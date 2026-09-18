# Commitment-Aware Semantic Sharding (CASS) with LiteQuorum verification

import os
import time
import numpy as np
import pandas as pd
import json
import hashlib
from mpi4py import MPI

import random
import threading
import math
from queue import Queue, Empty
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor


# Hashing and canonical layout
def stable_hash(obj):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_numpy_array(arr):
    arr = np.ascontiguousarray(np.asarray(arr))
    hasher = hashlib.sha256()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(str(tuple(arr.shape)).encode("utf-8"))
    hasher.update(memoryview(arr).cast("B"))
    return hasher.hexdigest()


CANONICAL_VECTOR_DTYPE   = np.float32
CANONICAL_CENTROID_DTYPE = np.float64


def as_canonical_vectors(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=CANONICAL_VECTOR_DTYPE))


def canonical_centroid(shard_vectors):
    sv = np.asarray(shard_vectors)
    if sv.ndim == 2 and sv.shape[0] > 0:
        return np.mean(sv, axis=0, dtype=np.float64)
    return np.zeros((1,), dtype=CANONICAL_CENTROID_DTYPE)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path, obj):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def block_metadata_bytes(block):
    return int(len(json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")))


# Dataset readers
def fvecs_info(path):
    d = int(np.fromfile(path, dtype="int32", count=1)[0])
    n = os.path.getsize(path) // (4 * (d + 1))
    return int(n), d


_BIGANN_DTYPES = {".fbin": np.float32, ".u8bin": np.uint8, ".i8bin": np.int8}


def is_bigann(path):
    return any(ext in os.path.basename(path) for ext in _BIGANN_DTYPES)


def bigann_dtype(path):
    name = os.path.basename(path)
    for ext, dt in _BIGANN_DTYPES.items():
        if ext in name:
            return dt
    raise ValueError(f"not a big-ann vector file: {name}")


def fbin_info(path):
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32, count=2)
    return int(n), int(d)


def read_fbin_range(path, start, count, dtype=np.float32):
    n, d = fbin_info(path)
    src = np.dtype(bigann_dtype(path))
    start = int(max(0, min(start, n)))
    count = int(max(0, min(count, n - start)))
    mm = np.memmap(path, dtype=src, mode="r",
                   offset=8 + start * d * src.itemsize, shape=(count, d))
    return np.ascontiguousarray(mm, dtype=dtype)


def read_fbin(path, max_rows=None, dtype=np.float32):
    n, _ = fbin_info(path)
    return read_fbin_range(path, 0, n if max_rows is None else min(max_rows, n),
                           dtype=dtype)


def read_ibin(path, max_rows=None):
    with open(path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32, count=2)
    n, k = int(n), int(k)
    rows = n if max_rows is None else min(max_rows, n)
    return np.ascontiguousarray(
        np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(rows, k)))


def read_fvecs(path, max_rows=None, dtype=np.float32):
    raw = np.memmap(path, dtype="int32", mode="r")
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    if max_rows is not None:
        rows = rows[:max_rows]
    return np.ascontiguousarray(rows[:, 1:].view("float32")).astype(dtype, copy=False)


def read_ivecs(path, max_rows=None):
    raw = np.memmap(path, dtype="int32", mode="r")
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    if max_rows is not None:
        rows = rows[:max_rows]
    return np.ascontiguousarray(rows[:, 1:])


def _safe_l2_normalize(arr, axis=1, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return arr / norm


# Sharding quality metrics
def compute_coefficient_of_variation(shard_sizes):
    if not shard_sizes:
        return float("nan")

    sizes = np.asarray(shard_sizes, dtype=np.float64)
    mean = float(np.mean(sizes))

    if mean == 0:
        return float("nan")

    std = float(np.std(sizes, ddof=0))
    return std / mean


def compute_centroid_based_sharding_quality(X, labels, noise_label=-1):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    valid_mask = labels != noise_label
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if X_valid.shape[0] == 0:
        return {
            "intra_shard_cosine": float("nan"),
            "inter_shard_cosine": float("nan"),
            "separation_gap": float("nan"),
            "balance_ratio": 0.0,
            "cv_shard_size": float("nan"),
            "noise_count": int(np.sum(labels == noise_label)),
            "noise_ratio": float(np.mean(labels == noise_label)),
            "per_shard": [],
        }

    unique_labels = sorted(np.unique(labels_valid).tolist())
    per_shard = []
    centroids = []
    weighted_intra_sum = 0.0
    total_points = 0
    shard_sizes = []

    for sid in unique_labels:
        shard_data = X_valid[labels_valid == sid]
        n_points = int(shard_data.shape[0])
        if n_points == 0:
            continue

        centroid = np.mean(shard_data, axis=0)
        centroid_norm = _safe_l2_normalize(centroid.reshape(1, -1), axis=1)[0]
        shard_norm = _safe_l2_normalize(shard_data, axis=1)

        similarities = shard_norm @ centroid_norm
        intra = float(np.mean(similarities))

        per_shard.append({
            "shard_id": int(sid),
            "num_points": n_points,
            "intra_shard_cosine": intra,
            "centroid": centroid.tolist(),
        })

        centroids.append(centroid)
        shard_sizes.append(n_points)
        weighted_intra_sum += intra * n_points
        total_points += n_points

    avg_intra = float(weighted_intra_sum / total_points) if total_points else float("nan")

    if len(centroids) >= 2:
        centroid_matrix = _safe_l2_normalize(np.vstack(centroids), axis=1)
        sim_matrix = centroid_matrix @ centroid_matrix.T
        upper = np.triu_indices_from(sim_matrix, k=1)
        avg_inter = float(np.mean(sim_matrix[upper]))
    else:
        avg_inter = float("nan")

    separation_gap = float(avg_intra - avg_inter) if np.isfinite(avg_inter) else float("nan")
    balance_ratio = float(min(shard_sizes) / max(shard_sizes)) if shard_sizes else 0.0
    cv_shard_size = compute_coefficient_of_variation(shard_sizes)

    return {
        "intra_shard_cosine": avg_intra,
        "inter_shard_cosine": avg_inter,
        "separation_gap": separation_gap,
        "balance_ratio": balance_ratio,
        "cv_shard_size": cv_shard_size,
        "noise_count": int(np.sum(labels == noise_label)),
        "noise_ratio": float(np.mean(labels == noise_label)),
        "per_shard": per_shard,
    }


def majority_fault_tolerance_summary(n_validators):
    n = int(n_validators)
    quorum = (n // 2) + 1

    return {
        "validators": n,
        "max_faulty_nodes_for_majority": n - quorum,
        "majority_commit_quorum": quorum,
        "quorum_rule": ">50% YES votes",
    }


# Merkle commitments
def _hash_leaf(row_index: int, row: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(row_index.to_bytes(8, "little"))
    hasher.update(np.ascontiguousarray(row, dtype=CANONICAL_VECTOR_DTYPE).tobytes())
    return hasher.hexdigest()


def _hash_block(block_index: int, block_bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(block_index.to_bytes(8, "little"))
    hasher.update(block_bytes)
    return hasher.hexdigest()


MERKLE_LEAF_ROWS = 1024

MERKLE_THREADS   = 1
MERKLE_MIN_ROWS  = 1 << 30


def _hash_leaf_range(mv, row_bytes, start, end):
    out = []
    stride = MERKLE_LEAF_ROWS * row_bytes
    idx = start // MERKLE_LEAF_ROWS
    for s in range(start, end, MERKLE_LEAF_ROWS):
        e = min(s + MERKLE_LEAF_ROWS, end)
        out.append(_hash_block(idx, mv[s * row_bytes:e * row_bytes]))
        idx += 1
    return out


def _hash_pair(left: str, right: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(bytes.fromhex(left))
    hasher.update(bytes.fromhex(right))
    return hasher.hexdigest()


def build_merkle_tree(shard_vectors: np.ndarray, threads: int = None) -> dict:
    if shard_vectors.ndim != 2 or shard_vectors.shape[0] == 0:
        return {"root": "0" * 64, "leaves": [], "depth": 0}

    buf       = as_canonical_vectors(shard_vectors)
    n_rows    = buf.shape[0]
    row_bytes = buf.shape[1] * buf.itemsize
    mv        = memoryview(buf).cast("B")

    nthreads = MERKLE_THREADS if threads is None else max(1, int(threads))
    if nthreads > 1 and n_rows >= MERKLE_MIN_ROWS:
        step   = (n_rows + nthreads - 1) // nthreads
        spans  = [(i, min(i + step, n_rows)) for i in range(0, n_rows, step)]
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            parts = list(ex.map(lambda sp: _hash_leaf_range(mv, row_bytes, *sp), spans))
        leaves = [h for part in parts for h in part]
    else:
        leaves = _hash_leaf_range(mv, row_bytes, 0, n_rows)

    level  = leaves[:]
    depth  = 0

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_hash_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        depth += 1

    return {"root": level[0], "leaves": leaves, "depth": depth}


def verify_merkle_root(shard_vectors: np.ndarray, expected_root: str,
                       precomputed_root: str = None) -> dict:
    result = {"check": "verify_merkle_root", "passed": False, "detail": ""}

    if precomputed_root is not None:
        actual = precomputed_root
        n_leaves = int(shard_vectors.shape[0]) if shard_vectors.ndim == 2 else 0
        depth    = 0 if n_leaves <= 1 else int(math.ceil(math.log2(n_leaves)))
        source   = "reused"
    else:
        tree     = build_merkle_tree(shard_vectors)
        actual   = tree["root"]
        n_leaves = len(tree["leaves"])
        depth    = tree["depth"]
        source   = "rebuilt"

    if actual != expected_root:
        raise VectorVerificationError(
            f"verify_merkle_root: root mismatch -- "
            f"expected {expected_root[:12]}... got {actual[:12]}... "
            f"-- at least one vector row was tampered"
        )

    result["passed"] = True
    result["detail"] = (
        f"root={actual[:12]}..., leaves={n_leaves}, depth={depth} ({source})"
    )
    return result


def build_shard_commitment(shard_vectors: np.ndarray) -> dict:
    sv       = as_canonical_vectors(shard_vectors)
    centroid = canonical_centroid(sv)
    tree     = build_merkle_tree(sv)
    return {
        "vectors":       sv,
        "centroid":      centroid,
        "leaves":        tree["leaves"],
        "data_hash":     hash_numpy_array(sv),
        "centroid_hash": hash_numpy_array(centroid),
        "merkle_root":   tree["root"],
        "merkle_depth":  tree["depth"],
    }


def merkle_proof_path(leaves: list, block_index: int) -> list:
    level = leaves[:]
    path  = []
    idx   = block_index

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        if idx % 2 == 0:
            path.append((level[idx + 1] if idx + 1 < len(level) else level[idx], "right"))
        else:
            path.append((level[idx - 1], "left"))
        level = [_hash_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2

    return path


def verify_merkle_proof(block_index: int, rows_block: np.ndarray,
                        proof_path: list, expected_root: str) -> bool:
    buf = as_canonical_vectors(rows_block)
    current = _hash_block(block_index, memoryview(buf).cast("B"))
    for sibling, position in proof_path:
        current = _hash_pair(current, sibling) if position == "right" else _hash_pair(sibling, current)
    return current == expected_root


def merkle_block_of(row_index: int):
    b = row_index // MERKLE_LEAF_ROWS
    return b, slice(b * MERKLE_LEAF_ROWS, (b + 1) * MERKLE_LEAF_ROWS)


def compute_vector_ids(shard_id: int, shard_vectors: np.ndarray) -> list:
    shard_bytes = shard_id.to_bytes(8, "little")
    ids = []
    for row in shard_vectors:
        h = hashlib.sha256()
        h.update(shard_bytes)
        h.update(np.ascontiguousarray(row, dtype=CANONICAL_VECTOR_DTYPE).tobytes())
        ids.append(h.hexdigest())
    return ids


def verify_no_duplicate_vector_ids(
    shard_id: int,
    shard_vectors: np.ndarray,
    global_seen_ids: set = None,
) -> dict:
    result = {"check": "verify_no_duplicate_vector_ids", "passed": False, "detail": ""}

    if global_seen_ids is None:
        global_seen_ids = set()

    ids        = compute_vector_ids(shard_id, shard_vectors)
    local_seen: set  = set()
    duplicates: list = []

    for i, vid in enumerate(ids):
        if vid in local_seen or vid in global_seen_ids:
            duplicates.append((i, vid[:12]))
        else:
            local_seen.add(vid)

    if duplicates:
        raise VectorVerificationError(
            f"verify_no_duplicate_vector_ids: {len(duplicates)} duplicate vector(s) "
            f"in shard {shard_id} -- first at row {duplicates[0][0]} "
            f"(id={duplicates[0][1]}...)"
        )

    global_seen_ids.update(local_seen)
    result["passed"] = True
    result["detail"] = f"{len(ids)} unique vector IDs, no duplicates"
    return result


# Verification checks
class VectorVerificationError(Exception):
    pass


def verify_timestamp(block: dict, max_drift_sec: float = 60.0) -> dict:
    result = {"check": "verify_timestamp", "passed": False, "detail": ""}

    if "timestamp" not in block:
        raise VectorVerificationError("verify_timestamp: missing 'timestamp' field")

    ts = block["timestamp"]
    if not isinstance(ts, (int, float)):
        raise VectorVerificationError(
            f"verify_timestamp: timestamp must be numeric, got {type(ts).__name__}"
        )

    age = time.time() - ts
    if age > max_drift_sec:
        raise VectorVerificationError(
            f"verify_timestamp: block is {age:.1f}s old (max {max_drift_sec}s) -- replay attack"
        )
    if age < -max_drift_sec:
        raise VectorVerificationError(
            f"verify_timestamp: block is {-age:.1f}s in the future -- pre-mining"
        )

    result["passed"] = True
    result["detail"] = f"age={age:.3f}s, within +/-{max_drift_sec}s"
    return result


def verify_dimensions(shard_vectors: np.ndarray, expected_dim: int) -> dict:
    result = {"check": "verify_dimensions", "passed": False, "detail": ""}

    if shard_vectors.ndim != 2:
        raise VectorVerificationError(
            f"verify_dimensions: expected 2-D, got {shard_vectors.ndim}-D {shard_vectors.shape}"
        )

    n, d = shard_vectors.shape
    if d != expected_dim:
        raise VectorVerificationError(
            f"verify_dimensions: expected {expected_dim} dims, got {d}"
        )
    FINITE_CHUNK = 8192
    for s in range(0, n, FINITE_CHUNK):
        block = shard_vectors[s:s + FINITE_CHUNK]
        if not np.isfinite(block).all():
            nan_c = int(np.sum(np.isnan(block)))
            inf_c = int(np.sum(np.isinf(block)))
            raise VectorVerificationError(
                f"verify_dimensions: {nan_c} NaN + {inf_c} Inf values in rows "
                f"[{s}, {min(s + FINITE_CHUNK, n)}) -- corrupted"
            )

    result["passed"] = True
    result["detail"] = f"shape=({n},{d}), all finite"
    return result


def verify_hash(
    shard_vectors: np.ndarray,
    centroid: np.ndarray,
    expected_data_hash: str,
    expected_centroid_hash: str,
    block_data_hash: str = None,
    block_centroid_hash: str = None,
    precomputed_data_hash: str = None,
    precomputed_centroid_hash: str = None,
) -> dict:
    result = {"check": "verify_hash", "passed": False, "detail": ""}

    recomputed_data = precomputed_data_hash if precomputed_data_hash is not None \
                      else hash_numpy_array(shard_vectors)
    recomputed_cent = precomputed_centroid_hash if precomputed_centroid_hash is not None \
                      else hash_numpy_array(centroid)

    if recomputed_data != expected_data_hash:
        raise VectorVerificationError(
            f"verify_hash: data_hash mismatch -- "
            f"expected {expected_data_hash[:12]}... got {recomputed_data[:12]}..."
        )
    if recomputed_cent != expected_centroid_hash:
        raise VectorVerificationError(
            f"verify_hash: centroid_hash mismatch -- "
            f"expected {expected_centroid_hash[:12]}... got {recomputed_cent[:12]}..."
        )
    if block_data_hash is not None and block_data_hash != expected_data_hash:
        raise VectorVerificationError(
            f"verify_hash: block-field data_hash forged "
            f"(block={block_data_hash[:12]}... vs validator={expected_data_hash[:12]}...)"
        )
    if block_centroid_hash is not None and block_centroid_hash != expected_centroid_hash:
        raise VectorVerificationError(
            f"verify_hash: block-field centroid_hash forged "
            f"(block={block_centroid_hash[:12]}... vs validator={expected_centroid_hash[:12]}...)"
        )

    result["passed"]        = True
    result["data_hash"]     = recomputed_data
    result["centroid_hash"] = recomputed_cent
    result["detail"] = f"data={recomputed_data[:12]}..., centroid={recomputed_cent[:12]}... match"
    return result


def run_minimum_verification(
    block: dict,
    shard_vectors: np.ndarray,
    expected_data_hash: str,
    expected_centroid_hash: str,
    max_drift_sec: float = 60.0,
    global_seen_ids: set = None,
    precomputed: dict = None,
) -> dict:
    precomputed  = precomputed or {}
    centroid     = np.asarray(block["centroid"], dtype=CANONICAL_CENTROID_DTYPE)
    expected_dim = int(block["vector_dim"])
    shard_id     = int(block["shard_id"])
    merkle_root  = block.get("merkle_root", "0" * 64)

    checks = [
        ("verify_timestamp",
         lambda: verify_timestamp(block, max_drift_sec)),

        ("verify_dimensions",
         lambda: verify_dimensions(shard_vectors, expected_dim)),


        ("verify_merkle_root",
         lambda: verify_merkle_root(shard_vectors, merkle_root,
                                    precomputed_root=precomputed.get("merkle_root"))),

        ("verify_hash",
         lambda: verify_hash(shard_vectors, centroid,
                             expected_data_hash, expected_centroid_hash,
                             block_data_hash=block.get("data_hash"),
                             block_centroid_hash=block.get("centroid_hash"),
                             precomputed_data_hash=precomputed.get("data_hash"),
                             precomputed_centroid_hash=precomputed.get("centroid_hash"))),
    ]

    summary = {"passed": True, "failed_check": None, "error": None, "results": {}}

    for name, fn in checks:
        try:
            summary["results"][name] = fn()
        except VectorVerificationError as exc:
            summary["passed"]        = False
            summary["failed_check"]  = name
            summary["error"]         = str(exc)
            summary["results"][name] = {"check": name, "passed": False, "detail": str(exc)}
            break

    return summary


def build_all_verification_ctxs(comm, rank: int, size: int,
                                local_shard_map: dict, owned_shards: list) -> dict:

    local_descriptors = [{"shard_id": int(sid), "computed_by": int(rank)}
                         for sid in owned_shards]

    all_descriptors = comm.allgather(local_descriptors)

    total_length = 0
    for rank_descriptors in all_descriptors:
        if len(rank_descriptors) > 0:
            for descriptor in rank_descriptors:
                total_length += len(descriptor)
        else:
            pass

    validator_rank = (rank + 1) % size

    proposer_rank    = (rank - 1) % size
    proposer_shard_ids = [sid for sid in range(
        sum(len(d) for d in all_descriptors)
    ) if sid % size == proposer_rank]

    proposer_owned = [d["shard_id"] for d in all_descriptors[proposer_rank]]

    my_payload = []
    for shard_id in owned_shards:
        sv = as_canonical_vectors(local_shard_map[shard_id])
        my_payload.append((int(shard_id), sv))

    received_payload = comm.sendrecv(
        sendobj  = my_payload,
        dest     = validator_rank,
        sendtag  = 0,
        source   = proposer_rank,
        recvtag  = 0,
    )

    independent_ctxs_for_proposer = {}
    for shard_id, sv in received_payload:
        c = build_shard_commitment(sv)
        independent_ctxs_for_proposer[shard_id] = {
            "shard_id":      int(shard_id),
            "data_hash":     c["data_hash"],
            "centroid_hash": c["centroid_hash"],
            "merkle_root":   c["merkle_root"],
            "computed_by":   int(rank),
        }

    my_ctxs = comm.sendrecv(
        sendobj  = independent_ctxs_for_proposer,
        dest     = proposer_rank,
        sendtag  = 1,
        source   = validator_rank,
        recvtag  = 1,
    )

    return my_ctxs


# Distributed Ledger

class DVD:
    def __init__(self, chain_file="DVD.json"):
        self.chain_file = chain_file
        self.chain = []
        if os.path.exists(self.chain_file):
            self.load()
        else:
            self.create_genesis_block()
            self.save()

    def create_genesis_block(self):
        g = {
            "index": 0, "timestamp": time.time(),
            "previous_hash": "0" * 64,
            "block_type": "genesis", "data": "Genesis Block"
        }
        g["block_hash"] = self.compute_block_hash(g)
        self.chain = [g]

    def get_last_block(self):
        return self.chain[-1]

    def compute_block_hash(self, block):
        bc = deepcopy(block)
        bc.pop("block_hash", None)
        return hashlib.sha256(
            json.dumps(bc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def add_block(self, committed_block):
        last    = self.get_last_block()
        new_blk = deepcopy(committed_block)
        new_blk["index"]         = len(self.chain)
        new_blk["timestamp"]     = time.time()
        new_blk["previous_hash"] = last["block_hash"]
        new_blk["block_hash"]    = self.compute_block_hash(new_blk)
        self.chain.append(new_blk)
        return new_blk

    def save(self):
        with open(self.chain_file, "w", encoding="utf-8") as f:
            json.dump(self.chain, f, indent=2)

    def load(self):
        with open(self.chain_file, "r", encoding="utf-8") as f:
            self.chain = json.load(f)

    def verify_chain(self):
        if not self.chain:
            return {"valid": False, "failure_index": -1,
                    "failure_reason": "empty chain", "checked_blocks": 0}

        for i, blk in enumerate(self.chain):
            recomputed = self.compute_block_hash(blk)
            if blk.get("block_hash") != recomputed:
                return {"valid": False, "failure_index": i,
                        "failure_reason": "block_hash mismatch (content tampered)",
                        "checked_blocks": i}
            if blk.get("index") != i:
                return {"valid": False, "failure_index": i,
                        "failure_reason": f"index mismatch (stored={blk.get('index')}, expected={i})",
                        "checked_blocks": i}
            if i > 0:
                prev_hash = self.chain[i - 1]["block_hash"]
                if blk.get("previous_hash") != prev_hash:
                    return {"valid": False, "failure_index": i,
                            "failure_reason": "previous_hash broken (reordering/deletion)",
                            "checked_blocks": i}
        return {"valid": True, "failure_index": None,
                "failure_reason": None, "checked_blocks": len(self.chain)}


BIGANN_PATH = "base.1B.fbin.crop_nb_100000000"
GIST_PATH   = os.path.join("gist", "gist_base.fvecs")

MAX_ROWS    = None
NUM_STEPS   = 100
SEED        = 42
TOL         = 1e-3

K           = 5

CHUNK_SIZE  = 32_768

PENALTY       = 0.0
PENALTY_DECAY = False
NORMALIZE     = True
FINAL_PENALTY = None

BALANCED      = True
BALANCE_TARGET_CV = 0.20

LAGRANGE_ITERS = 80
LAGRANGE_DECAY = 0.92

CHURN_TOL = 1e-4

SEED_SAMPLE_PER_RANK = 15_000

CORESET_FRAC = 0.20
CORESET_MIN  = 50_000
CORESET_MAX  = 400_000
CORESET_PER_CENTROID = 2_000
CORESET_PER_DIM_CENTROID = 10
CORESET_ALPHA = 0.5

METRIC_CHUNK  = 10_000

QUALITY_WEIGHTS = {"lift": 0.50, "separation": 0.30, "balance": 0.20}


def l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    X /= norms
    return X


def coreset_size(n, k=None, d=None):
    target = max(CORESET_MIN, int(CORESET_FRAC * n))
    if k is None:
        cap = CORESET_MAX
    else:
        floor_k  = CORESET_PER_CENTROID * int(k)
        floor_dk = CORESET_PER_DIM_CENTROID * int(k) * (int(d) if d else 0)
        cap = max(CORESET_MAX, floor_k, floor_dk)
    return int(min(target, cap, n))


# Coreset construction
def build_coreset(X, m_total, comm, seed=SEED, alpha=CORESET_ALPHA,
                  chunk=CHUNK_SIZE):
    rank = comm.Get_rank()
    n_local, d = X.shape

    local_sum = np.zeros(d + 1, dtype=np.float64)
    for s in range(0, n_local, chunk):
        e = min(s + chunk, n_local)
        local_sum[:d] += X[s:e].sum(axis=0, dtype=np.float64)
    local_sum[d] = float(n_local)
    g = np.empty_like(local_sum)
    comm.Allreduce(local_sum, g, op=MPI.SUM)
    n_global = max(g[d], 1.0)
    mu = (g[:d] / n_global).astype(np.float32)

    d2 = np.empty(n_local, dtype=np.float64)
    mu_sq = float(mu @ mu)
    for s in range(0, n_local, chunk):
        e = min(s + chunk, n_local)
        Xc = X[s:e]
        d2[s:e] = (np.einsum('ij,ij->i', Xc, Xc).astype(np.float64)
                   - 2.0 * (Xc @ mu).astype(np.float64) + mu_sq)
    np.maximum(d2, 0.0, out=d2)

    buf = np.array([d2.sum()], dtype=np.float64)
    out = np.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.SUM)
    d2_total = max(float(out[0]), 1e-30)

    q_local = (1.0 - alpha) / n_global + alpha * d2 / d2_total
    mass_local = float(q_local.sum())

    masses = np.zeros(comm.Get_size(), dtype=np.float64)
    masses[rank] = mass_local
    all_masses = np.empty_like(masses)
    comm.Allreduce(masses, all_masses, op=MPI.SUM)
    all_masses = np.maximum(all_masses, 0.0)
    tot = all_masses.sum()
    if tot <= 0:
        raise ValueError("degenerate coreset distribution")
    counts = np.random.default_rng(seed).multinomial(m_total, all_masses / tot)
    m_local = int(counts[rank])

    if m_local == 0:
        return (np.empty((0, d), dtype=np.float32),
                np.empty(0, dtype=np.float64), mu)

    rng = np.random.default_rng(seed + 1000 + rank)
    p = q_local / mass_local
    idx = rng.choice(n_local, size=m_local, replace=True, p=p)

    rows = np.ascontiguousarray(X[idx])
    weights = 1.0 / (m_total * q_local[idx])
    return rows, weights, mu


# Distributed Shard Construction
class DistributedKMeans:
    def __init__(self, k=5, num_steps=100, seed=42, tol=1e-3, verbose=True,
                 print_every=5, balance_penalty=0.0, penalty_decay=False,
                 final_penalty=None, dtype=np.float32, chunk_size=CHUNK_SIZE,
                 early_stop_on_labels=True, final_relabel=True):
        self.k = k
        self.num_steps = num_steps
        self.seed = seed
        self.tol = tol
        self.verbose = verbose
        self.print_every = print_every
        self.balance_penalty = balance_penalty
        self.penalty_decay = penalty_decay
        self.final_penalty = final_penalty
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.early_stop_on_labels = early_stop_on_labels
        self.final_relabel = final_relabel

    def _penalty_at(self, step):
        if self.balance_penalty <= 0.0:
            return 0.0
        if not self.penalty_decay:
            return self.balance_penalty
        span = max(1.0, 0.6 * self.num_steps)
        return self.balance_penalty * max(0.0, 1.0 - step / span)

    def _penalty_vector(self, global_counts, strength):
        if strength <= 0.0 or global_counts is None or np.sum(global_counts) <= 0:
            return None
        mean_count = np.sum(global_counts) / len(global_counts)
        if mean_count <= 0:
            return None
        excess = np.maximum(0.0, global_counts / mean_count - 1.0)
        return 1.0 + strength * excess

    def assign_clusters(self, local_data, centroids, global_counts=None,
                        penalty_strength=None):
        X = local_data
        n = X.shape[0]
        out = np.empty(n, dtype=np.int64)
        c_sq = np.einsum('ij,ij->i', centroids, centroids)
        strength = self.balance_penalty if penalty_strength is None else penalty_strength
        penalty = self._penalty_vector(global_counts, strength)

        for s in range(0, n, self.chunk_size):
            e = min(s + self.chunk_size, n)
            Xc = X[s:e]
            D = Xc @ centroids.T
            D *= -2.0
            D += c_sq[None, :]
            if penalty is not None:
                D += np.einsum('ij,ij->i', Xc, Xc)[:, None]
                np.maximum(D, 0.0, out=D)
                D *= penalty[None, :]
            out[s:e] = np.argmin(D, axis=1)
        return out

    def _assign_and_accumulate(self, X, centroids, weights=None,
                               prev_labels=None, global_counts=None,
                               penalty_strength=None):
        n, d = X.shape
        k = centroids.shape[0]
        labels = np.empty(n, dtype=np.int64)
        sums = np.zeros((k, d), dtype=np.float64)
        wcounts = np.zeros(k, dtype=np.float64)
        rcounts = np.zeros(k, dtype=np.int64)
        churn = 0.0

        c_sq = np.einsum('ij,ij->i', centroids, centroids)
        strength = self.balance_penalty if penalty_strength is None else penalty_strength
        penalty = self._penalty_vector(global_counts, strength)

        for s in range(0, n, self.chunk_size):
            e = min(s + self.chunk_size, n)
            Xc = X[s:e]
            m = e - s

            D = Xc @ centroids.T
            D *= -2.0
            D += c_sq[None, :]
            if penalty is not None:
                D += np.einsum('ij,ij->i', Xc, Xc)[:, None]
                np.maximum(D, 0.0, out=D)
                D *= penalty[None, :]
            lab = np.argmin(D, axis=1)
            labels[s:e] = lab

            O = np.zeros((m, k), dtype=X.dtype)
            if weights is None:
                O[np.arange(m), lab] = 1.0
                wcounts += np.bincount(lab, minlength=k)
            else:
                wc = weights[s:e]
                O[np.arange(m), lab] = wc.astype(X.dtype, copy=False)
                wcounts += np.bincount(lab, weights=wc, minlength=k)
            sums += (O.T @ Xc).astype(np.float64)
            rcounts += np.bincount(lab, minlength=k)

            if prev_labels is not None:
                churn += float(np.count_nonzero(lab != prev_labels[s:e]))
            else:
                churn += float(m)

        return labels, sums, wcounts, rcounts, churn

    def initialize_centroids(self, X, weights=None):
        X = np.ascontiguousarray(X, dtype=self.dtype)
        n = X.shape[0]
        if n < self.k:
            raise ValueError(f"k={self.k} but only {n} samples")
        w = np.ones(n, dtype=np.float64) if weights is None \
            else np.asarray(weights, dtype=np.float64)
        rng = np.random.default_rng(self.seed)

        first = rng.choice(n, p=w / w.sum())
        centroids = [X[first]]
        best = np.full(n, np.inf, dtype=np.float64)
        x_sq = np.einsum('ij,ij->i', X, X).astype(np.float64)

        for _ in range(1, self.k):
            c = centroids[-1]
            d2 = x_sq - 2.0 * (X @ c) + float(c @ c)
            np.maximum(d2, 0.0, out=d2)
            np.minimum(best, d2, out=best)
            score = best * w
            total = score.sum()
            idx = rng.integers(0, n) if (not np.isfinite(total) or total <= 0) \
                  else rng.choice(n, p=score / total)
            centroids.append(X[idx])

        C = np.ascontiguousarray(np.vstack(centroids), dtype=self.dtype)
        if self.verbose:
            print(f"[KMeans] Centroids shape: {C.shape} ({C.dtype}, weighted "
                  f"k-means++ from {n:,} coreset points)")
        return C

    def run(self, local_data, comm, weights=None, init_data=None,
            init_weights=None):
        rank = comm.Get_rank()
        X = np.ascontiguousarray(local_data, dtype=self.dtype)
        if X.ndim != 2:
            raise ValueError(f"[Rank {rank}] Expected 2D, got {X.shape}")
        w = None if weights is None else np.asarray(weights, dtype=np.float64)

        if init_data is None:
            n_take = min(X.shape[0], max(1, SEED_SAMPLE_PER_RANK))
            if X.shape[0] > n_take:
                sel = np.random.default_rng(self.seed + 31 + rank).choice(
                    X.shape[0], size=n_take, replace=False)
                part_X, part_w = np.ascontiguousarray(X[sel]), (
                    None if w is None else w[sel])
            else:
                part_X, part_w = X, w
            parts = comm.gather((part_X, part_w), root=0)
            if rank == 0:
                src = np.ascontiguousarray(np.vstack([p[0] for p in parts]))
                src_w = (None if parts[0][1] is None
                         else np.concatenate([p[1] for p in parts]))
            else:
                src = src_w = None
        else:
            src, src_w = init_data, init_weights

        C = self.initialize_centroids(src, src_w) if rank == 0 else None
        C = comm.bcast(C, root=0)
        k, d = C.shape

        mb = np.array([float(X.shape[0])], dtype=np.float64)
        mg = np.empty_like(mb)
        comm.Allreduce(mb, mg, op=MPI.SUM)
        m_global = float(mg[0])

        global_counts = np.zeros(k, dtype=np.float64)
        prev_labels = None
        packed = np.empty(k * d + 2 * k + 1, dtype=np.float64)
        gpacked = np.empty_like(packed)
        empty_ids = np.array([], dtype=np.int64)

        for step in range(self.num_steps):
            strength = self._penalty_at(step)

            labels, local_sums, local_w, local_r, churn = \
                self._assign_and_accumulate(X, C, w, prev_labels,
                                            global_counts, strength)

            packed[:k * d] = local_sums.ravel()
            packed[k * d:k * d + k] = local_w
            packed[k * d + k:k * d + 2 * k] = local_r
            packed[-1] = churn
            comm.Allreduce(packed, gpacked, op=MPI.SUM)

            global_sums = gpacked[:k * d].reshape(k, d)
            global_counts = gpacked[k * d:k * d + k].copy()
            global_raw = gpacked[k * d + k:k * d + 2 * k].astype(np.int64)
            global_churn = gpacked[-1]

            new_c = C.astype(np.float64, copy=True)
            nonempty = global_counts > 0
            new_c[nonempty] = global_sums[nonempty] / global_counts[nonempty][:, None]

            empty_ids = np.where(~nonempty)[0]
            if len(empty_ids) > 0:
                if rank == 0 and X.shape[0] > 0:
                    d_own = np.empty(X.shape[0], dtype=np.float64)
                    for s in range(0, X.shape[0], self.chunk_size):
                        e = min(s + self.chunk_size, X.shape[0])
                        diff = X[s:e] - C[labels[s:e]]
                        d_own[s:e] = np.einsum('ij,ij->i', diff, diff)
                    for eid in empty_ids:
                        far = int(np.argmax(d_own))
                        new_c[eid] = X[far]
                        d_own[far] = -1.0
                new_c = comm.bcast(new_c, root=0)

            shift_abs = float(np.linalg.norm(new_c - C))
            shift     = shift_abs / max(float(np.linalg.norm(C)), 1e-12)
            C = np.ascontiguousarray(new_c, dtype=self.dtype)
            prev_labels = labels

            if rank == 0 and self.verbose and (step == 0 or (step + 1) % self.print_every == 0):
                cv = float(np.std(global_raw) / np.mean(global_raw)) \
                     if global_raw.sum() > 0 else float("nan")
                print(f"[KMeans] Step {step + 1}, shift={shift:.6f} (rel), "
                      f"abs={shift_abs:.6f}, coreset_cv={cv:.4f}, "
                      f"churn={int(global_churn)}, empty={len(empty_ids)}, "
                      f"pen={strength:.3f}", flush=True)

            if shift < self.tol:
                if rank == 0 and self.verbose:
                    print(f"[KMeans] Converged at step {step + 1}")
                break
            if (self.early_stop_on_labels and step > 0
                    and global_churn <= CHURN_TOL * max(m_global, 1.0)):
                if rank == 0 and self.verbose:
                    print(f"[KMeans] Converged (churn {int(global_churn)} <= "
                          f"{CHURN_TOL:.1%} of {int(m_global):,}) at step {step + 1}")
                break

        if not self.final_relabel:
            return None, C.astype(np.float64), global_counts

        final_strength = self.balance_penalty if self.final_penalty is None \
                         else self.final_penalty
        final_labels = self.assign_clusters(X, C, global_counts, final_strength)
        cbuf = np.bincount(final_labels, minlength=k).astype(np.float64)
        gbuf = np.empty_like(cbuf)
        comm.Allreduce(cbuf, gbuf, op=MPI.SUM)
        return final_labels, C.astype(np.float64), gbuf


def _unit_rows(V, eps=1e-12):
    V = np.asarray(V, dtype=np.float64)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)


# Distributed shard metrics
def compute_shard_metrics(X, labels, k, comm, weights=None, chunk=METRIC_CHUNK):
    w = dict(QUALITY_WEIGHTS if weights is None else weights)
    n_local, d = X.shape

    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)
    for s in range(0, n_local, chunk):
        e = min(s + chunk, n_local)
        Xc = X[s:e]
        lab = labels[s:e]
        m = e - s
        O = np.zeros((m, k), dtype=np.float32)
        O[np.arange(m), lab] = 1.0
        sums += (O.T @ Xc).astype(np.float64)
        counts += np.bincount(lab, minlength=k)

    packed = np.concatenate([sums.ravel(), counts, [float(n_local)]])
    g = np.empty_like(packed)
    comm.Allreduce(packed, g, op=MPI.SUM)

    gsum = g[:k * d].reshape(k, d)
    gcnt = g[k * d:k * d + k]
    n_total = g[-1]

    C = np.zeros_like(gsum)
    nz = gcnt > 0
    C[nz] = gsum[nz] / gcnt[nz][:, None]
    gmean = gsum.sum(axis=0) / max(n_total, 1.0)

    Cn = np.zeros_like(C)
    if nz.any():
        Cn[nz] = _unit_rows(C[nz])
    gmean_u = gmean / max(np.linalg.norm(gmean), 1e-12)

    Cn32    = np.ascontiguousarray(Cn, dtype=np.float32)
    gmean32 = np.ascontiguousarray(gmean_u, dtype=np.float32)

    own_s = other_s = base_s = pure_s = 0.0
    for s in range(0, n_local, chunk):
        e = min(s + chunk, n_local)
        Xb  = X[s:e]
        nrm = np.linalg.norm(Xb, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        Xn  = Xb / nrm
        lab = labels[s:e]

        S = Xn @ Cn32.T
        rows = np.arange(S.shape[0])
        own = S[rows, lab]

        S_masked = S.copy()
        S_masked[rows, lab] = -np.inf
        best_other = S_masked.max(axis=1)

        own_s   += float(own.sum(dtype=np.float64))
        other_s += float(best_other.sum(dtype=np.float64))
        base_s  += float((Xn @ gmean32).sum(dtype=np.float64))
        pure_s  += float((S.argmax(axis=1) == lab).sum())

    buf = np.array([own_s, other_s, base_s, pure_s, float(n_local)])
    out = np.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.SUM)
    n = out[4]

    intra = out[0] / n
    baseline = out[2] / n
    lift = intra - baseline
    headroom = max(1.0 - baseline, 1e-9)

    idx = np.arange(k)[nz]
    if idx.size >= 2:
        Uc = _unit_rows(C[nz] - gmean[None, :])
        Sc = Uc @ Uc.T
        iu = np.triu_indices(idx.size, k=1)
        pairs = Sc[iu]
        worst = int(np.argmax(pairs))
        inter = float(pairs.mean())
        inter_max = float(pairs.max())
        worst_pair = (int(idx[iu[0][worst]]), int(idx[iu[1][worst]]))
    else:
        inter = inter_max = float("nan")
        worst_pair = None

    balance = float(gcnt.min() / gcnt.max()) if gcnt.max() > 0 else 0.0
    lift_captured = float(np.clip(lift / headroom, 0.0, 1.0))
    separation = float(np.clip((1.0 - inter) / 2.0, 0.0, 1.0)) \
        if np.isfinite(inter) else 0.0
    score = (w["lift"] * lift_captured
             + w["separation"] * separation
             + w["balance"] * balance)

    return {
        "intra_shard_cosine": intra,
        "baseline_cosine": baseline,
        "cohesion_lift": lift,
        "lift_captured": lift_captured,
        "inter_shard_cosine": inter,
        "inter_max": inter_max,
        "worst_pair": worst_pair,
        "point_margin": (out[0] - out[1]) / n,
        "purity": out[3] / n,
        "balance_ratio": balance,
        "shard_size_cv": float(gcnt.std() / gcnt.mean()) if gcnt.mean() else float("nan"),
        "empty_shards": int((gcnt == 0).sum()),
        "shard_quality_score": score,
        "centroids": C,
        "counts": gcnt.astype(np.int64),
        "weights": w,
    }


def print_shard_metrics(m, latency, throughput):
    w = m["weights"]
    print(f"  latency              : {latency:.2f} s")
    print(f"  throughput           : {throughput:,.0f} vec/s")
    print(f"  intra_shard_cosine   : {m['intra_shard_cosine']:.6f}")
    print(f"  inter_shard_cosine   : {m['inter_shard_cosine']:+.6f}   (centered, lower is better)")
    print(f"  shard_size_cv        : {m['shard_size_cv']:.6f}")
    print(f"  balance_ratio        : {m['balance_ratio']:.6f}")
    print(f"  purity               : {m['purity']:.4f}   (in their nearest shard)")
    print(f"  shard_quality_score  : {m['shard_quality_score']:.6f}"
          f"   (w: lift {w['lift']}, sep {w['separation']}, bal {w['balance']})")


# Balance-aware assignment
def balanced_assign(X, C, comm, target_cv=BALANCE_TARGET_CV,
                    iters=LAGRANGE_ITERS, chunk=CHUNK_SIZE, verbose=False):
    n, k = X.shape[0], C.shape[0]
    Cf = np.ascontiguousarray(C, dtype=np.float32)
    c_sq = np.einsum('ij,ij->i', Cf, Cf)

    D = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = X[s:e] @ Cf.T
        blk *= -2.0
        blk += c_sq[None, :]
        D[s:e] = blk

    part = np.partition(D, 1, axis=1)
    sb = np.array([float((part[:, 1] - part[:, 0]).sum()), float(n)])
    sg = np.empty_like(sb)
    comm.Allreduce(sb, sg, op=MPI.SUM)
    scale = max(sg[0] / max(sg[1], 1.0), 1e-6)

    lam = np.zeros(k, dtype=np.float32)
    cbuf = np.empty(k, dtype=np.float64)
    gbuf = np.empty(k, dtype=np.float64)
    eta = scale
    natural_cv = None
    best_lam, best_cv = lam.copy(), np.inf

    for t in range(iters):
        labels = np.argmin(D + lam[None, :], axis=1)
        cbuf[:] = np.bincount(labels, minlength=k)
        comm.Allreduce(cbuf, gbuf, op=MPI.SUM)

        mean = gbuf.mean()
        cv = float(gbuf.std() / mean) if mean > 0 else 0.0
        if t == 0:
            natural_cv = cv
        if cv < best_cv:
            best_cv, best_lam = cv, lam.copy()
        if cv <= target_cv:
            best_cv, best_lam = cv, lam.copy()
            break

        lam += (eta * (gbuf / mean - 1.0)).astype(np.float32)
        lam -= lam.min()
        eta *= LAGRANGE_DECAY

    lam = best_lam
    labels = np.argmin(D + lam[None, :], axis=1).astype(np.int64)

    if verbose:
        note = "" if best_cv <= target_cv else \
               f"   (target {target_cv:.3f} not reached)"
        print(f"[balance] natural_cv={natural_cv:.4f} -> cv={best_cv:.4f}"
              f"{note}  lambda={np.round(lam, 5)}", flush=True)
    return labels


# Commitment block
def build_metadata_block(rank, shard_id, shard_vectors, shard_file, commitment=None):
    c             = commitment if commitment is not None \
                    else build_shard_commitment(shard_vectors)
    shard_vectors = c["vectors"]
    centroid      = c["centroid"]
    tree          = {"root": c["merkle_root"], "leaves": c["leaves"],
                     "depth": c["merkle_depth"]}

    leaves_ref = f"merkle_leaves_{shard_id}.json"
    leaves_path = os.path.join(os.path.dirname(shard_file), leaves_ref)
    ensure_parent_dir(leaves_path)
    with open(leaves_path, "w") as f:
        json.dump(tree["leaves"], f)

    return {
        "rank_id":           int(rank),
        "shard_id":          int(shard_id),
        "timestamp":         time.time(),
        "data_hash":         c["data_hash"],
        "centroid":          centroid.tolist(),
        "centroid_hash":     c["centroid_hash"],
        "merkle_root":       tree["root"],
        "merkle_depth":      tree["depth"],
        "merkle_leaves_ref": leaves_ref,
        "num_points":        int(shard_vectors.shape[0]),
        "vector_dim":        int(shard_vectors.shape[1])
                               if shard_vectors.ndim == 2 and shard_vectors.size > 0
                               else (int(centroid.shape[0]) if centroid.ndim == 1 else 0),
        "offchain_ref":      os.path.basename(shard_file),
    }


# LiteQuorum Integrity

class PushPullHashConsensus:
    def __init__(
        self,
        output_dir="output_csv",
        seed=42,
        write_trace=False,
        validator_threads=4,
        max_drift_sec=60.0,
    ):
        self.output_dir        = output_dir
        self.seed              = seed
        self.write_trace       = write_trace
        self.validator_threads = validator_threads
        self.max_drift_sec     = max_drift_sec
        self._executor         = ThreadPoolExecutor(max_workers=max(1, validator_threads))
        self._global_seen_ids: set = set()

    def __del__(self):
        self._executor.shutdown(wait=False)

    def simulate_faulty_nodes(self, sub_cluster_size, fault_percentage):
        n   = int(sub_cluster_size * fault_percentage)
        rng = random.Random(self.seed)
        f   = set(rng.sample(range(sub_cluster_size), n)) if n > 0 else set()
        print(f"Fault percentage: {fault_percentage * 100:.2f}%  "
              f"Faulty nodes: {n}/{sub_cluster_size}")
        return f

    def consensus_success_rate(self, n, f, t):
        if f >= t:
            return 0.0
        fail_prob = 0.0
        for x in range(t, n + 1):
            if x <= f:
                fail_prob += math.comb(f, x) * (0.5 ** f)
        return 1.0 - fail_prob

    def _rejected_result(self, t_enter, sub_cluster_size, phase, error,
                         failed_check=None, verification_sec=0.0):
        r = {
            "committed": False,
            "quorum": majority_fault_tolerance_summary(sub_cluster_size)["majority_commit_quorum"],
            "max_faulty_nodes_for_majority": majority_fault_tolerance_summary(sub_cluster_size)["max_faulty_nodes_for_majority"],
            "fault_tolerance_status": "NOT_EVALUATED",
            "phase": phase, "verification_error": error,
            "prepare_yes": 0, "commit_yes": 0,
            "prepare_time_sec": 0.0, "commit_time_sec": 0.0,
            "push_time_sec": 0.0, "pull_time_sec": 0.0,
            "push_pull_time_sec": 0.0,
            "consensus_time_sec": 0.0,
            "verification_time_sec": float(verification_sec),
            "total_shard_time_sec": float(time.perf_counter() - t_enter),
            "min_validation_time_sec": 0.0, "avg_validation_time_sec": 0.0,
            "max_validation_time_sec": 0.0,
            "faulty_nodes": [], "consensus_success_rate": 0.0,
        }
        if failed_check:
            r["failed_check"] = failed_check
        return r

    def consensus(self, block, verification_ctx, rank, fault_percentage,
                  shard_vectors, sub_cluster_size=10, precomputed=None):
        t_enter = time.perf_counter()

        parsed_data = {k: block[k] for k in (
            "rank_id", "shard_id", "timestamp", "data_hash",
            "centroid", "centroid_hash",
            "merkle_root", "merkle_depth", "merkle_leaves_ref",
            "num_points", "vector_dim", "offchain_ref"
        )}

        expected_data_hash     = verification_ctx["data_hash"]
        expected_centroid_hash = verification_ctx["centroid_hash"]
        expected_merkle_root   = verification_ctx["merkle_root"]

        t_verify = time.perf_counter()

        if block["merkle_root"] != expected_merkle_root:
            print(f"[Shard {block['shard_id']}] REJECTED: merkle_root mismatch "
                  f"(proposer {block['merkle_root'][:12]}... vs "
                  f"validator {expected_merkle_root[:12]}...)")
            return self._rejected_result(
                t_enter, sub_cluster_size, "MERKLE_ROOT_MISMATCH",
                "proposer and validator merkle roots disagree",
                verification_sec=time.perf_counter() - t_verify)

        shard_vectors_np = as_canonical_vectors(shard_vectors)

        mv_res = run_minimum_verification(
            block                  = block,
            shard_vectors          = shard_vectors_np,
            expected_data_hash     = expected_data_hash,
            expected_centroid_hash = expected_centroid_hash,
            max_drift_sec          = self.max_drift_sec,
            global_seen_ids        = self._global_seen_ids,
            precomputed            = precomputed,
        )
        verify_dur = time.perf_counter() - t_verify

        print(f"[Shard {block['shard_id']}] Verification "
              f"{'PASSED' if mv_res['passed'] else 'FAILED'} "
              f"in {verify_dur * 1000:.3f} ms  "
              f"(ctx from rank {verification_ctx.get('computed_by', '?')})")

        for name, res in mv_res["results"].items():
            status = "PASS" if res["passed"] else "FAIL"
            print(f"  {status} {name}: {res.get('detail', res.get('error', ''))}")

        if not mv_res["passed"]:
            return self._rejected_result(
                t_enter, sub_cluster_size, "MIN_VERIFY_REJECT",
                mv_res["error"], mv_res["failed_check"],
                verification_sec=verify_dur)

        votes         = np.zeros(sub_cluster_size, dtype=bool)
        buffers       = [None] * sub_cluster_size
        local_ledgers = [[] for _ in range(sub_cluster_size)]
        shared_ledger = []

        faulty           = self.simulate_faulty_nodes(sub_cluster_size, fault_percentage)
        fault_summary = majority_fault_tolerance_summary(sub_cluster_size)
        max_faulty_nodes = fault_summary["max_faulty_nodes_for_majority"]
        commit_threshold = fault_summary["majority_commit_quorum"]
        quorum_threshold = commit_threshold

        _vh           = mv_res["results"]["verify_hash"]
        pre_data_hash = _vh["data_hash"]
        pre_cent_hash = _vh["centroid_hash"]

        val_times    = []
        timing_lock  = threading.Lock()
        state_lock   = threading.Lock()
        stop_event   = threading.Event()
        vq           = Queue()

        for i in range(sub_cluster_size):
            vq.put(i)

        state            = {"yes": 0, "done": 0}
        t_consensus      = time.perf_counter()
        push_start       = t_consensus

        def validator_worker():
            while not stop_event.is_set():
                try:
                    i = vq.get_nowait()
                except Empty:
                    break

                vote_yes = False
                temp     = None

                if i not in faulty:
                    t0 = time.perf_counter()
                    is_valid = (
                        mv_res["passed"] and
                        pre_data_hash == expected_data_hash and
                        pre_cent_hash == expected_centroid_hash and
                        parsed_data["data_hash"]     == expected_data_hash and
                        parsed_data["centroid_hash"] == expected_centroid_hash and
                        parsed_data["merkle_root"]   == expected_merkle_root
                    )
                    elapsed = time.perf_counter() - t0

                    if is_valid:
                        temp = parsed_data.copy()
                        temp["meta"] = {
                            "validator_node": i,
                            "coordinator_rank": rank,
                            "status": "PREPARED"
                        }
                        vote_yes = True

                    with timing_lock:
                        val_times.append(elapsed)

                with state_lock:
                    votes[i] = vote_yes
                    if temp is not None:
                        buffers[i] = temp
                    state["done"] += 1
                    if vote_yes:
                        state["yes"] += 1
                    remaining = sub_cluster_size - state["done"]
                    if state["yes"] >= commit_threshold:
                        stop_event.set()
                    elif state["yes"] + remaining < commit_threshold:
                        stop_event.set()

                vq.task_done()

        threads = max(1, min(self.validator_threads, sub_cluster_size))
        for f in [self._executor.submit(validator_worker) for _ in range(threads)]:
            f.result()

        push_dur  = time.perf_counter() - push_start
        committed = False

        if int(votes.sum()) >= commit_threshold:
            print(f"[Shard {block['shard_id']}] Quorum reached ({state['yes']}/{sub_cluster_size})")
            cc = parsed_data.copy()
            cc["meta"] = {"coordinator_rank": rank, "status": "COORDINATOR_COMMITTED"}
            shared_ledger.append(cc)
            committed = True
        else:
            print(f"[Shard {block['shard_id']}] Quorum failed ({state['yes']}/{sub_cluster_size})")

        pull_start = time.perf_counter()
        if committed:
            for i in range(sub_cluster_size):
                if votes[i] and buffers[i]:
                    buffers[i]["meta"]["status"] = "COMMITTED"
                else:
                    fb = parsed_data.copy()
                    fb["meta"] = {"coordinator_rank": rank, "node_id": i,
                                  "status": "COMMIT_READ_FROM_LEDGER"}
                    buffers[i] = fb
                local_ledgers[i].append(buffers[i])
        pull_dur = time.perf_counter() - pull_start
        push_pull_dur = push_dur + pull_dur

        consensus_dur = time.perf_counter() - t_consensus

        print(f"[Shard {block['shard_id']}] Push time: {push_dur * 1000:.4f} ms")
        print(f"[Shard {block['shard_id']}] Pull time: {pull_dur * 1000:.4f} ms")
        print(f"[Shard {block['shard_id']}] Push-Pull time: {push_pull_dur * 1000:.4f} ms")
        print(f"[Shard {block['shard_id']}] Consensus (control plane): "
              f"{consensus_dur * 1000:.4f} ms  |  "
              f"Verification (data plane, O(n)): {verify_dur * 1000:.4f} ms")

        if self.write_trace:
            write_json(
                os.path.join(self.output_dir,
                             f"consensus_trace_rank_{rank}_shard_{block['shard_id']}.json"),
                {
                    "min_verification":      mv_res,
                    "verification_ctx_from": verification_ctx.get("computed_by"),
                    "shared_ledger":         shared_ledger,
                    "local_ledgers":         local_ledgers,
                    "faulty_nodes":          sorted(faulty),
                    "votes_true":            int(votes.sum()),
                    "validators_processed":  int(state["done"]),
                    "early_terminated":      bool(state["done"] < sub_cluster_size),
                    "push_time_sec":        float(push_dur),
                    "pull_time_sec":        float(pull_dur),
                    "push_pull_time_sec":   float(push_pull_dur),
                }
            )

        t4r   = max(1, quorum_threshold + 1)
        c_rate = self.consensus_success_rate(sub_cluster_size, len(faulty), t4r)
        print(f"Consensus Success Rate: {c_rate * 100:.2f}%")

        min_vt = float(min(val_times)) if val_times else 0.0
        avg_vt = float(sum(val_times) / len(val_times)) if val_times else 0.0
        max_vt = float(max(val_times)) if val_times else 0.0

        return {
            "committed":               bool(committed),
            "quorum":                  int(quorum_threshold),
            "max_faulty_nodes_for_majority": int(max_faulty_nodes),
            "fault_tolerance_status": "WITHIN_51_PERCENT_BOUND" if len(faulty) <= max_faulty_nodes else "EXCEEDS_51_PERCENT_BOUND",
            "phase":                   "COMMIT_SUCCESS" if committed else "PREPARE_REJECT",
            "prepare_yes":             int(votes.sum()),
            "commit_yes":              int(votes.sum()) if committed else 0,
            "prepare_time_sec":        float(push_dur),
            "commit_time_sec":         float(pull_dur),
            "push_time_sec":           float(push_dur),
            "pull_time_sec":           float(pull_dur),
            "push_pull_time_sec":      float(push_pull_dur),
            "consensus_time_sec":      float(consensus_dur),
            "verification_time_sec":   float(verify_dur),
            "total_shard_time_sec":    float(time.perf_counter() - t_enter),
            "min_validation_time_sec": min_vt,
            "avg_validation_time_sec": avg_vt,
            "max_validation_time_sec": max_vt,
            "faulty_nodes":            sorted(faulty),
            "consensus_success_rate":  float(c_rate),
        }

    def run(self, block, verification_ctx, rank, fault_percentage,
            shard_vectors, sub_cluster_size=10, precomputed=None):
        return self.consensus(block, verification_ctx, rank,
                              fault_percentage, shard_vectors, sub_cluster_size,
                              precomputed=precomputed)


# Tamper detection experiment
class TamperDetectionExperiment:

    SHARD_ATTACKS = ["A1_row_tamper", "A2_centroid_tamper",
                     "A3_data_hash_forgery", "A4_replay"]
    CHAIN_ATTACKS = ["A5_reorder", "A6_blockhash_tamper"]

    def __init__(self, output_dir, validator_counts=None,
                 n_trials=5, seed=42):
        self.output_dir = output_dir
        self.validator_counts = validator_counts or [10, 30, 50, 100, 200, 500]
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.shards_dir = os.path.join(output_dir, "shards")
        self.chain_path = os.path.join(output_dir, "DVD.json")


    def _apply_shard_attack(self, attack, blk, sv, rng):
        blk_attacked = deepcopy(blk)
        sv_attacked  = as_canonical_vectors(sv).copy()

        if attack == "A1_row_tamper":
            row_idx = int(rng.integers(0, sv_attacked.shape[0]))
            col_idx = int(rng.integers(0, sv_attacked.shape[1]))
            sv_attacked[row_idx, col_idx] += 1.0
        elif attack == "A2_centroid_tamper":
            new_c = np.asarray(blk_attacked["centroid"],
                               dtype=CANONICAL_CENTROID_DTYPE) + 1.0
            blk_attacked["centroid"] = new_c.tolist()
        elif attack == "A3_data_hash_forgery":
            h = blk_attacked["data_hash"]
            blk_attacked["data_hash"] = ("0" if h[0] != "0" else "1") + h[1:]
        elif attack == "A4_replay":
            blk_attacked["timestamp"] = time.time() - 600.0
        else:
            raise ValueError(f"Unknown shard attack: {attack}")

        return blk_attacked, sv_attacked

    def _time_shard_attack(self, attack, blk, sv, rng):
        blk_attacked, sv_attacked = self._apply_shard_attack(attack, blk, sv, rng)
        clean = build_shard_commitment(sv)
        expected_data_hash     = clean["data_hash"]
        expected_centroid_hash = clean["centroid_hash"]

        t0 = time.perf_counter()
        detected = False
        failed_check = None
        try:
            mv = run_minimum_verification(
                block                  = blk_attacked,
                shard_vectors          = sv_attacked,
                expected_data_hash     = expected_data_hash,
                expected_centroid_hash = expected_centroid_hash,
            )
            if not mv["passed"]:
                detected = True
                failed_check = mv["failed_check"]
        except VectorVerificationError:
            detected = True
            failed_check = "exception"
        latency_ms = (time.perf_counter() - t0) * 1000
        return detected, latency_ms, failed_check


    def _time_chain_attack(self, attack):
        bc = DVD(chain_file=self.chain_path)
        bc_attacked = deepcopy(bc)

        if attack == "A5_reorder":
            if len(bc_attacked.chain) >= 3:
                bc_attacked.chain[1], bc_attacked.chain[2] = \
                    bc_attacked.chain[2], bc_attacked.chain[1]
        elif attack == "A6_blockhash_tamper":
            if len(bc_attacked.chain) >= 2:
                blk = bc_attacked.chain[1]
                h = blk["block_hash"]
                blk["block_hash"] = ("0" if h[0] != "0" else "1") + h[1:]
        else:
            raise ValueError(f"Unknown chain attack: {attack}")

        t0 = time.perf_counter()
        res = bc_attacked.verify_chain()
        latency_ms = (time.perf_counter() - t0) * 1000
        detected = (res["valid"] is False)
        return detected, latency_ms, res.get("failure_reason")


    def run(self):
        if not os.path.exists(self.chain_path):
            print(f"[TamperExp] No DVD at {self.chain_path}; skipping")
            return None
        if not os.path.isdir(self.shards_dir):
            print(f"[TamperExp] No shards dir at {self.shards_dir}; skipping")
            return None

        bc = DVD(chain_file=self.chain_path)
        committed = [b for b in bc.chain if b.get("block_type") != "genesis"
                                          and "shard_id" in b]
        if not committed:
            print("[TamperExp] No committed shard blocks to attack; skipping")
            return None

        shards_by_id = {}
        for blk in committed:
            sid = int(blk["shard_id"])
            sv_path = os.path.join(self.shards_dir, f"shard_{sid}.npy")
            if os.path.exists(sv_path):
                shards_by_id[sid] = as_canonical_vectors(np.load(sv_path))

        if not shards_by_id:
            print("[TamperExp] No shard .npy files found; skipping")
            return None

        rng = np.random.default_rng(self.seed)
        rows = []

        print("\n" + "=" * 78)
        print("Tamper detection -- shard-level attacks (sweep validator count)")
        print("=" * 78)
        target_blocks = list(shards_by_id.items())

        for n_val in self.validator_counts:
            for attack in self.SHARD_ATTACKS:
                detect_count = 0
                latencies = []
                checks = set()
                for trial in range(self.n_trials):
                    sid, sv = target_blocks[trial % len(target_blocks)]
                    blk = next(b for b in committed if int(b["shard_id"]) == sid)
                    detected, lat_ms, failed = self._time_shard_attack(
                        attack, blk, sv, rng)
                    if detected:
                        detect_count += 1
                        if failed:
                            checks.add(failed)
                    latencies.append(lat_ms)

                row = {
                    "experiment_type":  "shard_attack",
                    "attack":           attack,
                    "n_validators":     n_val,
                    "n_blocks":         len(committed),
                    "n_trials":         self.n_trials,
                    "detection_rate":   detect_count / self.n_trials,
                    "latency_ms_mean":  float(np.mean(latencies)),
                    "latency_ms_std":   float(np.std(latencies)),
                    "failed_checks":    "|".join(sorted(checks)),
                }
                rows.append(row)
                print(f"  [n_val={n_val:>4}  {attack:24s}] "
                      f"detect={row['detection_rate']*100:5.1f}%  "
                      f"latency={row['latency_ms_mean']:7.3f} +/- "
                      f"{row['latency_ms_std']:.3f} ms  "
                      f"caught_by={row['failed_checks']}")

        print("\n" + "=" * 78)
        print(f"Tamper detection -- chain-level attacks (chain length = {len(bc.chain)})")
        print("=" * 78)
        for attack in self.CHAIN_ATTACKS:
            detect_count = 0
            latencies = []
            reasons = set()
            for _ in range(self.n_trials):
                detected, lat_ms, reason = self._time_chain_attack(attack)
                if detected:
                    detect_count += 1
                    if reason:
                        reasons.add(reason)
                latencies.append(lat_ms)

            row = {
                "experiment_type":  "chain_attack",
                "attack":           attack,
                "n_validators":     0,
                "n_blocks":         len(bc.chain),
                "n_trials":         self.n_trials,
                "detection_rate":   detect_count / self.n_trials,
                "latency_ms_mean":  float(np.mean(latencies)),
                "latency_ms_std":   float(np.std(latencies)),
                "failed_checks":    "|".join(sorted(reasons)),
            }
            rows.append(row)
            print(f"  [chain_len={len(bc.chain):>3}  {attack:24s}] "
                  f"detect={row['detection_rate']*100:5.1f}%  "
                  f"latency={row['latency_ms_mean']:7.3f} +/- "
                  f"{row['latency_ms_std']:.3f} ms  "
                  f"caught_by={row['failed_checks']}")

        out_csv = os.path.join(self.output_dir, "tamper_detection.csv")
        fieldnames = ["experiment_type", "attack", "n_validators", "n_blocks",
                      "n_trials", "detection_rate", "latency_ms_mean",
                      "latency_ms_std", "failed_checks"]
        import csv as _csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[TamperExp] Wrote {len(rows)} rows to {out_csv}")
        return rows


# Node recovery experiment
class RecoveryExperiment:

    def __init__(self, output_dir, subcluster_size=30,
                 missing_counts=None, recovering_fractions=None,
                 rows_per_commit=None, uplink_mbps=100.0, rtt_sec=0.002,
                 batch_blocks=128, verify_merkle=False, seed=42,
                 make_plot=True, commit_mode="epoch"):
        self.output_dir     = output_dir
        self.n_nodes        = int(subcluster_size)
        self.missing_counts = sorted(missing_counts or [10, 100, 1000, 10000])
        self.recovering_fractions = recovering_fractions or [0.10, 0.25, 0.40]
        self.rows_per_commit = rows_per_commit
        self.uplink_bps     = float(uplink_mbps) * 1e6 / 8.0
        self.rtt_sec        = float(rtt_sec)
        self.batch_blocks   = int(batch_blocks)
        self.verify_merkle  = bool(verify_merkle)
        if commit_mode not in ("epoch", "microbatch"):
            raise ValueError("commit_mode must be 'epoch' or 'microbatch'")
        self.commit_mode    = commit_mode
        self.seed           = int(seed)
        self.make_plot      = bool(make_plot)

        self.shards_dir  = os.path.join(output_dir, "shards")
        self.chain_path  = os.path.join(output_dir, "DVD.json")
        self.work_dir    = os.path.join(output_dir, "recovery_work")


    def _load_real_vectors(self):
        files = sorted(f for f in os.listdir(self.shards_dir) if f.endswith(".npy"))
        if not files:
            raise FileNotFoundError(f"No shard .npy files in {self.shards_dir}")
        parts = []
        for fn in files:
            arr = np.load(os.path.join(self.shards_dir, fn))
            if arr.ndim == 2 and arr.shape[0] > 0:
                parts.append(as_canonical_vectors(arr))
        if not parts:
            raise ValueError("All shards are empty -- nothing to commit")
        return np.vstack(parts)


    def _open_fork(self):
        os.makedirs(self.work_dir, exist_ok=True)
        fork_path = os.path.join(self.work_dir, "DVD_backlog.json")
        if os.path.exists(fork_path):
            os.remove(fork_path)
        bc = DVD(chain_file=fork_path)
        if os.path.exists(self.chain_path):
            with open(self.chain_path, "r", encoding="utf-8") as f:
                bc.chain = json.load(f)
        return bc, len(bc.chain)

    def _build_backlog_epoch(self, max_k):
        files = sorted(f for f in os.listdir(self.shards_dir) if f.endswith(".npy"))
        shard_ids, shard_vecs, templates = [], [], []
        for fn in files:
            sid = int(fn.replace("shard_", "").replace(".npy", ""))
            sv = as_canonical_vectors(np.load(os.path.join(self.shards_dir, fn)))
            if sv.ndim != 2 or sv.shape[0] == 0:
                continue
            shard_file = os.path.join(self.shards_dir, fn)
            templates.append(build_metadata_block(0, sid, sv, shard_file))
            shard_ids.append(sid)
            shard_vecs.append(sv)

        if not templates:
            raise ValueError("No non-empty shards to re-commit")

        bc, anchor_len = self._open_fork()
        n_sh = len(templates)
        epochs = math.ceil(max_k / n_sh)
        print(f"[RecoveryExp] Building {max_k} commitments as "
              f"{epochs} epoch(s) x {n_sh} real shard(s)...")

        t0 = time.perf_counter()
        micro_vectors = []
        _tick = max(1, max_k // 10)
        for i in range(max_k):
            j = i % n_sh
            blk = dict(templates[j])
            blk["epoch"] = i // n_sh
            bc.add_block(blk)
            micro_vectors.append(shard_vecs[j])
            if (i + 1) % _tick == 0 or (i + 1) == max_k:
                print(f"[RecoveryExp]   backlog {i + 1:,}/{max_k:,} blocks "
                      f"({100.0 * (i + 1) / max_k:.0f}%) "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)

        rows_each = int(np.mean([s.shape[0] for s in shard_vecs]))
        print(f"[RecoveryExp] Backlog built in {time.perf_counter() - t0:.1f}s "
              f"(chain length {len(bc.chain)}, ~{rows_each} vectors/commitment)")
        return bc, anchor_len, micro_vectors, rows_each

    def _build_backlog(self, X, max_k):
        n_rows, dim = X.shape
        rpc = self.rows_per_commit or max(1, n_rows // max_k)
        if n_rows < max_k:
            print(f"[RecoveryExp] WARNING: corpus has {n_rows} rows but "
                  f"{max_k} commitments were requested, so vectors are reused "
                  f"across commitments. Block count and block bytes -- the "
                  f"only quantities recovery latency depends on -- are "
                  f"unaffected, but do not describe these as {max_k} distinct "
                  f"vectors. commit_mode='epoch' avoids the issue entirely.")
        os.makedirs(self.work_dir, exist_ok=True)

        fork_path = os.path.join(self.work_dir, "DVD_backlog.json")
        if os.path.exists(fork_path):
            os.remove(fork_path)
        bc = DVD(chain_file=fork_path)
        if os.path.exists(self.chain_path):
            with open(self.chain_path, "r", encoding="utf-8") as f:
                bc.chain = json.load(f)
        anchor_len = len(bc.chain)

        print(f"[RecoveryExp] Building {max_k} real commitments "
              f"({rpc} vector(s) each, dim={dim}) from {n_rows} dataset rows...")
        t0 = time.perf_counter()
        cursor = 0
        micro_vectors = []
        for i in range(max_k):
            idx = [(cursor + j) % n_rows for j in range(rpc)]
            cursor = (cursor + rpc) % n_rows
            sv = X[idx, :]

            shard_file = os.path.join(self.work_dir, f"micro_{i % 16}.npy")
            blk = build_metadata_block(rank=0, shard_id=i % 16,
                                       shard_vectors=sv, shard_file=shard_file)
            bc.add_block(blk)
            micro_vectors.append(sv)

            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{max_k} commitments "
                      f"({time.perf_counter() - t0:.1f}s)")

        print(f"[RecoveryExp] Backlog built in {time.perf_counter() - t0:.1f}s "
              f"(chain length {len(bc.chain)})")
        return bc, anchor_len, micro_vectors, rpc


    def run(self):
        print("\n" + "=" * 78)
        print("Node recovery experiment (real dataset, real chain)")
        print("=" * 78)

        max_k = max(self.missing_counts)
        if self.commit_mode == "epoch":
            bc, anchor_len, micro_vectors, rpc = self._build_backlog_epoch(max_k)
            dim = micro_vectors[0].shape[1]
        else:
            X = self._load_real_vectors()
            bc, anchor_len, micro_vectors, rpc = self._build_backlog(X, max_k)
            dim = X.shape[1]

        rows = []
        for k in self.missing_counts:
            segment = bc.chain[anchor_len:anchor_len + k]

            onchain_bytes = sum(block_metadata_bytes(b) for b in segment)
            wire_bytes = len(json.dumps(segment, separators=(",", ":")).encode("utf-8"))
            per_block = wire_bytes / k

            recovered = DVD.__new__(DVD)
            recovered.chain_file = os.path.join(self.work_dir, "_verify.json")
            recovered.chain = bc.chain[:anchor_len] + segment

            best = float("inf")
            for _ in range(3):
                t = time.perf_counter()
                res = recovered.verify_chain()
                best = min(best, time.perf_counter() - t)
            if not res["valid"]:
                raise RuntimeError(f"backlog failed verify_chain: {res}")
            verify_sec = best

            merkle_sec = 0.0
            if self.verify_merkle:
                t = time.perf_counter()
                for blk, sv in zip(segment, micro_vectors[:k]):
                    verify_merkle_root(sv, blk["merkle_root"])
                merkle_sec = time.perf_counter() - t
            verify_total = verify_sec + merkle_sec

            for frac in self.recovering_fractions:
                n_rec = max(1, int(round(self.n_nodes * frac)))
                n_on  = max(1, self.n_nodes - n_rec)

                n_batches   = math.ceil(k / self.batch_blocks)
                rtt_total   = n_batches * self.rtt_sec
                transfer    = (n_rec * wire_bytes) / (n_on * self.uplink_bps)
                latency     = rtt_total + transfer + verify_total

                rows.append({
                    "missing_commitments":  k,
                    "recovering_pct":       int(round(frac * 100)),
                    "recovering_nodes":     n_rec,
                    "serving_nodes":        n_on,
                    "rows_per_commit":      rpc,
                    "vector_dim":           int(dim),
                    "commit_mode":          self.commit_mode,
                    "onchain_bytes_per_block": round(onchain_bytes / k, 1),
                    "wire_bytes_per_block": round(per_block, 1),
                    "bytes_per_node":       wire_bytes,
                    "total_bytes":          n_rec * wire_bytes,
                    "rtt_sec":              round(rtt_total, 6),
                    "transfer_sec":         round(transfer, 6),
                    "verify_chain_sec":     round(verify_sec, 6),
                    "verify_merkle_sec":    round(merkle_sec, 6),
                    "recovery_latency_sec": round(latency, 6),
                })
                print(f"  K={k:>6}  {int(frac*100):>2}% recovering  "
                      f"latency={latency:9.4f}s  "
                      f"bytes/node={wire_bytes/1e6:8.2f} MB  "
                      f"total={n_rec*wire_bytes/1e6:9.2f} MB")

        out_csv = os.path.join(self.output_dir, "recovery_latency.csv")
        fieldnames = list(rows[0].keys())
        import csv as _csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[RecoveryExp] Wrote {len(rows)} rows to {out_csv}")


# Pipeline runner
class DistributedKMeansRunner:
    def __init__(
        self,
        csv_path="data.csv", k=5, num_steps=100, seed=42,
        output_dir="output_csv", subcluster_size=30, fault_percentage=0.0,
        label_column="Activity", drop_columns=None,
        max_rows=None, feature_dtype=np.float64,
        kmeans_tol=1e-3, kmeans_verbose=True, kmeans_print_every=5,
        kmeans_balance_penalty=0.0,
        kmeans_fit_mode="coreset",
        coreset_auto_threshold=2_000_000,
        independent_verification=False,
        write_trace=False, validator_threads=4, max_drift_sec=60.0,
        run_tamper_experiment=True,
        tamper_validator_counts=None,
        tamper_n_trials=5,
        run_recovery_experiment=True,
        recovery_missing_counts=None,
        recovery_fractions=None,
        recovery_rows_per_commit=None,
        recovery_uplink_mbps=100.0,
        recovery_rtt_sec=0.002,
        recovery_batch_blocks=128,
        recovery_verify_merkle=False,
        recovery_commit_mode="epoch",
    ):
        self.csv_path         = csv_path
        self.parallel_read    = is_bigann(csv_path)
        self.k                = k
        self.num_steps        = num_steps
        self.seed             = seed
        self.output_dir       = output_dir
        self.subcluster_size  = subcluster_size
        self.fault_percentage = fault_percentage
        self.label_column     = label_column
        self.drop_columns     = drop_columns or ["subject", "Activity"]
        self.max_rows         = max_rows
        self.feature_dtype    = feature_dtype
        self.max_drift_sec    = max_drift_sec
        self.write_trace      = write_trace
        self.validator_threads= validator_threads
        self.run_tamper_experiment   = bool(run_tamper_experiment)
        self.tamper_validator_counts = tamper_validator_counts or [10, 30, 50, 100, 200, 500]
        self.tamper_n_trials         = int(tamper_n_trials)
        self.run_recovery_experiment  = bool(run_recovery_experiment)
        self.recovery_missing_counts  = recovery_missing_counts or [10, 100, 1000, 10000]
        self.recovery_fractions       = recovery_fractions or [0.10, 0.25, 0.40]
        self.recovery_rows_per_commit = recovery_rows_per_commit
        self.recovery_uplink_mbps     = float(recovery_uplink_mbps)
        self.recovery_rtt_sec         = float(recovery_rtt_sec)
        self.recovery_batch_blocks    = int(recovery_batch_blocks)
        self.recovery_verify_merkle   = bool(recovery_verify_merkle)
        self.recovery_commit_mode     = recovery_commit_mode
        self.kmeans_verbose   = bool(kmeans_verbose)
        self.kmeans_fit_mode  = str(kmeans_fit_mode).lower()
        self.coreset_auto_threshold = int(coreset_auto_threshold)
        self.independent_verification = bool(independent_verification)
        self.kmeans           = DistributedKMeans(
            k=k, num_steps=num_steps, seed=seed,
            tol=kmeans_tol, verbose=kmeans_verbose, print_every=kmeans_print_every,
            balance_penalty=kmeans_balance_penalty,
            penalty_decay=PENALTY_DECAY,
            final_penalty=FINAL_PENALTY,
            chunk_size=CHUNK_SIZE,
            final_relabel=False,
        )

    def load_data_rank0(self, rank):
        if rank != 0:
            return None, None
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        max_rows = getattr(self, "max_rows", None)
        dtype    = getattr(self, "feature_dtype", np.float64)

        if is_bigann(self.csv_path):
            n_total, d = fbin_info(self.csv_path)
            print(f"[Rank 0] fbin header: {n_total:,} vectors x {d} dims "
                  f"({np.dtype(bigann_dtype(self.csv_path)).name})")
            fused = read_fbin(self.csv_path, max_rows=max_rows, dtype=dtype)
            print(f"[Rank 0] Feature matrix: {fused.shape} "
                  f"({fused.dtype}, {fused.nbytes / 1024**3:.2f} GB)")
            return fused, None

        if self.csv_path.endswith(".fvecs"):
            n_total, d = fvecs_info(self.csv_path)
            print(f"[Rank 0] fvecs header: {n_total} vectors x {d} dims")
            fused = read_fvecs(self.csv_path, max_rows=max_rows, dtype=dtype)
            gb = fused.nbytes / 1024 ** 3
            print(f"[Rank 0] Feature matrix: {fused.shape} "
                  f"({fused.dtype}, {gb:.2f} GB)")
            return fused, None

        if self.csv_path.endswith(".npy"):
            fused = np.load(self.csv_path).astype(dtype)
            if max_rows is not None:
                fused = fused[:max_rows]
            lab_path = self.csv_path.replace(".npy", "_labels.npy")
            labels = np.load(lab_path, allow_pickle=True).astype(str) \
                     if os.path.exists(lab_path) else None
            if labels is not None and max_rows is not None:
                labels = labels[:max_rows]
            print(f"[Rank 0] Feature matrix: {fused.shape}")
            return fused, labels
        if self.csv_path.endswith((".hdf5", ".h5")):
            import h5py
            with h5py.File(self.csv_path, "r") as f:
                train = f["train"]
                fused = np.asarray(train[:max_rows] if max_rows else train,
                                   dtype=dtype)
            print(f"[Rank 0] Feature matrix: {fused.shape}")
            return fused, None

        df     = pd.read_csv(self.csv_path, low_memory=False)
        print(f"[Rank 0] CSV shape: {df.shape}")
        labels = df[self.label_column].astype(str).to_numpy() \
                 if self.label_column in df.columns else None
        feat   = df.drop(columns=[c for c in self.drop_columns if c in df.columns],
                         errors="ignore")
        feat   = feat.apply(pd.to_numeric, errors="coerce") \
                     .replace([np.inf, -np.inf], np.nan) \
                     .fillna(feat.mean(numeric_only=True)).fillna(0.0)
        fused  = feat.to_numpy(dtype=np.float64)
        print(f"[Rank 0] Feature matrix: {fused.shape}")
        return fused, labels
    def shard_data_bigann(self, comm):

        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(
                    f"[Rank 0] Dataset not found: {self.csv_path}"
                )

            n_file, d = fbin_info(self.csv_path)
            src_dtype = np.dtype(bigann_dtype(self.csv_path))

            if self.max_rows is None:
                n = n_file
            else:
                n = min(int(self.max_rows), n_file)

            expected_size = 8 + n_file * d * src_dtype.itemsize
            actual_size = os.path.getsize(self.csv_path)

            print(
                f"[Rank 0] fbin header: "
                f"{n_file:,} vectors x {d} dims "
                f"({src_dtype.name})",
                flush=True
            )

            print(
                f"[Rank 0] Using {n:,} rows",
                flush=True
            )

            print(
                f"[Rank 0] File size: "
                f"{actual_size / (1024 ** 3):.2f} GiB",
                flush=True
            )

            if actual_size != expected_size:
                raise RuntimeError(
                    f"[Rank 0] Dataset size mismatch.\n"
                    f"Expected: {expected_size:,} bytes\n"
                    f"Actual:   {actual_size:,} bytes\n"
                    f"The .fbin file appears incomplete or corrupted."
                )

        else:
            n = 0
            d = 0

        n, d = comm.bcast((n, d), root=0)

        base, rem = divmod(n, size)

        rows = [
            base + (1 if r < rem else 0)
            for r in range(size)
        ]

        starts = [0] * size

        for r in range(1, size):
            starts[r] = starts[r - 1] + rows[r - 1]

        local_rows = int(rows[rank])
        local_start = int(starts[rank])

        dtype = np.dtype(self.feature_dtype)

        local = np.empty(
            (local_rows, d),
            dtype=dtype
        )

        try:
            mpi_dtype = MPI._typedict[dtype.char]
        except KeyError:
            raise RuntimeError(
                f"Unsupported MPI dtype: {dtype}"
            )

        STREAM_ROWS = 1_000_000

        if rank == 0:

            if local_rows > 0:

                local[:, :] = read_fbin_range(
                    self.csv_path,
                    local_start,
                    local_rows,
                    dtype=dtype
                )

                print(
                    f"[Rank 0] loaded own rows "
                    f"[{local_start:,}, "
                    f"{local_start + local_rows:,})",
                    flush=True
                )

            for dest in range(1, size):

                dest_start = int(starts[dest])
                dest_rows = int(rows[dest])

                sent = 0

                while sent < dest_rows:

                    chunk_rows = min(
                        STREAM_ROWS,
                        dest_rows - sent
                    )

                    global_start = dest_start + sent

                    chunk = read_fbin_range(
                        self.csv_path,
                        global_start,
                        chunk_rows,
                        dtype=dtype
                    )

                    comm.Send(
                        [chunk.reshape(-1), mpi_dtype],
                        dest=dest,
                        tag=2000 + dest
                    )

                    sent += chunk_rows

                    print(
                        f"[Rank 0] sent Rank {dest}: "
                        f"{sent:,}/{dest_rows:,} rows",
                        flush=True
                    )

                    del chunk

        else:

            received = 0

            while received < local_rows:

                chunk_rows = min(
                    STREAM_ROWS,
                    local_rows - received
                )

                recv_view = local[
                    received:received + chunk_rows
                ]

                comm.Recv(
                    [recv_view.reshape(-1), mpi_dtype],
                    source=0,
                    tag=2000 + rank
                )

                received += chunk_rows

                print(
                    f"[Rank {rank}] received "
                    f"{received:,}/{local_rows:,} rows",
                    flush=True
                )

        comm.Barrier()

        print(
            f"[Rank {rank}] local shard: "
            f"rows [{local_start:,}, "
            f"{local_start + local_rows:,}) "
            f"-> {local.shape} "
            f"({local.nbytes / (1024 ** 3):.2f} GiB)",
            flush=True
        )

        return local, n, d
    
    def shard_data(self, global_fused, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            n, dim = int(global_fused.shape[0]), int(global_fused.shape[1])
            dt_char = np.dtype(global_fused.dtype).char
        else:
            n = dim = 0
            dt_char = ""
        n, dim, dt_char = comm.bcast((n, dim, dt_char), root=0)
        dt     = np.dtype(dt_char)
        mpi_dt = MPI._typedict[dt.char]

        base, rem = divmod(n, size)
        rows      = np.array([base + (1 if r < rem else 0) for r in range(size)],
                             dtype=np.int64)
        counts    = rows * dim
        displs    = np.zeros(size, dtype=np.int64)
        displs[1:] = np.cumsum(counts)[:-1]

        if counts.max() > np.iinfo(np.int32).max:
            raise RuntimeError(
                f"[shard_data] per-rank chunk is {counts.max():,} elements, above the "
                f"MPI int limit ({np.iinfo(np.int32).max:,}). Use more ranks or fewer rows."
            )

        sendbuf = (np.ascontiguousarray(global_fused).reshape(-1) if rank == 0 else None)
        local   = np.empty((int(rows[rank]), dim), dtype=dt)

        comm.Scatterv([sendbuf, counts.astype(np.int32), displs.astype(np.int32), mpi_dt],
                      [local.reshape(-1), mpi_dt], root=0)

        print(f"[Rank {rank}] local shard: {local.shape}")
        return local

    def shard_owner(self, shard_id, size):
        return shard_id % size

    def redistribute_shards(self, local_fused, labels, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        dim  = local_fused.shape[1]
        dt      = np.dtype(local_fused.dtype)
        mpi_dt  = MPI._typedict[dt.char]

        dest_of_sid = np.fromiter((self.shard_owner(s, size) for s in range(self.k)),
                                  dtype=np.int64, count=self.k)

        key    = dest_of_sid[labels] * self.k + labels
        perm   = np.argsort(key, kind="stable")
        packed = np.ascontiguousarray(local_fused[perm], dtype=dt)
        del perm, key

        counts_by_sid = np.bincount(labels, minlength=self.k).astype(np.int64)
        order  = sorted(range(self.k), key=lambda s: (int(dest_of_sid[s]), s))
        offset, _o = {}, 0
        for sid in order:
            offset[sid] = _o
            _o += int(counts_by_sid[sid])

        all_counts = np.array(comm.allgather(counts_by_sid), dtype=np.int64)

        mpi_int_max = int(np.iinfo(np.int32).max)
        max_chunk_bytes = 512 * 1024 * 1024
        chunk_rows = min(mpi_int_max // dim,
                         max_chunk_bytes // (dim * dt.itemsize))
        if chunk_rows < 1:
            raise RuntimeError(
                f"[redistribute_shards] one row ({dim * dt.itemsize:,} bytes) "
                "is too large for a bounded MPI transfer."
            )

        tag_ub = comm.Get_attr(MPI.TAG_UB)
        if tag_ub is not None and self.k - 1 > int(tag_ub):
            raise RuntimeError(
                f"[redistribute_shards] k={self.k} requires message tag "
                f"{self.k - 1}, but MPI_TAG_UB is {int(tag_ub)}."
            )

        lsm = {}
        for sid in range(self.k):
            root     = int(dest_of_sid[sid])
            src_rows = all_counts[:, sid]
            nloc     = int(counts_by_sid[sid])
            sblk     = packed[offset[sid]:offset[sid] + nloc]

            if rank == root:
                recv = np.empty((int(src_rows.sum()), dim), dtype=dt)
                dst_row_offsets = np.zeros(size, dtype=np.int64)
                dst_row_offsets[1:] = np.cumsum(src_rows)[:-1]
            else:
                recv = None

            for src in range(size):
                rows_from_src = int(src_rows[src])

                if rank == root:
                    dst0 = int(dst_row_offsets[src])
                    if src == root:
                        if rows_from_src != nloc:
                            raise RuntimeError(
                                f"[redistribute_shards] shard {sid}: local count "
                                f"mismatch ({nloc:,} != {rows_from_src:,})."
                            )
                        for start in range(0, rows_from_src, chunk_rows):
                            stop = min(start + chunk_rows, rows_from_src)
                            np.copyto(recv[dst0 + start:dst0 + stop],
                                      sblk[start:stop], casting="no")
                    else:
                        for start in range(0, rows_from_src, chunk_rows):
                            stop = min(start + chunk_rows, rows_from_src)
                            target = recv[dst0 + start:dst0 + stop]
                            comm.Recv([target.reshape(-1), mpi_dt],
                                      source=src, tag=sid)

                elif rank == src:
                    for start in range(0, nloc, chunk_rows):
                        stop = min(start + chunk_rows, nloc)
                        comm.Send([sblk[start:stop].reshape(-1), mpi_dt],
                                  dest=root, tag=sid)

            if rank == root:
                lsm[sid] = np.ascontiguousarray(recv, dtype=CANONICAL_VECTOR_DTYPE)

        owned = [sid for sid in range(self.k) if int(dest_of_sid[sid]) == rank]
        for sid in owned:
            if sid not in lsm:
                lsm[sid] = np.empty((0, dim), dtype=CANONICAL_VECTOR_DTYPE)

        return lsm, owned

    def fit_kmeans(self, local_fused, comm):
        rank = comm.Get_rank()

        X = np.ascontiguousarray(local_fused, dtype=np.float32)
        if NORMALIZE:
            if X is local_fused or X.base is local_fused:
                X = X.copy()
            X = l2_normalize(X)

        nb = np.array([float(X.shape[0])], dtype=np.float64)
        ng = np.empty_like(nb)
        comm.Allreduce(nb, ng, op=MPI.SUM)
        n_global = int(ng[0])

        self.stage_times = {}
        mode = self.kmeans_fit_mode
        if mode == "auto":
            mode = "coreset" if n_global > self.coreset_auto_threshold else "full"

        t0 = MPI.Wtime()
        if mode == "full":
            Xc_local, wc_local = X, None
            if rank == 0 and self.kmeans_verbose:
                print(f"[fit] FULL-DATA clustering: all {n_global:,} vectors at "
                      f"{X.shape[1]}-d take part in every Lloyd iteration "
                      f"(no coreset, no sampling)", flush=True)
        else:
            m_core = coreset_size(n_global, self.k, X.shape[1])
            Xc_local, wc_local, _ = build_coreset(X, m_core, comm, seed=self.seed)
            if rank == 0 and self.kmeans_verbose:
                print(f"[coreset] {m_core:,} weighted rows at full {X.shape[1]}-d "
                      f"({100.0 * m_core / max(n_global, 1):.1f}% of corpus, "
                      f"alpha={CORESET_ALPHA})", flush=True)
        comm.Barrier()
        self.stage_times["km_coreset"] = MPI.Wtime() - t0

        t0 = MPI.Wtime()
        _, centroids_f64, _ = self.kmeans.run(Xc_local, comm, weights=wc_local)
        comm.Barrier()
        self.stage_times["km_lloyd"] = MPI.Wtime() - t0
        centroids = np.ascontiguousarray(centroids_f64, dtype=np.float32)

        t0 = MPI.Wtime()
        if BALANCED:
            local_labels = balanced_assign(X, centroids, comm,
                                           target_cv=BALANCE_TARGET_CV,
                                           verbose=(rank == 0 and self.kmeans_verbose))
        else:
            local_labels = self.kmeans.assign_clusters(X, centroids)

        comm.Barrier()
        self.stage_times["km_assign"] = MPI.Wtime() - t0

        local_counts = np.bincount(local_labels, minlength=self.k).astype(np.int64)
        cbuf = local_counts.astype(np.float64)
        gbuf = np.empty_like(cbuf)
        comm.Allreduce(cbuf, gbuf, op=MPI.SUM)
        global_counts = gbuf.astype(np.int64)

        return local_labels, centroids_f64, global_counts, local_counts

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "shards"), exist_ok=True)

        comm.Barrier()
        t_total = MPI.Wtime()

        _t = MPI.Wtime()
        if self.parallel_read:
            global_fused = global_labels = None
            local_fused, n_global, dim = self.shard_data_bigann(comm)
            comm.Barrier()
            t_load, t_scatter = MPI.Wtime() - _t, 0.0
            if rank == 0:
                print(f"Distributed K-means: {size} MPI ranks, "
                      f"global shape ({n_global:,}, {dim}) read in parallel")
        else:
            global_fused, global_labels = self.load_data_rank0(rank)
            comm.Barrier()
            t_load = MPI.Wtime() - _t
            if rank == 0:
                print(f"Distributed K-means: {size} MPI ranks, "
                      f"global shape {global_fused.shape}")

            _t = MPI.Wtime()
            local_fused = self.shard_data(global_fused, comm)
            comm.Barrier()
            t_scatter = MPI.Wtime() - _t

        comm.Barrier()
        t_km = MPI.Wtime()
        self.kmeans.verbose = self.kmeans_verbose and (rank == 0)
        local_labels, centroids, global_counts, _ = self.fit_kmeans(
            local_fused, comm
        )
        comm.Barrier()
        kmeans_time = comm.reduce(MPI.Wtime() - t_km, op=MPI.MAX, root=0)

        _t = MPI.Wtime()
        if self.parallel_read:
            gathered_labels = None
            _m = compute_shard_metrics(
                np.ascontiguousarray(local_fused, dtype=np.float32),
                local_labels, self.k, comm)
            dist_quality = {
                "intra_shard_cosine": float(_m["intra_shard_cosine"]),
                "inter_shard_cosine": float(_m["inter_shard_cosine"]),
                "separation_gap":     float(_m["intra_shard_cosine"]
                                            - _m["inter_shard_cosine"]),
                "balance_ratio":      float(_m["balance_ratio"]),
                "cv_shard_size":      float(_m["shard_size_cv"]),
                "purity":             float(_m["purity"]),
                "cohesion_lift":      float(_m["cohesion_lift"]),
                "point_margin":       float(_m["point_margin"]),
                "shard_quality_score": float(_m["shard_quality_score"]),
                "noise_count":        0,
                "noise_ratio":        0.0,
                "per_shard":          [],
            }
        else:
            gathered_labels = comm.gather(local_labels, root=0)
            dist_quality = None
        t_quality_dist = MPI.Wtime() - _t

        _t = MPI.Wtime()
        local_shard_map, owned = self.redistribute_shards(local_fused, local_labels, comm)
        comm.Barrier()
        t_redistribute = MPI.Wtime() - _t

        engine = PushPullHashConsensus(
            output_dir       = self.output_dir,
            seed             = self.seed,
            write_trace      = self.write_trace,
            validator_threads= self.validator_threads,
            max_drift_sec    = self.max_drift_sec,
        )

        _t = MPI.Wtime()
        commitments = {sid: build_shard_commitment(local_shard_map[sid])
                       for sid in owned}

        if self.independent_verification:
            all_vctxs        = build_all_verification_ctxs(
                comm, rank, size, local_shard_map, owned
            )
            precomputed_by_sid = {}
        else:
            all_vctxs = {
                sid: {
                    "shard_id":      sid,
                    "data_hash":     c["data_hash"],
                    "centroid_hash": c["centroid_hash"],
                    "merkle_root":   c["merkle_root"],
                    "computed_by":   rank,
                }
                for sid, c in commitments.items()
            }
            precomputed_by_sid = {
                sid: {"data_hash":     c["data_hash"],
                      "centroid_hash": c["centroid_hash"],
                      "merkle_root":   c["merkle_root"]}
                for sid, c in commitments.items()
            }

        comm.Barrier()
        t_vctx = MPI.Wtime() - _t

        t_shard_io   = 0.0
        t_consensus  = 0.0
        t_verify     = 0.0
        t_shard_wall = 0.0
        local_results = []
        for shard_id in owned:
            _t = MPI.Wtime()
            sv         = local_shard_map[shard_id]
            shard_file = os.path.join(self.output_dir, "shards", f"shard_{shard_id}.npy")
            np.save(shard_file, sv)
            print(f"[Rank {rank}] Shard {shard_id}: {len(sv)} points saved")

            block = build_metadata_block(rank, shard_id, sv, shard_file,
                                         commitment=commitments.get(shard_id))
            t_shard_io += MPI.Wtime() - _t
            vctx  = all_vctxs.get(shard_id, {
                "shard_id":      shard_id,
                "computed_by":   rank,
                "data_hash":     block["data_hash"],
                "centroid_hash": block["centroid_hash"],
                "merkle_root":   block["merkle_root"],
            })

            _t = MPI.Wtime()
            cr = engine.run(
                block, vctx,
                rank             = rank,
                fault_percentage = self.fault_percentage,
                shard_vectors    = sv,
                sub_cluster_size = self.subcluster_size,
                precomputed      = precomputed_by_sid.get(shard_id),
            )
            t_consensus  += cr["consensus_time_sec"]
            t_verify     += cr["verification_time_sec"]
            t_shard_wall += MPI.Wtime() - _t

            cb = None
            if cr["committed"]:
                cb = {
                    **block,
                    "verification_ctx_from": vctx.get("computed_by"),
                    "prepare_yes":             cr["prepare_yes"],
                    "commit_yes":              cr["commit_yes"],
                    "phase":                   cr["phase"],
                    "prepare_time_sec":        cr["prepare_time_sec"],
                    "commit_time_sec":         cr["commit_time_sec"],
                    "push_time_sec":           cr.get("push_time_sec", cr["prepare_time_sec"]),
                    "pull_time_sec":           cr.get("pull_time_sec", cr["commit_time_sec"]),
                    "push_pull_time_sec":      cr.get("push_pull_time_sec", cr["prepare_time_sec"] + cr["commit_time_sec"]),
                    "consensus_time_sec":      cr["consensus_time_sec"],
                    "verification_time_sec":   cr["verification_time_sec"],
                    "total_shard_time_sec":    cr["total_shard_time_sec"],
                    "min_validation_time_sec": cr["min_validation_time_sec"],
                    "avg_validation_time_sec": cr["avg_validation_time_sec"],
                    "max_validation_time_sec": cr["max_validation_time_sec"],
                }

            local_results.append({
                "shard_id":             shard_id,
                "committed_block":      cb,
                "consensus_result":     cr,
                "consensus_time_local":    cr["consensus_time_sec"],
                "verification_time_local": cr["verification_time_sec"],
                "total_time_local":        cr["total_shard_time_sec"],
                "shard_summary_item": {
                    "shard_id":           shard_id,
                    "num_points":         block["num_points"],
                    "vector_dim":         block["vector_dim"],
                    "timestamp":          block["timestamp"],
                    "data_hash":          block["data_hash"],
                    "centroid_hash":      block["centroid_hash"],
                    "merkle_root":        block["merkle_root"],
                    "merkle_depth":       block["merkle_depth"],
                    "merkle_leaves_ref":  block["merkle_leaves_ref"],
                    "offchain_ref":       block["offchain_ref"],
                    "ctx_from_rank":      vctx.get("computed_by"),
                    "committed":          bool(cr["committed"]),
                    "push_time_sec":      cr.get("push_time_sec", cr.get("prepare_time_sec", 0.0)),
                    "pull_time_sec":      cr.get("pull_time_sec", cr.get("commit_time_sec", 0.0)),
                    "push_pull_time_sec": cr.get("push_pull_time_sec", cr.get("prepare_time_sec", 0.0) + cr.get("commit_time_sec", 0.0)),
                    "consensus_time_sec":    cr["consensus_time_sec"],
                    "verification_time_sec": cr["verification_time_sec"],
                    "total_shard_time_sec":  cr["total_shard_time_sec"],
                }
            })

        t_vctx_g        = comm.reduce(t_vctx,        op=MPI.MAX, root=0)
        t_shard_io_g    = comm.reduce(t_shard_io,    op=MPI.MAX, root=0)
        t_consensus_g   = comm.reduce(t_consensus,   op=MPI.MAX, root=0)
        t_verify_g      = comm.reduce(t_verify,      op=MPI.MAX, root=0)
        t_shard_wall_g  = comm.reduce(t_shard_wall,  op=MPI.MAX, root=0)
        km_coreset_g    = comm.reduce(self.stage_times.get("km_coreset", 0.0), op=MPI.MAX, root=0)
        km_lloyd_g      = comm.reduce(self.stage_times.get("km_lloyd",   0.0), op=MPI.MAX, root=0)
        km_assign_g     = comm.reduce(self.stage_times.get("km_assign",  0.0), op=MPI.MAX, root=0)

        all_nested = comm.gather(local_results, root=0)

        if rank == 0:
            shard_blocks, shard_crs, shard_cts, shard_summary = [], [], [], []
            shard_vts, shard_tts = [], []

            for rr in all_nested:
                for res in rr:
                    shard_crs.append(res["consensus_result"])
                    shard_cts.append(res["consensus_time_local"])
                    shard_vts.append(res["verification_time_local"])
                    shard_tts.append(res["total_time_local"])
                    shard_summary.append(res["shard_summary_item"])
                    if res["committed_block"]:
                        shard_blocks.append(res["committed_block"])

            shard_blocks  = sorted(shard_blocks,  key=lambda x: x["shard_id"])
            shard_summary = sorted(shard_summary, key=lambda x: x["shard_id"])

            _t = MPI.Wtime()
            bc = DVD(chain_file=os.path.join(self.output_dir, "DVD.json"))
            for blk in shard_blocks:
                bc.add_block(blk)
            bc.save()

            write_json(os.path.join(self.output_dir, "shard_summary.json"), shard_summary)
            np.save(os.path.join(self.output_dir, "centroids.npy"), centroids)

            t_chain = MPI.Wtime() - _t

            sharding_quality = dist_quality
            t_quality        = t_quality_dist

            if gathered_labels is not None:
                acl = np.concatenate(gathered_labels)
                np.save(os.path.join(self.output_dir, "cluster_labels.npy"), acl)

                _t = MPI.Wtime()
                sharding_quality = compute_centroid_based_sharding_quality(global_fused, acl)
                t_quality = MPI.Wtime() - _t

                if global_labels is not None:
                    pd.DataFrame({"cluster_id": acl, self.label_column: global_labels}).to_csv(
                        os.path.join(self.output_dir, "cluster_vs_activity.csv"), index=False
                    )

            if sharding_quality is not None:
                write_json(os.path.join(self.output_dir, "sharding_quality.json"),
                           sharding_quality)

                q_by_sid = {q["shard_id"]: q for q in sharding_quality.get("per_shard", [])}
                for item in shard_summary:
                    q = q_by_sid.get(item["shard_id"])
                    if q:
                        item["intra_shard_cosine"] = q["intra_shard_cosine"]

                write_json(os.path.join(self.output_dir, "shard_summary.json"), shard_summary)

            total_time = (MPI.Wtime() - t_total) - t_quality
            min_ct = min(shard_cts) if shard_cts else 0.0
            max_ct = max(shard_cts) if shard_cts else 0.0
            min_vf = min(shard_vts) if shard_vts else 0.0
            max_vf = max(shard_vts) if shard_vts else 0.0
            min_tt = min(shard_tts) if shard_tts else 0.0
            max_tt = max(shard_tts) if shard_tts else 0.0
            avg_tt = float(np.mean(shard_tts)) if shard_tts else 0.0
            min_vt = min(r.get("min_validation_time_sec", 0.0) for r in shard_crs) if shard_crs else 0.0
            avg_vt = float(np.mean([r.get("avg_validation_time_sec", 0.0) for r in shard_crs])) if shard_crs else 0.0
            max_vt = max(r.get("max_validation_time_sec", 0.0) for r in shard_crs) if shard_crs else 0.0

            _pipeline = (MPI.Wtime() - t_total) - t_quality
            print("\n" + "=" * 62)
            print("STAGE TIMING BREAKDOWN (wall seconds, max across ranks)")
            print("=" * 62)
            _rows = [
                ("load (parallel read)" if self.parallel_read
                 else "load (rank 0 read)",   t_load),
                ("scatter to ranks",          t_scatter),
                ("K-MEANS total",             kmeans_time),
                ("    - coreset build",       km_coreset_g),
                ("    - Lloyd on coreset",    km_lloyd_g),
                ("    - final assignment",    km_assign_g),
                ("redistribute shards",       t_redistribute),
                ("hash + Merkle (vctx)",      t_vctx_g),
                ("shard save + block build",  t_shard_io_g),
                ("CONSENSUS (verify+vote+commit)", t_shard_wall_g),
                ("DVD + json write",   t_chain),
            ]
            _acc = t_load + t_scatter + kmeans_time + t_redistribute + \
                   t_vctx_g + t_shard_io_g + t_shard_wall_g + t_chain
            for _n, _v in _rows:
                _pct = 100.0 * _v / max(_pipeline, 1e-9)
                _ind = _n.startswith("    ")
                print(f"  {_n:<28} {_v:9.3f} s   {'' if _ind else f'{_pct:5.1f}%'}")
            print(f"  {'-' * 56}")
            print(f"  {'accounted for':<28} {_acc:9.3f} s   "
                  f"{100.0 * _acc / max(_pipeline, 1e-9):5.1f}%")
            print(f"  {'unattributed remainder':<28} {_pipeline - _acc:9.3f} s   "
                  f"{100.0 * (_pipeline - _acc) / max(_pipeline, 1e-9):5.1f}%")
            print(f"  {'PIPELINE TOTAL':<28} {_pipeline:9.3f} s   100.0%")
            print(f"  {'-' * 56}")
            print(f"  {'[excluded] sharding-quality':<28} {t_quality:9.3f} s   "
                  f"offline diagnostic, not in total")
            print("=" * 62)

            print("\nFinal shard counts:")
            for i, c in enumerate(global_counts):
                print(f"  Shard {i}: {int(c)} points")

            print("\nConsensus summary:")
            for s, r in zip(shard_summary, shard_crs):
                phase = r["phase"]
                extra = f"failed={r.get('failed_check','')} | " if "REJECT" in phase else ""
                print(
                    f"Shard {s['shard_id']} | committed={r['committed']} | "
                    f"ctx_from=rank{s.get('ctx_from_rank','?')} | "
                    f"merkle_depth={s['merkle_depth']} | "
                    f"prepare_yes={r['prepare_yes']} | quorum={r.get('quorum','?')} | "
                    f"fault_status={r.get('fault_tolerance_status','?')} | phase={phase} | {extra}"
                    f"avg_val={r.get('avg_validation_time_sec',0.0)*1000:.4f}ms | "
                    f"push={r.get('push_time_sec', r.get('prepare_time_sec',0.0))*1000:.4f}ms | "
                    f"pull={r.get('pull_time_sec', r.get('commit_time_sec',0.0))*1000:.4f}ms | "
                    f"push_pull={r.get('push_pull_time_sec', r.get('prepare_time_sec',0.0)+r.get('commit_time_sec',0.0))*1000:.4f}ms | "
                    f"consensus={r['consensus_time_sec']*1000:.4f}ms | "
                    f"verify={r.get('verification_time_sec',0.0)*1000:.4f}ms | "
                    f"total={r.get('total_shard_time_sec',0.0)*1000:.4f}ms"
                )

            print("\nPer-block on-chain metadata bytes:")
            for i, blk in enumerate(bc.chain[1:], start=1):
                print(f"  Block {i}: {block_metadata_bytes(blk)} bytes")

            if sharding_quality is not None:
                print("\nSemantic sharding quality:")
                print(f"  Intra-Shard Cosine Similarity: {sharding_quality['intra_shard_cosine']:.6f}  (higher is better)")
                print(f"  Inter-Shard Cosine Similarity: {sharding_quality['inter_shard_cosine']:.6f}  (lower is better)")
                print(f"  Separation Gap:                 {sharding_quality['separation_gap']:.6f}  (higher is better)")
                print(f"  Shard Balance Ratio:            {sharding_quality['balance_ratio']:.6f}  (closer to 1 is better)")
                print(f"  Shard Size CV:                   {sharding_quality['cv_shard_size']:.6f}  (closer to 0 is better)")
                if "purity" in sharding_quality:
                    print(f"  Purity:                          {sharding_quality['purity']:.6f}  (1.0 = nothing displaced by pricing)")
                per_shard = sharding_quality.get("per_shard", [])
                if per_shard:
                    print("\nPer-shard intra-shard cosine similarity:")
                    for q in per_shard:
                        print(f"  Shard {q['shard_id']}: intra_cosine={q['intra_shard_cosine']:.6f}, points={q['num_points']}")
                else:
                    print("  (per-shard cohesion not computed on the parallel-read path)")

            fault_summary = majority_fault_tolerance_summary(self.subcluster_size)
            print("\nFault tolerance setting:")
            print(f"  Validators per shard:       {fault_summary['validators']}")
            print(f"  Max Byzantine faults f:     {fault_summary['max_faulty_nodes_for_majority']}")
            print(f"  BFT commit quorum 2f + 1:   {fault_summary['majority_commit_quorum']}")
            print(f"  Quorum rule:                {fault_summary['quorum_rule']}")
            print(f"  Injected fault percentage:  {self.fault_percentage * 100:.2f}%")

            total_vectors = len(global_labels) if global_labels is not None \
                            else int(np.sum(global_counts))
            _mean_pts = (total_vectors / max(len(shard_cts), 1)) if shard_cts else 0

            print(f"\nK-means Time:           {kmeans_time:.4f} sec")
            print("\n--- Consensus (verification + voting + commit) ---")
            print(f"Consensus Time (min):   {min_tt * 1000:.4f} ms")
            print(f"Consensus Time (avg):   {avg_tt * 1000:.4f} ms")
            print(f"Consensus Time (max):   {max_tt * 1000:.4f} ms")
            print(f"Shards measured:        {len(shard_tts)}")
            if _mean_pts > 0 and avg_tt > 0:
                print(f"Per vector (avg shard): {avg_tt * 1e6 / _mean_pts:.4f} us  "
                      f"(~{_mean_pts:,.0f} vectors/shard)")
            print(f"\nExecution Time:         {total_time:.4f} sec")
            print(f"Throughput:             {total_vectors / total_time:.2f} vectors/sec")

            if self.run_tamper_experiment:
                exp = TamperDetectionExperiment(
                    output_dir       = self.output_dir,
                    validator_counts = self.tamper_validator_counts,
                    n_trials         = self.tamper_n_trials,
                    seed             = self.seed,
                )
                _te = MPI.Wtime()
                exp.run()
                print(f"\n[timing] TAMPER EXPERIMENT total: {MPI.Wtime() - _te:.3f} s")

            if self.run_recovery_experiment:
                _rc = sorted({min(int(c), int(total_vectors))
                              for c in self.recovery_missing_counts
                              if int(c) > 0})
                if _rc != sorted(self.recovery_missing_counts):
                    print(f"[RecoveryExp] missing_counts clamped to corpus size "
                          f"({total_vectors:,} vectors): "
                          f"{sorted(self.recovery_missing_counts)} -> {_rc}")

                rexp = RecoveryExperiment(
                    output_dir           = self.output_dir,
                    subcluster_size      = self.subcluster_size,
                    missing_counts       = _rc,
                    recovering_fractions = self.recovery_fractions,
                    rows_per_commit      = self.recovery_rows_per_commit,
                    uplink_mbps          = self.recovery_uplink_mbps,
                    rtt_sec              = self.recovery_rtt_sec,
                    batch_blocks         = self.recovery_batch_blocks,
                    verify_merkle        = self.recovery_verify_merkle,
                    commit_mode          = self.recovery_commit_mode,
                    seed                 = self.seed,
                )
                _te = MPI.Wtime()
                rexp.run()
                print(f"\n[timing] RECOVERY EXPERIMENT total: {MPI.Wtime() - _te:.3f} s")


if __name__ == "__main__":
    import argparse

    def _rows(text):
        t = str(text).strip().lower().replace("_", "").replace(",", "")
        if t in ("all", "full", "-1", "0"):
            return None
        mult = 1
        if t.endswith("k"):
            mult, t = 1_000, t[:-1]
        elif t.endswith("m"):
            mult, t = 1_000_000, t[:-1]
        return int(float(t) * mult)

    ap = argparse.ArgumentParser(description="Distributed sharding pipeline")
    ap.add_argument("--rows", type=_rows, default=100_000,
                    help="number of vectors to process (e.g. 100, 100k, 1M, all)")
    ap.add_argument("--k", type=int, default=5, help="number of shards")
    ap.add_argument("--data", default=BIGANN_PATH,
                    help="input path (.fbin/.u8bin/.i8bin, .fvecs, .npy, .h5, .csv)")
    ap.add_argument("--output", default="output_t2i", help="output directory")
    ap.add_argument("--fit-mode", default="coreset", choices=["coreset", "full", "auto"],
                    help="coreset = weighted subsample (default, fast); "
                         "full = every vector takes part in every iteration")
    ap.add_argument("--independent-verification", action="store_true",
                    help="re-enable the duplicate cross-rank hashing pass "
                         "(ships raw shards over MPI -- breaks above 2 GB/shard)")
    ap.add_argument("--validators", type=int, default=30,
                    help="validators per shard (sub_cluster_size)")
    ap.add_argument("--fault-pct", type=float, default=0.0,
                    help="fraction of validators that vote NO, 0.0-1.0")
    ap.add_argument("--experiments", action="store_true",
                    help="run the tamper and recovery experiments")
    args = ap.parse_args()

    runner = DistributedKMeansRunner(
        csv_path          = args.data,
        k                 = args.k,
        num_steps         = 30,
        seed              = 42,
        output_dir        = args.output,
        subcluster_size   = args.validators,
        fault_percentage  = args.fault_pct,
        label_column      = None,
        drop_columns      = [],
        max_rows          = args.rows,
        feature_dtype     = np.float32,
        kmeans_tol        = 1e-3,
        kmeans_verbose    = True,
        kmeans_print_every= 5,
        kmeans_fit_mode   = args.fit_mode,
        independent_verification = args.independent_verification,
        kmeans_balance_penalty = 0.0,
        write_trace       = False,
        validator_threads = 4,
        max_drift_sec     = 60.0,
        run_tamper_experiment   = args.experiments,
        tamper_validator_counts = [30, 60, 90, 120, 150],
        tamper_n_trials         = 5,
        run_recovery_experiment  = args.experiments,
        recovery_missing_counts  = [10, 100, 1000, 10000],
        recovery_fractions       = [0.10, 0.25, 0.40],
        recovery_rows_per_commit = None,
        recovery_uplink_mbps     = 100.0,
        recovery_rtt_sec         = 0.002,
        recovery_batch_blocks    = 128,
        recovery_verify_merkle   = False,
        recovery_commit_mode     = "epoch",
    )
    runner.execute()
