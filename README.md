DVD
DVD: Commitment-Aware Semantic Sharding for Large-Scale High-Dimensional Vector Data Management

Features

* Coreset-Accelerated Semantic Clustering via Parallel K-Means
Partitions high-dimensional vectors into similarity-preserving, load-balanced shards using a weighted coreset for fast convergence at scale across distributed nodes.
* Configurable LiteQuorum Consensus with Fault Injection
Implements a two-phase quorum consensus protocol over MPI, with an adjustable validator count and injectable fault rate to simulate sub-cluster committee-based consensus under Byzantine conditions.
* Hybrid On-Chain/Off-Chain Ledger Management
Raw vector shard data resides in off-chain distributed storage, while only lightweight shard metadata blocks are committed on-chain, ensuring tamper-evident provenance without excessive storage overhead.
* Scalable Ingestion for 100M+ Vector Datasets
Reads big-ann-formatted datasets via parallel memory-mapped row ranges, so each rank loads only its own slice and no single node ever holds the full corpus.
* Optional Tamper-Detection & Recovery Diagnostics
Measures tamper-detection latency under simulated shard and chain attacks, and node-recovery latency after simulated dropout, enabled with a single flag.

Dataset
DVD defaults to a slice of the [big-ann-benchmarks](https://big-ann-benchmarks.com/) text2image-1B dataset (200-dimensional float vectors), read as memory-mapped `.fbin` files so no single rank has to hold the full corpus in memory. It also accepts `.fvecs`, `.npy`, `.hdf5`/`.h5`, and `.csv` input via `--data`, so any dataset in one of those formats can be used in place of the default.

```
python create_dataset.py --dataset text2image-100M
```

Point `--data` at the resulting file to run against it:

```
mpiexec -n 4 python DVD.py --data base.1B.fbin.crop_nb_100000000
```

Development Setup
DVD should be run using python. First install [python](https://www.python.org/downloads/)
DVD is integrated with MPI Then install [mpi4py](https://github.com/mpi4py/mpi4py/)
To clone the code to your target directory

```
git clone https://github.com/doraautomation/DVD
cd DVD
```

Install all required package.

```
pip install -r requirements.txt
```

Run the Project Locally
After installing the dependencies, you can run the project using `mpiexec`. Here's an example with 4 processes:

```
mpiexec -n 4 python DVD.py
```

Run on HPC with SLURM
If you're working in an HPC environment, you can use the provided SLURM script to run your job.
Submit the Job

```
sbatch run_job.slurm
```
