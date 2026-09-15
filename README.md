
**Shard-Level Verifiability as a Native Vector-Database Abstraction**

## Features

**Verifiable Semantic Shard Construction**
Partitions high-dimensional vectors into similarity-preserving, load-balanced shards using a weighted coreset for fast convergence at scale across distributed nodes. Commitment generation is integrated directly into shard materialization, so every shard emits a compact, independently verifiable summary as it is formed.

**LiteQuorum Integrity Protocol with Fault Injection**
A three-phase quorum protocol over MPI that verifies a shard once against an independently computed context, then reaches majority agreement by exchanging only fixed-size digest tuples. Validator count and fault rate are configurable, allowing sub-cluster committee verification to be evaluated under crash failures up to the majority boundary.

**Verification Synchronization**
A push-pull mechanism that propagates accepted commitments to participating verifiers and lets recovering nodes identify and retrieve only the commitments they missed, so recovery cost scales with backlog depth rather than with the length of the verification history.

**Hybrid Data-Plane / Verification-Layer Storage**
Raw vector shards remain in the distributed data plane, while only compact shard commitments enter the verification layer, providing tamper-evident provenance at a storage cost that scales with shard count rather than vector volume.

**Scalable Ingestion for 100M+ Vector Datasets**
Reads big-ann-benchmarks-formatted datasets through parallel memory-mapped row ranges, so each rank loads only its own slice and no single node ever holds the full corpus.

## Dataset
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
After installing the dependencies and downloading a dataset (see Dataset above), you can run the project using `mpiexec`. Here's an example with 5 processes:
 
```
mpiexec -n 5 python DVD.py --data base.1B.fbin.crop_nb_100000000
```
 
To run on fixed vectors, set `--rows`. For example, 10M vectors:
 
```
mpiexec -n 5 python DVD.py --data base.1B.fbin.crop_nb_100000000 --rows 10M
```
 
Run on HPC with SLURM
If you're working in an HPC environment, you can use the provided SLURM script to run your job.
Submit the Job
 
```
sbatch run_job.slurm
```
