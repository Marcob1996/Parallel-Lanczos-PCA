# Lanczos Fast Data Visualization (LFDV)

For my AMSC 763: Advanced Numerical Linear Algebra (https://www.cs.umd.edu/users/elman/763.21/syl.html) final project, I am diving into the Lanczos algorithm (https://en.wikipedia.org/wiki/Lanczos_algorithm) further. Specifically, I am implementing a serial and parallel version of the Lanczos method which will be used to approximate the SVD of a data matrix <img src="https://render.githubusercontent.com/render/math?math=X">. The goal, is to use this approximated SVD to perform PCA on the data matrix for data visualization. Hopefully, using the Lanczos algorithm to construct the first two to three columns of <img src="https://render.githubusercontent.com/render/math?math=U,V">, from the SVD: <img src="https://render.githubusercontent.com/render/math?math=X = UDV^T">, will be a more efficient yet still accurate approah to computing the PCA of <img src="https://render.githubusercontent.com/render/math?math=X">.

## Dependencies
For this project I utilized CUDA 11.2.2 for the parallelization of the LFDV algorithm. I ran on one NVIDIA GeForce RTX 2080 Ti GPU. All my code is written in Python, using Python 3.9.6. I used the following packages: 
1) cupy-cuda112 10.0.0
2) tensorflow 2.7.0
3) keras 2.7.0
4) numpy 1.21.4
5) Sklearn
6) plotly 5.4.0
7) matplotlib 3.5.4

## Running LFDV
I have provided a batch script "run.sh" to run LFDV in serial and parallel, and compare their times and accuracies. This requires a GPU for the parallel code. Every script was run on SLURM in the UMD HPC cluster.
