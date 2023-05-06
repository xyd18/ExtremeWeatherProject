# ExtremeWeatherProject
Hi! This is our course project for CMU Parallel Computer Architecture and Programming. In this project, we parallelize the Transformer model using Tensor Model Parallelism and Pipeline Parallelism. We implemented the Encoder from scratch in C++, Tensor Model Parallelism inside a layer in OpenMP and MPI, and Pipeline Parallelism in MPI. Results can be found in the report shown in our project website:
[Project Website](https://xyd18.github.io/ExtremeWeatherProject/)


## How to run
### 0. Prerequisites
- C++ compiler
- OpenMP
- OpenMPI

### 1. Clone the repository and make
```
git clone https://github.com/xyd18/ExtremeWeatherProject.git
```

Next, under the root directly, run
```
make debug # for debug version, which has extensive printings
make release # for release version
```

### 2. Run the program
For the Tensor Model Parallelism version in OpenMP
```
./bin/debug-transformer-openmp # debug version
./bin/release-transformer-openmp # release version
```


For the Tensor Model Parallelism version in MPI, using 4 processes
```
mpirun -np 4 ./bin/debug-transformer-tmp-cube -o logs/debug-transformer-tmp-cube-4.txt > logs/debug-transformer-tmp-cube-4.log # debug version
mpirun -np 4 ./bin/release-transformer-tmp-cube -o logs/release-transformer-tmp-cube-4.txt > logs/release-transformer-tmp-cube-4.log # release version
```


For the Pipeline Parallelism version in MPI, using 4 processes
```
mpirun -np 4 ./bin/debug-ViT -pip -out Vit_output.bin > logs/debug-ViT.log # debug version
mpirun -np 4 ./bin/release-ViT -pip -out Vit_output.bin > logs/release-ViT.log # release version
```


For Pipeline Parallelism version in MPI with Tensor Model Parallelism inside a layer
```
mpirun -np 4 ./bin/debug-ViT -pip -tmp -out Vit_output.bin -micro 32 > logs/debug-ViT.log # debug version
mpirun -np 4 ./bin/release-ViT -pip -tmp -out Vit_output.bin -micro 32 > logs/release-ViT.log # release version
```


Alternatively, you can consider using the checker.py we provided.


## Future Work
- Implement the Auto Diff using C++ library
- Implement the CUDA/OpenCL version, enalbing training on GPU

