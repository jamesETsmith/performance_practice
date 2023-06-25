NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda/12.1/bin/nvcc

v0: v0.cu
	${NVCC} $? -o $@

v1: v1.cu
	${NVCC} $? -o $@

clean: 
	rm -f v0 v1