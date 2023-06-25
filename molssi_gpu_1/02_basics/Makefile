NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda/12.1/bin/nvcc

hello: hello.cu
	${NVCC} $? -o $@

clean:
	rm hello