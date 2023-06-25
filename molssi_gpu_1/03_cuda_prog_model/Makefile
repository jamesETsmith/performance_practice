NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda/12.1/bin/nvcc

cpu_printer: cpu_printer.cu
	${NVCC} $? -o $@

gpu_printer_sb: gpu_printer_sb.cu
	${NVCC} $? -o $@

gpu_printer_mb_local: gpu_printer_mb_local.cu
	${NVCC} $? -o $@

gpu_printer_mb_global: gpu_printer_mb_global.cu
	${NVCC} $? -o $@

gpu_printer_monolithic: gpu_printer_monolithic.cu
	${NVCC} $? -o $@

gpu_printer_grid_stride_loop: gpu_printer_grid_stride_loop.cu
	${NVCC} $? -o $@

gpu_printer_hierarchy: gpu_printer_hierarchy.cu
	${NVCC} $? -o $@

gpu_sum_vector: gpu_sum_vector.cu
	${NVCC} $? -o $@

clean:
	rm -f cpu_printer \
		gpu_printer_sb \
		gpu_printer_mb_global \
		gpu_printer_mb_local \
		gpu_printer_monolithic \
		gpu_printer_grid_stride_loop \
		gpu_printer_hierarchy \
		gpu_sum_vector
