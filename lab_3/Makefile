
NVCC = nvcc
CCFLAGS = -Xcompiler -O4 

ARCHFLAGS = -gencode arch=compute_35,code=sm_35 \
-gencode arch=compute_37,code=sm_37 \
-gencode arch=compute_50,code=sm_50 \
-gencode arch=compute_52,code=sm_52 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_86,code=compute_86

SRC = task2.cu task4.cu task5.cu task8.cu
EXECUTABLES = task2 task4 task8 task5 

all: $(EXECUTABLES)

%: %.cu
	$(NVCC) $(CCFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f $(EXECUTABLES)
