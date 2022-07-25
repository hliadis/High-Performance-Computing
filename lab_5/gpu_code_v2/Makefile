all: main contrast-enhancement.o histogram-equalization.o 

o: %.c
	gcc -o $@ $< -lm

main: contrast-enhancement.o histogram-equalization.o
	nvcc -O4 contrast-enhancement.o histogram-equalization.o main.cu -o main 

contrast-enhancement.o : hist-equ.h histogram-equalization.o
	nvcc -O4 -c contrast-enhancement.cu

histogram-equalization.o : hist-equ.h
	nvcc -O4 -c histogram-equalization.cu

clean:
	rm *.o main 
