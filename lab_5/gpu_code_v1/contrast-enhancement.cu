#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "hist-equ.h"
#define checkCudaErrors(ans) {                                \
          cudaError_t err = ans;                                   \
          if (err != cudaSuccess) {                                 \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(err)); \
            cudaDeviceReset();                                      \
            exit(EXIT_FAILURE);                                     \
          }                                                          \
        }                                                         \


PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    
    double exec_time = 0.0;
    
    checkCudaErrors (cudaHostAlloc (&result.img, result.w * result.h * sizeof(unsigned char),0));
    
    clock_t start = clock();
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    clock_t end = clock();

    double cpuElapsedTime =  (double(end- start) ) / (CLOCKS_PER_SEC/1000);
    exec_time = cpuElapsedTime;
    printf("Histogram Computation Time:%f ms\n",exec_time);
    
    exec_time += histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    printf("Total Computation Time: %f ms\n", exec_time);
    
    return result;
}
