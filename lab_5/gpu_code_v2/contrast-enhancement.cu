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
{   //int hist[256];
    PGM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;
    
    //histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    
    checkCudaErrors (cudaHostAlloc (&result.img, result.w * result.h * sizeof(unsigned char),0));
    
    histogram_equalization(result.img,img_in.img,result.w*result.h, 256);
    return result;
}

