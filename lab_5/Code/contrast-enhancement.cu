#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#define BILLION 1000000000L
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    

    /*struct timespec start, stop;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, & start) == -1) {
      perror("clock gettime");
      exit(EXIT_FAILURE);
    }*/
    clock_t start = clock();


    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    clock_t end = clock();
    double cpuElapsedTime =  (double(end- start) ) / CLOCKS_PER_SEC;
    double exec_time = cpuElapsedTime;
    /*if (clock_gettime(CLOCK_MONOTONIC_RAW, &stop) == -1) {
      perror("clock gettime");
      exit(EXIT_FAILURE);
    }
    double accum = (double)(stop.tv_sec - start.tv_sec) \
            + (double)(stop.tv_nsec - start.tv_nsec) / BILLION;*/    

    printf("CPU computation time: %10.8f\n", exec_time);

    return result;
}
