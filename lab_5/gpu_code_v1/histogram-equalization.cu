#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "hist-equ.h"
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
        ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#define checkCudaErrors(ans) {                                \
          cudaError_t err = ans;                                   \
          if (err != cudaSuccess) {                                 \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(err)); \
            cudaDeviceReset();                                      \
            exit(EXIT_FAILURE);                                     \
          }                                                          \
        }                                                         \


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

//-------------------------DEVICE_CODE---------------------------------//

__global__ void cdf_lut(int d_in, int min, int *d_hist, int nbr_bin, int img_size){
    extern __shared__ int cdf[];
    int thread_id = threadIdx.x;
    int offset = 1;
    
    int ai = thread_id;
    int bi = thread_id + (nbr_bin / 2);
    int ai_hist = d_hist[ai];
    int bi_hist = d_hist[bi];
    int temp;

    cdf[ai + CONFLICT_FREE_OFFSET(ai)] = ai_hist;
    cdf[bi + CONFLICT_FREE_OFFSET(bi)] = bi_hist;


    for(int d = nbr_bin >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if(thread_id < d)
        {
            
            ai = offset * ((2*thread_id) + 1) - 1;
            bi = offset * ((2*thread_id) + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            cdf[bi] += cdf[ai];
        }

        offset *= 2;
    }

    if(thread_id == 0) {
        cdf[nbr_bin - 1 + CONFLICT_FREE_OFFSET(nbr_bin - 1)] = 0;
    }
    
    for(int d = 1; d < nbr_bin; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if(thread_id < d)
        {
            ai = offset * ((2*thread_id) + 1) - 1;
            bi = offset * ((2*thread_id) + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp = cdf[ai];
            cdf[ai] = cdf[bi];
            cdf[bi] += temp; 
        }
    }

    __syncthreads();
    ai = thread_id;
    bi = thread_id + nbr_bin / 2;

    cdf[ai + CONFLICT_FREE_OFFSET(ai)] += ai_hist;
    cdf[bi + CONFLICT_FREE_OFFSET(bi)] += bi_hist;


   
    d_hist[ai] = (int)(((float)cdf[ai + CONFLICT_FREE_OFFSET(ai)]-min)*255/d_in+0.5);
    if(d_hist[ai] < 0) { d_hist[ai] = 0; }
    d_hist[bi] = (int)(((float)cdf[bi+CONFLICT_FREE_OFFSET(bi)]-min)*255/d_in+0.5);
    if(d_hist[bi] < 0) { d_hist[bi] = 0; }
}

__global__ void resultImage(int *lut_GPU, unsigned char *img_in, unsigned char * img_out,size_t img_size)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	int val = img_in[tx];
	extern __shared__ int lut[];

	if( threadIdx.x < 256) {
		lut[threadIdx.x] = lut_GPU[threadIdx.x];
	}
	__syncthreads();
	if( (size_t)tx  < img_size) {
		if(lut[val] > 255 ) 
			img_out[tx] = 255;
		else
			img_out[tx] = (unsigned char)lut[val];
	}
}


double histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *d_hist;
    unsigned char *d_img_out, *d_buffer;
    int i, min, d;
    dim3 grid, block;
    float elapsed= 0.0f;
    cudaEvent_t start, stop;
    
    checkCudaErrors (cudaMalloc((void **)&d_hist, nbr_bin * sizeof(int)));
    checkCudaErrors (cudaMemset((void *)d_hist, 0,  nbr_bin * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_buffer, img_size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void **)&d_img_out, img_size * sizeof(unsigned char)));
    checkCudaErrors (cudaMemset((void *)d_img_out, 0,  img_size * sizeof(unsigned char)));

    /* Construct the LUT by calculating the CDF */
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;

    grid = 1;
    block.x = 128;
    block.y = 1;

    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    checkCudaErrors( cudaEventRecord(start, 0) );
    
    checkCudaErrors( cudaMemcpyAsync(d_hist,hist_in, nbr_bin * sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyAsync(d_buffer,img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice) );

    cdf_lut<<<grid, block, (nbr_bin + (nbr_bin - 1) / NUM_BANKS) * sizeof(int)>>>(d, min, d_hist,nbr_bin,img_size);
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); 
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    block.x = 1024;
    block.y = 1;
    grid.x = img_size / block.x + 1;

    resultImage<<<grid, block, (256*sizeof(int))>>>(d_hist, d_buffer, d_img_out,img_size);
    
    error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); 
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    error = cudaGetLastError();
    
    if(error != cudaSuccess){
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); 
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    
    checkCudaErrors(cudaMemcpyAsync(img_out,d_img_out,img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost)); 

    checkCudaErrors( cudaEventRecord(stop, 0) );
    checkCudaErrors( cudaEventSynchronize (stop) );
    checkCudaErrors( cudaEventElapsedTime(&elapsed, start, stop) );

    printf("GPU Computation Time: %f ms\n", elapsed);
    
    cudaFree(d_img_out);
    cudaFree(d_buffer);
    cudaFree(d_hist);
    checkCudaErrors( cudaEventDestroy(start) );
    checkCudaErrors( cudaEventDestroy(stop) );

    return(elapsed);

}
