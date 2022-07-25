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
__global__ void histogram_GPU(int *hist_GPU, unsigned char *img,int size)
{   //Padding 1 to nbr_bins (256)
     __shared__ int sh_hist[(256+1)*8];
    //Indexes
    const int warpid = (int)(threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    const int warpsperblock = blockDim.x / 32;
    
    //offset to per-block sub-histogram
    const int offset = (256+1) * (threadIdx.x % 8);

    //Interleaved read access 
    const int begin = (size/warpsperblock)*warpid + 32 * blockIdx.x + lane;
    const int end = (size/warpsperblock) * (warpid + 1);
    const int step = 32 * gridDim.x;

    //Initialization
    for(int pos = threadIdx.x; pos < ((256+1)*8); pos+=blockDim.x){
        sh_hist[pos] = 0;
    }
    __syncthreads();

    for(int i = begin; i < end; i+=step){
        int d = img[i];
        atomicAdd(&sh_hist[offset + d], 1);
    }

    __syncthreads();

    //Merge sub_histograms and write in global memory
    for(int pos = threadIdx.x; pos < 256; pos+=blockDim.x){
        int sum = 0;
        for(int base = 0; base < ((256+1) * 8); base+= 256+1){
                sum+= sh_hist[base+pos];
        }
        atomicAdd(hist_GPU + pos, sum);
    }

}

__global__ void cdf_lut(int *d_hist, int nbr_bin, int img_size){
    extern __shared__ int cdf[];
    int thread_id = threadIdx.x;
    int offset = 1;
    int i=0;
    int d_in, min = 0;
    int ai = thread_id;
    int bi = thread_id + (nbr_bin / 2);
    int ai_hist = d_hist[ai];
    int bi_hist = d_hist[bi];
    int temp;

    cdf[ai + CONFLICT_FREE_OFFSET(ai)] = ai_hist;
    cdf[bi + CONFLICT_FREE_OFFSET(bi)] = bi_hist;

    __syncthreads();

    while(min == 0){
        min = d_hist[i];
        i = i+1;
    }

    d_in = img_size - min;
    
    for(int d = nbr_bin >> 1; d > 0; d >>= 1)
    {
        
        if(thread_id < d)
        {
            
            ai = offset * ((2*thread_id) + 1) - 1;
            bi = offset * ((2*thread_id) + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            cdf[bi] += cdf[ai];
        }

        offset *= 2;
        __syncthreads();
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


void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int img_size, int nbr_bin){
    int *d_hist;
    unsigned char *d_img_out, *d_buffer;
    dim3 grid, block;
    float elapsed= 0.0f;
    cudaEvent_t start, stop;
    
    int *h_hist = (int *) malloc(nbr_bin* sizeof(int));
    checkCudaErrors (cudaMalloc((void **)&d_hist, nbr_bin * sizeof(int)));
    checkCudaErrors (cudaMemset((void *)d_hist, 0,  nbr_bin * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_buffer, img_size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void **)&d_img_out, img_size * sizeof(unsigned char)));
    checkCudaErrors (cudaMemset((void *)d_img_out, 0,  img_size * sizeof(unsigned char)));

    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    checkCudaErrors( cudaEventRecord(start, 0) );
    
    checkCudaErrors( cudaMemcpyAsync(d_buffer,img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice) );
   
    histogram_GPU<<<img_size/1024 , 1024>>>(d_hist, d_buffer,img_size);
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); 
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }


    grid = 1;
    block.x = 128;
    block.y = 1;

    cdf_lut<<<grid, block, (nbr_bin + (nbr_bin - 1) / NUM_BANKS) * sizeof(int)>>>(d_hist,nbr_bin,img_size);
    
    error = cudaGetLastError();
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

}
