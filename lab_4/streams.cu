/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;
#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005
#define MAX_X_DIM 32
#define MAX_Y_DIM 32
__constant__ double d_Filter[257];
#define CPU

#define checkCudaErrors(ans) {                                \
          cudaError_t err = ans;                                   \
          if (err != cudaSuccess) {                                 \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(err)); \
            cudaDeviceReset();                                      \
            exit(EXIT_FAILURE);                                     \
          }                                                          \
        }                                                         \

//------------------------------DEVICE_CODE------------------------------

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, int pad_imageW, int filterR) {

    int k;

      int x =  blockIdx.x*blockDim.x + threadIdx.x +filterR;   
      int y =  blockIdx.y*blockDim.y +threadIdx.y + filterR;
      int idx = y *(pad_imageW) + x;
      int x_Dim = 2*filterR + blockDim.x;

      extern __shared__ double s_Input[];

      for(int k = 0; k + threadIdx.x < x_Dim ; k+= blockDim.x){
	      s_Input[threadIdx.y*x_Dim + threadIdx.x + k] = d_Src[idx - filterR + k];
      }

      __syncthreads();

    double sum = 0;
    
    for (k = -filterR; k <= filterR; k++) {
        sum += s_Input[threadIdx.y*x_Dim + threadIdx.x + filterR + k] * d_Filter[filterR - k];
    }

    d_Dst[idx] = sum; 
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, int pad_imageH ,int filterR) {
    
    int k;

    int x =  blockIdx.x*blockDim.x + threadIdx.x + filterR;   
    int y =  blockIdx.y*blockDim.y +threadIdx.y + filterR;
    int idx = y*pad_imageH + x;
    int y_Dim = 2*filterR + blockDim.y;
    int dst_idx = (y-filterR)*(pad_imageH - 2*filterR) + (x-filterR);

    extern __shared__ double s_Input[];

    for(int k = 0; k + threadIdx.y < y_Dim; k+= blockDim.y ){
        s_Input[threadIdx.y * blockDim.x + k*blockDim.x + threadIdx.x] = d_Src[idx + (k - filterR)*pad_imageH];
    }

    __syncthreads();
    
    double sum = 0;

    for (k = -filterR; k <= filterR; k++) {
        sum += s_Input[(threadIdx.y * blockDim.x) + (filterR + k)*blockDim.x + threadIdx.x] * d_Filter[filterR - k];
    }  

    d_Dst[dst_idx] = sum;
}

//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH,  int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH,  int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

/*void case_free(void *d_Input, void *d_OutputGPU, void *h_Filter, void *h_Buffer, void *h_OutputCPU, void * h_OutputGPU, void *h_temp, void *h_temp2, cudaError_t error){

  if(error != cudaSuccess){
      //Clean up CPU Memory
    free(h_OutputCPU);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_temp);
    free(h_temp2);
    free(h_Input);
    free(h_Filter);

    //Clean up GPU Memory
    cudaFree(d_OutputGPU);
    cudaFree(d_Input);

  }


}*/

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,

#ifdef CPU
    *h_Buffer,
    *h_OutputCPU,
    *h_Input,
#endif

    *h_temp,
    *h_temp2,
    *d_Input0,
    *d_Input1,
    *d_OutputGPU0,
    *d_OutputGPU1,
    *h_OutputGPU;

    float elapsed= 0.0f;
    cudaEvent_t start, stop;

    int imageW;
    int imageH;
    int pad_imageW;
    int pad_imageH;
    int block_size;
    int chunk;
    int tile_idx;
    int pos;

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    dim3 grid, block;

    unsigned int i, padding_size;
    int posx, posy;

	printf("Enter filter radius: ");
	if (scanf("%d", &filter_radius) == EOF){
    fprintf(stderr,"Invalid Input!\n");
    exit(EXIT_FAILURE);
  }

    padding_size = 2*filter_radius;

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    if (scanf("%d", &imageW) == EOF){
      fprintf(stderr,"Invalid Input!\n");
      exit(EXIT_FAILURE);
    }

    imageH = imageW;

    printf(" Enter block size (should be a power of 2): ");
    if(scanf("%d", &block_size) == EOF){
      fprintf(stderr,"Invalid Input!\n");
      exit(EXIT_FAILURE);
    }

    chunk = imageH*imageW / block_size;

    //Determine padded array size
    pad_imageW = padding_size + imageW;
    pad_imageH = padding_size + imageH;


    if(imageW > MAX_X_DIM){
      block.x = MAX_X_DIM;
      grid.x = imageW/block.x;
      //
      if(block_size/imageW > MAX_X_DIM){
        block.y = MAX_Y_DIM;
        grid.y = (block_size/imageW)/MAX_Y_DIM;
      }

      else{
        block.y = block_size/imageW;
        grid.y = 1;
      }

      tile_idx = block_size/imageW;
    }

    else{
      grid =1;
      block.x = imageW;
      block.y = imageH;
      chunk = 1;
      tile_idx = imageH;
    }


    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    //h_temp = (double *)malloc(pad_imageW * pad_imageH * sizeof(double));
    //h_temp2 = (double *)malloc(pad_imageW * pad_imageH * sizeof(double));
    
#ifdef CPU
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
#endif
    /*checkCudaErrors(cudaHostAlloc((void **) &h_OutputGPU, imageH * imageW * sizeof(double), cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **) &h_temp, pad_imageH * pad_imageW * sizeof(double), cudaHostAllocMapped));
    h_temp2 = h_temp;

    //Check for malloc success
    if(h_Filter == NULL){
        fprintf(stderr, "Failed to allocate host variable h_Filter!\n");
        exit(EXIT_FAILURE);
    }
    else if(h_temp == NULL){
          fprintf(stderr, "Failed to allocate host variable h_temp!\n");
          exit(EXIT_FAILURE);
    }
    else if(h_OutputGPU == NULL){
      fprintf(stderr, "Failed to allocate host variable h_OutputGPU!\n");
          exit(EXIT_FAILURE);
    }*/

#ifdef CPU
    if(h_Buffer == NULL || h_OutputCPU == NULL  || h_Input == NULL){
        fprintf(stderr, "Failed to allocate host variables!\n");
        exit(EXIT_FAILURE);
    }
#endif

    //allocate device (GPU) memory
    checkCudaErrors( cudaMalloc((void**)&d_Filter,FILTER_LENGTH * sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&d_Input0,pad_imageW * (filter_radius * 2 + tile_idx) * sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&d_Input1,pad_imageW * (filter_radius * 2 + tile_idx) * sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&d_OutputGPU0,pad_imageW * (filter_radius * 2 + tile_idx) * sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&d_OutputGPU1,pad_imageW * (filter_radius * 2 + tile_idx) * sizeof(double)) );
    checkCudaErrors(cudaMemset(d_Input0, 0, pad_imageW*(filter_radius * 2 + tile_idx) *sizeof(double)));
    checkCudaErrors(cudaMemset(d_OutputGPU0, 0, pad_imageW*(filter_radius * 2 + tile_idx) *sizeof(double)));
    checkCudaErrors(cudaMemset(d_Input1, 0, pad_imageW*(filter_radius * 2 + tile_idx) *sizeof(double)));
    checkCudaErrors(cudaMemset(d_OutputGPU1, 0, pad_imageW*(filter_radius * 2 + tile_idx) *sizeof(double)));

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);
    // initialization of host arrays
    checkCudaErrors(cudaHostAlloc((void **) &h_OutputGPU, imageH * imageW * sizeof(double), cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **) &h_temp, pad_imageH * pad_imageW * sizeof(double), cudaHostAllocMapped));
    h_temp2 = h_temp;
    //memset((void*)h_temp2,0,(size_t)((pad_imageW*pad_imageH)*(size_t)sizeof(double)));

    //Check for malloc success
    if(h_Filter == NULL){
        fprintf(stderr, "Failed to allocate host variable h_Filter!\n");
        exit(EXIT_FAILURE);
    }
    else if(h_temp == NULL){
          fprintf(stderr, "Failed to allocate host variable h_temp!\n");
          exit(EXIT_FAILURE);
    }
    else if(h_OutputGPU == NULL){
      fprintf(stderr, "Failed to allocate host variable h_OutputGPU!\n");
          exit(EXIT_FAILURE);
    }
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    i = 0;

    for(posy=0; posy < pad_imageH; posy++){
        for(posx=0; posx < pad_imageW; posx++){

            if(posy < filter_radius || posy >= imageH + filter_radius){
                h_temp[posy*pad_imageW + posx] = 0;
                //since y coordinate is out of [filter_radius, imageH + filter_radius]
                //we do not have to check x coordinate
                continue;
            }

            if(posx < filter_radius || posx >= imageW + filter_radius){
                h_temp[posy*pad_imageW + posx] = 0;
            }

            else{

                h_temp[posy*pad_imageW + posx] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
            #ifdef CPU
                h_Input[i] = h_temp[posy*pad_imageW + posx];
            #endif
                i++;
            }
        }
    }


    //GPU Computation
    printf("GPU computation...\n");
    
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    checkCudaErrors( cudaEventRecord(start, 0) );
    
    checkCudaErrors( cudaMemcpyToSymbol(d_Filter, h_Filter,FILTER_LENGTH * sizeof(double),0,cudaMemcpyHostToDevice) );
    
    pos = 0;

    for(i=0; (i < (chunk/2) ) || (chunk == 1); i++){
      checkCudaErrors(cudaMemcpyAsync(d_Input0, h_temp + pos, (pad_imageW)*(tile_idx+2*filter_radius)*sizeof(double),
                      cudaMemcpyHostToDevice, stream0));
      if(chunk != 1)
        checkCudaErrors(cudaMemcpyAsync(d_Input1, h_temp + pos + (pad_imageW * tile_idx) , (pad_imageW)*(tile_idx+2*filter_radius)*sizeof(double),
                        cudaMemcpyHostToDevice, stream1));

      convolutionRowGPU<<<grid,block,(block.x + 2*filter_radius)*block.y*sizeof(double),stream0>>>(d_OutputGPU0,d_Input0,pad_imageW, filter_radius);
      
      if(chunk != 1)
        convolutionRowGPU<<<grid,block,(block.x + 2*filter_radius)*block.y*sizeof(double),stream1>>>(d_OutputGPU1,d_Input1,pad_imageW, filter_radius);

      checkCudaErrors(cudaMemcpyAsync(h_temp2 + pad_imageW*filter_radius + pos, d_OutputGPU0 + pad_imageW*filter_radius, pad_imageW * tile_idx * sizeof(double),
                      cudaMemcpyDeviceToHost,stream0));
      if(chunk == 1)
        break;

      checkCudaErrors(cudaMemcpyAsync(h_temp2 + pad_imageW*filter_radius + pos + (pad_imageW * tile_idx) , d_OutputGPU1 + pad_imageW*filter_radius, pad_imageW * tile_idx * sizeof(double),
                      cudaMemcpyDeviceToHost,stream1));

      pos += 2*(pad_imageW * tile_idx);
    }

    cudaDeviceSynchronize();

    pos = 0;
    int final_pos = 0;

    for(i=0; (i < (chunk/2) ) || (chunk == 1); i++){
      checkCudaErrors(cudaMemcpyAsync(d_Input0, h_temp2 + pos, (pad_imageW)*(tile_idx+2*filter_radius)*sizeof(double), 
                      cudaMemcpyHostToDevice, stream0));
      
      if(chunk != 1)
        checkCudaErrors(cudaMemcpyAsync(d_Input1, h_temp2 + pos + (pad_imageW * tile_idx) , (pad_imageW)*(tile_idx+2*filter_radius)*sizeof(double), 
                        cudaMemcpyHostToDevice, stream1));

      convolutionColumnGPU<<<grid,block,(block.y + 2*filter_radius)*block.x*sizeof(double),stream0>>>(d_OutputGPU0,d_Input0,pad_imageH,filter_radius);

      if(chunk != 1)
        convolutionColumnGPU<<<grid,block,(block.y + 2*filter_radius)*block.x*sizeof(double),stream1>>>(d_OutputGPU1,d_Input1,pad_imageH,filter_radius);

      
      checkCudaErrors(cudaMemcpyAsync(h_OutputGPU + final_pos, d_OutputGPU0, imageW * tile_idx *sizeof(double), 
                      cudaMemcpyDeviceToHost,stream0));
      if(chunk == 1)
        break;

      checkCudaErrors(cudaMemcpyAsync(h_OutputGPU + final_pos + (imageW * tile_idx) , d_OutputGPU1, imageW * tile_idx *sizeof(double), 
                      cudaMemcpyDeviceToHost,stream1));

      pos += 2*(pad_imageW * tile_idx);
      final_pos += 2*(imageW * tile_idx);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    checkCudaErrors( cudaEventRecord(stop, 0) );
    checkCudaErrors( cudaEventSynchronize (stop) );
    checkCudaErrors( cudaEventElapsedTime(&elapsed, start, stop) );

    printf("GPU computation: COMPLETED!\n");

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.

#ifdef CPU
    printf("CPU computation...\n");

    clock_t cpu_startTime, cpu_endTime;
    
    cpu_startTime = clock();

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    printf("CPU computation: COMPLETED!\n");

    cpu_endTime = clock();

    double cpuElapsedTime =  (double(cpu_endTime - cpu_startTime) ) / CLOCKS_PER_SEC;
#endif

    printf("GPU Elapsed Time: %.3f msec\n", elapsed);

    bool correct = true;

#ifdef CPU
    printf("CPU Elapsed Time: %10.8f sec\n", cpuElapsedTime);

    printf("Checking computed result for correctness: \n");

    for (i = 0; i < imageW * imageH; i++) {
        if(h_OutputGPU[i] > h_OutputCPU[i] + accuracy || h_OutputGPU[i] < h_OutputCPU[i] - accuracy){
            printf("Error: h_OutputGPU[%d] = %f, with difference: %f\n", i, h_OutputGPU[i], h_OutputGPU[i] - h_OutputCPU[i]);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    //Clean up CPU Memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
#endif
    //free(h_temp);
    //free(h_temp2);
    checkCudaErrors( cudaFreeHost(h_OutputGPU) );
    checkCudaErrors( cudaFreeHost(h_temp) );
    free(h_Filter);

    //Clean up GPU Memory
    checkCudaErrors( cudaFree(d_OutputGPU0) );
    checkCudaErrors( cudaFree(d_OutputGPU1) );
    //checkCudaErrors( cudaFree(d_Buffer) );
    checkCudaErrors( cudaFree(d_Input0) );
    checkCudaErrors( cudaFree(d_Input1) );
    //checkCudaErrors( cudaFree(d_Filter) );
    checkCudaErrors( cudaEventDestroy(start) );
    checkCudaErrors( cudaEventDestroy(stop) );

    cudaDeviceReset();

    if(correct) 
        return 0;

    else
        return 1;
}
