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

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                       int imageW, int imageH, int filterR) {

    int k;

      int x =  blockIdx.x*blockDim.x + threadIdx.x;   
      int y =  blockIdx.y*blockDim.y +threadIdx.y;

    float sum = 0;

    for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
        }     
    } 
    d_Dst[y * imageW + x] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
    			   int imageW, int imageH, int filterR) {

    int k;

    int x =  blockIdx.x*blockDim.x + threadIdx.x;   
    int y =  blockIdx.y*blockDim.y +threadIdx.y;
                      
    float sum = 0;

    for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
        }   
    }
    d_Dst[y * imageW + x] = sum;  
}

//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU,
    acc = accuracy;

    float elapsed= 0.0f;
    cudaEvent_t start, stop;

    int imageW;
    int imageH;
    unsigned int i;

    //dim3 threads(imageH,imageW);


	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    dim3 threads(imageW,imageH);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

    //Check for malloc success
    if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL){
        fprintf(stderr, "Failed to allocate host variables!\n");
        exit(EXIT_FAILURE);
    }

    //allocate device (GPU) memory
    checkCudaErrors( cudaMallocManaged((void**)&d_Filter,FILTER_LENGTH * sizeof(float)) );
    checkCudaErrors( cudaMallocManaged((void**)&d_Input,imageH * imageW * sizeof(float)) );
    checkCudaErrors( cudaMallocManaged((void**)&d_Buffer,imageH * imageW * sizeof(float)) );
    checkCudaErrors( cudaMallocManaged((void**)&d_OutputGPU,imageH * imageW * sizeof(float)) );

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);
    // initialization of host arrays
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    //GPU Computation
    printf("GPU computation...\n");
    
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );

    checkCudaErrors( cudaEventRecord(start, 0) );

    //initialization of device arrays
    checkCudaErrors( cudaMemcpy(d_Filter, h_Filter,FILTER_LENGTH * sizeof(float),cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_Input, h_Input,imageH * imageW * sizeof(float),cudaMemcpyHostToDevice) );

    convolutionRowGPU<<<1,threads>>>(d_Buffer,d_Input,d_Filter,imageW,imageH,filter_radius);

    cudaError_t error=cudaGetLastError();

    checkCudaErrors(error);
    
    cudaDeviceSynchronize();
    
    convolutionColumnGPU<<<1,threads>>>(d_OutputGPU,d_Buffer,d_Filter,imageW,imageH,filter_radius);

    error=cudaGetLastError();
    checkCudaErrors(error);

    checkCudaErrors( cudaMemcpy(h_OutputGPU,d_OutputGPU,imageH * imageW * sizeof(float),cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaEventRecord(stop, 0) );
    checkCudaErrors( cudaEventSynchronize (stop) );
    checkCudaErrors( cudaEventElapsedTime(&elapsed, start, stop) );

    printf("GPU computation: COMPLETED!\n");

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    clock_t cpu_startTime, cpu_endTime;
    
    cpu_startTime = clock();

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    printf("CPU computation: COMPLETED!\n");

    cpu_endTime = clock();

    double cpuElapsedTime =  (double(cpu_endTime - cpu_startTime) ) / CLOCKS_PER_SEC;

    printf("GPU Elapsed Time: %.3f msec\n", elapsed);
    printf("CPU Elapsed Time: %10.8f sec\n", cpuElapsedTime);
    
    printf("Checking computed result for correctness: \n");
    bool correct = true;

    for (i = 0; i < imageW * imageH; i++) {
        if(h_OutputGPU[i] > h_OutputCPU[i] + accuracy || h_OutputGPU[i] < h_OutputCPU[i] - accuracy){
            printf("Error: h_OutputGPU[%d] = %f, with difference: %f\n", i, h_OutputGPU[i], h_OutputGPU[i] - h_OutputCPU[i]);
            correct = false;
            break;
        }
    }

    printf("accuracy: %f\n", acc);

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    //Clean up CPU Memory
    free(h_OutputCPU);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    //Clean up GPU Memory
    checkCudaErrors( cudaFree(d_OutputGPU) );
    checkCudaErrors( cudaFree(d_Buffer) );
    checkCudaErrors( cudaFree(d_Input) );
    checkCudaErrors( cudaFree(d_Filter) );
    checkCudaErrors( cudaEventDestroy(start) );
    checkCudaErrors( cudaEventDestroy(stop) );

    cudaDeviceReset();

    if(correct) 
        return 0;

    else
        return 1;


    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  



    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    // cudaDeviceReset();


    return 0;
}
