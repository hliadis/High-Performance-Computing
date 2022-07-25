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

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
                       int imageW, int imageH, int pad_imageW, int filterR) {

    int k;

      int x =  blockIdx.x*blockDim.x + threadIdx.x;   
      int y =  blockIdx.y*blockDim.y +threadIdx.y;

    double sum = 0;
    for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        sum += d_Src[y * pad_imageW + d] * d_Filter[filterR - k];
    }

    d_Dst[(y+filterR)*pad_imageW + x + filterR] = sum; 
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter,
    			   int imageW, int imageH, int pad_imageH, int filterR) {

    int k;

    int x =  blockIdx.x*blockDim.x + threadIdx.x;   
    int y =  blockIdx.y*blockDim.y +threadIdx.y;
    
    double sum = 0;

    for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        sum += d_Src[d * pad_imageH + x + filterR] * d_Filter[filterR - k];
    }  

    d_Dst[y * imageW + x] = sum;
}

//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

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
    			   int imageW, int imageH, int filterR) {

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


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_temp,
    *h_Buffer,
    *h_OutputCPU,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU;

    float elapsed= 0.0f;
    cudaEvent_t start, stop;

    int imageW;
    int imageH;
    int pad_imageW;
    int pad_imageH;

    unsigned int i, padding_size;
    int posx, posy;

	printf("Enter filter radius : ");
	if( scanf("%d", &filter_radius) == EOF){
    fprintf(stderr,"Invalid Input!\n");
      exit(EXIT_FAILURE);
  }

    padding_size = 2*filter_radius;

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    if( scanf("%d", &imageW) == EOF){
      fprintf(stderr,"Invalid Input!\n");
      exit(EXIT_FAILURE);
    }
    imageH = imageW;

    //Determine padded array size
    pad_imageW = padding_size + imageW;
    pad_imageH = padding_size + imageH;


    //Each block can have up to 32*32 threads. Thus, each block can process up to 32*32 pixels
    dim3 threads(MAX_X_DIM,MAX_Y_DIM);

    //Each blocks processes 32*32 pixels, and there are (imageW / 32) * (imageH / 32) blocks
    dim3 blocks(imageH/MAX_X_DIM,imageW/MAX_Y_DIM);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_temp = (double *)malloc(pad_imageW * pad_imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

    //Check for malloc success
    if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL){
        fprintf(stderr, "Failed to allocate host variables!\n");
        exit(EXIT_FAILURE);
    }

    //allocate device (GPU) memory
    checkCudaErrors( cudaMallocManaged((void**)&d_Filter,FILTER_LENGTH * sizeof(double)) );
    checkCudaErrors( cudaMallocManaged((void**)&d_Input,pad_imageH * pad_imageW * sizeof(double)) );
    checkCudaErrors( cudaMallocManaged((void**)&d_Buffer,pad_imageH * pad_imageW * sizeof(double)) );
    cudaMemset(d_Buffer, 0, pad_imageH*pad_imageW*sizeof(double));
    checkCudaErrors( cudaMallocManaged((void**)&d_OutputGPU,imageH * imageW * sizeof(double)) );

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);
    // initialization of host arrays
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
                h_Input[i] = h_temp[posy*pad_imageW + posx];
                i++;
            }
        }
    }

    //GPU Computation
    printf("GPU computation...\n");
    
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    checkCudaErrors( cudaEventRecord(start, 0) );
    
    checkCudaErrors( cudaMemcpy(d_Filter, h_Filter,FILTER_LENGTH * sizeof(double),cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_Input, h_temp,pad_imageH * pad_imageW * sizeof(double),cudaMemcpyHostToDevice) );

    convolutionRowGPU<<<blocks,threads>>>(d_Buffer,&d_Input[filter_radius * pad_imageH + filter_radius],d_Filter,imageW,imageH,pad_imageW,filter_radius);
    
    cudaDeviceSynchronize();
    
    convolutionColumnGPU<<<blocks,threads>>>(d_OutputGPU,&d_Buffer[filter_radius*pad_imageH],d_Filter,imageW,imageH,pad_imageH,filter_radius);

   checkCudaErrors( cudaMemcpy(h_OutputGPU,d_OutputGPU,imageH * imageW * sizeof(double),cudaMemcpyDeviceToHost) );

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
}
