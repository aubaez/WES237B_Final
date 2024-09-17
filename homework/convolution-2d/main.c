#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <arm_neon.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"



#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

float clamp(float x, float lower, float upper){
    if (x > lower){
        if ( x < upper) return x;
        else return upper;
    }
    else if ( lower > x){
        if (lower > upper) return lower;
        else return upper;
    }
    return 0;
}

void Neon2DConvolution(Matrix *input0, Matrix *input1, Matrix *result)
{
    int imageChannels = IMAGE_CHANNELS;
    int maskWidth = input1->shape[0];
    int maskRadius = maskWidth/2;
    int width = input0->shape[1];
    int height = input0->shape[0];

    float accum, imagePixel, maskValue;


    for(int xIndex = 0; xIndex < height; xIndex++){
        for(int yIndex = 0; yIndex < width; yIndex++ ){
            for(int k = 0; k < imageChannels; k++){
                accum = 0;
                for(int y = -maskRadius; y <= maskRadius; y++){
                    for (int x = -maskRadius; x <= maskRadius; x++){
                        int xOffset = yIndex + x;
                        int yOffset = xIndex + y;
                        if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height){
                            imagePixel = input0->data[(yOffset * width + xOffset) * imageChannels + k];
                            maskValue = input1->data[(y+maskRadius)*maskWidth+x+maskRadius];
                            accum += imagePixel * maskValue;
                        }
                    } 
                }
                // pixels are in the range of 0 to 1
                result->data[(xIndex * width + yIndex)*imageChannels + k] = clamp(accum, 0.0f, 1.0f);
            } 
        
        }
        
     }

}


int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadImg(input_file_a, &host_a);
    CHECK_ERR(err, "LoadImg");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadImg(input_file_c, &answer);
    CHECK_ERR(err, "LoadImg");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer image
    rows = host_a.shape[0];
    cols = host_a.shape[1];
    
    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1] * IMAGE_CHANNELS);

    //Call Neon 2D Convolution
    Neon2DConvolution(&host_a, &host_b, &host_c);

    // Save the image
    SaveImg(input_file_d, &host_c);

    // Check the result of the convolution
    CheckImg(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}