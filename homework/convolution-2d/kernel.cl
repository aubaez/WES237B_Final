
__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth,  int imageChannels){
    //@@ Insert code to implement matrix multiplication here

    /* channel index is 0 for R, 1 for G, and 2 for B */
    int maskRadius = maskWidth/2;
    int xIndex = get_global_id(0); //i in pseudo
    int yIndex = get_global_id(1);  //j in pseudo
    float accum, imagePixel, maskValue;


    if(xIndex < height && yIndex < width){
        for(int k = 0; k < imageChannels; k++){
            accum = 0;
            for(int y = -maskRadius; y <= maskRadius; y++){
                for (int x = -maskRadius; x <= maskRadius; x++){
                    int xOffset = yIndex + x;
                    int yOffset = xIndex + y;
                    if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height){
                        imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                        maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
                        accum += imagePixel * maskValue;
                    }
                } 

            } 
        // pixels are in the range of 0 to 1
        outputData[(xIndex * width + yIndex)*imageChannels + k] = clamp(accum, 0.0f, 1.0f);
        }
    }

}