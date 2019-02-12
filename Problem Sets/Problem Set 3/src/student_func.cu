/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

// should make a copy of d_logLuminance
//__global__ void getMaxMin(const float *const d_logLuminance, float *LogOut,
//                          const size_t numRows, const size_t numCols,
//                          bool isMax) {
//  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
//  int tx = threadIdx.x;
//  for (unsigned int s = blockDim.x / 2; s > 0; s >> 1) {
//    if (tx < s) {
//      if (isMax)
//        LogOut[Idx] = max(d_logLuminance[Idx + s], d_logLuminance[Idx]);
//      else
//        LogOut[Idx] = min(d_logLuminance[Idx + s], d_logLuminance[Idx]);
//    }
//    __syncthreads();
//  }
//  if (tx == 0) {
//    LogOut[blockIdx.x] = LogOut[Idx];
//  }
//}

__global__ void getMaxMin_share(const float *const d_logLuminance,
                                float *LogOut, const size_t numRows,
                                const size_t numCols, bool isMax) {
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  extern __shared__ float sdata[];
  sdata[tx] = d_logLuminance[Idx];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tx < s) {
      if (isMax)
        sdata[tx] = max(sdata[tx + s], sdata[tx]);
      else
        sdata[tx] = min(sdata[tx + s], sdata[tx]);
    }
    __syncthreads();
  }
  if (tx == 0) {
    LogOut[blockIdx.x] = sdata[0];
  }
}

__global__ void gen_histo(const float *const d_logLuminance, const float *d_min,
                          const float *d_max, int *d_histo, const int BINCNT) {
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  int range = *d_max - *d_min;
  int whichBin = (d_logLuminance[Idx] - *d_min) ;
  whichBin /= range ;
  whichBin *=BINCNT;
  if (whichBin == BINCNT)
    whichBin--;
  atomicAdd(&d_histo[whichBin], 1);
}

__global__ void Hscan(float *g_odata, float *g_idata, int n) {
  extern __shared__ int temp[]; // allocated on invocation
  int thid = threadIdx.x;
  int pout = 0, pin = 1;
  // Load input into shared memory.
  // This is exclusive scan, so shift right by one
  // and set first element to 0
  temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
  __syncthreads();
  for (int offset = 1; offset < n; offset *= 2) {
    pout = 1 - pout; // swap double buffer indices
    pin = 1 - pout;
    if (thid >= offset)
      temp[pout * n + thid] += temp[pin * n + thid - offset];
    else
      temp[pout * n + thid] = temp[pin * n + thid];
    __syncthreads();
  }
  g_odata[thid] = temp[pout * n + thid]; // write output
}

__global__ void Bscan(unsigned int *g_odata, int *g_idata, int n) {
  extern __shared__ int temp[]; // allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;
  temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
  temp[2 * thid + 1] = g_idata[2 * thid + 1];
  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0) {
    temp[n - 1] = 0;
  }                              // clear the last element
  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
  g_odata[2 * thid + 1] = temp[2 * thid + 1];
}
void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf, float &min_logLum,
                                  float &max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
  // TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  bool useShare = false;
  int blockSize(256);
  int gridSzize((blockSize + numCols * numRows - 1) / blockSize);

  float *d_intermediate; // should not modify d_idim3n
  checkCudaErrors(cudaMalloc(&d_intermediate,
                             sizeof(float) * gridSzize)); // store max and min
  float *d_min, *d_max;
  checkCudaErrors(cudaMalloc((void **)&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_max, sizeof(float)));
  int *d_histo;
  checkCudaErrors(cudaMalloc((void **)&d_histo, sizeof(int) * numBins));

//  if (useShare) {
    getMaxMin_share<<<gridSzize, blockSize, blockSize * sizeof(float)>>>(
        d_logLuminance, d_intermediate, numRows, numCols, true);

    getMaxMin_share<<<1, gridSzize, gridSzize * sizeof(float)>>>(
        d_max, d_intermediate, numRows, numCols, true);

    getMaxMin_share<<<gridSzize, blockSize, blockSize * sizeof(float)>>>(
        d_logLuminance, d_intermediate, numRows, numCols, false);

    getMaxMin_share<<<1, gridSzize, gridSzize * sizeof(float)>>>(
        d_min, d_intermediate, numRows, numCols, false);
//  } else {
//    getMaxMin<<<gridSzize, blockSize>>>(d_logLuminance, d_intermediate, numRows,
//                                        numCols, true);
//    getMaxMin<<<1, gridSzize>>>(d_max, d_intermediate, numRows, numCols, true);
//    getMaxMin<<<gridSzize, blockSize>>>(d_logLuminance, d_intermediate, numRows,
//                                        numCols, false);
//    getMaxMin<<<1, gridSzize>>>(d_min, d_intermediate, numRows, numCols, false);
//  }

  checkCudaErrors(cudaGetLastError());

  gen_histo<<<gridSzize, blockSize>>>(d_logLuminance, d_min, d_max, d_histo,
                                      numBins);

  checkCudaErrors(cudaGetLastError());

  int blocks=1;
  int threads=numBins;
  Bscan<<<blocks,threads,2*numBins*sizeof(int)>>>(d_cdf,d_histo,numBins);
//  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum,d_min,sizeof(float),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum,d_max,sizeof(float),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));
  checkCudaErrors(cudaFree(d_histo));
}
