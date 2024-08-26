#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <map>
#include <algorithm>
#include "../../include/feature_extraction.h"

// Constants
//const int SIGNAL_LENGTH = 1024; // Can be adjusted based on input signal

// Feature extraction structure to store all the features for machine learning.
struct Features {
    float magnitude[SIGNAL_LENGTH]; // FFT Magnitudes
    float phase[SIGNAL_LENGTH];     // FFT Phases
    float spectralCentroid;         // Spectral Centroid
    float spectralFlatness;         // Spectral Flatness
    float spectralBandwidth;        // Spectral Bandwidth
    float zcr;                      // Zero Crossing Rate
    float energy;                   // Signal Energy
    float temporalMean;            // Temporal Mean
    float temporalKurtosis;        // Temporal Variance
    float temporalSkewness;        // Temporal Skewness
    float temporalVariance;        // Temporal Variance
};

// Convert string to lowercase
std::string toLowerCase(const std::string& str) {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

__global__ void scaleSignal(float* d_signal, float scale) {
    /*
    Documentation:
        Scales the input signal on the GPU by a specified factor using CUDA parallelization.

    Inputs:
        float* d_signal:
            - A pointer to the signal data stored on the device (GPU).
            - Each element in the signal is a floating-point number representing the signal's amplitude at a specific time point.

        float scale:
            - The scaling factor by which each element in the signal will be multiplied.
            - This value is applied uniformly across the entire signal.

    Outputs:
        void, no return:
            - The function modifies the input signal in place on the GPU, so there is no return value.
            - Each element in the `d_signal` array will be scaled by the specified factor.
    */

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < SIGNAL_LENGTH) {
        d_signal[idx] *= scale;
    }
}

// Calculate zero crossing rate with CUDA kernel.
__global__ void calculateZCR(float* d_signal, int length, float* d_zcr) {
    /*
    Documentation:
        Calculates the zero crossing rate for the signal.
    Inputs:
        float* d_signal:
            - A pointer to the signal data stored on the device (GPU).
            - Each element in the signal is a floating-point number representing the signal's amplitude at a specific time point.
        int length:
            - An integer representing the length of the input signal data array.
        float* d_zcr:
            - A pointer to the computed ZCR data stored on the device (GPU).
    Outputs:
        void, no return:
            - The function uses the input signal to calculate the ZCR and store the data in d_zcr.
    */
   int zero_crossings = 0;
    for (int i = 1; i < length; ++i) {
        if ((d_signal[i - 1] > 0 && d_signal[i] < 0) || (d_signal[i - 1] < 0 && d_signal[i] > 0)) {
            zero_crossings++;
        }
    }
    *d_zcr = static_cast<float>(zero_crossings) / length;
}


void fft_feature_extraction(float* h_signal, int length, Features* h_features) {
    /*
    Documentation:
        Extracts features from the input signal by performing a Fast Fourier Transform (FFT) and calculating the magnitude of the FFT result. This process is accelerated using Nvidia Performance Primitives (NPP) and CUDA.

    Inputs:
        float* h_signal:
            - A pointer to the signal data stored on the host (CPU).
            - The input signal is a time-domain signal, represented as an array of floating-point numbers.

        int length:
            - The length of the input signal array.
            - This value determines the number of elements to process in the FFT and feature extraction.

        float* h_features:
            - A pointer to the array where the extracted features will be stored on the host (CPU).
            - The array should be pre-allocated to have enough space to store the magnitude values resulting from the FFT.

    Outputs:
        void, no return:
            - The function performs FFT on the input signal and stores the extracted features (magnitude of the FFT) in the `h_features` array on the host.
            - No value is returned, but the `h_features` array is modified to contain the extracted features.
    */
   
    // Allocate device memory first.
    float *d_signal;
    cufftComplex *d_fft_result;
    cudaError_t err = cudaMalloc((void**)&d_signal, length * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    cudaError_t err = cudaMalloc((void**)&d_fft_result, length * sizeof(cufftComplex)); 
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_signal);
        return;
    }
    // This is a complex float for FFT

    // Create a 1D FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, length, CUFFT_R2C, 1);

    // Execute FFT plan
    cufftExecR2C(plan, d_signal, d_fft_result);

    // Copy FFT result back to host
    cufftComplex h_fft_result[length];
    cudaMemcpy(h_fft_result, d_fft_result, length * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Calculate magnitude and phase from FFT results
    for (int i = 0; i < length; ++i) {
        h_features->magnitude[i] = sqrtf(h_fft_result[i].x * h_fft_result[i].x + h_fft_result[i].y * h_fft_result[i].y);
        h_features->phase[i] = atan2f(h_fft_result[i].y, h_fft_result[i].x);
    }

    // Calculate spectral centroid and bandwidth
    float spectral_centroid = 0.0f;
    float sum_magnitudes = 0.0f;
    for (int i = 0; i < length; ++i) {
        spectral_centroid += h_features->magnitude[i] * i;
        sum_magnitudes += h_features->magnitude[i];
    }
    h_features->spectralCentroid = spectral_centroid / sum_magnitudes;

    float spectral_bandwidth = 0.0f;
    for (int i = 0; i < length; ++i) {
        spectral_bandwidth += h_features->magnitude[i] * powf((i - h_features->spectralCentroid), 2);
    }
    h_features->spectralBandwidth = sqrtf(spectral_bandwidth / sum_magnitudes);

    // Calculate spectral flatness
    float geom_mean = 1.0f;
    for (int i = 0; i < length; ++i) {
        geom_mean *= h_features->magnitude[i];
    }
    geom_mean = powf(geom_mean, 1.0f / length);
    
    float arithm_mean = sum_magnitudes / length;
    h_features->spectralFlatness = geom_mean / arithm_mean;

    // Using CUDA kernel, calculate ZCR
    float* d_zcr;
    cudaError_t err = cudaMalloc((void**)&d_zcr, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        cufftDestroy(plan);
        cudaFree(d_signal);
        cudaFree(d_fft_result);
        return;
    }
    calculateZCR<<<1, 1>>>(d_signal, length, d_zcr);
    cudaMemcpy(&h_features->zcr, d_zcr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_zcr);

    // Calculate energy
    float energy = 0.0f;
    for (int i = 0; i < length; ++i) {
        energy += h_signal[i] * h_signal[i];
    }
    h_features->energy = sqrtf(energy / length);

    // Calculate temporal features
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        sum += h_signal[i];
    }
    h_features->temporalMean = sum / length;

    float variance_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        variance_sum += powf(h_signal[i] - h_features->temporalMean, 2.0f);
    }
    h_features->temporalVariance = variance_sum / length;

    float kurtosis_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        kurtosis_sum += powf(h_signal[i] - h_features->temporalMean, 4.0f);
    }
    h_features->temporalKurtosis = kurtosis_sum / (length * powf(h_features->temporalVariance, 2.0f)) - 3.0f;

    float skewness_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        skewness_sum += powf(h_signal[i] - h_features->temporalMean, 3.0f);
    }
    h_features->temporalSkewness = skewness_sum / (length * powf(h_features->temporalVariance, 1.5f));

    // Cleanup!
    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_fft_result);
}