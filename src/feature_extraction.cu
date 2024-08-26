#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
// Comment out if not testing
#include <cmath>
#define PI 3.14159265358979323846

// Constants
const int SIGNAL_LENGTH = 1024; // Can be adjusted based on input signal
const int FFT_LENGTH = SIGNAL_LENGTH;

// Feature extraction structure to store all the features for machine learning.
struct Features {
    float magnitude[SIGNAL_LENGTH]; // FFT Magnitudes
    float phase[SIGNAL_LENGTH];     // FFT Phases
    float spectralCentroid;         // Spectral Centroid
    float spectralFlatness;         // Spectral Flatness
    float spectralBandwidth;        // Spectral Bandwidth
    float zcr;                      // Zero Crossing Rate
    float energy;                   // Signal Energy
    float temporal_mean;            // Temporal Mean
    float temporal_kurtosis;        // Temporal Variance
    float temporal_skewness;        // Temporal Skewness
    float temporal_variance;        // Temporal Variance
};

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


void fft_feature_extraction(float* h_signal, int length, float* h_features) {
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
   float *d_signal, *d_fft_result;
   cudaMalloc((void**)&d_signal, length * sizeof(float));
   cudaMalloc((void**)&d_fft_result, length * sizeof(Npp32fc)); 
   // This is a complex float for FFT

    // Copy signal from host to device
    cudaMemcpy(d_signal, h_signal, length * sizeof(float), cudaMemcpyHostToDevice);

    // Set up FFT configuration
    NppiFFTSpec_R_32f* pFFTSpec = nullptr;
    Npp8u* pBuffer = nullptr;
    int bufferSize = 0;
    nppiFFTGetSize_R_32f(NppiFFT_R_32f_SIZE, length, &bufferSize, nullptr);
    cudaMalloc((void**)&pBuffer, bufferSize);
    nppiFFTInit_R_32f(&pFFTSpec, length, pBuffer);

    // Perform FFT
    nppStatus = nppiFFTFwd_RToCCS_32f(d_signal, 1, d_fft_result, pFFTSpec, pBuffer);
    if (nppStatus != NPP_SUCCESS) {
        std::cerr << "FFT failed!" << std::endl;
        cudaFree(d_signal);
        cudaFree(d_fft_result);
        return;
    }

    // Calculate magnitude and phase from FFT results
    nppiMagnitude_32fc(d_fft_result, 1, h_features->magnitude, 1, {length, 1});
    nppiPhase_32fc(d_fft_result, 1, h_features->phase, 1, {length, 1});

    // Calculate spectral centroid and bandwidth
    float spectral_centroid = 0.0f;
    float sum_magnitudes = 0.0f;
    for (int i = 0; i < length; ++i) {
        spectral_centroid += h_features->magnitude[i] * i;
        sum_magnitudes += h_features->magnitude[i];
    }
    h_features->spectral_centroid /= sum_magnitudes;

    float spectral_bandwidth = 0.0f;
    for (int i = 0; i < length; ++i) {
        spectral_bandwith += h_features->magnitude[i] * powf((i - h_features->spectral_centroid), 2);
    }
    h_features->spectral_bandwidth = sqrtf(spectral_bandwidth / sum_magnitudes);

    // Calculate spectral flatness
    float geom_mean = 1.0f;
    for (int i = 0; i < length; ++i) {
        geom_mean *= h_features->magnitude[i];
    }
    geom_mean = powf(geom_mean, 1.0f / length);
    
    float arithm_mean = sum_magnitudes / length;
    h_features->spectral_flatness = geom_mean / arith_mean;

    // Using CUDA kernel, calculate ZCR
    float* d_zcr;
    cudaMalloc((void**)&d_zcr, sizeof(float));
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
    h_features->temporal_mean = sum / length;

    float variance_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        variance_sum += powf(h_signal[i] - h_features->temporal_mean, 2.0f);
    }
    h_features->temporal_variance = variance_sum / length;

    float kurtosis_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        kurtosis_sum += powf(h_signal[i] - h_features->temporal_mean, 4.0f);
    }
    h_features->temporal_kurtosis = kurtosis_sum / (length * powf(h_features->temporal_variance, 2.0f)) - 3.0f;

    float skewness_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        skewness_sum += powf(h_signal[i] - h_features->temporal_mean, 3.0f);
    }
    h_features->temporal_skewness = skewness_sum / (length * powf(h_features->temporal_variance, 1.5f));

    // Cleanup!
    cudaFree(d_signal);
    cudaFree(d_fft_result);
    cudaFree(pBuffer);
    nppiFFTFree_R_32f(pFFTSpec);
}

int main() {
    // Testing code:
    /////////////////////////////
    // float h_signal[SIGNAL_LENGTH];
    // float frequency = 440.0f; // A4 note, 440 Hz
    // float sampleRate = 44100.f; // Standard 44.1 kHz audio sample rate.
    // // Generate a sine wave
    // for (int i = 0; i < SIGNAL_LENGTH; ++i) {
    //     h_signal[i] = sinf(2.0f * PI * frequency * i / sampleRate);
    // }
    ///////////////////////////////
    float h_signal[SIGNAL_LENGTH] = {/* Fill with sample signal data */};
    Features h_features = {};

    // Perform feature extraction
    fft_feature_extraction(h_signal, SIGNAL_LENGTH, &h_features);

    // Print out the magnitude and other features.
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        std::cout << "Magnitude[" << i << "]: " << h_features.magnitude[i] << std::endl;
    }
    std::cout << "Spectral Centroid: " << h_features.spectral_centroid << std::endl;
    std::cout << "Spectral Bandwidth: " << h_features.spectral_bandwidth << std::endl;
    std::cout << "Spectral Flatness: " << h_features.spectral_flatness << std::endl;
    std::cout << "Zero Crossing Rate: " << h_features.zcr << std::endl;
    std::cout << "Energy: " << h_features.energy << std::endl;
    std::cout << "Temporal Mean: " << h_features.temporal_mean << std::endl;
    std::cout << "Temporal Variance: " << h_features.temporal_variance << std::endl;
    std::cout << "Temporal Skewness: " << h_features.temporal_skewness << std::endl;
    std::cout << "Temporal Kurtosis: " << h_features.temporal_kurtosis << std::endl;

    return 0;
}
