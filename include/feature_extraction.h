#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>

// Constants
const int SIGNAL_LENGTH = 1024;

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
    float temporalKurtosis;        // Temporal Kurtosis
    float temporalSkewness;        // Temporal Skewness
    float temporalVariance;        // Temporal Variance
};

// Function prototypes
std::string toLowerCase(const std::string& str);

__global__ void scaleSignal(float* d_signal, float scale);

__global__ void calculateZCR(float* d_signal, int length, float* d_zcr);

void fft_feature_extraction(float* h_signal, int length, Features* h_features);

#endif // FEATURE_EXTRACTION_H
