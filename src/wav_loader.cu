#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <aquila/global.h>
#include <aquila/source/WaveFile.h>
#include "feature_extraction.cu"

#define SIGNAL_LENGTH 1024

void loadWavFileAquila(const char* filename, float* signal, int length) {
    /*
    Documentation:
        Loads a WAV file into Aquila for processing signal data.
    Inputs:
        const char* filename:
            - Name of the wav file.

        float* signal:
            - The signal contained within the wav file.

        int length:
            - Length of signal data array.

    Outputs:
        void, no return:
            - The function analyzes WAV files to get signal data using Aquila.
    */
    
    Aquila::WaveFile wav(filename);

    if (wav.getSamplesCount() < length) {
        std::cerr << "Warning: WAV file is shorter than expected length ("
            << length << " samples)" << std::endl;
    }

    // Ensure the data fits into the signal buffer
    for (int i = 0; i < length && i < wav.getSamplesCount(); ++i) {
        signal[i] = wav.sample(i);
    }
}

int main() {
    std::vector<std::string> wavFiles;
    std::vector<std::vector<float>> featuresMatrix;

    // Load all WAV files from the directory
    for (const auto& entry : std::filesystem::directory_iterator("data/WAV_files")) {
        if (entry.path().extension() == ".wav") {
            wavFiles.push_back(entry.path().string());
        }
    }

    for (const auto& file : wavFiles) {
        float h_signal[SIGNAL_LENGTH];
        loadWavFileAquila(file.c_str(), h_signal, SIGNAL_LENGTH);

        Features h_features = {};
        fft_feature_extraction(h_signal, SIGNAL_LENGTH, &h_features);

        // Store features in a vector
        std::vector<float> features_row = {
            h_features.spectralCentroid,
            h_features.spectralFlatness,
            h_features.spectralBandwidth,
            h_features.zcr,
            h_features.energy,
            h_features.temporal_mean,
            h_features.temporal_variance,
            h_features.temporal_skewness,
            h_features.temporal_kurtosis
        };
        featuresMatrix.push_back(features_row);
    }

    // Write features to a CSV file
    std::ofstream outFile("features.csv");
    for (const auto& row : featuresMatrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();

    return 0;
}