#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <aquila/global.h>
#include <aquila/source/WaveFile.h>
#include "../../include/wav_loader.h"
#include "../../include/feature_extraction.h"

// Use Aquila to load in WAV file
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
void loadWavFileAquila(const char* filename, float* signal, int length) {
    // Ensure the signal array is not a nullptr
    if (signal == nullptr) {
        std::cerr << "Error: Signal array is null." << std::endl;
        return;
    }

    Aquila::WaveFile wav(filename);

    if (wav.getSamplesCount() < length) {
        std::cerr << "Warning: WAV file is shorter than expected length ("
            << length << " samples)" << std::endl;
    }

    // Ensure the data fits into the signal buffer
    for (int i = 0; i < length && i < wav.getSamplesCount(); ++i) {
        signal[i] = wav.sample(i);
    }

    // Optionally, fill the remaining buffer with zeros if WAV file is shorter
    for (int i = wav.getSamplesCount(); i < length; ++i) {
        signal[i] = 0.0f;
    }
}