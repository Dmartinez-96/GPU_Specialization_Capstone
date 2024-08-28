#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include "../include/wav_loader.h"
#include "../include/feature_extraction.h"
#include "../include/pca.h"

int main() {
    // std::vector<std::string> wavFiles;
    // std::map<std::string, std::string> fileToInstrumentMap;

    // // Populate the wavFiles vector and fileToInstrumentMap based on filenames in data/WAV_files
    // for (const auto& entry : std::filesystem::directory_iterator("data/WAV_files")) {
    //     if (entry.path().extension() == ".wav") {
    //         std::string filePath = entry.path().string();
    //         wavFiles.push_back(filePath);
            
    //         std::string lowerFilePath = toLowerCase(filePath);
    //         // Map each filename to an instrument label
    //         if (lowerFilePath.find("guitar") != std::string::npos || lowerFilePath.find("gtr") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "guitar";
    //         } else if (lowerFilePath.find("piano") != std::string::npos || lowerFilePath.find("pno") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "piano";
    //         } else if (lowerFilePath.find("violin") != std::string::npos || lowerFilePath.find("vln") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "violin";
    //         } else if (lowerFilePath.find("cello") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "cello";
    //         } else if (lowerFilePath.find("harpsichord") != std::string::npos || lowerFilePath.find("harpsi") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "harpsichord";
    //         } else if (lowerFilePath.find("gongs") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "gongs";
    //         } else if (lowerFilePath.find("bass") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "bass";
    //         } else if (lowerFilePath.find("marimba") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "marimba";
    //         } else if (lowerFilePath.find("oboe") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "oboe";
    //         } else if (lowerFilePath.find("shakuhachi") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "shakuhachi";
    //         } else if (lowerFilePath.find("sitar") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "sitar";
    //         } else if (lowerFilePath.find("flute") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "flute";
    //         } else if (lowerFilePath.find("sax") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "saxophone";
    //         } else if (lowerFilePath.find("trumpet") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "trumpet";
    //         } else if (lowerFilePath.find("viola") != std::string::npos) {
    //             fileToInstrumentMap[filePath] = "viola";
    //         }
    //     }
    // }

    // // Process each WAV file, extract features, and associate with instrument label
    // std::vector<std::vector<float>> featuresMatrix;
    // std::vector<std::string> labels;

    // std::cout << "Extracting data features with cuFFT and Aquila. Please wait, this may take a while." << std::endl;
    // int progCounter = 0;
    // for (const auto& file : wavFiles) {
    //     float h_signal[SIGNAL_LENGTH];
    //     loadWavFileAquila(file.c_str(), h_signal, SIGNAL_LENGTH);

    //     Features h_features;
    //     fft_feature_extraction(h_signal, SIGNAL_LENGTH, &h_features);

    //     // Store extracted features
    //     std::vector<float> features_row = {
    //         h_features.spectralCentroid,
    //         h_features.spectralFlatness,
    //         h_features.spectralBandwidth,
    //         h_features.zcr,
    //         h_features.energy,
    //         h_features.temporalMean,
    //         h_features.temporalVariance,
    //         h_features.temporalSkewness,
    //         h_features.temporalKurtosis
    //     };
    //     featuresMatrix.push_back(features_row);

    //     // Store the corresponding instrument label
    //     labels.push_back(fileToInstrumentMap[file]);
    //     progCounter++;

    //     // Output a message indicating that processing is done for this file
    //     std::cout << "Completed processing file: " << file << "      (" << progCounter << "/14 done)" << std::endl;
    // }

    // // Optionally write the features and labels to a CSV file
    // std::ofstream outFile("features_with_labels.csv");
    
    // for (size_t i = 0; i < featuresMatrix.size(); ++i) {
    //     for (size_t j = 0; j < featuresMatrix[i].size(); ++j) {
    //         outFile << featuresMatrix[i][j];
    //         if (j < featuresMatrix[i].size() - 1) outFile << ",";
    //     }
    //     outFile << "," << labels[i] << "\n";
    // }
    // outFile.close();
    // std::cout << "Completed features_with_labels.csv!" << std::endl;

    // Comment out next line of code until next comment to restore full program -- this is a debug point, basically
    std::vector<std::vector<float>> featuresMatrix;
    // End comment section, uncomment all of beginning stuff now
    // Load featuresMatrix from the CSV file
    std::cout << "Loading features from CSV file." << std::endl;
    loadFeatureMatrix("features_with_labels.csv", featuresMatrix);
    std::cout << "Completed loading features from CSV!" << std::endl;

    // Perform PCA
    std::vector<std::vector<float>> covarianceMatrix(NUM_FEATURES, std::vector<float>(NUM_FEATURES, 0));
    computeCovarianceMatrix(featuresMatrix, covarianceMatrix);
    
    std::cout << "Completed computeCovarianceMatrix!" << std::endl;

    std::vector<float> eigenvalues(NUM_FEATURES, 0);
    std::vector<std::vector<float>> eigenvectors(NUM_FEATURES, std::vector<float>(NUM_FEATURES, 0));
    performEigenDecomposition(covarianceMatrix, eigenvalues, eigenvectors);
    
    std::cout << "Completed performEigenDecomposition!" << std::endl;

    std::vector<std::vector<float>> pca_result(featuresMatrix.size(), std::vector<float>(NUM_FEATURES));
    projectOntoPrincipalComponents(featuresMatrix, eigenvectors, pca_result);
    
    std::cout << "Completed projectOntoPrincipalComponents!" << std::endl;

    // Write PCA results to a CSV file
    std::ofstream pcaFile("pca_results.csv");
    for (const auto& row : pca_result) {
        for (size_t i = 0; i < row.size(); ++i) {
            pcaFile << row[i];
            if (i < row.size() - 1) pcaFile << ",";
        }
        pcaFile << "\n";
    }
    pcaFile.close();

    return 0;
}
