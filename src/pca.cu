#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <npp.h>

#define NUM_FEATURES 9  // Adjust based on the number of features you are using

// Utility function to load the feature matrix from a CSV file
void loadFeatureMatrix(const char* filename, std::vector<std::vector<float>>& featuresMatrix) {
    /*
    Description:
        This function loads a feature matrix from a CSV file into a 2D vector. 
        Each row of the CSV corresponds to a row in the feature matrix, and 
        each column corresponds to a feature.

    Inputs:
        const char* filename:
            - The path to the CSV file containing the feature matrix.
            - The CSV file should be formatted such that each row represents 
              a different data sample and each column represents a different feature.
              
        std::vector<std::vector<float>>& featuresMatrix:
            - A reference to a 2D vector where the loaded feature matrix will be stored.
            - The function will populate this vector with data from the CSV file.
    
    Outputs:
        void, no return:
            - The function loads the feature matrix into the provided vector, 
              modifying it directly. No return value is needed.
    */
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }

        featuresMatrix.push_back(row);
    }
}

// Compute the covariance matrix using NPP
void computeCovarianceMatrix(const std::vector<std::vector<float>>& featuresMatrix, std::vector<std::vector<float>>& covarianceMatrix) {
    /*
    Description:
        This function computes the covariance matrix of a given feature matrix. 
        The feature matrix is first centered by subtracting the mean of each feature, 
        and then the covariance matrix is calculated.

    Inputs:
        const std::vector<std::vector<float>>& featuresMatrix:
            - A 2D vector where each row represents a data sample and each column 
              represents a feature.
            - The input matrix is expected to be centered before covariance 
              computation, but this function also handles centering.

        std::vector<std::vector<float>>& covarianceMatrix:
            - A reference to a 2D vector where the computed covariance matrix 
              will be stored.
            - The function will populate this vector with the covariance values 
              for each pair of features.
    
    Outputs:
        void, no return:
            - The function directly modifies the provided covariance matrix vector 
              with computed values. No return value is needed.
    */
    int num_samples = featuresMatrix.size();
    int num_features = featuresMatrix[0].size();

    // Center the feature matrix by subtracting the mean of each feature
    std::vector<float> means(num_features, 0.0f);
    for (int j = 0; j < num_features; ++j) {
        for (int i = 0; i < num_samples; ++i) {
            means[j] += featuresMatrix[i][j];
        }
        means[j] /= num_samples;
    }

    std::vector<std::vector<float>> centeredMatrix = featuresMatrix;
    for (int j = 0; j < num_features; ++j) {
        for (int i = 0; i < num_samples; ++i) {
            centeredMatrix[i][j] -= means[j];
        }
    }

    // Convert centeredMatrix to a single array for NPP
    float* d_centeredMatrix;
    cudaMalloc((void**)&d_centeredMatrix, num_samples * num_features * sizeof(float));
    cudaMemcpy(d_centeredMatrix, &centeredMatrix[0][0], num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);

    float* d_covarianceMatrix;
    cudaMalloc((void**)&d_covarianceMatrix, num_features * num_features * sizeof(float));

    // NPP function to compute covariance matrix (this is a pseudocode placeholder)
    // Replace with the appropriate NPP function if available
    // nppiCovariance_32f(d_centeredMatrix, num_features, d_covarianceMatrix, num_samples, num_features);

    // Copy the covariance matrix back to the host
    cudaMemcpy(&covarianceMatrix[0][0], d_covarianceMatrix, num_features * num_features * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_centeredMatrix);
    cudaFree(d_covarianceMatrix);
}

// Perform eigenvalue decomposition using NPP
void performEigenDecomposition(const std::vector<std::vector<float>>& covarianceMatrix, std::vector<float>& eigenvalues, std::vector<std::vector<float>>& eigenvectors) {
    /*
    Description:
        This function performs eigenvalue decomposition on a covariance matrix.
        It calculates the eigenvalues and corresponding eigenvectors, which are 
        essential for Principal Component Analysis (PCA).

    Inputs:
        const std::vector<std::vector<float>>& covarianceMatrix:
            - A 2D vector representing the covariance matrix of the feature data.
            - The matrix is symmetric and square, with each dimension equal to 
              the number of features.

        std::vector<float>& eigenvalues:
            - A reference to a vector where the computed eigenvalues will be stored.
            - The function will populate this vector with the eigenvalues in 
              descending order of magnitude.

        std::vector<std::vector<float>>& eigenvectors:
            - A reference to a 2D vector where the computed eigenvectors will be stored.
            - Each column in this matrix corresponds to an eigenvector associated 
              with an eigenvalue in the `eigenvalues` vector.
    
    Outputs:
        void, no return:
            - The function modifies the `eigenvalues` and `eigenvectors` vectors 
              directly. No return value is needed.
    */
    int num_features = covarianceMatrix.size();

    float* d_covarianceMatrix;
    cudaMalloc((void**)&d_covarianceMatrix, num_features * num_features * sizeof(float));
    cudaMemcpy(d_covarianceMatrix, &covarianceMatrix[0][0], num_features * num_features * sizeof(float), cudaMemcpyHostToDevice);

    float* d_eigenvalues;
    float* d_eigenvectors;
    cudaMalloc((void**)&d_eigenvalues, num_features * sizeof(float));
    cudaMalloc((void**)&d_eigenvectors, num_features * num_features * sizeof(float));

    // NPP function to compute eigenvalues and eigenvectors (pseudocode placeholder)
    // nppiEigenValuesVectors_32f(d_covarianceMatrix, d_eigenvalues, d_eigenvectors, num_features);

    // Copy the results back to the host
    cudaMemcpy(&eigenvalues[0], d_eigenvalues, num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&eigenvectors[0][0], d_eigenvectors, num_features * num_features * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_covarianceMatrix);
    cudaFree(d_eigenvalues);
    cudaFree(d_eigenvectors);
}

// Project the data onto the principal components
void projectOntoPrincipalComponents(const std::vector<std::vector<float>>& featuresMatrix, const std::vector<std::vector<float>>& eigenvectors, std::vector<std::vector<float>>& pca_result) {
    /*
    Description:
        This function projects the original feature matrix onto the principal 
        components determined by the eigenvectors. The result is a new matrix 
        where each row represents the data in the reduced feature space.

    Inputs:
        const std::vector<std::vector<float>>& featuresMatrix:
            - A 2D vector where each row represents a data sample and each column 
              represents a feature.
            - This is the original data that will be projected onto the principal 
              components.

        const std::vector<std::vector<float>>& eigenvectors:
            - A 2D vector where each column represents an eigenvector corresponding 
              to one of the principal components.
            - The data will be projected onto these eigenvectors.

        std::vector<std::vector<float>>& pca_result:
            - A reference to a 2D vector where the projected data will be stored.
            - The function will populate this vector with the data represented 
              in the new, reduced feature space.
    
    Outputs:
        void, no return:
            - The function directly modifies the `pca_result` vector with the 
              projection of the data onto the principal components. No return value is needed.
    */
    int num_samples = featuresMatrix.size();
    int num_components = eigenvectors.size();

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_components; ++j) {
            pca_result[i][j] = 0;
            for (int k = 0; k < num_components; ++k) {
                pca_result[i][j] += featuresMatrix[i][k] * eigenvectors[k][j];
            }
        }
    }
}

int main() {
    std::vector<std::vector<float>> featuresMatrix;
    loadFeatureMatrix("features.csv", featuresMatrix);

    int num_samples = featuresMatrix.size();
    int num_features = featuresMatrix[0].size();

    // Initialize covariance matrix and perform PCA
    std::vector<std::vector<float>> covarianceMatrix(num_features, std::vector<float>(num_features, 0));
    computeCovarianceMatrix(featuresMatrix, covarianceMatrix);

    std::vector<float> eigenvalues(num_features, 0);
    std::vector<std::vector<float>> eigenvectors(num_features, std::vector<float>(num_features, 0));
    performEigenDecomposition(covarianceMatrix, eigenvalues, eigenvectors);

    // Project the data onto the principal components
    std::vector<std::vector<float>> pca_result(num_samples, std::vector<float>(num_features));
    projectOntoPrincipalComponents(featuresMatrix, eigenvectors, pca_result);

    // Write PCA results to a new CSV file
    std::ofstream outFile("pca_results.csv");
    for (const auto& row : pca_result) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();

    return 0;
}
