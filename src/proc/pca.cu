#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <npp.h>
#include <cusolverDn.h>
#include "../../include/pca.h"

// Utility function to load the feature matrix from a CSV file
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
void loadFeatureMatrix(const char* filename, std::vector<std::vector<float>>& featuresMatrix) {
    std::cout << "Loading feature matrix in pca.cu started." << std::endl;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        bool has_nan = false;

        while (std::getline(ss, value, ',')) {
            float num = std::stof(value);
            if (std::isnan(num)) {
                has_nan = true;
                break;
            }
            row.push_back(num);
        }

        if (!has_nan) {
            featuresMatrix.push_back(row);
        }
    }
    std::cout << "Loading feature matrix in pca.cu complete." << std::endl;
}

// Compute the covariance matrix using NPP and a custom CUDA kernel.

/*
    Description:
        This kernel manually computes the covariance matrix of a given feature matrix. 
        The feature matrix is first centered by subtracting the mean of each feature, 
        and then the covariance matrix is calculated.

    Inputs:
        float* d_centeredMatrix:
            - A pointer to the centered feature matrix on the device (GPU).
            - Each element is a floating-point value representing the deviation of a feature from its mean.

        float* d_covarianceMatrix:
            - A pointer to the covariance matrix on the device (GPU) where the computed covariance values will be stored.
            - This matrix will be populated with the covariance values for each pair of features.

        int num_samples:
            - The number of samples in the dataset.
            - This value determines how many data points are used in the covariance calculation.

        int num_features:
            - The number of features in the dataset.
            - This value determines the dimensions of the covariance matrix.
    
    Outputs:
        void, no return:
            - The function directly modifies the provided covariance matrix on the device with computed values.
*/
__global__ void computeCovarianceKernel(float* d_centeredMatrix, float* d_covarianceMatrix, int num_samples, int num_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_features && col < num_features) {
        float cov = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            cov += d_centeredMatrix[i * num_features + row] * d_centeredMatrix[i * num_features + col];
        }
        d_covarianceMatrix[row * num_features + col] = cov / (num_samples - 1);
    }
}

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
void computeCovarianceMatrix(const std::vector<std::vector<float>>& featuresMatrix, std::vector<std::vector<float>>& covarianceMatrix) {
    std::cout << "Starting computeCovarianceMatrix in pca.cu." << std::endl;
    int num_samples = featuresMatrix.size();
    int num_features = NUM_FEATURES;

    // Center the feature matrix by subtracting the mean of each feature
    std::vector<float> means(num_features, 0.0f);
    for (int j = 0; j < num_features; ++j) {
        for (int i = 0; i < num_samples; ++i) {
            means[j] += featuresMatrix[i][j];
        }
        means[j] /= num_samples;
    }

    std::vector<std::vector<float>> centeredMatrix(num_samples, std::vector<float>(num_features));
    for (int j = 0; j < num_features; ++j) {
        for (int i = 0; i < num_samples; ++i) {
            centeredMatrix[i][j] = featuresMatrix[i][j] - means[j];
        }
    }

    // Convert centeredMatrix to a single array for NPP
    float* d_centeredMatrix;
    cudaError_t err = cudaMalloc((void**)&d_centeredMatrix, num_samples * num_features * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_centeredMatrix: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMemcpy(d_centeredMatrix, centeredMatrix[0].data(), num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_centeredMatrix: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_centeredMatrix);
        return;
    }

    float* d_covarianceMatrix;
    err = cudaMalloc((void**)&d_covarianceMatrix, num_features * num_features * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_covarianceMatrix: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_centeredMatrix);
        return;
    }

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((num_features + blockSize.x - 1) / blockSize.x, (num_features + blockSize.y - 1) / blockSize.y);

    // Launch kernel to compute covariance matrix
    computeCovarianceKernel<<<gridSize, blockSize>>>(d_centeredMatrix, d_covarianceMatrix, num_samples, num_features);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed for computeCovarianceKernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_centeredMatrix);
        cudaFree(d_covarianceMatrix);
        return;
    }

    // Synchronize to ensure kernel execution is complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Device Synchronization failed after computeCovarianceKernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_centeredMatrix);
        cudaFree(d_covarianceMatrix);
        return;
    }

    // Copy the covariance matrix back to the host
    err = cudaMemcpy(covarianceMatrix[0].data(), d_covarianceMatrix, num_features * num_features * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for covarianceMatrix: " << cudaGetErrorString(err) << std::endl;
    }

    // Free device memory
    cudaFree(d_centeredMatrix);
    cudaFree(d_covarianceMatrix);
    std::cout << "Finishing computeCovarianceMatrix in pca.cu." << std::endl;
}

// Perform eigenvalue decomposition using cuSolver
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
void performEigenDecomposition(const std::vector<std::vector<float>>& covarianceMatrix, std::vector<float>& eigenvalues, std::vector<std::vector<float>>& eigenvectors) {
    std::cout << "Starting performEigenDecomposition in pca.cu." << std::endl;
    int num_features = covarianceMatrix.size();

    // Allocate covariance matrix memory
    float* d_covarianceMatrix = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_covarianceMatrix, num_features * num_features * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_covarianceMatrix: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMemcpy(d_covarianceMatrix, covarianceMatrix[0].data(), num_features * num_features * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_covarianceMatrix: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_covarianceMatrix);
        return;
    }

    // Allocate memory for eigenvalues and eigenvectors
    float* d_eigenvalues = nullptr;
    float* d_eigenvectors = nullptr;
    err = cudaMalloc((void**)&d_eigenvalues, num_features * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_eigenvalues: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_covarianceMatrix);
        return;
    }
    err = cudaMalloc((void**)&d_eigenvectors, num_features * num_features * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_eigenvectors: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_covarianceMatrix);
        cudaFree(d_eigenvalues);
        return;
    }

    // Initialize solver
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization failed" << std::endl;
        cudaFree(d_covarianceMatrix);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        return;
    }

    // Allocate memory for solver stuff
    int work_size = 0;
    int* devInfo = NULL;
    err = cudaMalloc((void**)&devInfo, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for devInfo: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_covarianceMatrix);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        cusolverDnDestroy(cusolverH);
        return;
    }

    // Allocate space for the workspace
    cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, num_features, d_covarianceMatrix, num_features, d_eigenvalues, &work_size);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to compute buffer size for eigenvalue decomposition" << std::endl;
        cudaFree(d_covarianceMatrix);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        cudaFree(devInfo);
        cusolverDnDestroy(cusolverH);
        return;
    }
    std::cout << "Buffer size for eigenvalue decomposition: " << work_size << std::endl;
    float* work = nullptr;
    err = cudaMalloc((void**)&work, work_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for work: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_covarianceMatrix);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        cusolverDnDestroy(cusolverH);
        cudaFree(devInfo);
        return;
    }

    // Perform the eigenvalue decomposition
    std::cout << "Starting cusolverDnSsyevd for eigenvalue decomposition." << std::endl;
    cusolver_status = cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, num_features, d_covarianceMatrix, num_features, d_eigenvalues, work, work_size, devInfo);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to perform eigenvalue decomposition" << std::endl;
        cudaFree(d_covarianceMatrix);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        cudaFree(work);
        cudaFree(devInfo);
        cusolverDnDestroy(cusolverH);
        return;
    }

    std::cout << "Eigenvalue decomposition completed." << std::endl;

    // Copy the results back to the host
    err = cudaMemcpy(eigenvalues.data(), d_eigenvalues, num_features * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_eigenvalues: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemcpy(eigenvectors[0].data(), d_eigenvectors, num_features * num_features * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_eigenvectors: " << cudaGetErrorString(err) << std::endl;
    }

    
    cudaFree(work);
    cudaFree(devInfo);
    cudaFree(d_eigenvectors);
    cudaFree(d_eigenvalues);
    cudaFree(d_covarianceMatrix);
    cusolverDnDestroy(cusolverH);
    std::cout << "Finishing performEigenDecomposition in pca.cu." << std::endl;
}

// Project the data onto the principal components
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
void projectOntoPrincipalComponents(const std::vector<std::vector<float>>& featuresMatrix, const std::vector<std::vector<float>>& eigenvectors, std::vector<std::vector<float>>& pca_result) {
    std::cout << "Starting projectOntoPrincipalComponents in pca.cu." << std::endl;
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
    std::cout << "Finishing projectOntoPrincipalComponents in pca.cu." << std::endl;
}
