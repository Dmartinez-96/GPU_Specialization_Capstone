#ifndef PCA_H
#define PCA_H

#include <vector>

// Define the number of features
#define NUM_FEATURES 9  // Adjust based on the number of features you are using

// Function prototypes
void loadFeatureMatrix(const char* filename, std::vector<std::vector<float>>& featuresMatrix);
void computeCovarianceMatrix(const std::vector<std::vector<float>>& featuresMatrix, std::vector<std::vector<float>>& covarianceMatrix);
void performEigenDecomposition(const std::vector<std::vector<float>>& covarianceMatrix, std::vector<float>& eigenvalues, std::vector<std::vector<float>>& eigenvectors);
void projectOntoPrincipalComponents(const std::vector<std::vector<float>>& featuresMatrix, const std::vector<std::vector<float>>& eigenvectors, std::vector<std::vector<float>>& pca_result);

#endif // PCA_H
