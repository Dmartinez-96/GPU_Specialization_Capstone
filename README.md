# PCA of Instrumental Sound Data using NVIDIA NPP with CUDA

## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) library with CUDA and Aquila to perform signal processing on audio data for different musical instruments. The goal is to utilize GPU acceleration to efficiently analyze several .WAV files of instrumental audio and extract audio features, then perform Principal Component Analysis on these features, leveraging the computational power of modern GPUs. The project is a part of the CUDA at Scale for the Enterprise course and serves to demonstrate my ability to utilize CUDA, NPP, and other functionalities to perform signal processing.

## Code Organization

- ```bin/``` This folder holds the binary/executable code that is built automatically or manually by the make commands.

- ```data/``` This folder holds the example WAV files used as data in the computational steps for signal processing and Principal Component Analysis.
    - ```data/WAV_files``` This subfolder holds the WAV files for 14 instruments used in the main program.
    - ```data/Extra_WAV_files``` Extra WAV files for various instruments if the user is interested in doing further analysis.

- ```lib/``` This folder is here if anyone wants to add more libraries to link, but all others can be installed via the operating-system specific package manager as in the instructions in ```INSTALL``` below.

- ```src/``` The source code is here, with programs split in a hierarchical fashion according to function.
    - ```src/proc/``` This subdirectory contains the .cu files that perform processing on the signal data.
        - ```src/proc/wav_loader.cu``` This file loads WAV files using the Aquila library, extracts the signal, and helps to pass it to the feature_extraction function.
        - ```src/proc/feature_extraction.cu``` This file handles extracting various features from the WAV files, such as spectral centroid, flatness, bandwidth, zero-crossing rate (ZCR), energy, and temporal features using CUDA and NPP signal processing routines. Logic maps the filenames to instrument labels based on substrings, saving extracted and calculated features and corresponding instrument labels to a CSV.
        - ```pca.cu``` This file loads the feature matrix from the CSV file and uses NPP features to compute the covariance matrix, perform eigenvalue decomposition, and project the data onto the principal components. The eigenvalue results are saved to a second CSV and the PCA results are then saved to a third CSV.
    - ```src/vis/```
        - ```src/vis/VisualizeResults.ipynb``` This Jupyter Notebook visualizes the features extracted by ```feature_extraction.cu``` and the PCA data from ```pca.cu```.


- ```README.md``` This file is what you are reading now -- it gives descriptions of how the program runs and instructions.

- ```INSTALL``` This file holds a human-readable set of instructions for installing the code so that it can be executed on various operating systems, like Windows, Linux (Ubuntu), and MacOS. Extensive testing has been performed on Linux (Ubuntu). 

- ```Makefile``` This is a script to compile the executable program into the ```bin/``` directory. Current compiler flags are set with ```NVCCFLAGS = -arch=sm_75```, which may need to be adjusted for other architectures.

- ```requirements.txt``` A list of Python modules required by the visualization Jupyter Notebook in ```src/vis/PCA_visualization.ipynb```. Install with ```pip install -r requirements.txt```.

- ```results/``` This folder holds the results of the program once it has executed. It also has sample results to provide the user something to compare against.
    - ```results/Example_results/``` This folder holds examples of the three CSV files produced by analyzing the files in ```data/WAV_files/```. It also holds example .png images for the resultant plots from the Jupyter Notebook in ```src/vis/```.

## Key Concepts

Performance Strategies, Signal Processing, NPP Library

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows, possibly MacOS with further testing

## Supported CPU Architecture

x86_64

## CUDA APIs involved
This project involves several key CUDA APIs, particularly those related to NVIDIA Performance Primitives (NPP) and cuSOLVER:
- NPP Signal Processing Functions: NPP provides a set of signal processing functions that are heavily optimized for NVIDIA GPUs. These functions are used in feature_extraction.cu for tasks such as calculating the spectral centroid, flatness, and other features.

- cuBLAS: While not directly invoked, cuBLAS is used indirectly by cuSOLVER for matrix operations. cuBLAS provides GPU-accelerated linear algebra routines and is a core part of the CUDA toolkit.

- cuSOLVER: The cuSOLVER library is used in pca.cu to perform eigenvalue decomposition. This is essential for Principal Component Analysis (PCA), where we decompose the covariance matrix into its eigenvalues and eigenvectors to identify the principal components.

- CUDA Runtime API: Standard CUDA runtime functions such as cudaMalloc, cudaMemcpy, and cudaFree are used extensively for managing memory on the GPU.

## Dependencies needed to build/run
To build and run this project, you'll need the following dependencies:
- CUDA Toolkit: Ensure you have the appropriate version of the CUDA toolkit installed. The project is developed with CUDA 11.4, but it should be compatible with other versions as well. Installation instructions are available in the INSTALL file.

- NVIDIA Performance Primitives (NPP): This is part of the CUDA toolkit and is used for signal processing tasks.

- cuSOLVER: Also part of the CUDA toolkit, used for performing eigenvalue decomposition in PCA.

- Aquila: A C++ library for signal processing, particularly for handling WAV files. Installation instructions are provided in the INSTALL file. If not installed system-wide, consider placing Aquila in the lib/ folder and linking it during the build process.

- Python3 (for visualization): The src/vis/VisualizeResults.ipynb Jupyter Notebook requires Python. Install the necessary Python packages using the requirements.txt file by running: ```pip install -r requirements.txt```.

## Prerequisites
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

- NVIDIA GPU: Ensure your system has an NVIDIA GPU with compute capability 3.5 or higher.

- CUDA Toolkit: Download and install the [CUDA Toolkit 11.5](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

- C++ Compiler: A modern C++ compiler compatible with CUDA, such as GCC on Linux or MSVC on Windows.

- Python: Required for running the visualization Jupyter Notebook. Ensure you have Python 3.6+ installed along with Jupyter Notebook.

## Build and Run

### Windows

### Linux
Navigate to the project directory and run make to build the project:
```
$ cd <project_dir>
$ make
```

The results should look similar to the image below:

![Example_image_make_command](https://github.com/user-attachments/assets/f8d7e93f-9f51-4eff-8a5f-667085b83eae)

## Running the Program
After building the project, you can run the program using the following commands. This command will execute the compiled binary, saving three CSV files with signal features, covariance matrix eigenvalues, and its projection onto the principal components. The results can then be run in the Jupyter Notebook:

__Linux__:

```bash
./bin/PCA_Instrument.exe
```
__Windows__:

```bash
.\bin\PCA_Instrument.exe
```

Various progress statements will be printed to the terminal as the program proceeds.

The result should look similar to the image below:

![Example_image_execute](https://github.com/user-attachments/assets/468d00e9-34e4-4286-a6d1-c758003a0807)

- Cleaning Up
To clean up the compiled binaries and other generated files, run:

```bash
- Copy code
make clean
```

This will remove all files in the ```bin/``` directory and all generated CSV files in the ```results/``` directory.

## Analyzing the Results with VisualizeResults.ipynb

After successfully running, the results may be visualized using the Jupyter Notebook in ```src/vis/```, which has three sections. We can do the following.

### Visualize the original signal features with labels.

![signal_feature_pair_plots](https://github.com/user-attachments/assets/099ebe87-d148-4ecc-9f6e-73e69fb6cf21)

### Plot eigenvalues in a scree plot to explain variance by each principal component.

![scree_plot](https://github.com/user-attachments/assets/62f42d34-215d-486b-8131-94baebc9f7b9)

### Plot all principal components against one another in pairs by instrument..

![principal_component_pair_plots](https://github.com/user-attachments/assets/36e3f07e-80ab-4f57-8cfc-acce79c6784e)
