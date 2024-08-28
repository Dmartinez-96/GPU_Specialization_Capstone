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

## Key Concepts

Performance Strategies, Signal Processing, NPP Library

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

## Dependencies needed to build/run
[FreeImage](../../README.md#freeimage), [NPP](../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```


## Running the Program
After building the project, you can run the program using the following command:

```bash
Copy code
make run
```

This command will execute the compiled binary, rotating the input image (Lena.png) by 45 degrees, and save the result as Lena_rotated.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
- Copy code
./bin/imageRotationNPP --input data/Lena.png --output data/Lena_rotated.png
```

- Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
- Copy code
make clean
```

This will remove all files in the bin/ directory.
