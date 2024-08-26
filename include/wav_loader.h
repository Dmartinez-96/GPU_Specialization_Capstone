#ifndef WAV_LOADER_H
#define WAV_LOADER_H

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <aquila/global.h>
#include <aquila/source/WaveFile.h>

// Function prototype
void loadWavFileAquila(const char* filename, float* signal, int length);

#endif // WAV_LOADER_H
