# IACV_background_subtractor

## Description
Fast C++ implementation of the PBAS Algorithm for Background Subtraction tasks in traffic environments.

## Compiling the source
Run the following to compile the source code (requires OpenCV libraries):
```
# RGB version
g++ -std=c++11 -g -o run_PBASrbg $(pkg-config --cflags --libs opencv4) run_PBASrbg.cpp

# Grayscale version
g++ -std=c++11 -g -o run_PBASgrayscale $(pkg-config --cflags --libs opencv4) run_PBASgrayscale.cpp
```

## Running the algorithm
```
# RGB version
./run_PBASrbg path/to/video.mp4

# Grayscale version
./run_PBASgrayscale path/to/video.mp4
```
