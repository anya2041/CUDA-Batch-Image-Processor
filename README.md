# CUDA-Batch-Image-Processor
A  GPU-accelerated image processing tool built with CUDA 11.8 and OpenCV, capable of processing hundreds of images in batch mode. The project demonstrates CUDA kernels for scaling and filtering images, with efficient memory handling and multi-threaded execution. Includes a Makefile and CLI for flexible execution.

# CUDA Batch Image Processor (CUDA at Scale Independent Project)

## Goal
Process **hundreds of images** on GPU with CUDA, measure transfer/compute time, and prove scaling using **multiple CUDA streams**.

## What it does
- Loads all images from `data/input/`.
- Copies them to the GPU in **batches across N streams** (default 4).
- Applies a **per-pixel CUDA kernel** (brightness/contrast + channel remap) to every image.
- Copies results back and writes to `output/`.
- Logs timings (H2D, kernel, D2H, total) to `output/timings.csv`.

## Build
- Windows, CUDA 11.8 (for Maxwell cc 5.0 GPUs like MX130).
- Open a **x64 Native Tools Command Prompt**.
```bat
make clean
make
