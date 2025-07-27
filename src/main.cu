// src/main.cu
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " - " \
              << cudaGetErrorString(err) << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

struct DeviceBuf {
    unsigned char* d_in  = nullptr;
    unsigned char* d_out = nullptr;
    size_t bytes = 0;
    cudaStream_t stream = nullptr;
};

__global__ void stylize_kernel(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int num_pixels, float alpha, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;
    int base = idx * 3; // BGR
    // Simple brightness/contrast-ish per-pixel op:
    float b = in[base + 0] * alpha + beta;
    float g = in[base + 1] * alpha + beta;
    float r = in[base + 2] * alpha + beta;
    // mild channel remap for visible change
    out[base + 0] = (unsigned char)max(0.f, min(255.f, 0.7f * b + 0.3f * g));
    out[base + 1] = (unsigned char)max(0.f, min(255.f, 0.7f * g + 0.3f * r));
    out[base + 2] = (unsigned char)max(0.f, min(255.f, 0.7f * r + 0.3f * b));
}

static void ensure_dir(const std::string& p) {
    std::filesystem::create_directories(p);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <input_dir> <output_dir> <alpha> [beta] [num_streams]\n"
                  << "Example: " << argv[0] << " data/input output 1.2 20 4\n";
        return 1;
    }
    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    float alpha = std::stof(argv[3]);
    float beta  = (argc >= 5) ? std::stof(argv[4]) : 0.0f;
    int n_streams = (argc >= 6) ? std::stoi(argv[5]) : NUM_STREAMS;

    ensure_dir(output_dir);

    // Collect images
    std::vector<std::string> paths;
    for (auto& e : std::filesystem::directory_iterator(input_dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext==".tif" || ext==".tiff")
            paths.push_back(e.path().string());
    }
    if (paths.empty()) {
        std::cerr << "No images found in " << input_dir << "\n";
        return 1;
    }

    std::cout << "Found " << paths.size() << " images\n";

    // Create streams and per-stream device buffers (ping-pong over streams)
    n_streams = std::max(1, n_streams);
    std::vector<DeviceBuf> dev(n_streams);

    for (int i = 0; i < n_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&dev[i].stream));
    }

    std::ofstream log_csv(output_dir + "/timings.csv");
    log_csv << "image,rows,cols,bytes,ms_h2d,ms_kernel,ms_d2h,total_ms\n";

    auto t0_all = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < paths.size(); ++i) {
        const int sid = static_cast<int>(i % n_streams);

        cv::Mat img = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to read: " << paths[i] << "\n";
            continue;
        }
        if (!img.isContinuous()) img = img.clone();

        const int rows = img.rows, cols = img.cols;
        const int num_pixels = rows * cols;
        const size_t bytes = static_cast<size_t>(num_pixels) * 3;

        // (Re)alloc per stream buffers if needed
        if (bytes != dev[sid].bytes) {
            if (dev[sid].d_in)  CUDA_CHECK(cudaFree(dev[sid].d_in));
            if (dev[sid].d_out) CUDA_CHECK(cudaFree(dev[sid].d_out));
            CUDA_CHECK(cudaMalloc(&dev[sid].d_in,  bytes));
            CUDA_CHECK(cudaMalloc(&dev[sid].d_out, bytes));
            dev[sid].bytes = bytes;
        }

        cv::Mat out(rows, cols, CV_8UC3);
        cudaEvent_t e_h2d_start, e_h2d_end, e_k_start, e_k_end, e_d2h_start, e_d2h_end;
        CUDA_CHECK(cudaEventCreate(&e_h2d_start));
        CUDA_CHECK(cudaEventCreate(&e_h2d_end));
        CUDA_CHECK(cudaEventCreate(&e_k_start));
        CUDA_CHECK(cudaEventCreate(&e_k_end));
        CUDA_CHECK(cudaEventCreate(&e_d2h_start));
        CUDA_CHECK(cudaEventCreate(&e_d2h_end));

        CUDA_CHECK(cudaEventRecord(e_h2d_start, dev[sid].stream));
        CUDA_CHECK(cudaMemcpyAsync(dev[sid].d_in, img.ptr<unsigned char>(), bytes,
                                   cudaMemcpyHostToDevice, dev[sid].stream));
        CUDA_CHECK(cudaEventRecord(e_h2d_end, dev[sid].stream));

        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;
        CUDA_CHECK(cudaEventRecord(e_k_start, dev[sid].stream));
        stylize_kernel<<<blocks, threads, 0, dev[sid].stream>>>(
            dev[sid].d_in, dev[sid].d_out, num_pixels, alpha, beta);
        CUDA_CHECK(cudaEventRecord(e_k_end, dev[sid].stream));

        CUDA_CHECK(cudaEventRecord(e_d2h_start, dev[sid].stream));
        CUDA_CHECK(cudaMemcpyAsync(out.ptr<unsigned char>(), dev[sid].d_out, bytes,
                                   cudaMemcpyDeviceToHost, dev[sid].stream));
        CUDA_CHECK(cudaEventRecord(e_d2h_end, dev[sid].stream));

        CUDA_CHECK(cudaStreamSynchronize(dev[sid].stream));

        float ms_h2d=0, ms_k=0, ms_d2h=0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, e_h2d_start, e_h2d_end));
        CUDA_CHECK(cudaEventElapsedTime(&ms_k,   e_k_start,   e_k_end));
        CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, e_d2h_start, e_d2h_end));

        float total_ms = ms_h2d + ms_k + ms_d2h;

        std::string fname = std::filesystem::path(paths[i]).filename().string();
        std::string out_path = output_dir + "/proc_" + fname;
        cv::imwrite(out_path, out);

        log_csv << fname << "," << rows << "," << cols << "," << bytes
                << "," << ms_h2d << "," << ms_k << "," << ms_d2h << "," << total_ms << "\n";

        CUDA_CHECK(cudaEventDestroy(e_h2d_start));
        CUDA_CHECK(cudaEventDestroy(e_h2d_end));
        CUDA_CHECK(cudaEventDestroy(e_k_start));
        CUDA_CHECK(cudaEventDestroy(e_k_end));
        CUDA_CHECK(cudaEventDestroy(e_d2h_start));
        CUDA_CHECK(cudaEventDestroy(e_d2h_end));

        if ((i+1) % 10 == 0) {
            std::cout << "Processed " << (i+1) << "/" << paths.size() << " images\r";
        }
    }

    for (int i = 0; i < n_streams; ++i) {
        if (dev[i].d_in)  cudaFree(dev[i].d_in);
        if (dev[i].d_out) cudaFree(dev[i].d_out);
        cudaStreamDestroy(dev[i].stream);
    }

    auto t1_all = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t1_all - t0_all).count();
    std::cout << "\nDone. Total wall time: " << total_s << " s\n";
    log_csv.close();

    return 0;
}
