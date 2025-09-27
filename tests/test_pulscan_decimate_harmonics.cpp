#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#ifndef AA_WITH_PULSCAN
#define AA_WITH_PULSCAN 1
#endif
#include "aa_device_pulscan.hpp"

using namespace astroaccelerate;

namespace {

constexpr float kTolerance = 1e-5f;

bool approx_equal(float a, float b) {
  return std::fabs(a - b) <= kTolerance;
}

void check_cuda(cudaError_t status, const char *message) {
  if (status != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(1);
  }
}

void cpu_decimate(const std::vector<float> &magnitudes,
                  std::vector<float> &out_dec2,
                  std::vector<float> &out_dec3,
                  std::vector<float> &out_dec4) {
  const long num_magnitudes = static_cast<long>(magnitudes.size());

  for (long idx = 0; idx < num_magnitudes; ++idx) {
    if (idx * 2 + 1 < num_magnitudes) {
      float fundamental = magnitudes[idx];
      float harmonic1a = magnitudes[idx * 2];
      float harmonic1b = magnitudes[idx * 2 + 1];
      out_dec2[idx] = fundamental + harmonic1a + harmonic1b;

      if (idx * 3 + 2 < num_magnitudes) {
        float harmonic2a = magnitudes[idx * 3];
        float harmonic2b = magnitudes[idx * 3 + 1];
        float harmonic2c = magnitudes[idx * 3 + 2];
        out_dec3[idx] = out_dec2[idx] + harmonic2a + harmonic2b + harmonic2c;

        if (idx * 4 + 3 < num_magnitudes) {
          float harmonic3a = magnitudes[idx * 4];
          float harmonic3b = magnitudes[idx * 4 + 1];
          float harmonic3c = magnitudes[idx * 4 + 2];
          float harmonic3d = magnitudes[idx * 4 + 3];
          out_dec4[idx] = out_dec3[idx] + harmonic3a + harmonic3b + harmonic3c + harmonic3d;
        }
      }
    }
  }
}

} // namespace

int main() {
  std::cout << "Running test_pulscan_decimate_harmonics.cpp" << std::endl;

  constexpr long num_magnitudes = 96;
  std::vector<float> magnitudes(num_magnitudes);
  for (long i = 0; i < num_magnitudes; ++i) {
    magnitudes[i] = std::sin(static_cast<float>(i) * 0.11f) + static_cast<float>(i) * 0.05f;
  }

  std::vector<float> expected_dec2(num_magnitudes, 0.0f);
  std::vector<float> expected_dec3(num_magnitudes, 0.0f);
  std::vector<float> expected_dec4(num_magnitudes, 0.0f);
  cpu_decimate(magnitudes, expected_dec2, expected_dec3, expected_dec4);

  float *device_magnitudes = nullptr;
  float *device_dec2 = nullptr;
  float *device_dec3 = nullptr;
  float *device_dec4 = nullptr;

  check_cuda(cudaMalloc(&device_magnitudes, num_magnitudes * sizeof(float)), "cudaMalloc magnitudes failed");
  check_cuda(cudaMalloc(&device_dec2, num_magnitudes * sizeof(float)), "cudaMalloc dec2 failed");
  check_cuda(cudaMalloc(&device_dec3, num_magnitudes * sizeof(float)), "cudaMalloc dec3 failed");
  check_cuda(cudaMalloc(&device_dec4, num_magnitudes * sizeof(float)), "cudaMalloc dec4 failed");

  check_cuda(cudaMemcpy(device_magnitudes, magnitudes.data(), num_magnitudes * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy magnitudes failed");

  check_cuda(cudaMemset(device_dec2, 0, num_magnitudes * sizeof(float)), "cudaMemset dec2 failed");
  check_cuda(cudaMemset(device_dec3, 0, num_magnitudes * sizeof(float)), "cudaMemset dec3 failed");
  check_cuda(cudaMemset(device_dec4, 0, num_magnitudes * sizeof(float)), "cudaMemset dec4 failed");

  dim3 block_size(96);
  dim3 grid_size(1);
  call_kernel_pulscan_decimate_harmonics(grid_size, block_size, 0,
                                         device_magnitudes, device_dec2, device_dec3, device_dec4,
                                         num_magnitudes);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after decimate_harmonics failed");
  check_cuda(cudaGetLastError(), "decimate_harmonics kernel launch error");

  std::vector<float> gpu_dec2(num_magnitudes, 0.0f);
  std::vector<float> gpu_dec3(num_magnitudes, 0.0f);
  std::vector<float> gpu_dec4(num_magnitudes, 0.0f);

  check_cuda(cudaMemcpy(gpu_dec2.data(), device_dec2, num_magnitudes * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy dec2 failed");
  check_cuda(cudaMemcpy(gpu_dec3.data(), device_dec3, num_magnitudes * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy dec3 failed");
  check_cuda(cudaMemcpy(gpu_dec4.data(), device_dec4, num_magnitudes * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy dec4 failed");

  for (long i = 0; i < num_magnitudes; ++i) {
    if (!approx_equal(gpu_dec2[i], expected_dec2[i])) {
      std::cerr << "dec2 mismatch at index " << i << ": got " << gpu_dec2[i]
                << ", expected " << expected_dec2[i] << std::endl;
      return 1;
    }
    if (!approx_equal(gpu_dec3[i], expected_dec3[i])) {
      std::cerr << "dec3 mismatch at index " << i << ": got " << gpu_dec3[i]
                << ", expected " << expected_dec3[i] << std::endl;
      return 1;
    }
    if (!approx_equal(gpu_dec4[i], expected_dec4[i])) {
      std::cerr << "dec4 mismatch at index " << i << ": got " << gpu_dec4[i]
                << ", expected " << expected_dec4[i] << std::endl;
      return 1;
    }
  }

  cudaFree(device_dec4);
  cudaFree(device_dec3);
  cudaFree(device_dec2);
  cudaFree(device_magnitudes);

  std::cout << "Runs" << std::endl;
  return 0;
}
