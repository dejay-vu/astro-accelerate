#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

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

void cpu_median_normalisation(std::vector<float> &values) {
  constexpr int block_width = 1024;
  constexpr int per_block = block_width * 4;

  std::vector<float> median_array(values.begin(), values.end());
  std::vector<float> mad_array(values.begin(), values.end());
  std::vector<float> normalised_array(values.begin(), values.end());

  for (int upper = block_width; upper > 0; upper >>= 2) {
    for (int idx = 0; idx < upper; ++idx) {
      float a = median_array[idx];
      float b = median_array[idx + upper];
      float c = median_array[idx + upper * 2];
      float d = median_array[idx + upper * 3];
      float min_value = std::min(std::min(a, b), std::min(c, d));
      float max_value = std::max(std::max(a, b), std::max(c, d));
      median_array[idx] = (a + b + c + d - min_value - max_value) * 0.5f;
    }
  }

  float median = median_array[0];

  for (int i = 0; i < per_block; ++i) {
    mad_array[i] = std::fabs(mad_array[i] - median);
  }

  for (int upper = block_width; upper > 0; upper >>= 2) {
    for (int idx = 0; idx < upper; ++idx) {
      float a = mad_array[idx];
      float b = mad_array[idx + upper];
      float c = mad_array[idx + upper * 2];
      float d = mad_array[idx + upper * 3];
      float min_value = std::min(std::min(a, b), std::min(c, d));
      float max_value = std::max(std::max(a, b), std::max(c, d));
      mad_array[idx] = (a + b + c + d - min_value - max_value) * 0.5f;
    }
  }

  float mad = mad_array[0] * 1.4826f;

  for (int i = 0; i < per_block; ++i) {
    normalised_array[i] = (normalised_array[i] - median) / mad;
  }

  values.assign(normalised_array.begin(), normalised_array.end());
}

} // namespace

int main() {
  std::cout << "Running test_pulscan_median_normalisation.cpp" << std::endl;

  constexpr long block_elements = 4096;
  std::vector<float> host_values(block_elements);

  for (long i = 0; i < block_elements; ++i) {
    host_values[i] = std::sin(static_cast<float>(i) * 0.17f) * 3.0f + static_cast<float>((i % 13) - 6);
  }

  std::vector<float> expected(host_values.begin(), host_values.end());
  cpu_median_normalisation(expected);

  float *device_values = nullptr;
  check_cuda(cudaMalloc(&device_values, block_elements * sizeof(float)), "cudaMalloc device_values failed");
  check_cuda(cudaMemcpy(device_values, host_values.data(), block_elements * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy host_values -> device_values failed");

  dim3 block_size(1024);
  dim3 grid_size(1);
  call_kernel_pulscan_median_normalisation(grid_size, block_size, 0, device_values);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after median_normalisation failed");
  check_cuda(cudaGetLastError(), "median_normalisation kernel launch error");

  std::vector<float> device_result(block_elements, 0.0f);
  check_cuda(cudaMemcpy(device_result.data(), device_values, block_elements * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy device_values -> host failed");

  for (long i = 0; i < block_elements; ++i) {
    if (!approx_equal(device_result[i], expected[i])) {
      std::cerr << "Mismatch at index " << i << ": got " << device_result[i]
                << ", expected " << expected[i] << std::endl;
      return 1;
    }
  }

  cudaFree(device_values);

  std::cout << "Runs" << std::endl;
  return 0;
}
