#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "aa_device_pulscan.hpp"

using namespace astroaccelerate;

namespace {

constexpr float kTolerance = 1e-6f;

bool approx_equal(float a, float b) {
  return std::fabs(a - b) <= kTolerance;
}

void check_cuda(cudaError_t status, const char *message) {
  if (status != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(1);
  }
}

} // namespace

int main() {
  std::cout << "Running test_pulscan_magnitude_squared.cpp" << std::endl;

  constexpr long num_samples = 16;

  std::vector<float> host_real(num_samples);
  std::vector<float> host_imag(num_samples);
  std::vector<float> expected(num_samples);
  for (long i = 0; i < num_samples; ++i) {
    host_real[i] = static_cast<float>(i - 4);
    host_imag[i] = static_cast<float>((i % 3) - 1);
    expected[i] = host_real[i] * host_real[i] + host_imag[i] * host_imag[i];
  }

  float *device_real = nullptr;
  float *device_imag = nullptr;
  float *device_power = nullptr;

  check_cuda(cudaMalloc(&device_real, num_samples * sizeof(float)), "cudaMalloc device_real failed");
  check_cuda(cudaMalloc(&device_imag, num_samples * sizeof(float)), "cudaMalloc device_imag failed");
  check_cuda(cudaMalloc(&device_power, num_samples * sizeof(float)), "cudaMalloc device_power failed");

  check_cuda(cudaMemcpy(device_real, host_real.data(), num_samples * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy host_real -> device_real failed");
  check_cuda(cudaMemcpy(device_imag, host_imag.data(), num_samples * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy host_imag -> device_imag failed");

  dim3 block_size(16);
  dim3 grid_size(1);
  call_kernel_pulscan_magnitude_squared(grid_size, block_size, 0, device_real, device_imag, device_power, num_samples);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after magnitude_squared failed");
  check_cuda(cudaGetLastError(), "Kernel launch reported error");

  std::vector<float> host_power(num_samples, 0.0f);
  check_cuda(cudaMemcpy(host_power.data(), device_power, num_samples * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy device_power -> host_power failed");

  for (long i = 0; i < num_samples; ++i) {
    if (!approx_equal(host_power[i], expected[i])) {
      std::cerr << "Mismatch at index " << i << ": got " << host_power[i] << ", expected " << expected[i]
                << std::endl;
      return 1;
    }
  }

  cudaFree(device_power);
  cudaFree(device_imag);
  cudaFree(device_real);

  std::cout << "Runs" << std::endl;
  return 0;
}
