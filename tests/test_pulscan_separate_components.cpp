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
  std::cout << "Running test_pulscan_separate_components.cpp" << std::endl;

  constexpr long num_complex = 12;
  std::vector<float2> host_in(num_complex);
  std::vector<float> expected_real(num_complex);
  std::vector<float> expected_imag(num_complex);

  for (long i = 0; i < num_complex; ++i) {
    float real_value = static_cast<float>(i * 0.5f - 1.0f);
    float imag_value = static_cast<float>((num_complex - i) * 0.25f);
    host_in[i] = make_float2(real_value, imag_value);
    expected_real[i] = real_value;
    expected_imag[i] = imag_value;
  }

  float2 *device_in = nullptr;
  float *device_real = nullptr;
  float *device_imag = nullptr;

  check_cuda(cudaMalloc(&device_in, num_complex * sizeof(float2)), "cudaMalloc device_in failed");
  check_cuda(cudaMalloc(&device_real, num_complex * sizeof(float)), "cudaMalloc device_real failed");
  check_cuda(cudaMalloc(&device_imag, num_complex * sizeof(float)), "cudaMalloc device_imag failed");

  check_cuda(cudaMemcpy(device_in, host_in.data(), num_complex * sizeof(float2), cudaMemcpyHostToDevice),
             "cudaMemcpy host_in -> device_in failed");

  dim3 block_size(32);
  dim3 grid_size(1);
  call_kernel_pulscan_separate_components(grid_size, block_size, 0, device_in, device_real, device_imag, num_complex);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after separate_components failed");
  check_cuda(cudaGetLastError(), "separate_components kernel launch error");

  std::vector<float> host_real(num_complex, 0.0f);
  std::vector<float> host_imag(num_complex, 0.0f);

  check_cuda(cudaMemcpy(host_real.data(), device_real, num_complex * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy device_real -> host_real failed");
  check_cuda(cudaMemcpy(host_imag.data(), device_imag, num_complex * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy device_imag -> host_imag failed");

  for (long i = 0; i < num_complex; ++i) {
    if (!approx_equal(host_real[i], expected_real[i])) {
      std::cerr << "Real mismatch at index " << i << ": got " << host_real[i]
                << ", expected " << expected_real[i] << std::endl;
      return 1;
    }
    if (!approx_equal(host_imag[i], expected_imag[i])) {
      std::cerr << "Imag mismatch at index " << i << ": got " << host_imag[i]
                << ", expected " << expected_imag[i] << std::endl;
      return 1;
    }
  }

  cudaFree(device_imag);
  cudaFree(device_real);
  cudaFree(device_in);

  std::cout << "Runs" << std::endl;
  return 0;
}
