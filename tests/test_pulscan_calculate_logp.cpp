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

constexpr float kTolerance = 1e-4f;

bool approx_equal(float a, float b) {
  return std::fabs(a - b) <= kTolerance;
}

void check_cuda(cudaError_t status, const char *message) {
  if (status != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(1);
  }
}

double cpu_power_to_logp(float chi2, float dof) {
  double double_dof = static_cast<double>(dof);
  double double_chi2 = static_cast<double>(chi2);

  if (dof >= chi2 * 1.05f) {
    return 0.0;
  }

  double x = 1500.0 * double_dof / double_chi2;
  double fx = (-4.460405902717228e-46 * std::pow(x, 16) + 9.492786384945832e-42 * std::pow(x, 15) -
               9.147045144529116e-38 * std::pow(x, 14) + 5.281085384219971e-34 * std::pow(x, 13) -
               2.0376166670276118e-30 * std::pow(x, 12) + 5.548033164083744e-27 * std::pow(x, 11) -
               1.0973877021703706e-23 * std::pow(x, 10) + 1.5991806841151474e-20 * std::pow(x, 9) -
               1.7231488066853853e-17 * std::pow(x, 8) + 1.3660070957914896e-14 * std::pow(x, 7) -
               7.861795249869729e-12 * std::pow(x, 6) + 3.2136336591718867e-09 * std::pow(x, 5) -
               9.046641813341226e-07 * std::pow(x, 4) + 1.6945948004599545e-04 * std::pow(x, 3) -
               2.14942314851717e-02 * std::pow(x, 2) + 2.951595476316614 * x -
               7.55240918031251e+02);

  double logp = chi2 * fx / 1500.0;
  return logp;
}

} // namespace

int main() {
  std::cout << "Running test_pulscan_calculate_logp.cpp" << std::endl;

  constexpr long num_candidates = 32;
  constexpr int num_sum = 4;

  std::vector<pulscan_candidate> host_candidates(num_candidates);
  std::vector<float> expected_logp(num_candidates, 0.0f);

  for (long i = 0; i < num_candidates; ++i) {
    int z = static_cast<int>((i % 8) + 1);
    float dof = static_cast<float>(z * num_sum * 2);
    float power = dof * (1.4f + static_cast<float>(i) * 0.05f);

    if ((i % 5) == 0) {
      power = dof * 0.8f;
    }

    host_candidates[i].power = power;
    host_candidates[i].logp = -123.0f;
    host_candidates[i].r = static_cast<int>(i * 3);
    host_candidates[i].z = z;
    host_candidates[i].numharm = 1;

    expected_logp[i] = static_cast<float>(cpu_power_to_logp(power, dof));
  }

  pulscan_candidate *device_candidates = nullptr;
  check_cuda(cudaMalloc(&device_candidates, num_candidates * sizeof(pulscan_candidate)), "cudaMalloc candidates failed");
  check_cuda(cudaMemcpy(device_candidates, host_candidates.data(), num_candidates * sizeof(pulscan_candidate), cudaMemcpyHostToDevice),
             "cudaMemcpy host -> device candidates failed");

  dim3 block_size(32);
  dim3 grid_size(1);
  call_kernel_pulscan_calculate_logp(grid_size, block_size, 0, device_candidates, num_candidates, num_sum);

  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after calculate_logp failed");
  check_cuda(cudaGetLastError(), "calculate_logp kernel launch error");

  std::vector<pulscan_candidate> gpu_candidates(num_candidates);
  check_cuda(cudaMemcpy(gpu_candidates.data(), device_candidates, num_candidates * sizeof(pulscan_candidate), cudaMemcpyDeviceToHost),
             "cudaMemcpy device -> host candidates failed");

  for (long i = 0; i < num_candidates; ++i) {
    if (!approx_equal(gpu_candidates[i].logp, expected_logp[i])) {
      std::cerr << "logp mismatch at index " << i << ": got " << gpu_candidates[i].logp
                << ", expected " << expected_logp[i] << std::endl;
      return 1;
    }
  }

  cudaFree(device_candidates);

  std::cout << "Runs" << std::endl;
  return 0;
}
