#include <algorithm>
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

struct cpu_candidate {
  float power;
  int r;
  int z;
  int numharm;
};

std::vector<cpu_candidate> cpu_boxcar_filter(const std::vector<float> &power, int numharm, long num_floats, int candidates_per_block) {
  const int block_size = 256;
  const int window = 256; // lookup_array covers 512 entries, but sums expand progressively
  int num_blocks = static_cast<int>(num_floats) / block_size;
  std::vector<cpu_candidate> all_candidates;
  all_candidates.reserve(num_blocks * candidates_per_block);

  std::vector<float> lookup(512, 0.0f);
  std::vector<float> sums(block_size, 0.0f);

  auto emit_candidate = [&](int block_index, float power_value, int r_index, int z_value) {
    cpu_candidate candidate{};
    candidate.power = power_value;
    candidate.r = r_index;
    candidate.z = z_value;
    candidate.numharm = numharm;
    all_candidates.push_back(candidate);
  };

  for (int block = 0; block < num_blocks; ++block) {
    long base_index = static_cast<long>(block) * block_size;
    for (int i = 0; i < block_size; ++i) {
      lookup[i] = power[base_index + i];
      long offset = base_index + i + 256;
      if (offset < num_floats) {
        lookup[i + 256] = power[offset];
      } else {
        lookup[i + 256] = 0.0f;
      }
      sums[i] = 0.0f;
    }

    auto emit_top = [&](int z_value, int output_counter) {
      std::vector<std::pair<float, int>> ranking;
      ranking.reserve(block_size);
      for (int i = 0; i < block_size; ++i) {
        ranking.emplace_back(sums[i], static_cast<int>(base_index + i));
      }
      auto best = std::max_element(ranking.begin(), ranking.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });
      emit_candidate(block, best->first, best->second, z_value);
    };

    sums.assign(block_size, 0.0f);
    for (int i = 0; i < block_size; ++i) {
      sums[i] += lookup[i + 0];
    }
    emit_top(0, 0);

    for (int i = 0; i < block_size; ++i) {
      sums[i] += lookup[i + 1];
    }
    emit_top(1, 1);

    for (int i = 0; i < block_size; ++i) {
      sums[i] += lookup[i + 2];
    }
    emit_top(2, 2);

    for (int delta = 3; delta <= 4; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(4, 3);

    for (int delta = 5; delta <= 8; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(8, 4);

    for (int delta = 9; delta <= 16; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(16, 5);

    for (int delta = 17; delta <= 32; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(32, 6);

    for (int delta = 33; delta <= 64; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(64, 7);

    for (int delta = 65; delta <= 128; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(128, 8);

    for (int delta = 129; delta <= 256; ++delta) {
      for (int i = 0; i < block_size; ++i) {
        sums[i] += lookup[i + delta];
      }
    }
    emit_top(256, 9);
  }

  return all_candidates;
}

} // namespace

int main() {
  std::cout << "Running test_pulscan_boxcar_filter.cpp" << std::endl;

  constexpr long num_floats = 512;
  constexpr int numharm = 1;
  constexpr int candidates_per_block = 10;
  constexpr int block_size = 256;

  std::vector<float> power(num_floats);
  for (long i = 0; i < num_floats; ++i) {
    power[i] = std::sin(static_cast<float>(i) * 0.07f) + std::cos(static_cast<float>(i) * 0.17f) + static_cast<float>(i) * 0.005f;
  }

  std::vector<cpu_candidate> expected = cpu_boxcar_filter(power, numharm, num_floats, candidates_per_block);

  float *device_power = nullptr;
  pulscan_candidate *device_candidates = nullptr;

  check_cuda(cudaMalloc(&device_power, num_floats * sizeof(float)), "cudaMalloc power failed");
  check_cuda(cudaMalloc(&device_candidates, (num_floats / block_size) * candidates_per_block * sizeof(pulscan_candidate)),
             "cudaMalloc candidates failed");

  check_cuda(cudaMemcpy(device_power, power.data(), num_floats * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy power failed");
  check_cuda(cudaMemset(device_candidates, 0, (num_floats / block_size) * candidates_per_block * sizeof(pulscan_candidate)),
             "cudaMemset candidates failed");

  dim3 grid_size(num_floats / block_size);
  dim3 block_dim(block_size);

  call_kernel_pulscan_boxcar_filter(grid_size, block_dim, 0, device_power, device_candidates, numharm, num_floats, candidates_per_block);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after boxcar_filter failed");
  check_cuda(cudaGetLastError(), "boxcar_filter kernel launch error");

  std::vector<pulscan_candidate> gpu_candidates(expected.size());
  check_cuda(cudaMemcpy(gpu_candidates.data(), device_candidates, gpu_candidates.size() * sizeof(pulscan_candidate), cudaMemcpyDeviceToHost),
             "cudaMemcpy candidates failed");

  if (gpu_candidates.size() != expected.size()) {
    std::cerr << "Candidate count mismatch: got " << gpu_candidates.size() << ", expected " << expected.size() << std::endl;
    return 1;
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!approx_equal(gpu_candidates[i].power, expected[i].power)) {
      std::cerr << "Power mismatch at index " << i << ": got " << gpu_candidates[i].power
                << ", expected " << expected[i].power << std::endl;
      return 1;
    }
    if (gpu_candidates[i].r != expected[i].r) {
      std::cerr << "Index mismatch at " << i << ": got " << gpu_candidates[i].r
                << ", expected " << expected[i].r << std::endl;
      return 1;
    }
    if (gpu_candidates[i].z != expected[i].z) {
      std::cerr << "Z mismatch at " << i << ": got " << gpu_candidates[i].z
                << ", expected " << expected[i].z << std::endl;
      return 1;
    }
    if (gpu_candidates[i].numharm != expected[i].numharm) {
      std::cerr << "numharm mismatch at " << i << std::endl;
      return 1;
    }
  }

  cudaFree(device_candidates);
  cudaFree(device_power);

  std::cout << "Runs" << std::endl;
  return 0;
}
