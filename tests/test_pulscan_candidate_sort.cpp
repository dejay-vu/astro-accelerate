#include <algorithm>
#include <iostream>
#include <vector>

#ifndef AA_WITH_PULSCAN
#define AA_WITH_PULSCAN 1
#endif
#include "aa_device_pulscan.hpp"

using namespace astroaccelerate;

int main() {
  std::cout << "Running test_pulscan_candidate_sort.cpp" << std::endl;

  const std::size_t total_candidates = 4105;
  std::vector<pulscan_host_candidate> candidates;
  candidates.reserve(total_candidates);

  for (std::size_t i = 0; i < total_candidates; ++i) {
    pulscan_host_candidate candidate{};
    candidate.logp = -0.01f * static_cast<float>(i % 500);
    candidate.sigma = pulscan_logp_to_sigma(candidate.logp);
    candidate.power = static_cast<float>(i);
    candidate.r = static_cast<int>(i % 256);
    candidate.z = static_cast<int>((i % 10) + 1);
    candidate.numharm = 1;
    candidate.dm = 100.0 + static_cast<double>(i) * 0.5;
    candidate.frequency_hz = 100.0 + static_cast<double>(i);
    candidate.period_s = 1.0 / candidate.frequency_hz;
    candidate.range_index = static_cast<int>(i % 4);
    candidate.dm_index = static_cast<int>(i % 16);
    candidate.bin_index = candidate.r;
    candidates.push_back(candidate);
  }

  std::vector<pulscan_host_candidate> unsorted = candidates;

  std::sort(candidates.begin(), candidates.end(), [](const pulscan_host_candidate &a, const pulscan_host_candidate &b) {
    return a.sigma > b.sigma;
  });

  const std::size_t max_pulscan_candidates = 4096;
  if (candidates.size() > max_pulscan_candidates) {
    candidates.resize(max_pulscan_candidates);
  }

  if (candidates.size() != max_pulscan_candidates) {
    std::cout << "Unexpected size after trim: " << candidates.size() << std::endl;
    return 1;
  }

  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i - 1].sigma < candidates[i].sigma) {
      std::cout << "Sigma order violated at index " << i << std::endl;
      return 1;
    }
  }

  for (const auto &candidate : candidates) {
    auto it = std::find_if(unsorted.begin(), unsorted.end(), [&](const pulscan_host_candidate &original) {
      return original.power == candidate.power;
    });
    if (it == unsorted.end()) {
      std::cout << "Candidate missing after sorting" << std::endl;
      return 1;
    }
    if (candidate.sigma != it->sigma) {
      std::cout << "Sigma mismatch after sorting" << std::endl;
      return 1;
    }
  }

  std::cout << "Runs" << std::endl;
  return 0;
}
