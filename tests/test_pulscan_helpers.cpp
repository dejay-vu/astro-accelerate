#include <cmath>
#include <iostream>

#ifndef AA_WITH_PULSCAN
#define AA_WITH_PULSCAN 1
#endif
#include "aa_device_pulscan.hpp"

using namespace astroaccelerate;

namespace {

bool approx_equal(float a, float b, float tolerance) {
  return std::fabs(a - b) <= tolerance;
}

} // namespace

int main() {
  std::cout << "Running test_pulscan_helpers.cpp" << std::endl;

  const float sigma_zero = pulscan_logp_to_sigma(0.0f);
  if (!approx_equal(sigma_zero, 0.0f, 1e-6f)) {
    std::cout << "sigma_zero unexpected: " << sigma_zero << std::endl;
    return 1;
  }

  const float sigma_neg1 = pulscan_logp_to_sigma(-1.0f);
  if (!(sigma_neg1 > 0.0f && sigma_neg1 < 1.0f)) {
    std::cout << "sigma_neg1 out of expected range: " << sigma_neg1 << std::endl;
    return 1;
  }

  const float sigma_neg10 = pulscan_logp_to_sigma(-10.0f);
  if (!(sigma_neg10 > 3.0f && sigma_neg10 < 5.0f)) {
    std::cout << "sigma_neg10 out of expected range: " << sigma_neg10 << std::endl;
    return 1;
  }

  const float sigma_neg1000 = pulscan_logp_to_sigma(-1000.0f);
  if (!(sigma_neg1000 > sigma_neg10 && sigma_neg1000 < 50.0f)) {
    std::cout << "sigma_neg1000 out of expected range: " << sigma_neg1000 << std::endl;
    return 1;
  }

  if (!(sigma_neg10 > sigma_neg1)) {
    std::cout << "Monotonicity check failed: sigma_neg10 <= sigma_neg1" << std::endl;
    return 1;
  }

  if (!std::isfinite(sigma_neg1000)) {
    std::cout << "sigma_neg1000 not finite" << std::endl;
    return 1;
  }

  std::cout << "Runs" << std::endl;
  return 0;
}
