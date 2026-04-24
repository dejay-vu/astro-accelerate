#ifndef ASTRO_ACCELERATE_AA_DEVICE_PULSCAN_HPP
#define ASTRO_ACCELERATE_AA_DEVICE_PULSCAN_HPP

#include "aa_params.hpp"

#if AA_ENABLE_PULSCAN

#include <cuda_runtime.h>

namespace astroaccelerate {

  struct pulscan_candidate {
    float power;
    float logp;
    int r;
    int z;
    int numharm;
  };

  struct pulscan_host_candidate {
    double dm;
    double frequency_hz;
    double period_s;
    float power;
    float logp;
    float sigma;
    int r;
    int z;
    int numharm;
    int range_index;
    int dm_index;
    int bin_index;
  };

  void call_kernel_pulscan_separate_components(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                               const float2 *raw_data, float *real_out, float *imag_out, long num_complex_floats);

  void call_kernel_pulscan_median_normalisation(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                                float *array);

  void call_kernel_pulscan_magnitude_squared(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                             const float *real_data, const float *imag_data, float *power_out, long num_floats);

  void call_kernel_pulscan_decimate_harmonics(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                              const float *power_in, float *harmonic2_out, float *harmonic3_out, float *harmonic4_out,
                                              long num_magnitudes);

  void call_kernel_pulscan_boxcar_filter(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                         const float *power_in, pulscan_candidate *candidate_out, int num_harmonics,
                                         long num_floats, int candidates_per_block);

  void call_kernel_pulscan_calculate_logp(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                          pulscan_candidate *candidate_array, long num_candidates, int num_sum);

  float pulscan_logp_to_sigma(float logp);

} // namespace astroaccelerate

#endif // AA_ENABLE_PULSCAN

#endif // ASTRO_ACCELERATE_AA_DEVICE_PULSCAN_HPP
