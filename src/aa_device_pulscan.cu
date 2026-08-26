#include "aa_params.hpp"

#if AA_ENABLE_PULSCAN

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include "aa_device_pulscan.hpp"
#include "cdflib.hpp"

namespace astroaccelerate {

  struct pulscan_power_index {
    float power;
    int index;
  };

  static double pulscan_extended_equiv_gaussian_sigma_host(double logp) {
    double t = std::sqrt(-2.0 * logp);
    double num = 2.515517 + t * (0.802853 + t * 0.010328);
    double denom = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
    return t - num / denom;
  }

  static double pulscan_equivalent_gaussian_sigma_host(double logp) {
    double x;

    if (logp < -600.0) {
      x = pulscan_extended_equiv_gaussian_sigma_host(logp);
    } else {
      int which = 2;
      int status = 0;
      double p;
      double q;
      double bound = 0.0;
      double mean = 0.0;
      double sd = 1.0;

      q = std::exp(logp);
      p = 1.0 - q;
      cdfnor(&which, &p, &q, &x, &mean, &sd, &status, &bound);
      if (status != 0) {
        if (status == -2) {
          x = 0.0;
        } else if (status == -3) {
          x = 38.5;
        } else {
          x = 0.0;
        }
      }
    }

    if (x < 0.0) {
      return 0.0;
    }
    return x;
  }

  float pulscan_logp_to_sigma(float logp) {
    return static_cast<float>(pulscan_equivalent_gaussian_sigma_host(static_cast<double>(logp)));
  }

  __device__ double pulscan_power_to_logp(float chi2, float dof) {
    double double_dof = static_cast<double>(dof);
    double double_chi2 = static_cast<double>(chi2);

    if (dof >= chi2 * 1.05f) {
      return 0.0;
    }

    double x = 1500.0 * double_dof / double_chi2;
    double f_x = (-4.460405902717228e-46 * pow(x, 16) + 9.492786384945832e-42 * pow(x, 15) -
                  9.147045144529116e-38 * pow(x, 14) + 5.281085384219971e-34 * pow(x, 13) -
                  2.0376166670276118e-30 * pow(x, 12) + 5.548033164083744e-27 * pow(x, 11) -
                  1.0973877021703706e-23 * pow(x, 10) + 1.5991806841151474e-20 * pow(x, 9) -
                  1.7231488066853853e-17 * pow(x, 8) + 1.3660070957914896e-14 * pow(x, 7) -
                  7.861795249869729e-12 * pow(x, 6) + 3.2136336591718867e-09 * pow(x, 5) -
                  9.046641813341226e-07 * pow(x, 4) + 1.6945948004599545e-04 * pow(x, 3) -
                  2.14942314851717e-02 * pow(x, 2) + 2.951595476316614 * x -
                  7.55240918031251e+02);
    double logp = chi2 * f_x / 1500.0;
    return logp;
  }

  __global__ void pulscan_separate_components_kernel(const float2 *raw_data_device, float *real_data, float *imaginary_data, long num_complex_floats) {
    long global_thread_index = static_cast<long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (global_thread_index < num_complex_floats) {
      float2 current_value = raw_data_device[global_thread_index];
      real_data[global_thread_index] = current_value.x;
      imaginary_data[global_thread_index] = current_value.y;
    }
  }

  __global__ void pulscan_median_normalisation_kernel(float *global_array) {
    __shared__ float median_array[4096];
    __shared__ float mad_array[4096];
    __shared__ float normalised_array[4096];

    int local_thread_index = threadIdx.x;
    int global_array_index = blockDim.x * blockIdx.x * 4 + threadIdx.x;

    median_array[local_thread_index] = global_array[global_array_index];
    median_array[local_thread_index + 1024] = global_array[global_array_index + 1024];
    median_array[local_thread_index + 2048] = global_array[global_array_index + 2048];
    median_array[local_thread_index + 3072] = global_array[global_array_index + 3072];

    mad_array[local_thread_index] = median_array[local_thread_index];
    mad_array[local_thread_index + 1024] = median_array[local_thread_index + 1024];
    mad_array[local_thread_index + 2048] = median_array[local_thread_index + 2048];
    mad_array[local_thread_index + 3072] = median_array[local_thread_index + 3072];

    normalised_array[local_thread_index] = median_array[local_thread_index];
    normalised_array[local_thread_index + 1024] = median_array[local_thread_index + 1024];
    normalised_array[local_thread_index + 2048] = median_array[local_thread_index + 2048];
    normalised_array[local_thread_index + 3072] = median_array[local_thread_index + 3072];

    __syncthreads();

    float a, b, c, d, min_value, max_value;

    for (int upper_thread_index = 1024; upper_thread_index > 0; upper_thread_index >>= 2) {
      if (local_thread_index < upper_thread_index) {
        a = median_array[local_thread_index];
        b = median_array[local_thread_index + upper_thread_index];
        c = median_array[local_thread_index + upper_thread_index * 2];
        d = median_array[local_thread_index + upper_thread_index * 3];
        min_value = fminf(fminf(fminf(a, b), c), d);
        max_value = fmaxf(fmaxf(fmaxf(a, b), c), d);
        median_array[local_thread_index] = (a + b + c + d - min_value - max_value) * 0.5f;
      }
      __syncthreads();
    }

    float median = median_array[0];
    __syncthreads();

    mad_array[local_thread_index] = fabsf(mad_array[local_thread_index] - median);
    mad_array[local_thread_index + 1024] = fabsf(mad_array[local_thread_index + 1024] - median);
    mad_array[local_thread_index + 2048] = fabsf(mad_array[local_thread_index + 2048] - median);
    mad_array[local_thread_index + 3072] = fabsf(mad_array[local_thread_index + 3072] - median);

    __syncthreads();

    for (int upper_thread_index = 1024; upper_thread_index > 0; upper_thread_index >>= 2) {
      if (local_thread_index < upper_thread_index) {
        a = mad_array[local_thread_index];
        b = mad_array[local_thread_index + upper_thread_index];
        c = mad_array[local_thread_index + upper_thread_index * 2];
        d = mad_array[local_thread_index + upper_thread_index * 3];
        min_value = fminf(fminf(fminf(a, b), c), d);
        max_value = fmaxf(fmaxf(fmaxf(a, b), c), d);
        mad_array[local_thread_index] = (a + b + c + d - min_value - max_value) * 0.5f;
      }
      __syncthreads();
    }

    float mad = mad_array[0] * 1.4826f;
    __syncthreads();

    normalised_array[local_thread_index] = (normalised_array[local_thread_index] - median) / mad;
    normalised_array[local_thread_index + 1024] = (normalised_array[local_thread_index + 1024] - median) / mad;
    normalised_array[local_thread_index + 2048] = (normalised_array[local_thread_index + 2048] - median) / mad;
    normalised_array[local_thread_index + 3072] = (normalised_array[local_thread_index + 3072] - median) / mad;

    __syncthreads();

    global_array[global_array_index] = normalised_array[local_thread_index];
    global_array[global_array_index + 1024] = normalised_array[local_thread_index + 1024];
    global_array[global_array_index + 2048] = normalised_array[local_thread_index + 2048];
    global_array[global_array_index + 3072] = normalised_array[local_thread_index + 3072];
  }

  __global__ void pulscan_magnitude_squared_kernel(const float *real_data, const float *imaginary_data, float *magnitude_squared_array, long num_floats) {
    long global_thread_index = static_cast<long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (global_thread_index < num_floats) {
      float real = real_data[global_thread_index];
      float imaginary = imaginary_data[global_thread_index];
      magnitude_squared_array[global_thread_index] = real * real + imaginary * imaginary;
    }
  }

  __global__ void pulscan_decimate_harmonics_kernel(const float *magnitude_squared_array, float *decimated_array2, float *decimated_array3, float *decimated_array4, long num_magnitudes) {
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    float fundamental;
    float harmonic1a, harmonic1b;
    float harmonic2a, harmonic2b, harmonic2c;
    float harmonic3a, harmonic3b, harmonic3c, harmonic3d;

    if (global_thread_index * 2 + 1 < num_magnitudes) {
      fundamental = magnitude_squared_array[global_thread_index];
      harmonic1a = magnitude_squared_array[global_thread_index * 2];
      harmonic1b = magnitude_squared_array[global_thread_index * 2 + 1];
      decimated_array2[global_thread_index] = fundamental + harmonic1a + harmonic1b;
    }

    if (global_thread_index * 3 + 2 < num_magnitudes) {
      harmonic2a = magnitude_squared_array[global_thread_index * 3];
      harmonic2b = magnitude_squared_array[global_thread_index * 3 + 1];
      harmonic2c = magnitude_squared_array[global_thread_index * 3 + 2];
      decimated_array3[global_thread_index] = fundamental + harmonic1a + harmonic1b +
                                              harmonic2a + harmonic2b + harmonic2c;
    }

    if (global_thread_index * 4 + 3 < num_magnitudes) {
      harmonic3a = magnitude_squared_array[global_thread_index * 4];
      harmonic3b = magnitude_squared_array[global_thread_index * 4 + 1];
      harmonic3c = magnitude_squared_array[global_thread_index * 4 + 2];
      harmonic3d = magnitude_squared_array[global_thread_index * 4 + 3];
      decimated_array4[global_thread_index] = fundamental + harmonic1a + harmonic1b +
                                              harmonic2a + harmonic2b + harmonic2c +
                                              harmonic3a + harmonic3b + harmonic3c + harmonic3d;
    }
  }

  __device__ void pulscan_search_and_update(float *sum_array, pulscan_power_index *search_array, pulscan_candidate *local_candidate_array,
                                            int z, int output_counter, int local_thread_index, int global_thread_index, int numharm) {
    search_array[local_thread_index].power = sum_array[local_thread_index];
    search_array[local_thread_index].index = global_thread_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (local_thread_index < stride) {
        if (search_array[local_thread_index].power < search_array[local_thread_index + stride].power) {
          search_array[local_thread_index] = search_array[local_thread_index + stride];
        }
      }
      __syncthreads();
    }
    if (local_thread_index == 0) {
      local_candidate_array[output_counter].power = search_array[0].power;
      local_candidate_array[output_counter].r = search_array[0].index;
      local_candidate_array[output_counter].z = z;
      local_candidate_array[output_counter].logp = 0.0f;
      local_candidate_array[output_counter].numharm = numharm;
    }
  }

  __global__ void pulscan_boxcar_filter_kernel(const float *magnitude_squared_array, pulscan_candidate *global_candidate_array, int numharm,
                                               long num_floats, int num_candidates_per_block) {
    __shared__ float lookup_array[512];
    __shared__ float sum_array[256];
    __shared__ pulscan_power_index search_array[256];
    __shared__ pulscan_candidate local_candidate_array[16];

    long global_thread_index = static_cast<long>(blockDim.x) * blockIdx.x + threadIdx.x;
    int local_thread_index = threadIdx.x;
    bool valid_start = global_thread_index < num_floats;
    bool valid_offset = (global_thread_index + 256) < num_floats;

    lookup_array[local_thread_index] =
      (global_thread_index < num_floats)
        ? magnitude_squared_array[global_thread_index]
        : 0.0f;
    lookup_array[local_thread_index + 256] =
      valid_offset ? magnitude_squared_array[global_thread_index + 256] : 0.0f;

    __syncthreads();

    sum_array[local_thread_index] = valid_start ? 0.0f : -1.0e30f;
    __syncthreads();

    sum_array[local_thread_index] += lookup_array[local_thread_index + 0];
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 0, 0, local_thread_index, global_thread_index, numharm);

    sum_array[local_thread_index] += lookup_array[local_thread_index + 1];
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 1, 1, local_thread_index, global_thread_index, numharm);

    sum_array[local_thread_index] += lookup_array[local_thread_index + 2];
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 2, 2, local_thread_index, global_thread_index, numharm);

    sum_array[local_thread_index] += lookup_array[local_thread_index + 3];
    sum_array[local_thread_index] += lookup_array[local_thread_index + 4];
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 4, 3, local_thread_index, global_thread_index, numharm);

    sum_array[local_thread_index] += lookup_array[local_thread_index + 5];
    sum_array[local_thread_index] += lookup_array[local_thread_index + 6];
    sum_array[local_thread_index] += lookup_array[local_thread_index + 7];
    sum_array[local_thread_index] += lookup_array[local_thread_index + 8];
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 8, 4, local_thread_index, global_thread_index, numharm);

    #pragma unroll
    for (int z = 9; z < 17; z++) {
      sum_array[local_thread_index] += lookup_array[local_thread_index + z];
    }
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 16, 5, local_thread_index, global_thread_index, numharm);

    #pragma unroll
    for (int z = 17; z < 33; z++) {
      sum_array[local_thread_index] += lookup_array[local_thread_index + z];
    }
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 32, 6, local_thread_index, global_thread_index, numharm);

    #pragma unroll
    for (int z = 33; z < 65; z++) {
      sum_array[local_thread_index] += lookup_array[local_thread_index + z];
    }
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 64, 7, local_thread_index, global_thread_index, numharm);

    #pragma unroll
    for (int z = 65; z < 129; z++) {
      sum_array[local_thread_index] += lookup_array[local_thread_index + z];
    }
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 128, 8, local_thread_index, global_thread_index, numharm);

    #pragma unroll
    for (int z = 129; z < 257; z++) {
      sum_array[local_thread_index] += lookup_array[local_thread_index + z];
    }
    __syncthreads();
    pulscan_search_and_update(sum_array, search_array, local_candidate_array, 256, 9, local_thread_index, global_thread_index, numharm);

    __syncthreads();

    if (local_thread_index < num_candidates_per_block) {
      global_candidate_array[blockIdx.x * num_candidates_per_block + local_thread_index] = local_candidate_array[local_thread_index];
    }
  }

  __global__ void pulscan_calculate_logp_kernel(pulscan_candidate *global_candidate_array, long num_candidates, int num_sum) {
    long global_thread_index = static_cast<long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (global_thread_index < num_candidates) {
      double logp = pulscan_power_to_logp(global_candidate_array[global_thread_index].power,
                                          static_cast<float>(global_candidate_array[global_thread_index].z * num_sum * 2));
      global_candidate_array[global_thread_index].logp = static_cast<float>(logp);
    }
  }

  void call_kernel_pulscan_separate_components(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                               const float2 *raw_data, float *real_out, float *imag_out, long num_complex_floats) {
    pulscan_separate_components_kernel<<<grid_size, block_size, 0, stream>>>(raw_data, real_out, imag_out, num_complex_floats);
  }

  void call_kernel_pulscan_median_normalisation(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                                float *array) {
    pulscan_median_normalisation_kernel<<<grid_size, block_size, 0, stream>>>(array);
  }

  void call_kernel_pulscan_magnitude_squared(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                             const float *real_data, const float *imag_data, float *power_out, long num_floats) {
    pulscan_magnitude_squared_kernel<<<grid_size, block_size, 0, stream>>>(real_data, imag_data, power_out, num_floats);
  }

  void call_kernel_pulscan_decimate_harmonics(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                              const float *power_in, float *harmonic2_out, float *harmonic3_out, float *harmonic4_out,
                                              long num_magnitudes) {
    pulscan_decimate_harmonics_kernel<<<grid_size, block_size, 0, stream>>>(power_in, harmonic2_out, harmonic3_out, harmonic4_out, num_magnitudes);
  }

  void call_kernel_pulscan_boxcar_filter(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                         const float *power_in, pulscan_candidate *candidate_out, int num_harmonics,
                                         long num_floats, int candidates_per_block) {
    pulscan_boxcar_filter_kernel<<<grid_size, block_size, 0, stream>>>(power_in, candidate_out, num_harmonics, num_floats, candidates_per_block);
  }

  void call_kernel_pulscan_calculate_logp(const dim3 &grid_size, const dim3 &block_size, const cudaStream_t &stream,
                                          pulscan_candidate *candidate_array, long num_candidates, int num_sum) {
    pulscan_calculate_logp_kernel<<<grid_size, block_size, 0, stream>>>(candidate_array, num_candidates, num_sum);
}

} // namespace astroaccelerate

#endif // AA_ENABLE_PULSCAN
