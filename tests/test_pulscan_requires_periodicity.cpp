#include <iostream>
#include <vector>


#include "aa_ddtr_plan.hpp"
#include "aa_ddtr_strategy.hpp"
#include "aa_device_info.hpp"
#include "aa_filterbank_metadata.hpp"
#include "aa_pipeline_api.hpp"

using namespace astroaccelerate;

int main() {
  std::cout << "Running test_pulscan_requires_periodicity.cpp" << std::endl;

  aa_ddtr_plan ddtr_plan;
  ddtr_plan.add_dm(0.0f, 20.0f, 1.0f, 1, 1);

  const double tstart = 58000.0;
  const double tsamp = 6.4e-5;
  const unsigned int nsamples = 4096;
  const int nchans = 64;
  const double fch1 = 1400.0;
  const double bandwidth = 200.0;
  const double foff = -bandwidth / nchans;
  const double nbits = 8;

  aa_filterbank_metadata metadata(tstart, tsamp, nbits, nsamples, fch1, foff, nchans);
  const std::size_t total_samples = static_cast<std::size_t>(metadata.nsamples());
  const std::size_t total_channels = static_cast<std::size_t>(metadata.nchans());

  std::vector<unsigned short> input(total_samples * total_channels, 0);
  for(std::size_t chan = 0; chan < total_channels; ++chan) {
    for(std::size_t sample = 0; sample < total_samples; ++sample) {
      input[chan * total_samples + sample] =
          static_cast<unsigned short>((sample + chan) % 255);
    }
  }

  aa_device_info device(0);
  aa_pipeline::pipeline components;
  components.insert(aa_pipeline::component::dedispersion);

  aa_pipeline::pipeline_option options;
  options.insert(aa_pipeline::component_option::copy_ddtr_data_to_host);

  aa_pipeline_api<unsigned short> runner(components, options, metadata, input.data(), device);
  runner.bind(ddtr_plan);

  if(!runner.ready()) {
    std::cerr << "Pipeline runner not ready" << std::endl;
    return 1;
  }

  aa_pipeline_runner::status status_code;
  while(runner.run(status_code)) {
    if(status_code == aa_pipeline_runner::status::error) {
      std::cerr << "Pipeline reported error status" << std::endl;
      return 1;
    }
  }

  const auto& candidates = runner.get_pulscan_candidates();
  if(!candidates.empty()) {
    std::cerr << "Pulscan ran without periodicity and returned "
              << candidates.size() << " candidates" << std::endl;
    return 1;
  }

  std::cout << "Runs" << std::endl;
  return 0;
}