#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include "SlowKernel.hpp"

using namespace sycl;

int main() {

  auto plaform_list = platform::get_platforms();
  std::vector<device> root_devices;
  // Enumerated root devices(GPU cards) from GPU Platform firstly.
  for (const auto& platform : plaform_list) {
    if (platform.get_backend() != backend::ext_oneapi_level_zero)
      continue;
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }

  std::cout << "root device count" << root_devices.size() << std::endl;
  sycl::queue queue = sycl::queue(root_devices[0], {property::queue::in_order(),
           property::queue::enable_profiling()});

  test_slow_kernel(queue, 8096*16);

  std::cout << "done" << std::endl;
}