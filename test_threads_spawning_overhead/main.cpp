#include "ThreadsSpawnTest.hpp"
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>

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
  std::cout << "run test on device:" << root_devices[0].get_info<sycl::info::device::name>() << std::endl;
  std::cout << "      slice number:" << root_devices[0].get_info<sycl::ext::intel::info::device::gpu_slices>() << std::endl;
  std::cout << "   subslice number:" << root_devices[0].get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>() << std::endl;
  std::cout << "         eu number:" << root_devices[0].get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>() << std::endl;
  std::cout << "physthreads number:" << root_devices[0].get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() << std::endl;
  std::cout << " simd width number:" << root_devices[0].get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>() << std::endl;
  sycl::queue queue = sycl::queue(root_devices[0], {property::queue::in_order(),
           property::queue::enable_profiling()});

  test_subgroup_ballot(queue);
}