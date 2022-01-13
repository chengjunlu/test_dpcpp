#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>

using namespace sycl;


static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

void print_mem_stats(std::vector<device>& devices) {

  for (const auto& device : devices) {
    auto mem_size = device.get_info<info::device::global_mem_size>();
    std::cout << "Device: " << device.get_info<info::device::name>()
              << " Mem size:" << format_size(mem_size) << std::endl;
  }

  std::cout << std::endl;

}

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

  std::cout << "root device mem size" << std::endl;

  print_mem_stats(root_devices);

  std::vector<device> tile_devices;
  // Mapping framework device to physical tile by default.
  // If IPEX_DISABLE_TILE_PARTITION enabled, mapping framework device to
  // physical device.
  constexpr info::partition_property partition_by_affinity =
          info::partition_property::partition_by_affinity_domain;
  constexpr info::partition_affinity_domain next_partitionable =
          info::partition_affinity_domain::next_partitionable;
  for (const auto& root_device : root_devices) {
    std::vector<device> sub_devices;
    try {
      sub_devices = root_device.create_sub_devices<partition_by_affinity>(
              next_partitionable);
      tile_devices.insert(
              tile_devices.end(), sub_devices.begin(), sub_devices.end());
    } catch (sycl::exception& e) {
      // FIXME: should only check feature_not_supported here.
      // But for now we got invalid here if partition is not supported.
      if (e.code() != errc::feature_not_supported &&
          e.code() != errc::invalid) {
        throw std::runtime_error(
                std::string("Failed to apply tile partition: ") + e.what());
      }
      std::cout<< "WARNING: Tile partition is UNSUPPORTED : " << root_device.get_info<info::device::name>() << std::endl;
    }
  }

  std::cout << "tile device mem size" << std::endl;

  print_mem_stats(tile_devices);
}