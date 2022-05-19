/*
 *
 * Copyright (C) 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <array>
#include <iostream>
#include <string.h>


#include "logging.hpp"
#include "utils.hpp"
#include "test_harness.hpp"

namespace lzt = level_zero_tests;

#include <level_zero/ze_api.h>
#include <CL/sycl.hpp>

const size_t size = 8;

int build_kernel(const std::string& build_option) {
  FILE *fp = fopen(bin_path_.c_str(), "rb");
  if (NULL == fp) {
    TORCH_WARN("CMKernel::build_kernel: ", "Open ", bin_path_, " failed\n");
    return DPCPP_FAILURE;
  }

  auto sycl_device = xpu::dpcpp::dpcppGetRawDevice(0);
  // ze_device_handle_t hDevice = sycl_device.get();

  auto ze_device = sycl_device.get_native<cl::sycl::backend::level_zero>();

  auto ze_ctx = context_.get_native<cl::sycl::backend::level_zero>();
  ze_module_handle_t ze_module;

  ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, //
                                 nullptr,
                                 ZE_MODULE_FORMAT_NATIVE, //
                                 progs_size, //
                                 progs, //
                                 "-cmc",
                                 nullptr};
  ze_module_handle_t hModule;
  L0_SAFE_CALL(
          zeModuleCreate(ze_ctx, ze_device, &moduleDesc, &hModule, nullptr));

  std::unique_ptr<cl::sycl::program> sycl_program;
  sycl_program.reset(new cl::sycl::program(
          cl::sycl::level_zero::make<cl::sycl::program>(context_, hModule)));
}

void KernelCopyTests() {
  ze_memory_type_t memory_type = ZE_MEMORY_TYPE_SHARED;
  int offset = 0;

  for (auto driver : lzt::get_all_driver_handles()) {
    for (auto device : lzt::get_devices(driver)) {
      // set up
      auto command_queue = lzt::create_command_queue();
      auto command_list = lzt::create_command_list();


      auto module = lzt::create_module(device, "/home/john/CLionProjects/test_dpcpp/test_lz_module/sycl_kernels/copy_module.spv");
      auto kernel = lzt::create_function(module, "_ZTSN2cl4sycl6detail19__pf_kernel_wrapperI9copy_dataEE");
//      auto module = lzt::create_module(device, "/home/john/CLionProjects/test_dpcpp/test_lz_module/kernels/copy_module.spv");
//      auto kernel = lzt::create_function(module, "copy_data");

      int *input_data, *output_data;
      if (memory_type == ZE_MEMORY_TYPE_HOST) {
        input_data =
            static_cast<int *>(lzt::allocate_host_memory(size * sizeof(int)));
        output_data =
            static_cast<int *>(lzt::allocate_host_memory(size * sizeof(int)));
      } else {
        input_data =
            static_cast<int *>(lzt::allocate_shared_memory(size * sizeof(int)));
        output_data =
            static_cast<int *>(lzt::allocate_shared_memory(size * sizeof(int)));
      }

      lzt::write_data_pattern(input_data, size * sizeof(int), 1);
      memset(output_data, 0, size * sizeof(int));

      std::cout << std::hex;
      std::cout << "John lu input:  ";
      for (int i = 0; i < size; i++) {
        std::cout << input_data[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "John lu output: ";
      for (int i = 0; i < size; i++) {
        std::cout << output_data[i] << " ";
      }
      std::cout << std::endl;

      lzt::set_argument_value(kernel, 0, sizeof(input_data), &input_data);
      lzt::set_argument_value(kernel, 1, sizeof(output_data), &output_data);
      lzt::set_argument_value(kernel, 2, sizeof(int), &offset);
      lzt::set_argument_value(kernel, 3, sizeof(int), &size);

      lzt::set_group_size(kernel, 1, 1, 1);

      ze_group_count_t group_count;
      group_count.groupCountX = 1;
      group_count.groupCountY = 1;
      group_count.groupCountZ = 1;

      lzt::append_launch_function(command_list, kernel, &group_count, nullptr,
                                  0, nullptr);

      lzt::close_command_list(command_list);
      lzt::execute_command_lists(command_queue, 1, &command_list, nullptr);
      lzt::synchronize(command_queue, UINT64_MAX);

      std::cout << "John lu input:  ";
      for (int i = 0; i < size; i++) {
        std::cout << input_data[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "John lu output: ";
      for (int i = 0; i < size; i++) {
        std::cout << output_data[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "L0 kernel module test end. memory compare result : " <<  memcmp(input_data, output_data + offset,
                          (size - offset) * sizeof(int)) << std::endl;

      // cleanup
      lzt::free_memory(input_data);
      lzt::free_memory(output_data);
      lzt::destroy_function(kernel);
      lzt::destroy_module(module);
      lzt::destroy_command_list(command_list);
      lzt::destroy_command_queue(command_queue);
    }
  }
}

int main() {
  ze_result_t result = zeInit(0);
  if (result) {
    throw std::runtime_error("zeInit failed: " +
                             level_zero_tests::to_string(result));
  }
  LOG_TRACE << "Driver initialized";
  level_zero_tests::print_platform_overview();

  KernelCopyTests();

  return 0;
}
//
//INSTANTIATE_TEST_CASE_P(
//    LZT, KernelCopyTests,
//    ::testing::Combine(::testing::Values(ZE_MEMORY_TYPE_HOST,
//                                         ZE_MEMORY_TYPE_SHARED),
//                       ::testing::Values(0, 1, size / 4, size / 2)));

//} // namespace
