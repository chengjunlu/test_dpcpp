#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdlib>
#include "triton_kernel.h"

static bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

#define EXPECT_EQ(value1, value2)                               \
  {                                                             \
    auto result = (value2);                                     \
    if ((value1) != (result)) {                                 \
      std::cout << "L0 API error code:" << std::hex << result << std::endl; \
      exit(-1);                                                \
    }                                                           \
  }

#define EXPECT_TRUE(value1) EXPECT_EQ(true, value1)

ze_module_handle_t create_module(
    ze_context_handle_t context,
    ze_device_handle_t device,
    uint32_t* binary_ptr,
    size_t binary_size) {

  const char* build_flags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;

  ze_module_desc_t module_description = {};
  module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  ze_module_constants_t module_constants = {};
  module_constants.numConstants=0;
  module_constants.pConstantIds=0;
  module_constants.pConstantValues=0;

  module_description.pNext = nullptr;
  module_description.format = format;
  module_description.inputSize =
      static_cast<uint32_t>(binary_size * sizeof(uint32_t));
  module_description.pInputModule = (uint8_t*)binary_ptr;
  module_description.pBuildFlags = build_flags;
  module_description.pConstants = &module_constants;

  ze_module_build_log_handle_t buildlog;
  ze_module_handle_t module;
  auto context_initial = context;
  auto device_initial = device;
  auto error_no = zeModuleCreate(
      context, device, &module_description, &module, &buildlog);

  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeModuleBuildLogGetString(buildlog, &szLog, nullptr));

    char* strLog = (char*)malloc(szLog);
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeModuleBuildLogGetString(buildlog, &szLog, strLog));

    std::cout << "L0 build module failed. Log:\\n" << strLog << std::endl;
    free(strLog);
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeModuleBuildLogDestroy(buildlog));
  }

  EXPECT_EQ(ZE_RESULT_SUCCESS, error_no);

  return module;
}

void printModuleKernelName(ze_module_handle_t hModule) {
  uint32_t Count = 0;
  auto ret = zeModuleGetKernelNames(hModule, &Count, nullptr);
  assert(ret == ZE_RESULT_SUCCESS);
  std::unique_ptr<const char*[]> PNames(new const char*[Count]);
  ret = zeModuleGetKernelNames(hModule, &Count, PNames.get());
  assert(ret == ZE_RESULT_SUCCESS);
  for (uint32_t i = 0; i < Count; ++i) {
    std::cout << std::string(PNames[i]) << std::endl;
  }
}

ze_kernel_handle_t create_function(
    ze_module_handle_t module,
    ze_kernel_flags_t flag,
    std::string func_name) {
  ze_kernel_handle_t kernel;
  ze_kernel_desc_t kernel_description = {};
  kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;

  kernel_description.pNext = nullptr;
  kernel_description.flags = flag;
  kernel_description.pKernelName = func_name.c_str();
  auto module_initial = module;
  std::cout << "create kernel:" << func_name << std::endl;
  EXPECT_EQ(
      ZE_RESULT_SUCCESS, zeKernelCreate(module, &kernel_description, &kernel));
  return kernel;
}

ze_kernel_handle_t create_function(
    ze_module_handle_t module,
    std::string func_name) {
  return create_function(module, 0, func_name);
}

std::vector<std::unique_ptr<sycl::kernel>> compiled_kernel;

sycl::kernel& spirv_to_sycl_kernel(
    sycl::device& device,
    uint32_t* binary_ptr,
    size_t binary_size,
    std::string kernel_name) {
  auto ctx = device.get_platform().ext_oneapi_get_default_context();
  auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      device);
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      ctx);

  auto l0_module =
      create_module(l0_context, l0_device, binary_ptr, binary_size);
  printModuleKernelName(l0_module);

  auto l0_kernel = create_function(l0_module, kernel_name);

  auto kernel_bundle = sycl::make_kernel_bundle<
      sycl::backend::ext_oneapi_level_zero,
      sycl::bundle_state::executable>(
      {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx);

  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernel_bundle,
       l0_kernel,
       sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx);
  compiled_kernel.push_back(std::make_unique<sycl::kernel>(kernel));
  sycl::kernel* ptr =  compiled_kernel[compiled_kernel.size() - 1].get();
//  if (getBoolEnv("MLIR_ENABLE_DUMP")){
    std::cout << "compiled kernel ptr: " << ptr << std::endl;
    std::cout << "total kernels:" << compiled_kernel.size() << std::endl;
    for (auto& k : compiled_kernel) {
      std::cout << "  kernel:" << k->get_info<sycl::info::kernel::function_name>() << " @" << k.get() << std::endl;
    }
//  }
  return *ptr;
}

static void set_scalar_arg(
    cl::sycl::handler& cgh,
    int index,
    size_t size,
    const void* value) {
  switch (size) {
  case sizeof(uint8_t):
    cgh.set_arg(index, *static_cast<const uint8_t*>(value));
    break;
  case sizeof(uint16_t):
    cgh.set_arg(index, *static_cast<const uint16_t*>(value));
    break;
  case sizeof(uint32_t):
    cgh.set_arg(index, *static_cast<const uint32_t*>(value));
    break;
  case sizeof(uint64_t):
    cgh.set_arg(index, *static_cast<const uint64_t*>(value));
    break;
  default:
    assert(false && "wrong scalar size in sycl gen.");
  }
}

static void sycl_kernel_launch(int gridX,
                               int gridY,
                               int gridZ,
                               int num_warps,
                               int threads_per_warp,
                               int shared_memory,
                               sycl::queue& stream,
                               sycl::kernel& kernel_ptr,
                               void* input,
                               void* output) {
  std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
  void *params[] = { &input, &output };
  uint32_t num_params = sizeof(params)/sizeof(params[0]);
  uint32_t expected_num_params = kernel_ptr.get_info<sycl::info::kernel::num_args>();

  size_t global_range_x = gridX*threads_per_warp*num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;

  size_t local_range_x = num_warps*threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;

  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);

  if (getBoolEnv("MLIR_ENABLE_DUMP")){
    std::cout << "kernel info name:" << kernel_name << " @" << &kernel_ptr << std::endl;
    std::cout << "kernel info attributes:" << kernel_ptr.get_info<sycl::info::kernel::attributes>() << std::endl;
    std::cout << "kernel info reference_count:" << kernel_ptr.get_info<sycl::info::kernel::reference_count>() << std::endl;
    std::cout << "kernel info num_args:" << kernel_ptr.get_info<sycl::info::kernel::num_args>() << std::endl;

    std::cout << "launch num param:" << num_params << std::endl;
    std::cout << "  gridx: " << gridX << std::endl;
    std::cout << "  gridY: " << gridY << std::endl;
    std::cout << "  gridZ: " << gridZ << std::endl;
    std::cout << "  num_warps: " << num_warps << std::endl;
    std::cout << "  threads_per_warp: " << threads_per_warp << std::endl;
    std::cout << "  global range:[" << "x:"<< global_range_x << ", y:" << global_range_y << ", z:" << global_range_z << "]" << std::endl;
    std::cout << "  local range:[" << "x:"<< local_range_x << ", y:" << local_range_y << ", z:" << local_range_z << "]" << std::endl;
    std::cout << "  shared_memory: " << shared_memory << std::endl;

    std::cout << "  param 0:" << *(void**)params[0] << std::endl;
    std::cout << "  param 1:" << *(void**)params[1] << std::endl;
  }
  if (shared_memory) {
    expected_num_params -= 1;
  }
  assert(num_params == expected_num_params && "number of kernel param not matched");

  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {

    set_scalar_arg(cgh, 0, sizeof(void*), params[0]);
    set_scalar_arg(cgh, 1, sizeof(void*), params[1]);

    if (shared_memory) {
      using share_mem_t = sycl::accessor<int8_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;
      share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
      cgh.set_arg(num_params, local_buffer.get_pointer());
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    } else {
      cgh.parallel_for(parallel_work_size, kernel_ptr);
    }

  };

  auto event = stream.submit(cgf);
}


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

  std::cout << "root device count: " << root_devices.size() << std::endl;
  std::cout << "compile kernel on device: " << root_devices[0].get_info<sycl::info::device::name>() << std::endl;
  sycl::queue queue = sycl::queue(root_devices[0], {property::queue::in_order(),
                                                    property::queue::enable_profiling()});
  sycl::device device = root_devices[0];
  auto binary_size_in_char = sizeof(__void_kernel_spv) / sizeof(__void_kernel_spv[0]);
  sycl::kernel& kernel = spirv_to_sycl_kernel(device,
                                              (uint32_t*)__void_kernel_spv,
                                              binary_size_in_char/4,
                                              "triton__0d1d2d3d4d5d6d7d8d9d10d11d12d13d14d15d16d17d18d19d20d21d22d23d24d25d26d27d28d29d30d31d32d33d34d35d36d37d38d39d40d41d42d43d44d45d46d47d48d49d50d51d52d53d54d55d56d57d58d59d60d61d62d63d64d65d66d67d68d69d70d71d72d73d74d75d76d77d78d79d80d81d82d83d84d85d86d");

  return 0;
}