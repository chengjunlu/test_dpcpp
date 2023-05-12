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
  if (getBoolEnv("MLIR_ENABLE_DUMP")){
    for (uint32_t i = 0; i < Count; ++i) {
      std::cout << std::string(PNames[i]) << std::endl;
    }
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
  if (getBoolEnv("MLIR_ENABLE_DUMP")){ std::cout << "create kernel:" << func_name << std::endl;}
  EXPECT_EQ(
      ZE_RESULT_SUCCESS, zeKernelCreate(module, &kernel_description, &kernel));
  //  EXPECT_EQ(module, module_initial);
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
  if (getBoolEnv("MLIR_ENABLE_DUMP")){
    //  auto kernel_ids = kernel_bundle.get_kernel_ids();
    //  std::cout << "num_kernels:" << kernel_ids.size() << std::endl;
    //  for (auto& kernel_id : kernel_ids) {
    //    std::cout << "kernel name: " << kernel_id.get_name() << std::endl;
    //  }
  }
  compiled_kernel.push_back(std::make_unique<sycl::kernel>(kernel));
  sycl::kernel* ptr =  compiled_kernel[compiled_kernel.size() - 1].get();
  if (getBoolEnv("MLIR_ENABLE_DUMP")){
    std::cout << "compiled kernel ptr: " << ptr << std::endl;
    std::cout << "total kernels:" << compiled_kernel.size() << std::endl;
    for (auto& k : compiled_kernel) {
      std::cout << "  kernel:" << k->get_info<sycl::info::kernel::function_name>() << " @" << k.get() << std::endl;
    }
  }
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
        //cgh.parallel_for(sycl::nd_range{sycl::range{(uint32_t)gridX*threads_per_warp*num_warps}, sycl::range{work_group_size}}, kernel_ptr);
        cgh.parallel_for(parallel_work_size, kernel_ptr);
    } else {
        cgh.parallel_for(parallel_work_size, kernel_ptr);
    }

  };

  auto event = stream.submit(cgf);
}


using namespace sycl;


constexpr size_t row = 4;
constexpr size_t row_malloced = 8;
constexpr size_t column = 128;

float input_host[row_malloced*column] = { 2.7627e-01, -1.8546e+00,  6.2390e-01,  1.1453e+00,  1.0372e+00,
       1.8866e+00, -1.1170e-01, -3.6210e-01,  1.4868e-01, -4.3778e-01,
       2.1713e+00,  1.1523e+00, -1.8188e+00, -1.3805e-01,  5.3984e-01,
       -1.7753e+00,  1.3149e+00, -4.7345e-01, -1.0922e+00, -2.5003e-01,
       -9.8229e-01,  1.0313e+00,  4.9133e-01, -4.4665e-01, -8.0636e-01,
       1.3127e-01, -1.2126e+00,  1.5999e-01, -7.5522e-01,  3.4990e-01,
       9.7754e-01, -1.3859e-01,  1.0386e-01,  3.0059e-01,  9.6821e-01,
       8.6962e-01,  5.6778e-01,  4.6528e-01, -1.1654e+00, -2.0360e+00,
       -1.1554e+00,  3.3452e+00,  1.2673e-01, -6.9418e-01,  5.5767e-01,
       9.9147e-02,  6.3793e-01,  7.0311e-01, -9.1609e-01, -7.8601e-01,
       1.1192e+00, -9.8340e-01,  2.4452e-01, -5.8141e-01,  4.2956e-01,
       7.9840e-01, -6.1007e-01,  1.1854e+00, -7.1083e-01, -7.8113e-01,
       -2.3037e-01,  1.2020e-01, -7.8847e-01, -2.9601e+00, -7.9558e-01,
       2.8458e-01,  5.2254e-01,  8.3554e-02,  2.7208e+00,  6.1101e-01,
       -8.3594e-01, -3.6960e-03,  3.9309e-01,  6.8450e-01,  7.7520e-01,
       5.5600e-01, -5.4098e-01,  1.5244e+00,  1.2084e+00,  5.1277e-02,
       -4.2663e-01,  8.0794e-01,  5.7280e-02, -1.0402e+00,  9.3415e-01,
       -1.2003e-01,  2.2070e+00, -3.3999e+00,  1.2335e+00, -5.8788e-01,
       -9.7694e-01,  4.6807e-01,  2.8711e+00, -1.2081e+00,  1.0036e+00,
       1.1307e+00,  5.6004e-01, -6.1840e-01, -2.2480e-01,  2.2159e+00,
       -1.5426e-01,  2.8761e-01,  5.7936e-01,  9.3325e-01,  1.6696e+00,
       1.7172e+00, -2.5365e-01, -1.1630e+00,  5.3445e-01, -1.4090e+00,
       -1.0598e+00, -4.4612e-01, -1.1624e+00, -6.2199e-01,  6.0933e-01,
       6.1522e-01,  5.0430e-01, -6.7570e-01,  1.3135e+00,  9.0337e-01,
       -2.0981e+00, -2.5777e-01,  1.0236e+00, -3.1466e-01,  1.0729e+00,
       3.2602e-02,  2.5253e+00, -1.7490e-01,
     -8.4196e-01,  7.6601e-02, -6.7123e-01, -3.1716e-01, -6.2536e-01,
       -1.0576e+00,  1.0561e+00,  4.4917e-03,  2.1383e-01, -1.6957e+00,
       1.3481e+00, -1.9640e+00, -2.3166e-01,  4.4385e-02, -1.7123e+00,
       1.6121e+00, -6.6185e-01, -5.7917e-01,  1.7439e-01,  7.9887e-01,
       5.3740e-01, -8.2387e-01,  1.4616e-01, -9.0632e-02,  9.4136e-01,
       1.1773e+00,  2.4901e-02,  3.0500e-01,  9.5457e-01, -6.4567e-03,
       2.1487e-01,  1.2424e+00,  1.4691e-01,  2.9256e-01, -6.3105e-01,
       -4.0905e-01,  3.7377e-02,  9.3387e-01, -1.3786e+00, -3.2798e-01,
       3.9181e-01,  7.4832e-01,  1.6054e-01, -6.1209e-01,  6.8850e-01,
       1.8209e-01,  4.6806e-01, -7.1571e-01,  2.2349e-01, -2.2955e-01,
       -4.0992e-01, -1.1947e+00, -1.2146e+00, -1.6637e+00,  1.2883e+00,
       -6.2526e-01,  1.9553e+00, -1.8948e+00, -2.4864e-01,  1.2950e+00,
       -1.2771e+00, -1.2307e+00,  3.3829e-01,  5.1029e-01,  5.0414e-02,
       -8.1231e-01, -1.4680e-01,  2.0523e+00,  1.1454e+00,  7.6480e-02,
       -8.6066e-01, -3.8770e-01,  1.6050e+00,  1.0989e+00, -1.6032e+00,
       -8.2106e-01,  5.5889e-01,  1.8189e+00, -1.5310e+00,  8.7489e-02,
       1.3380e+00,  1.1304e+00, -1.4418e+00, -6.8681e-01,  4.6613e-01,
       -5.5181e-01,  6.2251e-01, -2.7146e-02,  1.2424e+00, -1.6539e+00,
       -3.3841e-01,  1.4056e+00, -1.1278e+00,  6.4675e-01, -7.1116e-01,
       -4.8019e-01,  1.3080e+00,  9.5647e-01,  1.0249e+00, -1.2952e+00,
       3.3733e-01,  2.6570e-01,  9.9643e-01,  1.2568e+00, -2.2161e+00,
       -3.2871e-01,  1.2152e-01,  5.0576e-01, -5.2233e-01, -5.9177e-01,
       8.2925e-03,  4.0168e-01,  1.8776e+00, -5.4460e-01,  3.7619e-01,
       -7.1118e-01,  1.9510e+00, -7.5899e-01,  2.2927e-01,  1.4068e+00,
       -1.0694e+00, -1.2194e+00,  6.3516e-01, -6.3990e-01,  1.6001e+00,
       -7.3053e-01, -1.0096e+00, -1.0026e+00,
      6.8184e-02, -2.0946e+00,  2.0400e-01,  3.8814e-01, -1.7571e+00,
       -4.8244e-01, -5.0773e-01,  1.3887e+00,  6.0554e-02, -4.8930e-02,
       -5.4880e-01,  1.0755e+00,  8.3342e-01, -1.4758e+00,  7.4712e-01,
       1.0026e+00,  2.2585e-01,  9.7528e-01,  7.3848e-01, -1.5390e+00,
       -6.1467e-01,  2.0815e-01, -1.8790e-01, -8.6317e-01, -1.2375e+00,
       -1.2924e+00, -1.9992e+00, -7.3822e-02, -2.7682e-01, -1.3603e+00,
       5.8557e-01, -9.7529e-01, -1.6441e+00, -1.2712e+00, -1.0530e+00,
       2.7930e-01, -8.2507e-01,  3.2111e-01, -1.1374e-01,  1.3579e+00,
       -1.1545e-01, -7.6371e-01, -2.7592e+00, -1.5617e-01,  8.4418e-01,
       -9.5833e-02, -4.7003e-02,  2.7680e-01, -1.2600e-01,  5.3985e-01,
       -8.3125e-01, -1.9962e-01, -5.5307e-01,  8.9622e-01,  1.5723e+00,
       1.6326e+00, -4.7598e-01, -1.6583e-02,  5.4232e-01, -2.7075e-01,
       -1.1582e+00,  1.2743e-01, -2.9932e-01, -1.1013e+00,  2.6241e-01,
       -3.1469e-01, -1.3805e+00, -2.8419e-01,  1.0606e+00,  1.0542e+00,
       6.3905e-01, -1.6993e-02,  2.7999e-01, -1.9614e-01,  6.5637e-02,
       -8.6654e-01,  1.0631e+00, -7.4120e-01,  1.4012e+00, -2.1439e+00,
       -5.4569e-01, -6.0899e-02,  7.1062e-01,  7.7846e-01, -4.5766e-01,
       7.1397e-01,  7.4384e-01, -2.3738e-01,  7.0466e-01, -1.5229e+00,
       -5.7317e-01,  6.9417e-01, -1.4055e-01,  3.7642e-01, -8.3164e-01,
       -8.6053e-03, -1.0751e+00, -1.4123e+00, -1.1322e+00,  1.1267e+00,
       -5.2204e-01,  5.7308e-01, -1.4531e+00, -5.0066e-01,  7.6590e-01,
       4.0344e-02,  1.6316e-01, -2.4132e+00,  2.6974e+00,  2.8218e-01,
       -1.5469e+00, -2.6948e-01, -9.5871e-01,  3.5715e-01, -1.0329e+00,
       2.9174e-01,  2.9756e-01, -5.9401e-01,  5.9039e-01, -3.2883e-02,
       -6.5060e-01, -9.9815e-02, -3.8818e-02, -1.6336e-01,  1.9469e-02,
       -3.2170e-01,  1.1410e+00,  1.7365e-01,
      1.0382e+00,  1.3050e+00,  5.9897e-02,  8.9994e-01, -6.4546e-02,
       -3.7747e-01,  1.2916e+00,  6.8393e-01,  1.0032e-01, -6.5040e-01,
       4.4506e-01,  5.9353e-01, -2.3753e-01,  8.5415e-02,  2.6710e-01,
       -1.6325e+00,  1.6956e+00,  7.3484e-01, -1.9250e+00,  1.3467e+00,
       7.7702e-01, -7.1233e-01, -1.0545e+00, -8.1190e-04, -7.1854e-01,
       3.3566e-01,  1.5182e+00, -1.4622e+00, -3.0652e-02,  1.6284e+00,
       1.9907e+00,  3.7310e-01, -5.8678e-01,  6.0845e-01, -8.7355e-01,
       -8.8298e-01,  1.2889e+00,  1.9978e-01,  5.4162e-01, -1.0576e+00,
       -3.5953e-01, -4.7393e-01, -9.3910e-01,  4.5105e-01,  1.7048e-01,
       -9.0499e-01,  9.8979e-01, -4.2435e-01, -1.2809e+00, -5.6502e-01,
       1.3094e-01, -3.5723e-03, -3.6889e-01,  7.7671e-01,  4.0690e-01,
       -1.6294e+00, -1.1894e+00,  3.2147e+00,  1.7822e+00, -1.5393e+00,
       2.2535e+00, -7.0167e-01, -4.8634e-01, -3.6024e-01,  2.6748e-01,
       1.6660e+00,  1.5488e+00,  1.5052e+00,  6.5937e-01,  4.6897e-01,
       -1.2033e-01, -7.4920e-01,  2.2442e+00, -4.6256e-01, -1.5933e+00,
       1.2497e+00, -7.1713e-01,  4.8672e-01,  3.8527e-01, -1.5106e+00,
       1.2836e-02,  5.4131e-01, -1.1752e+00,  7.0986e-01,  2.9898e-02,
       2.1972e-02,  1.1299e+00,  4.4052e-01, -8.6699e-01, -9.4279e-01,
       8.8402e-01, -2.6083e-01, -6.8501e-02, -5.3390e-01,  6.0158e-01,
       -2.1960e-01, -6.8975e-01,  3.0067e-01,  4.6570e-01,  2.5932e+00,
       -1.1267e+00,  1.1161e+00, -6.4139e-01,  2.8112e-02,  8.2196e-01,
       2.6284e-01,  1.3920e+00,  4.2538e-01, -2.1853e-01, -7.0953e-02,
       -3.2521e+00, -9.1573e-01,  4.7960e-01,  9.0359e-01,  5.8932e-01,
       5.8507e-01, -3.8548e-01,  2.2366e+00, -2.4201e+00,  2.9837e-01,
       -1.3647e+00,  3.2501e-01,  8.8251e-01, -6.7234e-01, -1.3131e+00,
       -1.2201e+00, -4.2845e-01, -1.8115e+00,
};

#define ceil(m, n) ((m) + (n) - 1) / (n)

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
  sycl::device device = root_devices[0];
  auto binary_size_in_char = sizeof(_home_guangyey__triton_cache_0afb8f49797367a0a82878706cb8e2c6_kernel_spvbin) / sizeof(_home_guangyey__triton_cache_0afb8f49797367a0a82878706cb8e2c6_kernel_spvbin[0]);
  sycl::kernel& kernel = spirv_to_sycl_kernel(device,
                                              (uint32_t*)_home_guangyey__triton_cache_0afb8f49797367a0a82878706cb8e2c6_kernel_spvbin,
                                              binary_size_in_char/4,
                                              "kernel_0d1d");


  float* input = (float*)malloc_device(sizeof(float)* row_malloced * column, queue);
  float* output = (float*)malloc_device(sizeof(float)* column, queue);

  for (size_t i = row * column; i < row_malloced * column; i++) {
    input_host[i] = 100.0;
  }
#if 0
  for(int i =0; i < row_malloced; i++ ) {
    for (int j = 0; j < column; j++) {
      input_host[i * column + j] = j / 32 + i * 10;
    }
  }
#endif
  std::cout << " input_host value" << std::endl;
  for(int i =0; i < row_malloced; i++ ) {
    std::cout << "[";
    for (int j = 0; j < column; j++) {
      std::cout << " " << input_host[i * column + j];
    }
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;

  auto e = queue.memcpy(input, input_host, sizeof(float) * row_malloced * column);
  e.wait();

  sycl_kernel_launch(1, 1, 1, 8, 32, 2048,
                     queue, kernel, input, output);

  float* output_host = (float*)malloc(sizeof(float)* 128);
  e = queue.memcpy(output_host, output, sizeof(float)* 128);
  e.wait();

  for(int i =0; i < 128; i++ ) {
    std::cout << " " << output_host[i];
  }
  std::cout << std::endl;

#if 0
  auto cgf = [&](sycl::handler &cgh) {
    using share_mem_t = sycl::accessor<int8_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;
    share_mem_t local_buffer = share_mem_t(2048, cgh);

    auto kfn = [=](sycl::nd_item<1> item) {
      // The ids mimic the
      // #blocked = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
      auto threadId = item.get_local_id(0);
      unsigned laneId = (unsigned)threadId % 32;
      unsigned warpId = (unsigned)threadId / 32;

      constexpr unsigned sizePerThread[2] = {1, 2};
      constexpr unsigned threadsPerWarp[2] = {1, 32};
      constexpr unsigned warpsPerCTA[2] = {4, 2};
      constexpr unsigned order[2] = {1, 0};

      const unsigned multiDimWarpId[2] = {(warpId / warpsPerCTA[1]), warpId % warpsPerCTA[1]};

      const unsigned multiDimThreadId[2] = {(laneId / sizePerThread[1]), laneId % sizePerThread[1]};

      // Wrap around multiDimWarpId/multiDimThreadId incase
      // shape[k] < shapePerCTA[k]
      constexpr unsigned shape[] = {4, 128};
      constexpr unsigned maxWarps_0 = ceil(4, 1 * 1);
      constexpr unsigned maxWarps_1 = ceil(128, 2 * 32);
      constexpr unsigned maxThreads_0 = ceil(4, 1);
      constexpr unsigned maxThreads_1 = ceil(128, 2);

      const size_t roundMultiDimWarpId[2] = {(multiDimWarpId[0] % maxWarps_0), (multiDimWarpId[1] % maxWarps_1)};

      const size_t roundMultiDimThreadId[2] = {(multiDimThreadId[0] % maxThreads_0), (multiDimThreadId[1] % maxThreads_1)};

      const size_t multiDimBase[2] = { ((roundMultiDimWarpId[0] * threadsPerWarp[0]) + roundMultiDimThreadId[0]) * sizePerThread[0],
          ((roundMultiDimWarpId[1] * threadsPerWarp[1]) + roundMultiDimThreadId[1]) * sizePerThread[1], };

      constexpr unsigned shapePerCTA[] = {sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0],
          sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1],};

      constexpr unsigned tilesPerDim[] = {ceil(shape[0], shapePerCTA[0]),
          ceil(shape[1], shapePerCTA[1]),};

      unsigned offset[2][] = {
          { 0 * sizePerThread[0] *
               threadsPerWarp[0] * warpsPerCTA[0] +
           0 * sizePerThread[0] *
               threadsPerWarp[0] +
           0 * sizePerThread[0] + 0,
              0 * sizePerThread[0] *
                    threadsPerWarp[0] * warpsPerCTA[0] +
                0 * sizePerThread[0] *
                    threadsPerWarp[0] +
                0 * sizePerThread[0] + 0,},
          {}
      };

      auto threadsPerWarp = (local_id / 2) % 32;
      auto threadsPerWarp_X = threadsPerWarp % 32;
      auto threadsPerWarp_Y = (threadsPerWarp / 32) % 1;

      auto warpsPerCTA = local_id / (2 * 32);
      auto warpsPerCTA_X = warpsPerCTA % 2;
      auto warpsPerCTA_Y = (warpsPerCTA / 2) % 4;

      auto elem0 = input[0 + (threadsPerWarp_X + threadsPerWarp_Y * 32) * 2 + (warpsPerCTA_X + warpsPerCTA_Y * 2) * 2 * 32];
      auto elem1 = input[1 + (threadsPerWarp_X + threadsPerWarp_Y * 32) * 2 + (warpsPerCTA_X + warpsPerCTA_Y * 2) * 2 * 32];
      auto share_ptr = (float*)local_buffer.get_pointer().get();

      // put data
      share_ptr[0] = elem0;
      item.barrier(sycl::access::fence_space::local_space);

      // get data
      auto rhs = share_ptr[0];

      // compare
      auto max = std::max(elem0, rhs);


      output[local_id] = 1;
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(256 * 1),
            sycl::range<1>(256)),
        kfn);
  };

  queue.submit(cgf);

  e = queue.memcpy(output_host, output, sizeof(float)* 128);
  e.wait();

  for(int i =0; i < 128; i++ ) {
    std::cout << " " << output_host[i];
  }
  std::cout << std::endl;
#endif
}