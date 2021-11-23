#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include "FunctionTraits.h"

using namespace sycl;

// aligned vector generates vectorized load/store on XPU
template <int N_BYTES>
struct aligned_element {};
template <>
struct aligned_element<1> {
  using element_type = uint8_t;
};

template <>
struct aligned_element<2> {
  using element_type = uint16_t;
};

template <>
struct aligned_element<4> {
  using element_type = uint32_t;
};

template <>
struct aligned_element<8> {
  using element_type = uint64_t;
};

template <typename scalar_t, int vec_size>
struct aligned_vector {
  using element_type = typename aligned_element<sizeof(scalar_t)>::element_type;
  using type = sycl::vec<element_type, vec_size>;
};

template <int vec_size, typename T>
struct vectorized_args_tuple {
};

template <int vec_size, typename... Args>
struct vectorized_args_tuple<vec_size, std::tuple<Args...> > {
using vec_args_t = std::tuple< sycl::vec<
        typename aligned_element<sizeof(Args)>::element_type, vec_size>... >;
};

// Result of div/mod operation stored together.
template <typename Value>
struct DivMod {
  Value div, mod;

  DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider {
  IntDivider() {} // Dummy constructor for arrays.
  IntDivider(Value d) : divisor(d) {}

  inline Value div(Value n) const {
    return n / divisor;
  }
  inline Value mod(Value n) const {
    return n % divisor;
  }
  inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};


template <typename T, int size>
struct alignas(16) Array {
  T data[size];

  T operator[](int i) const {
    return data[i];
  }
  T& operator[](int i) {
    return data[i];
  }

  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;

  // Fill the array with x.
  Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
public:
  static constexpr int MAX_DIMS = 12;

  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  using offset_type = Array<index_t, NARGS>;

  static inline int __wa(int dim, int arg) {
    return NARGS * dim + arg;
  }

  OffsetCalculator(
          int dims,
          const int64_t* sizes,
          const int64_t* const* strides,
          const int64_t* element_sizes = nullptr)
          : dims(dims) {
//    static_assert(dims <= MAX_DIMS, "tensor has too many (>25) dims");
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < dims) {
        sizes_[dim] = IntDivider<index_t>(sizes[dim]);
      } else {
        sizes_[dim] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
                (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[__wa(dim, arg)] =
                dim < dims ? strides[arg][dim] / element_size : 0;
      }
    }
  }

  inline offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[__wa(dim, arg)];
      }
    }
    return offsets;
  }

public:
  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS * NARGS];
};


template <int NARGS, typename sizes, typename strides>
auto get(int32_t linear_idx, int dims,  sizes sizes_, strides strides_) {
  using offset_type = Array<int32_t, NARGS>;
  offset_type offsets;
#pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }

#pragma unroll
  for (int dim = 0; dim < 12; ++dim) {
    if (dim == dims) {
      break;
    }
    auto divmod = sizes_[dim].divmod(linear_idx);
    linear_idx = divmod.div;

#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] += divmod.mod * strides_[NARGS * dim + arg];
    }
  }
  return offsets;
}

template <typename TO, typename FROM>
inline std::decay_t<TO>& bitwise_cast(FROM& value) {
  static_assert(
          sizeof(TO) == sizeof(FROM), "in-compatible type size in bitwise_cast.");
  std::decay_t<TO>& transport_bits = *((std::decay_t<TO>*)&value);
  return transport_bits;
}

template <template <int i> typename func, int end, int current = 0>
struct static_unroll {
  template <typename... Args>
  static inline void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current + 1>::with_args(args...);
  }
};

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
  template <typename... Args>
  static inline void with_args(Args... args) {}
};


template <int unroll_index, typename Result, class F, class TupleVector, std::size_t... I>
constexpr void apply_vec_test_impl(Result&& results, F& f, TupleVector& t, std::index_sequence<I...>)
{
  using traits = function_traits<F>;
  using result_t = decltype(results[unroll_index]);
  auto ret =  std::__invoke(std::forward<F>(f),
                            bitwise_cast<typename traits::template arg<I>::type>(
                                    std::get<I>(std::forward<TupleVector>(t))[unroll_index])...);

  results[unroll_index] = bitwise_cast<result_t>(ret);
}

template <int unroll_index>
struct apply_vec_test {
  template <typename Result, typename F, typename TupleVector>
  static void apply(Result& results, F& f, TupleVector& t) {
    using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<TupleVector>>::value>;
    apply_vec_test_impl<unroll_index>(std::forward<Result>(results), f, t, Indices{});
  }
};

template <int arg_index>
struct unroll_load_test {
  template <
          typename args_t,
          typename array_t,
          typename offset_t,
          typename loader_t>
  static void apply(
          array_t& data,
          args_t& args,
          offset_t offset,
          loader_t loader,
          int unroll_index,
          int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    using dtype = decltype(std::get<arg_index>(args)[0]);
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args)[unroll_index] = loader.template load<std::decay_t<dtype>>(
            data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <
        int vec_size,
        typename func_t,
        typename array_t,
        typename inp_calc_t,
        typename out_calc_t,
        typename loader_t,
        typename storer_t>
static inline void launch_unrolled_kernel(
        cl::sycl::queue& dpcpp_queue,
        int64_t N,
        const func_t& f,
        array_t data,
        inp_calc_t ic,
        out_calc_t oc,
        loader_t l,
        storer_t s) {
  using traits = function_traits<func_t>;
  using ret_t = typename traits::result_type;
  int thread_num = (N + vec_size - 1) / vec_size;

  char* data_ptr[3];
  for(int i =0; i < 3; i++)
    data_ptr[i] = data[i];
  auto cgf = [&](handler &cgh) {
    auto kfn = [=](cl::sycl::item<1> item_id) {
      int thread_idx = item_id.get_linear_id();
      using traits = function_traits<func_t>;
      using result_t = typename aligned_element<sizeof(typename traits::result_type)>::element_type;
      using return_t = sycl::vec<result_t, vec_size>;
      using args_t = typename vectorized_args_tuple<vec_size, typename traits::ArgsTuple>::vec_args_t;

      return_t results;
      args_t args;

      constexpr int arity = std::tuple_size<args_t>::value;

      int i = 0;
      int linear_idx = thread_idx * vec_size + i;
#ifdef CALL_AS_MEMBER_FUNC
      auto offset = ic.get(linear_idx);
#else
//      auto offset = get<arity>(linear_idx, ic.dims, ic.sizes_, ic.strides_);
        using offset_type = Array<int32_t, arity>;
        offset_type offset;
#pragma unroll
        for (int arg = 0; arg < arity; arg++) {
          offset[arg] = 0;
        }

#pragma unroll
        for (int dim = 0; dim < 12; ++dim) {
          if (dim == ic.dims) {
            break;
          }
          auto divmod = ic.sizes_[dim].divmod(linear_idx);
          linear_idx = divmod.div;

#pragma unroll
          for (int arg = 0; arg < arity; arg++) {
            offset[arg] += divmod.mod * ic.strides_[arity * dim + arg];
          }
        }
#endif
      static_unroll<unroll_load_test, arity>::with_args(data_ptr, args, offset, l, i, 1);

      // unroll the compute multiple times
      static_unroll<apply_vec_test, vec_size>::with_args(results, f, args);

      linear_idx = thread_idx * vec_size + i;
      int __offset = linear_idx;
      s.store(results[i], data_ptr[0], __offset);
    };

    cgh.parallel_for(cl::sycl::range</*dim=*/1>(thread_num), kfn);
  };
  dpcpp_queue.submit(cgf);
}

struct LoadWithoutCast {
  template <typename scalar_t>
  scalar_t load(char* base_ptr, uint32_t offset, int arg) {
    return *(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }
};

struct StoreWithoutCast {
  template <typename scalar_t>
  void store(scalar_t value, char* base_ptr, uint32_t offset) const {
    *(reinterpret_cast<scalar_t*>(base_ptr) + offset) = value;
  }
};


struct TensorIterator {
public:

  using DimVector = std::vector<int64_t>;


  TensorIterator(int ndim, int num_outputs) : dims(ndim), num_outputs_(num_outputs) {};

  int ndim() const {
    return dims;
  }

  int noutputs() const {
    return num_outputs_;
  }

  const DimVector& strides(int arg) const {
    return strides_[arg];
  }

  const DimVector& shape() const {
    return shape_;
  }

  int element_size(int arg) const {
    return 4;
  }

  int dims;
  int num_outputs_ = 0;
  DimVector shape_;
  DimVector strides_[4];
};

template <int N>
static OffsetCalculator<N> make_input_offset_calculator(
        const TensorIterator& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(
          iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(
        const TensorIterator& iter) {
  assert(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(
          iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

int main() {
  constexpr int size=16;
  assert(0 && "cannot run");

  char* data[3];
// Create queue on implementation-chosen default device
  queue Q;
  TensorIterator iter(1, 1);
// Create buffer using host allocated "data" array
  auto input_offset_calculator = make_input_offset_calculator<2>(iter);
  auto output_offset_calculator = make_output_offset_calculator(iter);
  auto loader = LoadWithoutCast();
  auto storer = StoreWithoutCast();
  launch_unrolled_kernel<4>(Q,
          size,
          [=](int a, int b){ return a + b;},
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);


// Obtain access to buffer on the host
// Will wait for device kernel to execute to generate data
//  cl::sycl::host_accessor<int> A{B};
//  for (int i = 0; i < size; i++)
//    std::cout << "data[" << i << "] = " << A[i] << "\n";
  return 0;
}