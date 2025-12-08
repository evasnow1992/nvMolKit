// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NVMOLKIT_CUB_HELPERS_H
#define NVMOLKIT_CUB_HELPERS_H

#include <cub/cub.cuh>

#include "nvtx.h"

// Default to 0 if not defined (no CCCL or unknown version)
// Version encoding: MAJOR * 10000 + MINOR * 100 + PATCH
// Examples: 2.8.0 = 20800, 3.0.0 = 30000
#ifndef NVMOLKIT_CCCL_VERSION
#define NVMOLKIT_CCCL_VERSION 0
#endif

// Check for modern C++ operators support:
// - CCCL >= 3.0.0 (detected via CMake), OR
// - CUDA >= 13.0 when CCCL version detection failed (bundled CCCL >= 3.0.0)
#if NVMOLKIT_CCCL_VERSION >= 30000 || (NVMOLKIT_CCCL_VERSION == 0 && CUDART_VERSION >= 13000)
// CCCL >= 3.0.0 provides modern C++ functional operators
using cubMax = cuda::maximum<>;
using cubSum = cuda::std::plus<>;
#else
// Fall back to CUB operators for older CCCL or bundled CUDA headers
// Suppress deprecation warnings for cub::Max and cub::Sum in CCCL 2.x
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
using cubMax = cub::Max;
using cubSum = cub::Sum;
#pragma GCC diagnostic pop
#endif

#include <cub/device/device_reduce.cuh>
#include <iterator>

namespace nvmolkit {
namespace detail {

#if NVMOLKIT_CCCL_VERSION >= 20800 || (NVMOLKIT_CCCL_VERSION == 0 && CUDART_VERSION >= 12090)
// CCCL >= 2.8.0: Use CUB's DeviceReduce::ArgMax with new API

//! Wrapper for CUB's DeviceReduce::ArgMax (CCCL >= 2.8.0)
//! Uses the new API that returns max value and index separately
template <typename InputIteratorT>
inline cudaError_t DeviceArgMax(void*          d_temp_storage,
                                size_t&        temp_storage_bytes,
                                InputIteratorT d_in,
                                int*           d_max_value_out,
                                int*           d_max_index_out,
                                int            num_items,
                                cudaStream_t   stream = 0) {
  nvMolKit::ScopedNvtxRange range("CUB ArgMax (CCCL >= 2.8.0)");
  return cub::DeviceReduce::ArgMax(d_temp_storage,
                                   temp_storage_bytes,
                                   d_in,
                                   d_max_value_out,
                                   d_max_index_out,
                                   static_cast<int64_t>(num_items),
                                   stream);
}

#else
// CCCL < 2.8.0: Use custom kernel (faster than CUB wrapper with KeyValuePair overhead)

constexpr int kArgMaxBlockSize = 256;

//! Custom ArgMax kernel for CCCL < 2.8.0
//! Uses CUB's BlockReduce for intra-block reduction, avoiding KeyValuePair extraction overhead
__global__ void lastArgMaxKernel(const int* values, int num_items, int* outVal, int* outIdx) {
  int            maxVal = cuda::std::numeric_limits<int>::min();
  int            maxID  = -1;
  __shared__ int foundMaxVal[kArgMaxBlockSize];
  __shared__ int foundMaxIds[kArgMaxBlockSize];
  const auto     tid = static_cast<int>(threadIdx.x);

  for (int i = tid; i < num_items; i += kArgMaxBlockSize) {
    if (const int val = values[i]; val >= maxVal) {
      maxID  = i;
      maxVal = val;
    }
  }

  foundMaxVal[tid] = maxVal;
  foundMaxIds[tid] = maxID;

  __shared__ cub::BlockReduce<int, kArgMaxBlockSize>::TempStorage storage;
  const int actualMaxVal = cub::BlockReduce<int, kArgMaxBlockSize>(storage).Reduce(maxVal, cubMax());
  __syncthreads();  // For shared memory write of maxVal and maxID

  if (tid == 0) {
    *outVal = actualMaxVal;
    for (int i = kArgMaxBlockSize - 1; i >= 0; i--) {
      if (foundMaxVal[i] == actualMaxVal) {
        *outIdx = foundMaxIds[i];
        break;
      }
    }
  }
}

#endif

}  // namespace detail
}  // namespace nvmolkit

#endif  // NVMOLKIT_CUB_HELPERS_H
