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

//! Wrapper for CUB's ArgMax that handles API differences across CCCL versions.
//! 
//! CCCL >= 2.8.0 changed the ArgMax API to return both the max value and index
//! separately, while older versions returned them as a KeyValuePair.
//!
//! This wrapper provides a consistent interface that always returns both the
//! max value and its index.
template <typename InputIteratorT>
inline cudaError_t DeviceArgMax(void*         d_temp_storage,
                                size_t&       temp_storage_bytes,
                                InputIteratorT d_in,
                                int*          d_max_value_out,
                                int*          d_max_index_out,
                                int           num_items,
                                cudaStream_t  stream = 0) {
// Check for new ArgMax API:
// - CCCL >= 2.8.0 (detected via CMake), OR
// - CUDA >= 12.9 when CCCL version detection failed (bundled CCCL >= 2.8.0)
#if NVMOLKIT_CCCL_VERSION >= 20800 || (NVMOLKIT_CCCL_VERSION == 0 && CUDART_VERSION >= 12090)
  // New API (CCCL >= 2.8.0): Returns max value and index separately
  nvMolKit::ScopedNvtxRange range("CUB ArgMax (CCCL >= 2.8.0)");
  return cub::DeviceReduce::ArgMax(d_temp_storage,
                                    temp_storage_bytes,
                                    d_in,
                                    d_max_value_out,
                                    d_max_index_out,
                                    static_cast<int64_t>(num_items),
                                    stream);
#else
  // Old API (CCCL < 2.8.0): Returns KeyValuePair<offset, value>
  // We need a temporary buffer to hold the KeyValuePair result
  nvMolKit::ScopedNvtxRange range("CUB ArgMax (CCCL < 2.8.0)");
  using InputValueT = typename std::iterator_traits<InputIteratorT>::value_type;
  using KeyValuePairT = cub::KeyValuePair<int, InputValueT>;
  
  // Calculate required temp storage for the KeyValuePair output
  size_t kvp_size = sizeof(KeyValuePairT);
  size_t kvp_storage_offset = 0;
  
  if (d_temp_storage == nullptr) {
    // First call: determine temp storage requirements
    size_t reduce_temp_bytes = 0;
    cudaError_t error = cub::DeviceReduce::ArgMax(nullptr,
                                                    reduce_temp_bytes,
                                                    d_in,
                                                    static_cast<KeyValuePairT*>(nullptr),
                                                    num_items,
                                                    stream);
    if (error != cudaSuccess) return error;
    
    // Align kvp_storage_offset to 256 bytes
    kvp_storage_offset = (reduce_temp_bytes + 255) & ~255;
    temp_storage_bytes = kvp_storage_offset + kvp_size;
    return cudaSuccess;
  }
  
  // Second call: perform the reduction
  // Recalculate kvp_storage_offset
  size_t reduce_temp_bytes = 0;
  cub::DeviceReduce::ArgMax(nullptr,
                             reduce_temp_bytes,
                             d_in,
                             static_cast<KeyValuePairT*>(nullptr),
                             num_items,
                             stream);
  kvp_storage_offset = (reduce_temp_bytes + 255) & ~255;
  
  // Use part of temp storage for KeyValuePair output
  KeyValuePairT* d_kvp_out = reinterpret_cast<KeyValuePairT*>(
      static_cast<char*>(d_temp_storage) + kvp_storage_offset);
  
  // Perform the reduction
  cudaError_t error = cub::DeviceReduce::ArgMax(d_temp_storage,
                                                  reduce_temp_bytes,
                                                  d_in,
                                                  d_kvp_out,
                                                  num_items,
                                                  stream);
  if (error != cudaSuccess) return error;
  
  // Extract the max value and index from KeyValuePair
  // Note: KeyValuePair has .key (index) and .value (max value)
  error = cudaMemcpyAsync(d_max_index_out,
                          &d_kvp_out->key,
                          sizeof(int),
                          cudaMemcpyDeviceToDevice,
                          stream);
  if (error != cudaSuccess) return error;
  
  error = cudaMemcpyAsync(d_max_value_out,
                          &d_kvp_out->value,
                          sizeof(InputValueT),
                          cudaMemcpyDeviceToDevice,
                          stream);
  return error;
#endif
}

}  // namespace detail
}  // namespace nvmolkit

#endif  // NVMOLKIT_CUB_HELPERS_H
