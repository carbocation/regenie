/*

   This file is part of the regenie software package.

   Copyright (c) 2020-2024 Joelle Mbatchou, Andrey Ziyatdinov & Jonathan Marchini

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

*/

#ifndef CUDA_RESOURCES_H
#define CUDA_RESOURCES_H

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace regenie {
namespace cuda {

inline void check_resource_status(
    cudaError_t status, const char* operation) {
  if(status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
      cudaGetErrorString(status));
}

class EventPair {
 public:
  EventPair() : start_(nullptr), stop_(nullptr) {
    check_resource_status(cudaEventCreate(&start_), "cudaEventCreate(start)");
    try {
      check_resource_status(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
    } catch(...) {
      cudaEventDestroy(start_);
      throw;
    }
  }

  ~EventPair() {
    if(stop_) cudaEventDestroy(stop_);
    if(start_) cudaEventDestroy(start_);
  }

  EventPair(const EventPair&) = delete;
  EventPair& operator=(const EventPair&) = delete;

  void record_start() {
    check_resource_status(
      cudaEventRecord(start_), "cudaEventRecord(start)");
  }

  double record_stop_and_elapsed_ms() {
    check_resource_status(
      cudaEventRecord(stop_), "cudaEventRecord(stop)");
    check_resource_status(
      cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");
    float milliseconds = 0;
    check_resource_status(
      cudaEventElapsedTime(&milliseconds, start_, stop_),
      "cudaEventElapsedTime");
    return milliseconds;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

class HostRegistration {
 public:
  HostRegistration() noexcept : pointer_(nullptr) {}

  ~HostRegistration() {
    if(pointer_) cudaHostUnregister(pointer_);
  }

  HostRegistration(const HostRegistration&) = delete;
  HostRegistration& operator=(const HostRegistration&) = delete;

  HostRegistration(HostRegistration&& other) noexcept
      : pointer_(other.pointer_) {
    other.pointer_ = nullptr;
  }

  HostRegistration& operator=(HostRegistration&& other) noexcept {
    if(this == &other) return *this;
    if(pointer_) cudaHostUnregister(pointer_);
    pointer_ = other.pointer_;
    other.pointer_ = nullptr;
    return *this;
  }

  bool try_register(void* pointer, size_t bytes,
      unsigned int flags = cudaHostRegisterPortable) noexcept {
    if(pointer_ || !pointer || bytes == 0) return false;
    const cudaError_t status = cudaHostRegister(pointer, bytes, flags);
    if(status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    pointer_ = pointer;
    return true;
  }

  bool registered() const noexcept {
    return pointer_ != nullptr;
  }

  cudaError_t unregister_now() noexcept {
    if(!pointer_) return cudaSuccess;
    const cudaError_t status = cudaHostUnregister(pointer_);
    if(status == cudaSuccess) pointer_ = nullptr;
    return status;
  }

 private:
  void* pointer_;
};

}  // namespace cuda
}  // namespace regenie

#endif
