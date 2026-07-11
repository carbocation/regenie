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

#include "Step1_Compute.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using ComputeClock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
  if(status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream message;
    message << operation << ": cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
  }
}

int checked_int(Eigen::Index value, const char* dimension) {
  if(value < 0 || value > INT_MAX)
    throw std::runtime_error(std::string("CUDA backend dimension exceeds cuBLAS limits: ") + dimension);
  return static_cast<int>(value);
}

double elapsed_ms(const ComputeClock::time_point& start) {
  return std::chrono::duration<double, std::milli>(ComputeClock::now() - start).count();
}

__global__ void mirror_lower_triangle(double* matrix, int size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if(row < size && column < size && row < column)
    matrix[row + column * size] = matrix[column + row * size];
}

class CudaEventPair {
  public:
    CudaEventPair() : start_(nullptr), stop_(nullptr) {
      check_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
      try {
        check_cuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
      } catch(...) {
        cudaEventDestroy(start_);
        throw;
      }
    }

    ~CudaEventPair() {
      if(stop_) cudaEventDestroy(stop_);
      if(start_) cudaEventDestroy(start_);
    }

    void record_start() {
      check_cuda(cudaEventRecord(start_), "cudaEventRecord(start)");
    }

    double record_stop_and_elapsed_ms() {
      check_cuda(cudaEventRecord(stop_), "cudaEventRecord(stop)");
      check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");
      float milliseconds = 0;
      check_cuda(cudaEventElapsedTime(&milliseconds, start_, stop_), "cudaEventElapsedTime");
      return milliseconds;
    }

  private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

class CudaStep1ComputeBackend : public Step1ComputeBackend {
  public:
    explicit CudaStep1ComputeBackend(int device)
      : device_(device), handle_(nullptr), d_genotypes_(nullptr), d_phenotypes_(nullptr),
        d_gram_(nullptr), d_crossproduct_(nullptr), genotypes_capacity_(0),
        phenotypes_capacity_(0), gram_capacity_(0), crossproduct_capacity_(0) {

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      check_cuda(cudaGetDeviceProperties(&properties_, device_), "cudaGetDeviceProperties");
      check_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CudaStep1ComputeBackend() override {
      cudaSetDevice(device_);
      if(d_crossproduct_) cudaFree(d_crossproduct_);
      if(d_gram_) cudaFree(d_gram_);
      if(d_phenotypes_) cudaFree(d_phenotypes_);
      if(d_genotypes_) cudaFree(d_genotypes_);
      if(handle_) cublasDestroy(handle_);
    }

    const char* name() const override {
      return "cuda";
    }

    std::string description() const override {
      std::ostringstream result;
      result << properties_.name << " (device " << device_ << ", compute capability "
             << properties_.major << "." << properties_.minor << ")";
      return result.str();
    }

    void compute_products(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1GramMode mode,
      Step1ComputeTimings* timings) override {

      if(genotypes.cols() != phenotypes.rows())
        throw std::invalid_argument(
          "Step 1 compute backend received incompatible genotype and phenotype matrices");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int blocks = checked_int(genotypes.rows(), "genotype rows");
      const int samples = checked_int(genotypes.cols(), "genotype columns");
      const int phenotype_count = checked_int(phenotypes.cols(), "phenotype columns");

      gram.resize(blocks, blocks);
      crossproduct.resize(blocks, phenotype_count);
      if(blocks == 0 || samples == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }

      ensure_capacity(d_genotypes_, genotypes_capacity_, genotypes.size(), "cudaMalloc(genotypes)");
      ensure_capacity(d_gram_, gram_capacity_, gram.size(), "cudaMalloc(Gram matrix)");
      if(phenotype_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_, phenotypes.size(), "cudaMalloc(phenotypes)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, crossproduct.size(), "cudaMalloc(crossproduct)");
      }

      const Eigen::MatrixXd packed_genotypes = contiguous_copy_if_needed(genotypes);
      const Eigen::MatrixXd packed_phenotypes = phenotype_count > 0 ?
        contiguous_copy_if_needed(phenotypes) : Eigen::MatrixXd();
      const double* genotype_data = packed_genotypes.size() ? packed_genotypes.data() : genotypes.data();
      const double* phenotype_data = packed_phenotypes.size() ? packed_phenotypes.data() : phenotypes.data();

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_genotypes_, genotype_data, genotypes.size() * sizeof(double),
        cudaMemcpyHostToDevice), "copy genotypes to CUDA device");
      if(phenotype_count > 0)
        check_cuda(cudaMemcpy(d_phenotypes_, phenotype_data, phenotypes.size() * sizeof(double),
          cudaMemcpyHostToDevice), "copy phenotypes to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      const double alpha = 1.0;
      const double beta = 0.0;

      if(phenotype_count > 0) {
        std::unique_ptr<CudaEventPair> crossproduct_events;
        if(timings) {
          crossproduct_events.reset(new CudaEventPair());
          crossproduct_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
          blocks, phenotype_count, samples, &alpha,
          d_genotypes_, blocks, d_phenotypes_, samples, &beta,
          d_crossproduct_, blocks), "cublasDgemm(genotypes * phenotypes)");
        if(timings)
          timings->crossproduct_ms += crossproduct_events->record_stop_and_elapsed_ms();
      }

      std::unique_ptr<CudaEventPair> gram_events;
      if(timings) {
        gram_events.reset(new CudaEventPair());
        gram_events->record_start();
      }
      if(mode == Step1GramMode::selfadjoint_rank_update) {
        check_cublas(cublasDsyrk(handle_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
          blocks, samples, &alpha, d_genotypes_, blocks, &beta, d_gram_, blocks),
          "cublasDsyrk(genotype Gram matrix)");
        const dim3 threads(16, 16);
        const dim3 grid((blocks + threads.x - 1) / threads.x,
                        (blocks + threads.y - 1) / threads.y);
        mirror_lower_triangle<<<grid, threads>>>(d_gram_, blocks);
        check_cuda(cudaGetLastError(), "mirror Gram triangle kernel");
      } else {
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
          blocks, blocks, samples, &alpha,
          d_genotypes_, blocks, d_genotypes_, blocks, &beta,
          d_gram_, blocks), "cublasDgemm(genotype Gram matrix)");
      }
      if(timings) timings->gram_ms += gram_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      if(phenotype_count > 0)
        check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_, crossproduct.size() * sizeof(double),
          cudaMemcpyDeviceToHost), "copy crossproduct from CUDA device");
      check_cuda(cudaMemcpy(gram.data(), d_gram_, gram.size() * sizeof(double),
        cudaMemcpyDeviceToHost), "copy Gram matrix from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

  private:
    static Eigen::MatrixXd contiguous_copy_if_needed(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
      if(matrix.innerStride() == 1 && matrix.outerStride() == matrix.rows())
        return Eigen::MatrixXd();
      return Eigen::MatrixXd(matrix);
    }

    static void ensure_capacity(double*& pointer, size_t& capacity,
      Eigen::Index required, const char* label) {
      const size_t required_size = static_cast<size_t>(required);
      if(required_size <= capacity) return;
      if(required_size > std::numeric_limits<size_t>::max() / sizeof(double))
        throw std::runtime_error(std::string("CUDA allocation size overflow for ") + label);
      if(pointer) check_cuda(cudaFree(pointer), "cudaFree while growing buffer");
      pointer = nullptr;
      capacity = 0;
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), required_size * sizeof(double)), label);
      capacity = required_size;
    }

    int device_;
    cudaDeviceProp properties_;
    cublasHandle_t handle_;
    double* d_genotypes_;
    double* d_phenotypes_;
    double* d_gram_;
    double* d_crossproduct_;
    size_t genotypes_capacity_;
    size_t phenotypes_capacity_;
    size_t gram_capacity_;
    size_t crossproduct_capacity_;
};

}

bool cuda_step1_compute_backend_available(int device, std::string& reason) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if(status != cudaSuccess) {
    reason = cudaGetErrorString(status);
    return false;
  }
  if(device < 0 || device >= device_count) {
    std::ostringstream message;
    message << "device " << device << " was requested, but " << device_count
            << " CUDA device(s) are visible";
    reason = message.str();
    return false;
  }
  cudaDeviceProp properties;
  const cudaError_t properties_status = cudaGetDeviceProperties(&properties, device);
  if(properties_status != cudaSuccess) {
    reason = cudaGetErrorString(properties_status);
    return false;
  }
  reason.clear();
  return true;
}

std::unique_ptr<Step1ComputeBackend> make_cuda_step1_compute_backend(int device) {
  return std::unique_ptr<Step1ComputeBackend>(new CudaStep1ComputeBackend(device));
}
