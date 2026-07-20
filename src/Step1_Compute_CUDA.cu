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

#define EIGEN_NO_CUDA
#include "Step1_Compute.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

void check_cusolver(cusolverStatus_t status, const char* operation) {
  if(status != CUSOLVER_STATUS_SUCCESS) {
    std::ostringstream message;
    message << operation << ": cuSOLVER status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
  }
}

int checked_int(Eigen::Index value, const char* dimension) {
  if(value < 0 || value > INT_MAX)
    throw std::runtime_error(std::string("CUDA backend dimension exceeds cuBLAS limits: ") + dimension);
  return static_cast<int>(value);
}

int checked_element_count(int first, int second, const char* label) {
  const long long count = static_cast<long long>(first) * second;
  if(count < 0 || count > INT_MAX)
    throw std::runtime_error(std::string("CUDA kernel element count exceeds integer limits: ") + label);
  return static_cast<int>(count);
}

Eigen::Index bounded_cuda_chunk_rows(Eigen::Index rows, Eigen::Index columns) {
  Eigen::Index max_elements = 125000000; // approximately 1 GB of FP64 data
  const char* chunk_mb_text = std::getenv("REGENIE_CUDA_CHUNK_MB");
  if(chunk_mb_text && *chunk_mb_text) {
    char* end = nullptr;
    const long chunk_mb = std::strtol(chunk_mb_text, &end, 10);
    if(end != chunk_mb_text && *end == '\0' && chunk_mb > 0 &&
       chunk_mb <= std::numeric_limits<int>::max())
      max_elements = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(chunk_mb) * 1000000 / sizeof(double));
  }
  max_elements = std::min<Eigen::Index>(max_elements, INT_MAX);
  if(rows <= 0) return 0;
  const Eigen::Index rows_per_chunk = columns > 0 ?
    std::max<Eigen::Index>(1, max_elements / columns) : rows;
  return std::min(rows, rows_per_chunk);
}

Eigen::Index cuda_resident_preprocess_max_elements() {
  const char* resident_mb_text = std::getenv("REGENIE_CUDA_RESIDENT_MB");
  if(resident_mb_text && *resident_mb_text) {
    char* end = nullptr;
    const long resident_mb = std::strtol(resident_mb_text, &end, 10);
    if(end == resident_mb_text || *end != '\0' || resident_mb < 0 ||
       resident_mb > std::numeric_limits<int>::max())
      throw std::invalid_argument(
        "REGENIE_CUDA_RESIDENT_MB must be a non-negative integer");
    return std::min<Eigen::Index>(
      static_cast<Eigen::Index>(resident_mb) *
        1000000 / sizeof(double),
      INT_MAX);
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  free_bytes = std::min(free_bytes, total_bytes);
  const size_t automatic_cap_bytes = size_t(6000) * 1000000;
  const size_t automatic_budget_bytes = std::min(
    automatic_cap_bytes, free_bytes / 5 * 3);
  return std::min<Eigen::Index>(
    static_cast<Eigen::Index>(automatic_budget_bytes / sizeof(double)),
    INT_MAX);
}

Eigen::Index cuda_level1_resident_max_elements() {
  const char* resident_mb_text =
    std::getenv("REGENIE_CUDA_LEVEL1_RESIDENT_MB");
  if(resident_mb_text && *resident_mb_text) {
    char* end = nullptr;
    const long resident_mb = std::strtol(resident_mb_text, &end, 10);
    if(end == resident_mb_text || *end != '\0' || resident_mb < 0 ||
       resident_mb > std::numeric_limits<int>::max())
      throw std::invalid_argument(
        "REGENIE_CUDA_LEVEL1_RESIDENT_MB must be a non-negative integer");
    return std::min<Eigen::Index>(
      static_cast<Eigen::Index>(resident_mb) *
        1000000 / sizeof(double),
      INT_MAX);
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  free_bytes = std::min(free_bytes, total_bytes);
  // Weighted IRLS needs both the resident design and an equally sized
  // weighted-design workspace.  Limit one copy to 40% of device memory and
  // leave the remainder for cuBLAS/cuSOLVER workspaces and model state.
  const size_t automatic_cap_bytes = size_t(16000) * 1000000;
  const size_t automatic_budget_bytes = std::min(
    automatic_cap_bytes, free_bytes / 5 * 2);
  return std::min<Eigen::Index>(
    static_cast<Eigen::Index>(automatic_budget_bytes / sizeof(double)),
    INT_MAX);
}

size_t cuda_pinned_staging_bytes() {
  const char* value = std::getenv("REGENIE_CUDA_PINNED_STAGING_MB");
  if(!value || !*value) return size_t(64) * 1000000;
  char* end = nullptr;
  const unsigned long long megabytes = std::strtoull(value, &end, 10);
  if(end == value || *end != '\0' || megabytes >
      std::numeric_limits<size_t>::max() / 1000000)
    throw std::invalid_argument(
      "REGENIE_CUDA_PINNED_STAGING_MB must be a non-negative integer");
  return static_cast<size_t>(megabytes) * 1000000;
}

bool cuda_level0_cholesky_enabled() {
  const char* value = std::getenv("REGENIE_CUDA_LEVEL0_CHOLESKY");
  if(!value || !*value || std::string(value) == "1") return true;
  if(std::string(value) == "0") return false;
  throw std::invalid_argument(
    "REGENIE_CUDA_LEVEL0_CHOLESKY must be '0' or '1'");
}

bool cuda_level0_fold_batch_enabled() {
  const char* value = std::getenv("REGENIE_CUDA_LEVEL0_FOLD_BATCH");
  if(!value || !*value || std::string(value) == "1") return true;
  if(std::string(value) == "0") return false;
  throw std::invalid_argument(
    "REGENIE_CUDA_LEVEL0_FOLD_BATCH must be '0' or '1'");
}

bool cuda_level0_resident_folds_enabled() {
  const char* value = std::getenv("REGENIE_CUDA_LEVEL0_RESIDENT_FOLDS");
  if(!value || !*value || std::string(value) == "1") return true;
  if(std::string(value) == "0") return false;
  throw std::invalid_argument(
    "REGENIE_CUDA_LEVEL0_RESIDENT_FOLDS must be '0' or '1'");
}

bool cuda_register_packed_hardcalls_enabled() {
  const char* value = std::getenv("REGENIE_CUDA_REGISTER_PACKED");
  if(!value || !*value || std::string(value) == "1") return true;
  if(std::string(value) == "0") return false;
  throw std::invalid_argument(
    "REGENIE_CUDA_REGISTER_PACKED must be '0' or '1'");
}

bool cuda_host_pointer_is_registered(const void* pointer) {
  if(!pointer) return false;
  cudaPointerAttributes attributes;
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if(status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeHost;
#else
  return attributes.memoryType == cudaMemoryTypeHost;
#endif
}

bool cuda_direct_grouped_upload_enabled() {
  const char* value = std::getenv("REGENIE_CUDA_DIRECT_GROUPED_UPLOAD");
  if(!value || !*value || std::string(value) == "1") return true;
  if(std::string(value) == "0") return false;
  throw std::invalid_argument(
    "REGENIE_CUDA_DIRECT_GROUPED_UPLOAD must be '0' or '1'");
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

__global__ void build_ridge_inverse(const double* eigenvalues,
  const double* ridge_parameters, double* inverse, int size, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) {
    const int row = index % size;
    const int parameter = index / size;
    inverse[index] = 1.0 / (eigenvalues[row] + ridge_parameters[parameter]);
  }
}

__global__ void build_scaled_right_hand_sides(const double* inverse,
  const double* right_hand_sides, double* scaled, int size,
  int phenotype_count, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) {
    const int row = index % size;
    const int combination = index / size;
    const int phenotype = combination % phenotype_count;
    const int parameter = combination / phenotype_count;
    scaled[index] = inverse[row + parameter * size] *
      right_hand_sides[row + phenotype * size];
  }
}

__global__ void fill_constant(double* values, double value, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) values[index] = value;
}

__global__ void normalize_design_columns(double* design, int rows,
  int count, const double* means, const double* inverse_standard_deviations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) {
    const int column = index / rows;
    design[index] = (design[index] - means[column]) *
      inverse_standard_deviations[column];
  }
}

__global__ void square_elements(const double* input, double* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) output[index] = input[index] * input[index];
}

__global__ void scale_matrix_rows(const double* input, const double* weights,
  double* output, int rows, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) output[index] = input[index] * weights[index % rows];
}

__global__ void mask_genotype_columns(double* genotypes,
  const double* sample_weights, int rows, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count)
    genotypes[index] *= sample_weights[index / rows];
}

__global__ void packed_hardcall_row_statistics(
  const unsigned char* packed_hardcalls, size_t packed_stride_bytes,
  const double* sample_weights, double* row_sums,
  unsigned int* row_counts, int rows, int columns) {
  const int row = blockIdx.x;
  if(row >= rows) return;
  double sum = 0.0;
  unsigned int count = 0;
  const unsigned char* packed_row = packed_hardcalls +
    static_cast<size_t>(row) * packed_stride_bytes;
  const int packed_columns = (columns + 3) / 4;
  for(int packed_column = threadIdx.x; packed_column < packed_columns;
      packed_column += blockDim.x) {
    const unsigned char codes = packed_row[packed_column];
    const int first_column = packed_column * 4;
    for(int lane = 0; lane < 4; ++lane) {
      const int column = first_column + lane;
      if(column >= columns || sample_weights[column] <= 0) continue;
      const unsigned int code = (codes >> (2 * lane)) & 3u;
      if(code != 3u) {
        sum += static_cast<double>(code);
        count++;
      }
    }
  }
  __shared__ double partial_sums[256];
  __shared__ unsigned int partial_counts[256];
  partial_sums[threadIdx.x] = sum;
  partial_counts[threadIdx.x] = count;
  __syncthreads();
  for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if(threadIdx.x < offset) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
      partial_counts[threadIdx.x] += partial_counts[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    row_sums[row] = partial_sums[0];
    row_counts[row] = partial_counts[0];
  }
}

__global__ void transpose_packed_hardcalls(
  const unsigned char* input, unsigned char* output,
  int rows, int packed_columns, size_t input_stride) {
  __shared__ unsigned char tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  for(int offset = 0; offset < 32; offset += 8)
    if(x < packed_columns && y + offset < rows)
      tile[threadIdx.y + offset][threadIdx.x] =
        input[static_cast<size_t>(y + offset) * input_stride + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  for(int offset = 0; offset < 32; offset += 8)
    if(x < rows && y + offset < packed_columns)
      output[static_cast<size_t>(y + offset) * rows + x] =
        tile[threadIdx.x][threadIdx.y + offset];
}

__global__ void expand_packed_hardcalls(
  const unsigned char* transposed_hardcalls,
  const double* sample_weights, const double* row_sums,
  const unsigned int* row_counts, double* genotypes,
  int rows, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= count) return;
  const int row = index % rows;
  const int column = index / rows;
  if(sample_weights[column] <= 0) {
    genotypes[index] = 0.0;
    return;
  }
  const unsigned int code =
    (transposed_hardcalls[
       static_cast<size_t>(column >> 2) * rows + row] >>
       (2 * (column & 3))) & 3u;
  const double value = code == 3u ?
    (row_counts[row] ? row_sums[row] / row_counts[row] : 0.0) :
    static_cast<double>(code);
  genotypes[index] = value * sample_weights[column];
}

__global__ void compute_genotype_row_scales(const double* genotypes,
  double* row_scales, int rows, int columns, double degrees_of_freedom) {
  const int row = blockIdx.x;
  if(row >= rows) return;
  double sum = 0.0;
  for(int column = threadIdx.x; column < columns; column += blockDim.x) {
    const double value = genotypes[row + column * rows];
    sum += value * value;
  }
  extern __shared__ double partial_sums[];
  partial_sums[threadIdx.x] = sum;
  __syncthreads();
  for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if(threadIdx.x < offset)
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
    __syncthreads();
  }
  if(threadIdx.x == 0)
    row_scales[row] = sqrt(partial_sums[0] / degrees_of_freedom);
}

__global__ void scale_genotype_rows(double* genotypes,
  const double* row_scales, const double* row_multipliers,
  int rows, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) {
    const int row = index % rows;
    const double multiplier = row_multipliers ? row_multipliers[row] : 1.0;
    genotypes[index] *= multiplier / row_scales[row];
  }
}

__global__ void add_diagonal_penalty(double* matrix,
  const double* penalty_multipliers, double ridge_parameter, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size)
    matrix[index + index * size] += ridge_parameter * penalty_multipliers[index];
}

__global__ void add_uniform_diagonal_penalty(double* matrix,
  double ridge_parameter, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size)
    matrix[index + index * size] += ridge_parameter;
}

__global__ void compute_weighted_leverage_diagonal(
  const double* design, const double* inverse_design,
  const double* weights, double* leverage, int size, int sample_count) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  if(sample < sample_count) {
    double value = 0.0;
    for(int feature = 0; feature < size; ++feature) {
      const int index = sample + feature * sample_count;
      value += design[index] * inverse_design[index];
    }
    leverage[sample] = weights[sample] * value;
  }
}

__global__ void grouped_leave_one_out_predictions(
  const double* design, const double* inverse_design,
  const double* residuals,
  const double* leverage, double* predictions, int sample_count,
  int group, int group_offset, int group_size) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  if(sample < sample_count) {
    double influence = 0.0;
    const int group_end = group_offset + group_size;
    for(int feature = group_offset; feature < group_end; ++feature) {
      const int index = sample + feature * sample_count;
      const double design_value = design[index];
      influence += design_value * inverse_design[index];
    }
    predictions[sample + group * sample_count] -=
      influence * residuals[sample] / (1.0 - leverage[sample]);
  }
}

__global__ void apply_leave_one_out_correction(double* predictions,
  const double* leverage, const double* outcomes, int sample_count,
  int phenotype_count, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) {
    const int sample = index % sample_count;
    const int combination = index / sample_count;
    const int phenotype = combination % phenotype_count;
    const int parameter = combination / phenotype_count;
    const double gamma = leverage[sample + parameter * sample_count];
    predictions[index] = (predictions[index] -
      gamma * outcomes[sample + phenotype * sample_count]) / (1.0 - gamma);
  }
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

struct CudaLevel0CholeskyLane {
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  double* gram = nullptr;
  double* factor = nullptr;
  double* right_hand_sides = nullptr;
  double* solve = nullptr;
  double* coefficients = nullptr;
  double* predictions = nullptr;
  double* workspace = nullptr;
  int* info = nullptr;
  size_t gram_capacity = 0;
  size_t factor_capacity = 0;
  size_t right_hand_sides_capacity = 0;
  size_t solve_capacity = 0;
  size_t coefficients_capacity = 0;
  size_t predictions_capacity = 0;
  size_t workspace_capacity = 0;
  size_t info_capacity = 0;
};

class CudaStep1ComputeBackend : public Step1ComputeBackend {
  public:
    explicit CudaStep1ComputeBackend(int device)
      : device_(device), handle_(nullptr), solver_handle_(nullptr),
        pinned_staging_chunk_bytes_(cuda_pinned_staging_bytes()),
        level0_cholesky_enabled_(cuda_level0_cholesky_enabled()),
        level0_fold_batch_enabled_(cuda_level0_fold_batch_enabled()),
        level0_resident_folds_enabled_(
          cuda_level0_resident_folds_enabled()),
        register_packed_hardcalls_enabled_(
          cuda_register_packed_hardcalls_enabled()),
        direct_grouped_upload_(cuda_direct_grouped_upload_enabled()),
        resident_preprocess_max_elements_(0),
        level1_resident_max_elements_(0),
        d_genotypes_(nullptr), d_resident_genotypes_(nullptr),
        d_phenotypes_(nullptr), d_gram_(nullptr), d_crossproduct_(nullptr),
        d_factorized_(nullptr),
        d_ridge_vectors_(nullptr), d_ridge_values_(nullptr),
        d_ridge_rhs_(nullptr),
        d_eigenvalues_(nullptr), d_solver_workspace_(nullptr), d_solver_info_(nullptr),
        d_ridge_parameters_(nullptr), d_inverse_(nullptr), d_scaled_rhs_(nullptr),
        d_predictions_(nullptr), d_outcomes_(nullptr), d_projected_(nullptr),
        d_level1_design_(nullptr), d_level1_ones_(nullptr),
        d_level0_phenotypes_(nullptr),
        d_squared_(nullptr), d_leverage_(nullptr),
        d_preprocess_covariates_(nullptr), d_preprocess_weights_(nullptr),
        d_preprocess_coefficients_(nullptr), d_preprocess_scales_(nullptr),
        d_preprocess_multipliers_(nullptr), d_packed_hardcalls_(nullptr),
        d_transposed_hardcalls_(nullptr), d_packed_row_counts_(nullptr),
        genotypes_capacity_(0), resident_genotypes_capacity_(0),
        resident_host_data_(nullptr), resident_rows_(0), resident_columns_(0),
        resident_valid_(false), resident_design_rows_(0),
        resident_design_columns_(0), resident_design_valid_(false),
        resident_design_uses_level1_cache_(false),
        level1_design_rows_(0), level1_design_columns_(0),
        level1_design_cached_columns_(0),
        resident_fold_system_count_(0),
        resident_fold_rhs_count_(0),
        resident_fold_systems_valid_(false),
        resident_fold_systems_design_orientation_(false),
        packed_static_inputs_valid_(false),
        packed_static_covariates_(nullptr),
        packed_static_weights_(nullptr),
        packed_static_samples_(0),
        packed_static_covariate_count_(0),
        level0_phenotypes_host_(nullptr),
        level0_phenotype_rows_(0), level0_phenotype_columns_(0),
        phenotypes_capacity_(0), gram_capacity_(0),
        factorized_capacity_(0), factorized_size_(-1),
        ridge_vectors_capacity_(0), ridge_values_capacity_(0),
        ridge_rhs_capacity_(0),
        crossproduct_capacity_(0), eigenvalues_capacity_(0), solver_workspace_capacity_(0),
        pinned_staging_capacity_(0), pinned_staging_available_(true),
        ridge_factorized_size_(-1), ridge_factorized_rhs_count_(0) {

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      resident_preprocess_max_elements_ =
        cuda_resident_preprocess_max_elements();
      level1_resident_max_elements_ =
        cuda_level1_resident_max_elements();
      check_cuda(cudaGetDeviceProperties(&properties_, device_), "cudaGetDeviceProperties");
      check_cublas(cublasCreate(&handle_), "cublasCreate");
      try {
        check_cusolver(cusolverDnCreate(&solver_handle_), "cusolverDnCreate");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_solver_info_), sizeof(int)),
          "cudaMalloc(cuSOLVER info)");
      } catch(...) {
        if(solver_handle_) cusolverDnDestroy(solver_handle_);
        if(handle_) cublasDestroy(handle_);
        throw;
      }
    }

    ~CudaStep1ComputeBackend() override {
      cudaSetDevice(device_);
      release_packed_hardcall_buffers_noexcept();
      for(auto& lane : level0_cholesky_lanes_)
        release_level0_cholesky_lane(lane);
      for(int index = 0; index < 2; ++index) {
        if(upload_streams_[index])
          cudaStreamSynchronize(upload_streams_[index]);
        if(pinned_staging_[index]) cudaFreeHost(pinned_staging_[index]);
        if(upload_streams_[index])
          cudaStreamDestroy(upload_streams_[index]);
      }
      if(d_preprocess_multipliers_) cudaFree(d_preprocess_multipliers_);
      if(d_preprocess_scales_) cudaFree(d_preprocess_scales_);
      if(d_preprocess_coefficients_) cudaFree(d_preprocess_coefficients_);
      if(d_preprocess_weights_) cudaFree(d_preprocess_weights_);
      if(d_preprocess_covariates_) cudaFree(d_preprocess_covariates_);
      if(d_packed_row_counts_) cudaFree(d_packed_row_counts_);
      if(d_transposed_hardcalls_) cudaFree(d_transposed_hardcalls_);
      if(d_packed_hardcalls_) cudaFree(d_packed_hardcalls_);
      if(d_leverage_) cudaFree(d_leverage_);
      if(d_squared_) cudaFree(d_squared_);
      if(d_projected_) cudaFree(d_projected_);
      if(d_level1_design_) cudaFree(d_level1_design_);
      if(d_level1_ones_) cudaFree(d_level1_ones_);
      if(d_level0_phenotypes_) cudaFree(d_level0_phenotypes_);
      if(d_outcomes_) cudaFree(d_outcomes_);
      if(d_predictions_) cudaFree(d_predictions_);
      if(d_scaled_rhs_) cudaFree(d_scaled_rhs_);
      if(d_inverse_) cudaFree(d_inverse_);
      if(d_ridge_parameters_) cudaFree(d_ridge_parameters_);
      if(d_solver_info_) cudaFree(d_solver_info_);
      if(d_solver_workspace_) cudaFree(d_solver_workspace_);
      if(d_eigenvalues_) cudaFree(d_eigenvalues_);
      if(d_ridge_rhs_) cudaFree(d_ridge_rhs_);
      if(d_ridge_values_) cudaFree(d_ridge_values_);
      if(d_ridge_vectors_) cudaFree(d_ridge_vectors_);
      if(d_factorized_) cudaFree(d_factorized_);
      if(d_crossproduct_) cudaFree(d_crossproduct_);
      if(d_gram_) cudaFree(d_gram_);
      if(d_phenotypes_) cudaFree(d_phenotypes_);
      if(d_resident_genotypes_) cudaFree(d_resident_genotypes_);
      if(d_genotypes_) cudaFree(d_genotypes_);
      if(solver_handle_) cusolverDnDestroy(solver_handle_);
      if(handle_) cublasDestroy(handle_);
    }

    const char* name() const override {
      return "cuda";
    }

    std::string description() const override {
      std::ostringstream result;
      result << properties_.name << " (device " << device_ << ", compute capability "
             << properties_.major << "." << properties_.minor;
      if(pinned_staging_chunk_bytes_ > 0)
        result << ", pinned upload staging " <<
          (pinned_staging_chunk_bytes_ / 1000000.0) << " MB";
      result << ", level0 solver " <<
        (level0_cholesky_enabled_ ? "cholesky" : "eigendecomposition");
      if(level0_cholesky_enabled_)
        result << ", level0 folds " <<
          (level0_fold_batch_enabled_ ? "batched" : "sequential");
      result << ", resident preprocess " <<
        (resident_preprocess_max_elements_ * sizeof(double) / 1000000.0) <<
        " MB";
      result << ", resident level1 " <<
        (level1_resident_max_elements_ * sizeof(double) / 1000000.0) <<
        " MB";
      result << ", grouped upload " <<
        (direct_grouped_upload_ ? "direct" : "materialized");
      result << ", resident fold systems " <<
        (level0_resident_folds_enabled_ ? "enabled" : "disabled");
      result << ", registered packed input " <<
        (register_packed_hardcalls_enabled_ ? "enabled" : "disabled");
      result << ")";
      return result.str();
    }

    bool can_preprocess_packed_hardcalls(
      Eigen::Index variants, Eigen::Index samples) const override {
      if(variants < 0 || samples < 0) return false;
      if(variants == 0) return true;
      if(samples == 0) return false;
      if(variants > INT_MAX || samples > INT_MAX) return false;
      const long long element_count =
        static_cast<long long>(variants) * samples;
      return element_count <= INT_MAX &&
        element_count <= resident_preprocess_max_elements_;
    }

    bool preprocess_packed_hardcalls(
      const unsigned char* packed_hardcalls,
      size_t packed_bytes,
      size_t packed_stride_bytes,
      Eigen::Index variants,
      Eigen::Index samples,
      const Eigen::Ref<const Eigen::MatrixXd>& covariates,
      const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
      double degrees_of_freedom,
      double minimum_scale,
      Eigen::VectorXd& row_scales,
      Step1ComputeTimings* timings) override {

      const ComputeClock::time_point backend_wall_start =
        ComputeClock::now();
      const ComputeClock::time_point validation_start =
        ComputeClock::now();
      const bool static_inputs_cached = packed_static_inputs_valid_ &&
        packed_static_covariates_ == covariates.data() &&
        packed_static_weights_ == sample_weights.data() &&
        packed_static_samples_ == samples &&
        packed_static_covariate_count_ == covariates.cols();
      if(static_inputs_cached) {
        if(variants < 0 || samples < 0 || covariates.rows() != samples ||
           sample_weights.size() != samples)
          throw std::invalid_argument(
            "Step 1 packed hardcall preprocessing received incompatible dimensions");
        if(!std::isfinite(degrees_of_freedom) || degrees_of_freedom <= 0)
          throw std::invalid_argument(
            "Step 1 packed hardcall preprocessing requires positive degrees of freedom");
        if(!std::isfinite(minimum_scale) || minimum_scale < 0)
          throw std::invalid_argument(
            "Step 1 packed hardcall preprocessing requires a non-negative minimum scale");
        const size_t minimum_stride =
          (static_cast<size_t>(samples) + 3) / 4;
        if(packed_stride_bytes < minimum_stride ||
           (variants > 0 && packed_stride_bytes >
             std::numeric_limits<size_t>::max() /
               static_cast<size_t>(variants)) ||
           packed_bytes < static_cast<size_t>(variants) *
             packed_stride_bytes ||
           (variants > 0 && !packed_hardcalls))
          throw std::invalid_argument(
            "Step 1 packed hardcall preprocessing received an invalid packed buffer");
      } else {
        validate_packed_hardcall_preprocessing_inputs(
          packed_hardcalls, packed_bytes, packed_stride_bytes,
          variants, samples, covariates, sample_weights,
          degrees_of_freedom, minimum_scale);
      }
      if(timings)
        timings->packed_hardcall_validation_ms +=
          elapsed_ms(validation_start);

      const ComputeClock::time_point allocation_start =
        ComputeClock::now();
      invalidate_resident_design();
      invalidate_resident_genotypes();
      if(!can_preprocess_packed_hardcalls(variants, samples)) {
        if(timings) {
          timings->packed_hardcall_allocation_ms +=
            elapsed_ms(allocation_start);
          timings->packed_hardcall_backend_wall_ms +=
            elapsed_ms(backend_wall_start);
        }
        return false;
      }
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        variants, "packed hardcall preprocessing variant count");
      const int columns = checked_int(
        samples, "packed hardcall preprocessing sample count");
      const int covariate_count = checked_int(covariates.cols(),
        "packed hardcall preprocessing covariate count");
      const int element_count = checked_element_count(
        rows, columns, "packed hardcall preprocessing block");
      const size_t required_packed_bytes =
        static_cast<size_t>(rows) * packed_stride_bytes;
      row_scales.resize(rows);
      if(rows == 0) {
        resident_rows_ = 0;
        resident_columns_ = columns;
        resident_valid_ = true;
        if(timings) {
          timings->packed_hardcall_allocation_ms +=
            elapsed_ms(allocation_start);
          timings->packed_hardcall_backend_wall_ms +=
            elapsed_ms(backend_wall_start);
        }
        return true;
      }

      ensure_capacity(d_resident_genotypes_, resident_genotypes_capacity_,
        element_count,
        "cudaMalloc(packed hardcall resident genotype block)");
      ensure_capacity(d_packed_hardcalls_, packed_hardcalls_capacity_,
        required_packed_bytes, "cudaMalloc(packed hardcall block)");
      ensure_capacity(d_transposed_hardcalls_,
        transposed_hardcalls_capacity_, required_packed_bytes,
        "cudaMalloc(transposed packed hardcall block)");
      ensure_capacity(d_packed_row_counts_, packed_row_counts_capacity_,
        rows, "cudaMalloc(packed hardcall row counts)");
      ensure_capacity(d_preprocess_weights_, preprocess_weights_capacity_,
        columns, "cudaMalloc(packed hardcall sample weights)");
      ensure_capacity(d_preprocess_scales_, preprocess_scales_capacity_,
        rows, "cudaMalloc(packed hardcall row statistics)");
      if(covariate_count > 0) {
        ensure_capacity(d_preprocess_covariates_,
          preprocess_covariates_capacity_, covariates.size(),
          "cudaMalloc(packed hardcall covariates)");
        ensure_capacity(d_preprocess_coefficients_,
          preprocess_coefficients_capacity_,
          static_cast<Eigen::Index>(rows) * covariate_count,
          "cudaMalloc(packed hardcall projection coefficients)");
      }
      if(timings)
        timings->packed_hardcall_allocation_ms +=
          elapsed_ms(allocation_start);

      const ComputeClock::time_point host_prepare_start =
        ComputeClock::now();
      const Eigen::MatrixXd packed_covariates = covariate_count > 0 ?
        contiguous_copy_if_needed(covariates) : Eigen::MatrixXd();
      const double* covariate_data = packed_covariates.size() ?
        packed_covariates.data() : covariates.data();
      if(timings)
        timings->packed_hardcall_host_prepare_ms +=
          elapsed_ms(host_prepare_start);
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(cuda_host_pointer_is_registered(packed_hardcalls)) {
        check_cuda(cudaMemcpy(d_packed_hardcalls_, packed_hardcalls,
          required_packed_bytes, cudaMemcpyHostToDevice),
          "copy registered packed hardcalls to CUDA device");
        if(timings) {
          timings->registered_packed_upload_count++;
          timings->registered_packed_upload_bytes += required_packed_bytes;
        }
      } else {
        copy_host_to_device_staged(d_packed_hardcalls_, packed_hardcalls,
          required_packed_bytes,
          "copy packed hardcalls to CUDA device", timings);
      }
      if(!static_inputs_cached) {
        check_cuda(cudaMemcpy(d_preprocess_weights_, sample_weights.data(),
          static_cast<size_t>(columns) * sizeof(double),
          cudaMemcpyHostToDevice),
          "copy packed hardcall sample weights to CUDA device");
        if(covariate_count > 0)
          check_cuda(cudaMemcpy(d_preprocess_covariates_, covariate_data,
            covariates.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy packed hardcall covariates to CUDA device");
        packed_static_covariates_ = covariates.data();
        packed_static_weights_ = sample_weights.data();
        packed_static_samples_ = samples;
        packed_static_covariate_count_ = covariates.cols();
        packed_static_inputs_valid_ = true;
      }
      if(timings) {
        timings->upload_ms += elapsed_ms(transfer_start);
        timings->packed_hardcall_upload_count++;
        timings->packed_hardcall_upload_bytes += required_packed_bytes;
      }

      const int threads = 256;
      std::unique_ptr<CudaEventPair> expand_events;
      if(timings) {
        expand_events.reset(new CudaEventPair());
        expand_events->record_start();
      }
      packed_hardcall_row_statistics<<<rows, threads>>>(
        d_packed_hardcalls_, packed_stride_bytes, d_preprocess_weights_,
        d_preprocess_scales_, d_packed_row_counts_, rows, columns);
      check_cuda(cudaGetLastError(),
        "compute packed hardcall row statistics kernel");
      const int packed_columns = (columns + 3) / 4;
      const dim3 transpose_threads(32, 8);
      const dim3 transpose_grid(
        (packed_columns + 31) / 32, (rows + 31) / 32);
      transpose_packed_hardcalls<<<transpose_grid, transpose_threads>>>(
        d_packed_hardcalls_, d_transposed_hardcalls_, rows,
        packed_columns, packed_stride_bytes);
      check_cuda(cudaGetLastError(),
        "transpose packed hardcalls kernel");
      const int element_blocks = (element_count - 1) / threads + 1;
      expand_packed_hardcalls<<<element_blocks, threads>>>(
        d_transposed_hardcalls_, d_preprocess_weights_,
        d_preprocess_scales_, d_packed_row_counts_,
        d_resident_genotypes_, rows, element_count);
      check_cuda(cudaGetLastError(), "expand packed hardcalls kernel");
      if(timings) {
        const double expand_ms =
          expand_events->record_stop_and_elapsed_ms();
        timings->packed_hardcall_expand_ms += expand_ms;
        timings->preprocess_ms += expand_ms;
      }

      std::unique_ptr<CudaEventPair> projection_events;
      if(timings) {
        projection_events.reset(new CudaEventPair());
        projection_events->record_start();
      }
      if(covariate_count > 0) {
        const double one = 1.0;
        const double zero = 0.0;
        const double minus_one = -1.0;
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
          rows, covariate_count, columns, &one,
          d_resident_genotypes_, rows,
          d_preprocess_covariates_, columns, &zero,
          d_preprocess_coefficients_, rows),
          "cublasDgemm(packed hardcall projection coefficients)");
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
          rows, columns, covariate_count, &minus_one,
          d_preprocess_coefficients_, rows,
          d_preprocess_covariates_, columns, &one,
          d_resident_genotypes_, rows),
          "cublasDgemm(packed hardcall residuals)");
      }
      compute_genotype_row_scales<<<rows, threads,
        threads * sizeof(double)>>>(d_resident_genotypes_,
        d_preprocess_scales_, rows, columns, degrees_of_freedom);
      check_cuda(cudaGetLastError(),
        "compute packed hardcall row scales kernel");
      if(timings)
        timings->preprocess_ms +=
          projection_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(row_scales.data(), d_preprocess_scales_,
        static_cast<size_t>(rows) * sizeof(double), cudaMemcpyDeviceToHost),
        "copy packed hardcall row scales from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
      if(!row_scales.allFinite())
        throw std::runtime_error(
          "CUDA packed hardcall preprocessing produced non-finite row scales");
      if(row_scales.minCoeff() < minimum_scale) {
        if(timings)
          timings->packed_hardcall_backend_wall_ms +=
            elapsed_ms(backend_wall_start);
        return true;
      }

      std::unique_ptr<CudaEventPair> scale_events;
      if(timings) {
        scale_events.reset(new CudaEventPair());
        scale_events->record_start();
      }
      scale_genotype_rows<<<element_blocks, threads>>>(
        d_resident_genotypes_, d_preprocess_scales_, nullptr,
        rows, element_count);
      check_cuda(cudaGetLastError(),
        "scale packed hardcall genotype rows kernel");
      if(timings)
        timings->preprocess_ms +=
          scale_events->record_stop_and_elapsed_ms();

      resident_host_data_ = nullptr;
      resident_rows_ = variants;
      resident_columns_ = samples;
      resident_valid_ = true;
      if(timings)
        timings->packed_hardcall_backend_wall_ms +=
          elapsed_ms(backend_wall_start);
      return true;
    }

    bool preprocess_genotypes(
      Eigen::MatrixXd& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& covariates,
      const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
      double degrees_of_freedom,
      double minimum_scale,
      const Eigen::Ref<const Eigen::VectorXd>& row_multipliers,
      bool copy_to_host,
      Eigen::VectorXd& row_scales,
      Step1ComputeTimings* timings) override {

      packed_static_inputs_valid_ = false;
      Step1ComputeBackend::preprocess_genotypes(genotypes, covariates,
        sample_weights, degrees_of_freedom, minimum_scale,
        row_multipliers, copy_to_host, row_scales, timings);
      invalidate_resident_design();
      invalidate_resident_genotypes();
      const long long required_elements_long =
        static_cast<long long>(genotypes.rows()) * genotypes.cols();
      if(required_elements_long > INT_MAX ||
         required_elements_long > resident_preprocess_max_elements_)
        return false;

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(genotypes.rows(),
        "genotype preprocessing row count");
      const int columns = checked_int(genotypes.cols(),
        "genotype preprocessing sample count");
      const int covariate_count = checked_int(covariates.cols(),
        "genotype preprocessing covariate count");
      const int element_count = static_cast<int>(required_elements_long);
      row_scales.resize(rows);
      if(rows == 0) return true;
      if(columns == 0) return false;

      ensure_capacity(d_resident_genotypes_, resident_genotypes_capacity_,
        element_count,
        "cudaMalloc(resident genotype preprocessing block)");
      ensure_capacity(d_preprocess_weights_, preprocess_weights_capacity_,
        columns, "cudaMalloc(genotype preprocessing sample weights)");
      ensure_capacity(d_preprocess_scales_, preprocess_scales_capacity_,
        rows, "cudaMalloc(genotype preprocessing row scales)");
      if(covariate_count > 0) {
        ensure_capacity(d_preprocess_covariates_,
          preprocess_covariates_capacity_, covariates.size(),
          "cudaMalloc(genotype preprocessing covariates)");
        ensure_capacity(d_preprocess_coefficients_,
          preprocess_coefficients_capacity_,
          static_cast<Eigen::Index>(rows) * covariate_count,
          "cudaMalloc(genotype preprocessing coefficients)");
      }
      if(row_multipliers.size() > 0)
        ensure_capacity(d_preprocess_multipliers_,
          preprocess_multipliers_capacity_, rows,
          "cudaMalloc(genotype preprocessing row multipliers)");

      const Eigen::MatrixXd packed_covariates = covariate_count > 0 ?
        contiguous_copy_if_needed(covariates) : Eigen::MatrixXd();
      const double* covariate_data = packed_covariates.size() ?
        packed_covariates.data() : covariates.data();
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      copy_host_to_device_staged(d_resident_genotypes_, genotypes.data(),
        static_cast<size_t>(element_count) * sizeof(double),
        "copy resident genotype preprocessing block to CUDA device",
        timings);
      check_cuda(cudaMemcpy(d_preprocess_weights_, sample_weights.data(),
        static_cast<size_t>(columns) * sizeof(double),
        cudaMemcpyHostToDevice),
        "copy genotype preprocessing sample weights to CUDA device");
      if(covariate_count > 0)
        check_cuda(cudaMemcpy(d_preprocess_covariates_, covariate_data,
          covariates.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy genotype preprocessing covariates to CUDA device");
      if(row_multipliers.size() > 0)
        check_cuda(cudaMemcpy(d_preprocess_multipliers_,
          row_multipliers.data(), row_multipliers.size() * sizeof(double),
          cudaMemcpyHostToDevice),
          "copy genotype preprocessing row multipliers to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      std::unique_ptr<CudaEventPair> projection_events;
      if(timings) {
        projection_events.reset(new CudaEventPair());
        projection_events->record_start();
      }
      const int threads = 256;
      const int element_blocks = (element_count - 1) / threads + 1;
      mask_genotype_columns<<<element_blocks, threads>>>(
        d_resident_genotypes_, d_preprocess_weights_, rows, element_count);
      check_cuda(cudaGetLastError(),
        "mask genotype preprocessing columns kernel");
      if(covariate_count > 0) {
        const double one = 1.0;
        const double zero = 0.0;
        const double minus_one = -1.0;
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
          rows, covariate_count, columns, &one,
          d_resident_genotypes_, rows,
          d_preprocess_covariates_, columns, &zero,
          d_preprocess_coefficients_, rows),
          "cublasDgemm(genotype preprocessing projection coefficients)");
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
          rows, columns, covariate_count, &minus_one,
          d_preprocess_coefficients_, rows,
          d_preprocess_covariates_, columns, &one,
          d_resident_genotypes_, rows),
          "cublasDgemm(genotype preprocessing residuals)");
      }
      compute_genotype_row_scales<<<rows, threads,
        threads * sizeof(double)>>>(d_resident_genotypes_,
        d_preprocess_scales_, rows, columns, degrees_of_freedom);
      check_cuda(cudaGetLastError(),
        "compute genotype preprocessing row scales kernel");
      if(timings)
        timings->preprocess_ms +=
          projection_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(row_scales.data(), d_preprocess_scales_,
        static_cast<size_t>(rows) * sizeof(double), cudaMemcpyDeviceToHost),
        "copy genotype preprocessing row scales from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
      if(!row_scales.allFinite())
        throw std::runtime_error(
          "CUDA genotype preprocessing produced non-finite row scales");
      if(row_scales.minCoeff() < minimum_scale)
        return true;

      std::unique_ptr<CudaEventPair> scale_events;
      if(timings) {
        scale_events.reset(new CudaEventPair());
        scale_events->record_start();
      }
      scale_genotype_rows<<<element_blocks, threads>>>(
        d_resident_genotypes_, d_preprocess_scales_,
        row_multipliers.size() ? d_preprocess_multipliers_ : nullptr,
        rows, element_count);
      check_cuda(cudaGetLastError(),
        "scale genotype preprocessing rows kernel");
      if(timings)
        timings->preprocess_ms +=
          scale_events->record_stop_and_elapsed_ms();

      if(copy_to_host) {
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(genotypes.data(), d_resident_genotypes_,
          static_cast<size_t>(element_count) * sizeof(double),
          cudaMemcpyDeviceToHost),
          "copy normalized genotype preprocessing block from CUDA device");
        if(timings) timings->download_ms += elapsed_ms(transfer_start);
      }
      resident_host_data_ = genotypes.data();
      resident_rows_ = genotypes.rows();
      resident_columns_ = genotypes.cols();
      resident_valid_ = true;
      return true;
    }

    bool register_packed_hardcall_buffer(
      unsigned char* buffer, size_t bytes) override {
      if(!register_packed_hardcalls_enabled_ || !buffer || bytes == 0)
        return false;
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      for(const auto& registration : registered_packed_hardcall_buffers_)
        if(registration.first == buffer && registration.second >= bytes)
          return true;
      const cudaError_t status = cudaHostRegister(
        buffer, bytes, cudaHostRegisterPortable);
      if(status == cudaSuccess) {
        registered_packed_hardcall_buffers_.push_back(
          std::make_pair(buffer, bytes));
        return true;
      }
      if(status == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        return true;
      }
      cudaGetLastError();
      return false;
    }

    void release_packed_hardcall_buffers() override {
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      for(const auto& registration : registered_packed_hardcall_buffers_)
        check_cuda(cudaHostUnregister(registration.first),
          "cudaHostUnregister(packed hardcall buffer)");
      registered_packed_hardcall_buffers_.clear();
    }

    void release_preprocessed_genotypes() override {
      invalidate_resident_genotypes();
    }

    bool cache_design_partitions(
      const std::vector<Eigen::MatrixXd>& partitions,
      Step1ComputeTimings* timings) override {

      invalidate_resident_design();
      invalidate_resident_genotypes();
      if(partitions.empty()) return false;

      const Eigen::Index columns = partitions.front().cols();
      Eigen::Index rows = 0;
      for(const Eigen::MatrixXd& partition : partitions) {
        if(partition.cols() != columns)
          throw std::invalid_argument(
            "Step 1 cached design partitions have inconsistent columns");
        if(partition.rows() > std::numeric_limits<Eigen::Index>::max() - rows)
          throw std::runtime_error(
            "Step 1 cached design row count overflows Eigen dimensions");
        rows += partition.rows();
      }
      const long long required_elements_long =
        static_cast<long long>(rows) * columns;
      if(required_elements_long < 0 || required_elements_long > INT_MAX ||
         required_elements_long > level1_resident_max_elements_)
        return false;

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const Eigen::Index required_elements =
        static_cast<Eigen::Index>(required_elements_long);
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
      (void)total_bytes;
      const size_t resident_growth = required_elements >
        static_cast<Eigen::Index>(resident_genotypes_capacity_) ?
        static_cast<size_t>(required_elements -
          static_cast<Eigen::Index>(resident_genotypes_capacity_)) *
            sizeof(double) : 0;
      const size_t workspace_growth = required_elements >
        static_cast<Eigen::Index>(projected_capacity_) ?
        static_cast<size_t>(required_elements -
          static_cast<Eigen::Index>(projected_capacity_)) *
            sizeof(double) : 0;
      const size_t reserve_bytes = size_t(512) * 1000000;
      if(resident_growth > free_bytes ||
         workspace_growth > free_bytes - resident_growth ||
         reserve_bytes > free_bytes - resident_growth - workspace_growth)
        return false;

      ensure_capacity(d_resident_genotypes_, resident_genotypes_capacity_,
        required_elements, "cudaMalloc(resident Level 1 design)");
      ensure_capacity(d_projected_, projected_capacity_, required_elements,
        "cudaMalloc(resident Level 1 weighted-design workspace)");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      Eigen::Index row_offset = 0;
      for(const Eigen::MatrixXd& partition : partitions) {
        if(partition.rows() > 0 && columns > 0) {
          const size_t row_bytes =
            static_cast<size_t>(partition.rows()) * sizeof(double);
          check_cuda(cudaMemcpy2D(
            d_resident_genotypes_ + row_offset,
            static_cast<size_t>(rows) * sizeof(double),
            partition.data(),
            static_cast<size_t>(partition.outerStride()) * sizeof(double),
            row_bytes, static_cast<size_t>(columns),
            cudaMemcpyHostToDevice),
            "copy Level 1 design partition to CUDA device");
        }
        row_offset += partition.rows();
      }
      if(timings) {
        timings->upload_ms += elapsed_ms(transfer_start);
        timings->resident_design_upload_count += partitions.size();
        timings->resident_design_upload_bytes +=
          static_cast<uint64_t>(required_elements) * sizeof(double);
      }
      resident_design_rows_ = rows;
      resident_design_columns_ = columns;
      resident_design_valid_ = true;
      return true;
    }

    bool cache_design_matrix(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      Step1ComputeTimings* timings) override {

      invalidate_resident_design();
      invalidate_resident_genotypes();
      const Eigen::Index rows = design.rows();
      const Eigen::Index columns = design.cols();
      const long long required_elements_long =
        static_cast<long long>(rows) * columns;
      if(required_elements_long < 0 || required_elements_long > INT_MAX ||
         required_elements_long > level1_resident_max_elements_)
        return false;

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const Eigen::Index required_elements =
        static_cast<Eigen::Index>(required_elements_long);
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
      (void)total_bytes;
      const size_t resident_growth = required_elements >
        static_cast<Eigen::Index>(resident_genotypes_capacity_) ?
        static_cast<size_t>(required_elements -
          static_cast<Eigen::Index>(resident_genotypes_capacity_)) *
            sizeof(double) : 0;
      const size_t workspace_growth = required_elements >
        static_cast<Eigen::Index>(projected_capacity_) ?
        static_cast<size_t>(required_elements -
          static_cast<Eigen::Index>(projected_capacity_)) *
            sizeof(double) : 0;
      const size_t reserve_bytes = size_t(512) * 1000000;
      if(resident_growth > free_bytes ||
         workspace_growth > free_bytes - resident_growth ||
         reserve_bytes > free_bytes - resident_growth - workspace_growth)
        return false;

      ensure_capacity(d_resident_genotypes_, resident_genotypes_capacity_,
        required_elements, "cudaMalloc(resident Level 1 design)");
      ensure_capacity(d_projected_, projected_capacity_, required_elements,
        "cudaMalloc(resident Level 1 weighted-design workspace)");

      const Eigen::MatrixXd packed_design =
        contiguous_copy_if_needed(design);
      const double* design_data = packed_design.size() ?
        packed_design.data() : design.data();
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(required_elements > 0)
        copy_host_to_device_staged(d_resident_genotypes_, design_data,
          static_cast<size_t>(required_elements) * sizeof(double),
          "copy Level 1 design matrix to CUDA device", timings);
      if(timings) {
        timings->upload_ms += elapsed_ms(transfer_start);
        timings->resident_design_upload_count++;
        timings->resident_design_upload_bytes +=
          static_cast<uint64_t>(required_elements) * sizeof(double);
      }
      resident_design_rows_ = rows;
      resident_design_columns_ = columns;
      resident_design_valid_ = true;
      return true;
    }

    bool initialize_level1_design_cache(
      Eigen::Index rows, Eigen::Index columns) override {

      release_level1_design_cache();
      if(rows <= 0 || columns <= 0) return false;
      const long long required_elements_long =
        static_cast<long long>(rows) * columns;
      if(required_elements_long <= 0 ||
         required_elements_long > INT_MAX ||
         required_elements_long > level1_resident_max_elements_)
        return false;

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
      (void)total_bytes;
      const size_t required_bytes =
        static_cast<size_t>(required_elements_long) * sizeof(double);
      const size_t reserve_bytes = size_t(6000) * 1000000;
      if(required_bytes > free_bytes ||
         reserve_bytes > free_bytes - required_bytes)
        return false;

      ensure_capacity(d_level1_design_, level1_design_capacity_,
        static_cast<Eigen::Index>(required_elements_long),
        "cudaMalloc(persistent Level 1 design)");
      level1_design_rows_ = rows;
      level1_design_columns_ = columns;
      level1_design_cached_columns_ = 0;
      return true;
    }

    void append_level1_design_cache(
      Eigen::Index start_column,
      const Eigen::Ref<const Eigen::MatrixXd>& columns,
      Step1ComputeTimings* timings) override {

      if(!d_level1_design_ || level1_design_rows_ <= 0 ||
         columns.rows() != level1_design_rows_ ||
         start_column != level1_design_cached_columns_ ||
         start_column < 0 || columns.cols() < 0 ||
         start_column > level1_design_columns_ - columns.cols())
        throw std::invalid_argument(
          "Step 1 persistent Level 1 design append is out of order or has incompatible dimensions");
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(columns.size() > 0)
        check_cuda(cudaMemcpy(
          d_level1_design_ + start_column * level1_design_rows_,
          columns.data(),
          static_cast<size_t>(columns.size()) * sizeof(double),
          cudaMemcpyHostToDevice),
          "append persistent Level 1 design columns to CUDA device");
      level1_design_cached_columns_ += columns.cols();
      if(timings) {
        timings->upload_ms += elapsed_ms(transfer_start);
        timings->resident_design_upload_count++;
        timings->resident_design_upload_bytes +=
          static_cast<uint64_t>(columns.size()) * sizeof(double);
      }
    }

    bool activate_level1_design_cache(
      Eigen::Index rows, Eigen::Index columns) override {

      if(!d_level1_design_ || rows != level1_design_rows_ ||
         columns != level1_design_columns_ ||
         level1_design_cached_columns_ != level1_design_columns_)
        return false;
      invalidate_resident_design();
      invalidate_resident_genotypes();
      resident_design_rows_ = rows;
      resident_design_columns_ = columns;
      resident_design_valid_ = true;
      resident_design_uses_level1_cache_ = true;
      return true;
    }

    void release_level1_design_cache() override {
      if(resident_design_uses_level1_cache_)
        invalidate_resident_design();
      if(d_level1_design_) {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        check_cuda(cudaFree(d_level1_design_),
          "cudaFree(persistent Level 1 design)");
      }
      d_level1_design_ = nullptr;
      level1_design_capacity_ = 0;
      level1_design_rows_ = 0;
      level1_design_columns_ = 0;
      level1_design_cached_columns_ = 0;
    }

    void predict_cached_design(
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      Eigen::VectorXd& predictions,
      Step1ComputeTimings* timings) override {

      if(!resident_design_valid_ ||
         coefficients.size() != resident_design_columns_)
        throw std::invalid_argument(
          "Step 1 cached design prediction received incompatible dimensions");
      if(!coefficients.allFinite())
        throw std::invalid_argument(
          "Step 1 cached design prediction requires finite coefficients");
      predictions.resize(resident_design_rows_);
      if(resident_design_rows_ == 0) return;

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        resident_design_rows_, "cached design prediction row count");
      const int columns = checked_int(
        resident_design_columns_, "cached design prediction column count");
      ensure_capacity(d_inverse_, inverse_capacity_, coefficients.size(),
        "cudaMalloc(cached design coefficients)");
      ensure_capacity(d_predictions_, predictions_capacity_, predictions.size(),
        "cudaMalloc(cached design predictions)");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(coefficients.size() > 0)
        check_cuda(cudaMemcpy(d_inverse_, coefficients.data(),
          static_cast<size_t>(coefficients.size()) * sizeof(double),
          cudaMemcpyHostToDevice),
          "copy cached design coefficients to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      std::unique_ptr<CudaEventPair> prediction_events;
      if(timings) {
        prediction_events.reset(new CudaEventPair());
        prediction_events->record_start();
      }
      if(columns > 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        check_cublas(cublasDgemv(handle_, CUBLAS_OP_N,
          rows, columns, &alpha, resident_design_data(), rows,
          d_inverse_, 1, &beta, d_predictions_, 1),
          "cublasDgemv(cached design prediction)");
      } else {
        check_cuda(cudaMemset(d_predictions_, 0,
          static_cast<size_t>(rows) * sizeof(double)),
          "clear empty cached design prediction");
      }
      if(timings)
        timings->ridge_ms +=
          prediction_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(predictions.data(), d_predictions_,
        static_cast<size_t>(rows) * sizeof(double),
        cudaMemcpyDeviceToHost),
        "copy cached design predictions from CUDA device");
      if(timings) {
        timings->download_ms += elapsed_ms(transfer_start);
        timings->resident_design_reuse_count++;
      }
    }

    void compute_cached_weighted_design_products(
      const Eigen::Ref<const Eigen::VectorXd>& weights,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings) override {

      if(!resident_design_valid_ || weights.size() != resident_design_rows_ ||
         outcomes.rows() != resident_design_rows_)
        throw std::invalid_argument(
          "Step 1 cached weighted design products received incompatible dimensions");
      if(!weights.allFinite() || (weights.array() < 0).any() ||
         !outcomes.allFinite())
        throw std::invalid_argument(
          "Step 1 cached weighted design products require finite inputs and non-negative weights");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        resident_design_rows_, "cached weighted design row count");
      const int features = checked_int(
        resident_design_columns_, "cached weighted design feature count");
      const int outcome_count = checked_int(
        outcomes.cols(), "cached weighted design outcome count");
      gram.resize(features, features);
      crossproduct.resize(features, outcome_count);
      if(features == 0 || rows == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }

      compute_cached_weighted_design_products_device(
        weights, outcomes, timings);
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(outcome_count > 0)
        check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_,
          static_cast<size_t>(crossproduct.size()) * sizeof(double),
          cudaMemcpyDeviceToHost),
          "copy cached weighted crossproduct from CUDA device");
      check_cuda(cudaMemcpy(gram.data(), d_gram_,
        static_cast<size_t>(gram.size()) * sizeof(double),
        cudaMemcpyDeviceToHost),
        "copy cached weighted Gram matrix from CUDA device");
      if(timings) {
        timings->download_ms += elapsed_ms(transfer_start);
      }
    }

    bool solve_cached_weighted_design(
      const Eigen::Ref<const Eigen::VectorXd>& weights,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings) override {

      if(!resident_design_valid_ || weights.size() != resident_design_rows_ ||
         outcomes.rows() != resident_design_rows_ ||
         penalty_multipliers.size() != resident_design_columns_)
        throw std::invalid_argument(
          "Step 1 cached weighted solve received incompatible dimensions");
      if(!weights.allFinite() || (weights.array() < 0).any() ||
         !outcomes.allFinite() || !ridge_parameters.allFinite() ||
         (ridge_parameters.array() < 0).any() ||
         !penalty_multipliers.allFinite() ||
         (penalty_multipliers.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 cached weighted solve requires finite inputs and non-negative weights and penalties");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        resident_design_rows_, "cached weighted solve row count");
      const int features = checked_int(
        resident_design_columns_, "cached weighted solve feature count");
      const int outcome_count = checked_int(
        outcomes.cols(), "cached weighted solve outcome count");
      const int parameter_count = checked_int(
        ridge_parameters.size(), "cached weighted solve parameter count");
      const long long solution_column_count_long =
        static_cast<long long>(outcome_count) * parameter_count;
      if(solution_column_count_long > INT_MAX)
        throw std::runtime_error(
          "CUDA cached weighted solution column count exceeds integer limits");
      solutions.resize(features,
        static_cast<Eigen::Index>(solution_column_count_long));
      if(features == 0 || outcome_count == 0 || parameter_count == 0) {
        solutions.setZero();
        return true;
      }
      if(rows == 0) return false;

      compute_cached_weighted_design_products_device(
        weights, outcomes, timings);
      diagonal_penalty_solve_device(features, outcome_count,
        ridge_parameters, penalty_multipliers, solutions, timings);
      return true;
    }

    void compute_cached_design_crossproduct(
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings) override {

      if(!resident_design_valid_ ||
         outcomes.rows() != resident_design_rows_)
        throw std::invalid_argument(
          "Step 1 cached design crossproduct received incompatible dimensions");
      if(!outcomes.allFinite())
        throw std::invalid_argument(
          "Step 1 cached design crossproduct requires finite outcomes");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        resident_design_rows_, "cached crossproduct row count");
      const int features = checked_int(
        resident_design_columns_, "cached crossproduct feature count");
      const int outcome_count = checked_int(
        outcomes.cols(), "cached crossproduct outcome count");
      crossproduct.resize(features, outcome_count);
      if(features == 0 || outcome_count == 0 || rows == 0) {
        crossproduct.setZero();
        return;
      }

      ensure_capacity(d_phenotypes_, phenotypes_capacity_, outcomes.size(),
        "cudaMalloc(cached crossproduct outcomes)");
      ensure_capacity(d_crossproduct_, crossproduct_capacity_,
        crossproduct.size(), "cudaMalloc(cached design crossproduct)");
      const Eigen::MatrixXd packed_outcomes =
        contiguous_copy_if_needed(outcomes);
      const double* outcome_data = packed_outcomes.size() ?
        packed_outcomes.data() : outcomes.data();
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_phenotypes_, outcome_data,
        static_cast<size_t>(outcomes.size()) * sizeof(double),
        cudaMemcpyHostToDevice),
        "copy cached crossproduct outcomes to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      std::unique_ptr<CudaEventPair> crossproduct_events;
      if(timings) {
        crossproduct_events.reset(new CudaEventPair());
        crossproduct_events->record_start();
      }
      const double alpha = 1.0;
      const double beta = 0.0;
      check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        features, outcome_count, rows, &alpha,
        resident_design_data(), rows, d_phenotypes_, rows, &beta,
        d_crossproduct_, features),
        "cublasDgemm(cached design crossproduct)");
      if(timings)
        timings->crossproduct_ms +=
          crossproduct_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_,
        static_cast<size_t>(crossproduct.size()) * sizeof(double),
        cudaMemcpyDeviceToHost),
        "copy cached design crossproduct from CUDA device");
      if(timings) {
        timings->download_ms += elapsed_ms(transfer_start);
        timings->resident_design_reuse_count++;
      }
    }

    void release_cached_design() override {
      invalidate_resident_design();
    }

    void compute_preprocessed_products(
      Eigen::Index start_column,
      Eigen::Index column_count,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1GramMode mode,
      Step1ComputeTimings* timings) override {

      if(!resident_valid_ || start_column < 0 || column_count < 0 ||
         start_column > resident_columns_ - column_count ||
         phenotypes.rows() != column_count)
        throw std::invalid_argument(
          "Step 1 resident genotype products received incompatible dimensions");
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        resident_rows_, "resident genotype product row count");
      const int columns = checked_int(
        column_count, "resident genotype product sample count");
      const int phenotype_count = checked_int(
        phenotypes.cols(), "resident genotype product phenotype count");
      gram.resize(rows, rows);
      crossproduct.resize(rows, phenotype_count);
      if(rows == 0 || columns == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }

      ensure_capacity(d_gram_, gram_capacity_, gram.size(),
        "cudaMalloc(resident genotype Gram matrix)");
      if(phenotype_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_,
          phenotypes.size(),
          "cudaMalloc(resident genotype phenotype block)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_,
          crossproduct.size(),
          "cudaMalloc(resident genotype crossproduct)");
      }

      const Eigen::MatrixXd packed_phenotypes = phenotype_count > 0 ?
        contiguous_copy_if_needed(phenotypes) : Eigen::MatrixXd();
      const double* phenotype_data = packed_phenotypes.size() ?
        packed_phenotypes.data() : phenotypes.data();
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(phenotype_count > 0)
        check_cuda(cudaMemcpy(d_phenotypes_, phenotype_data,
          phenotypes.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy resident genotype phenotypes to CUDA device");
      if(timings) {
        timings->upload_ms += elapsed_ms(transfer_start);
        timings->resident_reuse_count++;
      }

      const double* device_genotypes = d_resident_genotypes_ +
        start_column * resident_rows_;
      const double alpha = 1.0;
      const double beta = 0.0;
      if(phenotype_count > 0) {
        std::unique_ptr<CudaEventPair> crossproduct_events;
        if(timings) {
          crossproduct_events.reset(new CudaEventPair());
          crossproduct_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
          rows, phenotype_count, columns, &alpha,
          device_genotypes, rows, d_phenotypes_, columns, &beta,
          d_crossproduct_, rows),
          "cublasDgemm(resident genotype product)");
        if(timings)
          timings->crossproduct_ms +=
            crossproduct_events->record_stop_and_elapsed_ms();
      }

      std::unique_ptr<CudaEventPair> gram_events;
      if(timings) {
        gram_events.reset(new CudaEventPair());
        gram_events->record_start();
      }
      if(mode == Step1GramMode::selfadjoint_rank_update)
        check_cublas(cublasDsyrk(handle_, CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_N, rows, columns, &alpha,
          device_genotypes, rows, &beta, d_gram_, rows),
          "cublasDsyrk(resident genotype Gram product)");
      else
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
          rows, rows, columns, &alpha,
          device_genotypes, rows, device_genotypes, rows, &beta,
          d_gram_, rows),
          "cublasDgemm(resident genotype Gram product)");
      if(timings)
        timings->gram_ms += gram_events->record_stop_and_elapsed_ms();
      if(mode == Step1GramMode::selfadjoint_rank_update) {
        const dim3 threads(16, 16);
        const dim3 grid((rows + threads.x - 1) / threads.x,
          (rows + threads.y - 1) / threads.y);
        mirror_lower_triangle<<<grid, threads>>>(d_gram_, rows);
        check_cuda(cudaGetLastError(),
          "mirror resident genotype Gram triangle kernel");
      }

      if(timings) transfer_start = ComputeClock::now();
      if(phenotype_count > 0)
        check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_,
          crossproduct.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "copy resident genotype crossproduct from CUDA device");
      check_cuda(cudaMemcpy(gram.data(), d_gram_,
        gram.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy resident genotype Gram matrix from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

    bool cache_preprocessed_fold_systems(
      const Eigen::Ref<const Eigen::VectorXi>& start_columns,
      const Eigen::Ref<const Eigen::VectorXi>& column_counts,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Step1ComputeTimings* timings) override {

      invalidate_resident_fold_systems();
      const Eigen::Index system_count_index = start_columns.size();
      if(!level0_resident_folds_enabled_ || !level0_cholesky_enabled_ ||
         !level0_fold_batch_enabled_ || system_count_index < 2)
        return false;
      if(!resident_valid_ || column_counts.size() != system_count_index ||
         phenotypes.rows() != resident_columns_)
        throw std::invalid_argument(
          "Step 1 resident fold products received incompatible dimensions");
      for(Eigen::Index system = 0; system < system_count_index; ++system) {
        if(start_columns(system) < 0 || column_counts(system) < 0 ||
           start_columns(system) > resident_columns_ - column_counts(system))
          throw std::invalid_argument(
            "Step 1 resident fold products received an invalid fold");
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const size_t system_count = static_cast<size_t>(system_count_index);
      const int rows = checked_int(
        resident_rows_, "resident fold product row count");
      const int phenotype_count = checked_int(
        phenotypes.cols(), "resident fold product phenotype count");
      const Eigen::Index gram_elements = resident_rows_ * resident_rows_;
      const Eigen::Index rhs_elements = resident_rows_ * phenotypes.cols();
      ensure_level0_cholesky_lane_count(system_count);
      ensure_capacity(d_gram_, gram_capacity_, gram_elements,
        "cudaMalloc(resident fold total Gram matrix)");
      if(rhs_elements > 0)
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, rhs_elements,
          "cudaMalloc(resident fold total right-hand sides)");

      const bool cache_full_phenotypes = phenotypes.innerStride() == 1 &&
        phenotypes.outerStride() == phenotypes.rows();
      const bool reuse_cached_phenotypes = cache_full_phenotypes &&
        level0_phenotypes_host_ == phenotypes.data() &&
        level0_phenotype_rows_ == phenotypes.rows() &&
        level0_phenotype_columns_ == phenotypes.cols();
      if(cache_full_phenotypes && !reuse_cached_phenotypes)
        ensure_capacity(d_level0_phenotypes_, level0_phenotypes_capacity_,
          phenotypes.size(), "cudaMalloc(static Level 0 phenotypes)");
      std::vector<Eigen::MatrixXd> packed_phenotypes(
        cache_full_phenotypes ? 0 : system_count);
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        const Eigen::Index fold = static_cast<Eigen::Index>(system);
        const Eigen::Index sample_count = column_counts(fold);
        ensure_capacity(lane.gram, lane.gram_capacity, gram_elements,
          "cudaMalloc(resident fold Gram matrix)");
        ensure_capacity(lane.right_hand_sides,
          lane.right_hand_sides_capacity, rhs_elements,
          "cudaMalloc(resident fold right-hand sides)");
        if(phenotype_count > 0 && !cache_full_phenotypes) {
          packed_phenotypes[system] = phenotypes.middleRows(
            start_columns(fold), sample_count);
          ensure_capacity(lane.predictions, lane.predictions_capacity,
            packed_phenotypes[system].size(),
            "cudaMalloc(resident fold phenotype staging)");
        }
      }

      ComputeClock::time_point phase_start;
      if(timings) phase_start = ComputeClock::now();
      if(cache_full_phenotypes && !reuse_cached_phenotypes &&
         phenotypes.size() > 0) {
        check_cuda(cudaMemcpy(d_level0_phenotypes_, phenotypes.data(),
          static_cast<size_t>(phenotypes.size()) * sizeof(double),
          cudaMemcpyHostToDevice),
          "copy static Level 0 phenotypes to CUDA device");
        level0_phenotypes_host_ = phenotypes.data();
        level0_phenotype_rows_ = phenotypes.rows();
        level0_phenotype_columns_ = phenotypes.cols();
      } else if(!cache_full_phenotypes) {
        for(size_t system = 0; system < system_count; ++system) {
          if(phenotype_count == 0) continue;
          CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
          check_cuda(cudaMemcpyAsync(lane.predictions,
            packed_phenotypes[system].data(),
            packed_phenotypes[system].size() * sizeof(double),
            cudaMemcpyHostToDevice, lane.stream),
            "copy resident fold phenotypes to CUDA device");
        }
        synchronize_level0_cholesky_lanes(system_count);
      }
      if(timings) timings->upload_ms += elapsed_ms(phase_start);

      if(timings) phase_start = ComputeClock::now();
      const double alpha = 1.0;
      const double beta = 0.0;
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        const Eigen::Index fold = static_cast<Eigen::Index>(system);
        const int start = checked_int(start_columns(fold),
          "resident fold product start column");
        const int samples = checked_int(column_counts(fold),
          "resident fold product sample count");
        const double* device_genotypes = d_resident_genotypes_ +
          static_cast<Eigen::Index>(start) * resident_rows_;
        const double* device_fold_phenotypes = phenotype_count == 0 ?
          nullptr : (cache_full_phenotypes ?
            d_level0_phenotypes_ + start : lane.predictions);
        const int phenotype_leading_dimension = cache_full_phenotypes ?
          checked_int(phenotypes.rows(),
            "static Level 0 phenotype leading dimension") : samples;
        if(phenotype_count > 0)
          check_cublas(cublasDgemm(lane.blas, CUBLAS_OP_N, CUBLAS_OP_N,
            rows, phenotype_count, samples, &alpha,
            device_genotypes, rows, device_fold_phenotypes,
            phenotype_leading_dimension, &beta,
            lane.right_hand_sides, rows),
            "cublasDgemm(resident fold crossproduct)");
        check_cublas(cublasDgemm(lane.blas, CUBLAS_OP_N, CUBLAS_OP_T,
          rows, rows, samples, &alpha, device_genotypes, rows,
          device_genotypes, rows, &beta, lane.gram, rows),
          "cublasDgemm(resident fold Gram product)");
      }
      synchronize_level0_cholesky_lanes(system_count);

      check_cuda(cudaMemset(d_gram_, 0,
        static_cast<size_t>(gram_elements) * sizeof(double)),
        "clear resident fold total Gram matrix");
      if(rhs_elements > 0)
        check_cuda(cudaMemset(d_crossproduct_, 0,
          static_cast<size_t>(rhs_elements) * sizeof(double)),
          "clear resident fold total right-hand sides");
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cublas(cublasDaxpy(handle_, checked_int(gram_elements,
          "resident fold Gram element count"), &alpha, lane.gram, 1,
          d_gram_, 1), "sum resident fold Gram matrix");
        if(rhs_elements > 0)
          check_cublas(cublasDaxpy(handle_, checked_int(rhs_elements,
            "resident fold right-hand-side element count"), &alpha,
            lane.right_hand_sides, 1, d_crossproduct_, 1),
            "sum resident fold right-hand sides");
      }
      const double minus_one = -1.0;
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cublas(cublasDscal(handle_, checked_int(gram_elements,
          "resident fold Gram element count"), &minus_one,
          lane.gram, 1), "negate resident fold Gram matrix");
        check_cublas(cublasDaxpy(handle_, checked_int(gram_elements,
          "resident fold Gram element count"), &alpha, d_gram_, 1,
          lane.gram, 1), "form resident training Gram matrix");
        if(rhs_elements > 0) {
          check_cublas(cublasDscal(handle_, checked_int(rhs_elements,
            "resident fold right-hand-side element count"), &minus_one,
            lane.right_hand_sides, 1),
            "negate resident fold right-hand sides");
          check_cublas(cublasDaxpy(handle_, checked_int(rhs_elements,
            "resident fold right-hand-side element count"), &alpha,
            d_crossproduct_, 1, lane.right_hand_sides, 1),
            "form resident training right-hand sides");
        }
      }
      check_cuda(cudaDeviceSynchronize(),
        "synchronize resident fold system preparation");
      if(timings) {
        timings->gram_ms += elapsed_ms(phase_start);
        timings->resident_reuse_count += system_count;
      }
      resident_fold_system_count_ = system_count_index;
      resident_fold_rhs_count_ = phenotypes.cols();
      resident_fold_systems_valid_ = true;
      resident_fold_systems_design_orientation_ = false;
      return true;
    }

    bool cache_resident_design_fold_systems(
      const Eigen::Ref<const Eigen::VectorXi>& start_rows,
      const Eigen::Ref<const Eigen::VectorXi>& row_counts,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Step1ComputeTimings* timings) override {

      invalidate_resident_fold_systems();
      const Eigen::Index system_count_index = start_rows.size();
      if(!level0_cholesky_enabled_ || !level0_fold_batch_enabled_ ||
         system_count_index < 2)
        return false;
      if(!resident_design_valid_ ||
         row_counts.size() != system_count_index ||
         outcomes.rows() != resident_design_rows_)
        throw std::invalid_argument(
          "Step 1 resident design fold products received incompatible dimensions");
      for(Eigen::Index system = 0; system < system_count_index; ++system) {
        if(start_rows(system) < 0 || row_counts(system) < 0 ||
           start_rows(system) > resident_design_rows_ - row_counts(system))
          throw std::invalid_argument(
            "Step 1 resident design fold products received an invalid fold");
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const size_t system_count = static_cast<size_t>(system_count_index);
      const int features = checked_int(
        resident_design_columns_, "resident design fold feature count");
      const int outcome_count = checked_int(
        outcomes.cols(), "resident design fold outcome count");
      if(features == 0) return false;
      const Eigen::Index gram_elements =
        resident_design_columns_ * resident_design_columns_;
      const Eigen::Index rhs_elements =
        resident_design_columns_ * outcomes.cols();
      const Eigen::Index resident_design_elements =
        resident_design_rows_ * resident_design_columns_;
      if(!resident_design_uses_level1_cache_ && projected_capacity_ <
           static_cast<size_t>(resident_design_elements))
        throw std::runtime_error(
          "Step 1 resident design fold staging workspace is unavailable");
      ensure_level0_cholesky_lane_count(system_count);
      ensure_capacity(d_gram_, gram_capacity_, gram_elements,
        "cudaMalloc(resident design fold total Gram matrix)");
      if(rhs_elements > 0)
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, rhs_elements,
          "cudaMalloc(resident design fold total right-hand sides)");

      std::vector<Eigen::MatrixXd> packed_outcomes(system_count);
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        const Eigen::Index fold = static_cast<Eigen::Index>(system);
        const Eigen::Index sample_count = row_counts(fold);
        ensure_capacity(lane.gram, lane.gram_capacity, gram_elements,
          "cudaMalloc(resident design fold Gram matrix)");
        ensure_capacity(lane.right_hand_sides,
          lane.right_hand_sides_capacity, rhs_elements,
          "cudaMalloc(resident design fold right-hand sides)");
        if(outcome_count > 0) {
          packed_outcomes[system] = outcomes.middleRows(
            start_rows(fold), sample_count);
          ensure_capacity(lane.predictions, lane.predictions_capacity,
            packed_outcomes[system].size(),
            "cudaMalloc(resident design fold outcome staging)");
        }
      }

      ComputeClock::time_point phase_start;
      if(timings) phase_start = ComputeClock::now();
      for(size_t system = 0; system < system_count; ++system) {
        if(outcome_count == 0) continue;
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cuda(cudaMemcpyAsync(lane.predictions,
          packed_outcomes[system].data(),
          packed_outcomes[system].size() * sizeof(double),
          cudaMemcpyHostToDevice, lane.stream),
          "copy resident design fold outcomes to CUDA device");
      }
      synchronize_level0_cholesky_lanes(system_count);
      if(timings) timings->upload_ms += elapsed_ms(phase_start);

      if(timings) phase_start = ComputeClock::now();
      const double alpha = 1.0;
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        const Eigen::Index fold = static_cast<Eigen::Index>(system);
        const Eigen::Index fold_rows = row_counts(fold);
        if(fold_rows == 0) {
          check_cuda(cudaMemsetAsync(lane.gram, 0,
            static_cast<size_t>(gram_elements) * sizeof(double),
            lane.stream),
            "clear empty resident design fold Gram matrix");
          if(rhs_elements > 0)
            check_cuda(cudaMemsetAsync(lane.right_hand_sides, 0,
              static_cast<size_t>(rhs_elements) * sizeof(double),
              lane.stream),
              "clear empty resident design fold right-hand sides");
          continue;
        }
        const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
          fold_rows, resident_design_columns_);
        for(Eigen::Index start = 0; start < fold_rows;
            start += chunk_rows) {
          const Eigen::Index count_index = std::min(
            chunk_rows, fold_rows - start);
          const int count = checked_int(count_index,
            "resident design fold product chunk row count");
          const double beta = start == 0 ? 0.0 : 1.0;
          const double* design_chunk = nullptr;
          int design_leading_dimension = 0;
          if(resident_design_uses_level1_cache_) {
            design_chunk = d_level1_design_ + start_rows(fold) + start;
            design_leading_dimension = checked_int(
              resident_design_rows_,
              "persistent Level 1 design leading dimension");
          } else {
            double* staged_design_chunk = d_projected_ +
              start_rows(fold) * resident_design_columns_;
            check_cuda(cudaMemcpy2DAsync(staged_design_chunk,
              static_cast<size_t>(count) * sizeof(double),
              d_resident_genotypes_ + start_rows(fold) + start,
              static_cast<size_t>(resident_design_rows_) * sizeof(double),
              static_cast<size_t>(count) * sizeof(double),
              static_cast<size_t>(features), cudaMemcpyDeviceToDevice,
              lane.stream),
              "stage resident design fold product chunk on CUDA device");
            design_chunk = staged_design_chunk;
            design_leading_dimension = count;
          }
          if(outcome_count > 0)
            check_cublas(cublasDgemm(lane.blas,
              CUBLAS_OP_T, CUBLAS_OP_N,
              features, outcome_count, count, &alpha,
              design_chunk, design_leading_dimension,
              lane.predictions + start,
              checked_int(fold_rows,
                "resident design fold outcome leading dimension"),
              &beta, lane.right_hand_sides, features),
              "cublasDgemm(resident design fold crossproduct)");
          check_cublas(cublasDsyrk(lane.blas, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, features, count, &alpha,
            design_chunk, design_leading_dimension,
            &beta, lane.gram, features),
            "cublasDsyrk(resident design fold Gram product)");
        }
        const dim3 threads(16, 16);
        const dim3 grid((features + threads.x - 1) / threads.x,
          (features + threads.y - 1) / threads.y);
        mirror_lower_triangle<<<grid, threads, 0, lane.stream>>>(
          lane.gram, features);
        check_cuda(cudaGetLastError(),
          "mirror resident design fold Gram triangle kernel");
      }
      synchronize_level0_cholesky_lanes(system_count);

      check_cuda(cudaMemset(d_gram_, 0,
        static_cast<size_t>(gram_elements) * sizeof(double)),
        "clear resident design fold total Gram matrix");
      if(rhs_elements > 0)
        check_cuda(cudaMemset(d_crossproduct_, 0,
          static_cast<size_t>(rhs_elements) * sizeof(double)),
          "clear resident design fold total right-hand sides");
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cublas(cublasDaxpy(handle_, checked_int(gram_elements,
          "resident design fold Gram element count"), &alpha,
          lane.gram, 1, d_gram_, 1),
          "sum resident design fold Gram matrix");
        if(rhs_elements > 0)
          check_cublas(cublasDaxpy(handle_, checked_int(rhs_elements,
            "resident design fold right-hand-side element count"), &alpha,
            lane.right_hand_sides, 1, d_crossproduct_, 1),
            "sum resident design fold right-hand sides");
      }
      const double minus_one = -1.0;
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cublas(cublasDscal(handle_, checked_int(gram_elements,
          "resident design fold Gram element count"), &minus_one,
          lane.gram, 1), "negate resident design fold Gram matrix");
        check_cublas(cublasDaxpy(handle_, checked_int(gram_elements,
          "resident design fold Gram element count"), &alpha,
          d_gram_, 1, lane.gram, 1),
          "form resident design training Gram matrix");
        if(rhs_elements > 0) {
          check_cublas(cublasDscal(handle_, checked_int(rhs_elements,
            "resident design fold right-hand-side element count"),
            &minus_one, lane.right_hand_sides, 1),
            "negate resident design fold right-hand sides");
          check_cublas(cublasDaxpy(handle_, checked_int(rhs_elements,
            "resident design fold right-hand-side element count"), &alpha,
            d_crossproduct_, 1, lane.right_hand_sides, 1),
            "form resident design training right-hand sides");
        }
      }
      check_cuda(cudaDeviceSynchronize(),
        "synchronize resident design fold system preparation");
      if(timings) {
        timings->gram_ms += elapsed_ms(phase_start);
        timings->resident_design_reuse_count += system_count;
      }
      resident_fold_system_count_ = system_count_index;
      resident_fold_rhs_count_ = outcomes.cols();
      resident_fold_systems_valid_ = true;
      resident_fold_systems_design_orientation_ = true;
      return true;
    }

    void ridge_predict_preprocessed(
      Eigen::Index start_column,
      Eigen::Index column_count,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings) override {

      if(ridge_factorized_size_ < 0)
        throw std::runtime_error(
          "Step 1 resident ridge prediction requested before factorization");
      if(!resident_valid_ || resident_rows_ != ridge_factorized_size_ ||
         start_column < 0 || column_count < 0 ||
         start_column > resident_columns_ - column_count)
        throw std::invalid_argument(
          "Step 1 resident ridge prediction received incompatible dimensions");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 resident ridge parameters must be non-negative");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = ridge_factorized_size_;
      const int sample_count = checked_int(
        column_count, "resident ridge sample count");
      const int phenotype_count = ridge_factorized_rhs_count_;
      const int parameter_count = checked_int(
        ridge_parameters.size(), "resident ridge parameter count");
      const long long combination_count_long =
        static_cast<long long>(phenotype_count) * parameter_count;
      if(combination_count_long > INT_MAX)
        throw std::runtime_error(
          "CUDA resident ridge phenotype/parameter count exceeds integer limits");
      const int combination_count =
        static_cast<int>(combination_count_long);
      predictions.resize(sample_count, combination_count);
      coefficients.resize(size, combination_count);
      if(size == 0 || sample_count == 0 || combination_count == 0) {
        predictions.setZero();
        coefficients.setZero();
        return;
      }

      const Eigen::Index chunk_samples = bounded_cuda_chunk_rows(
        column_count, combination_count);
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_,
        ridge_parameters.size(),
        "cudaMalloc(resident ridge parameters)");
      ensure_capacity(d_inverse_, inverse_capacity_,
        static_cast<Eigen::Index>(size) * parameter_count,
        "cudaMalloc(resident ridge inverse)");
      ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_,
        static_cast<Eigen::Index>(size) * combination_count,
        "cudaMalloc(resident ridge scaled right-hand sides)");
      ensure_capacity(d_phenotypes_, phenotypes_capacity_,
        coefficients.size(), "cudaMalloc(resident ridge coefficients)");
      ensure_capacity(d_predictions_, predictions_capacity_,
        chunk_samples * combination_count,
        "cudaMalloc(resident ridge prediction chunk)");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_ridge_parameters_, ridge_parameters.data(),
        ridge_parameters.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy resident ridge parameters to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      std::unique_ptr<CudaEventPair> coefficient_events;
      if(timings) {
        coefficient_events.reset(new CudaEventPair());
        coefficient_events->record_start();
      }
      const int threads = 256;
      const int inverse_count = checked_element_count(
        size, parameter_count, "resident ridge inverse");
      build_ridge_inverse<<<
        (inverse_count + threads - 1) / threads, threads>>>(
        d_ridge_values_, d_ridge_parameters_, d_inverse_, size,
        inverse_count);
      check_cuda(cudaGetLastError(),
        "build resident ridge inverse kernel");
      const int scaled_count = checked_element_count(
        size, combination_count,
        "resident ridge scaled right-hand sides");
      build_scaled_right_hand_sides<<<
        (scaled_count + threads - 1) / threads, threads>>>(
        d_inverse_, d_ridge_rhs_, d_scaled_rhs_, size,
        phenotype_count, scaled_count);
      check_cuda(cudaGetLastError(),
        "build resident ridge scaled right-hand sides kernel");
      const double alpha = 1.0;
      const double beta = 0.0;
      check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        size, combination_count, size, &alpha,
        d_ridge_vectors_, size, d_scaled_rhs_, size, &beta,
        d_phenotypes_, size),
        "cublasDgemm(resident ridge coefficients)");
      if(timings)
        timings->ridge_ms +=
          coefficient_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(coefficients.data(), d_phenotypes_,
        coefficients.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy resident ridge coefficients from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);

      for(Eigen::Index start = 0; start < column_count;
          start += chunk_samples) {
        const Eigen::Index count_index = std::min(
          chunk_samples, column_count - start);
        const int count = checked_int(
          count_index, "resident ridge prediction chunk sample count");
        const double* device_prediction_chunk = d_resident_genotypes_ +
          (start_column + start) * resident_rows_;
        if(timings) timings->resident_reuse_count++;

        std::unique_ptr<CudaEventPair> prediction_events;
        if(timings) {
          prediction_events.reset(new CudaEventPair());
          prediction_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          count, combination_count, size, &alpha,
          device_prediction_chunk, size,
          d_phenotypes_, size, &beta,
          d_predictions_, count),
          "cublasDgemm(resident ridge prediction chunk)");
        if(timings)
          timings->ridge_ms +=
            prediction_events->record_stop_and_elapsed_ms();

        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy2D(
          predictions.data() + start,
          static_cast<size_t>(predictions.outerStride()) * sizeof(double),
          d_predictions_, static_cast<size_t>(count) * sizeof(double),
          static_cast<size_t>(count) * sizeof(double),
          combination_count, cudaMemcpyDeviceToHost),
          "copy resident ridge prediction chunk from CUDA device");
        if(timings) timings->download_ms += elapsed_ms(transfer_start);
      }
    }

    bool ridge_predict_preprocessed_system(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::Index start_column,
      Eigen::Index column_count,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings) override {

      if(!level0_cholesky_enabled_) return false;
      if(!resident_valid_ || gram.rows() != gram.cols() ||
         gram.rows() != resident_rows_ ||
         right_hand_sides.rows() != resident_rows_ ||
         start_column < 0 || column_count < 0 ||
         start_column > resident_columns_ - column_count)
        throw std::invalid_argument(
          "Step 1 resident Cholesky ridge prediction received incompatible dimensions");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 resident Cholesky ridge parameters must be non-negative");
      if((ridge_parameters.array() == 0).any()) return false;

      const Eigen::VectorXd penalty_multipliers =
        Eigen::VectorXd::Ones(gram.rows());
      diagonal_penalty_solve(gram, right_hand_sides, ridge_parameters,
        penalty_multipliers, coefficients, timings);

      const int size = checked_int(
        resident_rows_, "resident Cholesky ridge system size");
      const int sample_count = checked_int(
        column_count, "resident Cholesky ridge sample count");
      const int combination_count = checked_int(
        coefficients.cols(), "resident Cholesky ridge combination count");
      predictions.resize(sample_count, combination_count);
      if(size == 0 || sample_count == 0 || combination_count == 0) {
        predictions.setZero();
        return true;
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      ensure_capacity(d_inverse_, inverse_capacity_, predictions.size(),
        "cudaMalloc(resident Cholesky ridge predictions)");
      const double* device_prediction_matrix = d_resident_genotypes_ +
        start_column * resident_rows_;
      const double alpha = 1.0;
      const double beta = 0.0;
      std::unique_ptr<CudaEventPair> prediction_events;
      if(timings) {
        prediction_events.reset(new CudaEventPair());
        prediction_events->record_start();
      }
      check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        sample_count, combination_count, size, &alpha,
        device_prediction_matrix, size, d_predictions_, size, &beta,
        d_inverse_, sample_count),
        "cublasDgemm(resident Cholesky ridge predictions)");
      if(timings) {
        timings->ridge_ms +=
          prediction_events->record_stop_and_elapsed_ms();
        timings->resident_reuse_count++;
      }

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(predictions.data(), d_inverse_,
        predictions.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy resident Cholesky ridge predictions from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
      return true;
    }

    bool ridge_predict_preprocessed_systems(
      const std::vector<Eigen::MatrixXd>& grams,
      const std::vector<Eigen::MatrixXd>& right_hand_sides,
      const Eigen::Ref<const Eigen::VectorXi>& start_columns,
      const Eigen::Ref<const Eigen::VectorXi>& column_counts,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      std::vector<Eigen::MatrixXd>& predictions,
      std::vector<Eigen::MatrixXd>& coefficients,
      Step1ComputeTimings* timings) override {

      const size_t system_count = grams.size();
      if(!level0_cholesky_enabled_ || !level0_fold_batch_enabled_ ||
         system_count < 2)
        return false;
      if(!resident_valid_ || right_hand_sides.size() != system_count ||
         start_columns.size() != static_cast<Eigen::Index>(system_count) ||
         column_counts.size() != static_cast<Eigen::Index>(system_count))
        throw std::invalid_argument(
          "Step 1 resident batched Cholesky ridge prediction received incompatible systems");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 resident batched Cholesky ridge parameters must be non-negative");
      if((ridge_parameters.array() == 0).any()) return false;

      const Eigen::Index size_index = resident_rows_;
      const Eigen::Index right_hand_side_count_index =
        right_hand_sides.empty() ? 0 : right_hand_sides.front().cols();
      for(size_t system = 0; system < system_count; ++system) {
        if(grams[system].rows() != size_index ||
           grams[system].cols() != size_index ||
           right_hand_sides[system].rows() != size_index ||
           right_hand_sides[system].cols() != right_hand_side_count_index ||
           start_columns(static_cast<Eigen::Index>(system)) < 0 ||
           column_counts(static_cast<Eigen::Index>(system)) < 0 ||
           start_columns(static_cast<Eigen::Index>(system)) >
             resident_columns_ -
               column_counts(static_cast<Eigen::Index>(system)))
          throw std::invalid_argument(
            "Step 1 resident batched Cholesky ridge prediction received incompatible dimensions");
      }

      const int size = checked_int(
        size_index, "resident batched Cholesky ridge system size");
      const int right_hand_side_count = checked_int(
        right_hand_side_count_index,
        "resident batched Cholesky ridge right-hand-side count");
      const int parameter_count = checked_int(
        ridge_parameters.size(),
        "resident batched Cholesky ridge parameter count");
      const long long combination_count_long =
        static_cast<long long>(right_hand_side_count) * parameter_count;
      if(combination_count_long > INT_MAX)
        throw std::runtime_error(
          "resident batched Cholesky ridge combination count exceeds integer limits");
      const int combination_count =
        static_cast<int>(combination_count_long);

      predictions.resize(system_count);
      coefficients.resize(system_count);
      for(size_t system = 0; system < system_count; ++system) {
        const Eigen::Index sample_count =
          column_counts(static_cast<Eigen::Index>(system));
        predictions[system].resize(sample_count, combination_count);
        coefficients[system].resize(size_index, combination_count);
      }
      if(size == 0 || right_hand_side_count == 0 ||
         parameter_count == 0) {
        for(size_t system = 0; system < system_count; ++system) {
          predictions[system].setZero();
          coefficients[system].setZero();
        }
        return true;
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      ensure_level0_cholesky_lane_count(system_count);
      const Eigen::Index gram_elements = size_index * size_index;
      const Eigen::Index rhs_elements =
        size_index * right_hand_side_count_index;
      const Eigen::Index coefficient_elements =
        size_index * combination_count;
      std::vector<int> workspace_sizes(system_count, 0);
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        ensure_capacity(lane.gram, lane.gram_capacity, gram_elements,
          "cudaMalloc(batched Cholesky Gram matrix)");
        const bool factor_grew = static_cast<size_t>(gram_elements) >
          lane.factor_capacity;
        ensure_capacity(lane.factor, lane.factor_capacity, gram_elements,
          "cudaMalloc(batched Cholesky factorization matrix)");
        ensure_capacity(lane.right_hand_sides,
          lane.right_hand_sides_capacity, rhs_elements,
          "cudaMalloc(batched Cholesky right-hand sides)");
        ensure_capacity(lane.solve, lane.solve_capacity, rhs_elements,
          "cudaMalloc(batched Cholesky solve workspace)");
        ensure_capacity(lane.coefficients, lane.coefficients_capacity,
          coefficient_elements,
          "cudaMalloc(batched Cholesky coefficients)");
        ensure_capacity(lane.predictions, lane.predictions_capacity,
          predictions[system].size(),
          "cudaMalloc(batched Cholesky predictions)");
        ensure_capacity(lane.info, lane.info_capacity,
          2 * parameter_count,
          "cudaMalloc(batched Cholesky solver status)");
        if(factor_grew || lane.workspace_capacity == 0) {
          check_cusolver(cusolverDnDpotrf_bufferSize(lane.solver,
            CUBLAS_FILL_MODE_LOWER, size, lane.factor, size,
            &workspace_sizes[system]),
            "cusolverDnDpotrf_bufferSize(batched resident ridge)");
          ensure_capacity(lane.workspace, lane.workspace_capacity,
            workspace_sizes[system],
            "cudaMalloc(batched Cholesky solver workspace)");
        } else {
          workspace_sizes[system] = checked_int(
            static_cast<Eigen::Index>(lane.workspace_capacity),
            "batched Cholesky solver workspace size");
        }
      }

      ComputeClock::time_point phase_start;
      if(timings) phase_start = ComputeClock::now();
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cuda(cudaMemcpyAsync(lane.gram, grams[system].data(),
          gram_elements * sizeof(double), cudaMemcpyHostToDevice,
          lane.stream), "copy batched Cholesky Gram matrix to CUDA device");
        check_cuda(cudaMemcpyAsync(lane.right_hand_sides,
          right_hand_sides[system].data(), rhs_elements * sizeof(double),
          cudaMemcpyHostToDevice, lane.stream),
          "copy batched Cholesky right-hand sides to CUDA device");
      }
      synchronize_level0_cholesky_lanes(system_count);
      if(timings) timings->upload_ms += elapsed_ms(phase_start);

      std::vector<std::vector<int>> solver_status(
        system_count, std::vector<int>(2 * parameter_count));
      if(timings) phase_start = ComputeClock::now();
      const int threads = 256;
      const double alpha = 1.0;
      const double beta = 0.0;
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        for(int parameter = 0; parameter < parameter_count; ++parameter) {
          check_cuda(cudaMemcpyAsync(lane.factor, lane.gram,
            gram_elements * sizeof(double), cudaMemcpyDeviceToDevice,
            lane.stream),
            "copy batched Cholesky factorization matrix");
          check_cuda(cudaMemcpyAsync(lane.solve, lane.right_hand_sides,
            rhs_elements * sizeof(double), cudaMemcpyDeviceToDevice,
            lane.stream), "copy batched Cholesky solve right-hand sides");
          add_uniform_diagonal_penalty<<<
            (size + threads - 1) / threads, threads, 0, lane.stream>>>(
              lane.factor, ridge_parameters(parameter), size);
          check_cuda(cudaGetLastError(),
            "add batched uniform diagonal penalty kernel");
          check_cusolver(cusolverDnDpotrf(lane.solver,
            CUBLAS_FILL_MODE_LOWER, size, lane.factor, size,
            lane.workspace, workspace_sizes[system],
            lane.info + 2 * parameter),
            "cusolverDnDpotrf(batched resident ridge)");
          check_cusolver(cusolverDnDpotrs(lane.solver,
            CUBLAS_FILL_MODE_LOWER, size, right_hand_side_count,
            lane.factor, size, lane.solve, size,
            lane.info + 2 * parameter + 1),
            "cusolverDnDpotrs(batched resident ridge)");
          check_cuda(cudaMemcpyAsync(
            lane.coefficients +
              static_cast<Eigen::Index>(parameter) * rhs_elements,
            lane.solve, rhs_elements * sizeof(double),
            cudaMemcpyDeviceToDevice, lane.stream),
            "store batched Cholesky coefficients");
        }

        const int sample_count = checked_int(
          column_counts(static_cast<Eigen::Index>(system)),
          "resident batched Cholesky ridge sample count");
        if(sample_count > 0) {
          const double* prediction_matrix = d_resident_genotypes_ +
            static_cast<Eigen::Index>(
              start_columns(static_cast<Eigen::Index>(system))) *
              resident_rows_;
          check_cublas(cublasDgemm(lane.blas, CUBLAS_OP_T, CUBLAS_OP_N,
            sample_count, combination_count, size, &alpha,
            prediction_matrix, size, lane.coefficients, size, &beta,
            lane.predictions, sample_count),
            "cublasDgemm(batched resident Cholesky predictions)");
        }
      }
      synchronize_level0_cholesky_lanes(system_count);
      for(size_t system = 0; system < system_count; ++system)
        check_cuda(cudaMemcpy(solver_status[system].data(),
          level0_cholesky_lanes_[system].info,
          solver_status[system].size() * sizeof(int),
          cudaMemcpyDeviceToHost),
          "copy batched Cholesky solver status to host");
      if(timings) timings->ridge_ms += elapsed_ms(phase_start);

      for(size_t system = 0; system < system_count; ++system) {
        for(int parameter = 0; parameter < parameter_count; ++parameter) {
          const int factor_status = solver_status[system][2 * parameter];
          const int solve_status = solver_status[system][2 * parameter + 1];
          if(factor_status != 0 || solve_status != 0) {
            std::ostringstream message;
            message << "cuSOLVER batched resident Cholesky failed for system="
                    << system << " parameter=" << parameter
                    << " factor_info=" << factor_status
                    << " solve_info=" << solve_status;
            throw std::runtime_error(message.str());
          }
        }
      }

      if(timings) phase_start = ComputeClock::now();
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        check_cuda(cudaMemcpyAsync(coefficients[system].data(),
          lane.coefficients, coefficients[system].size() * sizeof(double),
          cudaMemcpyDeviceToHost, lane.stream),
          "copy batched Cholesky coefficients from CUDA device");
        if(predictions[system].size() > 0)
          check_cuda(cudaMemcpyAsync(predictions[system].data(),
            lane.predictions, predictions[system].size() * sizeof(double),
            cudaMemcpyDeviceToHost, lane.stream),
            "copy batched Cholesky predictions from CUDA device");
      }
      synchronize_level0_cholesky_lanes(system_count);
      if(timings) {
        timings->download_ms += elapsed_ms(phase_start);
        timings->resident_reuse_count += system_count;
      }
      return true;
    }

    bool ridge_predict_cached_preprocessed_systems(
      const Eigen::Ref<const Eigen::VectorXi>& start_columns,
      const Eigen::Ref<const Eigen::VectorXi>& column_counts,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      std::vector<Eigen::MatrixXd>& predictions,
      std::vector<Eigen::MatrixXd>& coefficients,
      Step1ComputeTimings* timings) override {

      return ridge_predict_cached_preprocessed_systems_impl(
        start_columns, column_counts, ridge_parameters,
        predictions, coefficients, true, timings);
    }

    bool ridge_predict_cached_preprocessed_systems_normalized(
      const Eigen::Ref<const Eigen::VectorXi>& start_columns,
      const Eigen::Ref<const Eigen::VectorXi>& column_counts,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      double effective_sample_count,
      Eigen::Index level1_start_column,
      Eigen::MatrixXd& normalized_predictions,
      Step1ComputeTimings* timings) override {

      if(!resident_fold_systems_valid_ ||
         resident_fold_systems_design_orientation_ ||
         resident_fold_rhs_count_ != 1 || !d_level1_design_ ||
         level1_design_rows_ <= 0 ||
         level1_start_column != level1_design_cached_columns_ ||
         ridge_parameters.size() >
           level1_design_columns_ - level1_start_column ||
         effective_sample_count != level1_design_rows_)
        return false;
      Eigen::Index covered_rows = 0;
      for(Eigen::Index fold = 0; fold < start_columns.size(); ++fold) {
        if(start_columns(fold) != covered_rows) return false;
        covered_rows += column_counts(fold);
      }
      if(covered_rows != level1_design_rows_) return false;

      std::vector<Eigen::MatrixXd> unused_predictions;
      std::vector<Eigen::MatrixXd> unused_coefficients;
      if(!ridge_predict_cached_preprocessed_systems_impl(
           start_columns, column_counts, ridge_parameters,
           unused_predictions, unused_coefficients, false, timings))
        return false;

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int rows = checked_int(
        level1_design_rows_, "normalized Level 0 prediction row count");
      const int columns = checked_int(ridge_parameters.size(),
        "normalized Level 0 prediction column count");
      const int element_count = checked_element_count(
        rows, columns, "normalized Level 0 predictions");
      normalized_predictions.resize(rows, columns);
      if(rows == 0 || columns == 0) {
        normalized_predictions.setZero();
        return true;
      }

      ComputeClock::time_point phase_start;
      if(timings) phase_start = ComputeClock::now();
      double* destination = d_level1_design_ +
        level1_start_column * level1_design_rows_;
      for(Eigen::Index fold = 0; fold < start_columns.size(); ++fold) {
        const int fold_rows = checked_int(column_counts(fold),
          "normalized Level 0 fold row count");
        if(fold_rows == 0) continue;
        CudaLevel0CholeskyLane& lane =
          level0_cholesky_lanes_[static_cast<size_t>(fold)];
        check_cuda(cudaMemcpy2DAsync(
          destination + start_columns(fold),
          static_cast<size_t>(rows) * sizeof(double),
          lane.predictions,
          static_cast<size_t>(fold_rows) * sizeof(double),
          static_cast<size_t>(fold_rows) * sizeof(double),
          static_cast<size_t>(columns), cudaMemcpyDeviceToDevice,
          lane.stream),
          "assemble normalized Level 0 predictions on CUDA device");
      }
      synchronize_level0_cholesky_lanes(
        static_cast<size_t>(start_columns.size()));

      ensure_capacity(d_level1_ones_, level1_ones_capacity_, rows,
        "cudaMalloc(Level 0 normalization ones)");
      const int threads = 256;
      fill_constant<<<(rows + threads - 1) / threads, threads>>>(
        d_level1_ones_, 1.0, rows);
      check_cuda(cudaGetLastError(),
        "fill Level 0 normalization ones kernel");
      std::vector<double> means(columns);
      std::vector<double> inverse_standard_deviations(columns);
      for(int column = 0; column < columns; ++column) {
        const double* values = destination +
          static_cast<Eigen::Index>(column) * rows;
        double sum = 0.0;
        double sum_of_squares = 0.0;
        check_cublas(cublasDdot(handle_, rows, values, 1,
          d_level1_ones_, 1, &sum),
          "cublasDdot(Level 0 prediction sum)");
        check_cublas(cublasDdot(handle_, rows, values, 1,
          values, 1, &sum_of_squares),
          "cublasDdot(Level 0 prediction sum of squares)");
        means[column] = sum / effective_sample_count;
        const double centered_sum_of_squares = sum_of_squares -
          effective_sample_count * means[column] * means[column];
        if(!(centered_sum_of_squares > 0) ||
           !std::isfinite(centered_sum_of_squares))
          throw std::runtime_error(
            "CUDA Level 0 predictions have invalid variance");
        inverse_standard_deviations[column] = std::sqrt(
          (effective_sample_count - 1.0) / centered_sum_of_squares);
      }
      ensure_capacity(d_inverse_, inverse_capacity_, columns,
        "cudaMalloc(Level 0 prediction means)");
      ensure_capacity(d_eigenvalues_, eigenvalues_capacity_, columns,
        "cudaMalloc(Level 0 prediction inverse standard deviations)");
      check_cuda(cudaMemcpy(d_inverse_, means.data(),
        static_cast<size_t>(columns) * sizeof(double),
        cudaMemcpyHostToDevice),
        "copy Level 0 prediction means to CUDA device");
      check_cuda(cudaMemcpy(d_eigenvalues_,
        inverse_standard_deviations.data(),
        static_cast<size_t>(columns) * sizeof(double),
        cudaMemcpyHostToDevice),
        "copy Level 0 prediction inverse standard deviations to CUDA device");
      normalize_design_columns<<<
        (element_count + threads - 1) / threads, threads>>>(
          destination, rows, element_count,
          d_inverse_, d_eigenvalues_);
      check_cuda(cudaGetLastError(),
        "normalize Level 0 prediction columns kernel");
      check_cuda(cudaDeviceSynchronize(),
        "finish Level 0 prediction normalization");
      if(timings) timings->ridge_ms += elapsed_ms(phase_start);

      if(timings) phase_start = ComputeClock::now();
      check_cuda(cudaMemcpy(normalized_predictions.data(), destination,
        static_cast<size_t>(element_count) * sizeof(double),
        cudaMemcpyDeviceToHost),
        "copy normalized Level 0 predictions from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(phase_start);
      level1_design_cached_columns_ += columns;
      return true;
    }

    bool ridge_predict_cached_preprocessed_systems_impl(
      const Eigen::Ref<const Eigen::VectorXi>& start_columns,
      const Eigen::Ref<const Eigen::VectorXi>& column_counts,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      std::vector<Eigen::MatrixXd>& predictions,
      std::vector<Eigen::MatrixXd>& coefficients,
      bool copy_results_to_host,
      Step1ComputeTimings* timings) {

      if(!resident_fold_systems_valid_) return false;
      const size_t system_count = static_cast<size_t>(
        resident_fold_system_count_);
      const bool design_orientation =
        resident_fold_systems_design_orientation_;
      if((design_orientation ? !resident_design_valid_ : !resident_valid_) ||
         system_count < 2 ||
         start_columns.size() != resident_fold_system_count_ ||
         column_counts.size() != resident_fold_system_count_)
        throw std::invalid_argument(
          "Step 1 cached fold ridge prediction received incompatible systems");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 cached fold ridge parameters must be non-negative");
      if((ridge_parameters.array() == 0).any()) return false;
      for(size_t system = 0; system < system_count; ++system) {
        const Eigen::Index fold = static_cast<Eigen::Index>(system);
        if(start_columns(fold) < 0 || column_counts(fold) < 0 ||
           start_columns(fold) >
             (design_orientation ? resident_design_rows_ :
               resident_columns_) - column_counts(fold))
          throw std::invalid_argument(
            "Step 1 cached fold ridge prediction received invalid dimensions");
      }

      const Eigen::Index size_index = design_orientation ?
        resident_design_columns_ : resident_rows_;
      const Eigen::Index right_hand_side_count_index =
        resident_fold_rhs_count_;
      const int size = checked_int(
        size_index, "cached fold Cholesky ridge system size");
      const int right_hand_side_count = checked_int(
        right_hand_side_count_index,
        "cached fold Cholesky ridge right-hand-side count");
      const int parameter_count = checked_int(
        ridge_parameters.size(),
        "cached fold Cholesky ridge parameter count");
      const long long combination_count_long =
        static_cast<long long>(right_hand_side_count) * parameter_count;
      if(combination_count_long > INT_MAX)
        throw std::runtime_error(
          "cached fold Cholesky ridge combination count exceeds integer limits");
      const int combination_count =
        static_cast<int>(combination_count_long);

      predictions.resize(system_count);
      coefficients.resize(system_count);
      for(size_t system = 0; system < system_count; ++system) {
        const Eigen::Index sample_count = column_counts(
          static_cast<Eigen::Index>(system));
        if(copy_results_to_host) {
          predictions[system].resize(sample_count, combination_count);
          coefficients[system].resize(size_index, combination_count);
        }
      }
      if(size == 0 || right_hand_side_count == 0 ||
         parameter_count == 0) {
        for(size_t system = 0; system < system_count; ++system) {
          predictions[system].setZero();
          coefficients[system].setZero();
        }
        return true;
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      ensure_level0_cholesky_lane_count(system_count);
      const Eigen::Index gram_elements = size_index * size_index;
      const Eigen::Index rhs_elements =
        size_index * right_hand_side_count_index;
      const Eigen::Index coefficient_elements =
        size_index * combination_count;
      std::vector<int> workspace_sizes(system_count, 0);
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        if(lane.gram_capacity < static_cast<size_t>(gram_elements) ||
           lane.right_hand_sides_capacity <
             static_cast<size_t>(rhs_elements))
          throw std::runtime_error(
            "Step 1 cached fold systems were released before ridge prediction");
        const bool factor_grew = static_cast<size_t>(gram_elements) >
          lane.factor_capacity;
        ensure_capacity(lane.factor, lane.factor_capacity, gram_elements,
          "cudaMalloc(cached fold Cholesky factorization matrix)");
        ensure_capacity(lane.solve, lane.solve_capacity, rhs_elements,
          "cudaMalloc(cached fold Cholesky solve workspace)");
        ensure_capacity(lane.coefficients, lane.coefficients_capacity,
          coefficient_elements,
          "cudaMalloc(cached fold Cholesky coefficients)");
        ensure_capacity(lane.predictions, lane.predictions_capacity,
          column_counts(static_cast<Eigen::Index>(system)) *
            combination_count,
          "cudaMalloc(cached fold Cholesky predictions)");
        ensure_capacity(lane.info, lane.info_capacity,
          2 * parameter_count,
          "cudaMalloc(cached fold Cholesky solver status)");
        if(factor_grew || lane.workspace_capacity == 0) {
          check_cusolver(cusolverDnDpotrf_bufferSize(lane.solver,
            CUBLAS_FILL_MODE_LOWER, size, lane.factor, size,
            &workspace_sizes[system]),
            "cusolverDnDpotrf_bufferSize(cached fold ridge)");
          ensure_capacity(lane.workspace, lane.workspace_capacity,
            workspace_sizes[system],
            "cudaMalloc(cached fold Cholesky solver workspace)");
        } else {
          workspace_sizes[system] = checked_int(
            static_cast<Eigen::Index>(lane.workspace_capacity),
            "cached fold Cholesky solver workspace size");
        }
      }

      std::vector<std::vector<int>> solver_status(
        system_count, std::vector<int>(2 * parameter_count));
      ComputeClock::time_point phase_start;
      if(timings) phase_start = ComputeClock::now();
      const int threads = 256;
      const double alpha = 1.0;
      const double beta = 0.0;
      for(size_t system = 0; system < system_count; ++system) {
        CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
        for(int parameter = 0; parameter < parameter_count; ++parameter) {
          check_cuda(cudaMemcpyAsync(lane.factor, lane.gram,
            gram_elements * sizeof(double), cudaMemcpyDeviceToDevice,
            lane.stream),
            "copy cached fold Cholesky factorization matrix");
          check_cuda(cudaMemcpyAsync(lane.solve, lane.right_hand_sides,
            rhs_elements * sizeof(double), cudaMemcpyDeviceToDevice,
            lane.stream), "copy cached fold Cholesky right-hand sides");
          add_uniform_diagonal_penalty<<<
            (size + threads - 1) / threads, threads, 0, lane.stream>>>(
              lane.factor, ridge_parameters(parameter), size);
          check_cuda(cudaGetLastError(),
            "add cached fold uniform diagonal penalty kernel");
          check_cusolver(cusolverDnDpotrf(lane.solver,
            CUBLAS_FILL_MODE_LOWER, size, lane.factor, size,
            lane.workspace, workspace_sizes[system],
            lane.info + 2 * parameter),
            "cusolverDnDpotrf(cached fold ridge)");
          check_cusolver(cusolverDnDpotrs(lane.solver,
            CUBLAS_FILL_MODE_LOWER, size, right_hand_side_count,
            lane.factor, size, lane.solve, size,
            lane.info + 2 * parameter + 1),
            "cusolverDnDpotrs(cached fold ridge)");
          check_cuda(cudaMemcpyAsync(
            lane.coefficients +
              static_cast<Eigen::Index>(parameter) * rhs_elements,
            lane.solve, rhs_elements * sizeof(double),
            cudaMemcpyDeviceToDevice, lane.stream),
            "store cached fold Cholesky coefficients");
        }

        const int sample_count = checked_int(
          column_counts(static_cast<Eigen::Index>(system)),
          "cached fold Cholesky ridge sample count");
        if(sample_count > 0) {
          const Eigen::Index start =
            start_columns(static_cast<Eigen::Index>(system));
          if(design_orientation) {
            const Eigen::Index sample_count_index =
              column_counts(static_cast<Eigen::Index>(system));
            if(resident_design_uses_level1_cache_) {
              check_cublas(cublasDgemm(lane.blas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                sample_count, combination_count, size, &alpha,
                d_level1_design_ + start,
                checked_int(resident_design_rows_,
                  "persistent Level 1 design leading dimension"),
                lane.coefficients, size, &beta,
                lane.predictions, sample_count),
                "cublasDgemm(persistent Level 1 fold predictions)");
            } else {
              const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
                sample_count_index, size_index);
              double* prediction_chunk = d_projected_ +
                start * resident_design_columns_;
              for(Eigen::Index chunk_start = 0;
                  chunk_start < sample_count_index;
                  chunk_start += chunk_rows) {
                const Eigen::Index count_index = std::min(
                  chunk_rows, sample_count_index - chunk_start);
                const int count = checked_int(count_index,
                  "cached design prediction chunk sample count");
                check_cuda(cudaMemcpy2DAsync(prediction_chunk,
                  static_cast<size_t>(count) * sizeof(double),
                  d_resident_genotypes_ + start + chunk_start,
                  static_cast<size_t>(resident_design_rows_) * sizeof(double),
                  static_cast<size_t>(count) * sizeof(double),
                  static_cast<size_t>(size), cudaMemcpyDeviceToDevice,
                  lane.stream),
                  "stage cached design prediction chunk on CUDA device");
                check_cublas(cublasDgemm(lane.blas,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  count, combination_count, size, &alpha,
                  prediction_chunk, count,
                  lane.coefficients, size, &beta,
                  lane.predictions + chunk_start, sample_count),
                  "cublasDgemm(cached design fold Cholesky predictions)");
              }
            }
          } else {
            const double* prediction_matrix = d_resident_genotypes_ +
              start * resident_rows_;
            check_cublas(cublasDgemm(lane.blas,
              CUBLAS_OP_T, CUBLAS_OP_N,
              sample_count, combination_count, size, &alpha,
              prediction_matrix, size, lane.coefficients, size, &beta,
              lane.predictions, sample_count),
              "cublasDgemm(cached fold Cholesky predictions)");
          }
        }
      }
      synchronize_level0_cholesky_lanes(system_count);
      for(size_t system = 0; system < system_count; ++system)
        check_cuda(cudaMemcpy(solver_status[system].data(),
          level0_cholesky_lanes_[system].info,
          solver_status[system].size() * sizeof(int),
          cudaMemcpyDeviceToHost),
          "copy cached fold Cholesky solver status to host");
      if(timings) timings->ridge_ms += elapsed_ms(phase_start);

      for(size_t system = 0; system < system_count; ++system) {
        for(int parameter = 0; parameter < parameter_count; ++parameter) {
          const int factor_status = solver_status[system][2 * parameter];
          const int solve_status = solver_status[system][2 * parameter + 1];
          if(factor_status != 0 || solve_status != 0) {
            std::ostringstream message;
            message << "cuSOLVER cached fold Cholesky failed for system="
                    << system << " parameter=" << parameter
                    << " factor_info=" << factor_status
                    << " solve_info=" << solve_status;
            throw std::runtime_error(message.str());
          }
        }
      }

      if(copy_results_to_host) {
        if(timings) phase_start = ComputeClock::now();
        for(size_t system = 0; system < system_count; ++system) {
          CudaLevel0CholeskyLane& lane = level0_cholesky_lanes_[system];
          check_cuda(cudaMemcpyAsync(coefficients[system].data(),
            lane.coefficients, coefficients[system].size() * sizeof(double),
            cudaMemcpyDeviceToHost, lane.stream),
            "copy cached fold Cholesky coefficients from CUDA device");
          if(predictions[system].size() > 0)
            check_cuda(cudaMemcpyAsync(predictions[system].data(),
              lane.predictions, predictions[system].size() * sizeof(double),
              cudaMemcpyDeviceToHost, lane.stream),
              "copy cached fold Cholesky predictions from CUDA device");
        }
        synchronize_level0_cholesky_lanes(system_count);
        if(timings) timings->download_ms += elapsed_ms(phase_start);
      }
      if(timings) {
        timings->resident_reuse_count += system_count;
      }
      return true;
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

      const Eigen::Index chunk_samples = bounded_cuda_chunk_rows(
        genotypes.cols(), genotypes.rows());
      const bool genotypes_are_resident =
        resident_genotype_columns(genotypes, 0, genotypes.cols()) != nullptr;
      if(!genotypes_are_resident)
        ensure_capacity(d_genotypes_, genotypes_capacity_,
          chunk_samples * genotypes.rows(), "cudaMalloc(genotype chunk)");
      ensure_capacity(d_gram_, gram_capacity_, gram.size(), "cudaMalloc(Gram matrix)");
      if(phenotype_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_,
          chunk_samples * phenotypes.cols(), "cudaMalloc(phenotype chunk)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, crossproduct.size(), "cudaMalloc(crossproduct)");
      }

      const double alpha = 1.0;
      const bool genotypes_have_contiguous_columns =
        genotypes.innerStride() == 1 &&
        genotypes.outerStride() == genotypes.rows();
      for(Eigen::Index start = 0; start < genotypes.cols();
          start += chunk_samples) {
        const Eigen::Index count_index = std::min(
          chunk_samples, genotypes.cols() - start);
        const int count = checked_int(count_index,
          "genotype product chunk sample count");
        const double* device_genotype_chunk =
          resident_genotype_columns(genotypes, start, count_index);
        Eigen::MatrixXd packed_genotype_chunk;
        const double* genotype_chunk_data = nullptr;
        if(!device_genotype_chunk) {
          if(genotypes_have_contiguous_columns)
            genotype_chunk_data = genotypes.data() +
              start * genotypes.outerStride();
          else {
            packed_genotype_chunk =
              genotypes.middleCols(start, count_index);
            genotype_chunk_data = packed_genotype_chunk.data();
          }
        }
        const Eigen::MatrixXd phenotype_chunk = phenotype_count > 0 ?
          Eigen::MatrixXd(phenotypes.middleRows(start, count_index)) :
          Eigen::MatrixXd();

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        if(!device_genotype_chunk) {
          check_cuda(cudaMemcpy(d_genotypes_, genotype_chunk_data,
            count_index * genotypes.rows() * sizeof(double),
            cudaMemcpyHostToDevice),
            "copy genotype chunk to CUDA device");
          device_genotype_chunk = d_genotypes_;
        } else if(timings) {
          timings->resident_reuse_count++;
        }
        if(phenotype_count > 0)
          check_cuda(cudaMemcpy(d_phenotypes_, phenotype_chunk.data(),
            phenotype_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy phenotype chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        const double beta = start == 0 ? 0.0 : 1.0;
        if(phenotype_count > 0) {
          std::unique_ptr<CudaEventPair> crossproduct_events;
          if(timings) {
            crossproduct_events.reset(new CudaEventPair());
            crossproduct_events->record_start();
          }
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            blocks, phenotype_count, count, &alpha,
            device_genotype_chunk, blocks, d_phenotypes_, count, &beta,
            d_crossproduct_, blocks),
            "cublasDgemm(genotype product chunk)");
          if(timings)
            timings->crossproduct_ms +=
              crossproduct_events->record_stop_and_elapsed_ms();
        }

        std::unique_ptr<CudaEventPair> gram_events;
        if(timings) {
          gram_events.reset(new CudaEventPair());
          gram_events->record_start();
        }
        if(mode == Step1GramMode::selfadjoint_rank_update)
          check_cublas(cublasDsyrk(handle_, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, blocks, count, &alpha,
            device_genotype_chunk, blocks,
            &beta, d_gram_, blocks),
            "cublasDsyrk(genotype Gram chunk)");
        else
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            blocks, blocks, count, &alpha,
            device_genotype_chunk, blocks,
            device_genotype_chunk, blocks, &beta,
            d_gram_, blocks), "cublasDgemm(genotype Gram chunk)");
        if(timings)
          timings->gram_ms += gram_events->record_stop_and_elapsed_ms();
      }

      if(mode == Step1GramMode::selfadjoint_rank_update) {
        const dim3 threads(16, 16);
        const dim3 grid((blocks + threads.x - 1) / threads.x,
                        (blocks + threads.y - 1) / threads.y);
        mirror_lower_triangle<<<grid, threads>>>(d_gram_, blocks);
        check_cuda(cudaGetLastError(), "mirror Gram triangle kernel");
      }

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(phenotype_count > 0)
        check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_, crossproduct.size() * sizeof(double),
          cudaMemcpyDeviceToHost), "copy crossproduct from CUDA device");
      check_cuda(cudaMemcpy(gram.data(), d_gram_, gram.size() * sizeof(double),
        cudaMemcpyDeviceToHost), "copy Gram matrix from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

    void compute_design_products(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings) override {

      if(design.rows() != outcomes.rows())
        throw std::invalid_argument(
          "Step 1 design products received incompatible design and outcome matrices");
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int samples = checked_int(design.rows(), "design sample count");
      const int features = checked_int(design.cols(), "design feature count");
      const int outcome_count = checked_int(outcomes.cols(), "design outcome count");
      gram.resize(features, features);
      crossproduct.resize(features, outcome_count);
      if(features == 0 || samples == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }

      const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
        design.rows(), design.cols());
      ensure_capacity(d_genotypes_, genotypes_capacity_,
        chunk_rows * design.cols(), "cudaMalloc(design product chunk)");
      ensure_capacity(d_gram_, gram_capacity_, gram.size(), "cudaMalloc(design Gram matrix)");
      if(outcome_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_,
          chunk_rows * outcomes.cols(), "cudaMalloc(design outcome chunk)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, crossproduct.size(),
          "cudaMalloc(design crossproduct)");
      }

      const double alpha = 1.0;
      for(Eigen::Index start = 0; start < design.rows(); start += chunk_rows) {
        const Eigen::Index count_index = std::min(
          chunk_rows, design.rows() - start);
        const int count = checked_int(
          count_index, "design product chunk row count");

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        copy_matrix_row_chunk_to_device(design, start, count_index,
          d_genotypes_, "copy design product chunk to CUDA device");
        if(outcome_count > 0)
          copy_matrix_row_chunk_to_device(outcomes, start, count_index,
            d_phenotypes_,
            "copy design product outcome chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        const double beta = start == 0 ? 0.0 : 1.0;
        if(outcome_count > 0) {
          std::unique_ptr<CudaEventPair> crossproduct_events;
          if(timings) {
            crossproduct_events.reset(new CudaEventPair());
            crossproduct_events->record_start();
          }
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            features, outcome_count, count, &alpha,
            d_genotypes_, count, d_phenotypes_, count, &beta,
            d_crossproduct_, features),
            "cublasDgemm(design product crossproduct chunk)");
          if(timings)
            timings->crossproduct_ms +=
              crossproduct_events->record_stop_and_elapsed_ms();
        }

        std::unique_ptr<CudaEventPair> gram_events;
        if(timings) {
          gram_events.reset(new CudaEventPair());
          gram_events->record_start();
        }
        check_cublas(cublasDsyrk(handle_, CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_T, features, count, &alpha, d_genotypes_, count,
          &beta, d_gram_, features),
          "cublasDsyrk(design Gram chunk)");
        if(timings)
          timings->gram_ms += gram_events->record_stop_and_elapsed_ms();
      }

      const dim3 threads(16, 16);
      const dim3 grid((features + threads.x - 1) / threads.x,
                      (features + threads.y - 1) / threads.y);
      mirror_lower_triangle<<<grid, threads>>>(d_gram_, features);
      check_cuda(cudaGetLastError(), "mirror design Gram triangle kernel");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(outcome_count > 0)
        check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_,
          crossproduct.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "copy design crossproduct from CUDA device");
      check_cuda(cudaMemcpy(gram.data(), d_gram_, gram.size() * sizeof(double),
        cudaMemcpyDeviceToHost), "copy design Gram matrix from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

    void compute_design_crossproduct(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings) override {

      if(design.rows() != outcomes.rows())
        throw std::invalid_argument(
          "Step 1 design crossproduct received incompatible dimensions");
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int samples = checked_int(
        design.rows(), "design crossproduct sample count");
      const int features = checked_int(
        design.cols(), "design crossproduct feature count");
      const int outcome_count = checked_int(
        outcomes.cols(), "design crossproduct outcome count");
      crossproduct.resize(features, outcome_count);
      if(features == 0 || outcome_count == 0 || samples == 0) {
        crossproduct.setZero();
        return;
      }

      const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
        design.rows(), design.cols());
      ensure_capacity(d_genotypes_, genotypes_capacity_,
        chunk_rows * design.cols(),
        "cudaMalloc(design crossproduct design)");
      ensure_capacity(d_phenotypes_, phenotypes_capacity_,
        chunk_rows * outcomes.cols(),
        "cudaMalloc(design crossproduct outcomes)");
      ensure_capacity(d_crossproduct_, crossproduct_capacity_, crossproduct.size(),
        "cudaMalloc(design crossproduct)");
      const double alpha = 1.0;
      for(Eigen::Index start = 0; start < design.rows(); start += chunk_rows) {
        const Eigen::Index count_index = std::min(
          chunk_rows, design.rows() - start);
        const int count = checked_int(
          count_index, "design crossproduct chunk row count");
        const Eigen::MatrixXd design_chunk = design.middleRows(start, count_index);
        const Eigen::MatrixXd outcomes_chunk = outcomes.middleRows(start, count_index);

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
          design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy design crossproduct chunk to CUDA device");
        check_cuda(cudaMemcpy(d_phenotypes_, outcomes_chunk.data(),
          outcomes_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy design crossproduct outcome chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        std::unique_ptr<CudaEventPair> crossproduct_events;
        if(timings) {
          crossproduct_events.reset(new CudaEventPair());
          crossproduct_events->record_start();
        }
        const double beta = start == 0 ? 0.0 : 1.0;
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          features, outcome_count, count, &alpha,
          d_genotypes_, count, d_phenotypes_, count, &beta,
          d_crossproduct_, features),
          "cublasDgemm(design-only crossproduct chunk)");
        if(timings) timings->crossproduct_ms +=
          crossproduct_events->record_stop_and_elapsed_ms();
      }

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_,
        crossproduct.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy design-only crossproduct from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

    void compute_weighted_design_products(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& weights,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings) override {

      if(design.rows() != weights.size() || design.rows() != outcomes.rows())
        throw std::invalid_argument(
          "Step 1 weighted design products received incompatible dimensions");
      if(!weights.allFinite() || (weights.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 weighted design products require finite non-negative weights");
      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int samples = checked_int(design.rows(), "weighted design sample count");
      const int features = checked_int(design.cols(), "weighted design feature count");
      const int outcome_count = checked_int(outcomes.cols(), "weighted design outcome count");
      gram.resize(features, features);
      crossproduct.resize(features, outcome_count);
      if(features == 0 || samples == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }

      const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
        design.rows(), design.cols());
      ensure_capacity(d_genotypes_, genotypes_capacity_,
        chunk_rows * design.cols(), "cudaMalloc(weighted design chunk)");
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_, chunk_rows,
        "cudaMalloc(design weight chunk)");
      ensure_capacity(d_projected_, projected_capacity_,
        chunk_rows * design.cols(), "cudaMalloc(weighted design matrix chunk)");
      ensure_capacity(d_gram_, gram_capacity_, gram.size(),
        "cudaMalloc(weighted Gram matrix)");
      if(outcome_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_,
          chunk_rows * outcomes.cols(), "cudaMalloc(weighted outcome chunk)");
        ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_,
          chunk_rows * outcomes.cols(),
          "cudaMalloc(weighted outcome matrix chunk)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, crossproduct.size(),
          "cudaMalloc(weighted crossproduct)");
      }

      const double alpha = 1.0;
      const int threads = 256;
      for(Eigen::Index start = 0; start < design.rows(); start += chunk_rows) {
        const Eigen::Index count_index = std::min(
          chunk_rows, design.rows() - start);
        const int count = checked_int(
          count_index, "weighted design product chunk row count");
        const Eigen::MatrixXd design_chunk =
          design.middleRows(start, count_index);
        const Eigen::VectorXd weights_chunk =
          weights.segment(start, count_index);
        const Eigen::MatrixXd outcomes_chunk = outcome_count > 0 ?
          Eigen::MatrixXd(outcomes.middleRows(start, count_index)) :
          Eigen::MatrixXd();

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
          design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy weighted design chunk to CUDA device");
        check_cuda(cudaMemcpy(d_ridge_parameters_, weights_chunk.data(),
          weights_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy design weight chunk to CUDA device");
        if(outcome_count > 0)
          check_cuda(cudaMemcpy(d_phenotypes_, outcomes_chunk.data(),
            outcomes_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy weighted outcome chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        const int design_count = checked_element_count(
          count, features, "weighted design chunk");
        scale_matrix_rows<<<
          (design_count + threads - 1) / threads, threads>>>(
          d_genotypes_, d_ridge_parameters_, d_projected_, count,
          design_count);
        check_cuda(cudaGetLastError(),
          "scale weighted design chunk rows kernel");
        if(outcome_count > 0) {
          const int outcome_element_count = checked_element_count(
            count, outcome_count, "weighted outcome chunk");
          scale_matrix_rows<<<
            (outcome_element_count + threads - 1) / threads, threads>>>(
            d_phenotypes_, d_ridge_parameters_, d_scaled_rhs_, count,
            outcome_element_count);
          check_cuda(cudaGetLastError(),
            "scale weighted outcome chunk rows kernel");
        }

        const double beta = start == 0 ? 0.0 : 1.0;
        if(outcome_count > 0) {
          std::unique_ptr<CudaEventPair> crossproduct_events;
          if(timings) {
            crossproduct_events.reset(new CudaEventPair());
            crossproduct_events->record_start();
          }
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            features, outcome_count, count, &alpha,
            d_genotypes_, count, d_scaled_rhs_, count, &beta,
            d_crossproduct_, features),
            "cublasDgemm(weighted crossproduct chunk)");
          if(timings)
            timings->crossproduct_ms +=
              crossproduct_events->record_stop_and_elapsed_ms();
        }

        std::unique_ptr<CudaEventPair> gram_events;
        if(timings) {
          gram_events.reset(new CudaEventPair());
          gram_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          features, features, count, &alpha,
          d_genotypes_, count, d_projected_, count, &beta,
          d_gram_, features), "cublasDgemm(weighted Gram chunk)");
        if(timings)
          timings->gram_ms += gram_events->record_stop_and_elapsed_ms();
      }

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(outcome_count > 0)
        check_cuda(cudaMemcpy(crossproduct.data(), d_crossproduct_,
          crossproduct.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "copy weighted crossproduct from CUDA device");
      check_cuda(cudaMemcpy(gram.data(), d_gram_, gram.size() * sizeof(double),
        cudaMemcpyDeviceToHost), "copy weighted Gram matrix from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

    void diagonal_penalty_solve(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings) override {

      const Eigen::Index size_index = gram.rows();
      if(gram.cols() != size_index || right_hand_sides.rows() != size_index ||
         penalty_multipliers.size() != size_index)
        throw std::invalid_argument(
          "Step 1 diagonal-penalty solve received incompatible matrix dimensions");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 diagonal-penalty parameters must be non-negative");
      if((penalty_multipliers.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 diagonal-penalty multipliers must be non-negative");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = checked_int(size_index, "diagonal-penalty matrix size");
      const int right_hand_side_count = checked_int(
        right_hand_sides.cols(), "diagonal-penalty right-hand-side count");
      const int parameter_count = checked_int(
        ridge_parameters.size(), "diagonal-penalty parameter count");
      const long long solution_column_count_long =
        static_cast<long long>(right_hand_side_count) * parameter_count;
      if(solution_column_count_long > INT_MAX)
        throw std::runtime_error(
          "CUDA diagonal-penalty solution column count exceeds integer limits");
      solutions.resize(size, static_cast<Eigen::Index>(solution_column_count_long));
      if(size == 0 || right_hand_side_count == 0 || parameter_count == 0) {
        solutions.setZero();
        return;
      }

      ensure_capacity(d_gram_, gram_capacity_, gram.size(),
        "cudaMalloc(diagonal-penalty Gram matrix)");
      ensure_capacity(d_crossproduct_, crossproduct_capacity_, right_hand_sides.size(),
        "cudaMalloc(diagonal-penalty right-hand sides)");

      const Eigen::MatrixXd packed_gram = contiguous_copy_if_needed(gram);
      const Eigen::MatrixXd packed_rhs = contiguous_copy_if_needed(right_hand_sides);
      const double* gram_data = packed_gram.size() ? packed_gram.data() : gram.data();
      const double* rhs_data = packed_rhs.size() ?
        packed_rhs.data() : right_hand_sides.data();

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_gram_, gram_data, gram.size() * sizeof(double),
        cudaMemcpyHostToDevice), "copy diagonal-penalty Gram matrix to CUDA device");
      check_cuda(cudaMemcpy(d_crossproduct_, rhs_data,
        right_hand_sides.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy diagonal-penalty right-hand sides to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);
      diagonal_penalty_solve_device(size, right_hand_side_count,
        ridge_parameters, penalty_multipliers, solutions, timings);
    }

    void diagonal_penalty_predict(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      const Eigen::Ref<const Eigen::MatrixXd>& prediction_matrix,
      bool samples_in_columns,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      const Eigen::Ref<const Eigen::MatrixXd>& leave_one_out_outcomes,
      bool leave_one_out,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings) override {

      const Eigen::Index size_index = gram.rows();
      const Eigen::Index sample_count_index = samples_in_columns ?
        prediction_matrix.cols() : prediction_matrix.rows();
      const Eigen::Index outcome_count_index = right_hand_sides.cols();
      const Eigen::Index parameter_count_index = ridge_parameters.size();
      if(gram.cols() != size_index || right_hand_sides.rows() != size_index ||
         penalty_multipliers.size() != size_index ||
         (samples_in_columns ? prediction_matrix.rows() : prediction_matrix.cols()) !=
           size_index)
        throw std::invalid_argument(
          "Step 1 diagonal-penalty solve received incompatible matrix dimensions");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 diagonal-penalty parameters must be non-negative");
      if((penalty_multipliers.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 diagonal-penalty multipliers must be non-negative");
      if(leave_one_out &&
         (leave_one_out_outcomes.rows() != sample_count_index ||
          leave_one_out_outcomes.cols() != outcome_count_index))
        throw std::invalid_argument(
          "Step 1 diagonal-penalty LOOCV outcomes have incompatible dimensions");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = checked_int(size_index, "diagonal-penalty matrix size");
      const int sample_count = checked_int(
        sample_count_index, "diagonal-penalty sample count");
      const int outcome_count = checked_int(
        outcome_count_index, "diagonal-penalty outcome count");
      const int parameter_count = checked_int(
        parameter_count_index, "diagonal-penalty parameter count");
      const long long combination_count_long =
        static_cast<long long>(outcome_count) * parameter_count;
      if(combination_count_long > INT_MAX)
        throw std::runtime_error(
          "CUDA diagonal-penalty outcome/parameter count exceeds integer limits");
      const int combination_count = static_cast<int>(combination_count_long);

      predictions.resize(sample_count, combination_count);
      coefficients.resize(size, combination_count);
      if(size == 0 || sample_count == 0 || combination_count == 0) {
        predictions.setZero();
        coefficients.setZero();
        return;
      }

      if(!leave_one_out) {
        diagonal_penalty_solve(gram, right_hand_sides, ridge_parameters,
          penalty_multipliers, coefficients, timings);

        const Eigen::Index streaming_columns = std::max<Eigen::Index>(
          size, combination_count);
        const Eigen::Index chunk_samples = bounded_cuda_chunk_rows(
          sample_count_index, streaming_columns);
        ensure_capacity(d_genotypes_, genotypes_capacity_,
          chunk_samples * size,
          "cudaMalloc(diagonal-penalty prediction chunk)");
        ensure_capacity(d_inverse_, inverse_capacity_, coefficients.size(),
          "cudaMalloc(diagonal-penalty streamed coefficients)");
        ensure_capacity(d_predictions_, predictions_capacity_,
          chunk_samples * combination_count,
          "cudaMalloc(diagonal-penalty prediction result chunk)");

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_inverse_, coefficients.data(),
          coefficients.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy diagonal-penalty streamed coefficients to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        const double alpha = 1.0;
        const double beta = 0.0;
        for(Eigen::Index start = 0; start < sample_count_index;
            start += chunk_samples) {
          const Eigen::Index count_index = std::min(
            chunk_samples, sample_count_index - start);
          const int count = checked_int(
            count_index, "diagonal-penalty prediction chunk sample count");
          Eigen::MatrixXd prediction_chunk;
          if(samples_in_columns)
            prediction_chunk =
              prediction_matrix.middleCols(start, count_index);
          else
            prediction_chunk =
              prediction_matrix.middleRows(start, count_index);

          if(timings) transfer_start = ComputeClock::now();
          check_cuda(cudaMemcpy(d_genotypes_, prediction_chunk.data(),
            prediction_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy diagonal-penalty prediction chunk to CUDA device");
          if(timings) timings->upload_ms += elapsed_ms(transfer_start);

          std::unique_ptr<CudaEventPair> prediction_events;
          if(timings) {
            prediction_events.reset(new CudaEventPair());
            prediction_events->record_start();
          }
          if(samples_in_columns)
            check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
              count, combination_count, size, &alpha,
              d_genotypes_, size, d_inverse_, size, &beta,
              d_predictions_, count),
              "cublasDgemm(diagonal-penalty prediction chunk)");
          else
            check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
              count, combination_count, size, &alpha,
              d_genotypes_, count, d_inverse_, size, &beta,
              d_predictions_, count),
              "cublasDgemm(design diagonal-penalty prediction chunk)");
          if(timings) timings->ridge_ms +=
            prediction_events->record_stop_and_elapsed_ms();

          if(timings) transfer_start = ComputeClock::now();
          for(int combination = 0; combination < combination_count;
              ++combination)
            check_cuda(cudaMemcpy(
              predictions.col(combination).segment(start, count_index).data(),
              d_predictions_ +
                static_cast<Eigen::Index>(combination) * count,
              count * sizeof(double), cudaMemcpyDeviceToHost),
              "copy diagonal-penalty prediction chunk from CUDA device");
          if(timings) timings->download_ms += elapsed_ms(transfer_start);
        }
        return;
      }

      // Reuse the resident Cholesky factor and the bounded grouped-LOOCV
      // primitive. Level 1 calls normally contain one outcome, while this
      // loop preserves the general multi-outcome contract without staging a
      // sample-by-feature influence matrix for every ridge parameter.
      const Eigen::VectorXd leverage_weights =
        Eigen::VectorXd::Ones(sample_count);
      const Eigen::VectorXi full_group_offset =
        Eigen::VectorXi::Zero(1);
      const Eigen::VectorXi full_group_size =
        Eigen::VectorXi::Constant(1, size);
      const int saved_factorized_size = factorized_size_;
      if(saved_factorized_size > 0) {
        ensure_capacity(d_gram_, gram_capacity_,
          static_cast<Eigen::Index>(saved_factorized_size) *
            saved_factorized_size,
          "cudaMalloc(saved diagonal-penalty factorization)");
        check_cuda(cudaMemcpy(d_gram_, d_factorized_,
          static_cast<size_t>(saved_factorized_size) *
            saved_factorized_size * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "save reusable diagonal-penalty factorization");
      }
      const auto restore_factorized_state = [&] () {
        if(saved_factorized_size > 0)
          check_cuda(cudaMemcpy(d_factorized_, d_gram_,
            static_cast<size_t>(saved_factorized_size) *
              saved_factorized_size * sizeof(double),
            cudaMemcpyDeviceToDevice),
            "restore reusable diagonal-penalty factorization");
        factorized_size_ = saved_factorized_size;
      };
      const auto run_leave_one_out = [&] (
          const Eigen::Ref<const Eigen::MatrixXd>& loo_design) {
        for(int parameter = 0; parameter < parameter_count; ++parameter) {
          factorize_diagonal_penalty(gram, ridge_parameters(parameter),
            penalty_multipliers, timings);
          Eigen::MatrixXd parameter_coefficients;
          solve_factorized(right_hand_sides, parameter_coefficients, timings);
          coefficients.middleCols(
            static_cast<Eigen::Index>(parameter) * outcome_count,
            outcome_count) = parameter_coefficients;
          for(int outcome = 0; outcome < outcome_count; ++outcome) {
            Eigen::MatrixXd outcome_predictions;
            grouped_leave_one_out_predict_factorized_impl(
              loo_design, parameter_coefficients.col(outcome),
              leave_one_out_outcomes.col(outcome), leverage_weights,
              full_group_offset, full_group_size, true,
              outcome_predictions, timings);
            predictions.col(
              static_cast<Eigen::Index>(parameter) * outcome_count +
              outcome) = outcome_predictions.col(0);
          }
        }
      };
      try {
        if(samples_in_columns) {
          const Eigen::MatrixXd transposed_prediction =
            prediction_matrix.transpose();
          run_leave_one_out(transposed_prediction);
        } else {
          run_leave_one_out(prediction_matrix);
        }
      } catch(...) {
        restore_factorized_state();
        throw;
      }
      restore_factorized_state();
      return;

    }

    void factorize_diagonal_penalty(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      double ridge_parameter,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      Step1ComputeTimings* timings) override {

      if(gram.rows() != gram.cols() || penalty_multipliers.size() != gram.rows())
        throw std::invalid_argument(
          "Step 1 reusable factorization received incompatible matrix dimensions");
      if(ridge_parameter < 0)
        throw std::invalid_argument(
          "Step 1 reusable factorization parameter must be non-negative");
      if((penalty_multipliers.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 reusable factorization multipliers must be non-negative");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = checked_int(gram.rows(), "reusable factorization matrix size");
      factorized_size_ = -1;
      if(size == 0) {
        factorized_size_ = 0;
        return;
      }
      ensure_capacity(d_factorized_, factorized_capacity_, gram.size(),
        "cudaMalloc(reusable factorization matrix)");
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_,
        penalty_multipliers.size(), "cudaMalloc(reusable factorization multipliers)");

      const Eigen::MatrixXd packed_gram = contiguous_copy_if_needed(gram);
      const double* gram_data = packed_gram.size() ? packed_gram.data() : gram.data();
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_factorized_, gram_data, gram.size() * sizeof(double),
        cudaMemcpyHostToDevice), "copy reusable factorization matrix to CUDA device");
      check_cuda(cudaMemcpy(d_ridge_parameters_, penalty_multipliers.data(),
        penalty_multipliers.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy reusable factorization multipliers to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      const int threads = 256;
      add_diagonal_penalty<<<(size + threads - 1) / threads, threads>>>(
        d_factorized_, d_ridge_parameters_, ridge_parameter, size);
      check_cuda(cudaGetLastError(), "add reusable diagonal penalty kernel");

      int workspace_size = 0;
      check_cusolver(cusolverDnDpotrf_bufferSize(solver_handle_,
        CUBLAS_FILL_MODE_LOWER, size, d_factorized_, size, &workspace_size),
        "cusolverDnDpotrf_bufferSize(reusable factorization)");
      ensure_capacity(d_solver_workspace_, solver_workspace_capacity_, workspace_size,
        "cudaMalloc(cuSOLVER reusable Cholesky workspace)");

      std::unique_ptr<CudaEventPair> solve_events;
      if(timings) {
        solve_events.reset(new CudaEventPair());
        solve_events->record_start();
      }
      check_cusolver(cusolverDnDpotrf(solver_handle_, CUBLAS_FILL_MODE_LOWER,
        size, d_factorized_, size, d_solver_workspace_, workspace_size,
        d_solver_info_), "cusolverDnDpotrf(reusable factorization)");
      if(timings) timings->ridge_ms += solve_events->record_stop_and_elapsed_ms();

      int solver_info = 0;
      check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
        cudaMemcpyDeviceToHost), "copy reusable Cholesky factorization status to host");
      if(solver_info != 0) {
        std::ostringstream message;
        message << "cuSOLVER reusable Cholesky factorization failed with info="
                << solver_info;
        throw std::runtime_error(message.str());
      }
      factorized_size_ = size;
    }

    void solve_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings) override {

      if(factorized_size_ < 0)
        throw std::runtime_error(
          "Step 1 reusable solve requested before factorization");
      if(right_hand_sides.rows() != factorized_size_)
        throw std::invalid_argument(
          "Step 1 reusable solve received incompatible right-hand sides");
      solutions.resize(right_hand_sides.rows(), right_hand_sides.cols());
      if(right_hand_sides.size() == 0) {
        solutions.setZero();
        return;
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      checked_int(right_hand_sides.cols(),
        "reusable solve right-hand-side count");
      const Eigen::Index chunk_columns = bounded_cuda_chunk_rows(
        right_hand_sides.cols(), factorized_size_);
      ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_,
        chunk_columns * factorized_size_,
        "cudaMalloc(reusable solve right-hand-side chunk)");

      for(Eigen::Index start = 0; start < right_hand_sides.cols();
          start += chunk_columns) {
        const Eigen::Index count_index = std::min(
          chunk_columns, right_hand_sides.cols() - start);
        const int count = checked_int(
          count_index, "reusable solve chunk right-hand-side count");
        const Eigen::MatrixXd rhs_chunk =
          right_hand_sides.middleCols(start, count_index);

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_scaled_rhs_, rhs_chunk.data(),
          rhs_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy reusable solve right-hand-side chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        std::unique_ptr<CudaEventPair> solve_events;
        if(timings) {
          solve_events.reset(new CudaEventPair());
          solve_events->record_start();
        }
        check_cusolver(cusolverDnDpotrs(solver_handle_,
          CUBLAS_FILL_MODE_LOWER, factorized_size_, count, d_factorized_,
          factorized_size_, d_scaled_rhs_, factorized_size_, d_solver_info_),
          "cusolverDnDpotrs(reusable solve chunk)");
        if(timings)
          timings->ridge_ms +=
            solve_events->record_stop_and_elapsed_ms();

        int solver_info = 0;
        check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
          cudaMemcpyDeviceToHost),
          "copy reusable Cholesky chunk solve status to host");
        if(solver_info != 0) {
          std::ostringstream message;
          message << "cuSOLVER reusable Cholesky chunk solve failed with info="
                  << solver_info;
          throw std::runtime_error(message.str());
        }

        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(solutions.col(start).data(), d_scaled_rhs_,
          rhs_chunk.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "copy reusable solve result chunk from CUDA device");
        if(timings) timings->download_ms += elapsed_ms(transfer_start);
      }
    }

    void grouped_leave_one_out_predict_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      const Eigen::Ref<const Eigen::VectorXd>& residuals,
      const Eigen::Ref<const Eigen::VectorXd>& leverage_weights,
      const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
      const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
      Eigen::MatrixXd& predictions,
      Step1ComputeTimings* timings) override {

      grouped_leave_one_out_predict_factorized_impl(
        design, coefficients, residuals, leverage_weights,
        group_offsets, group_sizes, false, predictions, timings);
    }

    void grouped_leave_one_out_predict_factorized_impl(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      const Eigen::Ref<const Eigen::VectorXd>& residuals_or_outcomes,
      const Eigen::Ref<const Eigen::VectorXd>& leverage_weights,
      const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
      const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
      bool inputs_are_outcomes,
      Eigen::MatrixXd& predictions,
      Step1ComputeTimings* timings) {

      if(factorized_size_ < 0)
        throw std::runtime_error(
          "Step 1 grouped LOOCV prediction requested before factorization");
      if(design.cols() != factorized_size_ ||
         coefficients.size() != factorized_size_ ||
         design.rows() != residuals_or_outcomes.size() ||
         design.rows() != leverage_weights.size() ||
         group_offsets.size() != group_sizes.size())
        throw std::invalid_argument(
          "Step 1 grouped LOOCV prediction received incompatible dimensions");
      if(!coefficients.allFinite() || !residuals_or_outcomes.allFinite() ||
         !leverage_weights.allFinite())
        throw std::invalid_argument(
          "Step 1 grouped LOOCV prediction requires finite inputs");
      if((leverage_weights.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 grouped LOOCV prediction requires non-negative leverage weights");
      for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
        if(group_offsets(group) < 0 || group_sizes(group) < 0 ||
           group_offsets(group) > design.cols() - group_sizes(group))
          throw std::invalid_argument(
            "Step 1 grouped LOOCV prediction received an invalid feature group");
      }
      if(inputs_are_outcomes &&
         (group_offsets.size() != 1 || group_offsets(0) != 0 ||
          group_sizes(0) != design.cols()))
        throw std::invalid_argument(
          "Step 1 outcome-based LOOCV prediction requires one full feature group");

      predictions.resize(design.rows(), group_offsets.size());
      if(design.rows() == 0 || group_offsets.size() == 0 || factorized_size_ == 0) {
        predictions.setZero();
        return;
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = factorized_size_;
      const int group_count = checked_int(
        group_offsets.size(), "grouped LOOCV group count");

      const Eigen::Index streaming_columns = std::max<Eigen::Index>(
        size, group_count);
      const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
        design.rows(), streaming_columns);
      ensure_capacity(d_genotypes_, genotypes_capacity_,
        chunk_rows * size, "cudaMalloc(grouped LOOCV design chunk)");
      ensure_capacity(d_projected_, projected_capacity_,
        chunk_rows * size,
        "cudaMalloc(grouped LOOCV transposed design chunk)");
      ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_,
        chunk_rows * size,
        "cudaMalloc(grouped LOOCV influence solve chunk)");
      ensure_capacity(d_inverse_, inverse_capacity_, coefficients.size(),
        "cudaMalloc(grouped LOOCV coefficients)");
      ensure_capacity(d_outcomes_, outcomes_capacity_, chunk_rows,
        "cudaMalloc(grouped LOOCV residual chunk)");
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_,
        chunk_rows, "cudaMalloc(grouped LOOCV leverage weight chunk)");
      ensure_capacity(d_leverage_, leverage_capacity_, chunk_rows,
        "cudaMalloc(grouped LOOCV leverage chunk)");
      ensure_capacity(d_predictions_, predictions_capacity_,
        chunk_rows * group_count,
        "cudaMalloc(grouped LOOCV prediction chunk)");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_inverse_, coefficients.data(),
        coefficients.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy grouped LOOCV coefficients to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      const double alpha = 1.0;
      const double beta = 0.0;
      const int threads = 256;
      for(Eigen::Index start = 0; start < design.rows(); start += chunk_rows) {
        const Eigen::Index count_index = std::min(
          chunk_rows, design.rows() - start);
        const int count = checked_int(
          count_index, "grouped LOOCV chunk row count");
        const Eigen::MatrixXd design_chunk =
          design.middleRows(start, count_index);
        const Eigen::VectorXd residual_or_outcome_chunk =
          residuals_or_outcomes.segment(start, count_index);
        const Eigen::VectorXd weight_chunk =
          leverage_weights.segment(start, count_index);

        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
          design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy grouped LOOCV design chunk to CUDA device");
        check_cuda(cudaMemcpy(d_outcomes_, residual_or_outcome_chunk.data(),
          residual_or_outcome_chunk.size() * sizeof(double),
          cudaMemcpyHostToDevice),
          inputs_are_outcomes ?
            "copy grouped LOOCV outcome chunk to CUDA device" :
            "copy grouped LOOCV residual chunk to CUDA device");
        check_cuda(cudaMemcpy(d_ridge_parameters_, weight_chunk.data(),
          weight_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy grouped LOOCV leverage weight chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        std::unique_ptr<CudaEventPair> solve_events;
        if(timings) {
          solve_events.reset(new CudaEventPair());
          solve_events->record_start();
        }
        check_cublas(cublasDgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
          size, count, &alpha, d_genotypes_, count,
          &beta, d_genotypes_, count, d_projected_, size),
          "cublasDgeam(transpose grouped LOOCV design chunk)");
        check_cuda(cudaMemcpy(d_scaled_rhs_, d_projected_,
          design_chunk.size() * sizeof(double), cudaMemcpyDeviceToDevice),
          "copy grouped LOOCV influence chunk right-hand sides");
        check_cusolver(cusolverDnDpotrs(solver_handle_,
          CUBLAS_FILL_MODE_LOWER, size, count, d_factorized_, size,
          d_scaled_rhs_, size, d_solver_info_),
          "cusolverDnDpotrs(grouped LOOCV influence chunk)");
        int solver_info = 0;
        check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
          cudaMemcpyDeviceToHost),
          "copy grouped LOOCV chunk solve status to host");
        if(solver_info != 0) {
          std::ostringstream message;
          message << "cuSOLVER grouped LOOCV chunk solve failed with info="
                  << solver_info;
          throw std::runtime_error(message.str());
        }

        check_cublas(cublasDgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
          count, size, &alpha, d_scaled_rhs_, size,
          &beta, d_scaled_rhs_, size, d_projected_, count),
          "cublasDgeam(transpose grouped LOOCV influence chunk)");
        compute_weighted_leverage_diagonal<<<
          (count + threads - 1) / threads, threads>>>(
          d_genotypes_, d_projected_, d_ridge_parameters_, d_leverage_,
          size, count);
        check_cuda(cudaGetLastError(),
          "compute grouped LOOCV weighted leverage chunk kernel");
        for(int group = 0; group < group_count; ++group) {
          const int group_size = group_sizes(group);
          double* group_predictions = d_predictions_ + group * count;
          if(group_size > 0)
            check_cublas(cublasDgemv(handle_, CUBLAS_OP_N,
              count, group_size, &alpha,
              d_genotypes_ +
                static_cast<Eigen::Index>(group_offsets(group)) * count,
              count, d_inverse_ + group_offsets(group), 1,
              &beta, group_predictions, 1),
              "cublasDgemv(grouped LOOCV base prediction chunk)");
          else
            check_cuda(cudaMemset(group_predictions, 0,
              count * sizeof(double)),
              "clear empty grouped LOOCV prediction chunk");
          if(inputs_are_outcomes) {
            apply_leave_one_out_correction<<<
              (count + threads - 1) / threads, threads>>>(
              group_predictions, d_leverage_, d_outcomes_, count, 1,
              count);
            check_cuda(cudaGetLastError(),
              "apply full-group LOOCV outcome correction chunk kernel");
          } else {
            grouped_leave_one_out_predictions<<<
              (count + threads - 1) / threads, threads>>>(
              d_genotypes_, d_projected_, d_outcomes_, d_leverage_,
              d_predictions_, count, group,
              group_offsets(group), group_size);
            check_cuda(cudaGetLastError(),
              "compute grouped LOOCV prediction chunk kernel");
          }
        }
        if(timings) timings->ridge_ms +=
          solve_events->record_stop_and_elapsed_ms();

        if(timings) transfer_start = ComputeClock::now();
        for(int group = 0; group < group_count; ++group)
          check_cuda(cudaMemcpy(
            predictions.col(group).segment(start, count_index).data(),
            d_predictions_ + group * count, count * sizeof(double),
            cudaMemcpyDeviceToHost),
            "copy grouped LOOCV prediction chunk from CUDA device");
        if(timings) timings->download_ms += elapsed_ms(transfer_start);
      }
    }

    void grouped_predict(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
      const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
      Eigen::MatrixXd& predictions,
      Step1ComputeTimings* timings) override {

      if(design.cols() != coefficients.size() ||
         group_offsets.size() != group_sizes.size())
        throw std::invalid_argument(
          "Step 1 grouped prediction received incompatible dimensions");
      if(!coefficients.allFinite())
        throw std::invalid_argument(
          "Step 1 grouped prediction requires finite coefficients");
      for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
        if(group_offsets(group) < 0 || group_sizes(group) < 0 ||
           group_offsets(group) > design.cols() - group_sizes(group))
          throw std::invalid_argument(
            "Step 1 grouped prediction received an invalid feature group");
      }

      predictions.resize(design.rows(), group_offsets.size());
      if(design.rows() == 0 || group_offsets.size() == 0) {
        predictions.setZero();
        return;
      }

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int group_count = checked_int(
        group_offsets.size(), "grouped prediction group count");
      const Eigen::Index chunk_rows = bounded_cuda_chunk_rows(
        design.rows(), design.cols());
      ensure_capacity(d_genotypes_, genotypes_capacity_,
        chunk_rows * design.cols(),
        "cudaMalloc(grouped prediction design)");
      ensure_capacity(d_inverse_, inverse_capacity_, coefficients.size(),
        "cudaMalloc(grouped prediction coefficients)");
      ensure_capacity(d_predictions_, predictions_capacity_,
        chunk_rows * group_offsets.size(),
        "cudaMalloc(grouped predictions)");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      if(coefficients.size() > 0)
        check_cuda(cudaMemcpy(d_inverse_, coefficients.data(),
          coefficients.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy grouped prediction coefficients to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      const double alpha = 1.0;
      const double beta = 0.0;
      for(Eigen::Index start = 0; start < design.rows(); start += chunk_rows) {
        const Eigen::Index count_index = std::min(
          chunk_rows, design.rows() - start);
        const int count = checked_int(
          count_index, "grouped prediction chunk row count");
        if(direct_grouped_upload_) {
          if(timings) transfer_start = ComputeClock::now();
          copy_matrix_row_chunk_to_device(design, start, count_index,
            d_genotypes_,
            "copy grouped prediction chunk directly to CUDA device");
          if(timings) timings->upload_ms += elapsed_ms(transfer_start);
        } else {
          ComputeClock::time_point materialization_start;
          if(timings) materialization_start = ComputeClock::now();
          const Eigen::MatrixXd design_chunk =
            design.middleRows(start, count_index);
          if(timings)
            timings->host_materialization_ms +=
              elapsed_ms(materialization_start);
          if(timings) transfer_start = ComputeClock::now();
          if(design_chunk.size() > 0) {
            check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
              design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
              "copy materialized grouped prediction chunk to CUDA device");
          }
          if(timings) timings->upload_ms += elapsed_ms(transfer_start);
        }
        if(timings) {
          timings->design_upload_count++;
          timings->design_upload_bytes +=
            static_cast<uint64_t>(count_index) *
            static_cast<uint64_t>(design.cols()) * sizeof(double);
        }

        std::unique_ptr<CudaEventPair> prediction_events;
        if(timings) {
          prediction_events.reset(new CudaEventPair());
          prediction_events->record_start();
        }
        for(int group = 0; group < group_count; ++group) {
          const int group_size = group_sizes(group);
          double* group_predictions = d_predictions_ + group * count;
          if(group_size > 0)
            check_cublas(cublasDgemv(handle_, CUBLAS_OP_N,
              count, group_size, &alpha,
              d_genotypes_ + static_cast<Eigen::Index>(group_offsets(group)) *
                count,
              count, d_inverse_ + group_offsets(group), 1,
              &beta, group_predictions, 1),
              "cublasDgemv(grouped prediction chunk)");
          else
            check_cuda(cudaMemset(group_predictions, 0,
              count * sizeof(double)),
              "clear empty grouped prediction chunk");
        }
        if(timings) timings->ridge_ms +=
          prediction_events->record_stop_and_elapsed_ms();

        if(timings) transfer_start = ComputeClock::now();
        for(int group = 0; group < group_count; ++group)
          check_cuda(cudaMemcpy(
            predictions.col(group).segment(start, count_index).data(),
            d_predictions_ + group * count, count * sizeof(double),
            cudaMemcpyDeviceToHost),
            "copy grouped prediction chunk from CUDA device");
        if(timings) timings->download_ms += elapsed_ms(transfer_start);
      }
    }

    void ridge_predict(
      const Eigen::Ref<const Eigen::MatrixXd>& eigenvectors,
      const Eigen::Ref<const Eigen::MatrixXd>& eigenvalues,
      const Eigen::Ref<const Eigen::MatrixXd>& transformed_right_hand_sides,
      const Eigen::Ref<const Eigen::MatrixXd>& prediction_matrix,
      bool samples_in_columns,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::MatrixXd>& leave_one_out_outcomes,
      bool leave_one_out,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings) override {

      validate_ridge_dimensions(eigenvectors, eigenvalues,
        transformed_right_hand_sides, prediction_matrix, samples_in_columns,
        ridge_parameters, leave_one_out_outcomes, leave_one_out);

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = checked_int(eigenvectors.rows(), "ridge matrix size");
      const int phenotype_count = checked_int(
        transformed_right_hand_sides.cols(), "ridge phenotype count");
      const int saved_factorized_size = ridge_factorized_size_;
      const int saved_rhs_count = ridge_factorized_rhs_count_;
      if(saved_factorized_size > 0) {
        ensure_capacity(d_gram_, gram_capacity_,
          static_cast<Eigen::Index>(saved_factorized_size) *
            saved_factorized_size,
          "cudaMalloc(saved ridge eigenvectors)");
        ensure_capacity(d_eigenvalues_, eigenvalues_capacity_,
          saved_factorized_size, "cudaMalloc(saved ridge eigenvalues)");
        check_cuda(cudaMemcpy(d_gram_, d_ridge_vectors_,
          static_cast<size_t>(saved_factorized_size) *
            saved_factorized_size * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "save reusable ridge eigenvectors");
        check_cuda(cudaMemcpy(d_eigenvalues_, d_ridge_values_,
          static_cast<size_t>(saved_factorized_size) * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "save reusable ridge eigenvalues");
        if(saved_rhs_count > 0) {
          ensure_capacity(d_crossproduct_, crossproduct_capacity_,
            static_cast<Eigen::Index>(saved_factorized_size) *
              saved_rhs_count,
            "cudaMalloc(saved transformed ridge right-hand sides)");
          check_cuda(cudaMemcpy(d_crossproduct_, d_ridge_rhs_,
            static_cast<size_t>(saved_factorized_size) * saved_rhs_count *
              sizeof(double), cudaMemcpyDeviceToDevice),
            "save reusable transformed ridge right-hand sides");
        }
      }
      const auto restore_factorized_state = [&] () {
        if(saved_factorized_size > 0) {
          check_cuda(cudaMemcpy(d_ridge_vectors_, d_gram_,
            static_cast<size_t>(saved_factorized_size) *
              saved_factorized_size * sizeof(double),
            cudaMemcpyDeviceToDevice),
            "restore reusable ridge eigenvectors");
          check_cuda(cudaMemcpy(d_ridge_values_, d_eigenvalues_,
            static_cast<size_t>(saved_factorized_size) * sizeof(double),
            cudaMemcpyDeviceToDevice),
            "restore reusable ridge eigenvalues");
          if(saved_rhs_count > 0)
            check_cuda(cudaMemcpy(d_ridge_rhs_, d_crossproduct_,
              static_cast<size_t>(saved_factorized_size) * saved_rhs_count *
                sizeof(double), cudaMemcpyDeviceToDevice),
              "restore reusable transformed ridge right-hand sides");
        }
        ridge_factorized_size_ = saved_factorized_size;
        ridge_factorized_rhs_count_ = saved_rhs_count;
      };
      try {
        if(size > 0) {
          ensure_capacity(d_ridge_vectors_, ridge_vectors_capacity_,
            eigenvectors.size(),
            "cudaMalloc(ridge eigenvectors)");
          ensure_capacity(d_ridge_values_, ridge_values_capacity_,
            eigenvalues.size(), "cudaMalloc(ridge eigenvalues)");
          if(phenotype_count > 0)
            ensure_capacity(d_ridge_rhs_, ridge_rhs_capacity_,
              transformed_right_hand_sides.size(),
              "cudaMalloc(ridge transformed right-hand sides)");
        }

        const Eigen::MatrixXd packed_vectors =
          contiguous_copy_if_needed(eigenvectors);
        const Eigen::MatrixXd packed_values =
          contiguous_copy_if_needed(eigenvalues);
        const Eigen::MatrixXd packed_rhs = phenotype_count > 0 ?
          contiguous_copy_if_needed(transformed_right_hand_sides) :
          Eigen::MatrixXd();
        const double* vectors_data = packed_vectors.size() ?
          packed_vectors.data() : eigenvectors.data();
        const double* values_data = packed_values.size() ?
          packed_values.data() : eigenvalues.data();
        const double* rhs_data = packed_rhs.size() ? packed_rhs.data() :
          transformed_right_hand_sides.data();

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        if(size > 0) {
          check_cuda(cudaMemcpy(d_ridge_vectors_, vectors_data,
            eigenvectors.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy ridge eigenvectors to CUDA device");
          check_cuda(cudaMemcpy(d_ridge_values_, values_data,
            eigenvalues.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy ridge eigenvalues to CUDA device");
          if(phenotype_count > 0)
            check_cuda(cudaMemcpy(d_ridge_rhs_, rhs_data,
              transformed_right_hand_sides.size() * sizeof(double),
              cudaMemcpyHostToDevice),
              "copy ridge transformed right-hand sides to CUDA device");
        }
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        ridge_factorized_size_ = size;
        ridge_factorized_rhs_count_ = phenotype_count;
        ridge_predict_factorized(prediction_matrix, samples_in_columns,
          ridge_parameters, leave_one_out_outcomes, leave_one_out,
          predictions, coefficients, timings);
      } catch(...) {
        restore_factorized_state();
        throw;
      }
      restore_factorized_state();
    }

    void factorize_ridge_system(
      const Eigen::Ref<const Eigen::MatrixXd>& symmetric_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Step1ComputeTimings* timings) override {

      if(symmetric_matrix.rows() != symmetric_matrix.cols())
        throw std::invalid_argument(
          "Step 1 reusable ridge factorization requires a square matrix");
      if(symmetric_matrix.rows() != right_hand_sides.rows())
        throw std::invalid_argument(
          "Step 1 reusable ridge factorization received incompatible right-hand sides");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = checked_int(
        symmetric_matrix.rows(), "reusable ridge factorization size");
      const int right_hand_side_count = checked_int(
        right_hand_sides.cols(), "reusable ridge right-hand-side count");
      ridge_factorized_size_ = -1;
      ridge_factorized_rhs_count_ = 0;
      if(size == 0) {
        ridge_factorized_size_ = 0;
        ridge_factorized_rhs_count_ = right_hand_side_count;
        return;
      }

      ensure_capacity(d_ridge_vectors_, ridge_vectors_capacity_,
        symmetric_matrix.size(),
        "cudaMalloc(reusable ridge factorization matrix)");
      ensure_capacity(d_ridge_values_, ridge_values_capacity_, size,
        "cudaMalloc(reusable ridge eigenvalues)");
      if(right_hand_side_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_, right_hand_sides.size(),
          "cudaMalloc(reusable ridge right-hand sides)");
        ensure_capacity(d_ridge_rhs_, ridge_rhs_capacity_,
          right_hand_sides.size(),
          "cudaMalloc(reusable transformed ridge right-hand sides)");
      }

      const Eigen::MatrixXd packed_matrix =
        contiguous_copy_if_needed(symmetric_matrix);
      const Eigen::MatrixXd packed_rhs = right_hand_side_count > 0 ?
        contiguous_copy_if_needed(right_hand_sides) : Eigen::MatrixXd();
      const double* matrix_data = packed_matrix.size() ?
        packed_matrix.data() : symmetric_matrix.data();
      const double* rhs_data = packed_rhs.size() ?
        packed_rhs.data() : right_hand_sides.data();

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_ridge_vectors_, matrix_data,
        symmetric_matrix.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy reusable ridge factorization matrix to CUDA device");
      if(right_hand_side_count > 0)
        check_cuda(cudaMemcpy(d_phenotypes_, rhs_data,
          right_hand_sides.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy reusable ridge right-hand sides to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      int workspace_size = 0;
      check_cusolver(cusolverDnDsyevd_bufferSize(solver_handle_,
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, size,
        d_ridge_vectors_, size, d_ridge_values_, &workspace_size),
        "cusolverDnDsyevd_bufferSize(reusable ridge)");
      ensure_capacity(d_solver_workspace_, solver_workspace_capacity_, workspace_size,
        "cudaMalloc(cuSOLVER reusable ridge workspace)");

      std::unique_ptr<CudaEventPair> eigensolve_events;
      if(timings) {
        eigensolve_events.reset(new CudaEventPair());
        eigensolve_events->record_start();
      }
      check_cusolver(cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, size, d_ridge_vectors_, size,
        d_ridge_values_,
        d_solver_workspace_, workspace_size, d_solver_info_),
        "cusolverDnDsyevd(reusable ridge)");
      if(timings)
        timings->eigensolve_ms +=
          eigensolve_events->record_stop_and_elapsed_ms();

      int solver_info = 0;
      check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
        cudaMemcpyDeviceToHost),
        "copy reusable ridge eigensolver status to host");
      if(solver_info != 0) {
        std::ostringstream message;
        message << "cuSOLVER reusable ridge eigendecomposition failed with info="
                << solver_info;
        throw std::runtime_error(message.str());
      }

      if(right_hand_side_count > 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        std::unique_ptr<CudaEventPair> transform_events;
        if(timings) {
          transform_events.reset(new CudaEventPair());
          transform_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          size, right_hand_side_count, size, &alpha,
          d_ridge_vectors_, size, d_phenotypes_, size, &beta,
          d_ridge_rhs_, size),
          "cublasDgemm(reusable ridge transform)");
        if(timings)
          timings->transform_ms += transform_events->record_stop_and_elapsed_ms();
      }

      ridge_factorized_size_ = size;
      ridge_factorized_rhs_count_ = right_hand_side_count;
    }

    void compute_products_and_factorize_ridge(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Step1GramMode mode,
      Step1ComputeTimings* timings) override {

      if(genotypes.cols() != phenotypes.rows())
        throw std::invalid_argument(
          "Step 1 fused ridge backend received incompatible genotype and phenotype matrices");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int blocks = checked_int(genotypes.rows(), "fused ridge genotype rows");
      const int samples = checked_int(genotypes.cols(), "fused ridge genotype columns");
      const int phenotype_count = checked_int(
        phenotypes.cols(), "fused ridge phenotype columns");
      ridge_factorized_size_ = -1;
      ridge_factorized_rhs_count_ = 0;
      if(blocks == 0) {
        ridge_factorized_size_ = 0;
        ridge_factorized_rhs_count_ = phenotype_count;
        return;
      }

      const Eigen::Index chunk_samples = bounded_cuda_chunk_rows(
        genotypes.cols(), genotypes.rows());
      const bool genotypes_are_resident =
        resident_genotype_columns(genotypes, 0, genotypes.cols()) != nullptr;
      if(!genotypes_are_resident)
        ensure_capacity(d_genotypes_, genotypes_capacity_,
          std::max<Eigen::Index>(1, chunk_samples * genotypes.rows()),
          "cudaMalloc(fused ridge genotype chunk)");
      ensure_capacity(d_gram_, gram_capacity_,
        static_cast<Eigen::Index>(blocks) * blocks,
        "cudaMalloc(fused ridge Gram matrix)");
      ensure_capacity(d_ridge_vectors_, ridge_vectors_capacity_,
        static_cast<Eigen::Index>(blocks) * blocks,
        "cudaMalloc(fused ridge eigenvectors)");
      ensure_capacity(d_ridge_values_, ridge_values_capacity_, blocks,
        "cudaMalloc(fused ridge eigenvalues)");
      if(phenotype_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_,
          std::max<Eigen::Index>(1, chunk_samples * phenotypes.cols()),
          "cudaMalloc(fused ridge phenotype chunk)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_,
          static_cast<Eigen::Index>(blocks) * phenotype_count,
          "cudaMalloc(fused ridge crossproduct)");
        ensure_capacity(d_ridge_rhs_, ridge_rhs_capacity_,
          static_cast<Eigen::Index>(blocks) * phenotype_count,
          "cudaMalloc(fused transformed ridge right-hand sides)");
      }

      const double alpha = 1.0;
      const bool genotypes_have_contiguous_columns =
        genotypes.innerStride() == 1 &&
        genotypes.outerStride() == genotypes.rows();
      if(samples == 0) {
        check_cuda(cudaMemset(d_gram_, 0,
          static_cast<size_t>(blocks) * blocks * sizeof(double)),
          "clear empty fused ridge Gram matrix");
        if(phenotype_count > 0)
          check_cuda(cudaMemset(d_crossproduct_, 0,
            static_cast<size_t>(blocks) * phenotype_count * sizeof(double)),
            "clear empty fused ridge crossproduct");
      } else {
        for(Eigen::Index start = 0; start < genotypes.cols();
            start += chunk_samples) {
          const Eigen::Index count_index = std::min(
            chunk_samples, genotypes.cols() - start);
          const int count = checked_int(
            count_index, "fused ridge chunk sample count");
          const double* device_genotype_chunk =
            resident_genotype_columns(genotypes, start, count_index);
          Eigen::MatrixXd packed_genotype_chunk;
          const double* genotype_chunk_data = nullptr;
          if(!device_genotype_chunk) {
            if(genotypes_have_contiguous_columns)
              genotype_chunk_data = genotypes.data() +
                start * genotypes.outerStride();
            else {
              packed_genotype_chunk =
                genotypes.middleCols(start, count_index);
              genotype_chunk_data = packed_genotype_chunk.data();
            }
          }
          const Eigen::MatrixXd phenotype_chunk = phenotype_count > 0 ?
            Eigen::MatrixXd(phenotypes.middleRows(start, count_index)) :
            Eigen::MatrixXd();

          ComputeClock::time_point transfer_start;
          if(timings) transfer_start = ComputeClock::now();
          if(!device_genotype_chunk) {
            check_cuda(cudaMemcpy(d_genotypes_, genotype_chunk_data,
              count_index * genotypes.rows() * sizeof(double),
              cudaMemcpyHostToDevice),
              "copy fused ridge genotype chunk to CUDA device");
            device_genotype_chunk = d_genotypes_;
          } else if(timings) {
            timings->resident_reuse_count++;
          }
          if(phenotype_count > 0)
            check_cuda(cudaMemcpy(d_phenotypes_, phenotype_chunk.data(),
              phenotype_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
              "copy fused ridge phenotype chunk to CUDA device");
          if(timings) timings->upload_ms += elapsed_ms(transfer_start);

          const double beta = start == 0 ? 0.0 : 1.0;
          if(phenotype_count > 0) {
            std::unique_ptr<CudaEventPair> crossproduct_events;
            if(timings) {
              crossproduct_events.reset(new CudaEventPair());
              crossproduct_events->record_start();
            }
            check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
              blocks, phenotype_count, count, &alpha,
              device_genotype_chunk, blocks,
              d_phenotypes_, count, &beta,
              d_crossproduct_, blocks),
              "cublasDgemm(fused ridge crossproduct chunk)");
            if(timings)
              timings->crossproduct_ms +=
                crossproduct_events->record_stop_and_elapsed_ms();
          }

          std::unique_ptr<CudaEventPair> gram_events;
          if(timings) {
            gram_events.reset(new CudaEventPair());
            gram_events->record_start();
          }
          if(mode == Step1GramMode::selfadjoint_rank_update)
            check_cublas(cublasDsyrk(handle_, CUBLAS_FILL_MODE_LOWER,
              CUBLAS_OP_N, blocks, count, &alpha,
              device_genotype_chunk, blocks,
              &beta, d_gram_, blocks),
              "cublasDsyrk(fused ridge Gram chunk)");
          else
            check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
              blocks, blocks, count, &alpha,
              device_genotype_chunk, blocks,
              device_genotype_chunk, blocks, &beta,
              d_gram_, blocks), "cublasDgemm(fused ridge Gram chunk)");
          if(timings)
            timings->gram_ms += gram_events->record_stop_and_elapsed_ms();
        }
      }

      if(mode == Step1GramMode::selfadjoint_rank_update && samples > 0) {
        const dim3 threads(16, 16);
        const dim3 grid((blocks + threads.x - 1) / threads.x,
                        (blocks + threads.y - 1) / threads.y);
        mirror_lower_triangle<<<grid, threads>>>(d_gram_, blocks);
        check_cuda(cudaGetLastError(), "mirror fused ridge Gram triangle kernel");
      }

      check_cuda(cudaMemcpy(d_ridge_vectors_, d_gram_,
        static_cast<size_t>(blocks) * blocks * sizeof(double),
        cudaMemcpyDeviceToDevice),
        "copy fused ridge Gram matrix into persistent state");

      int workspace_size = 0;
      check_cusolver(cusolverDnDsyevd_bufferSize(solver_handle_,
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, blocks,
        d_ridge_vectors_, blocks, d_ridge_values_, &workspace_size),
        "cusolverDnDsyevd_bufferSize(fused ridge)");
      ensure_capacity(d_solver_workspace_, solver_workspace_capacity_, workspace_size,
        "cudaMalloc(cuSOLVER fused ridge workspace)");

      std::unique_ptr<CudaEventPair> eigensolve_events;
      if(timings) {
        eigensolve_events.reset(new CudaEventPair());
        eigensolve_events->record_start();
      }
      check_cusolver(cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, blocks, d_ridge_vectors_, blocks,
        d_ridge_values_,
        d_solver_workspace_, workspace_size, d_solver_info_),
        "cusolverDnDsyevd(fused ridge)");
      if(timings)
        timings->eigensolve_ms +=
          eigensolve_events->record_stop_and_elapsed_ms();

      int solver_info = 0;
      check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
        cudaMemcpyDeviceToHost), "copy fused ridge eigensolver status to host");
      if(solver_info != 0) {
        std::ostringstream message;
        message << "cuSOLVER fused ridge eigendecomposition failed with info="
                << solver_info;
        throw std::runtime_error(message.str());
      }

      if(phenotype_count > 0) {
        const double beta = 0.0;
        std::unique_ptr<CudaEventPair> transform_events;
        if(timings) {
          transform_events.reset(new CudaEventPair());
          transform_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          blocks, phenotype_count, blocks, &alpha,
          d_ridge_vectors_, blocks, d_crossproduct_, blocks, &beta,
          d_ridge_rhs_, blocks),
          "cublasDgemm(fused ridge transform)");
        if(timings)
          timings->transform_ms += transform_events->record_stop_and_elapsed_ms();
      }

      ridge_factorized_size_ = blocks;
      ridge_factorized_rhs_count_ = phenotype_count;
    }

    void ridge_predict_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& prediction_matrix,
      bool samples_in_columns,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::MatrixXd>& leave_one_out_outcomes,
      bool leave_one_out,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings) override {

      if(ridge_factorized_size_ < 0)
        throw std::runtime_error(
          "Step 1 factorized ridge prediction requested before factorization");
      if((samples_in_columns ? prediction_matrix.rows() : prediction_matrix.cols()) !=
         ridge_factorized_size_)
        throw std::invalid_argument(
          "Step 1 factorized ridge prediction received incompatible matrix dimensions");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument(
          "Step 1 factorized ridge parameters must be non-negative");

      const Eigen::Index sample_count_index = samples_in_columns ?
        prediction_matrix.cols() : prediction_matrix.rows();
      if(leave_one_out &&
         (leave_one_out_outcomes.rows() != sample_count_index ||
          leave_one_out_outcomes.cols() != ridge_factorized_rhs_count_))
        throw std::invalid_argument(
          "Step 1 factorized ridge LOOCV outcomes have incompatible dimensions");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = ridge_factorized_size_;
      const int sample_count = checked_int(
        sample_count_index, "factorized ridge sample count");
      const int phenotype_count = ridge_factorized_rhs_count_;
      const int parameter_count = checked_int(
        ridge_parameters.size(), "factorized ridge parameter count");
      const long long combination_count_long =
        static_cast<long long>(phenotype_count) * parameter_count;
      if(combination_count_long > INT_MAX)
        throw std::runtime_error(
          "CUDA factorized ridge phenotype/parameter count exceeds integer limits");
      const int combination_count = static_cast<int>(combination_count_long);

      predictions.resize(sample_count, combination_count);
      coefficients.resize(size, combination_count);
      if(size == 0 || sample_count == 0 || combination_count == 0) {
        predictions.setZero();
        coefficients.setZero();
        return;
      }

      const Eigen::Index streaming_columns = std::max<Eigen::Index>(
        size, combination_count);
      const Eigen::Index chunk_samples = bounded_cuda_chunk_rows(
        sample_count_index, streaming_columns);
      const bool prediction_is_resident = samples_in_columns &&
        resident_genotype_columns(
          prediction_matrix, 0, prediction_matrix.cols()) != nullptr;
      if(!prediction_is_resident)
        ensure_capacity(d_genotypes_, genotypes_capacity_,
          chunk_samples * size,
          "cudaMalloc(factorized ridge prediction chunk)");
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_,
        ridge_parameters.size(), "cudaMalloc(factorized ridge parameters)");
      ensure_capacity(d_inverse_, inverse_capacity_,
        static_cast<Eigen::Index>(size) * parameter_count,
        "cudaMalloc(factorized ridge inverse)");
      ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_,
        static_cast<Eigen::Index>(size) * combination_count,
        "cudaMalloc(factorized ridge scaled right-hand sides)");
      ensure_capacity(d_phenotypes_, phenotypes_capacity_, coefficients.size(),
        "cudaMalloc(factorized ridge coefficients)");
      ensure_capacity(d_predictions_, predictions_capacity_,
        chunk_samples * combination_count,
        "cudaMalloc(factorized ridge prediction results chunk)");
      if(leave_one_out) {
        ensure_capacity(d_outcomes_, outcomes_capacity_,
          chunk_samples * phenotype_count,
          "cudaMalloc(factorized ridge LOOCV outcome chunk)");
        ensure_capacity(d_projected_, projected_capacity_,
          chunk_samples * size,
          "cudaMalloc(factorized ridge projected matrix chunk)");
        ensure_capacity(d_squared_, squared_capacity_,
          chunk_samples * size,
          "cudaMalloc(factorized ridge squared projected matrix chunk)");
        ensure_capacity(d_leverage_, leverage_capacity_,
          chunk_samples * parameter_count,
          "cudaMalloc(factorized ridge LOOCV leverage chunk)");
      }

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_ridge_parameters_, ridge_parameters.data(),
        ridge_parameters.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy factorized ridge parameters to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      std::unique_ptr<CudaEventPair> coefficient_events;
      if(timings) {
        coefficient_events.reset(new CudaEventPair());
        coefficient_events->record_start();
      }
      const int threads = 256;
      const int inverse_count = checked_element_count(
        size, parameter_count, "factorized ridge inverse");
      build_ridge_inverse<<<(inverse_count + threads - 1) / threads, threads>>>(
        d_ridge_values_, d_ridge_parameters_, d_inverse_, size,
        inverse_count);
      check_cuda(cudaGetLastError(), "build factorized ridge inverse kernel");
      const int scaled_count = checked_element_count(
        size, combination_count, "factorized ridge scaled right-hand sides");
      build_scaled_right_hand_sides<<<
        (scaled_count + threads - 1) / threads, threads>>>(
        d_inverse_, d_ridge_rhs_, d_scaled_rhs_, size,
        phenotype_count, scaled_count);
      check_cuda(cudaGetLastError(),
        "build factorized ridge scaled right-hand sides kernel");

      const double alpha = 1.0;
      const double beta = 0.0;
      check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        size, combination_count, size, &alpha,
        d_ridge_vectors_, size, d_scaled_rhs_, size, &beta,
        d_phenotypes_, size),
        "cublasDgemm(factorized ridge coefficients)");
      if(timings) timings->ridge_ms +=
        coefficient_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(coefficients.data(), d_phenotypes_,
        coefficients.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy factorized ridge coefficients from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);

      const bool prediction_has_contiguous_columns =
        prediction_matrix.innerStride() == 1 &&
        prediction_matrix.outerStride() == prediction_matrix.rows();
      for(Eigen::Index start = 0; start < sample_count_index;
          start += chunk_samples) {
        const Eigen::Index count_index = std::min(
          chunk_samples, sample_count_index - start);
        const int count = checked_int(
          count_index, "factorized ridge prediction chunk sample count");
        const double* device_prediction_chunk = samples_in_columns ?
          resident_genotype_columns(
            prediction_matrix, start, count_index) : nullptr;
        Eigen::MatrixXd packed_prediction_chunk;
        const double* prediction_chunk_data = nullptr;
        if(!device_prediction_chunk && samples_in_columns) {
          if(prediction_has_contiguous_columns)
            prediction_chunk_data = prediction_matrix.data() +
              start * prediction_matrix.outerStride();
          else {
            packed_prediction_chunk =
              prediction_matrix.middleCols(start, count_index);
            prediction_chunk_data = packed_prediction_chunk.data();
          }
        }

        if(timings) transfer_start = ComputeClock::now();
        if(!device_prediction_chunk) {
          if(samples_in_columns)
            check_cuda(cudaMemcpy(d_genotypes_, prediction_chunk_data,
              count_index * size * sizeof(double), cudaMemcpyHostToDevice),
              "copy factorized ridge prediction chunk to CUDA device");
          else
            copy_matrix_row_chunk_to_device(
              prediction_matrix, start, count_index, d_genotypes_,
              "copy factorized ridge prediction chunk to CUDA device");
          device_prediction_chunk = d_genotypes_;
        } else if(timings) {
          timings->resident_reuse_count++;
        }
        if(leave_one_out)
          copy_matrix_row_chunk_to_device(
            leave_one_out_outcomes, start, count_index, d_outcomes_,
            "copy factorized ridge LOOCV outcome chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

        std::unique_ptr<CudaEventPair> prediction_events;
        if(timings) {
          prediction_events.reset(new CudaEventPair());
          prediction_events->record_start();
        }
        if(samples_in_columns)
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            count, combination_count, size, &alpha,
            device_prediction_chunk, size,
            d_phenotypes_, size, &beta,
            d_predictions_, count),
            "cublasDgemm(factorized ridge prediction chunk)");
        else
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            count, combination_count, size, &alpha,
            device_prediction_chunk, count,
            d_phenotypes_, size, &beta,
            d_predictions_, count),
            "cublasDgemm(design factorized ridge prediction chunk)");

        if(leave_one_out) {
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_T,
            samples_in_columns ? CUBLAS_OP_N : CUBLAS_OP_T,
            size, count, size, &alpha,
            d_ridge_vectors_, size, device_prediction_chunk,
            samples_in_columns ? size : count,
            &beta, d_projected_, size),
            "cublasDgemm(factorized ridge projected matrix chunk)");
          const int projected_count = checked_element_count(
            size, count, "factorized ridge projected matrix chunk");
          square_elements<<<
            (projected_count + threads - 1) / threads, threads>>>(
            d_projected_, d_squared_, projected_count);
          check_cuda(cudaGetLastError(),
            "square factorized ridge projected matrix chunk kernel");
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            count, parameter_count, size, &alpha,
            d_squared_, size, d_inverse_, size, &beta,
            d_leverage_, count),
            "cublasDgemm(factorized ridge LOOCV leverage chunk)");
          const int prediction_count = checked_element_count(
            count, combination_count,
            "factorized ridge LOOCV prediction chunk");
          apply_leave_one_out_correction<<<
            (prediction_count + threads - 1) / threads, threads>>>(
            d_predictions_, d_leverage_, d_outcomes_, count,
            phenotype_count, prediction_count);
          check_cuda(cudaGetLastError(),
            "apply factorized ridge LOOCV correction chunk kernel");
        }
        if(timings) timings->ridge_ms +=
          prediction_events->record_stop_and_elapsed_ms();

        if(timings) transfer_start = ComputeClock::now();
        for(int combination = 0; combination < combination_count;
            ++combination)
          check_cuda(cudaMemcpy(
            predictions.col(combination).segment(start, count_index).data(),
            d_predictions_ + static_cast<Eigen::Index>(combination) * count,
            count * sizeof(double), cudaMemcpyDeviceToHost),
            "copy factorized ridge prediction chunk from CUDA device");
        if(timings) timings->download_ms += elapsed_ms(transfer_start);
      }
    }

    void eigendecompose_and_transform(
      const Eigen::Ref<const Eigen::MatrixXd>& symmetric_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::MatrixXd& eigenvectors,
      Eigen::MatrixXd& eigenvalues,
      Eigen::MatrixXd& transformed_right_hand_sides,
      Step1ComputeTimings* timings) override {

      if(symmetric_matrix.rows() != symmetric_matrix.cols())
        throw std::invalid_argument("Step 1 eigendecomposition requires a square matrix");
      if(symmetric_matrix.rows() != right_hand_sides.rows())
        throw std::invalid_argument(
          "Step 1 eigendecomposition received incompatible right-hand sides");

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
      const int size = checked_int(symmetric_matrix.rows(), "eigendecomposition size");
      const int right_hand_side_count = checked_int(
        right_hand_sides.cols(), "right-hand-side columns");
      eigenvectors.resize(size, size);
      eigenvalues.resize(size, 1);
      transformed_right_hand_sides.resize(size, right_hand_side_count);
      if(size == 0) return;

      ensure_capacity(d_gram_, gram_capacity_, symmetric_matrix.size(),
        "cudaMalloc(eigendecomposition matrix)");
      ensure_capacity(d_eigenvalues_, eigenvalues_capacity_, size,
        "cudaMalloc(eigenvalues)");
      if(right_hand_side_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_, right_hand_sides.size(),
          "cudaMalloc(eigendecomposition right-hand sides)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, right_hand_sides.size(),
          "cudaMalloc(transformed right-hand sides)");
      }

      const Eigen::MatrixXd packed_matrix = contiguous_copy_if_needed(symmetric_matrix);
      const Eigen::MatrixXd packed_rhs = right_hand_side_count > 0 ?
        contiguous_copy_if_needed(right_hand_sides) : Eigen::MatrixXd();
      const double* matrix_data = packed_matrix.size() ? packed_matrix.data() : symmetric_matrix.data();
      const double* rhs_data = packed_rhs.size() ? packed_rhs.data() : right_hand_sides.data();

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_gram_, matrix_data, symmetric_matrix.size() * sizeof(double),
        cudaMemcpyHostToDevice), "copy eigendecomposition matrix to CUDA device");
      if(right_hand_side_count > 0)
        check_cuda(cudaMemcpy(d_phenotypes_, rhs_data, right_hand_sides.size() * sizeof(double),
          cudaMemcpyHostToDevice), "copy eigendecomposition right-hand sides to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      int workspace_size = 0;
      check_cusolver(cusolverDnDsyevd_bufferSize(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, size, d_gram_, size, d_eigenvalues_, &workspace_size),
        "cusolverDnDsyevd_bufferSize");
      ensure_capacity(d_solver_workspace_, solver_workspace_capacity_, workspace_size,
        "cudaMalloc(cuSOLVER workspace)");

      std::unique_ptr<CudaEventPair> eigensolve_events;
      if(timings) {
        eigensolve_events.reset(new CudaEventPair());
        eigensolve_events->record_start();
      }
      check_cusolver(cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, size, d_gram_, size, d_eigenvalues_,
        d_solver_workspace_, workspace_size, d_solver_info_), "cusolverDnDsyevd");
      if(timings)
        timings->eigensolve_ms += eigensolve_events->record_stop_and_elapsed_ms();

      int solver_info = 0;
      check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int), cudaMemcpyDeviceToHost),
        "copy cuSOLVER status to host");
      if(solver_info != 0) {
        std::ostringstream message;
        message << "cuSOLVER symmetric eigendecomposition failed with info=" << solver_info;
        throw std::runtime_error(message.str());
      }

      if(right_hand_side_count > 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        std::unique_ptr<CudaEventPair> transform_events;
        if(timings) {
          transform_events.reset(new CudaEventPair());
          transform_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          size, right_hand_side_count, size, &alpha,
          d_gram_, size, d_phenotypes_, size, &beta,
          d_crossproduct_, size), "cublasDgemm(eigenvectors^T * right-hand sides)");
        if(timings)
          timings->transform_ms += transform_events->record_stop_and_elapsed_ms();
      }

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(eigenvectors.data(), d_gram_, eigenvectors.size() * sizeof(double),
        cudaMemcpyDeviceToHost), "copy eigenvectors from CUDA device");
      check_cuda(cudaMemcpy(eigenvalues.data(), d_eigenvalues_, eigenvalues.size() * sizeof(double),
        cudaMemcpyDeviceToHost), "copy eigenvalues from CUDA device");
      if(right_hand_side_count > 0)
        check_cuda(cudaMemcpy(transformed_right_hand_sides.data(), d_crossproduct_,
          transformed_right_hand_sides.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "copy transformed right-hand sides from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

  private:
    void compute_cached_weighted_design_products_device(
      const Eigen::Ref<const Eigen::VectorXd>& weights,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Step1ComputeTimings* timings) {

      const int rows = checked_int(
        resident_design_rows_, "cached weighted design row count");
      const int features = checked_int(
        resident_design_columns_, "cached weighted design feature count");
      const int outcome_count = checked_int(
        outcomes.cols(), "cached weighted design outcome count");
      const int design_count = checked_element_count(
        rows, features, "cached weighted design matrix");
      const Eigen::Index gram_elements =
        static_cast<Eigen::Index>(features) * features;
      const Eigen::Index crossproduct_elements =
        static_cast<Eigen::Index>(features) * outcome_count;

      ensure_capacity(d_projected_, projected_capacity_, design_count,
        "cudaMalloc(cached weighted design matrix)");
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_, rows,
        "cudaMalloc(cached design weights)");
      ensure_capacity(d_gram_, gram_capacity_, gram_elements,
        "cudaMalloc(cached weighted Gram matrix)");
      if(outcome_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_, outcomes.size(),
          "cudaMalloc(cached weighted outcomes)");
        ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_, outcomes.size(),
          "cudaMalloc(cached weighted outcome matrix)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_,
          crossproduct_elements, "cudaMalloc(cached weighted crossproduct)");
      }

      const Eigen::MatrixXd packed_outcomes = outcome_count > 0 ?
        contiguous_copy_if_needed(outcomes) : Eigen::MatrixXd();
      const double* outcome_data = packed_outcomes.size() ?
        packed_outcomes.data() : outcomes.data();
      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_ridge_parameters_, weights.data(),
        static_cast<size_t>(rows) * sizeof(double), cudaMemcpyHostToDevice),
        "copy cached design weights to CUDA device");
      if(outcome_count > 0)
        check_cuda(cudaMemcpy(d_phenotypes_, outcome_data,
          static_cast<size_t>(outcomes.size()) * sizeof(double),
          cudaMemcpyHostToDevice),
          "copy cached weighted outcomes to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      const int threads = 256;
      scale_matrix_rows<<<
        (design_count + threads - 1) / threads, threads>>>(
        resident_design_data(), d_ridge_parameters_, d_projected_, rows,
        design_count);
      check_cuda(cudaGetLastError(),
        "scale cached weighted design rows kernel");
      if(outcome_count > 0) {
        const int outcome_elements = checked_element_count(
          rows, outcome_count, "cached weighted outcomes");
        scale_matrix_rows<<<
          (outcome_elements + threads - 1) / threads, threads>>>(
          d_phenotypes_, d_ridge_parameters_, d_scaled_rhs_, rows,
          outcome_elements);
        check_cuda(cudaGetLastError(),
          "scale cached weighted outcomes kernel");
      }

      const double alpha = 1.0;
      const double beta = 0.0;
      if(outcome_count > 0) {
        std::unique_ptr<CudaEventPair> crossproduct_events;
        if(timings) {
          crossproduct_events.reset(new CudaEventPair());
          crossproduct_events->record_start();
        }
        check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          features, outcome_count, rows, &alpha,
          resident_design_data(), rows, d_scaled_rhs_, rows, &beta,
          d_crossproduct_, features),
          "cublasDgemm(cached weighted crossproduct)");
        if(timings)
          timings->crossproduct_ms +=
            crossproduct_events->record_stop_and_elapsed_ms();
      }

      std::unique_ptr<CudaEventPair> gram_events;
      if(timings) {
        gram_events.reset(new CudaEventPair());
        gram_events->record_start();
      }
      check_cublas(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        features, features, rows, &alpha,
        resident_design_data(), rows, d_projected_, rows, &beta,
        d_gram_, features),
        "cublasDgemm(cached weighted Gram matrix)");
      if(timings) {
        timings->gram_ms += gram_events->record_stop_and_elapsed_ms();
        timings->resident_design_reuse_count++;
      }
    }

    void diagonal_penalty_solve_device(
      int size,
      int right_hand_side_count,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings) {

      const int parameter_count = checked_int(
        ridge_parameters.size(), "diagonal-penalty parameter count");
      const Eigen::Index gram_elements =
        static_cast<Eigen::Index>(size) * size;
      const Eigen::Index right_hand_side_elements =
        static_cast<Eigen::Index>(size) * right_hand_side_count;

      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_,
        penalty_multipliers.size(), "cudaMalloc(diagonal-penalty multipliers)");
      ensure_capacity(d_projected_, projected_capacity_, gram_elements,
        "cudaMalloc(diagonal-penalty factorization matrix)");
      ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_,
        right_hand_side_elements,
        "cudaMalloc(diagonal-penalty solve workspace)");
      ensure_capacity(d_predictions_, predictions_capacity_, solutions.size(),
        "cudaMalloc(diagonal-penalty solutions)");

      ComputeClock::time_point transfer_start;
      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(d_ridge_parameters_, penalty_multipliers.data(),
        penalty_multipliers.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy diagonal-penalty multipliers to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      int workspace_size = 0;
      check_cusolver(cusolverDnDpotrf_bufferSize(solver_handle_,
        CUBLAS_FILL_MODE_LOWER, size, d_projected_, size, &workspace_size),
        "cusolverDnDpotrf_bufferSize");
      ensure_capacity(d_solver_workspace_, solver_workspace_capacity_,
        workspace_size, "cudaMalloc(cuSOLVER Cholesky workspace)");

      std::unique_ptr<CudaEventPair> solve_events;
      if(timings) {
        solve_events.reset(new CudaEventPair());
        solve_events->record_start();
      }
      const int threads = 256;
      for(int parameter = 0; parameter < parameter_count; ++parameter) {
        check_cuda(cudaMemcpy(d_projected_, d_gram_,
          static_cast<size_t>(gram_elements) * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "copy diagonal-penalty factorization matrix");
        check_cuda(cudaMemcpy(d_scaled_rhs_, d_crossproduct_,
          static_cast<size_t>(right_hand_side_elements) * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "copy diagonal-penalty solve right-hand sides");
        add_diagonal_penalty<<<(size + threads - 1) / threads, threads>>>(
          d_projected_, d_ridge_parameters_, ridge_parameters(parameter), size);
        check_cuda(cudaGetLastError(), "add diagonal penalty kernel");

        check_cusolver(cusolverDnDpotrf(solver_handle_, CUBLAS_FILL_MODE_LOWER,
          size, d_projected_, size, d_solver_workspace_, workspace_size,
          d_solver_info_), "cusolverDnDpotrf");
        int solver_info = 0;
        check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
          cudaMemcpyDeviceToHost),
          "copy Cholesky factorization status to host");
        if(solver_info != 0) {
          std::ostringstream message;
          message << "cuSOLVER Cholesky factorization failed with info="
                  << solver_info;
          throw std::runtime_error(message.str());
        }

        check_cusolver(cusolverDnDpotrs(solver_handle_, CUBLAS_FILL_MODE_LOWER,
          size, right_hand_side_count, d_projected_, size, d_scaled_rhs_, size,
          d_solver_info_), "cusolverDnDpotrs");
        check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
          cudaMemcpyDeviceToHost), "copy Cholesky solve status to host");
        if(solver_info != 0) {
          std::ostringstream message;
          message << "cuSOLVER Cholesky solve failed with info=" << solver_info;
          throw std::runtime_error(message.str());
        }

        check_cuda(cudaMemcpy(
          d_predictions_ + static_cast<Eigen::Index>(parameter) *
            right_hand_side_elements,
          d_scaled_rhs_,
          static_cast<size_t>(right_hand_side_elements) * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "store diagonal-penalty solutions on CUDA device");
      }
      if(timings)
        timings->ridge_ms += solve_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(solutions.data(), d_predictions_,
        solutions.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy diagonal-penalty solutions from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
    }

    static void release_level0_cholesky_lane(
      CudaLevel0CholeskyLane& lane) {
      if(lane.stream) cudaStreamSynchronize(lane.stream);
      if(lane.info) cudaFree(lane.info);
      if(lane.workspace) cudaFree(lane.workspace);
      if(lane.predictions) cudaFree(lane.predictions);
      if(lane.coefficients) cudaFree(lane.coefficients);
      if(lane.solve) cudaFree(lane.solve);
      if(lane.right_hand_sides) cudaFree(lane.right_hand_sides);
      if(lane.factor) cudaFree(lane.factor);
      if(lane.gram) cudaFree(lane.gram);
      if(lane.solver) cusolverDnDestroy(lane.solver);
      if(lane.blas) cublasDestroy(lane.blas);
      if(lane.stream) cudaStreamDestroy(lane.stream);
      lane = CudaLevel0CholeskyLane();
    }

    void ensure_level0_cholesky_lane_count(size_t required) {
      if(level0_cholesky_lanes_.size() >= required) return;
      level0_cholesky_lanes_.reserve(required);
      while(level0_cholesky_lanes_.size() < required) {
        CudaLevel0CholeskyLane lane;
        try {
          check_cuda(cudaStreamCreateWithFlags(
            &lane.stream, cudaStreamNonBlocking),
            "create batched Cholesky stream");
          check_cusolver(cusolverDnCreate(&lane.solver),
            "create batched Cholesky cuSOLVER handle");
          check_cusolver(cusolverDnSetStream(lane.solver, lane.stream),
            "set batched Cholesky cuSOLVER stream");
          check_cublas(cublasCreate(&lane.blas),
            "create batched Cholesky cuBLAS handle");
          check_cublas(cublasSetStream(lane.blas, lane.stream),
            "set batched Cholesky cuBLAS stream");
          level0_cholesky_lanes_.push_back(lane);
        } catch(...) {
          release_level0_cholesky_lane(lane);
          throw;
        }
      }
    }

    void synchronize_level0_cholesky_lanes(size_t count) {
      for(size_t lane = 0; lane < count; ++lane)
        check_cuda(cudaStreamSynchronize(
          level0_cholesky_lanes_[lane].stream),
          "synchronize batched Cholesky stream");
    }

    bool ensure_pinned_staging(size_t required_bytes) {
      if(pinned_staging_chunk_bytes_ == 0 || required_bytes < 65536 ||
         !pinned_staging_available_)
        return false;
      const size_t target_capacity = std::min(
        pinned_staging_chunk_bytes_, required_bytes);
      if(pinned_staging_capacity_ >= target_capacity) return true;

      for(int index = 0; index < 2; ++index) {
        if(upload_streams_[index])
          check_cuda(cudaStreamSynchronize(upload_streams_[index]),
            "synchronize pinned upload stream while growing staging");
        if(pinned_staging_[index]) {
          check_cuda(cudaFreeHost(pinned_staging_[index]),
            "cudaFreeHost while growing pinned upload staging");
          pinned_staging_[index] = nullptr;
        }
      }
      pinned_staging_capacity_ = 0;

      for(int index = 0; index < 2; ++index) {
        if(!upload_streams_[index])
          check_cuda(cudaStreamCreateWithFlags(&upload_streams_[index],
            cudaStreamNonBlocking), "create pinned upload stream");
        const cudaError_t allocation_status = cudaHostAlloc(
          &pinned_staging_[index], target_capacity, cudaHostAllocPortable);
        if(allocation_status == cudaErrorMemoryAllocation) {
          cudaGetLastError();
          for(int cleanup = 0; cleanup <= index; ++cleanup) {
            if(pinned_staging_[cleanup]) {
              cudaFreeHost(pinned_staging_[cleanup]);
              pinned_staging_[cleanup] = nullptr;
            }
          }
          pinned_staging_available_ = false;
          return false;
        }
        check_cuda(allocation_status,
          "allocate pinned upload staging buffer");
      }
      pinned_staging_capacity_ = target_capacity;
      return true;
    }

    void copy_host_to_device_staged(void* destination,
      const void* source, size_t bytes, const char* label,
      Step1ComputeTimings* timings) {
      if(!ensure_pinned_staging(bytes)) {
        check_cuda(cudaMemcpy(destination, source, bytes,
          cudaMemcpyHostToDevice), label);
        return;
      }

      bool in_flight[2] = {false, false};
      size_t offset = 0;
      size_t chunk_index = 0;
      while(offset < bytes) {
        const int slot = static_cast<int>(chunk_index % 2);
        if(in_flight[slot])
          check_cuda(cudaStreamSynchronize(upload_streams_[slot]),
            "synchronize reusable pinned upload staging slot");
        const size_t chunk_bytes = std::min(
          pinned_staging_capacity_, bytes - offset);
        std::memcpy(pinned_staging_[slot],
          reinterpret_cast<const char*>(source) + offset, chunk_bytes);
        check_cuda(cudaMemcpyAsync(
          reinterpret_cast<char*>(destination) + offset,
          pinned_staging_[slot], chunk_bytes, cudaMemcpyHostToDevice,
          upload_streams_[slot]), label);
        in_flight[slot] = true;
        offset += chunk_bytes;
        chunk_index++;
      }
      for(int slot = 0; slot < 2; ++slot)
        if(in_flight[slot])
          check_cuda(cudaStreamSynchronize(upload_streams_[slot]),
            "finish pinned staged upload");
      if(timings) {
        timings->pinned_staging_upload_count++;
        timings->pinned_staging_upload_bytes += bytes;
      }
    }

    void invalidate_resident_genotypes() {
      resident_host_data_ = nullptr;
      resident_rows_ = 0;
      resident_columns_ = 0;
      resident_valid_ = false;
      invalidate_resident_fold_systems();
    }

    void invalidate_resident_fold_systems() {
      resident_fold_system_count_ = 0;
      resident_fold_rhs_count_ = 0;
      resident_fold_systems_valid_ = false;
      resident_fold_systems_design_orientation_ = false;
    }

    void release_packed_hardcall_buffers_noexcept() {
      for(const auto& registration : registered_packed_hardcall_buffers_)
        cudaHostUnregister(registration.first);
      registered_packed_hardcall_buffers_.clear();
      cudaGetLastError();
    }

    void invalidate_resident_design() {
      resident_design_rows_ = 0;
      resident_design_columns_ = 0;
      resident_design_valid_ = false;
      resident_design_uses_level1_cache_ = false;
      if(resident_fold_systems_design_orientation_)
        invalidate_resident_fold_systems();
    }

    const double* resident_design_data() const {
      return resident_design_uses_level1_cache_ ?
        d_level1_design_ : d_resident_genotypes_;
    }

    const double* resident_genotype_columns(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix,
      Eigen::Index start_column, Eigen::Index column_count) const {

      if(!resident_valid_ || !resident_host_data_ ||
         matrix.rows() != resident_rows_ || matrix.innerStride() != 1 ||
         matrix.outerStride() != resident_rows_ || start_column < 0 ||
         column_count < 0 || start_column > matrix.cols() - column_count)
        return nullptr;

      const std::uintptr_t resident_address =
        reinterpret_cast<std::uintptr_t>(resident_host_data_);
      const std::uintptr_t matrix_address =
        reinterpret_cast<std::uintptr_t>(matrix.data());
      if(matrix_address < resident_address) return nullptr;
      const std::uintptr_t byte_offset = matrix_address - resident_address;
      if(byte_offset % sizeof(double) != 0) return nullptr;
      const Eigen::Index element_offset =
        static_cast<Eigen::Index>(byte_offset / sizeof(double));
      if(resident_rows_ <= 0 || element_offset % resident_rows_ != 0)
        return nullptr;
      const Eigen::Index first_column = element_offset / resident_rows_;
      if(first_column < 0 ||
         first_column > resident_columns_ - matrix.cols())
        return nullptr;
      return d_resident_genotypes_ +
        (first_column + start_column) * resident_rows_;
    }

    static Eigen::MatrixXd contiguous_copy_if_needed(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
      if(matrix.innerStride() == 1 && matrix.outerStride() == matrix.rows())
        return Eigen::MatrixXd();
      return Eigen::MatrixXd(matrix);
    }

    static void copy_matrix_row_chunk_to_device(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix,
      Eigen::Index start_row, Eigen::Index row_count,
      double* destination, const char* label) {

      if(matrix.innerStride() == 1) {
        const size_t row_bytes = static_cast<size_t>(row_count) *
          sizeof(double);
        if(start_row == 0 && row_count == matrix.rows() &&
           matrix.outerStride() == matrix.rows()) {
          check_cuda(cudaMemcpy(destination, matrix.data(),
            static_cast<size_t>(matrix.size()) * sizeof(double),
            cudaMemcpyHostToDevice), label);
        } else {
          check_cuda(cudaMemcpy2D(destination, row_bytes,
            matrix.data() + start_row,
            static_cast<size_t>(matrix.outerStride()) * sizeof(double),
            row_bytes, static_cast<size_t>(matrix.cols()),
            cudaMemcpyHostToDevice), label);
        }
        return;
      }

      const Eigen::MatrixXd packed =
        matrix.middleRows(start_row, row_count);
      check_cuda(cudaMemcpy(destination, packed.data(),
        static_cast<size_t>(packed.size()) * sizeof(double),
        cudaMemcpyHostToDevice), label);
    }

    static void validate_ridge_dimensions(
      const Eigen::Ref<const Eigen::MatrixXd>& eigenvectors,
      const Eigen::Ref<const Eigen::MatrixXd>& eigenvalues,
      const Eigen::Ref<const Eigen::MatrixXd>& transformed_right_hand_sides,
      const Eigen::Ref<const Eigen::MatrixXd>& prediction_matrix,
      bool samples_in_columns,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::MatrixXd>& leave_one_out_outcomes,
      bool leave_one_out) {

      const Eigen::Index size = eigenvectors.rows();
      if(eigenvectors.cols() != size || eigenvalues.rows() != size || eigenvalues.cols() != 1 ||
         transformed_right_hand_sides.rows() != size ||
         (samples_in_columns ? prediction_matrix.rows() : prediction_matrix.cols()) != size)
        throw std::invalid_argument("Step 1 ridge prediction received incompatible matrix dimensions");
      if((ridge_parameters.array() < 0).any())
        throw std::invalid_argument("Step 1 ridge parameters must be non-negative");
      if(leave_one_out &&
         (leave_one_out_outcomes.rows() !=
            (samples_in_columns ? prediction_matrix.cols() : prediction_matrix.rows()) ||
          leave_one_out_outcomes.cols() != transformed_right_hand_sides.cols()))
        throw std::invalid_argument("Step 1 LOOCV outcomes have incompatible dimensions");
    }

    static void ensure_capacity(double*& pointer, size_t& capacity,
      Eigen::Index required, const char* label) {
      if(required < 0)
        throw std::runtime_error(
          std::string("negative CUDA allocation size for ") + label);
      const size_t required_size = static_cast<size_t>(required);
      if(required_size <= capacity) return;
      if(required_size > std::numeric_limits<size_t>::max() / sizeof(double))
        throw std::runtime_error(std::string("CUDA allocation size overflow for ") + label);
      if(pointer) check_cuda(cudaFree(pointer), "cudaFree while growing buffer");
      pointer = nullptr;
      capacity = 0;
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
        required_size * sizeof(double)), label);
      capacity = required_size;
    }

    static void ensure_capacity(unsigned char*& pointer, size_t& capacity,
      size_t required, const char* label) {
      if(required <= capacity) return;
      if(pointer) check_cuda(cudaFree(pointer),
        "cudaFree while growing byte buffer");
      pointer = nullptr;
      capacity = 0;
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), required),
        label);
      capacity = required;
    }

    static void ensure_capacity(unsigned int*& pointer, size_t& capacity,
      Eigen::Index required, const char* label) {
      if(required < 0)
        throw std::runtime_error(
          std::string("negative CUDA allocation size for ") + label);
      const size_t required_size = static_cast<size_t>(required);
      if(required_size <= capacity) return;
      if(required_size >
         std::numeric_limits<size_t>::max() / sizeof(unsigned int))
        throw std::runtime_error(
          std::string("CUDA allocation size overflow for ") + label);
      if(pointer) check_cuda(cudaFree(pointer),
        "cudaFree while growing unsigned buffer");
      pointer = nullptr;
      capacity = 0;
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
        required_size * sizeof(unsigned int)), label);
      capacity = required_size;
    }

    static void ensure_capacity(int*& pointer, size_t& capacity,
      Eigen::Index required, const char* label) {
      if(required < 0)
        throw std::runtime_error(
          std::string("negative CUDA allocation size for ") + label);
      const size_t required_size = static_cast<size_t>(required);
      if(required_size <= capacity) return;
      if(required_size >
         std::numeric_limits<size_t>::max() / sizeof(int))
        throw std::runtime_error(
          std::string("CUDA allocation size overflow for ") + label);
      if(pointer) check_cuda(cudaFree(pointer),
        "cudaFree while growing integer buffer");
      pointer = nullptr;
      capacity = 0;
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
        required_size * sizeof(int)), label);
      capacity = required_size;
    }

    int device_;
    cudaDeviceProp properties_;
    cublasHandle_t handle_;
    cusolverDnHandle_t solver_handle_;
    size_t pinned_staging_chunk_bytes_;
    bool level0_cholesky_enabled_;
    bool level0_fold_batch_enabled_;
    bool level0_resident_folds_enabled_;
    bool register_packed_hardcalls_enabled_;
    bool direct_grouped_upload_;
    Eigen::Index resident_preprocess_max_elements_;
    Eigen::Index level1_resident_max_elements_;
    void* pinned_staging_[2] = {nullptr, nullptr};
    cudaStream_t upload_streams_[2] = {nullptr, nullptr};
    double* d_genotypes_;
    double* d_resident_genotypes_;
    double* d_phenotypes_;
    double* d_gram_;
    double* d_crossproduct_;
    double* d_factorized_;
    double* d_ridge_vectors_;
    double* d_ridge_values_;
    double* d_ridge_rhs_;
    double* d_eigenvalues_;
    double* d_solver_workspace_;
    int* d_solver_info_;
    double* d_ridge_parameters_;
    double* d_inverse_;
    double* d_scaled_rhs_;
    double* d_predictions_;
    double* d_outcomes_;
    double* d_projected_;
    double* d_level1_design_;
    double* d_level1_ones_;
    double* d_level0_phenotypes_;
    double* d_squared_;
    double* d_leverage_;
    double* d_preprocess_covariates_;
    double* d_preprocess_weights_;
    double* d_preprocess_coefficients_;
    double* d_preprocess_scales_;
    double* d_preprocess_multipliers_;
    unsigned char* d_packed_hardcalls_;
    unsigned char* d_transposed_hardcalls_;
    unsigned int* d_packed_row_counts_;
    size_t genotypes_capacity_;
    size_t resident_genotypes_capacity_;
    const double* resident_host_data_;
    Eigen::Index resident_rows_;
    Eigen::Index resident_columns_;
    bool resident_valid_;
    Eigen::Index resident_design_rows_;
    Eigen::Index resident_design_columns_;
    bool resident_design_valid_;
    bool resident_design_uses_level1_cache_;
    Eigen::Index level1_design_rows_;
    Eigen::Index level1_design_columns_;
    Eigen::Index level1_design_cached_columns_;
    Eigen::Index resident_fold_system_count_;
    Eigen::Index resident_fold_rhs_count_;
    bool resident_fold_systems_valid_;
    bool resident_fold_systems_design_orientation_;
    bool packed_static_inputs_valid_;
    const double* packed_static_covariates_;
    const double* packed_static_weights_;
    Eigen::Index packed_static_samples_;
    Eigen::Index packed_static_covariate_count_;
    const double* level0_phenotypes_host_;
    Eigen::Index level0_phenotype_rows_;
    Eigen::Index level0_phenotype_columns_;
    size_t phenotypes_capacity_;
    size_t gram_capacity_;
    size_t factorized_capacity_;
    int factorized_size_;
    size_t ridge_vectors_capacity_;
    size_t ridge_values_capacity_;
    size_t ridge_rhs_capacity_;
    size_t crossproduct_capacity_;
    size_t eigenvalues_capacity_;
    size_t solver_workspace_capacity_;
    size_t pinned_staging_capacity_;
    bool pinned_staging_available_;
    size_t ridge_parameters_capacity_ = 0;
    size_t inverse_capacity_ = 0;
    size_t scaled_rhs_capacity_ = 0;
    size_t predictions_capacity_ = 0;
    size_t outcomes_capacity_ = 0;
    size_t projected_capacity_ = 0;
    size_t level1_design_capacity_ = 0;
    size_t level1_ones_capacity_ = 0;
    size_t level0_phenotypes_capacity_ = 0;
    size_t squared_capacity_ = 0;
    size_t leverage_capacity_ = 0;
    size_t preprocess_covariates_capacity_ = 0;
    size_t preprocess_weights_capacity_ = 0;
    size_t preprocess_coefficients_capacity_ = 0;
    size_t preprocess_scales_capacity_ = 0;
    size_t preprocess_multipliers_capacity_ = 0;
    size_t packed_hardcalls_capacity_ = 0;
    size_t transposed_hardcalls_capacity_ = 0;
    size_t packed_row_counts_capacity_ = 0;
    std::vector<std::pair<unsigned char*, size_t>>
      registered_packed_hardcall_buffers_;
    int ridge_factorized_size_;
    int ridge_factorized_rhs_count_;
    std::vector<CudaLevel0CholeskyLane> level0_cholesky_lanes_;
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
