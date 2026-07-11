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
#include <cstdlib>
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

__global__ void square_elements(const double* input, double* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) output[index] = input[index] * input[index];
}

__global__ void scale_matrix_rows(const double* input, const double* weights,
  double* output, int rows, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < count) output[index] = input[index] * weights[index % rows];
}

__global__ void add_diagonal_penalty(double* matrix,
  const double* penalty_multipliers, double ridge_parameter, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size)
    matrix[index + index * size] += ridge_parameter * penalty_multipliers[index];
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

class CudaStep1ComputeBackend : public Step1ComputeBackend {
  public:
    explicit CudaStep1ComputeBackend(int device)
      : device_(device), handle_(nullptr), solver_handle_(nullptr), d_genotypes_(nullptr),
        d_phenotypes_(nullptr), d_gram_(nullptr), d_crossproduct_(nullptr),
        d_factorized_(nullptr),
        d_ridge_vectors_(nullptr), d_ridge_values_(nullptr),
        d_ridge_rhs_(nullptr),
        d_eigenvalues_(nullptr), d_solver_workspace_(nullptr), d_solver_info_(nullptr),
        d_ridge_parameters_(nullptr), d_inverse_(nullptr), d_scaled_rhs_(nullptr),
        d_predictions_(nullptr), d_outcomes_(nullptr), d_projected_(nullptr),
        d_squared_(nullptr), d_leverage_(nullptr),
        genotypes_capacity_(0), phenotypes_capacity_(0), gram_capacity_(0),
        factorized_capacity_(0), factorized_size_(-1),
        ridge_vectors_capacity_(0), ridge_values_capacity_(0),
        ridge_rhs_capacity_(0),
        crossproduct_capacity_(0), eigenvalues_capacity_(0), solver_workspace_capacity_(0),
        ridge_factorized_size_(-1), ridge_factorized_rhs_count_(0) {

      check_cuda(cudaSetDevice(device_), "cudaSetDevice");
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
      if(d_leverage_) cudaFree(d_leverage_);
      if(d_squared_) cudaFree(d_squared_);
      if(d_projected_) cudaFree(d_projected_);
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

      const Eigen::Index chunk_samples = bounded_cuda_chunk_rows(
        genotypes.cols(), genotypes.rows());
      ensure_capacity(d_genotypes_, genotypes_capacity_,
        chunk_samples * genotypes.rows(), "cudaMalloc(genotype chunk)");
      ensure_capacity(d_gram_, gram_capacity_, gram.size(), "cudaMalloc(Gram matrix)");
      if(phenotype_count > 0) {
        ensure_capacity(d_phenotypes_, phenotypes_capacity_,
          chunk_samples * phenotypes.cols(), "cudaMalloc(phenotype chunk)");
        ensure_capacity(d_crossproduct_, crossproduct_capacity_, crossproduct.size(), "cudaMalloc(crossproduct)");
      }

      const double alpha = 1.0;
      for(Eigen::Index start = 0; start < genotypes.cols();
          start += chunk_samples) {
        const Eigen::Index count_index = std::min(
          chunk_samples, genotypes.cols() - start);
        const int count = checked_int(count_index,
          "genotype product chunk sample count");
        const Eigen::MatrixXd genotype_chunk =
          genotypes.middleCols(start, count_index);
        const Eigen::MatrixXd phenotype_chunk = phenotype_count > 0 ?
          Eigen::MatrixXd(phenotypes.middleRows(start, count_index)) :
          Eigen::MatrixXd();

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, genotype_chunk.data(),
          genotype_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy genotype chunk to CUDA device");
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
            d_genotypes_, blocks, d_phenotypes_, count, &beta,
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
            CUBLAS_OP_N, blocks, count, &alpha, d_genotypes_, blocks,
            &beta, d_gram_, blocks),
            "cublasDsyrk(genotype Gram chunk)");
        else
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            blocks, blocks, count, &alpha,
            d_genotypes_, blocks, d_genotypes_, blocks, &beta,
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
        const Eigen::MatrixXd design_chunk =
          design.middleRows(start, count_index);
        const Eigen::MatrixXd outcomes_chunk = outcome_count > 0 ?
          Eigen::MatrixXd(outcomes.middleRows(start, count_index)) :
          Eigen::MatrixXd();

        ComputeClock::time_point transfer_start;
        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
          design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy design product chunk to CUDA device");
        if(outcome_count > 0)
          check_cuda(cudaMemcpy(d_phenotypes_, outcomes_chunk.data(),
            outcomes_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
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
      ensure_capacity(d_ridge_parameters_, ridge_parameters_capacity_,
        penalty_multipliers.size(), "cudaMalloc(diagonal-penalty multipliers)");
      ensure_capacity(d_projected_, projected_capacity_, gram.size(),
        "cudaMalloc(diagonal-penalty factorization matrix)");
      ensure_capacity(d_scaled_rhs_, scaled_rhs_capacity_, right_hand_sides.size(),
        "cudaMalloc(diagonal-penalty solve workspace)");
      ensure_capacity(d_predictions_, predictions_capacity_, solutions.size(),
        "cudaMalloc(diagonal-penalty solutions)");

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
      check_cuda(cudaMemcpy(d_ridge_parameters_, penalty_multipliers.data(),
        penalty_multipliers.size() * sizeof(double), cudaMemcpyHostToDevice),
        "copy diagonal-penalty multipliers to CUDA device");
      if(timings) timings->upload_ms += elapsed_ms(transfer_start);

      int workspace_size = 0;
      check_cusolver(cusolverDnDpotrf_bufferSize(solver_handle_,
        CUBLAS_FILL_MODE_LOWER, size, d_projected_, size, &workspace_size),
        "cusolverDnDpotrf_bufferSize");
      ensure_capacity(d_solver_workspace_, solver_workspace_capacity_, workspace_size,
        "cudaMalloc(cuSOLVER Cholesky workspace)");

      std::unique_ptr<CudaEventPair> solve_events;
      if(timings) {
        solve_events.reset(new CudaEventPair());
        solve_events->record_start();
      }
      const int threads = 256;
      for(int parameter = 0; parameter < parameter_count; ++parameter) {
        check_cuda(cudaMemcpy(d_projected_, d_gram_, gram.size() * sizeof(double),
          cudaMemcpyDeviceToDevice), "copy diagonal-penalty factorization matrix");
        check_cuda(cudaMemcpy(d_scaled_rhs_, d_crossproduct_,
          right_hand_sides.size() * sizeof(double), cudaMemcpyDeviceToDevice),
          "copy diagonal-penalty solve right-hand sides");
        add_diagonal_penalty<<<(size + threads - 1) / threads, threads>>>(
          d_projected_, d_ridge_parameters_, ridge_parameters(parameter), size);
        check_cuda(cudaGetLastError(), "add diagonal penalty kernel");

        check_cusolver(cusolverDnDpotrf(solver_handle_, CUBLAS_FILL_MODE_LOWER,
          size, d_projected_, size, d_solver_workspace_, workspace_size,
          d_solver_info_), "cusolverDnDpotrf");
        int solver_info = 0;
        check_cuda(cudaMemcpy(&solver_info, d_solver_info_, sizeof(int),
          cudaMemcpyDeviceToHost), "copy Cholesky factorization status to host");
        if(solver_info != 0) {
          std::ostringstream message;
          message << "cuSOLVER Cholesky factorization failed with info=" << solver_info;
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
            size * right_hand_side_count,
          d_scaled_rhs_, right_hand_sides.size() * sizeof(double),
          cudaMemcpyDeviceToDevice),
          "store diagonal-penalty solutions on CUDA device");
      }
      if(timings) timings->ridge_ms += solve_events->record_stop_and_elapsed_ms();

      if(timings) transfer_start = ComputeClock::now();
      check_cuda(cudaMemcpy(solutions.data(), d_predictions_,
        solutions.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy diagonal-penalty solutions from CUDA device");
      if(timings) timings->download_ms += elapsed_ms(transfer_start);
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
            grouped_leave_one_out_predict_factorized(
              loo_design, parameter_coefficients.col(outcome),
              leave_one_out_outcomes.col(outcome), leverage_weights,
              full_group_offset, full_group_size, outcome_predictions,
              timings);
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

      if(factorized_size_ < 0)
        throw std::runtime_error(
          "Step 1 grouped LOOCV prediction requested before factorization");
      if(design.cols() != factorized_size_ ||
         coefficients.size() != factorized_size_ ||
         design.rows() != residuals.size() ||
         design.rows() != leverage_weights.size() ||
         group_offsets.size() != group_sizes.size())
        throw std::invalid_argument(
          "Step 1 grouped LOOCV prediction received incompatible dimensions");
      if(!coefficients.allFinite() || !residuals.allFinite() ||
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
        const Eigen::VectorXd residual_chunk =
          residuals.segment(start, count_index);
        const Eigen::VectorXd weight_chunk =
          leverage_weights.segment(start, count_index);

        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
          design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy grouped LOOCV design chunk to CUDA device");
        check_cuda(cudaMemcpy(d_outcomes_, residual_chunk.data(),
          residual_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
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
          grouped_leave_one_out_predictions<<<
            (count + threads - 1) / threads, threads>>>(
            d_genotypes_, d_projected_, d_outcomes_, d_leverage_,
            d_predictions_, count, group,
            group_offsets(group), group_size);
          check_cuda(cudaGetLastError(),
            "compute grouped LOOCV prediction chunk kernel");
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
        const Eigen::MatrixXd design_chunk = design.middleRows(start, count_index);

        if(timings) transfer_start = ComputeClock::now();
        if(design_chunk.size() > 0)
          check_cuda(cudaMemcpy(d_genotypes_, design_chunk.data(),
            design_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy grouped prediction chunk to CUDA device");
        if(timings) timings->upload_ms += elapsed_ms(transfer_start);

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
          const Eigen::MatrixXd genotype_chunk =
            genotypes.middleCols(start, count_index);
          const Eigen::MatrixXd phenotype_chunk = phenotype_count > 0 ?
            Eigen::MatrixXd(phenotypes.middleRows(start, count_index)) :
            Eigen::MatrixXd();

          ComputeClock::time_point transfer_start;
          if(timings) transfer_start = ComputeClock::now();
          check_cuda(cudaMemcpy(d_genotypes_, genotype_chunk.data(),
            genotype_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy fused ridge genotype chunk to CUDA device");
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
              d_genotypes_, blocks, d_phenotypes_, count, &beta,
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
              CUBLAS_OP_N, blocks, count, &alpha, d_genotypes_, blocks,
              &beta, d_gram_, blocks),
              "cublasDsyrk(fused ridge Gram chunk)");
          else
            check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
              blocks, blocks, count, &alpha,
              d_genotypes_, blocks, d_genotypes_, blocks, &beta,
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

      for(Eigen::Index start = 0; start < sample_count_index;
          start += chunk_samples) {
        const Eigen::Index count_index = std::min(
          chunk_samples, sample_count_index - start);
        const int count = checked_int(
          count_index, "factorized ridge prediction chunk sample count");
        Eigen::MatrixXd prediction_chunk;
        if(samples_in_columns)
          prediction_chunk = prediction_matrix.middleCols(start, count_index);
        else
          prediction_chunk = prediction_matrix.middleRows(start, count_index);
        const Eigen::MatrixXd outcomes_chunk = leave_one_out ?
          Eigen::MatrixXd(
            leave_one_out_outcomes.middleRows(start, count_index)) :
          Eigen::MatrixXd();

        if(timings) transfer_start = ComputeClock::now();
        check_cuda(cudaMemcpy(d_genotypes_, prediction_chunk.data(),
          prediction_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
          "copy factorized ridge prediction chunk to CUDA device");
        if(leave_one_out)
          check_cuda(cudaMemcpy(d_outcomes_, outcomes_chunk.data(),
            outcomes_chunk.size() * sizeof(double), cudaMemcpyHostToDevice),
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
            d_genotypes_, size, d_phenotypes_, size, &beta,
            d_predictions_, count),
            "cublasDgemm(factorized ridge prediction chunk)");
        else
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            count, combination_count, size, &alpha,
            d_genotypes_, count, d_phenotypes_, size, &beta,
            d_predictions_, count),
            "cublasDgemm(design factorized ridge prediction chunk)");

        if(leave_one_out) {
          check_cublas(cublasDgemm(handle_, CUBLAS_OP_T,
            samples_in_columns ? CUBLAS_OP_N : CUBLAS_OP_T,
            size, count, size, &alpha,
            d_ridge_vectors_, size, d_genotypes_,
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
    static Eigen::MatrixXd contiguous_copy_if_needed(
      const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
      if(matrix.innerStride() == 1 && matrix.outerStride() == matrix.rows())
        return Eigen::MatrixXd();
      return Eigen::MatrixXd(matrix);
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

    int device_;
    cudaDeviceProp properties_;
    cublasHandle_t handle_;
    cusolverDnHandle_t solver_handle_;
    double* d_genotypes_;
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
    double* d_squared_;
    double* d_leverage_;
    size_t genotypes_capacity_;
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
    size_t ridge_parameters_capacity_ = 0;
    size_t inverse_capacity_ = 0;
    size_t scaled_rhs_capacity_ = 0;
    size_t predictions_capacity_ = 0;
    size_t outcomes_capacity_ = 0;
    size_t projected_capacity_ = 0;
    size_t squared_capacity_ = 0;
    size_t leverage_capacity_ = 0;
    int ridge_factorized_size_;
    int ridge_factorized_rhs_count_;
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
