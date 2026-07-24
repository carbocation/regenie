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
#include "Step2_Compute.hpp"
#include "Cuda_Resources.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kMaximumCovariates = 16;

enum class ScoreMode {
  none,
  quantitative,
  binary,
  cox
};

void check_cuda(cudaError_t status, const char* operation) {
  if(status == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " +
    cudaGetErrorString(status));
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

__inline__ __device__ double warp_sum(double value) {
  for(int offset = warpSize / 2; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__inline__ __device__ double decode_genotype(
    const unsigned char* packed, int sample, double missing_mean,
    bool flipped) {
  const unsigned int code =
    (packed[sample >> 2] >> (2 * (sample & 3))) & 3;
  if(code == 3) return flipped ? 2.0 - missing_mean : missing_mean;
  const double genotype = static_cast<double>(code);
  return flipped ? 2.0 - genotype : genotype;
}

template<int MaximumTerms>
__inline__ __device__ void reduce_terms(
    double (&terms)[MaximumTerms], int term_count,
    double* shared_terms) {
  for(int term = 0; term < term_count; ++term)
    terms[term] = warp_sum(terms[term]);

  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  if(lane == 0)
    for(int term = 0; term < term_count; ++term)
      shared_terms[term * warp_count + warp] = terms[term];
  __syncthreads();

  if(warp == 0) {
    for(int term = 0; term < term_count; ++term) {
      terms[term] = lane < warp_count ?
        shared_terms[term * warp_count + lane] : 0.0;
      terms[term] = warp_sum(terms[term]);
    }
  }
}

__global__ void quantitative_score_kernel(
    const unsigned char* packed, int packed_stride,
    const double* missing_means, const unsigned char* flipped,
    const double* residuals, const double* covariates,
    const double* outcome_covariate_products,
    int samples, int phenotypes, int covariate_count,
    double* numerators, double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const unsigned char* variant_packed =
    packed + static_cast<size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    residuals + static_cast<size_t>(phenotype) * samples;
  double terms[kMaximumCovariates + 2] = {};

  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const double genotype = decode_genotype(variant_packed, sample,
      missing_means[variant], flipped[variant] != 0);
    terms[0] += genotype * phenotype_residuals[sample];
    terms[1] += genotype * genotype;
    for(int covariate = 0; covariate < covariate_count; ++covariate)
      terms[covariate + 2] += genotype *
        covariates[static_cast<size_t>(covariate) * samples + sample];
  }

  extern __shared__ double shared_terms[];
  reduce_terms(terms, covariate_count + 2, shared_terms);
  if((threadIdx.x & (warpSize - 1)) != 0 ||
     threadIdx.x / warpSize != 0) return;

  double numerator = terms[0];
  double denominator = terms[1];
  for(int covariate = 0; covariate < covariate_count; ++covariate) {
    const double crossproduct = terms[covariate + 2];
    numerator -= crossproduct *
      outcome_covariate_products[
        static_cast<size_t>(phenotype) * covariate_count + covariate];
    denominator -= crossproduct * crossproduct;
  }
  const size_t output =
    static_cast<size_t>(variant) * phenotypes + phenotype;
  numerators[output] = numerator;
  denominators[output] = denominator;
}

__global__ void quantitative_missing_score_kernel(
    const unsigned char* packed, int packed_stride,
    const double* missing_means, const unsigned char* flipped,
    const unsigned char* sparse, const double* residuals,
    const double* covariates, const double* outcome_covariate_products,
    const unsigned char* observed, const double* observed_grams,
    int samples, int phenotypes, int covariate_count,
    double* numerators, double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const unsigned char* variant_packed =
    packed + static_cast<size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    residuals + static_cast<size_t>(phenotype) * samples;
  const unsigned char* phenotype_observed =
    observed + static_cast<size_t>(phenotype) * samples;
  double terms[2 + 2 * kMaximumCovariates] = {};

  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const double genotype = decode_genotype(variant_packed, sample,
      missing_means[variant], flipped[variant] != 0);
    const bool is_observed = phenotype_observed[sample] != 0;
    terms[0] += genotype * phenotype_residuals[sample];
    if(is_observed) terms[1] += genotype * genotype;
    for(int covariate = 0; covariate < covariate_count; ++covariate) {
      const double crossproduct = genotype *
        covariates[static_cast<size_t>(covariate) * samples + sample];
      terms[2 + covariate] += crossproduct;
      if(is_observed)
        terms[2 + covariate_count + covariate] += crossproduct;
    }
  }

  extern __shared__ double shared_terms[];
  reduce_terms(terms, 2 + 2 * covariate_count, shared_terms);
  if((threadIdx.x & (warpSize - 1)) != 0 ||
     threadIdx.x / warpSize != 0) return;

  double numerator = terms[0];
  double denominator = terms[1];
  const size_t phenotype_small =
    static_cast<size_t>(phenotype) * covariate_count;
  for(int row = 0; row < covariate_count; ++row) {
    const double complete_cross = terms[2 + row];
    const double observed_cross =
      terms[2 + covariate_count + row];
    numerator -= complete_cross *
      outcome_covariate_products[phenotype_small + row];
    denominator -= 2 * complete_cross * observed_cross;
    if(sparse[variant]) {
      // Match the existing sparse-QT approximation, which uses the complete
      // X'X=I projection norm after removing missing samples from the raw and
      // cross terms.
      denominator += complete_cross * complete_cross;
    } else {
      double gram_product = 0;
      for(int column = 0; column < covariate_count; ++column)
        gram_product += observed_grams[
          (static_cast<size_t>(phenotype) * covariate_count + row) *
          covariate_count + column] * terms[2 + column];
      denominator += complete_cross * gram_product;
    }
  }
  const size_t output =
    static_cast<size_t>(variant) * phenotypes + phenotype;
  numerators[output] = numerator;
  denominators[output] = denominator;
}

__global__ void observed_trait_counts_kernel(
    const unsigned char* packed, int packed_stride,
    const unsigned char* observed, int samples, int phenotypes,
    double* observed_allele_sums, double* observed_nonmissing_counts) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const unsigned char* variant_packed =
    packed + static_cast<size_t>(variant) * packed_stride;
  const unsigned char* phenotype_observed =
    observed + static_cast<size_t>(phenotype) * samples;
  double terms[2] = {};

  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    if(!phenotype_observed[sample]) continue;
    const unsigned int code =
      (variant_packed[sample >> 2] >> (2 * (sample & 3))) & 3;
    if(code == 3) continue;
    terms[0] += code;
    terms[1] += 1;
  }

  extern __shared__ double shared_terms[];
  reduce_terms(terms, 2, shared_terms);
  if((threadIdx.x & (warpSize - 1)) != 0 ||
     threadIdx.x / warpSize != 0) return;

  const size_t output =
    static_cast<size_t>(variant) * phenotypes + phenotype;
  observed_allele_sums[output] = terms[0];
  observed_nonmissing_counts[output] = terms[1];
}

__global__ void binary_score_kernel(
    const unsigned char* packed, int packed_stride,
    const double* missing_means, const unsigned char* flipped,
    const unsigned char* sparse, const unsigned char* active,
    const double* residuals, const double* weights,
    const double* designs, const double* design_residual_products,
    int samples, int phenotypes, int covariate_count,
    double* numerators, double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const size_t output =
    static_cast<size_t>(variant) * phenotypes + phenotype;
  if(!active[phenotype]) {
    if(threadIdx.x == 0) {
      numerators[output] = 0;
      denominators[output] = 1;
    }
    return;
  }

  const unsigned char* variant_packed =
    packed + static_cast<size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    residuals + static_cast<size_t>(phenotype) * samples;
  const double* phenotype_weights =
    weights + static_cast<size_t>(phenotype) * samples;
  double terms[kMaximumCovariates + 2] = {};

  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const double genotype = decode_genotype(variant_packed, sample,
      missing_means[variant], flipped[variant] != 0);
    const double weighted_genotype =
      genotype * phenotype_weights[sample];
    terms[0] += weighted_genotype * phenotype_residuals[sample];
    terms[1] += weighted_genotype * weighted_genotype;
    for(int covariate = 0; covariate < covariate_count; ++covariate) {
      const size_t design_offset =
        (static_cast<size_t>(phenotype) * covariate_count + covariate) *
        samples + sample;
      terms[covariate + 2] +=
        weighted_genotype * designs[design_offset];
    }
  }

  extern __shared__ double shared_terms[];
  reduce_terms(terms, covariate_count + 2, shared_terms);
  if((threadIdx.x & (warpSize - 1)) != 0 ||
     threadIdx.x / warpSize != 0) return;

  double numerator = terms[0];
  double denominator = terms[1];
  for(int covariate = 0; covariate < covariate_count; ++covariate) {
    const double crossproduct = terms[covariate + 2];
    if(!sparse[variant])
      numerator -= crossproduct *
        design_residual_products[
          static_cast<size_t>(phenotype) * covariate_count + covariate];
    denominator -= crossproduct * crossproduct;
  }
  numerators[output] = numerator;
  denominators[output] = denominator;
}

__global__ void cox_score_kernel(
    const unsigned char* packed, int packed_stride,
    const double* missing_means, const unsigned char* flipped,
    const unsigned char* active, const double* score_residuals,
    const double* weighted_designs, const double* projections,
    const double* projection_scores, const double* projection_grams,
    const double* residual_variances,
    int samples, int phenotypes, int covariate_count,
    double* numerators, double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const size_t output =
    static_cast<size_t>(variant) * phenotypes + phenotype;
  if(!active[phenotype]) {
    if(threadIdx.x == 0) {
      numerators[output] = 0;
      denominators[output] = 1;
    }
    return;
  }

  const unsigned char* variant_packed =
    packed + static_cast<size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    score_residuals + static_cast<size_t>(phenotype) * samples;
  double terms[2 + 2 * kMaximumCovariates] = {};

  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const double genotype = decode_genotype(variant_packed, sample,
      missing_means[variant], flipped[variant] != 0);
    terms[0] += genotype * phenotype_residuals[sample];
    terms[1] += genotype * genotype;
    for(int covariate = 0; covariate < covariate_count; ++covariate) {
      const size_t design_offset =
        (static_cast<size_t>(phenotype) * covariate_count + covariate) *
        samples + sample;
      terms[2 + covariate] +=
        genotype * weighted_designs[design_offset];
      terms[2 + covariate_count + covariate] +=
        genotype * projections[design_offset];
    }
  }

  extern __shared__ double shared_terms[];
  reduce_terms(terms, 2 + 2 * covariate_count, shared_terms);
  if((threadIdx.x & (warpSize - 1)) != 0 ||
     threadIdx.x / warpSize != 0) return;

  double numerator = terms[0];
  double denominator = terms[1];
  const size_t phenotype_small =
    static_cast<size_t>(phenotype) * covariate_count;
  for(int row = 0; row < covariate_count; ++row) {
    const double coefficient = terms[2 + row];
    const double raw_cross = terms[2 + covariate_count + row];
    numerator -= coefficient * projection_scores[phenotype_small + row];
    denominator -= 2 * coefficient * raw_cross;
    double gram_product = 0;
    for(int column = 0; column < covariate_count; ++column)
      gram_product += projection_grams[
        (static_cast<size_t>(phenotype) * covariate_count + row) *
        covariate_count + column] * terms[2 + column];
    denominator += coefficient * gram_product;
  }
  numerators[output] = numerator;
  denominators[output] = residual_variances[phenotype] * denominator;
}

class CudaStep2ComputeBackend : public Step2ComputeBackend {
 public:
  CudaStep2ComputeBackend(int device, bool automatic) :
      device_(device), automatic_(automatic) {
    check_cuda(cudaSetDevice(device_), "select CUDA device for Step 2");
    check_cuda(cudaGetDeviceProperties(&properties_, device_),
      "query CUDA device for Step 2");
  }

  ~CudaStep2ComputeBackend() override {
    release_all();
  }

  const char* name() const override {
    if(attempted_ && mode_ == ScoreMode::none) return "cpu";
    if(mode_ != ScoreMode::none) return "cuda";
    return automatic_ ? "auto" : "cuda";
  }

  std::string description() const override {
    std::ostringstream out;
    out << properties_.name << " packed Step 2 scoring";
    if(automatic_) out << " with per-workflow CPU fallback";
    return out.str();
  }

  bool ready() const override { return mode_ != ScoreMode::none; }
  bool uses_packed_hardcalls() const override { return true; }
  bool provides_observed_trait_counts() const override {
    return ready() && trait_counts_required_;
  }

  void clear() override {
    mode_ = ScoreMode::none;
    attempted_ = false;
    samples_ = phenotypes_ = covariates_ = 0;
    trait_counts_required_ = false;
  }

  bool prepare_quantitative(
      const Eigen::Ref<const Eigen::MatrixXd>& residuals,
      const Eigen::Ref<const Eigen::MatrixXd>& covariates,
      const Eigen::Ref<const Eigen::MatrixXd>& outcome_covariate_products,
      const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
        Eigen::Dynamic>>& observed_masks,
      bool complete_masks,
      Step2ComputeTimings* timings) override {
    clear();
    attempted_ = true;
    // The packed quantitative kernels are throughput-negative on the measured
    // pre-Ampere device even when the GPU is saturated. Keep the explicit CUDA
    // path for benchmarking, but do not select it automatically there.
    if(automatic_ && properties_.major < 8) return false;
    if(residuals.rows() <= 0 || residuals.cols() <= 0 ||
       covariates.rows() != residuals.rows() || covariates.cols() <= 0 ||
       covariates.cols() > kMaximumCovariates ||
       outcome_covariate_products.rows() != residuals.cols() ||
       outcome_covariate_products.cols() != covariates.cols() ||
       observed_masks.rows() != residuals.rows() ||
       observed_masks.cols() != residuals.cols())
      return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    samples_ = static_cast<int>(residuals.rows());
    phenotypes_ = static_cast<int>(residuals.cols());
    covariates_ = static_cast<int>(covariates.cols());
    quantitative_complete_masks_ = complete_masks;
    trait_counts_required_ = !complete_masks;
    active_host_.assign(phenotypes_, 1);
    small_host_.resize(static_cast<size_t>(phenotypes_) * covariates_);
    for(int phenotype = 0; phenotype < phenotypes_; ++phenotype)
      for(int covariate = 0; covariate < covariates_; ++covariate)
        small_host_[static_cast<size_t>(phenotype) * covariates_ + covariate] =
          outcome_covariate_products(phenotype, covariate);

    observed_host_.clear();
    grams_host_.clear();
    if(!complete_masks) {
      observed_host_.resize(static_cast<size_t>(samples_) * phenotypes_);
      grams_host_.assign(static_cast<size_t>(phenotypes_) * covariates_ *
        covariates_, 0);
      const Eigen::MatrixXd complete_gram =
        covariates.transpose() * covariates;
      for(int phenotype = 0; phenotype < phenotypes_; ++phenotype) {
        for(int row = 0; row < covariates_; ++row)
          for(int column = 0; column < covariates_; ++column)
            grams_host_[
              (static_cast<size_t>(phenotype) * covariates_ + row) *
              covariates_ + column] = complete_gram(row, column);
        for(int sample = 0; sample < samples_; ++sample) {
          const bool is_observed = observed_masks(sample, phenotype);
          observed_host_[static_cast<size_t>(phenotype) * samples_ + sample] =
            is_observed ? 1 : 0;
          if(is_observed) continue;
          for(int row = 0; row < covariates_; ++row)
            for(int column = 0; column < covariates_; ++column)
              grams_host_[
                (static_cast<size_t>(phenotype) * covariates_ + row) *
                covariates_ + column] -= covariates(sample, row) *
                  covariates(sample, column);
        }
      }
    }

    upload_static(residuals.data(), residuals.size(), nullptr, 0,
      covariates.data(), covariates.size(), nullptr, 0,
      small_host_.data(), small_host_.size(), grams_host_.data(),
      grams_host_.size(), nullptr, 0);
    if(!complete_masks) {
      replace_buffer(reinterpret_cast<void**>(&d_observed_),
        &observed_capacity_, observed_host_.size(),
        "allocate Step 2 observed masks");
      check_cuda(cudaMemcpy(d_observed_, observed_host_.data(),
        observed_host_.size(), cudaMemcpyHostToDevice),
        "upload Step 2 observed masks");
    }
    mode_ = ScoreMode::quantitative;
    if(timings) {
      timings->prepared_chromosomes++;
      timings->prepare_upload_ms += elapsed_ms(start);
    }
    return true;
  }

  bool prepare_binary(
      const Eigen::Ref<const Eigen::MatrixXd>& residuals,
      const Eigen::Ref<const Eigen::MatrixXd>& weights,
      const std::vector<Eigen::MatrixXd>& designs,
      const std::vector<Eigen::VectorXd>& design_residual_products,
      const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
        Eigen::Dynamic>>& observed_masks,
      const Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, 1>>&
        active_phenotypes,
      Step2ComputeTimings* timings) override {
    clear();
    attempted_ = true;
    if(residuals.rows() <= 0 || residuals.cols() <= 0 ||
       weights.rows() != residuals.rows() ||
       weights.cols() != residuals.cols() ||
       designs.size() != static_cast<size_t>(residuals.cols()) ||
       design_residual_products.size() != designs.size() ||
       observed_masks.rows() != residuals.rows() ||
       observed_masks.cols() != residuals.cols() ||
       active_phenotypes.size() != residuals.cols())
      return false;

    int covariate_count = 0;
    for(int phenotype = 0; phenotype < residuals.cols(); ++phenotype) {
      if(!active_phenotypes(phenotype)) continue;
      covariate_count = static_cast<int>(designs[phenotype].cols());
      break;
    }
    if(covariate_count <= 0 || covariate_count > kMaximumCovariates)
      return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    samples_ = static_cast<int>(residuals.rows());
    phenotypes_ = static_cast<int>(residuals.cols());
    covariates_ = covariate_count;
    active_host_.resize(phenotypes_);
    designs_host_.assign(static_cast<size_t>(samples_) * phenotypes_ *
      covariates_, 0);
    small_host_.assign(static_cast<size_t>(phenotypes_) * covariates_, 0);
    for(int phenotype = 0; phenotype < phenotypes_; ++phenotype) {
      active_host_[phenotype] = active_phenotypes(phenotype) ? 1 : 0;
      if(!active_host_[phenotype]) continue;
      if(designs[phenotype].rows() != samples_ ||
         designs[phenotype].cols() != covariates_ ||
         design_residual_products[phenotype].size() != covariates_)
        return false;
      std::memcpy(designs_host_.data() +
          static_cast<size_t>(phenotype) * covariates_ * samples_,
        designs[phenotype].data(),
        static_cast<size_t>(samples_) * covariates_ * sizeof(double));
      for(int covariate = 0; covariate < covariates_; ++covariate)
        small_host_[static_cast<size_t>(phenotype) * covariates_ +
          covariate] = design_residual_products[phenotype](covariate);
    }

    upload_static(residuals.data(), residuals.size(), weights.data(),
      weights.size(), designs_host_.data(), designs_host_.size(), nullptr, 0,
      small_host_.data(), small_host_.size(), nullptr, 0, nullptr, 0);
    prepare_observed_masks(observed_masks);
    mode_ = ScoreMode::binary;
    if(timings) {
      timings->prepared_chromosomes++;
      timings->prepare_upload_ms += elapsed_ms(start);
    }
    return true;
  }

  bool prepare_cox(
      const std::vector<Eigen::VectorXd>& score_residuals,
      const std::vector<Eigen::MatrixXd>& weighted_designs,
      const std::vector<Eigen::MatrixXd>& projections,
      const Eigen::Ref<const Eigen::MatrixXd>& common_projection_design,
      const std::vector<Eigen::MatrixXd>& projection_transforms,
      const std::vector<Eigen::VectorXd>& projection_scores,
      const std::vector<Eigen::MatrixXd>& projection_grams,
      const Eigen::Ref<const Eigen::VectorXd>& residual_variances,
      const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
        Eigen::Dynamic>>& observed_masks,
      const Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, 1>>&
        active_phenotypes,
      Step2ComputeTimings* timings) override {
    (void)common_projection_design;
    (void)projection_transforms;
    clear();
    attempted_ = true;
    const int phenotype_count = static_cast<int>(score_residuals.size());
    if(phenotype_count <= 0 ||
       weighted_designs.size() != score_residuals.size() ||
       projections.size() != score_residuals.size() ||
       projection_scores.size() != score_residuals.size() ||
       projection_grams.size() != score_residuals.size() ||
       residual_variances.size() != phenotype_count ||
       observed_masks.cols() != phenotype_count ||
       active_phenotypes.size() != phenotype_count)
      return false;

    int sample_count = 0;
    int covariate_count = 0;
    for(int phenotype = 0; phenotype < phenotype_count; ++phenotype) {
      if(!active_phenotypes(phenotype)) continue;
      sample_count = static_cast<int>(score_residuals[phenotype].size());
      covariate_count = static_cast<int>(weighted_designs[phenotype].cols());
      break;
    }
    if(sample_count <= 0 || covariate_count <= 0 ||
       covariate_count > kMaximumCovariates ||
       observed_masks.rows() != sample_count)
      return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    samples_ = sample_count;
    phenotypes_ = phenotype_count;
    covariates_ = covariate_count;
    active_host_.resize(phenotypes_);
    residuals_host_.assign(static_cast<size_t>(samples_) * phenotypes_, 0);
    designs_host_.assign(static_cast<size_t>(samples_) * phenotypes_ *
      covariates_, 0);
    projections_host_.assign(designs_host_.size(), 0);
    small_host_.assign(static_cast<size_t>(phenotypes_) * covariates_, 0);
    grams_host_.assign(static_cast<size_t>(phenotypes_) * covariates_ *
      covariates_, 0);
    variances_host_.resize(phenotypes_);

    for(int phenotype = 0; phenotype < phenotypes_; ++phenotype) {
      active_host_[phenotype] = active_phenotypes(phenotype) ? 1 : 0;
      variances_host_[phenotype] = residual_variances(phenotype);
      if(!active_host_[phenotype]) continue;
      if(score_residuals[phenotype].size() != samples_ ||
         weighted_designs[phenotype].rows() != samples_ ||
         weighted_designs[phenotype].cols() != covariates_ ||
         projections[phenotype].rows() != samples_ ||
         projections[phenotype].cols() != covariates_ ||
         projection_scores[phenotype].size() != covariates_ ||
         projection_grams[phenotype].rows() != covariates_ ||
         projection_grams[phenotype].cols() != covariates_)
        return false;
      std::memcpy(residuals_host_.data() +
          static_cast<size_t>(phenotype) * samples_,
        score_residuals[phenotype].data(),
        static_cast<size_t>(samples_) * sizeof(double));
      std::memcpy(designs_host_.data() +
          static_cast<size_t>(phenotype) * covariates_ * samples_,
        weighted_designs[phenotype].data(),
        static_cast<size_t>(samples_) * covariates_ * sizeof(double));
      std::memcpy(projections_host_.data() +
          static_cast<size_t>(phenotype) * covariates_ * samples_,
        projections[phenotype].data(),
        static_cast<size_t>(samples_) * covariates_ * sizeof(double));
      for(int covariate = 0; covariate < covariates_; ++covariate)
        small_host_[static_cast<size_t>(phenotype) * covariates_ +
          covariate] = projection_scores[phenotype](covariate);
      for(int row = 0; row < covariates_; ++row)
        for(int column = 0; column < covariates_; ++column)
          grams_host_[
            (static_cast<size_t>(phenotype) * covariates_ + row) *
            covariates_ + column] =
              projection_grams[phenotype](row, column);
    }

    upload_static(residuals_host_.data(), residuals_host_.size(), nullptr, 0,
      designs_host_.data(), designs_host_.size(),
      projections_host_.data(), projections_host_.size(),
      small_host_.data(), small_host_.size(), grams_host_.data(),
      grams_host_.size(), variances_host_.data(), variances_host_.size());
    prepare_observed_masks(observed_masks);
    mode_ = ScoreMode::cox;
    if(timings) {
      timings->prepared_chromosomes++;
      timings->prepare_upload_ms += elapsed_ms(start);
    }
    return true;
  }

  bool score_packed_block(
      const std::vector<std::vector<unsigned char>>& packed_hardcalls,
      const std::vector<double>& missing_means,
      const std::vector<unsigned char>& flipped,
      const std::vector<unsigned char>& sparse,
      Eigen::Index samples,
      Eigen::MatrixXd& numerators,
      Eigen::MatrixXd& denominators,
      Eigen::MatrixXd& observed_allele_sums,
      Eigen::MatrixXd& observed_nonmissing_counts,
      Step2ComputeTimings* timings) override {
    // Block scoring may run on the Step 2 pipeline worker thread. CUDA's
    // selected device is thread-local, so select it explicitly here rather
    // than relying on the thread that prepared the chromosome.
    check_cuda(cudaSetDevice(device_), "select CUDA device for Step 2 score");
    if(!ready() || samples != samples_ || packed_hardcalls.empty())
      return false;
    const int variants = static_cast<int>(packed_hardcalls.size());
    if(missing_means.size() != packed_hardcalls.size() ||
       flipped.size() != packed_hardcalls.size() ||
       sparse.size() != packed_hardcalls.size())
      return false;
    const size_t packed_stride =
      (static_cast<size_t>(samples_) + 3) / 4;
    packed_host_.resize(static_cast<size_t>(variants) * packed_stride);

    const std::chrono::steady_clock::time_point wall_start =
      std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point pack_start = wall_start;
    for(int variant = 0; variant < variants; ++variant) {
      if(packed_hardcalls[variant].size() != packed_stride)
        return false;
      std::memcpy(packed_host_.data() +
          static_cast<size_t>(variant) * packed_stride,
        packed_hardcalls[variant].data(), packed_stride);
    }
    if(timings) timings->host_pack_ms += elapsed_ms(pack_start);

    const bool return_trait_counts = trait_counts_required_;
    const size_t output_count =
      static_cast<size_t>(variants) * phenotypes_;
    ensure_dynamic(packed_host_.size(), variants, output_count,
      return_trait_counts);
    regenie::cuda::EventPair events;

    events.record_start();
    check_cuda(cudaMemcpyAsync(d_packed_, packed_host_.data(),
      packed_host_.size(), cudaMemcpyHostToDevice),
      "upload Step 2 packed hardcalls");
    check_cuda(cudaMemcpyAsync(d_means_, missing_means.data(),
      static_cast<size_t>(variants) * sizeof(double), cudaMemcpyHostToDevice),
      "upload Step 2 missing means");
    check_cuda(cudaMemcpyAsync(d_flipped_, flipped.data(), variants,
      cudaMemcpyHostToDevice), "upload Step 2 flip flags");
    check_cuda(cudaMemcpyAsync(d_sparse_, sparse.data(), variants,
      cudaMemcpyHostToDevice), "upload Step 2 sparse flags");
    const double upload_ms = events.record_stop_and_elapsed_ms();
    if(timings) timings->upload_ms += upload_ms;

    const dim3 grid(variants, phenotypes_);
    const int threads = 256;
    const int warp_count = threads / 32;
    int term_count = covariates_ + 2;
    if(mode_ == ScoreMode::cox ||
       (mode_ == ScoreMode::quantitative &&
        !quantitative_complete_masks_))
      term_count = 2 + 2 * covariates_;
    const size_t shared_bytes =
      static_cast<size_t>(term_count) * warp_count * sizeof(double);
    events.record_start();
    if(mode_ == ScoreMode::quantitative && quantitative_complete_masks_) {
      quantitative_score_kernel<<<grid, threads, shared_bytes>>>(d_packed_,
        static_cast<int>(packed_stride), d_means_, d_flipped_, d_residuals_,
        d_designs_, d_small_, samples_, phenotypes_, covariates_,
        d_numerators_, d_denominators_);
    } else if(mode_ == ScoreMode::quantitative) {
      quantitative_missing_score_kernel<<<grid, threads, shared_bytes>>>(
        d_packed_, static_cast<int>(packed_stride), d_means_, d_flipped_,
        d_sparse_, d_residuals_, d_designs_, d_small_, d_observed_, d_grams_,
        samples_, phenotypes_, covariates_, d_numerators_, d_denominators_);
    } else if(mode_ == ScoreMode::binary) {
      binary_score_kernel<<<grid, threads, shared_bytes>>>(d_packed_,
        static_cast<int>(packed_stride), d_means_, d_flipped_, d_sparse_,
        d_active_, d_residuals_, d_weights_, d_designs_, d_small_, samples_,
        phenotypes_, covariates_, d_numerators_, d_denominators_);
    } else {
      cox_score_kernel<<<grid, threads, shared_bytes>>>(d_packed_,
        static_cast<int>(packed_stride), d_means_, d_flipped_, d_active_,
        d_residuals_, d_designs_, d_projections_, d_small_, d_grams_,
        d_variances_, samples_, phenotypes_, covariates_, d_numerators_,
        d_denominators_);
    }
    check_cuda(cudaGetLastError(), "launch Step 2 packed score kernel");
    if(return_trait_counts) {
      const size_t count_shared_bytes =
        static_cast<size_t>(2 * warp_count) * sizeof(double);
      observed_trait_counts_kernel<<<grid, threads, count_shared_bytes>>>(
        d_packed_, static_cast<int>(packed_stride), d_observed_, samples_,
        phenotypes_, d_observed_allele_sums_,
        d_observed_nonmissing_counts_);
      check_cuda(cudaGetLastError(),
        "launch Step 2 observed trait counts kernel");
    }
    const double kernel_ms = events.record_stop_and_elapsed_ms();
    if(timings) timings->kernel_ms += kernel_ms;

    numerators.resize(phenotypes_, variants);
    denominators.resize(phenotypes_, variants);
    if(return_trait_counts) {
      observed_allele_sums.resize(phenotypes_, variants);
      observed_nonmissing_counts.resize(phenotypes_, variants);
    } else {
      observed_allele_sums.resize(0, 0);
      observed_nonmissing_counts.resize(0, 0);
    }
    events.record_start();
    check_cuda(cudaMemcpyAsync(numerators.data(), d_numerators_,
      static_cast<size_t>(variants) * phenotypes_ * sizeof(double),
      cudaMemcpyDeviceToHost), "download Step 2 score numerators");
    check_cuda(cudaMemcpyAsync(denominators.data(), d_denominators_,
      static_cast<size_t>(variants) * phenotypes_ * sizeof(double),
      cudaMemcpyDeviceToHost), "download Step 2 score denominators");
    if(return_trait_counts) {
      check_cuda(cudaMemcpyAsync(observed_allele_sums.data(),
        d_observed_allele_sums_, output_count * sizeof(double),
        cudaMemcpyDeviceToHost),
        "download Step 2 observed allele sums");
      check_cuda(cudaMemcpyAsync(observed_nonmissing_counts.data(),
        d_observed_nonmissing_counts_, output_count * sizeof(double),
        cudaMemcpyDeviceToHost),
        "download Step 2 observed nonmissing counts");
    }
    const double download_ms = events.record_stop_and_elapsed_ms();
    if(timings) {
      timings->download_ms += download_ms;
      timings->scored_blocks++;
      timings->scored_variants += variants;
      timings->packed_upload_bytes += packed_host_.size();
      timings->wall_ms += elapsed_ms(wall_start);
    }
    return numerators.allFinite() && denominators.allFinite();
  }

  bool score_dense_block(
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const std::vector<unsigned char>&,
      const Eigen::RowVectorXd*,
      Eigen::MatrixXd&, Eigen::MatrixXd&,
      Step2ComputeTimings*) override {
    return false;
  }

 private:
  void replace_buffer(void** pointer, size_t* capacity, size_t bytes,
      const char* label) {
    if(bytes <= *capacity) return;
    if(*pointer) check_cuda(cudaFree(*pointer), "release CUDA Step 2 buffer");
    *pointer = nullptr;
    *capacity = 0;
    if(bytes == 0) return;
    check_cuda(cudaMalloc(pointer, bytes), label);
    *capacity = bytes;
  }

  void copy_static(double** pointer, size_t* capacity,
      const double* values, size_t count, const char* label) {
    if(count == 0) return;
    replace_buffer(reinterpret_cast<void**>(pointer), capacity,
      count * sizeof(double), label);
    check_cuda(cudaMemcpy(*pointer, values, count * sizeof(double),
      cudaMemcpyHostToDevice), label);
  }

  void upload_static(const double* residuals, size_t residual_count,
      const double* weights, size_t weight_count,
      const double* designs, size_t design_count,
      const double* projections, size_t projection_count,
      const double* small, size_t small_count,
      const double* grams, size_t gram_count,
      const double* variances, size_t variance_count) {
    copy_static(&d_residuals_, &residual_capacity_, residuals,
      residual_count, "upload Step 2 residuals");
    copy_static(&d_weights_, &weight_capacity_, weights,
      weight_count, "upload Step 2 weights");
    copy_static(&d_designs_, &design_capacity_, designs,
      design_count, "upload Step 2 designs");
    copy_static(&d_projections_, &projection_capacity_, projections,
      projection_count, "upload Step 2 projections");
    copy_static(&d_small_, &small_capacity_, small,
      small_count, "upload Step 2 small products");
    copy_static(&d_grams_, &gram_capacity_, grams,
      gram_count, "upload Step 2 projection Grams");
    copy_static(&d_variances_, &variance_capacity_, variances,
      variance_count, "upload Step 2 residual variances");
    replace_buffer(reinterpret_cast<void**>(&d_active_), &active_capacity_,
      active_host_.size(), "allocate Step 2 active traits");
    check_cuda(cudaMemcpy(d_active_, active_host_.data(), active_host_.size(),
      cudaMemcpyHostToDevice), "upload Step 2 active traits");
  }

  void prepare_observed_masks(
      const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
        Eigen::Dynamic>>& observed_masks) {
    trait_counts_required_ = !observed_masks.array().all();
    observed_host_.clear();
    if(!trait_counts_required_) return;
    observed_host_.resize(static_cast<size_t>(samples_) * phenotypes_);
    for(int phenotype = 0; phenotype < phenotypes_; ++phenotype)
      for(int sample = 0; sample < samples_; ++sample)
        observed_host_[static_cast<size_t>(phenotype) * samples_ + sample] =
          observed_masks(sample, phenotype) ? 1 : 0;
    replace_buffer(reinterpret_cast<void**>(&d_observed_),
      &observed_capacity_, observed_host_.size(),
      "allocate Step 2 observed masks");
    check_cuda(cudaMemcpy(d_observed_, observed_host_.data(),
      observed_host_.size(), cudaMemcpyHostToDevice),
      "upload Step 2 observed masks");
  }

  void ensure_dynamic(size_t packed_bytes, int variants,
      size_t output_count, bool trait_counts) {
    replace_buffer(reinterpret_cast<void**>(&d_packed_), &packed_capacity_,
      packed_bytes, "allocate Step 2 packed hardcalls");
    replace_buffer(reinterpret_cast<void**>(&d_means_), &mean_capacity_,
      static_cast<size_t>(variants) * sizeof(double),
      "allocate Step 2 missing means");
    replace_buffer(reinterpret_cast<void**>(&d_flipped_), &flip_capacity_,
      variants, "allocate Step 2 flip flags");
    replace_buffer(reinterpret_cast<void**>(&d_sparse_), &sparse_capacity_,
      variants, "allocate Step 2 sparse flags");
    replace_buffer(reinterpret_cast<void**>(&d_numerators_),
      &numerator_capacity_, output_count * sizeof(double),
      "allocate Step 2 score numerators");
    replace_buffer(reinterpret_cast<void**>(&d_denominators_),
      &denominator_capacity_, output_count * sizeof(double),
      "allocate Step 2 score denominators");
    if(trait_counts) {
      replace_buffer(reinterpret_cast<void**>(&d_observed_allele_sums_),
        &observed_allele_sum_capacity_, output_count * sizeof(double),
        "allocate Step 2 observed allele sums");
      replace_buffer(
        reinterpret_cast<void**>(&d_observed_nonmissing_counts_),
        &observed_nonmissing_count_capacity_, output_count * sizeof(double),
        "allocate Step 2 observed nonmissing counts");
    }
  }

  void release(void* pointer) {
    if(pointer) cudaFree(pointer);
  }

  void release_all() {
    release(d_residuals_);
    release(d_weights_);
    release(d_designs_);
    release(d_projections_);
    release(d_small_);
    release(d_grams_);
    release(d_variances_);
    release(d_active_);
    release(d_observed_);
    release(d_packed_);
    release(d_means_);
    release(d_flipped_);
    release(d_sparse_);
    release(d_numerators_);
    release(d_denominators_);
    release(d_observed_allele_sums_);
    release(d_observed_nonmissing_counts_);
  }

  int device_ = 0;
  bool automatic_ = false;
  bool attempted_ = false;
  cudaDeviceProp properties_{};
  ScoreMode mode_ = ScoreMode::none;
  int samples_ = 0;
  int phenotypes_ = 0;
  int covariates_ = 0;
  bool quantitative_complete_masks_ = true;
  bool trait_counts_required_ = false;

  std::vector<unsigned char> active_host_;
  std::vector<unsigned char> observed_host_;
  std::vector<unsigned char> packed_host_;
  std::vector<double> residuals_host_;
  std::vector<double> designs_host_;
  std::vector<double> projections_host_;
  std::vector<double> small_host_;
  std::vector<double> grams_host_;
  std::vector<double> variances_host_;

  double* d_residuals_ = nullptr;
  double* d_weights_ = nullptr;
  double* d_designs_ = nullptr;
  double* d_projections_ = nullptr;
  double* d_small_ = nullptr;
  double* d_grams_ = nullptr;
  double* d_variances_ = nullptr;
  unsigned char* d_active_ = nullptr;
  unsigned char* d_observed_ = nullptr;
  unsigned char* d_packed_ = nullptr;
  double* d_means_ = nullptr;
  unsigned char* d_flipped_ = nullptr;
  unsigned char* d_sparse_ = nullptr;
  double* d_numerators_ = nullptr;
  double* d_denominators_ = nullptr;
  double* d_observed_allele_sums_ = nullptr;
  double* d_observed_nonmissing_counts_ = nullptr;

  size_t residual_capacity_ = 0;
  size_t weight_capacity_ = 0;
  size_t design_capacity_ = 0;
  size_t projection_capacity_ = 0;
  size_t small_capacity_ = 0;
  size_t gram_capacity_ = 0;
  size_t variance_capacity_ = 0;
  size_t active_capacity_ = 0;
  size_t observed_capacity_ = 0;
  size_t packed_capacity_ = 0;
  size_t mean_capacity_ = 0;
  size_t flip_capacity_ = 0;
  size_t sparse_capacity_ = 0;
  size_t numerator_capacity_ = 0;
  size_t denominator_capacity_ = 0;
  size_t observed_allele_sum_capacity_ = 0;
  size_t observed_nonmissing_count_capacity_ = 0;
};

}  // namespace

bool cuda_step2_compute_backend_available(int device, std::string& reason) {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if(status != cudaSuccess) {
    reason = cudaGetErrorString(status);
    return false;
  }
  if(device < 0 || device >= count) {
    std::ostringstream out;
    out << "device " << device << " is outside the available range [0, "
        << count << ")";
    reason = out.str();
    return false;
  }
  return true;
}

std::unique_ptr<Step2ComputeBackend> make_cuda_step2_compute_backend(
    int device, bool automatic) {
  return std::unique_ptr<Step2ComputeBackend>(
    new CudaStep2ComputeBackend(device, automatic));
}
