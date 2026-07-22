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

#include "Step2_Compute.hpp"

#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#ifdef WITH_CUDA
std::unique_ptr<Step2ComputeBackend> make_cuda_step2_compute_backend(
  int device, bool automatic);
bool cuda_step2_compute_backend_available(int device, std::string& reason);
#endif

bool should_use_cpu_quantitative_block_scoring(
    Eigen::Index samples, Eigen::Index phenotypes, bool complete_masks) {
  if(samples <= 0 || phenotypes <= 0) return false;

  // Keep a diagnostic override so crossover measurements can compare both
  // paths without maintaining benchmark-only binaries.
  const char* value =
    std::getenv("REGENIE_STEP2_QT_BLOCK_MIN_PHENOTYPES");
  if(value && *value) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if(end != value && *end == '\0' && parsed >= 1 &&
       parsed <= std::numeric_limits<int>::max())
      return phenotypes >= static_cast<Eigen::Index>(parsed);
  }

  // With phenotype-specific missingness, the per-variant implementation
  // repeatedly rebuilds masked genotype projections; even a two-trait panel
  // amortizes the blockwise crossproducts. Complete traits have a cheaper
  // fallback, so retain it for narrow or modest-sized panels.
  if(!complete_masks) return phenotypes >= 2;
  if(phenotypes >= 12) return true;
  if(phenotypes < 4) return false;

  const Eigen::Index minimum_samples =
    (2000000 + phenotypes - 1) / phenotypes;
  return samples >= minimum_samples;
}

namespace {

enum class CpuScoreMode {
  none,
  quantitative_complete,
  quantitative_missing,
  binary,
  cox
};

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

class CpuStep2ComputeBackend : public Step2ComputeBackend {
 public:
  const char* name() const override { return "cpu"; }
  std::string description() const override {
    return "blockwise host Step 2 scoring";
  }
  bool ready() const override { return mode_ != CpuScoreMode::none; }
  bool uses_packed_hardcalls() const override { return false; }
  bool provides_observed_trait_counts() const override { return false; }
  void clear() override {
    mode_ = CpuScoreMode::none;
    samples_ = phenotypes_ = covariates_ = 0;
    linear_terms_.resize(0, 0);
    square_terms_.resize(0, 0);
    small_.resize(0, 0);
    grams_.clear();
    projection_transforms_.clear();
    cox_factored_projection_ = false;
    variances_.resize(0);
    active_.clear();
  }

  bool prepare_quantitative(
      const Eigen::Ref<const Eigen::MatrixXd>& residuals,
      const Eigen::Ref<const Eigen::MatrixXd>& covariates,
      const Eigen::Ref<const Eigen::MatrixXd>& outcome_covariate_products,
      const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
        Eigen::Dynamic>>& observed_masks,
      bool complete_masks, Step2ComputeTimings* timings) override {
    clear();
    if(!should_use_cpu_quantitative_block_scoring(
         residuals.rows(), residuals.cols(), complete_masks) ||
       covariates.rows() != residuals.rows() || covariates.cols() <= 0 ||
       outcome_covariate_products.rows() != residuals.cols() ||
       outcome_covariate_products.cols() != covariates.cols() ||
       observed_masks.rows() != residuals.rows() ||
       observed_masks.cols() != residuals.cols())
      return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    samples_ = residuals.rows();
    phenotypes_ = residuals.cols();
    covariates_ = covariates.cols();
    small_ = outcome_covariate_products;

    if(complete_masks) {
      linear_terms_.resize(samples_, phenotypes_ + covariates_);
      linear_terms_.leftCols(phenotypes_) = residuals;
      linear_terms_.rightCols(covariates_) = covariates;
      mode_ = CpuScoreMode::quantitative_complete;
    } else {
      const Eigen::Index observed_term_count = phenotypes_ * covariates_;
      linear_terms_.resize(samples_, phenotypes_ + covariates_ +
        observed_term_count);
      linear_terms_.leftCols(phenotypes_) = residuals;
      linear_terms_.middleCols(phenotypes_, covariates_) = covariates;
      square_terms_.resize(samples_, phenotypes_);
      grams_.resize(phenotypes_);
      const Eigen::MatrixXd complete_gram =
        covariates.transpose() * covariates;
      for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
          ++phenotype) {
        const Eigen::ArrayXd observed =
          observed_masks.col(phenotype).cast<double>().array();
        square_terms_.col(phenotype) = observed.matrix();
        for(Eigen::Index covariate = 0; covariate < covariates_;
            ++covariate) {
          const Eigen::Index column = phenotypes_ + covariates_ +
            phenotype * covariates_ + covariate;
          linear_terms_.col(column) =
            (covariates.col(covariate).array() * observed).matrix();
        }
        grams_[phenotype] = complete_gram;
        for(Eigen::Index sample = 0; sample < samples_; ++sample) {
          if(observed_masks(sample, phenotype)) continue;
          grams_[phenotype].noalias() -=
            covariates.row(sample).transpose() * covariates.row(sample);
        }
      }
      mode_ = CpuScoreMode::quantitative_missing;
    }

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
    if(residuals.rows() <= 0 || residuals.cols() < 4 ||
       weights.rows() != residuals.rows() ||
       weights.cols() != residuals.cols() ||
       designs.size() != static_cast<size_t>(residuals.cols()) ||
       design_residual_products.size() != designs.size() ||
       observed_masks.rows() != residuals.rows() ||
       observed_masks.cols() != residuals.cols() ||
       active_phenotypes.size() != residuals.cols())
      return false;

    Eigen::Index covariate_count = 0;
    for(Eigen::Index phenotype = 0; phenotype < residuals.cols();
        ++phenotype) {
      if(!active_phenotypes(phenotype)) continue;
      covariate_count = designs[phenotype].cols();
      break;
    }
    if(covariate_count <= 0) return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    samples_ = residuals.rows();
    phenotypes_ = residuals.cols();
    covariates_ = covariate_count;
    const Eigen::Index terms_per_phenotype = covariates_ + 1;
    linear_terms_ = Eigen::MatrixXd::Zero(samples_,
      phenotypes_ * terms_per_phenotype);
    square_terms_ = Eigen::MatrixXd::Zero(samples_, phenotypes_);
    small_ = Eigen::MatrixXd::Zero(phenotypes_, covariates_);
    active_.resize(phenotypes_, 0);
    for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
        ++phenotype) {
      active_[phenotype] = active_phenotypes(phenotype) ? 1 : 0;
      if(!active_[phenotype]) continue;
      if(designs[phenotype].rows() != samples_ ||
         designs[phenotype].cols() != covariates_ ||
         design_residual_products[phenotype].size() != covariates_)
        return false;
      const Eigen::Index base = phenotype * terms_per_phenotype;
      linear_terms_.col(base) = (weights.col(phenotype).array() *
        residuals.col(phenotype).array()).matrix();
      square_terms_.col(phenotype) =
        weights.col(phenotype).array().square().matrix();
      for(Eigen::Index covariate = 0; covariate < covariates_;
          ++covariate) {
        linear_terms_.col(base + 1 + covariate) =
          (weights.col(phenotype).array() *
            designs[phenotype].col(covariate).array()).matrix();
        small_(phenotype, covariate) =
          design_residual_products[phenotype](covariate);
      }
    }
    mode_ = CpuScoreMode::binary;
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
    clear();
    const Eigen::Index phenotype_count = score_residuals.size();
    if(phenotype_count < 4 ||
       weighted_designs.size() != score_residuals.size() ||
       projections.size() != score_residuals.size() ||
       (!projection_transforms.empty() &&
        projection_transforms.size() != score_residuals.size()) ||
       projection_scores.size() != score_residuals.size() ||
       projection_grams.size() != score_residuals.size() ||
       residual_variances.size() != phenotype_count ||
       observed_masks.cols() != phenotype_count ||
       active_phenotypes.size() != phenotype_count)
      return false;

    Eigen::Index sample_count = 0;
    Eigen::Index covariate_count = 0;
    for(Eigen::Index phenotype = 0; phenotype < phenotype_count;
        ++phenotype) {
      if(!active_phenotypes(phenotype)) continue;
      sample_count = score_residuals[phenotype].size();
      covariate_count = weighted_designs[phenotype].cols();
      break;
    }
    if(sample_count <= 0 || covariate_count <= 0 ||
       observed_masks.rows() != sample_count)
      return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    samples_ = sample_count;
    phenotypes_ = phenotype_count;
    covariates_ = covariate_count;
    cox_factored_projection_ = common_projection_design.size() > 0 ||
      !projection_transforms.empty();
    if(cox_factored_projection_ &&
       (common_projection_design.rows() != samples_ ||
        common_projection_design.cols() != covariates_ ||
        projection_transforms.size() !=
          static_cast<size_t>(phenotypes_)))
      return false;
    const Eigen::Index terms_per_phenotype = 1 + covariates_ +
      (cox_factored_projection_ ? 0 : covariates_);
    linear_terms_ = Eigen::MatrixXd::Zero(samples_,
      phenotypes_ * terms_per_phenotype +
      (cox_factored_projection_ ? covariates_ : 0));
    if(cox_factored_projection_)
      linear_terms_.rightCols(covariates_) = common_projection_design;
    small_ = Eigen::MatrixXd::Zero(phenotypes_, covariates_);
    grams_.resize(phenotypes_);
    projection_transforms_.resize(phenotypes_);
    variances_ = residual_variances;
    active_.resize(phenotypes_, 0);
    for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
        ++phenotype) {
      active_[phenotype] = active_phenotypes(phenotype) ? 1 : 0;
      grams_[phenotype] = Eigen::MatrixXd::Zero(covariates_, covariates_);
      if(!active_[phenotype]) continue;
      if(score_residuals[phenotype].size() != samples_ ||
         weighted_designs[phenotype].rows() != samples_ ||
         weighted_designs[phenotype].cols() != covariates_ ||
         projections[phenotype].rows() != samples_ ||
         projections[phenotype].cols() != covariates_ ||
         projection_scores[phenotype].size() != covariates_ ||
         projection_grams[phenotype].rows() != covariates_ ||
         projection_grams[phenotype].cols() != covariates_ ||
         (cox_factored_projection_ &&
          (projection_transforms[phenotype].rows() != covariates_ ||
           projection_transforms[phenotype].cols() != covariates_)))
        return false;
      const Eigen::Index base = phenotype * terms_per_phenotype;
      linear_terms_.col(base) = score_residuals[phenotype];
      linear_terms_.middleCols(base + 1, covariates_) =
        weighted_designs[phenotype];
      if(cox_factored_projection_)
        projection_transforms_[phenotype] =
          projection_transforms[phenotype];
      else
        linear_terms_.middleCols(base + 1 + covariates_, covariates_) =
          projections[phenotype];
      small_.row(phenotype) = projection_scores[phenotype].transpose();
      grams_[phenotype] = projection_grams[phenotype];
    }
    mode_ = CpuScoreMode::cox;
    if(timings) {
      timings->prepared_chromosomes++;
      timings->prepare_upload_ms += elapsed_ms(start);
    }
    return true;
  }

  bool score_packed_block(
      const std::vector<std::vector<unsigned char>>& ,
      const std::vector<double>&,
      const std::vector<unsigned char>&,
      const std::vector<unsigned char>&,
      Eigen::Index, Eigen::MatrixXd&, Eigen::MatrixXd&,
      Eigen::MatrixXd&, Eigen::MatrixXd&,
      Step2ComputeTimings*) override {
    return false;
  }

  bool score_dense_block(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const std::vector<unsigned char>& sparse,
      const Eigen::RowVectorXd* supplied_raw_squared_norms,
      Eigen::MatrixXd& numerators,
      Eigen::MatrixXd& denominators,
      Step2ComputeTimings* timings) override {
    if(!ready() || genotypes.rows() != samples_ ||
       genotypes.cols() <= 0 ||
       sparse.size() != static_cast<size_t>(genotypes.cols()))
      return false;

    const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
    const Eigen::Index variants = genotypes.cols();
    std::chrono::steady_clock::time_point phase_start =
      std::chrono::steady_clock::now();
    linear_crossproducts_.noalias() =
      linear_terms_.transpose() * genotypes;
    const double linear_crossproduct_ms = elapsed_ms(phase_start);
    phase_start = std::chrono::steady_clock::now();
    double square_materialization_ms = 0;
    double square_crossproduct_ms = 0;
    if(square_terms_.cols() > 0) {
      squared_genotypes_.resize(genotypes.rows(), variants);
      // Eigen's coefficient-wise assignment is single-threaded here. Each
      // production block is several GiB, so partition independent variant
      // columns across the existing OpenMP team before the second GEMM.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for(Eigen::Index variant = 0; variant < variants; ++variant)
        squared_genotypes_.col(variant) =
          genotypes.col(variant).array().square().matrix();
      square_materialization_ms = elapsed_ms(phase_start);
      phase_start = std::chrono::steady_clock::now();
      square_crossproducts_.noalias() =
        square_terms_.transpose() * squared_genotypes_;
      square_crossproduct_ms = elapsed_ms(phase_start);
    } else {
      const bool use_supplied_raw_squared_norms =
        mode_ == CpuScoreMode::cox && supplied_raw_squared_norms &&
        supplied_raw_squared_norms->size() == variants &&
        supplied_raw_squared_norms->allFinite() &&
        (supplied_raw_squared_norms->array() >= 0).all();
      if(use_supplied_raw_squared_norms)
        raw_squared_norms_ = *supplied_raw_squared_norms;
      else
        raw_squared_norms_ = genotypes.colwise().squaredNorm();
      square_materialization_ms = elapsed_ms(phase_start);
      square_crossproducts_.resize(0, 0);
    }

    phase_start = std::chrono::steady_clock::now();
    numerators.resize(phenotypes_, variants);
    denominators.resize(phenotypes_, variants);
    if(mode_ == CpuScoreMode::quantitative_complete)
      finish_quantitative_complete(variants, numerators, denominators);
    else if(mode_ == CpuScoreMode::quantitative_missing)
      finish_quantitative_missing(variants, sparse, numerators,
        denominators);
    else if(mode_ == CpuScoreMode::binary)
      finish_binary(variants, sparse, numerators, denominators);
    else
      finish_cox(variants, numerators, denominators);
    const double finalize_ms = elapsed_ms(phase_start);

    const double wall_ms = elapsed_ms(start);
    if(timings) {
      timings->scored_blocks++;
      timings->scored_variants += variants;
      timings->kernel_ms += wall_ms;
      timings->wall_ms += wall_ms;
      timings->linear_crossproduct_ms += linear_crossproduct_ms;
      timings->square_materialization_ms += square_materialization_ms;
      timings->square_crossproduct_ms += square_crossproduct_ms;
      timings->finalize_ms += finalize_ms;
    }
    return numerators.allFinite() && denominators.allFinite();
  }

 private:
  void finish_quantitative_complete(Eigen::Index variants,
      Eigen::MatrixXd& numerators, Eigen::MatrixXd& denominators) const {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(Eigen::Index variant = 0; variant < variants; ++variant) {
      double denominator = raw_squared_norms_(variant);
      for(Eigen::Index covariate = 0; covariate < covariates_;
          ++covariate) {
        const double crossproduct =
          linear_crossproducts_(phenotypes_ + covariate, variant);
        denominator -= crossproduct * crossproduct;
      }
      for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
          ++phenotype) {
        double numerator = linear_crossproducts_(phenotype, variant);
        for(Eigen::Index covariate = 0; covariate < covariates_;
            ++covariate)
          numerator -= linear_crossproducts_(phenotypes_ + covariate,
            variant) * small_(phenotype, covariate);
        numerators(phenotype, variant) = numerator;
        denominators(phenotype, variant) = denominator;
      }
    }
  }

  void finish_quantitative_missing(Eigen::Index variants,
      const std::vector<unsigned char>& sparse,
      Eigen::MatrixXd& numerators, Eigen::MatrixXd& denominators) const {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(Eigen::Index variant = 0; variant < variants; ++variant) {
      for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
          ++phenotype) {
        double numerator = linear_crossproducts_(phenotype, variant);
        double denominator = square_crossproducts_(phenotype, variant);
        const Eigen::Index observed_base = phenotypes_ + covariates_ +
          phenotype * covariates_;
        for(Eigen::Index row = 0; row < covariates_; ++row) {
          const double complete_cross =
            linear_crossproducts_(phenotypes_ + row, variant);
          const double observed_cross =
            linear_crossproducts_(observed_base + row, variant);
          numerator -= complete_cross * small_(phenotype, row);
          denominator -= 2 * complete_cross * observed_cross;
          if(sparse[variant])
            denominator += complete_cross * complete_cross;
        }
        if(!sparse[variant]) {
          for(Eigen::Index row = 0; row < covariates_; ++row) {
            const double row_cross =
              linear_crossproducts_(phenotypes_ + row, variant);
            for(Eigen::Index column = 0; column < covariates_; ++column)
              denominator += row_cross * grams_[phenotype](row, column) *
                linear_crossproducts_(phenotypes_ + column, variant);
          }
        }
        numerators(phenotype, variant) = numerator;
        denominators(phenotype, variant) = denominator;
      }
    }
  }

  void finish_binary(Eigen::Index variants,
      const std::vector<unsigned char>& sparse,
      Eigen::MatrixXd& numerators, Eigen::MatrixXd& denominators) const {
    const Eigen::Index terms_per_phenotype = covariates_ + 1;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
        ++phenotype) {
      if(!active_[phenotype]) {
        numerators.row(phenotype).setZero();
        denominators.row(phenotype).setOnes();
        continue;
      }
      const Eigen::Index base = phenotype * terms_per_phenotype;
      for(Eigen::Index variant = 0; variant < variants; ++variant) {
        double numerator = linear_crossproducts_(base, variant);
        double denominator = square_crossproducts_(phenotype, variant);
        for(Eigen::Index covariate = 0; covariate < covariates_;
            ++covariate) {
          const double crossproduct =
            linear_crossproducts_(base + 1 + covariate, variant);
          if(!sparse[variant])
            numerator -= crossproduct * small_(phenotype, covariate);
          denominator -= crossproduct * crossproduct;
        }
        numerators(phenotype, variant) = numerator;
        denominators(phenotype, variant) = denominator;
      }
    }
  }

  void finish_cox(Eigen::Index variants, Eigen::MatrixXd& numerators,
      Eigen::MatrixXd& denominators) const {
    const Eigen::Index terms_per_phenotype = 1 + covariates_ +
      (cox_factored_projection_ ? 0 : covariates_);
    const Eigen::Index common_base =
      phenotypes_ * terms_per_phenotype;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(Eigen::Index phenotype = 0; phenotype < phenotypes_;
        ++phenotype) {
      if(!active_[phenotype]) {
        numerators.row(phenotype).setZero();
        denominators.row(phenotype).setOnes();
        continue;
      }
      const Eigen::Index base = phenotype * terms_per_phenotype;
      for(Eigen::Index variant = 0; variant < variants; ++variant) {
        double numerator = linear_crossproducts_(base, variant);
        double denominator = raw_squared_norms_(variant);
        for(Eigen::Index covariate = 0; covariate < covariates_;
            ++covariate) {
          const double coefficient =
            linear_crossproducts_(base + 1 + covariate, variant);
          double raw_cross = 0;
          if(cox_factored_projection_) {
            for(Eigen::Index column = 0; column < covariates_; ++column)
              raw_cross += projection_transforms_[phenotype](
                column, covariate) * linear_crossproducts_(
                  common_base + column, variant);
          } else {
            raw_cross = linear_crossproducts_(
              base + 1 + covariates_ + covariate, variant);
          }
          numerator -= coefficient * small_(phenotype, covariate);
          denominator -= 2 * coefficient * raw_cross;
        }
        for(Eigen::Index row = 0; row < covariates_; ++row) {
          const double row_coefficient =
            linear_crossproducts_(base + 1 + row, variant);
          for(Eigen::Index column = 0; column < covariates_; ++column)
            denominator += row_coefficient *
              grams_[phenotype](row, column) *
              linear_crossproducts_(base + 1 + column, variant);
        }
        numerators(phenotype, variant) = numerator;
        denominators(phenotype, variant) =
          variances_(phenotype) * denominator;
      }
    }
  }

  CpuScoreMode mode_ = CpuScoreMode::none;
  Eigen::Index samples_ = 0;
  Eigen::Index phenotypes_ = 0;
  Eigen::Index covariates_ = 0;
  Eigen::MatrixXd linear_terms_;
  Eigen::MatrixXd square_terms_;
  Eigen::MatrixXd small_;
  std::vector<Eigen::MatrixXd> grams_;
  std::vector<Eigen::MatrixXd> projection_transforms_;
  Eigen::VectorXd variances_;
  std::vector<unsigned char> active_;
  bool cox_factored_projection_ = false;
  Eigen::MatrixXd linear_crossproducts_;
  Eigen::MatrixXd squared_genotypes_;
  Eigen::MatrixXd square_crossproducts_;
  Eigen::RowVectorXd raw_squared_norms_;
};

}  // namespace

std::unique_ptr<Step2ComputeBackend> make_step2_compute_backend(
    const std::string& requested_backend, int device) {
  if(requested_backend == "cpu")
    return std::unique_ptr<Step2ComputeBackend>(
      new CpuStep2ComputeBackend());

  if(requested_backend != "cuda" && requested_backend != "auto")
    throw std::runtime_error("unknown Step 2 compute backend '" +
      requested_backend + "' (expected cpu, cuda, or auto)");

#ifdef WITH_CUDA
  std::string reason;
  if(cuda_step2_compute_backend_available(device, reason))
    return make_cuda_step2_compute_backend(device,
      requested_backend == "auto");
  if(requested_backend == "cuda")
    throw std::runtime_error("CUDA Step 2 backend is unavailable: " + reason);
#else
  (void)device;
  if(requested_backend == "cuda")
    throw std::runtime_error(
      "CUDA Step 2 backend was requested, but this binary was built without REGENIE_WITH_CUDA");
#endif

  return std::unique_ptr<Step2ComputeBackend>(new CpuStep2ComputeBackend());
}
