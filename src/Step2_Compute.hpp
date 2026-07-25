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

#ifndef STEP2_COMPUTE_H
#define STEP2_COMPUTE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

struct Step2ComputeTimings {
  uint64_t prepared_chromosomes = 0;
  uint64_t scored_blocks = 0;
  uint64_t scored_variants = 0;
  uint64_t packed_upload_bytes = 0;
  double prepare_upload_ms = 0;
  double host_pack_ms = 0;
  double upload_ms = 0;
  double kernel_ms = 0;
  double download_ms = 0;
  double wall_ms = 0;
  double linear_crossproduct_ms = 0;
  double square_materialization_ms = 0;
  double square_crossproduct_ms = 0;
  double finalize_ms = 0;
};

class Step2ScoreBatch {
 public:
  void begin_write() {
    scores_valid_ = false;
    trait_counts_valid_ = false;
  }

  void reset() {
    begin_write();
  }

  bool publish(bool scored, Eigen::Index phenotypes,
      Eigen::Index variants) {
    scores_valid_ = scored &&
      numerators_.rows() == phenotypes &&
      numerators_.cols() == variants &&
      denominators_.rows() == phenotypes &&
      denominators_.cols() == variants;
    trait_counts_valid_ = scores_valid_ &&
      observed_allele_sums_.rows() == phenotypes &&
      observed_allele_sums_.cols() == variants &&
      observed_nonmissing_counts_.rows() == phenotypes &&
      observed_nonmissing_counts_.cols() == variants;
    return scores_valid_;
  }

  bool valid() const {
    return scores_valid_;
  }

  bool has_variant(Eigen::Index phenotypes, Eigen::Index variant) const {
    return scores_valid_ &&
      numerators_.rows() == phenotypes &&
      denominators_.rows() == phenotypes &&
      numerators_.cols() == denominators_.cols() &&
      variant >= 0 &&
      variant < numerators_.cols() &&
      variant < denominators_.cols();
  }

  bool has_score(Eigen::Index phenotype, Eigen::Index variant) const {
    return scores_valid_ &&
      phenotype >= 0 && variant >= 0 &&
      numerators_.rows() == denominators_.rows() &&
      numerators_.cols() == denominators_.cols() &&
      phenotype < numerators_.rows() &&
      phenotype < denominators_.rows() &&
      variant < numerators_.cols() &&
      variant < denominators_.cols();
  }

  bool trait_counts_valid() const {
    return trait_counts_valid_;
  }

  Eigen::MatrixXd& numerator_output() {
    begin_write();
    return numerators_;
  }
  Eigen::MatrixXd& denominator_output() {
    begin_write();
    return denominators_;
  }
  Eigen::MatrixXd& observed_allele_sum_output() {
    begin_write();
    return observed_allele_sums_;
  }
  Eigen::MatrixXd& observed_nonmissing_count_output() {
    begin_write();
    return observed_nonmissing_counts_;
  }

  const Eigen::MatrixXd& numerators() const { return numerators_; }
  const Eigen::MatrixXd& denominators() const { return denominators_; }
  const Eigen::MatrixXd& observed_allele_sums() const {
    return observed_allele_sums_;
  }
  const Eigen::MatrixXd& observed_nonmissing_counts() const {
    return observed_nonmissing_counts_;
  }

 private:
  Eigen::MatrixXd numerators_;
  Eigen::MatrixXd denominators_;
  Eigen::MatrixXd observed_allele_sums_;
  Eigen::MatrixXd observed_nonmissing_counts_;
  bool scores_valid_ = false;
  bool trait_counts_valid_ = false;
};

class Step2ComputeBackend {
 public:
  virtual ~Step2ComputeBackend() {}

  virtual const char* name() const = 0;
  virtual std::string description() const = 0;
  virtual bool ready() const = 0;
  virtual bool uses_packed_hardcalls() const = 0;
  virtual bool prefers_loco_prediction_prefetch() const = 0;
  virtual bool supports_packed_block_pipeline() const = 0;
  virtual bool provides_observed_trait_counts() const = 0;
  virtual void clear() = 0;

  virtual bool prepare_quantitative(
    const Eigen::Ref<const Eigen::MatrixXd>& residuals,
    const Eigen::Ref<const Eigen::MatrixXd>& covariates,
    const Eigen::Ref<const Eigen::MatrixXd>& outcome_covariate_products,
    const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
      Eigen::Dynamic>>& observed_masks,
    bool complete_masks,
    Step2ComputeTimings* timings = nullptr) = 0;

  virtual bool prepare_binary(
    const Eigen::Ref<const Eigen::MatrixXd>& residuals,
    const Eigen::Ref<const Eigen::MatrixXd>& weights,
    const std::vector<Eigen::MatrixXd>& designs,
    const std::vector<Eigen::VectorXd>& design_residual_products,
    const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
      Eigen::Dynamic>>& observed_masks,
    const Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, 1>>&
      active_phenotypes,
    Step2ComputeTimings* timings = nullptr) = 0;

  virtual bool prepare_cox(
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
    Step2ComputeTimings* timings = nullptr) = 0;

  virtual bool score_packed_block(
    const std::vector<std::vector<unsigned char>>& packed_hardcalls,
    const std::vector<double>& missing_means,
    const std::vector<unsigned char>& flipped,
    const std::vector<unsigned char>& sparse,
    Eigen::Index samples,
    Eigen::MatrixXd& numerators,
    Eigen::MatrixXd& denominators,
    Eigen::MatrixXd& observed_allele_sums,
    Eigen::MatrixXd& observed_nonmissing_counts,
    Step2ComputeTimings* timings = nullptr) = 0;

  virtual bool score_dense_block(
    const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
    const std::vector<unsigned char>& sparse,
    const Eigen::RowVectorXd* raw_squared_norms,
    Eigen::MatrixXd& numerators,
    Eigen::MatrixXd& denominators,
    Step2ComputeTimings* timings = nullptr) = 0;
};

bool should_use_cpu_quantitative_block_scoring(
  Eigen::Index samples, Eigen::Index phenotypes, bool complete_masks);

std::unique_ptr<Step2ComputeBackend> make_step2_compute_backend(
  const std::string& requested_backend, int device);

#endif
