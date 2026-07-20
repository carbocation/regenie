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
};

class Step2ComputeBackend {
 public:
  virtual ~Step2ComputeBackend() {}

  virtual const char* name() const = 0;
  virtual std::string description() const = 0;
  virtual bool ready() const = 0;
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
};

std::unique_ptr<Step2ComputeBackend> make_step2_compute_backend(
  const std::string& requested_backend, int device);

#endif
