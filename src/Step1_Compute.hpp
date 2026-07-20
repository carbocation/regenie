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

#ifndef STEP1_COMPUTE_H
#define STEP1_COMPUTE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>

enum class Step1GramMode {
  full_product,
  selfadjoint_rank_update
};

struct Step1ComputeTimings {
  double upload_ms = 0;
  double preprocess_ms = 0;
  double crossproduct_ms = 0;
  double gram_ms = 0;
  double eigensolve_ms = 0;
  double transform_ms = 0;
  double ridge_ms = 0;
  double download_ms = 0;
  uint64_t resident_reuse_count = 0;
  uint64_t pinned_staging_upload_count = 0;
  uint64_t pinned_staging_upload_bytes = 0;
  uint64_t packed_hardcall_upload_count = 0;
  uint64_t packed_hardcall_upload_bytes = 0;
  double packed_hardcall_expand_ms = 0;
  double packed_hardcall_validation_ms = 0;
  double packed_hardcall_allocation_ms = 0;
  double packed_hardcall_host_prepare_ms = 0;
  double packed_hardcall_backend_wall_ms = 0;
  uint64_t design_upload_count = 0;
  uint64_t design_upload_bytes = 0;
  uint64_t resident_design_upload_count = 0;
  uint64_t resident_design_upload_bytes = 0;
  uint64_t resident_design_reuse_count = 0;
  double host_materialization_ms = 0;
};

class Step1ComputeBackend {

  public:
    virtual ~Step1ComputeBackend() {}

    virtual const char* name() const = 0;
    virtual std::string description() const = 0;

    virtual bool preprocess_genotypes(
      Eigen::MatrixXd& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& covariates,
      const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
      double degrees_of_freedom,
      double minimum_scale,
      const Eigen::Ref<const Eigen::VectorXd>& row_multipliers,
      bool copy_to_host,
      Eigen::VectorXd& row_scales,
      Step1ComputeTimings* timings = nullptr);

    virtual bool can_preprocess_packed_hardcalls(
      Eigen::Index variants,
      Eigen::Index samples) const;

    virtual bool preprocess_packed_hardcalls(
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
      Step1ComputeTimings* timings = nullptr);

    virtual void compute_preprocessed_products(
      Eigen::Index start_column,
      Eigen::Index column_count,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1GramMode mode,
      Step1ComputeTimings* timings = nullptr);

    virtual void ridge_predict_preprocessed(
      Eigen::Index start_column,
      Eigen::Index column_count,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings = nullptr);

    virtual bool ridge_predict_preprocessed_system(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::Index start_column,
      Eigen::Index column_count,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings = nullptr);

    virtual bool ridge_predict_preprocessed_systems(
      const std::vector<Eigen::MatrixXd>& grams,
      const std::vector<Eigen::MatrixXd>& right_hand_sides,
      const Eigen::Ref<const Eigen::VectorXi>& start_columns,
      const Eigen::Ref<const Eigen::VectorXi>& column_counts,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      std::vector<Eigen::MatrixXd>& predictions,
      std::vector<Eigen::MatrixXd>& coefficients,
      Step1ComputeTimings* timings = nullptr);

    virtual void release_preprocessed_genotypes();

    virtual void compute_products(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1GramMode mode,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void eigendecompose_and_transform(
      const Eigen::Ref<const Eigen::MatrixXd>& symmetric_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::MatrixXd& eigenvectors,
      Eigen::MatrixXd& eigenvalues,
      Eigen::MatrixXd& transformed_right_hand_sides,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void compute_design_products(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void compute_design_crossproduct(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings = nullptr);

    virtual void compute_weighted_design_products(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& weights,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual bool cache_design_partitions(
      const std::vector<Eigen::MatrixXd>& partitions,
      Step1ComputeTimings* timings = nullptr);

    virtual bool cache_design_matrix(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      Step1ComputeTimings* timings = nullptr);

    virtual void predict_cached_design(
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      Eigen::VectorXd& predictions,
      Step1ComputeTimings* timings = nullptr);

    virtual void compute_cached_weighted_design_products(
      const Eigen::Ref<const Eigen::VectorXd>& weights,
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings = nullptr);

    virtual void compute_cached_design_crossproduct(
      const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
      Eigen::MatrixXd& crossproduct,
      Step1ComputeTimings* timings = nullptr);

    virtual void release_cached_design();

    virtual void ridge_predict(
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
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void factorize_ridge_system(
      const Eigen::Ref<const Eigen::MatrixXd>& symmetric_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void compute_products_and_factorize_ridge(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Step1GramMode mode,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void ridge_predict_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& prediction_matrix,
      bool samples_in_columns,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::MatrixXd>& leave_one_out_outcomes,
      bool leave_one_out,
      Eigen::MatrixXd& predictions,
      Eigen::MatrixXd& coefficients,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void diagonal_penalty_predict(
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
      Step1ComputeTimings* timings = nullptr);

    virtual void diagonal_penalty_solve(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings = nullptr);

    virtual void factorize_diagonal_penalty(
      const Eigen::Ref<const Eigen::MatrixXd>& gram,
      double ridge_parameter,
      const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void solve_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings = nullptr) = 0;

    virtual void grouped_leave_one_out_predict_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      const Eigen::Ref<const Eigen::VectorXd>& residuals,
      const Eigen::Ref<const Eigen::VectorXd>& leverage_weights,
      const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
      const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
      Eigen::MatrixXd& predictions,
      Step1ComputeTimings* timings = nullptr);

    virtual void grouped_predict(
      const Eigen::Ref<const Eigen::MatrixXd>& design,
      const Eigen::Ref<const Eigen::VectorXd>& coefficients,
      const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
      const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
      Eigen::MatrixXd& predictions,
      Step1ComputeTimings* timings = nullptr);

  protected:
    static void validate_packed_hardcall_preprocessing_inputs(
      const unsigned char* packed_hardcalls,
      size_t packed_bytes,
      size_t packed_stride_bytes,
      Eigen::Index variants,
      Eigen::Index samples,
      const Eigen::Ref<const Eigen::MatrixXd>& covariates,
      const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
      double degrees_of_freedom,
      double minimum_scale);
};

std::unique_ptr<Step1ComputeBackend> make_cpu_step1_compute_backend();
std::unique_ptr<Step1ComputeBackend> make_step1_compute_backend(
  const std::string& requested_backend,
  int device);
bool cuda_step1_compute_backend_compiled();

#endif
