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

#include <memory>
#include <string>
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
      Eigen::VectorXd& row_scales,
      Step1ComputeTimings* timings = nullptr);

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
};

std::unique_ptr<Step1ComputeBackend> make_cpu_step1_compute_backend();
std::unique_ptr<Step1ComputeBackend> make_step1_compute_backend(
  const std::string& requested_backend,
  int device);
bool cuda_step1_compute_backend_compiled();

#endif
