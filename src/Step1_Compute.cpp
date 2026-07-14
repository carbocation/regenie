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

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef WITH_CUDA
std::unique_ptr<Step1ComputeBackend> make_cuda_step1_compute_backend(int device);
bool cuda_step1_compute_backend_available(int device, std::string& reason);
#endif

namespace {

using ComputeClock = std::chrono::steady_clock;

double elapsed_ms(const ComputeClock::time_point& start) {
  return std::chrono::duration<double, std::milli>(ComputeClock::now() - start).count();
}

}

bool Step1ComputeBackend::preprocess_genotypes(
  Eigen::MatrixXd& genotypes,
  const Eigen::Ref<const Eigen::MatrixXd>& covariates,
  const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
  double degrees_of_freedom,
  double minimum_scale,
  const Eigen::Ref<const Eigen::VectorXd>& row_multipliers,
  bool copy_to_host,
  Eigen::VectorXd& row_scales,
  Step1ComputeTimings* timings) {

  (void)copy_to_host;
  (void)timings;
  if(genotypes.cols() != covariates.rows() ||
     genotypes.cols() != sample_weights.size() ||
     (row_multipliers.size() != 0 &&
      row_multipliers.size() != genotypes.rows()))
    throw std::invalid_argument(
      "Step 1 genotype preprocessing received incompatible dimensions");
  if(!std::isfinite(degrees_of_freedom) || degrees_of_freedom <= 0)
    throw std::invalid_argument(
      "Step 1 genotype preprocessing requires positive degrees of freedom");
  if(!std::isfinite(minimum_scale) || minimum_scale < 0)
    throw std::invalid_argument(
      "Step 1 genotype preprocessing requires a non-negative minimum scale");
  if((sample_weights.array() < 0).any() ||
     (row_multipliers.array() < 0).any() ||
     !sample_weights.allFinite() || !covariates.allFinite() ||
     !row_multipliers.allFinite())
    throw std::invalid_argument(
      "Step 1 genotype preprocessing requires finite, non-negative weights and multipliers");
  row_scales.resize(genotypes.rows());
  return false;
}

bool Step1ComputeBackend::can_preprocess_packed_hardcalls(
  Eigen::Index variants,
  Eigen::Index samples) const {
  (void)variants;
  (void)samples;
  return false;
}

void Step1ComputeBackend::validate_packed_hardcall_preprocessing_inputs(
  const unsigned char* packed_hardcalls,
  size_t packed_bytes,
  size_t packed_stride_bytes,
  Eigen::Index variants,
  Eigen::Index samples,
  const Eigen::Ref<const Eigen::MatrixXd>& covariates,
  const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
  double degrees_of_freedom,
  double minimum_scale) {

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
  if((sample_weights.array() < 0).any() ||
     !sample_weights.allFinite() || !covariates.allFinite())
    throw std::invalid_argument(
      "Step 1 packed hardcall preprocessing requires finite, non-negative weights");
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
}

bool Step1ComputeBackend::preprocess_packed_hardcalls(
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
  Step1ComputeTimings* timings) {
  (void)packed_hardcalls;
  (void)packed_bytes;
  (void)packed_stride_bytes;
  (void)variants;
  (void)samples;
  (void)covariates;
  (void)sample_weights;
  (void)degrees_of_freedom;
  (void)minimum_scale;
  (void)row_scales;
  (void)timings;
  return false;
}

int Step1ComputeBackend::configure_packed_hardcall_pipeline(
  Eigen::Index maximum_variants,
  Eigen::Index samples,
  Eigen::Index covariate_count) {
  (void)maximum_variants;
  (void)samples;
  (void)covariate_count;
  return 1;
}

bool Step1ComputeBackend::submit_packed_hardcall_preprocessing(
  const unsigned char* packed_hardcalls,
  size_t packed_bytes,
  size_t packed_stride_bytes,
  Eigen::Index variants,
  Eigen::Index samples,
  const Eigen::Ref<const Eigen::MatrixXd>& covariates,
  const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
  double degrees_of_freedom,
  double minimum_scale,
  int& pipeline_slot) {
  (void)packed_hardcalls;
  (void)packed_bytes;
  (void)packed_stride_bytes;
  (void)variants;
  (void)samples;
  (void)covariates;
  (void)sample_weights;
  (void)degrees_of_freedom;
  (void)minimum_scale;
  pipeline_slot = -1;
  return false;
}

bool Step1ComputeBackend::activate_packed_hardcall_preprocessing(
  int pipeline_slot,
  Eigen::VectorXd& row_scales,
  Step1ComputeTimings* timings) {
  (void)pipeline_slot;
  (void)row_scales;
  (void)timings;
  return false;
}

void Step1ComputeBackend::finish_packed_hardcall_pipeline() {
}

void Step1ComputeBackend::compute_preprocessed_products(
  Eigen::Index start_column,
  Eigen::Index column_count,
  const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
  Eigen::MatrixXd& gram,
  Eigen::MatrixXd& crossproduct,
  Step1GramMode mode,
  Step1ComputeTimings* timings) {
  (void)start_column;
  (void)column_count;
  (void)phenotypes;
  (void)gram;
  (void)crossproduct;
  (void)mode;
  (void)timings;
  throw std::runtime_error(
    "Step 1 backend has no resident preprocessed genotype block");
}

void Step1ComputeBackend::ridge_predict_preprocessed(
  Eigen::Index start_column,
  Eigen::Index column_count,
  const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
  Eigen::MatrixXd& predictions,
  Eigen::MatrixXd& coefficients,
  Step1ComputeTimings* timings) {
  (void)start_column;
  (void)column_count;
  (void)ridge_parameters;
  (void)predictions;
  (void)coefficients;
  (void)timings;
  throw std::runtime_error(
    "Step 1 backend has no resident preprocessed genotype block");
}

bool Step1ComputeBackend::ridge_predict_preprocessed_system(
  const Eigen::Ref<const Eigen::MatrixXd>& gram,
  const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
  Eigen::Index start_column,
  Eigen::Index column_count,
  const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
  Eigen::MatrixXd& predictions,
  Eigen::MatrixXd& coefficients,
  Step1ComputeTimings* timings) {
  (void)gram;
  (void)right_hand_sides;
  (void)start_column;
  (void)column_count;
  (void)ridge_parameters;
  (void)predictions;
  (void)coefficients;
  (void)timings;
  return false;
}

bool Step1ComputeBackend::ridge_predict_preprocessed_systems(
  const std::vector<Eigen::MatrixXd>& grams,
  const std::vector<Eigen::MatrixXd>& right_hand_sides,
  const Eigen::Ref<const Eigen::VectorXi>& start_columns,
  const Eigen::Ref<const Eigen::VectorXi>& column_counts,
  const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
  std::vector<Eigen::MatrixXd>& predictions,
  std::vector<Eigen::MatrixXd>& coefficients,
  Step1ComputeTimings* timings) {
  (void)grams;
  (void)right_hand_sides;
  (void)start_columns;
  (void)column_counts;
  (void)ridge_parameters;
  (void)predictions;
  (void)coefficients;
  (void)timings;
  return false;
}

void Step1ComputeBackend::release_preprocessed_genotypes() {
}

bool Step1ComputeBackend::cache_design_partitions(
  const std::vector<Eigen::MatrixXd>& partitions,
  Step1ComputeTimings* timings) {
  (void)partitions;
  (void)timings;
  return false;
}

void Step1ComputeBackend::predict_cached_design(
  const Eigen::Ref<const Eigen::VectorXd>& coefficients,
  Eigen::VectorXd& predictions,
  Step1ComputeTimings* timings) {
  (void)coefficients;
  (void)predictions;
  (void)timings;
  throw std::runtime_error("Step 1 backend has no cached design matrix");
}

void Step1ComputeBackend::compute_cached_weighted_design_products(
  const Eigen::Ref<const Eigen::VectorXd>& weights,
  const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
  Eigen::MatrixXd& gram,
  Eigen::MatrixXd& crossproduct,
  Step1ComputeTimings* timings) {
  (void)weights;
  (void)outcomes;
  (void)gram;
  (void)crossproduct;
  (void)timings;
  throw std::runtime_error("Step 1 backend has no cached design matrix");
}

void Step1ComputeBackend::compute_cached_design_crossproduct(
  const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
  Eigen::MatrixXd& crossproduct,
  Step1ComputeTimings* timings) {
  (void)outcomes;
  (void)crossproduct;
  (void)timings;
  throw std::runtime_error("Step 1 backend has no cached design matrix");
}

void Step1ComputeBackend::release_cached_design() {
}

void Step1ComputeBackend::diagonal_penalty_predict(
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
  Step1ComputeTimings* timings) {

  const Eigen::Index size = gram.rows();
  const Eigen::Index sample_count = samples_in_columns ?
    prediction_matrix.cols() : prediction_matrix.rows();
  const Eigen::Index outcome_count = right_hand_sides.cols();
  const Eigen::Index parameter_count = ridge_parameters.size();
  if(gram.cols() != size || right_hand_sides.rows() != size ||
     penalty_multipliers.size() != size ||
     (samples_in_columns ? prediction_matrix.rows() : prediction_matrix.cols()) != size)
    throw std::invalid_argument(
      "Step 1 diagonal-penalty solve received incompatible matrix dimensions");
  if((ridge_parameters.array() < 0).any())
    throw std::invalid_argument("Step 1 diagonal-penalty parameters must be non-negative");
  if((penalty_multipliers.array() < 0).any())
    throw std::invalid_argument("Step 1 diagonal-penalty multipliers must be non-negative");
  if(leave_one_out &&
     (leave_one_out_outcomes.rows() != sample_count ||
      leave_one_out_outcomes.cols() != outcome_count))
    throw std::invalid_argument(
      "Step 1 diagonal-penalty LOOCV outcomes have incompatible dimensions");

  predictions.resize(sample_count, outcome_count * parameter_count);
  coefficients.resize(size, outcome_count * parameter_count);
  if(size == 0 || sample_count == 0 || outcome_count == 0 || parameter_count == 0) {
    predictions.setZero();
    coefficients.setZero();
    return;
  }

  const Eigen::VectorXd no_additional_ridge = Eigen::VectorXd::Zero(1);
  for(Eigen::Index parameter = 0; parameter < parameter_count; ++parameter) {
    Eigen::MatrixXd penalized_gram = gram;
    penalized_gram.diagonal().array() +=
      ridge_parameters(parameter) * penalty_multipliers.array();
    Eigen::MatrixXd eigenvectors, eigenvalues, transformed;
    eigendecompose_and_transform(penalized_gram, right_hand_sides,
      eigenvectors, eigenvalues, transformed, timings);
    Eigen::MatrixXd parameter_predictions, parameter_coefficients;
    ridge_predict(eigenvectors, eigenvalues, transformed, prediction_matrix,
      samples_in_columns, no_additional_ridge, leave_one_out_outcomes,
      leave_one_out, parameter_predictions, parameter_coefficients, timings);
    predictions.middleCols(parameter * outcome_count, outcome_count) =
      parameter_predictions;
    coefficients.middleCols(parameter * outcome_count, outcome_count) =
      parameter_coefficients;
  }
}

void Step1ComputeBackend::diagonal_penalty_solve(
  const Eigen::Ref<const Eigen::MatrixXd>& gram,
  const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
  const Eigen::Ref<const Eigen::VectorXd>& ridge_parameters,
  const Eigen::Ref<const Eigen::VectorXd>& penalty_multipliers,
  Eigen::MatrixXd& solutions,
  Step1ComputeTimings* timings) {

  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(gram.rows(), gram.rows());
  const Eigen::MatrixXd no_outcomes(0, 0);
  Eigen::MatrixXd predictions;
  diagonal_penalty_predict(gram, right_hand_sides, identity, false,
    ridge_parameters, penalty_multipliers, no_outcomes, false,
    predictions, solutions, timings);
}

void Step1ComputeBackend::compute_design_crossproduct(
  const Eigen::Ref<const Eigen::MatrixXd>& design,
  const Eigen::Ref<const Eigen::MatrixXd>& outcomes,
  Eigen::MatrixXd& crossproduct,
  Step1ComputeTimings* timings) {

  if(design.rows() != outcomes.rows())
    throw std::invalid_argument(
      "Step 1 design crossproduct received incompatible dimensions");
  crossproduct.resize(design.cols(), outcomes.cols());
  if(design.cols() == 0 || outcomes.cols() == 0 || design.rows() == 0) {
    crossproduct.setZero();
    return;
  }
  ComputeClock::time_point start;
  if(timings) start = ComputeClock::now();
  crossproduct.noalias() = design.transpose() * outcomes;
  if(timings) timings->crossproduct_ms += elapsed_ms(start);
}

void Step1ComputeBackend::grouped_leave_one_out_predict_factorized(
  const Eigen::Ref<const Eigen::MatrixXd>& design,
  const Eigen::Ref<const Eigen::VectorXd>& coefficients,
  const Eigen::Ref<const Eigen::VectorXd>& residuals,
  const Eigen::Ref<const Eigen::VectorXd>& leverage_weights,
  const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
  const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
  Eigen::MatrixXd& predictions,
  Step1ComputeTimings* timings) {

  if(design.cols() != coefficients.size() ||
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

  Eigen::MatrixXd inverse_design_transpose;
  solve_factorized(design.transpose(), inverse_design_transpose, timings);
  predictions.resize(design.rows(), group_offsets.size());
  if(design.rows() == 0 || group_offsets.size() == 0) {
    predictions.setZero();
    return;
  }

  ComputeClock::time_point start;
  if(timings) start = ComputeClock::now();
  const Eigen::VectorXd leverage =
    ((design.array() * inverse_design_transpose.transpose().array())
      .rowwise().sum() * leverage_weights.array()).matrix();
  const Eigen::VectorXd adjustment =
    (residuals.array() / (1.0 - leverage.array())).matrix();
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const Eigen::Index offset = group_offsets(group);
    const Eigen::Index count = group_sizes(group);
    if(count == 0) {
      predictions.col(group).setZero();
      continue;
    }
    predictions.col(group).noalias() =
      design.middleCols(offset, count) * coefficients.segment(offset, count);
    predictions.col(group).array() -= adjustment.array() *
      (design.middleCols(offset, count).array() *
       inverse_design_transpose.middleRows(offset, count).transpose().array())
        .rowwise().sum();
  }
  if(timings) timings->ridge_ms += elapsed_ms(start);
}

void Step1ComputeBackend::grouped_predict(
  const Eigen::Ref<const Eigen::MatrixXd>& design,
  const Eigen::Ref<const Eigen::VectorXd>& coefficients,
  const Eigen::Ref<const Eigen::VectorXi>& group_offsets,
  const Eigen::Ref<const Eigen::VectorXi>& group_sizes,
  Eigen::MatrixXd& predictions,
  Step1ComputeTimings* timings) {

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
  ComputeClock::time_point start;
  if(timings) start = ComputeClock::now();
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const Eigen::Index offset = group_offsets(group);
    const Eigen::Index count = group_sizes(group);
    if(count == 0)
      predictions.col(group).setZero();
    else
      predictions.col(group).noalias() =
        design.middleCols(offset, count) * coefficients.segment(offset, count);
  }
  if(timings) timings->ridge_ms += elapsed_ms(start);
}

class CpuStep1ComputeBackend : public Step1ComputeBackend {

  public:
    const char* name() const override {
      return "cpu";
    }

    std::string description() const override {
      return "Eigen CPU";
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

      gram.resize(genotypes.rows(), genotypes.rows());
      crossproduct.resize(genotypes.rows(), phenotypes.cols());
      if(genotypes.rows() == 0 || genotypes.cols() == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }

      if(mode == Step1GramMode::selfadjoint_rank_update) {
        ComputeClock::time_point start;
        if(timings) start = ComputeClock::now();
        gram.setZero(genotypes.rows(), genotypes.rows());
        gram.selfadjointView<Eigen::Lower>().rankUpdate(genotypes);
        gram.triangularView<Eigen::Upper>() = gram.transpose();
        if(timings) timings->gram_ms += elapsed_ms(start);

        if(timings) start = ComputeClock::now();
        if(phenotypes.cols() > 0)
          crossproduct.noalias() = genotypes * phenotypes;
        else
          crossproduct.setZero();
        if(timings) timings->crossproduct_ms += elapsed_ms(start);
      } else {
        ComputeClock::time_point start;
        if(timings) start = ComputeClock::now();
        if(phenotypes.cols() > 0)
          crossproduct.noalias() = genotypes * phenotypes;
        else
          crossproduct.setZero();
        if(timings) timings->crossproduct_ms += elapsed_ms(start);

        if(timings) start = ComputeClock::now();
        gram = genotypes * genotypes.transpose();
        if(timings) timings->gram_ms += elapsed_ms(start);
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

      if(symmetric_matrix.rows() == 0) {
        eigenvectors.resize(0, 0);
        eigenvalues.resize(0, 1);
        transformed_right_hand_sides.resize(0, right_hand_sides.cols());
        return;
      }

      ComputeClock::time_point start;
      if(timings) start = ComputeClock::now();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric_matrix);
      if(solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen CPU symmetric eigendecomposition failed");
      eigenvectors = solver.eigenvectors();
      eigenvalues = solver.eigenvalues();
      if(timings) timings->eigensolve_ms += elapsed_ms(start);

      if(timings) start = ComputeClock::now();
      transformed_right_hand_sides = eigenvectors.transpose() * right_hand_sides;
      if(timings) timings->transform_ms += elapsed_ms(start);
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
      gram.resize(design.cols(), design.cols());
      crossproduct.resize(design.cols(), outcomes.cols());
      ComputeClock::time_point start;
      if(timings) start = ComputeClock::now();
      crossproduct.noalias() = design.transpose() * outcomes;
      if(timings) timings->crossproduct_ms += elapsed_ms(start);
      if(timings) start = ComputeClock::now();
      gram.noalias() = design.transpose() * design;
      if(timings) timings->gram_ms += elapsed_ms(start);
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
      gram.resize(design.cols(), design.cols());
      crossproduct.resize(design.cols(), outcomes.cols());
      if(design.cols() == 0 || design.rows() == 0) {
        gram.setZero();
        crossproduct.setZero();
        return;
      }
      const Eigen::MatrixXd weighted_design =
        (design.array().colwise() * weights.array()).matrix();
      ComputeClock::time_point start;
      if(timings) start = ComputeClock::now();
      crossproduct.noalias() = design.transpose() *
        (outcomes.array().colwise() * weights.array()).matrix();
      if(timings) timings->crossproduct_ms += elapsed_ms(start);
      if(timings) start = ComputeClock::now();
      gram.noalias() = design.transpose() * weighted_design;
      if(timings) timings->gram_ms += elapsed_ms(start);
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

      Eigen::MatrixXd system = gram;
      system.diagonal().array() += ridge_parameter * penalty_multipliers.array();
      ComputeClock::time_point start;
      if(timings) start = ComputeClock::now();
      factorization_.compute(system);
      if(factorization_.info() != Eigen::Success)
        throw std::runtime_error("Eigen CPU reusable Cholesky factorization failed");
      factorization_size_ = gram.rows();
      if(timings) timings->ridge_ms += elapsed_ms(start);
    }

    void solve_factorized(
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Eigen::MatrixXd& solutions,
      Step1ComputeTimings* timings) override {

      if(factorization_size_ < 0)
        throw std::runtime_error(
          "Step 1 reusable solve requested before factorization");
      if(right_hand_sides.rows() != factorization_size_)
        throw std::invalid_argument(
          "Step 1 reusable solve received incompatible right-hand sides");
      solutions.resize(right_hand_sides.rows(), right_hand_sides.cols());
      if(right_hand_sides.size() == 0) {
        solutions.setZero();
        return;
      }
      ComputeClock::time_point start;
      if(timings) start = ComputeClock::now();
      solutions = factorization_.solve(right_hand_sides);
      if(factorization_.info() != Eigen::Success)
        throw std::runtime_error("Eigen CPU reusable Cholesky solve failed");
      if(timings) timings->ridge_ms += elapsed_ms(start);
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

      const Eigen::Index size = eigenvectors.rows();
      const Eigen::Index sample_count = samples_in_columns ?
        prediction_matrix.cols() : prediction_matrix.rows();
      const Eigen::Index phenotype_count = transformed_right_hand_sides.cols();
      const Eigen::Index parameter_count = ridge_parameters.size();
      const Eigen::Index combination_count = phenotype_count * parameter_count;
      predictions.resize(sample_count, combination_count);
      coefficients.resize(size, combination_count);
      if(size == 0 || sample_count == 0 || combination_count == 0) {
        predictions.setZero();
        coefficients.setZero();
        return;
      }

      ComputeClock::time_point start;
      if(timings) start = ComputeClock::now();
      Eigen::MatrixXd inverse(size, parameter_count);
      Eigen::MatrixXd scaled_rhs(size, combination_count);
      for(Eigen::Index parameter = 0; parameter < parameter_count; ++parameter) {
        inverse.col(parameter) =
          (eigenvalues.col(0).array() + ridge_parameters(parameter)).inverse().matrix();
        for(Eigen::Index phenotype = 0; phenotype < phenotype_count; ++phenotype)
          scaled_rhs.col(parameter * phenotype_count + phenotype) =
            inverse.col(parameter).array() * transformed_right_hand_sides.col(phenotype).array();
      }

      coefficients.noalias() = eigenvectors * scaled_rhs;
      if(samples_in_columns)
        predictions.noalias() = prediction_matrix.transpose() * coefficients;
      else
        predictions.noalias() = prediction_matrix * coefficients;

      if(leave_one_out) {
        Eigen::MatrixXd projected_genotypes;
        if(samples_in_columns)
          projected_genotypes = eigenvectors.transpose() * prediction_matrix;
        else
          projected_genotypes = eigenvectors.transpose() * prediction_matrix.transpose();
        const Eigen::MatrixXd leverage =
          projected_genotypes.array().square().matrix().transpose() * inverse;
        for(Eigen::Index parameter = 0; parameter < parameter_count; ++parameter)
          for(Eigen::Index phenotype = 0; phenotype < phenotype_count; ++phenotype) {
            const Eigen::Index column = parameter * phenotype_count + phenotype;
            predictions.col(column).array() -= leverage.col(parameter).array() *
              leave_one_out_outcomes.col(phenotype).array();
            predictions.col(column).array() /= 1.0 - leverage.col(parameter).array();
          }
      }
      if(timings) timings->ridge_ms += elapsed_ms(start);
    }

    void factorize_ridge_system(
      const Eigen::Ref<const Eigen::MatrixXd>& symmetric_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& right_hand_sides,
      Step1ComputeTimings* timings) override {

      eigendecompose_and_transform(symmetric_matrix, right_hand_sides,
        factorized_ridge_vectors_, factorized_ridge_values_,
        factorized_ridge_rhs_, timings);
      ridge_factorized_ = true;
    }

    void compute_products_and_factorize_ridge(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Step1GramMode mode,
      Step1ComputeTimings* timings) override {

      Eigen::MatrixXd gram, crossproduct;
      compute_products(genotypes, phenotypes, gram, crossproduct, mode, timings);
      factorize_ridge_system(gram, crossproduct, timings);
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

      if(!ridge_factorized_)
        throw std::runtime_error(
          "Step 1 factorized ridge prediction requested before factorization");
      ridge_predict(factorized_ridge_vectors_, factorized_ridge_values_,
        factorized_ridge_rhs_, prediction_matrix, samples_in_columns,
        ridge_parameters, leave_one_out_outcomes, leave_one_out,
        predictions, coefficients, timings);
    }

  private:
    Eigen::LLT<Eigen::MatrixXd> factorization_;
    Eigen::Index factorization_size_ = -1;
    Eigen::MatrixXd factorized_ridge_vectors_;
    Eigen::MatrixXd factorized_ridge_values_;
    Eigen::MatrixXd factorized_ridge_rhs_;
    bool ridge_factorized_ = false;

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
};

std::unique_ptr<Step1ComputeBackend> make_cpu_step1_compute_backend() {
  return std::unique_ptr<Step1ComputeBackend>(new CpuStep1ComputeBackend());
}

bool cuda_step1_compute_backend_compiled() {
#ifdef WITH_CUDA
  return true;
#else
  return false;
#endif
}

std::unique_ptr<Step1ComputeBackend> make_step1_compute_backend(
  const std::string& requested_backend,
  int device) {

  if(requested_backend == "cpu")
    return make_cpu_step1_compute_backend();

  if(requested_backend != "cuda" && requested_backend != "auto")
    throw std::runtime_error("unknown Step 1 compute backend '" + requested_backend +
      "' (expected cpu, cuda, or auto)");

#ifdef WITH_CUDA
  std::string reason;
  if(cuda_step1_compute_backend_available(device, reason))
    return make_cuda_step1_compute_backend(device);
  if(requested_backend == "cuda")
    throw std::runtime_error("CUDA Step 1 backend is unavailable: " + reason);
#else
  (void)device;
  if(requested_backend == "cuda")
    throw std::runtime_error(
      "CUDA Step 1 backend was requested, but this binary was built without REGENIE_WITH_CUDA");
#endif

  return make_cpu_step1_compute_backend();
}
