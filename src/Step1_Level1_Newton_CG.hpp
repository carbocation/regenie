#ifndef STEP1_LEVEL1_NEWTON_CG_H
#define STEP1_LEVEL1_NEWTON_CG_H

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "Step1_Compute.hpp"

struct Step1Level1NewtonCgResult {
  Eigen::ArrayXd accepted_coefficients;
  Eigen::ArrayXd score;
  bool supported = false;
  bool converged = false;
  bool invalid_variance = false;
  uint64_t accepted_steps = 0;
  uint64_t hessian_product_calls = 0;
  uint64_t preconditioner_calls = 0;
  uint64_t prediction_calls = 0;
  uint64_t score_calls = 0;
};

Step1Level1NewtonCgResult run_step1_level1_newton_cg_corrections(
  Step1ComputeBackend* compute_backend,
  Step1ComputeTimings* profile_timings,
  int held_out_fold,
  int phenotype,
  int fold_count,
  double penalty,
  double score_tolerance,
  double probability_tolerance,
  double variance_tolerance,
  const Eigen::ArrayXd& penalty_multipliers,
  const Eigen::ArrayXd& initial_coefficients,
  const std::vector<Eigen::Index>& fold_offsets,
  const std::vector<Eigen::MatrixXd>& fold_offset_values,
  const std::vector<Eigen::MatrixXd>& fold_outcomes,
  const std::vector<
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>& fold_masks,
  Eigen::VectorXd& cached_predictions,
  Eigen::VectorXd& cached_weights,
  Eigen::MatrixXd& cached_score_outcome,
  Eigen::ArrayXd& linear_predictor,
  Eigen::ArrayXd& probabilities,
  Eigen::ArrayXd& variances);

#endif
