#ifndef STEP1_COX_PATH_NEWTON_H
#define STEP1_COX_PATH_NEWTON_H

#include <cstdint>
#include <functional>

#include <Eigen/Dense>

struct Step1CoxPathNewtonPoint {
  Eigen::VectorXd coefficients;
  Eigen::VectorXd linear_predictor;
  Eigen::VectorXd ordered_linear_predictor;
  Eigen::VectorXd unpenalized_score;
  double negative_log_likelihood = 0;
  double deviance = 0;
};

struct Step1CoxPathNewtonStats {
  uint64_t attempts = 0;
  uint64_t correction_solves = 0;
  uint64_t accepted_corrections = 0;
  uint64_t converged_without_irls = 0;
  uint64_t fallbacks = 0;
  uint64_t cached_gram_unavailable = 0;
  uint64_t prediction_calls = 0;
  uint64_t exact_score_calls = 0;
  uint64_t halvings = 0;
  uint64_t rejected_nonfinite = 0;
  uint64_t rejected_non_descent = 0;
  uint64_t rejected_objective = 0;
  uint64_t rejected_score = 0;
  uint64_t ordinary_weighted_solves = 0;
};

struct Step1CoxPathNewtonResult {
  Step1CoxPathNewtonPoint accepted_point;
  Eigen::VectorXd score;
  double penalized_loss = 0;
  bool solver_supported = false;
  bool converged = false;
  Step1CoxPathNewtonStats stats;
};

using Step1CoxPathNewtonSolve =
  std::function<bool(
    const Eigen::VectorXd&, double, Eigen::VectorXd&)>;
using Step1CoxPathNewtonEvaluate =
  std::function<bool(
    const Eigen::VectorXd&, Step1CoxPathNewtonPoint&)>;

Step1CoxPathNewtonResult run_step1_cox_path_newton(
  const Step1CoxPathNewtonPoint& initial_point,
  double penalty,
  double score_tolerance,
  const Step1CoxPathNewtonSolve& solve,
  const Step1CoxPathNewtonEvaluate& evaluate);

void accumulate_step1_cox_path_newton_stats(
  Step1CoxPathNewtonStats& destination,
  const Step1CoxPathNewtonStats& source);

#endif
