#ifndef STEP1_NEWTON_CG_H
#define STEP1_NEWTON_CG_H

#include <functional>

#include <Eigen/Dense>

using Step1NewtonCgOperator =
  std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)>;

struct Step1NewtonCgResult {
  Eigen::VectorXd solution;
  double residual_norm = 0;
  int iterations = 0;
  int operator_calls = 0;
  int preconditioner_calls = 0;
  bool valid = false;
  bool converged = false;
};

Step1NewtonCgResult step1_preconditioned_conjugate_gradient(
  const Eigen::Ref<const Eigen::VectorXd>& right_hand_side,
  double relative_tolerance,
  int maximum_iterations,
  const Step1NewtonCgOperator& apply_matrix,
  const Step1NewtonCgOperator& apply_preconditioner);

#endif
