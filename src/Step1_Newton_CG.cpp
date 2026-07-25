#include "Step1_Newton_CG.hpp"

#include <cmath>
#include <stdexcept>

Step1NewtonCgResult step1_preconditioned_conjugate_gradient(
  const Eigen::Ref<const Eigen::VectorXd>& right_hand_side,
  double relative_tolerance,
  int maximum_iterations,
  const Step1NewtonCgOperator& apply_matrix,
  const Step1NewtonCgOperator& apply_preconditioner) {

  if(!right_hand_side.allFinite())
    throw std::invalid_argument(
      "Newton-CG requires a finite right-hand side");
  if(!std::isfinite(relative_tolerance) || relative_tolerance < 0)
    throw std::invalid_argument(
      "Newton-CG requires a finite non-negative relative tolerance");
  if(maximum_iterations < 1)
    throw std::invalid_argument(
      "Newton-CG requires at least one iteration");
  if(!apply_matrix || !apply_preconditioner)
    throw std::invalid_argument(
      "Newton-CG requires matrix and preconditioner operators");

  Step1NewtonCgResult result;
  result.solution = Eigen::VectorXd::Zero(right_hand_side.size());
  Eigen::VectorXd residual = right_hand_side;
  const double initial_residual_norm = residual.norm();
  result.residual_norm = initial_residual_norm;
  if(initial_residual_norm == 0) {
    result.valid = true;
    result.converged = true;
    return result;
  }

  Eigen::VectorXd preconditioned;
  if(!apply_preconditioner(residual, preconditioned))
    return result;
  ++result.preconditioner_calls;
  if(preconditioned.size() != residual.size() ||
     !preconditioned.allFinite())
    return result;
  double residual_dot_preconditioned = residual.dot(preconditioned);
  if(!std::isfinite(residual_dot_preconditioned) ||
     residual_dot_preconditioned <= 0)
    return result;

  Eigen::VectorXd direction = preconditioned;
  for(int iteration = 0; iteration < maximum_iterations; ++iteration) {
    Eigen::VectorXd matrix_direction;
    if(!apply_matrix(direction, matrix_direction))
      return result;
    ++result.operator_calls;
    if(matrix_direction.size() != direction.size() ||
       !matrix_direction.allFinite())
      return result;

    const double curvature = direction.dot(matrix_direction);
    if(!std::isfinite(curvature) || curvature <= 0)
      return result;
    const double step = residual_dot_preconditioned / curvature;
    if(!std::isfinite(step))
      return result;
    result.solution.noalias() += step * direction;
    residual.noalias() -= step * matrix_direction;
    result.iterations = iteration + 1;
    result.residual_norm = residual.norm();
    if(!std::isfinite(result.residual_norm) ||
       !result.solution.allFinite())
      return result;
    if(result.residual_norm <=
         relative_tolerance * initial_residual_norm) {
      result.valid = true;
      result.converged = true;
      return result;
    }

    if(!apply_preconditioner(residual, preconditioned))
      return result;
    ++result.preconditioner_calls;
    if(preconditioned.size() != residual.size() ||
       !preconditioned.allFinite())
      return result;
    const double next_residual_dot_preconditioned =
      residual.dot(preconditioned);
    if(!std::isfinite(next_residual_dot_preconditioned) ||
       next_residual_dot_preconditioned <= 0)
      return result;
    const double direction_scale =
      next_residual_dot_preconditioned / residual_dot_preconditioned;
    if(!std::isfinite(direction_scale))
      return result;
    direction = preconditioned + direction_scale * direction;
    residual_dot_preconditioned = next_residual_dot_preconditioned;
  }

  result.valid = true;
  return result;
}
