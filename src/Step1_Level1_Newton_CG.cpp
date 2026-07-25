#include "Step1_Level1_Newton_CG.hpp"

#include <cmath>
#include <limits>

#include "Step1_Newton_CG.hpp"

using Step1Level1BoolArray =
  Eigen::Array<bool, Eigen::Dynamic, 1>;

bool get_wvec(
  Eigen::ArrayXd& probabilities,
  Eigen::ArrayXd& variances,
  const Eigen::Ref<const Step1Level1BoolArray>& mask,
  const double& tolerance);

void get_pvec(
  Eigen::ArrayXd& probabilities,
  const Eigen::Ref<const Eigen::ArrayXd>& linear_predictor,
  const double& tolerance);

namespace {

constexpr int kMaximumCorrections = 2;
constexpr int kMaximumCgIterations = 8;
constexpr int kMaximumLineSearchIterations = 4;
constexpr double kCgRelativeTolerance = 0.1;

bool evaluate_point(
  Step1ComputeBackend* compute_backend,
  Step1ComputeTimings* profile_timings,
  int held_out_fold,
  int phenotype,
  int fold_count,
  double penalty,
  double probability_tolerance,
  double variance_tolerance,
  const Eigen::ArrayXd& penalty_multipliers,
  const Eigen::ArrayXd& coefficients,
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
  Eigen::ArrayXd& variances,
  Eigen::ArrayXd& score) {

  const Eigen::VectorXd coefficient_vector = coefficients.matrix();
  compute_backend->predict_cached_design(
    coefficient_vector, cached_predictions, profile_timings);
  cached_weights.setZero();
  cached_score_outcome.setZero();
  for(int fold = 0; fold < fold_count; ++fold) {
    if(fold == held_out_fold) continue;
    const Eigen::Index start = fold_offsets[fold];
    const Eigen::Index count = fold_offsets[fold + 1] - start;
    linear_predictor = fold_offset_values[fold].array() +
      cached_predictions.segment(start, count).array();
    get_pvec(probabilities, linear_predictor, probability_tolerance);
    if(get_wvec(
         probabilities, variances,
         fold_masks[fold].col(phenotype).array(),
         variance_tolerance))
      return false;
    cached_weights.segment(start, count) =
      fold_masks[fold].col(phenotype).array().select(
        variances, 0).matrix();
    cached_score_outcome.col(0).segment(start, count) =
      fold_masks[fold].col(phenotype).array().select(
        fold_outcomes[fold].array() - probabilities, 0).matrix();
  }

  Eigen::MatrixXd crossproduct;
  compute_backend->compute_cached_design_crossproduct(
    cached_score_outcome, crossproduct, profile_timings);
  score = crossproduct.col(0).array() -
    penalty * penalty_multipliers * coefficients;
  return score.allFinite();
}

}

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
  Eigen::ArrayXd& variances) {

  Step1Level1NewtonCgResult result;
  result.accepted_coefficients = initial_coefficients;
  if(!evaluate_point(
       compute_backend, profile_timings, held_out_fold, phenotype,
       fold_count, penalty, probability_tolerance, variance_tolerance,
       penalty_multipliers, result.accepted_coefficients, fold_offsets,
       fold_offset_values, fold_outcomes, fold_masks, cached_predictions,
       cached_weights, cached_score_outcome, linear_predictor,
       probabilities, variances, result.score)) {
    result.invalid_variance = true;
    return result;
  }
  ++result.prediction_calls;
  ++result.score_calls;

  double accepted_score_max = result.score.abs().maxCoeff();
  if(accepted_score_max < score_tolerance) {
    result.supported = true;
    result.converged = true;
    return result;
  }
  if(!compute_backend->factorize_cached_weighted_gram(
       penalty, penalty_multipliers.matrix(), profile_timings))
    return result;
  result.supported = true;

  for(int correction = 0; correction < kMaximumCorrections; ++correction) {
    const Eigen::VectorXd right_hand_side = result.score.matrix();
    const Step1NewtonCgOperator apply_hessian =
      [&] (const Eigen::VectorXd& vector, Eigen::VectorXd& product) {
        const Eigen::MatrixXd vectors = vector;
        Eigen::MatrixXd products;
        if(!compute_backend->
             compute_cached_weighted_design_hessian_product(
               cached_weights, vectors, products, profile_timings))
          return false;
        ++result.hessian_product_calls;
        if(products.cols() != 1) return false;
        product = products.col(0);
        product.array() +=
          penalty * penalty_multipliers * vector.array();
        return product.allFinite();
      };
    const Step1NewtonCgOperator apply_preconditioner =
      [&] (const Eigen::VectorXd& vector, Eigen::VectorXd& product) {
        const Eigen::MatrixXd right_hand_sides = vector;
        Eigen::MatrixXd solutions;
        if(!compute_backend->solve_factorized_cached_weighted_gram(
             right_hand_sides, solutions, profile_timings))
          return false;
        ++result.preconditioner_calls;
        if(solutions.cols() != 1) return false;
        product = solutions.col(0);
        return product.allFinite();
      };
    const Step1NewtonCgResult cg_result =
      step1_preconditioned_conjugate_gradient(
        right_hand_side, kCgRelativeTolerance, kMaximumCgIterations,
        apply_hessian, apply_preconditioner);
    if(!cg_result.valid || !cg_result.converged ||
       !cg_result.solution.allFinite())
      break;

    bool accepted = false;
    double step_scale = 1.0;
    for(int line_search = 0;
        line_search < kMaximumLineSearchIterations; ++line_search) {
      const Eigen::ArrayXd candidate =
        result.accepted_coefficients +
          step_scale * cg_result.solution.array();
      Eigen::ArrayXd candidate_score;
      const bool valid_candidate = evaluate_point(
        compute_backend, profile_timings, held_out_fold, phenotype,
        fold_count, penalty, probability_tolerance, variance_tolerance,
        penalty_multipliers, candidate, fold_offsets, fold_offset_values,
        fold_outcomes, fold_masks, cached_predictions, cached_weights,
        cached_score_outcome, linear_predictor, probabilities, variances,
        candidate_score);
      ++result.prediction_calls;
      ++result.score_calls;
      const double candidate_score_max =
        valid_candidate ? candidate_score.abs().maxCoeff() :
          std::numeric_limits<double>::infinity();
      if(std::isfinite(candidate_score_max) &&
         candidate_score_max < accepted_score_max) {
        result.accepted_coefficients = candidate;
        result.score = candidate_score;
        accepted_score_max = candidate_score_max;
        ++result.accepted_steps;
        accepted = true;
        break;
      }
      step_scale *= 0.5;
    }
    if(!accepted) break;
    if(accepted_score_max < score_tolerance) {
      result.converged = true;
      break;
    }
  }

  return result;
}
