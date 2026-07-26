#include "Step1_Cox_Path_Newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace {

constexpr int kMaximumCorrections = 4;
constexpr int kMaximumHalvings = 4;

bool point_is_finite(const Step1CoxPathNewtonPoint& point) {
  return point.coefficients.allFinite() &&
    point.linear_predictor.allFinite() &&
    point.ordered_linear_predictor.allFinite() &&
    point.unpenalized_score.allFinite() &&
    std::isfinite(point.negative_log_likelihood) &&
    std::isfinite(point.deviance);
}

double loss_roundoff_allowance(double loss) {
  return 32 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, std::fabs(loss));
}

}

Step1CoxPathNewtonResult run_step1_cox_path_newton(
  const Step1CoxPathNewtonPoint& initial_point,
  double penalty,
  double score_tolerance,
  const Step1CoxPathNewtonSolve& solve,
  const Step1CoxPathNewtonEvaluate& evaluate) {

  Step1CoxPathNewtonResult result;
  result.accepted_point = initial_point;
  ++result.stats.attempts;
  if(penalty < 0 || score_tolerance < 0 ||
     !std::isfinite(penalty) || !std::isfinite(score_tolerance) ||
     !point_is_finite(initial_point) ||
     initial_point.unpenalized_score.size() !=
       initial_point.coefficients.size() ||
     initial_point.linear_predictor.size() !=
       initial_point.ordered_linear_predictor.size()) {
    ++result.stats.rejected_nonfinite;
    return result;
  }

  result.score = initial_point.unpenalized_score -
    penalty * initial_point.coefficients;
  result.penalized_loss = initial_point.negative_log_likelihood +
    0.5 * penalty * initial_point.coefficients.squaredNorm();
  if(!result.score.allFinite() ||
     !std::isfinite(result.penalized_loss)) {
    ++result.stats.rejected_nonfinite;
    return result;
  }

  double accepted_score_max = result.score.size() ?
    result.score.cwiseAbs().maxCoeff() : 0;
  if(accepted_score_max <= score_tolerance) {
    result.converged = true;
    ++result.stats.converged_without_irls;
    return result;
  }

  for(int correction = 0;
      correction < kMaximumCorrections; ++correction) {
    Eigen::VectorXd step;
    ++result.stats.correction_solves;
    if(!solve(result.score, penalty, step)) {
      ++result.stats.cached_gram_unavailable;
      break;
    }
    result.solver_supported = true;
    if(step.size() != result.score.size() || !step.allFinite()) {
      ++result.stats.rejected_nonfinite;
      break;
    }
    const double directional_derivative = result.score.dot(step);
    if(!std::isfinite(directional_derivative) ||
       directional_derivative <= 0) {
      ++result.stats.rejected_non_descent;
      break;
    }

    bool accepted = false;
    double step_scale = 1.0;
    for(int halving = 0; halving <= kMaximumHalvings; ++halving) {
      const Eigen::VectorXd candidate_coefficients =
        result.accepted_point.coefficients + step_scale * step;
      Step1CoxPathNewtonPoint candidate;
      ++result.stats.prediction_calls;
      ++result.stats.exact_score_calls;
      const bool valid_candidate =
        evaluate(candidate_coefficients, candidate) &&
        candidate.coefficients.size() == result.score.size() &&
        candidate.unpenalized_score.size() == result.score.size() &&
        candidate.linear_predictor.size() ==
          initial_point.linear_predictor.size() &&
        candidate.ordered_linear_predictor.size() ==
          initial_point.ordered_linear_predictor.size() &&
        (candidate.coefficients.array() ==
          candidate_coefficients.array()).all() &&
        point_is_finite(candidate);
      if(!valid_candidate) {
        ++result.stats.rejected_nonfinite;
      } else {
        const Eigen::VectorXd candidate_score =
          candidate.unpenalized_score - penalty * candidate.coefficients;
        const double candidate_score_max = candidate_score.size() ?
          candidate_score.cwiseAbs().maxCoeff() : 0;
        const double candidate_loss =
          candidate.negative_log_likelihood +
            0.5 * penalty * candidate.coefficients.squaredNorm();
        const bool objective_improved =
          std::isfinite(candidate_loss) &&
          candidate_loss <= result.penalized_loss +
            loss_roundoff_allowance(result.penalized_loss);
        const bool score_improved =
          candidate_score.allFinite() &&
          std::isfinite(candidate_score_max) &&
          candidate_score_max < accepted_score_max;
        if(objective_improved && score_improved) {
          result.accepted_point = std::move(candidate);
          result.score = candidate_score;
          result.penalized_loss = candidate_loss;
          accepted_score_max = candidate_score_max;
          ++result.stats.accepted_corrections;
          accepted = true;
          break;
        }
        if(!objective_improved)
          ++result.stats.rejected_objective;
        else
          ++result.stats.rejected_score;
      }
      if(halving < kMaximumHalvings) {
        step_scale *= 0.5;
        ++result.stats.halvings;
      }
    }
    if(!accepted) break;
    if(accepted_score_max <= score_tolerance) {
      result.converged = true;
      ++result.stats.converged_without_irls;
      break;
    }
  }

  return result;
}

void accumulate_step1_cox_path_newton_stats(
  Step1CoxPathNewtonStats& destination,
  const Step1CoxPathNewtonStats& source) {

  destination.attempts += source.attempts;
  destination.correction_solves += source.correction_solves;
  destination.accepted_corrections += source.accepted_corrections;
  destination.converged_without_irls += source.converged_without_irls;
  destination.fallbacks += source.fallbacks;
  destination.cached_gram_unavailable += source.cached_gram_unavailable;
  destination.prediction_calls += source.prediction_calls;
  destination.exact_score_calls += source.exact_score_calls;
  destination.halvings += source.halvings;
  destination.rejected_nonfinite += source.rejected_nonfinite;
  destination.rejected_non_descent += source.rejected_non_descent;
  destination.rejected_objective += source.rejected_objective;
  destination.rejected_score += source.rejected_score;
  destination.ordinary_weighted_solves += source.ordinary_weighted_solves;
}
