#include "Step1_Level1_Optimizer.hpp"
#include "cox_ridge.hpp"
#include "survival_data.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

struct CoxFixture {
  survival_data data;
  Eigen::MatrixXd design;
  Eigen::VectorXd offset;
  ArrayXb mask;
};

CoxFixture make_fixture() {
  constexpr int sample_count = 96;
  Eigen::VectorXd time(sample_count);
  Eigen::VectorXd status = Eigen::VectorXd::Zero(sample_count);
  ArrayXb mask = ArrayXb::Constant(sample_count, true);
  Eigen::MatrixXd design(sample_count, 3);
  Eigen::VectorXd offset(sample_count);
  for(int row = 0; row < sample_count; ++row) {
    time(row) = 1 + row / 4;
    status(row) = (row % 3 == 0 || row % 13 == 0) ? 1 : 0;
    if(row == 5 || row == 37 || row == 81) mask(row) = false;
    design(row, 0) = (row % 5 - 2) / 3.0;
    design(row, 1) = std::sin(0.11 * row);
    design(row, 2) = row % 7 == 0 ? 1 : 0;
    offset(row) = 0.03 * std::cos(0.07 * row);
  }
  survival_data data;
  data.setup(time, status, mask, true);
  return {data, design, offset, mask};
}

cox_ridge evaluate(
  const CoxFixture& fixture,
  const Eigen::VectorXd& coefficients,
  double penalty) {
  return cox_ridge(
    fixture.data, fixture.design, fixture.offset, fixture.mask,
    penalty, 100, 30, 1e-10, false, coefficients);
}

void check_exact_penalized_score() {
  const CoxFixture fixture = make_fixture();
  const double penalty = 0.35;
  Eigen::VectorXd coefficients(3);
  coefficients << 0.08, -0.12, 0.05;
  cox_ridge model = evaluate(fixture, coefficients, penalty);
  model.coxGrad(fixture.data);
  const Eigen::VectorXd analytic_score =
    fixture.design.transpose() * model.get_gradient() -
      penalty * coefficients;

  constexpr double epsilon = 1e-6;
  Eigen::VectorXd numerical_score(coefficients.size());
  for(Eigen::Index coefficient = 0;
      coefficient < coefficients.size(); ++coefficient) {
    Eigen::VectorXd plus = coefficients;
    Eigen::VectorXd minus = coefficients;
    plus(coefficient) += epsilon;
    minus(coefficient) -= epsilon;
    cox_ridge plus_model = evaluate(fixture, plus, penalty);
    cox_ridge minus_model = evaluate(fixture, minus, penalty);
    const double plus_loss =
      plus_model.get_deviance_all()(0) / 2 +
        penalty * plus.squaredNorm() / 2;
    const double minus_loss =
      minus_model.get_deviance_all()(0) / 2 +
        penalty * minus.squaredNorm() / 2;
    numerical_score(coefficient) =
      -(plus_loss - minus_loss) / (2 * epsilon);
  }
  const double maximum_error =
    (analytic_score - numerical_score).cwiseAbs().maxCoeff();
  if(!std::isfinite(maximum_error) || maximum_error > 2e-7) {
    std::cerr << std::setprecision(17)
              << "analytic_score=" << analytic_score.transpose()
              << " numerical_score=" << numerical_score.transpose()
              << " maximum_error=" << maximum_error << "\n";
    throw std::runtime_error(
      "Cox penalized score finite-difference check failed");
  }
}

void check_nonresident_optimizer_fallback() {
  const CoxFixture fixture = make_fixture();
  Eigen::VectorXd penalties(3);
  penalties << 1.2, 0.45, 0.15;
  cox_ridge_path irls(
    fixture.data, fixture.design, fixture.offset, fixture.mask,
    penalties.size(), 1e-4, penalties, 80, 20, 1e-8, false,
    nullptr, false, nullptr, Step1Level1Optimizer::Irls);
  cox_ridge_path path_newton(
    fixture.data, fixture.design, fixture.offset, fixture.mask,
    penalties.size(), 1e-4, penalties, 80, 20, 1e-8, false,
    nullptr, false, nullptr, Step1Level1Optimizer::PathNewton);
  cox_ridge_path newton_cg(
    fixture.data, fixture.design, fixture.offset, fixture.mask,
    penalties.size(), 1e-4, penalties, 80, 20, 1e-8, false,
    nullptr, false, nullptr, Step1Level1Optimizer::NewtonCg);
  irls.fit(
    fixture.data, fixture.design, fixture.offset, fixture.mask);
  path_newton.fit(
    fixture.data, fixture.design, fixture.offset, fixture.mask);
  newton_cg.fit(
    fixture.data, fixture.design, fixture.offset, fixture.mask);
  if(!irls.beta_mx.isApprox(path_newton.beta_mx, 0) ||
     !irls.beta_mx.isApprox(newton_cg.beta_mx, 0) ||
     !irls.eta_mx.isApprox(path_newton.eta_mx, 0) ||
     !irls.eta_mx.isApprox(newton_cg.eta_mx, 0) ||
     !irls.deviance.isApprox(path_newton.deviance, 0) ||
     !irls.deviance.isApprox(newton_cg.deviance, 0) ||
     path_newton.path_newton_stats.attempts != 0 ||
     newton_cg.path_newton_stats.attempts != 0)
    throw std::runtime_error(
      "Cox optimizer did not preserve the nonresident IRLS fallback");
}

}

int main() {
  try {
    check_exact_penalized_score();
    check_nonresident_optimizer_fallback();
    std::cout << "COX_RIDGE_TEST status=PASS\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "COX_RIDGE_TEST status=FAIL error=\""
              << error.what() << "\"\n";
    return 1;
  }
}
