/* Deterministic conformance tests for blockwise Step 2 CPU scoring. */

#include "Step2_Compute.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

Eigen::MatrixXd deterministic_matrix(Eigen::Index rows,
    Eigen::Index columns, double phase) {
  Eigen::MatrixXd result(rows, columns);
  for(Eigen::Index column = 0; column < columns; ++column)
    for(Eigen::Index row = 0; row < rows; ++row) {
      const double index = 1.0 + row + rows * column;
      result(row, column) = std::sin(index * 0.17 + phase) +
        0.25 * std::cos(index * 0.31 - phase);
    }
  return result;
}

double relative_error(const Eigen::MatrixXd& actual,
    const Eigen::MatrixXd& expected) {
  if(actual.rows() != expected.rows() ||
     actual.cols() != expected.cols())
    return std::numeric_limits<double>::max();
  if(actual.size() == 0) return 0;
  const double scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
  return (actual - expected).cwiseAbs().maxCoeff() / scale;
}

void require_close(const Eigen::MatrixXd& actual,
    const Eigen::MatrixXd& expected, const char* label) {
  if(relative_error(actual, expected) > 2e-12)
    throw std::runtime_error(std::string(label) +
      " conformance tolerance exceeded");
}

void check_quantitative(Step2ComputeBackend& backend) {
  const Eigen::Index samples = 11;
  const Eigen::Index phenotypes = 16;
  const Eigen::Index covariates = 3;
  const Eigen::Index variants = 5;
  const Eigen::MatrixXd residuals =
    deterministic_matrix(samples, phenotypes, 0.13);
  const Eigen::MatrixXd design =
    deterministic_matrix(samples, covariates, -0.29);
  const Eigen::MatrixXd products =
    residuals.transpose() * design;
  const Eigen::MatrixXd genotypes =
    deterministic_matrix(samples, variants, 0.71);
  const std::vector<unsigned char> dense(variants, 0);
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);

  Step2ComputeTimings timings;
  if(!backend.prepare_quantitative(residuals, design, products,
       observed, true, &timings) || !backend.ready() ||
     backend.uses_packed_hardcalls())
    throw std::runtime_error("complete quantitative preparation failed");

  Eigen::MatrixXd numerators, denominators;
  if(!backend.score_dense_block(genotypes, dense, numerators,
       denominators, &timings))
    throw std::runtime_error("complete quantitative scoring failed");
  const Eigen::MatrixXd design_cross = design.transpose() * genotypes;
  const Eigen::MatrixXd expected_numerators =
    residuals.transpose() * genotypes - products * design_cross;
  Eigen::RowVectorXd expected_denominator =
    genotypes.colwise().squaredNorm();
  expected_denominator.array() -=
    design_cross.array().square().colwise().sum();
  const Eigen::MatrixXd expected_denominators =
    expected_denominator.replicate(phenotypes, 1);
  require_close(numerators, expected_numerators,
    "complete quantitative numerator");
  require_close(denominators, expected_denominators,
    "complete quantitative denominator");
  if(timings.prepared_chromosomes != 1 || timings.scored_blocks != 1 ||
     timings.scored_variants != static_cast<uint64_t>(variants))
    throw std::runtime_error("quantitative timing counters are invalid");

  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype)
    for(Eigen::Index sample = 0; sample < samples; ++sample)
      observed(sample, phenotype) =
        ((sample + 2 * phenotype) % 7) != 0;
  if(!backend.prepare_quantitative(residuals, design, products,
       observed, false, nullptr))
    throw std::runtime_error("missing quantitative preparation failed");
  if(!backend.score_dense_block(genotypes, dense, numerators,
       denominators, nullptr))
    throw std::runtime_error("missing quantitative scoring failed");
  Eigen::MatrixXd expected_missing_numerators(phenotypes, variants);
  Eigen::MatrixXd expected_missing_denominators(phenotypes, variants);
  for(Eigen::Index variant = 0; variant < variants; ++variant) {
    const Eigen::VectorXd coefficient = design_cross.col(variant);
    const Eigen::VectorXd residualized =
      genotypes.col(variant) - design * coefficient;
    for(Eigen::Index phenotype = 0; phenotype < phenotypes;
        ++phenotype) {
      expected_missing_numerators(phenotype, variant) =
        residuals.col(phenotype).dot(residualized);
      double denominator = 0;
      for(Eigen::Index sample = 0; sample < samples; ++sample)
        if(observed(sample, phenotype))
          denominator += residualized(sample) * residualized(sample);
      expected_missing_denominators(phenotype, variant) = denominator;
    }
  }
  require_close(numerators, expected_missing_numerators,
    "missing quantitative numerator");
  require_close(denominators, expected_missing_denominators,
    "missing quantitative denominator");

  const Eigen::MatrixXd narrow = residuals.leftCols(15);
  const Eigen::MatrixXd narrow_products = products.topRows(15);
  const auto narrow_observed = observed.leftCols(15).eval();
  if(backend.prepare_quantitative(narrow, design, narrow_products,
       narrow_observed, true, nullptr) || backend.ready())
    throw std::runtime_error("small quantitative panel bypass failed");
}

void check_binary(Step2ComputeBackend& backend) {
  const Eigen::Index samples = 13;
  const Eigen::Index phenotypes = 4;
  const Eigen::Index covariates = 3;
  const Eigen::Index variants = 6;
  const Eigen::MatrixXd residuals =
    deterministic_matrix(samples, phenotypes, -0.41);
  const Eigen::MatrixXd weights =
    deterministic_matrix(samples, phenotypes, 0.22).array().abs() + 0.5;
  const Eigen::MatrixXd genotypes =
    deterministic_matrix(samples, variants, 0.63);
  std::vector<Eigen::MatrixXd> designs(phenotypes);
  std::vector<Eigen::VectorXd> products(phenotypes);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    designs[phenotype] = deterministic_matrix(samples, covariates,
      0.07 * phenotype);
    products[phenotype] = deterministic_matrix(covariates, 1,
      -0.16 * phenotype);
  }
  const auto observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);
  const auto active =
    Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(phenotypes, true);
  if(!backend.prepare_binary(residuals, weights, designs, products,
       observed, active, nullptr))
    throw std::runtime_error("binary preparation failed");

  const std::vector<unsigned char> dense(variants, 0);
  Eigen::MatrixXd numerators, denominators;
  if(!backend.score_dense_block(genotypes, dense, numerators,
       denominators, nullptr))
    throw std::runtime_error("binary scoring failed");
  Eigen::MatrixXd expected_numerators(phenotypes, variants);
  Eigen::MatrixXd expected_denominators(phenotypes, variants);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    const Eigen::MatrixXd weighted_genotypes =
      genotypes.array().colwise() * weights.col(phenotype).array();
    const Eigen::MatrixXd cross =
      designs[phenotype].transpose() * weighted_genotypes;
    expected_numerators.row(phenotype) =
      (residuals.col(phenotype).transpose() * weighted_genotypes -
       products[phenotype].transpose() * cross);
    expected_denominators.row(phenotype) =
      weighted_genotypes.colwise().squaredNorm() -
      cross.colwise().squaredNorm();
  }
  require_close(numerators, expected_numerators, "binary numerator");
  require_close(denominators, expected_denominators,
    "binary denominator");
}

void check_cox(Step2ComputeBackend& backend) {
  const Eigen::Index samples = 12;
  const Eigen::Index phenotypes = 4;
  const Eigen::Index covariates = 2;
  const Eigen::Index variants = 5;
  const Eigen::MatrixXd genotypes =
    deterministic_matrix(samples, variants, -0.37);
  std::vector<Eigen::VectorXd> score_residuals(phenotypes);
  std::vector<Eigen::MatrixXd> weighted_designs(phenotypes);
  std::vector<Eigen::MatrixXd> projections(phenotypes);
  std::vector<Eigen::VectorXd> projection_scores(phenotypes);
  std::vector<Eigen::MatrixXd> projection_grams(phenotypes);
  Eigen::VectorXd variances(phenotypes);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    score_residuals[phenotype] =
      deterministic_matrix(samples, 1, 0.11 * phenotype);
    weighted_designs[phenotype] =
      deterministic_matrix(samples, covariates, 0.19 * phenotype);
    projections[phenotype] =
      deterministic_matrix(samples, covariates, -0.23 * phenotype);
    projection_scores[phenotype] =
      deterministic_matrix(covariates, 1, 0.31 * phenotype);
    const Eigen::MatrixXd gram_source =
      deterministic_matrix(covariates, covariates, 0.43 * phenotype);
    projection_grams[phenotype] =
      gram_source.transpose() * gram_source;
    variances(phenotype) = 0.8 + 0.1 * phenotype;
  }
  const auto observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);
  const auto active =
    Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(phenotypes, true);
  if(!backend.prepare_cox(score_residuals, weighted_designs,
       projections, projection_scores, projection_grams, variances,
       observed, active, nullptr))
    throw std::runtime_error("Cox preparation failed");

  const std::vector<unsigned char> dense(variants, 0);
  Eigen::MatrixXd numerators, denominators;
  if(!backend.score_dense_block(genotypes, dense, numerators,
       denominators, nullptr))
    throw std::runtime_error("Cox scoring failed");
  Eigen::MatrixXd expected_numerators(phenotypes, variants);
  Eigen::MatrixXd expected_denominators(phenotypes, variants);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    const Eigen::MatrixXd coefficients =
      weighted_designs[phenotype].transpose() * genotypes;
    const Eigen::MatrixXd raw_cross =
      projections[phenotype].transpose() * genotypes;
    expected_numerators.row(phenotype) =
      score_residuals[phenotype].transpose() * genotypes -
      projection_scores[phenotype].transpose() * coefficients;
    for(Eigen::Index variant = 0; variant < variants; ++variant) {
      const Eigen::VectorXd coefficient = coefficients.col(variant);
      expected_denominators(phenotype, variant) = variances(phenotype) *
        (genotypes.col(variant).squaredNorm() -
         2 * coefficient.dot(raw_cross.col(variant)) +
         coefficient.dot(projection_grams[phenotype] * coefficient));
    }
  }
  require_close(numerators, expected_numerators, "Cox numerator");
  require_close(denominators, expected_denominators, "Cox denominator");
}

}  // namespace

int main() {
  try {
    std::unique_ptr<Step2ComputeBackend> backend =
      make_step2_compute_backend("cpu", 0);
    check_quantitative(*backend);
    check_binary(*backend);
    check_cox(*backend);
    std::cout << "Step 2 CPU block scoring conformance passed\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
