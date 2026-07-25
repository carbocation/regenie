/* Deterministic conformance tests for blockwise Step 2 scoring. */

#include "Step2_Compute.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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

void check_capability_invariants(const Step2ComputeBackend& backend,
    const char* label) {
  if(backend.supports_packed_block_pipeline() &&
     !backend.uses_packed_hardcalls())
    throw std::runtime_error(std::string(label) +
      " packed-pipeline capability is inconsistent");
  if(!backend.uses_packed_hardcalls()) {
    if(backend.supports_packed_block_pipeline())
      throw std::runtime_error(std::string(label) +
        " CPU backend unexpectedly supports the packed pipeline");
    if(backend.prefers_loco_prediction_prefetch())
      throw std::runtime_error(std::string(label) +
        " CPU backend unexpectedly prefers LOCO prefetch");
  }
  if(backend.provides_observed_trait_counts() && !backend.ready())
    throw std::runtime_error(std::string(label) +
      " observed-count capability is inconsistent");
}

void check_score_batch_contract() {
  const Eigen::Index phenotypes = 2;
  const Eigen::Index variants = 3;
  Step2ScoreBatch batch;
  if(batch.valid() || batch.trait_counts_valid() ||
     batch.has_variant(phenotypes, 0) || batch.has_score(0, 0))
    throw std::runtime_error("new Step 2 score batch is unexpectedly valid");

  batch.begin_write();
  batch.numerator_output() =
    Eigen::MatrixXd::Zero(phenotypes, variants);
  batch.denominator_output() =
    Eigen::MatrixXd::Zero(phenotypes, variants - 1);
  batch.observed_allele_sum_output() =
    Eigen::MatrixXd::Zero(phenotypes, variants);
  batch.observed_nonmissing_count_output() =
    Eigen::MatrixXd::Zero(phenotypes, variants);
  if(batch.publish(true, phenotypes, variants) || batch.valid() ||
     batch.trait_counts_valid())
    throw std::runtime_error(
      "Step 2 score batch accepted mismatched denominator dimensions");

  batch.denominator_output() =
    Eigen::MatrixXd::Zero(phenotypes, variants);
  batch.observed_nonmissing_count_output() =
    Eigen::MatrixXd::Zero(phenotypes - 1, variants);
  if(!batch.publish(true, phenotypes, variants) || !batch.valid() ||
     batch.trait_counts_valid())
    throw std::runtime_error(
      "Step 2 score batch did not reject mismatched trait-count dimensions");

  batch.observed_nonmissing_count_output() =
    Eigen::MatrixXd::Zero(phenotypes, variants);
  if(!batch.publish(true, phenotypes, variants) ||
     !batch.trait_counts_valid() ||
     !batch.has_variant(phenotypes, variants - 1) ||
     batch.has_variant(phenotypes, variants) ||
     !batch.has_score(phenotypes - 1, variants - 1) ||
     batch.has_score(phenotypes, 0))
    throw std::runtime_error(
      "Step 2 score batch publication contract is invalid");

  batch.reset();
  if(batch.valid() || batch.trait_counts_valid() ||
     batch.has_variant(phenotypes, 0) || batch.has_score(0, 0))
    throw std::runtime_error("Step 2 score batch reset retained valid state");
}

struct PackedHardcallBlock {
  Eigen::MatrixXd genotypes;
  std::vector<std::vector<unsigned char>> packed;
  std::vector<double> missing_means;
  std::vector<unsigned char> flipped;
  std::vector<unsigned char> sparse;
};

PackedHardcallBlock deterministic_packed_hardcalls(
    Eigen::Index samples, Eigen::Index variants) {
  PackedHardcallBlock block;
  block.genotypes.resize(samples, variants);
  block.packed.assign(variants,
    std::vector<unsigned char>((samples + 3) / 4, 0));
  block.missing_means.resize(variants);
  block.flipped.resize(variants);
  block.sparse.assign(variants, 0);

  for(Eigen::Index variant = 0; variant < variants; ++variant) {
    double allele_sum = 0;
    Eigen::Index observed = 0;
    for(Eigen::Index sample = 0; sample < samples; ++sample) {
      const unsigned char code = static_cast<unsigned char>(
        (sample * 3 + variant * 5 + sample * variant) % 4);
      block.packed[variant][sample >> 2] |=
        static_cast<unsigned char>(code << (2 * (sample & 3)));
      if(code != 3) {
        allele_sum += code;
        ++observed;
      }
    }
    if(observed == 0)
      throw std::runtime_error("packed hardcall fixture has no observations");
    block.missing_means[variant] = allele_sum / observed;
    block.flipped[variant] = static_cast<unsigned char>(variant % 2);
    for(Eigen::Index sample = 0; sample < samples; ++sample) {
      const unsigned char code =
        (block.packed[variant][sample >> 2] >>
          (2 * (sample & 3))) & 3;
      const double unflipped = code == 3 ?
        block.missing_means[variant] : static_cast<double>(code);
      block.genotypes(sample, variant) =
        block.flipped[variant] ? 2.0 - unflipped : unflipped;
    }
  }
  return block;
}

void require_observed_counts(
    const PackedHardcallBlock& block,
    const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
      Eigen::Dynamic>>& observed,
    const Eigen::MatrixXd& allele_sums,
    const Eigen::MatrixXd& nonmissing_counts,
    const char* label) {
  Eigen::MatrixXd expected_sums =
    Eigen::MatrixXd::Zero(observed.cols(), block.genotypes.cols());
  Eigen::MatrixXd expected_counts =
    Eigen::MatrixXd::Zero(observed.cols(), block.genotypes.cols());
  for(Eigen::Index variant = 0; variant < block.genotypes.cols(); ++variant)
    for(Eigen::Index phenotype = 0; phenotype < observed.cols();
        ++phenotype)
      for(Eigen::Index sample = 0; sample < observed.rows(); ++sample) {
        if(!observed(sample, phenotype)) continue;
        const unsigned char code =
          (block.packed[variant][sample >> 2] >>
            (2 * (sample & 3))) & 3;
        if(code == 3) continue;
        expected_sums(phenotype, variant) += code;
        expected_counts(phenotype, variant) += 1;
      }
  require_close(allele_sums, expected_sums,
    (std::string(label) + " observed allele sums").c_str());
  require_close(nonmissing_counts, expected_counts,
    (std::string(label) + " observed nonmissing counts").c_str());
}

void score_packed_and_compare(
    Step2ComputeBackend& candidate,
    Step2ComputeBackend& reference,
    const PackedHardcallBlock& block,
    const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
      Eigen::Dynamic>>& observed,
    const Eigen::RowVectorXd* raw_squared_norms,
    Step2ComputeTimings* timings,
    const char* label) {
  Eigen::MatrixXd actual_numerators, actual_denominators;
  Eigen::MatrixXd observed_allele_sums, observed_nonmissing_counts;
  if(!candidate.score_packed_block(block.packed, block.missing_means,
       block.flipped, block.sparse, block.genotypes.rows(),
       actual_numerators, actual_denominators, observed_allele_sums,
       observed_nonmissing_counts, timings))
    throw std::runtime_error(std::string(label) +
      " packed scoring failed");

  Eigen::MatrixXd expected_numerators, expected_denominators;
  if(!reference.score_dense_block(block.genotypes, block.sparse,
       raw_squared_norms, expected_numerators, expected_denominators,
       nullptr))
    throw std::runtime_error(std::string(label) +
      " CPU reference scoring failed");
  require_close(actual_numerators, expected_numerators,
    (std::string(label) + " numerator").c_str());
  require_close(actual_denominators, expected_denominators,
    (std::string(label) + " denominator").c_str());

  if(candidate.provides_observed_trait_counts()) {
    require_observed_counts(block, observed, observed_allele_sums,
      observed_nonmissing_counts, label);
  } else if(observed_allele_sums.size() != 0 ||
            observed_nonmissing_counts.size() != 0) {
    throw std::runtime_error(std::string(label) +
      " returned unexpected observed-trait counts");
  }
}

void check_quantitative(Step2ComputeBackend& backend) {
  if(std::getenv("REGENIE_STEP2_QT_BLOCK_MIN_PHENOTYPES")) {
    if(!should_use_cpu_quantitative_block_scoring(5000, 1, true))
      throw std::runtime_error("quantitative block override is invalid");
  } else {
    if(should_use_cpu_quantitative_block_scoring(5000, 1, true) ||
       should_use_cpu_quantitative_block_scoring(500000, 2, true) ||
       should_use_cpu_quantitative_block_scoring(100000, 8, true) ||
       should_use_cpu_quantitative_block_scoring(50000, 11, true) ||
       !should_use_cpu_quantitative_block_scoring(500000, 4, true) ||
       !should_use_cpu_quantitative_block_scoring(250000, 8, true) ||
       !should_use_cpu_quantitative_block_scoring(50000, 12, true) ||
       should_use_cpu_quantitative_block_scoring(5000, 1, false) ||
       !should_use_cpu_quantitative_block_scoring(5000, 2, false))
      throw std::runtime_error("quantitative block dispatch is invalid");
  }

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
  if(!backend.score_dense_block(genotypes, dense, nullptr, numerators,
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
  if(!backend.score_dense_block(genotypes, dense, nullptr, numerators,
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

  if(!std::getenv("REGENIE_STEP2_QT_BLOCK_MIN_PHENOTYPES")) {
    const Eigen::MatrixXd narrow = residuals.leftCols(11);
    const Eigen::MatrixXd narrow_products = products.topRows(11);
    const auto narrow_observed = observed.leftCols(11).eval();
    if(backend.prepare_quantitative(narrow, design, narrow_products,
         narrow_observed, true, nullptr) || backend.ready())
      throw std::runtime_error("small quantitative panel bypass failed");
  }
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
  if(!backend.score_dense_block(genotypes, dense, nullptr, numerators,
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
  std::vector<Eigen::MatrixXd> projection_transforms(phenotypes);
  std::vector<Eigen::VectorXd> projection_scores(phenotypes);
  std::vector<Eigen::MatrixXd> projection_grams(phenotypes);
  Eigen::VectorXd variances(phenotypes);
  Eigen::MatrixXd common_projection_design =
    deterministic_matrix(samples, covariates, -0.71);
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);
  Eigen::Array<bool, Eigen::Dynamic, 1> active =
    Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(phenotypes, true);
  active(phenotypes - 1) = false;
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    score_residuals[phenotype] =
      deterministic_matrix(samples, 1, 0.11 * phenotype);
    for(Eigen::Index sample = 0; sample < samples; ++sample) {
      observed(sample, phenotype) =
        ((sample + 3 * phenotype) % 11) != 0;
      if(!observed(sample, phenotype)) score_residuals[phenotype](sample) = 0;
    }
    const Eigen::VectorXd weights =
      deterministic_matrix(samples, 1, 0.19 * phenotype).array().abs() + 0.5;
    weighted_designs[phenotype] =
      common_projection_design.array().colwise() * weights.array();
    projection_transforms[phenotype] =
      (common_projection_design.transpose() * weighted_designs[phenotype])
        .colPivHouseholderQr().inverse();
    projections[phenotype] = common_projection_design *
      projection_transforms[phenotype];
    projection_scores[phenotype] = projections[phenotype].transpose() *
      score_residuals[phenotype];
    projection_grams[phenotype] =
      projections[phenotype].transpose() * projections[phenotype];
    variances(phenotype) = 0.8 + 0.1 * phenotype;
  }
  if(!backend.prepare_cox(score_residuals, weighted_designs,
       projections, common_projection_design, projection_transforms,
       projection_scores, projection_grams, variances,
       observed, active, nullptr))
    throw std::runtime_error("Cox preparation failed");

  const std::vector<unsigned char> dense(variants, 0);
  Eigen::MatrixXd numerators, denominators;
  if(!backend.score_dense_block(genotypes, dense, nullptr, numerators,
       denominators, nullptr))
    throw std::runtime_error("Cox scoring failed");
  Eigen::MatrixXd expected_numerators(phenotypes, variants);
  Eigen::MatrixXd expected_denominators(phenotypes, variants);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    if(!active(phenotype)) {
      expected_numerators.row(phenotype).setZero();
      expected_denominators.row(phenotype).setOnes();
      continue;
    }
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

  const Eigen::MatrixXd factored_numerators = numerators;
  const Eigen::MatrixXd factored_denominators = denominators;
  Eigen::RowVectorXd supplied_squared_norms =
    genotypes.colwise().squaredNorm();
  if(!backend.score_dense_block(genotypes, dense,
       &supplied_squared_norms, numerators, denominators, nullptr))
    throw std::runtime_error("Cox supplied-norm scoring failed");
  require_close(numerators, factored_numerators,
    "Cox supplied-norm numerator");
  require_close(denominators, factored_denominators,
    "Cox supplied-norm denominator");

  supplied_squared_norms(0) = -1;
  if(!backend.score_dense_block(genotypes, dense,
       &supplied_squared_norms, numerators, denominators, nullptr))
    throw std::runtime_error("Cox invalid-norm fallback failed");
  require_close(numerators, factored_numerators,
    "Cox invalid-norm fallback numerator");
  require_close(denominators, factored_denominators,
    "Cox invalid-norm fallback denominator");

  const Eigen::MatrixXd empty_design;
  const std::vector<Eigen::MatrixXd> empty_transforms;
  if(!backend.prepare_cox(score_residuals, weighted_designs,
       projections, empty_design, empty_transforms, projection_scores,
       projection_grams, variances, observed, active, nullptr) ||
     !backend.score_dense_block(genotypes, dense, nullptr, numerators,
       denominators, nullptr))
    throw std::runtime_error("legacy Cox scoring failed");
  require_close(numerators, factored_numerators,
    "factored versus legacy Cox numerator");
  require_close(denominators, factored_denominators,
    "factored versus legacy Cox denominator");

  std::vector<Eigen::MatrixXd> malformed_transforms =
    projection_transforms;
  malformed_transforms.pop_back();
  if(backend.prepare_cox(score_residuals, weighted_designs,
       projections, common_projection_design, malformed_transforms,
       projection_scores, projection_grams, variances, observed, active,
       nullptr) || backend.ready())
    throw std::runtime_error("malformed factored Cox preparation accepted");
}

void check_packed_quantitative(Step2ComputeBackend& candidate,
    Step2ComputeBackend& reference, bool allow_workflow_fallback) {
  const Eigen::Index samples = 17;
  const Eigen::Index phenotypes = 16;
  const Eigen::Index covariates = 3;
  const Eigen::Index variants = 7;
  const Eigen::MatrixXd residuals =
    deterministic_matrix(samples, phenotypes, 0.13);
  const Eigen::MatrixXd design =
    deterministic_matrix(samples, covariates, -0.29);
  const Eigen::MatrixXd products = residuals.transpose() * design;
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);
  const PackedHardcallBlock block =
    deterministic_packed_hardcalls(samples, variants);

  Step2ComputeTimings timings;
  const bool candidate_prepared = candidate.prepare_quantitative(
    residuals, design, products, observed, true, &timings);
  if(!reference.prepare_quantitative(residuals, design, products,
       observed, true, nullptr))
    throw std::runtime_error(
      "complete packed quantitative preparation failed");
  if(!candidate_prepared) {
    if(!allow_workflow_fallback || candidate.ready() ||
       std::string(candidate.name()) != "cpu" ||
       candidate.prefers_loco_prediction_prefetch())
      throw std::runtime_error(
        "automatic quantitative workflow fallback is invalid");
    check_capability_invariants(candidate,
      "automatic quantitative fallback backend");
    check_quantitative(reference);
    return;
  }
  check_capability_invariants(candidate,
    "prepared packed quantitative backend");
  if(!candidate.prefers_loco_prediction_prefetch())
    throw std::runtime_error(
      "prepared CUDA backend does not prefer LOCO prefetch");
  score_packed_and_compare(candidate, reference, block, observed,
    nullptr, &timings, "complete packed quantitative");

  candidate.clear();
  if(candidate.ready() || candidate.provides_observed_trait_counts())
    throw std::runtime_error("cleared CUDA backend retained prepared state");
  check_capability_invariants(candidate, "cleared CUDA backend");
  const bool explicit_cuda = std::string(candidate.name()) == "cuda";
  if(candidate.prefers_loco_prediction_prefetch() != explicit_cuda)
    throw std::runtime_error(
      "cleared CUDA LOCO-prefetch preference is invalid");
  Eigen::MatrixXd rejected_numerators, rejected_denominators;
  Eigen::MatrixXd rejected_allele_sums, rejected_nonmissing_counts;
  if(candidate.score_packed_block(block.packed, block.missing_means,
       block.flipped, block.sparse, samples, rejected_numerators,
       rejected_denominators, rejected_allele_sums,
       rejected_nonmissing_counts, nullptr))
    throw std::runtime_error("cleared CUDA backend accepted packed scoring");

  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype)
    for(Eigen::Index sample = 0; sample < samples; ++sample)
      observed(sample, phenotype) =
        ((sample + 2 * phenotype) % 7) != 0;
  if(!candidate.prepare_quantitative(residuals, design, products,
       observed, false, &timings) ||
     !reference.prepare_quantitative(residuals, design, products,
       observed, false, nullptr))
    throw std::runtime_error(
      "missing packed quantitative preparation failed");
  check_capability_invariants(candidate,
    "reprepared packed quantitative backend");
  score_packed_and_compare(candidate, reference, block, observed,
    nullptr, &timings, "missing packed quantitative");

  const uint64_t packed_bytes = static_cast<uint64_t>(
    variants * ((samples + 3) / 4));
  if(timings.prepared_chromosomes != 2 || timings.scored_blocks != 2 ||
     timings.scored_variants != 2 * static_cast<uint64_t>(variants) ||
     timings.packed_upload_bytes != 2 * packed_bytes)
    throw std::runtime_error(
      "packed quantitative timing counters are invalid");
}

void check_packed_binary(Step2ComputeBackend& candidate,
    Step2ComputeBackend& reference) {
  const Eigen::Index samples = 17;
  const Eigen::Index phenotypes = 4;
  const Eigen::Index covariates = 3;
  const Eigen::Index variants = 7;
  const Eigen::MatrixXd residuals =
    deterministic_matrix(samples, phenotypes, -0.41);
  const Eigen::MatrixXd weights =
    deterministic_matrix(samples, phenotypes, 0.22).array().abs() + 0.5;
  std::vector<Eigen::MatrixXd> designs(phenotypes);
  std::vector<Eigen::VectorXd> products(phenotypes);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    designs[phenotype] = deterministic_matrix(samples, covariates,
      0.07 * phenotype);
    products[phenotype] = deterministic_matrix(covariates, 1,
      -0.16 * phenotype);
  }
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype)
    for(Eigen::Index sample = 0; sample < samples; ++sample)
      observed(sample, phenotype) =
        ((2 * sample + phenotype) % 9) != 0;
  const auto active =
    Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(phenotypes, true);
  const PackedHardcallBlock block =
    deterministic_packed_hardcalls(samples, variants);

  Step2ComputeTimings timings;
  if(!candidate.prepare_binary(residuals, weights, designs, products,
       observed, active, &timings) ||
     !reference.prepare_binary(residuals, weights, designs, products,
       observed, active, nullptr))
    throw std::runtime_error("packed binary preparation failed");
  score_packed_and_compare(candidate, reference, block, observed,
    nullptr, &timings, "packed binary");
  if(timings.prepared_chromosomes != 1 || timings.scored_blocks != 1 ||
     timings.scored_variants != static_cast<uint64_t>(variants))
    throw std::runtime_error("packed binary timing counters are invalid");
}

void check_packed_cox(Step2ComputeBackend& candidate,
    Step2ComputeBackend& reference) {
  const Eigen::Index samples = 17;
  const Eigen::Index phenotypes = 4;
  const Eigen::Index covariates = 2;
  const Eigen::Index variants = 7;
  std::vector<Eigen::VectorXd> score_residuals(phenotypes);
  std::vector<Eigen::MatrixXd> weighted_designs(phenotypes);
  std::vector<Eigen::MatrixXd> projections(phenotypes);
  std::vector<Eigen::MatrixXd> projection_transforms(phenotypes);
  std::vector<Eigen::VectorXd> projection_scores(phenotypes);
  std::vector<Eigen::MatrixXd> projection_grams(phenotypes);
  Eigen::VectorXd variances(phenotypes);
  const Eigen::MatrixXd common_projection_design =
    deterministic_matrix(samples, covariates, -0.71);
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> observed =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      samples, phenotypes, true);
  Eigen::Array<bool, Eigen::Dynamic, 1> active =
    Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(phenotypes, true);
  active(phenotypes - 1) = false;
  for(Eigen::Index phenotype = 0; phenotype < phenotypes; ++phenotype) {
    score_residuals[phenotype] =
      deterministic_matrix(samples, 1, 0.11 * phenotype);
    for(Eigen::Index sample = 0; sample < samples; ++sample) {
      observed(sample, phenotype) =
        ((sample + 3 * phenotype) % 11) != 0;
      if(!observed(sample, phenotype))
        score_residuals[phenotype](sample) = 0;
    }
    const Eigen::VectorXd weights =
      deterministic_matrix(samples, 1,
        0.19 * phenotype).array().abs() + 0.5;
    weighted_designs[phenotype] =
      common_projection_design.array().colwise() * weights.array();
    projection_transforms[phenotype] =
      (common_projection_design.transpose() *
        weighted_designs[phenotype]).colPivHouseholderQr().inverse();
    projections[phenotype] = common_projection_design *
      projection_transforms[phenotype];
    projection_scores[phenotype] = projections[phenotype].transpose() *
      score_residuals[phenotype];
    projection_grams[phenotype] =
      projections[phenotype].transpose() * projections[phenotype];
    variances(phenotype) = 0.8 + 0.1 * phenotype;
  }
  const PackedHardcallBlock block =
    deterministic_packed_hardcalls(samples, variants);
  const Eigen::RowVectorXd raw_squared_norms =
    block.genotypes.colwise().squaredNorm();

  Step2ComputeTimings timings;
  if(!candidate.prepare_cox(score_residuals, weighted_designs,
       projections, common_projection_design, projection_transforms,
       projection_scores, projection_grams, variances, observed, active,
       &timings) ||
     !reference.prepare_cox(score_residuals, weighted_designs,
       projections, common_projection_design, projection_transforms,
       projection_scores, projection_grams, variances, observed, active,
       nullptr))
    throw std::runtime_error("packed Cox preparation failed");
  score_packed_and_compare(candidate, reference, block, observed,
    &raw_squared_norms, &timings, "packed Cox");
  if(timings.prepared_chromosomes != 1 || timings.scored_blocks != 1 ||
     timings.scored_variants != static_cast<uint64_t>(variants))
    throw std::runtime_error("packed Cox timing counters are invalid");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    check_score_batch_contract();
    std::string requested_backend = "cpu";
    int gpu_device = 0;
    for(int argument = 1; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if(option == "--backend" && argument + 1 < argc)
        requested_backend = argv[++argument];
      else if(option == "--gpu-device" && argument + 1 < argc)
        gpu_device = std::atoi(argv[++argument]);
      else
        throw std::runtime_error(
          "usage: step2_compute_test "
          "[--backend cpu|auto|cuda] [--gpu-device N]");
    }

    std::unique_ptr<Step2ComputeBackend> backend =
      make_step2_compute_backend(requested_backend, gpu_device);
    check_capability_invariants(*backend, "new Step 2 backend");
    const std::string initial_backend = backend->name();
    if(initial_backend == "cuda" || initial_backend == "auto") {
      if(!backend->uses_packed_hardcalls() ||
         !backend->supports_packed_block_pipeline())
        throw std::runtime_error(
          "CUDA backend omitted a required packed-path capability");
      const bool expected_initial_prefetch = requested_backend == "cuda";
      if(backend->prefers_loco_prediction_prefetch() !=
         expected_initial_prefetch)
        throw std::runtime_error(
          "initial CUDA LOCO-prefetch preference is invalid");
    }
    if(backend->uses_packed_hardcalls()) {
      std::unique_ptr<Step2ComputeBackend> reference =
        make_step2_compute_backend("cpu", 0);
      check_packed_quantitative(*backend, *reference,
        requested_backend == "auto");
      check_packed_binary(*backend, *reference);
      check_packed_cox(*backend, *reference);
    } else {
      check_quantitative(*backend);
      check_binary(*backend);
      check_cox(*backend);
    }
    std::cout << "STEP2_BACKEND_TEST requested=" << requested_backend
      << " active=" << backend->name() << " status=pass\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
