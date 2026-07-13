/* Deterministic conformance and benchmark driver for Step 1 backends. */

#include "Step1_Compute.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string backend = "cpu";
  int device = 0;
  bool benchmark = false;
  bool level1_benchmark = false;
  int blocks = 512;
  int samples = 20000;
  int phenotypes = 10;
  int folds = 5;
  int ridge_parameters = 5;
  int repeats = 3;
  int warmup_repeats = 1;
};

int parse_positive(const char* value, const char* option, bool allow_zero = false) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if(!end || *end != '\0' || parsed < (allow_zero ? 0 : 1) || parsed > 2147483647L)
    throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for(int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if(argument == "--benchmark") options.benchmark = true;
    else if(argument == "--level1-benchmark") options.level1_benchmark = true;
    else if(argument == "--backend" && i + 1 < argc) options.backend = argv[++i];
    else if(argument == "--device" && i + 1 < argc)
      options.device = parse_positive(argv[++i], "--device", true);
    else if(argument == "--blocks" && i + 1 < argc)
      options.blocks = parse_positive(argv[++i], "--blocks");
    else if(argument == "--samples" && i + 1 < argc)
      options.samples = parse_positive(argv[++i], "--samples");
    else if(argument == "--phenotypes" && i + 1 < argc)
      options.phenotypes = parse_positive(argv[++i], "--phenotypes");
    else if(argument == "--folds" && i + 1 < argc)
      options.folds = parse_positive(argv[++i], "--folds");
    else if(argument == "--ridge-parameters" && i + 1 < argc)
      options.ridge_parameters = parse_positive(argv[++i], "--ridge-parameters");
    else if(argument == "--repeats" && i + 1 < argc)
      options.repeats = parse_positive(argv[++i], "--repeats");
    else if(argument == "--warmup-repeats" && i + 1 < argc)
      options.warmup_repeats = parse_positive(argv[++i], "--warmup-repeats");
    else
      throw std::runtime_error("unknown or incomplete option: " + argument);
  }
  if(options.folds > options.samples)
    throw std::runtime_error("--folds must not exceed --samples");
  return options;
}

Eigen::MatrixXd deterministic_matrix(int rows, int columns, double phase) {
  Eigen::MatrixXd result(rows, columns);
  for(int column = 0; column < columns; ++column)
    for(int row = 0; row < rows; ++row) {
      const double index = 1.0 + row + rows * column;
      result(row, column) = std::sin(index * 0.017 + phase) +
                            0.25 * std::cos(index * 0.031 - phase);
    }
  return result;
}

double relative_error(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected) {
  if(actual.rows() != expected.rows() || actual.cols() != expected.cols())
    return std::numeric_limits<double>::max();
  if(actual.size() == 0) return 0;
  const double scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
  return (actual - expected).cwiseAbs().maxCoeff() / scale;
}

bool uses_mixed_gram_products(const Step1ComputeBackend& backend) {
  const char* precision = std::getenv("REGENIE_CUDA_GRAM_PRECISION");
  return std::string(backend.name()) == "cuda" && precision &&
    std::string(precision) == "fp32";
}

double gram_conformance_tolerance(const Step1ComputeBackend& backend,
  double fp64_tolerance) {
  return uses_mixed_gram_products(backend) ? 5e-7 : fp64_tolerance;
}

double factorized_conformance_tolerance(
  const Step1ComputeBackend& backend, double fp64_tolerance) {
  return uses_mixed_gram_products(backend) ? 1e-5 : fp64_tolerance;
}

void reference_preprocess_genotypes(Eigen::MatrixXd& genotypes,
  const Eigen::Ref<const Eigen::MatrixXd>& covariates,
  const Eigen::Ref<const Eigen::VectorXd>& sample_weights,
  double degrees_of_freedom,
  const Eigen::Ref<const Eigen::VectorXd>& row_multipliers,
  Eigen::VectorXd& row_scales) {

  genotypes.array().rowwise() *= sample_weights.transpose().array();
  const Eigen::MatrixXd coefficients = genotypes * covariates;
  genotypes.noalias() -= coefficients * covariates.transpose();
  row_scales = genotypes.rowwise().norm() / std::sqrt(degrees_of_freedom);
  genotypes.array().colwise() /= row_scales.array();
  if(row_multipliers.size())
    genotypes.array().colwise() *= row_multipliers.array();
}

void check_genotype_preprocessing(Step1ComputeBackend& candidate) {
  const int rows = 13;
  // Slightly larger than a 1 MB upload so validation can force the reusable
  // two-slot staging path with REGENIE_CUDA_PINNED_STAGING_MB=1.
  const int samples = 10001;
  Eigen::MatrixXd raw_covariates =
    deterministic_matrix(samples, 3, -0.19);
  Eigen::VectorXd sample_weights = Eigen::VectorXd::Ones(samples);
  for(int sample = 0; sample < samples; sample += 7) {
    sample_weights(sample) = 0;
    raw_covariates.row(sample).setZero();
  }
  const Eigen::MatrixXd covariate_gram =
    raw_covariates.transpose() * raw_covariates;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covariate_solver(
    covariate_gram);
  if(covariate_solver.info() != Eigen::Success)
    throw std::runtime_error(
      "genotype preprocessing covariate setup failed");
  const Eigen::MatrixXd covariates = raw_covariates *
    covariate_solver.eigenvectors() *
    covariate_solver.eigenvalues().array().sqrt().inverse().matrix().asDiagonal() *
    covariate_solver.eigenvectors().transpose();
  const Eigen::VectorXd row_multipliers =
    Eigen::VectorXd::LinSpaced(rows, 0.55, 1.45);
  const double degrees_of_freedom = sample_weights.sum() - covariates.cols();
  Eigen::MatrixXd expected = deterministic_matrix(rows, samples, 0.61);
  Eigen::MatrixXd actual = expected;
  Eigen::VectorXd expected_scales, actual_scales;
  reference_preprocess_genotypes(expected, covariates, sample_weights,
    degrees_of_freedom, row_multipliers, expected_scales);

  Step1ComputeTimings timings;
  timings.upload_ms = timings.preprocess_ms = timings.download_ms = 1.0;
  const bool backend_processed = candidate.preprocess_genotypes(
    actual, covariates, sample_weights, degrees_of_freedom, 1e-12,
    row_multipliers, true, actual_scales, &timings);
  if(!backend_processed)
    reference_preprocess_genotypes(actual, covariates, sample_weights,
      degrees_of_freedom, row_multipliers, actual_scales);

  const double genotype_error = relative_error(actual, expected);
  const double scale_error =
    (actual_scales - expected_scales).cwiseAbs().maxCoeff() /
    std::max(1.0, expected_scales.cwiseAbs().maxCoeff());
  const double preprocessing_tolerance = 3e-11;
  const double gram_tolerance = gram_conformance_tolerance(
    candidate, preprocessing_tolerance);
  if(genotype_error > preprocessing_tolerance ||
     scale_error > preprocessing_tolerance ||
     !std::isfinite(timings.upload_ms) ||
     !std::isfinite(timings.preprocess_ms) ||
     !std::isfinite(timings.download_ms) || timings.upload_ms < 1.0 ||
     timings.preprocess_ms < 1.0 || timings.download_ms < 1.0)
    throw std::runtime_error(
      "genotype preprocessing conformance tolerance exceeded");

  const Eigen::MatrixXd outcomes = deterministic_matrix(samples, 2, -0.47);
  const Eigen::MatrixXd expected_gram = expected * expected.transpose();
  const Eigen::MatrixXd expected_crossproduct = expected * outcomes;
  Eigen::MatrixXd actual_gram, actual_crossproduct;
  Step1ComputeTimings reuse_timings;
  candidate.compute_products(actual, outcomes, actual_gram,
    actual_crossproduct, Step1GramMode::full_product, &reuse_timings);
  const double reuse_gram_error = relative_error(actual_gram, expected_gram);
  const double reuse_crossproduct_error =
    relative_error(actual_crossproduct, expected_crossproduct);
  if(reuse_gram_error > gram_tolerance ||
     reuse_crossproduct_error > preprocessing_tolerance ||
     (backend_processed && reuse_timings.resident_reuse_count == 0) ||
     (!backend_processed && reuse_timings.resident_reuse_count != 0))
    throw std::runtime_error(
      "resident genotype preprocessing reuse conformance failed");

  const int slice_start = 5;
  const int slice_size = 23;
  const Eigen::MatrixXd slice_outcomes =
    outcomes.middleRows(slice_start, slice_size);
  const Eigen::MatrixXd expected_slice =
    expected.middleCols(slice_start, slice_size);
  const Eigen::MatrixXd expected_slice_gram =
    expected_slice * expected_slice.transpose();
  const Eigen::MatrixXd expected_slice_crossproduct =
    expected_slice * slice_outcomes;
  Step1ComputeTimings slice_timings;
  candidate.compute_products(
    actual.middleCols(slice_start, slice_size), slice_outcomes,
    actual_gram, actual_crossproduct, Step1GramMode::full_product,
    &slice_timings);
  if(relative_error(actual_gram, expected_slice_gram) > gram_tolerance ||
     relative_error(actual_crossproduct, expected_slice_crossproduct) >
       preprocessing_tolerance ||
     (backend_processed && slice_timings.resident_reuse_count == 0) ||
     (!backend_processed && slice_timings.resident_reuse_count != 0))
    throw std::runtime_error(
      "resident genotype preprocessing slice reuse conformance failed");
  reuse_timings.resident_reuse_count +=
    slice_timings.resident_reuse_count;

  candidate.release_preprocessed_genotypes();
  Step1ComputeTimings released_timings;
  candidate.compute_products(actual, outcomes, actual_gram,
    actual_crossproduct, Step1GramMode::full_product, &released_timings);
  const double released_gram_error = relative_error(
    actual_gram, expected_gram);
  if(released_timings.resident_reuse_count != 0 ||
     released_gram_error > gram_tolerance ||
     relative_error(actual_crossproduct, expected_crossproduct) >
       preprocessing_tolerance)
    throw std::runtime_error(
      "released genotype preprocessing state was reused");

  Eigen::MatrixXd device_only_actual =
    deterministic_matrix(rows, samples, 0.61);
  const Eigen::MatrixXd device_only_host_input = device_only_actual;
  Eigen::VectorXd device_only_scales;
  Step1ComputeTimings device_only_timings;
  const bool device_only_processed = candidate.preprocess_genotypes(
    device_only_actual, covariates, sample_weights, degrees_of_freedom,
    1e-12, row_multipliers, false, device_only_scales,
    &device_only_timings);
  const double device_only_host_mutation =
    relative_error(device_only_actual, device_only_host_input);
  if(device_only_processed && device_only_host_mutation != 0)
    throw std::runtime_error(
      "device-only genotype preprocessing modified its host input");
  if(!device_only_processed)
    reference_preprocess_genotypes(device_only_actual, covariates,
      sample_weights, degrees_of_freedom, row_multipliers,
      device_only_scales);
  Step1ComputeTimings device_only_reuse_timings;
  candidate.compute_products(device_only_actual, outcomes, actual_gram,
    actual_crossproduct, Step1GramMode::full_product,
    &device_only_reuse_timings);
  const double device_only_gram_error = relative_error(
    actual_gram, expected_gram);
  const double device_only_scale_error =
    (device_only_scales - expected_scales).cwiseAbs().maxCoeff() /
    std::max(1.0, expected_scales.cwiseAbs().maxCoeff());
  if(device_only_gram_error > gram_tolerance ||
     relative_error(actual_crossproduct, expected_crossproduct) >
       preprocessing_tolerance ||
     device_only_scale_error > preprocessing_tolerance ||
     (device_only_processed &&
      device_only_reuse_timings.resident_reuse_count == 0) ||
     (!device_only_processed &&
      device_only_reuse_timings.resident_reuse_count != 0))
    throw std::runtime_error(
      "device-only genotype preprocessing conformance failed");
  candidate.release_preprocessed_genotypes();

  bool rejected_dimensions = false;
  try {
    Eigen::MatrixXd invalid = deterministic_matrix(rows, samples, 0.2);
    Eigen::VectorXd invalid_scales;
    const Eigen::VectorXd invalid_weights =
      Eigen::VectorXd::Ones(samples - 1);
    candidate.preprocess_genotypes(invalid, covariates,
      invalid_weights, degrees_of_freedom, 1e-12,
      row_multipliers, true, invalid_scales);
  } catch(const std::invalid_argument&) {
    rejected_dimensions = true;
  }
  if(!rejected_dimensions)
    throw std::runtime_error(
      "genotype preprocessing accepted incompatible dimensions");

  std::cout << "STEP1_BACKEND_TEST case=genotype_preprocessing"
            << " backend_processed=" << (backend_processed ? 1 : 0)
            << " resident_reuses=" << reuse_timings.resident_reuse_count
            << " device_only_processed=" <<
              (device_only_processed ? 1 : 0)
            << " device_only_reuses=" <<
              device_only_reuse_timings.resident_reuse_count
            << " pinned_staging_uploads=" <<
              (timings.pinned_staging_upload_count +
                device_only_timings.pinned_staging_upload_count)
            << " pinned_staging_bytes=" <<
              (timings.pinned_staging_upload_bytes +
                device_only_timings.pinned_staging_upload_bytes)
            << " genotype_relative_error=" << genotype_error
            << " scale_relative_error=" << scale_error
            << " gram_relative_error=" << reuse_gram_error
            << " released_gram_relative_error=" << released_gram_error
            << " device_only_gram_relative_error=" <<
              device_only_gram_error
            << " status=PASS\n";
}

void accumulate_timings(Step1ComputeTimings& destination,
  const Step1ComputeTimings& source) {
  destination.upload_ms += source.upload_ms;
  destination.preprocess_ms += source.preprocess_ms;
  destination.crossproduct_ms += source.crossproduct_ms;
  destination.gram_ms += source.gram_ms;
  destination.eigensolve_ms += source.eigensolve_ms;
  destination.transform_ms += source.transform_ms;
  destination.ridge_ms += source.ridge_ms;
  destination.download_ms += source.download_ms;
  destination.resident_reuse_count += source.resident_reuse_count;
  destination.pinned_staging_upload_count +=
    source.pinned_staging_upload_count;
  destination.pinned_staging_upload_bytes +=
    source.pinned_staging_upload_bytes;
}

double total_timing_ms(const Step1ComputeTimings& timings) {
  return timings.upload_ms + timings.preprocess_ms +
    timings.crossproduct_ms + timings.gram_ms +
    timings.eigensolve_ms + timings.transform_ms + timings.ridge_ms +
    timings.download_ms;
}

void check_case(Step1ComputeBackend& candidate, Step1ComputeBackend& oracle,
  const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
  const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
  Step1GramMode mode, const char* case_name) {

  Eigen::MatrixXd expected_gram, expected_crossproduct, actual_gram, actual_crossproduct;
  oracle.compute_products(genotypes, phenotypes, expected_gram, expected_crossproduct, mode);
  candidate.compute_products(genotypes, phenotypes, actual_gram, actual_crossproduct, mode);

  const double gram_error = relative_error(actual_gram, expected_gram);
  const double crossproduct_error = relative_error(actual_crossproduct, expected_crossproduct);
  const double symmetry_error = relative_error(actual_gram, actual_gram.transpose());
  const double gram_tolerance = gram_conformance_tolerance(candidate, 5e-12);
  if(gram_error > gram_tolerance || crossproduct_error > 5e-12 ||
     symmetry_error > gram_tolerance) {
    std::cerr << "STEP1_BACKEND_TEST case=" << case_name
              << " gram_relative_error=" << gram_error
              << " crossproduct_relative_error=" << crossproduct_error
              << " symmetry_relative_error=" << symmetry_error
              << " status=FAIL\n";
    throw std::runtime_error("backend conformance tolerance exceeded");
  }

  std::cout << "STEP1_BACKEND_TEST case=" << case_name
            << " gram_relative_error=" << gram_error
            << " crossproduct_relative_error=" << crossproduct_error
            << " symmetry_relative_error=" << symmetry_error
            << " status=PASS\n";
}

void check_eigendecomposition(Step1ComputeBackend& candidate,
  Step1ComputeBackend& oracle) {

  const Eigen::MatrixXd basis = deterministic_matrix(19, 11, 0.35);
  Eigen::MatrixXd symmetric = basis * basis.transpose();
  symmetric.diagonal().array() += Eigen::ArrayXd::LinSpaced(19, 0.25, 2.5);
  const Eigen::MatrixXd rhs = deterministic_matrix(19, 4, -0.6);

  Eigen::MatrixXd expected_vectors, expected_values, expected_transformed;
  Eigen::MatrixXd actual_vectors, actual_values, actual_transformed;
  oracle.eigendecompose_and_transform(symmetric, rhs, expected_vectors,
    expected_values, expected_transformed);

  Step1ComputeTimings timings;
  timings.upload_ms = timings.eigensolve_ms = timings.transform_ms =
    timings.download_ms = 1.0;
  candidate.eigendecompose_and_transform(symmetric, rhs, actual_vectors,
    actual_values, actual_transformed, &timings);

  const Eigen::MatrixXd reconstructed = actual_vectors *
    actual_values.col(0).asDiagonal() * actual_vectors.transpose();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(19, 19);
  const double eigenvalue_error = relative_error(actual_values, expected_values);
  const double reconstruction_error = relative_error(reconstructed, symmetric);
  const double orthogonality_error = relative_error(
    actual_vectors.transpose() * actual_vectors, identity);
  const double transform_error = relative_error(
    actual_transformed, actual_vectors.transpose() * rhs);

  const double lambda = 0.75;
  const Eigen::MatrixXd actual_solution = actual_vectors *
    (actual_values.col(0).array() + lambda).inverse().matrix().asDiagonal() *
    actual_transformed;
  const Eigen::MatrixXd expected_solution = expected_vectors *
    (expected_values.col(0).array() + lambda).inverse().matrix().asDiagonal() *
    expected_transformed;
  const double ridge_solution_error = relative_error(actual_solution, expected_solution);
  const double tolerance = 2e-11;

  if(eigenvalue_error > tolerance || reconstruction_error > tolerance ||
     orthogonality_error > tolerance || transform_error > tolerance ||
     ridge_solution_error > tolerance)
    throw std::runtime_error("eigendecomposition conformance tolerance exceeded");
  if(!std::isfinite(timings.upload_ms) || !std::isfinite(timings.eigensolve_ms) ||
     !std::isfinite(timings.transform_ms) || !std::isfinite(timings.download_ms) ||
     timings.upload_ms < 1.0 || timings.eigensolve_ms < 1.0 ||
     timings.transform_ms < 1.0 || timings.download_ms < 1.0)
    throw std::runtime_error("eigendecomposition returned invalid timings");

  std::cout << "STEP1_BACKEND_TEST case=eigendecomposition"
            << " eigenvalue_relative_error=" << eigenvalue_error
            << " reconstruction_relative_error=" << reconstruction_error
            << " orthogonality_relative_error=" << orthogonality_error
            << " transform_relative_error=" << transform_error
            << " ridge_solution_relative_error=" << ridge_solution_error
            << " status=PASS\n";

  Eigen::MatrixXd empty_matrix(0, 0), empty_rhs(0, 3);
  candidate.eigendecompose_and_transform(empty_matrix, empty_rhs, actual_vectors,
    actual_values, actual_transformed);
  if(actual_vectors.rows() != 0 || actual_vectors.cols() != 0 ||
     actual_values.rows() != 0 || actual_values.cols() != 1 ||
     actual_transformed.rows() != 0 || actual_transformed.cols() != 3)
    throw std::runtime_error("empty eigendecomposition returned incorrect dimensions");
  std::cout << "STEP1_BACKEND_TEST case=empty_eigendecomposition status=PASS\n";

  bool rejected_non_square = false;
  try {
    candidate.eigendecompose_and_transform(deterministic_matrix(3, 2, 0),
      deterministic_matrix(3, 1, 0), actual_vectors, actual_values, actual_transformed);
  } catch(const std::invalid_argument&) {
    rejected_non_square = true;
  }
  if(!rejected_non_square)
    throw std::runtime_error("backend accepted a non-square eigendecomposition matrix");
  std::cout << "STEP1_BACKEND_TEST case=eigendecomposition_validation status=PASS\n";

  Eigen::VectorXd ridge_parameters(3);
  ridge_parameters << 0.1, 1.0, 10.0;
  const Eigen::MatrixXd prediction_genotypes = deterministic_matrix(19, 13, 0.9);
  const Eigen::MatrixXd outcomes = deterministic_matrix(13, 4, -0.25);
  Eigen::MatrixXd expected_predictions, expected_coefficients;
  Eigen::MatrixXd actual_predictions, actual_coefficients;
  const Eigen::MatrixXd no_outcomes(0, 0);

  candidate.eigendecompose_and_transform(symmetric, rhs, actual_vectors,
    actual_values, actual_transformed);

  oracle.ridge_predict(expected_vectors, expected_values, expected_transformed,
    prediction_genotypes, true, ridge_parameters, no_outcomes, false,
    expected_predictions, expected_coefficients);
  candidate.ridge_predict(actual_vectors, actual_values, actual_transformed,
    prediction_genotypes, true, ridge_parameters, no_outcomes, false,
    actual_predictions, actual_coefficients);
  const double prediction_error = relative_error(actual_predictions, expected_predictions);
  const double coefficient_error = relative_error(actual_coefficients, expected_coefficients);
  if(prediction_error > tolerance || coefficient_error > tolerance)
    throw std::runtime_error("k-fold ridge prediction conformance tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=ridge_kfold"
            << " prediction_relative_error=" << prediction_error
            << " coefficient_relative_error=" << coefficient_error
            << " status=PASS\n";

  oracle.ridge_predict(expected_vectors, expected_values, expected_transformed,
    prediction_genotypes, true, ridge_parameters, outcomes, true,
    expected_predictions, expected_coefficients);
  Step1ComputeTimings ridge_timings;
  ridge_timings.upload_ms = ridge_timings.ridge_ms = ridge_timings.download_ms = 1.0;
  candidate.ridge_predict(actual_vectors, actual_values, actual_transformed,
    prediction_genotypes, true, ridge_parameters, outcomes, true,
    actual_predictions, actual_coefficients, &ridge_timings);
  const double loocv_prediction_error = relative_error(actual_predictions, expected_predictions);
  const double loocv_coefficient_error = relative_error(actual_coefficients, expected_coefficients);
  if(loocv_prediction_error > tolerance || loocv_coefficient_error > tolerance)
    throw std::runtime_error("LOOCV ridge prediction conformance tolerance exceeded");
  if(!std::isfinite(ridge_timings.upload_ms) || !std::isfinite(ridge_timings.ridge_ms) ||
     !std::isfinite(ridge_timings.download_ms) || ridge_timings.upload_ms < 1.0 ||
     ridge_timings.ridge_ms < 1.0 || ridge_timings.download_ms < 1.0)
    throw std::runtime_error("ridge prediction returned invalid timings");
  std::cout << "STEP1_BACKEND_TEST case=ridge_loocv"
            << " prediction_relative_error=" << loocv_prediction_error
            << " coefficient_relative_error=" << loocv_coefficient_error
            << " status=PASS\n";

  const Eigen::MatrixXd prediction_design = prediction_genotypes.transpose();
  candidate.ridge_predict(actual_vectors, actual_values, actual_transformed,
    prediction_design, false, ridge_parameters, outcomes, true,
    actual_predictions, actual_coefficients);
  const double design_prediction_error = relative_error(
    actual_predictions, expected_predictions);
  const double design_coefficient_error = relative_error(
    actual_coefficients, expected_coefficients);
  if(design_prediction_error > tolerance || design_coefficient_error > tolerance)
    throw std::runtime_error("design-layout ridge conformance tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=ridge_design_layout"
            << " prediction_relative_error=" << design_prediction_error
            << " coefficient_relative_error=" << design_coefficient_error
            << " status=PASS\n";

  Step1ComputeTimings factorized_ridge_timings;
  factorized_ridge_timings.upload_ms = factorized_ridge_timings.eigensolve_ms =
    factorized_ridge_timings.transform_ms = factorized_ridge_timings.ridge_ms =
    factorized_ridge_timings.download_ms = 1.0;
  candidate.factorize_ridge_system(
    symmetric, rhs, &factorized_ridge_timings);
  candidate.ridge_predict_factorized(
    prediction_genotypes, true, ridge_parameters, outcomes, true,
    actual_predictions, actual_coefficients, &factorized_ridge_timings);
  const double factorized_prediction_error = relative_error(
    actual_predictions, expected_predictions);
  const double factorized_coefficient_error = relative_error(
    actual_coefficients, expected_coefficients);
  const Eigen::MatrixXd scratch_design = deterministic_matrix(29, 5, -0.17);
  const Eigen::MatrixXd scratch_outcomes = deterministic_matrix(29, 2, 0.63);
  Eigen::VectorXd scratch_weights(29);
  for(Eigen::Index row = 0; row < scratch_weights.size(); ++row)
    scratch_weights(row) = 0.2 + std::fmod(0.11 * (row + 1), 0.7);
  Eigen::MatrixXd scratch_gram, scratch_crossproduct;
  candidate.compute_weighted_design_products(scratch_design, scratch_weights,
    scratch_outcomes, scratch_gram, scratch_crossproduct);
  Eigen::MatrixXd scratch_vectors, scratch_values, scratch_transformed;
  candidate.eigendecompose_and_transform(scratch_gram, scratch_crossproduct,
    scratch_vectors, scratch_values, scratch_transformed);
  Eigen::VectorXd scratch_ridge_parameter =
    Eigen::VectorXd::Constant(1, 0.4);
  Eigen::MatrixXd expected_scratch_predictions, expected_scratch_coefficients;
  Eigen::MatrixXd actual_scratch_predictions, actual_scratch_coefficients;
  oracle.ridge_predict(scratch_vectors, scratch_values, scratch_transformed,
    scratch_design, false, scratch_ridge_parameter, no_outcomes, false,
    expected_scratch_predictions, expected_scratch_coefficients);
  candidate.ridge_predict(scratch_vectors, scratch_values, scratch_transformed,
    scratch_design, false, scratch_ridge_parameter, no_outcomes, false,
    actual_scratch_predictions, actual_scratch_coefficients);
  const double scratch_prediction_error = relative_error(
    actual_scratch_predictions, expected_scratch_predictions);
  const double scratch_coefficient_error = relative_error(
    actual_scratch_coefficients, expected_scratch_coefficients);
  candidate.ridge_predict_factorized(
    prediction_design, false, ridge_parameters, outcomes, true,
    actual_predictions, actual_coefficients, &factorized_ridge_timings);
  const double second_factorized_prediction_error = relative_error(
    actual_predictions, expected_predictions);
  const double second_factorized_coefficient_error = relative_error(
    actual_coefficients, expected_coefficients);
  if(factorized_prediction_error > tolerance ||
     factorized_coefficient_error > tolerance ||
     scratch_prediction_error > tolerance ||
     scratch_coefficient_error > tolerance ||
     second_factorized_prediction_error > tolerance ||
     second_factorized_coefficient_error > tolerance)
    throw std::runtime_error("factorized ridge prediction tolerance exceeded");
  if(!std::isfinite(factorized_ridge_timings.upload_ms) ||
     !std::isfinite(factorized_ridge_timings.eigensolve_ms) ||
     !std::isfinite(factorized_ridge_timings.transform_ms) ||
     !std::isfinite(factorized_ridge_timings.ridge_ms) ||
     !std::isfinite(factorized_ridge_timings.download_ms) ||
     factorized_ridge_timings.upload_ms < 1.0 ||
     factorized_ridge_timings.eigensolve_ms < 1.0 ||
     factorized_ridge_timings.transform_ms < 1.0 ||
     factorized_ridge_timings.ridge_ms < 1.0 ||
     factorized_ridge_timings.download_ms < 1.0)
    throw std::runtime_error("factorized ridge prediction returned invalid timings");
  std::cout << "STEP1_BACKEND_TEST case=factorized_ridge_prediction"
            << " first_prediction_relative_error=" << factorized_prediction_error
            << " first_coefficient_relative_error=" << factorized_coefficient_error
            << " nested_prediction_relative_error=" << scratch_prediction_error
            << " nested_coefficient_relative_error=" << scratch_coefficient_error
            << " second_prediction_relative_error="
            << second_factorized_prediction_error
            << " second_coefficient_relative_error="
            << second_factorized_coefficient_error
            << " status=PASS\n";

  Eigen::MatrixXd chunked_ridge_predictions(expected_predictions.rows(),
    expected_predictions.cols());
  double chunked_ridge_coefficient_error = 0.0;
  const Eigen::Index ridge_chunk_boundaries[] = {0, 3, 9, 13};
  for(int chunk = 0; chunk < 3; ++chunk) {
    const Eigen::Index start = ridge_chunk_boundaries[chunk];
    const Eigen::Index count = ridge_chunk_boundaries[chunk + 1] - start;
    candidate.ridge_predict_factorized(
      prediction_design.middleRows(start, count), false, ridge_parameters,
      outcomes.middleRows(start, count), true,
      actual_predictions, actual_coefficients);
    chunked_ridge_predictions.middleRows(start, count) = actual_predictions;
    chunked_ridge_coefficient_error = std::max(
      chunked_ridge_coefficient_error,
      relative_error(actual_coefficients, expected_coefficients));
  }
  const double chunked_ridge_prediction_error = relative_error(
    chunked_ridge_predictions, expected_predictions);
  if(chunked_ridge_prediction_error > tolerance ||
     chunked_ridge_coefficient_error > tolerance)
    throw std::runtime_error("chunked factorized ridge prediction tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=chunked_factorized_ridge_loocv"
            << " prediction_relative_error=" << chunked_ridge_prediction_error
            << " coefficient_relative_error=" << chunked_ridge_coefficient_error
            << " status=PASS\n";

  const Eigen::MatrixXd penalized_design = deterministic_matrix(23, 7, 0.42);
  const Eigen::MatrixXd penalized_outcomes = deterministic_matrix(23, 2, -0.73);
  const Eigen::MatrixXd penalized_gram =
    penalized_design.transpose() * penalized_design;
  const Eigen::MatrixXd penalized_rhs =
    penalized_design.transpose() * penalized_outcomes;
  Eigen::VectorXd penalty_parameters(3);
  penalty_parameters << 0.05, 0.8, 4.0;
  Eigen::VectorXd penalty_multipliers(7);
  penalty_multipliers << 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0;
  Eigen::MatrixXd expected_penalized_predictions(23, 6);
  Eigen::MatrixXd expected_penalized_coefficients(7, 6);
  for(Eigen::Index parameter = 0; parameter < penalty_parameters.size(); ++parameter) {
    Eigen::MatrixXd system = penalized_gram;
    system.diagonal().array() +=
      penalty_parameters(parameter) * penalty_multipliers.array();
    const Eigen::LDLT<Eigen::MatrixXd> factorization(system);
    const Eigen::MatrixXd parameter_coefficients = factorization.solve(penalized_rhs);
    const Eigen::MatrixXd inverse_design_transpose =
      factorization.solve(penalized_design.transpose());
    const Eigen::VectorXd leverage =
      (penalized_design.array() * inverse_design_transpose.transpose().array())
        .rowwise().sum().matrix();
    Eigen::MatrixXd parameter_predictions = penalized_design * parameter_coefficients;
    for(Eigen::Index outcome = 0; outcome < penalized_outcomes.cols(); ++outcome) {
      parameter_predictions.col(outcome).array() -=
        leverage.array() * penalized_outcomes.col(outcome).array();
      parameter_predictions.col(outcome).array() /= 1.0 - leverage.array();
    }
    expected_penalized_predictions.middleCols(parameter * 2, 2) =
      parameter_predictions;
    expected_penalized_coefficients.middleCols(parameter * 2, 2) =
      parameter_coefficients;
  }

  candidate.diagonal_penalty_predict(penalized_gram, penalized_rhs,
    penalized_design, false, penalty_parameters, penalty_multipliers,
    penalized_outcomes, true, actual_predictions, actual_coefficients);
  const double penalized_prediction_error = relative_error(
    actual_predictions, expected_penalized_predictions);
  const double penalized_coefficient_error = relative_error(
    actual_coefficients, expected_penalized_coefficients);
  if(penalized_prediction_error > tolerance || penalized_coefficient_error > tolerance)
    throw std::runtime_error("diagonal-penalty solve conformance tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=diagonal_penalty_loocv"
            << " prediction_relative_error=" << penalized_prediction_error
            << " coefficient_relative_error=" << penalized_coefficient_error
            << " status=PASS\n";

  Eigen::MatrixXd chunked_penalized_predictions(
    expected_penalized_predictions.rows(), expected_penalized_predictions.cols());
  double chunked_penalized_coefficient_error = 0.0;
  const Eigen::Index penalty_chunk_boundaries[] = {0, 4, 15, 23};
  for(int chunk = 0; chunk < 3; ++chunk) {
    const Eigen::Index start = penalty_chunk_boundaries[chunk];
    const Eigen::Index count = penalty_chunk_boundaries[chunk + 1] - start;
    candidate.diagonal_penalty_predict(penalized_gram, penalized_rhs,
      penalized_design.middleRows(start, count), false,
      penalty_parameters, penalty_multipliers,
      penalized_outcomes.middleRows(start, count), true,
      actual_predictions, actual_coefficients);
    chunked_penalized_predictions.middleRows(start, count) = actual_predictions;
    chunked_penalized_coefficient_error = std::max(
      chunked_penalized_coefficient_error,
      relative_error(actual_coefficients, expected_penalized_coefficients));
  }
  const double chunked_penalized_prediction_error = relative_error(
    chunked_penalized_predictions, expected_penalized_predictions);
  if(chunked_penalized_prediction_error > tolerance ||
     chunked_penalized_coefficient_error > tolerance)
    throw std::runtime_error("chunked diagonal-penalty prediction tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=chunked_diagonal_penalty_loocv"
            << " prediction_relative_error="
            << chunked_penalized_prediction_error
            << " coefficient_relative_error="
            << chunked_penalized_coefficient_error
            << " status=PASS\n";

  Eigen::MatrixXd actual_solutions;
  candidate.diagonal_penalty_solve(penalized_gram, penalized_rhs,
    penalty_parameters, penalty_multipliers, actual_solutions);
  const double solve_error = relative_error(
    actual_solutions, expected_penalized_coefficients);
  if(solve_error > tolerance)
    throw std::runtime_error("diagonal-penalty solve-only tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=diagonal_penalty_solve"
            << " solution_relative_error=" << solve_error
            << " status=PASS\n";

  const Eigen::Index reusable_parameter = 1;
  Step1ComputeTimings reusable_timings;
  reusable_timings.upload_ms = reusable_timings.ridge_ms =
    reusable_timings.download_ms = 1.0;
  candidate.factorize_diagonal_penalty(penalized_gram,
    penalty_parameters(reusable_parameter), penalty_multipliers,
    &reusable_timings);
  const Eigen::VectorXd nested_penalty_parameter =
    Eigen::VectorXd::Constant(1, penalty_parameters(0));
  candidate.diagonal_penalty_predict(penalized_gram, penalized_rhs,
    penalized_design, false, nested_penalty_parameter, penalty_multipliers,
    penalized_outcomes, true, actual_predictions, actual_coefficients);
  const double nested_penalty_prediction_error = relative_error(
    actual_predictions, expected_penalized_predictions.leftCols(2));
  const double nested_penalty_coefficient_error = relative_error(
    actual_coefficients, expected_penalized_coefficients.leftCols(2));
  Eigen::MatrixXd interleaved_crossproduct;
  candidate.compute_design_crossproduct(penalized_design, penalized_outcomes,
    interleaved_crossproduct);
  Eigen::MatrixXd reusable_solutions;
  candidate.solve_factorized(penalized_rhs, reusable_solutions,
    &reusable_timings);
  const Eigen::MatrixXd expected_reusable =
    expected_penalized_coefficients.middleCols(reusable_parameter * 2, 2);
  const double reusable_solution_error = relative_error(
    reusable_solutions, expected_reusable);
  Eigen::MatrixXd reusable_system = penalized_gram;
  reusable_system.diagonal().array() += penalty_parameters(reusable_parameter) *
    penalty_multipliers.array();

  const Eigen::VectorXd grouped_residuals =
    deterministic_matrix(23, 1, 0.19).col(0);
  Eigen::VectorXd grouped_weights(23);
  for(Eigen::Index sample = 0; sample < grouped_weights.size(); ++sample)
    grouped_weights(sample) = 0.15 + std::fmod(0.17 * (sample + 1), 0.8);
  Eigen::VectorXi group_offsets(3), group_sizes(3);
  group_offsets << 0, 2, 5;
  group_sizes << 2, 3, 2;
  const Eigen::VectorXd grouped_coefficients = expected_reusable.col(0);
  Eigen::MatrixXd expected_linear_grouped_predictions(23, 3);
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const Eigen::Index offset = group_offsets(group);
    const Eigen::Index count = group_sizes(group);
    expected_linear_grouped_predictions.col(group).noalias() =
      penalized_design.middleCols(offset, count) *
      grouped_coefficients.segment(offset, count);
  }
  Eigen::MatrixXd linear_grouped_predictions;
  candidate.grouped_predict(penalized_design, grouped_coefficients,
    group_offsets, group_sizes, linear_grouped_predictions,
    &reusable_timings);
  const double linear_grouped_prediction_error = relative_error(
    linear_grouped_predictions, expected_linear_grouped_predictions);
  Eigen::MatrixXd chunked_linear_grouped_predictions(23, 3);
  const Eigen::Index grouped_chunk_boundaries[] = {0, 4, 15, 23};
  for(int chunk = 0; chunk < 3; ++chunk) {
    const Eigen::Index start = grouped_chunk_boundaries[chunk];
    const Eigen::Index count = grouped_chunk_boundaries[chunk + 1] - start;
    candidate.grouped_predict(penalized_design.middleRows(start, count),
      grouped_coefficients, group_offsets, group_sizes,
      linear_grouped_predictions, &reusable_timings);
    chunked_linear_grouped_predictions.middleRows(start, count) =
      linear_grouped_predictions;
  }
  const double chunked_linear_grouped_prediction_error = relative_error(
    chunked_linear_grouped_predictions, expected_linear_grouped_predictions);
  if(linear_grouped_prediction_error > tolerance ||
     chunked_linear_grouped_prediction_error > tolerance)
    throw std::runtime_error("grouped linear prediction tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=grouped_linear_prediction"
            << " prediction_relative_error="
            << linear_grouped_prediction_error
            << " chunked_prediction_relative_error="
            << chunked_linear_grouped_prediction_error
            << " status=PASS\n";

  const Eigen::MatrixXd grouped_inverse_design =
    reusable_system.llt().solve(penalized_design.transpose());
  const Eigen::VectorXd grouped_leverage =
    ((penalized_design.array() * grouped_inverse_design.transpose().array())
      .rowwise().sum() * grouped_weights.array()).matrix();
  const Eigen::VectorXd grouped_adjustment =
    (grouped_residuals.array() / (1.0 - grouped_leverage.array())).matrix();
  Eigen::MatrixXd expected_grouped_predictions(23, 3);
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const Eigen::Index offset = group_offsets(group);
    const Eigen::Index count = group_sizes(group);
    expected_grouped_predictions.col(group).noalias() =
      penalized_design.middleCols(offset, count) *
      grouped_coefficients.segment(offset, count);
    expected_grouped_predictions.col(group).array() -=
      grouped_adjustment.array() *
      (penalized_design.middleCols(offset, count).array() *
       grouped_inverse_design.middleRows(offset, count).transpose().array())
        .rowwise().sum();
  }

  Eigen::MatrixXd grouped_predictions;
  candidate.grouped_leave_one_out_predict_factorized(
    penalized_design, grouped_coefficients, grouped_residuals,
    grouped_weights, group_offsets, group_sizes, grouped_predictions,
    &reusable_timings);
  const double grouped_prediction_error = relative_error(
    grouped_predictions, expected_grouped_predictions);
  Eigen::MatrixXd chunked_grouped_predictions(23, 3);
  for(int chunk = 0; chunk < 3; ++chunk) {
    const Eigen::Index start = grouped_chunk_boundaries[chunk];
    const Eigen::Index count = grouped_chunk_boundaries[chunk + 1] - start;
    candidate.grouped_leave_one_out_predict_factorized(
      penalized_design.middleRows(start, count), grouped_coefficients,
      grouped_residuals.segment(start, count),
      grouped_weights.segment(start, count), group_offsets, group_sizes,
      grouped_predictions, &reusable_timings);
    chunked_grouped_predictions.middleRows(start, count) = grouped_predictions;
  }
  const double chunked_grouped_prediction_error = relative_error(
    chunked_grouped_predictions, expected_grouped_predictions);
  if(grouped_prediction_error > tolerance ||
     chunked_grouped_prediction_error > tolerance)
    throw std::runtime_error("grouped reusable LOOCV prediction tolerance exceeded");

  const Eigen::VectorXi no_groups(0);
  candidate.grouped_leave_one_out_predict_factorized(
    penalized_design, grouped_coefficients, grouped_residuals,
    grouped_weights, no_groups, no_groups, grouped_predictions);
  if(grouped_predictions.rows() != 23 || grouped_predictions.cols() != 0)
    throw std::runtime_error("zero-group reusable LOOCV prediction has wrong shape");
  const Eigen::MatrixXd no_grouped_samples(0, 7);
  const Eigen::VectorXd no_grouped_values(0);
  candidate.grouped_leave_one_out_predict_factorized(
    no_grouped_samples, grouped_coefficients, no_grouped_values,
    no_grouped_values, group_offsets, group_sizes, grouped_predictions);
  if(grouped_predictions.rows() != 0 || grouped_predictions.cols() != 3)
    throw std::runtime_error("zero-sample reusable LOOCV prediction has wrong shape");

  bool rejected_bad_group = false;
  bool rejected_negative_group_weight = false;
  try {
    const Eigen::VectorXi bad_offsets = Eigen::VectorXi::Constant(1, 6);
    const Eigen::VectorXi bad_sizes = Eigen::VectorXi::Constant(1, 2);
    candidate.grouped_leave_one_out_predict_factorized(
      penalized_design, grouped_coefficients, grouped_residuals,
      grouped_weights, bad_offsets, bad_sizes, grouped_predictions);
  } catch(const std::invalid_argument&) {
    rejected_bad_group = true;
  }
  try {
    Eigen::VectorXd bad_weights = grouped_weights;
    bad_weights(0) = -1.0;
    candidate.grouped_leave_one_out_predict_factorized(
      penalized_design, grouped_coefficients, grouped_residuals,
      bad_weights, group_offsets, group_sizes, grouped_predictions);
  } catch(const std::invalid_argument&) {
    rejected_negative_group_weight = true;
  }
  if(!rejected_bad_group || !rejected_negative_group_weight)
    throw std::runtime_error("grouped reusable LOOCV validation failed");
  std::cout << "STEP1_BACKEND_TEST case=grouped_reusable_loocv_prediction"
            << " prediction_relative_error=" << grouped_prediction_error
            << " chunked_prediction_relative_error="
            << chunked_grouped_prediction_error
            << " status=PASS\n";

  const Eigen::MatrixXd reusable_rhs = penalized_design.topRows(5).transpose();
  Eigen::MatrixXd second_reusable_solutions;
  candidate.solve_factorized(reusable_rhs, second_reusable_solutions,
    &reusable_timings);
  const Eigen::MatrixXd expected_second_reusable =
    reusable_system.llt().solve(reusable_rhs);
  const double second_reusable_error = relative_error(
    second_reusable_solutions, expected_second_reusable);
  if(reusable_solution_error > tolerance ||
     nested_penalty_prediction_error > tolerance ||
     nested_penalty_coefficient_error > tolerance ||
     second_reusable_error > tolerance)
    throw std::runtime_error("reusable diagonal-penalty solve tolerance exceeded");
  if(!std::isfinite(reusable_timings.upload_ms) ||
     !std::isfinite(reusable_timings.ridge_ms) ||
     !std::isfinite(reusable_timings.download_ms) ||
     reusable_timings.upload_ms < 1.0 || reusable_timings.ridge_ms < 1.0 ||
     reusable_timings.download_ms < 1.0)
    throw std::runtime_error("reusable diagonal-penalty solve returned invalid timings");
  std::cout << "STEP1_BACKEND_TEST case=reusable_diagonal_penalty_solve"
            << " first_solution_relative_error=" << reusable_solution_error
            << " nested_prediction_relative_error="
            << nested_penalty_prediction_error
            << " nested_coefficient_relative_error="
            << nested_penalty_coefficient_error
            << " second_solution_relative_error=" << second_reusable_error
            << " status=PASS\n";

  const Eigen::MatrixXd empty_penalty_gram(0, 0);
  const Eigen::MatrixXd empty_penalty_rhs(0, 2);
  const Eigen::MatrixXd empty_penalty_design(6, 0);
  const Eigen::MatrixXd empty_penalty_outcomes =
    deterministic_matrix(6, 2, 0.31);
  const Eigen::VectorXd empty_penalty_multipliers(0);
  Eigen::VectorXd empty_penalty_parameters(2);
  empty_penalty_parameters << 0.25, 2.0;
  candidate.diagonal_penalty_predict(empty_penalty_gram, empty_penalty_rhs,
    empty_penalty_design, false, empty_penalty_parameters,
    empty_penalty_multipliers, empty_penalty_outcomes, true,
    actual_predictions, actual_coefficients);
  if(actual_predictions.rows() != 6 || actual_predictions.cols() != 4 ||
     actual_coefficients.rows() != 0 || actual_coefficients.cols() != 4 ||
     actual_predictions.cwiseAbs().maxCoeff() != 0)
    throw std::runtime_error(
      "zero-feature diagonal-penalty prediction returned incorrect output");
  candidate.diagonal_penalty_solve(empty_penalty_gram, empty_penalty_rhs,
    empty_penalty_parameters, empty_penalty_multipliers, actual_solutions);
  if(actual_solutions.rows() != 0 || actual_solutions.cols() != 4)
    throw std::runtime_error(
      "zero-feature diagonal-penalty solve returned incorrect output");
  std::cout << "STEP1_BACKEND_TEST case=diagonal_penalty_empty_shapes status=PASS\n";
}

void check_streamed_design_operations(Step1ComputeBackend& candidate) {
  const char* chunk_mb_text = std::getenv("REGENIE_CUDA_CHUNK_MB");
  if(!chunk_mb_text || !*chunk_mb_text) return;

  char* end = nullptr;
  const long chunk_mb = std::strtol(chunk_mb_text, &end, 10);
  if(end == chunk_mb_text || *end != '\0' || chunk_mb <= 0 || chunk_mb > 128)
    return;

  // Make the design slightly larger than the configured backend buffer so
  // both operations must execute at least two row chunks. The cap above keeps
  // this opt-in conformance case modest with a user-supplied value.
  const Eigen::Index max_elements =
    static_cast<Eigen::Index>(chunk_mb) * 1000000 / sizeof(double);
  const Eigen::Index columns = 16;
  const Eigen::Index rows = max_elements / columns + 17;
  Eigen::MatrixXd design = deterministic_matrix(
    static_cast<int>(rows), static_cast<int>(columns), 0.71);
  // deterministic_matrix has only two row frequencies, so sufficiently tall
  // instances have rank at most four regardless of the column count. Add
  // column-specific frequencies so this streaming stress case compares
  // coefficients from a well-conditioned full-rank system instead of
  // amplifying harmless accumulation-order differences in null directions.
  for(Eigen::Index column = 0; column < columns; ++column)
    for(Eigen::Index row = 0; row < rows; ++row)
      design(row, column) += 0.2 * std::sin(
        0.0013 * (row + 1) * (column + 1) + 0.17 * column);
  const Eigen::MatrixXd outcomes = deterministic_matrix(
    static_cast<int>(rows), 2, -0.38);

  const Eigen::MatrixXd expected_crossproduct = design.transpose() * outcomes;
  Eigen::MatrixXd actual_crossproduct;
  candidate.compute_design_crossproduct(
    design, outcomes, actual_crossproduct);
  const double crossproduct_error = relative_error(
    actual_crossproduct, expected_crossproduct);

  const Eigen::MatrixXd expected_gram = design.transpose() * design;
  Eigen::MatrixXd actual_gram, actual_product_crossproduct;
  candidate.compute_design_products(
    design, outcomes, actual_gram, actual_product_crossproduct);
  const double design_gram_error = relative_error(actual_gram, expected_gram);
  const double design_product_crossproduct_error = relative_error(
    actual_product_crossproduct, expected_crossproduct);

  Eigen::VectorXd weights(rows);
  for(Eigen::Index row = 0; row < rows; ++row)
    weights(row) = 0.1 + std::fmod(0.013 * (row + 1), 0.9);
  const Eigen::MatrixXd weighted_design =
    (design.array().colwise() * weights.array()).matrix();
  const Eigen::MatrixXd expected_weighted_gram =
    design.transpose() * weighted_design;
  const Eigen::MatrixXd expected_weighted_crossproduct = design.transpose() *
    (outcomes.array().colwise() * weights.array()).matrix();
  Eigen::MatrixXd actual_weighted_gram, actual_weighted_crossproduct;
  candidate.compute_weighted_design_products(
    design, weights, outcomes, actual_weighted_gram,
    actual_weighted_crossproduct);
  const double weighted_gram_error = relative_error(
    actual_weighted_gram, expected_weighted_gram);
  const double weighted_crossproduct_error = relative_error(
    actual_weighted_crossproduct, expected_weighted_crossproduct);

  const Eigen::MatrixXd genotypes = design.transpose();
  Eigen::MatrixXd actual_l0_gram, actual_l0_crossproduct;
  candidate.compute_products(genotypes, outcomes, actual_l0_gram,
    actual_l0_crossproduct, Step1GramMode::full_product);
  const double l0_full_gram_error = relative_error(
    actual_l0_gram, expected_gram);
  const double l0_full_crossproduct_error = relative_error(
    actual_l0_crossproduct, expected_crossproduct);
  candidate.compute_products(genotypes, outcomes, actual_l0_gram,
    actual_l0_crossproduct, Step1GramMode::selfadjoint_rank_update);
  const double l0_rank_gram_error = relative_error(
    actual_l0_gram, expected_gram);
  const double l0_rank_crossproduct_error = relative_error(
    actual_l0_crossproduct, expected_crossproduct);

  std::unique_ptr<Step1ComputeBackend> oracle =
    make_cpu_step1_compute_backend();
  oracle->compute_products_and_factorize_ridge(genotypes, outcomes,
    Step1GramMode::selfadjoint_rank_update);
  candidate.compute_products_and_factorize_ridge(genotypes, outcomes,
    Step1GramMode::selfadjoint_rank_update);
  Eigen::VectorXd streamed_ridge_parameters(2);
  streamed_ridge_parameters << 0.2, 2.0;
  const Eigen::Index prediction_samples = rows;
  const Eigen::MatrixXd no_loo_outcomes(0, 0);
  Eigen::MatrixXd expected_fused_predictions, expected_fused_coefficients;
  Eigen::MatrixXd actual_fused_predictions, actual_fused_coefficients;
  oracle->ridge_predict_factorized(
    genotypes.leftCols(prediction_samples), true, streamed_ridge_parameters,
    no_loo_outcomes, false, expected_fused_predictions,
    expected_fused_coefficients);
  candidate.ridge_predict_factorized(
    genotypes.leftCols(prediction_samples), true, streamed_ridge_parameters,
    no_loo_outcomes, false, actual_fused_predictions,
    actual_fused_coefficients);
  const double fused_prediction_error = relative_error(
    actual_fused_predictions, expected_fused_predictions);
  const double fused_coefficient_error = relative_error(
    actual_fused_coefficients, expected_fused_coefficients);
  const Eigen::VectorXd streamed_loo_parameter =
    Eigen::VectorXd::Constant(1, 0.7);
  oracle->ridge_predict_factorized(
    genotypes, true, streamed_loo_parameter, outcomes, true,
    expected_fused_predictions, expected_fused_coefficients);
  candidate.ridge_predict_factorized(
    genotypes, true, streamed_loo_parameter, outcomes, true,
    actual_fused_predictions, actual_fused_coefficients);
  const double fused_loo_prediction_error = relative_error(
    actual_fused_predictions, expected_fused_predictions);
  const double fused_loo_coefficient_error = relative_error(
    actual_fused_coefficients, expected_fused_coefficients);

  Eigen::VectorXd streamed_diagonal_parameters(2);
  streamed_diagonal_parameters << 0.3, 1.1;
  const Eigen::VectorXd streamed_diagonal_multipliers =
    Eigen::VectorXd::LinSpaced(columns, 0.25, 1.75);
  Eigen::MatrixXd expected_diagonal_predictions;
  Eigen::MatrixXd expected_diagonal_coefficients;
  Eigen::MatrixXd actual_diagonal_predictions;
  Eigen::MatrixXd actual_diagonal_coefficients;
  oracle->diagonal_penalty_predict(expected_gram, expected_crossproduct,
    design, false, streamed_diagonal_parameters,
    streamed_diagonal_multipliers, no_loo_outcomes, false,
    expected_diagonal_predictions, expected_diagonal_coefficients);
  candidate.diagonal_penalty_predict(expected_gram, expected_crossproduct,
    design, false, streamed_diagonal_parameters,
    streamed_diagonal_multipliers, no_loo_outcomes, false,
    actual_diagonal_predictions, actual_diagonal_coefficients);
  const double diagonal_prediction_error = relative_error(
    actual_diagonal_predictions, expected_diagonal_predictions);
  const double diagonal_coefficient_error = relative_error(
    actual_diagonal_coefficients, expected_diagonal_coefficients);
  oracle->diagonal_penalty_predict(expected_gram, expected_crossproduct,
    design, false, streamed_diagonal_parameters,
    streamed_diagonal_multipliers, outcomes, true,
    expected_diagonal_predictions, expected_diagonal_coefficients);
  candidate.diagonal_penalty_predict(expected_gram, expected_crossproduct,
    design, false, streamed_diagonal_parameters,
    streamed_diagonal_multipliers, outcomes, true,
    actual_diagonal_predictions, actual_diagonal_coefficients);
  const double diagonal_loo_prediction_error = relative_error(
    actual_diagonal_predictions, expected_diagonal_predictions);
  const double diagonal_loo_coefficient_error = relative_error(
    actual_diagonal_coefficients, expected_diagonal_coefficients);

  const Eigen::VectorXd coefficients = deterministic_matrix(
    static_cast<int>(columns), 1, 0.19).col(0);
  Eigen::VectorXi group_offsets(3), group_sizes(3);
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const Eigen::Index start = group * columns / group_offsets.size();
    const Eigen::Index finish =
      (group + 1) * columns / group_offsets.size();
    group_offsets(group) = static_cast<int>(start);
    group_sizes(group) = static_cast<int>(finish - start);
  }

  Eigen::MatrixXd expected_predictions(rows, group_offsets.size());
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const Eigen::Index offset = group_offsets(group);
    const Eigen::Index size = group_sizes(group);
    expected_predictions.col(group).noalias() =
      design.middleCols(offset, size) * coefficients.segment(offset, size);
  }
  Eigen::MatrixXd actual_predictions;
  candidate.grouped_predict(design, coefficients, group_offsets, group_sizes,
    actual_predictions);
  const double prediction_error = relative_error(
    actual_predictions, expected_predictions);

  const double grouped_loo_parameter = 0.5;
  const Eigen::VectorXd grouped_loo_multipliers =
    Eigen::VectorXd::LinSpaced(columns, 0.25, 1.75);
  Eigen::MatrixXd grouped_loo_system = expected_gram;
  grouped_loo_system.diagonal().array() +=
    grouped_loo_parameter * grouped_loo_multipliers.array();
  const Eigen::VectorXd grouped_loo_coefficients =
    grouped_loo_system.llt().solve(expected_crossproduct.col(0));
  oracle->factorize_diagonal_penalty(expected_gram, grouped_loo_parameter,
    grouped_loo_multipliers);
  candidate.factorize_diagonal_penalty(expected_gram, grouped_loo_parameter,
    grouped_loo_multipliers);
  Eigen::MatrixXd expected_streamed_solutions;
  Eigen::MatrixXd actual_streamed_solutions;
  oracle->solve_factorized(genotypes, expected_streamed_solutions);
  candidate.solve_factorized(genotypes, actual_streamed_solutions);
  const double streamed_solve_error = relative_error(
    actual_streamed_solutions, expected_streamed_solutions);
  Eigen::MatrixXd expected_grouped_loo_predictions;
  Eigen::MatrixXd actual_grouped_loo_predictions;
  oracle->grouped_leave_one_out_predict_factorized(
    design, grouped_loo_coefficients, outcomes.col(0), weights,
    group_offsets, group_sizes, expected_grouped_loo_predictions);
  candidate.grouped_leave_one_out_predict_factorized(
    design, grouped_loo_coefficients, outcomes.col(0), weights,
    group_offsets, group_sizes, actual_grouped_loo_predictions);
  const double grouped_loo_prediction_error = relative_error(
    actual_grouped_loo_predictions, expected_grouped_loo_predictions);

  const double tolerance = 5e-12;
  const bool mixed_gram_products = uses_mixed_gram_products(candidate);
  const double l0_gram_tolerance = mixed_gram_products ? 5e-8 : tolerance;
  const double fused_tolerance = mixed_gram_products ? 2e-7 : 2e-11;
  const bool passed = !(crossproduct_error > tolerance ||
     design_gram_error > tolerance ||
     design_product_crossproduct_error > tolerance ||
     weighted_gram_error > tolerance ||
     weighted_crossproduct_error > tolerance ||
     l0_full_gram_error > l0_gram_tolerance ||
     l0_full_crossproduct_error > tolerance ||
     l0_rank_gram_error > l0_gram_tolerance ||
     l0_rank_crossproduct_error > tolerance ||
     fused_prediction_error > fused_tolerance ||
     fused_coefficient_error > fused_tolerance ||
     fused_loo_prediction_error > fused_tolerance ||
     fused_loo_coefficient_error > fused_tolerance ||
     diagonal_prediction_error > 2e-11 ||
     diagonal_coefficient_error > 2e-11 ||
     diagonal_loo_prediction_error > 2e-11 ||
     diagonal_loo_coefficient_error > 2e-11 ||
     streamed_solve_error > 2e-11 ||
     grouped_loo_prediction_error > 2e-11 ||
     prediction_error > tolerance);
  std::cout << "STEP1_BACKEND_TEST case=streamed_design_operations"
            << " chunk_mb=" << chunk_mb
            << " rows=" << rows
            << " columns=" << columns
            << " crossproduct_relative_error=" << crossproduct_error
            << " design_gram_relative_error=" << design_gram_error
            << " weighted_gram_relative_error=" << weighted_gram_error
            << " weighted_crossproduct_relative_error="
            << weighted_crossproduct_error
            << " l0_full_gram_relative_error=" << l0_full_gram_error
            << " l0_rank_gram_relative_error=" << l0_rank_gram_error
            << " fused_prediction_relative_error=" << fused_prediction_error
            << " fused_coefficient_relative_error=" << fused_coefficient_error
            << " fused_loo_prediction_relative_error="
            << fused_loo_prediction_error
            << " fused_loo_coefficient_relative_error="
            << fused_loo_coefficient_error
            << " diagonal_prediction_relative_error="
            << diagonal_prediction_error
            << " diagonal_coefficient_relative_error="
            << diagonal_coefficient_error
            << " diagonal_loo_prediction_relative_error="
            << diagonal_loo_prediction_error
            << " diagonal_loo_coefficient_relative_error="
            << diagonal_loo_coefficient_error
            << " streamed_solve_relative_error=" << streamed_solve_error
            << " grouped_loo_prediction_relative_error="
            << grouped_loo_prediction_error
            << " prediction_relative_error=" << prediction_error
            << " status=" << (passed ? "PASS" : "FAIL") << "\n";
  if(!passed)
    throw std::runtime_error(
      "streamed design operation conformance tolerance exceeded");
}

void run_conformance(Step1ComputeBackend& candidate) {
  std::unique_ptr<Step1ComputeBackend> oracle = make_cpu_step1_compute_backend();
  const Eigen::MatrixXd genotypes = deterministic_matrix(17, 43, 0.2);
  const Eigen::MatrixXd phenotypes = deterministic_matrix(43, 5, -0.4);

  bool rejected_unfactorized_ridge = false;
  bool rejected_unfactorized_cholesky = false;
  bool rejected_unfactorized_grouped_loo = false;
  Eigen::MatrixXd state_predictions, state_coefficients, state_solutions;
  const Eigen::MatrixXd state_prediction_matrix = deterministic_matrix(3, 2, 0.1);
  const Eigen::MatrixXd state_outcomes(0, 0);
  const Eigen::VectorXd state_parameters = Eigen::VectorXd::Ones(1);
  try {
    candidate.ridge_predict_factorized(state_prediction_matrix, false,
      state_parameters, state_outcomes, false,
      state_predictions, state_coefficients);
  } catch(const std::runtime_error&) {
    rejected_unfactorized_ridge = true;
  }
  try {
    candidate.solve_factorized(Eigen::MatrixXd::Ones(2, 1), state_solutions);
  } catch(const std::runtime_error&) {
    rejected_unfactorized_cholesky = true;
  }
  try {
    const Eigen::VectorXd state_coefficients_vector = Eigen::VectorXd::Ones(2);
    const Eigen::VectorXd state_residuals = Eigen::VectorXd::Ones(3);
    const Eigen::VectorXd state_weights = Eigen::VectorXd::Ones(3);
    const Eigen::VectorXi state_group_offsets = Eigen::VectorXi::Zero(1);
    const Eigen::VectorXi state_group_sizes = Eigen::VectorXi::Constant(1, 2);
    candidate.grouped_leave_one_out_predict_factorized(
      state_prediction_matrix, state_coefficients_vector, state_residuals,
      state_weights, state_group_offsets, state_group_sizes,
      state_predictions);
  } catch(const std::runtime_error&) {
    rejected_unfactorized_grouped_loo = true;
  }
  if(!rejected_unfactorized_ridge || !rejected_unfactorized_cholesky ||
     !rejected_unfactorized_grouped_loo)
    throw std::runtime_error("backend accepted a reusable solve before factorization");
  std::cout << "STEP1_BACKEND_TEST case=reusable_state_validation status=PASS\n";

  check_genotype_preprocessing(candidate);

  check_case(candidate, *oracle, genotypes, phenotypes,
    Step1GramMode::full_product, "contiguous_full_product");
  check_case(candidate, *oracle, genotypes, phenotypes,
    Step1GramMode::selfadjoint_rank_update, "contiguous_rank_update");

  Eigen::MatrixXd expected_design_gram, expected_design_crossproduct;
  Eigen::MatrixXd actual_design_gram, actual_design_crossproduct;
  oracle->compute_design_products(genotypes.transpose(), phenotypes,
    expected_design_gram, expected_design_crossproduct);
  candidate.compute_design_products(genotypes.transpose(), phenotypes,
    actual_design_gram, actual_design_crossproduct);
  const double design_gram_error = relative_error(actual_design_gram, expected_design_gram);
  const double design_crossproduct_error = relative_error(
    actual_design_crossproduct, expected_design_crossproduct);
  if(design_gram_error > 5e-12 || design_crossproduct_error > 5e-12)
    throw std::runtime_error("design product conformance tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=design_products"
            << " gram_relative_error=" << design_gram_error
            << " crossproduct_relative_error=" << design_crossproduct_error
            << " status=PASS\n";

  Eigen::MatrixXd crossproduct_only;
  candidate.compute_design_crossproduct(
    genotypes.transpose(), phenotypes, crossproduct_only);
  const double crossproduct_only_error = relative_error(
    crossproduct_only, expected_design_crossproduct);
  const Eigen::MatrixXd no_crossproduct_outcomes(43, 0);
  candidate.compute_design_crossproduct(
    genotypes.transpose(), no_crossproduct_outcomes, crossproduct_only);
  if(crossproduct_only_error > 5e-12 ||
     crossproduct_only.rows() != genotypes.rows() ||
     crossproduct_only.cols() != 0)
    throw std::runtime_error("design-only crossproduct tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=design_crossproduct"
            << " crossproduct_relative_error=" << crossproduct_only_error
            << " status=PASS\n";

  Eigen::VectorXd weights(43);
  for(Eigen::Index row = 0; row < weights.size(); ++row)
    weights(row) = 0.05 + std::fmod(0.37 * (row + 1), 1.75);
  Eigen::MatrixXd expected_weighted_gram, expected_weighted_crossproduct;
  Eigen::MatrixXd actual_weighted_gram, actual_weighted_crossproduct;
  oracle->compute_weighted_design_products(genotypes.transpose(), weights, phenotypes,
    expected_weighted_gram, expected_weighted_crossproduct);
  candidate.compute_weighted_design_products(genotypes.transpose(), weights, phenotypes,
    actual_weighted_gram, actual_weighted_crossproduct);
  const double weighted_gram_error = relative_error(
    actual_weighted_gram, expected_weighted_gram);
  const double weighted_crossproduct_error = relative_error(
    actual_weighted_crossproduct, expected_weighted_crossproduct);
  if(weighted_gram_error > 5e-12 || weighted_crossproduct_error > 5e-12)
    throw std::runtime_error("weighted design product conformance tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=weighted_design_products"
            << " gram_relative_error=" << weighted_gram_error
            << " crossproduct_relative_error=" << weighted_crossproduct_error
            << " status=PASS\n";

  const Eigen::MatrixXd zero_weighted_outcomes(43, 0);
  oracle->compute_weighted_design_products(genotypes.transpose(), weights,
    zero_weighted_outcomes, expected_weighted_gram,
    expected_weighted_crossproduct);
  candidate.compute_weighted_design_products(genotypes.transpose(), weights,
    zero_weighted_outcomes, actual_weighted_gram,
    actual_weighted_crossproduct);
  const double zero_outcome_weighted_gram_error = relative_error(
    actual_weighted_gram, expected_weighted_gram);
  if(zero_outcome_weighted_gram_error > 5e-12 ||
     actual_weighted_crossproduct.rows() != genotypes.rows() ||
     actual_weighted_crossproduct.cols() != 0)
    throw std::runtime_error(
      "zero-outcome weighted design product conformance tolerance exceeded");
  std::cout << "STEP1_BACKEND_TEST case=weighted_design_zero_outcomes"
            << " gram_relative_error=" << zero_outcome_weighted_gram_error
            << " status=PASS\n";

  Eigen::VectorXd fused_ridge_parameters(3);
  fused_ridge_parameters << 0.1, 1.0, 10.0;
  const Eigen::MatrixXd no_outcomes(0, 0);
  Eigen::MatrixXd expected_fused_predictions, expected_fused_coefficients;
  Eigen::MatrixXd actual_fused_predictions, actual_fused_coefficients;
  oracle->factorize_ridge_system(expected_design_gram, expected_design_crossproduct);
  oracle->ridge_predict_factorized(genotypes, true, fused_ridge_parameters,
    no_outcomes, false, expected_fused_predictions, expected_fused_coefficients);
  Step1ComputeTimings fused_timings;
  fused_timings.upload_ms = fused_timings.crossproduct_ms =
    fused_timings.gram_ms = fused_timings.eigensolve_ms =
    fused_timings.transform_ms = fused_timings.ridge_ms =
    fused_timings.download_ms = 1.0;
  candidate.compute_products_and_factorize_ridge(
    genotypes, phenotypes, Step1GramMode::selfadjoint_rank_update,
    &fused_timings);
  candidate.compute_design_products(genotypes.transpose(), phenotypes,
    actual_design_gram, actual_design_crossproduct);
  Eigen::MatrixXd interleaved_vectors, interleaved_values;
  Eigen::MatrixXd interleaved_transformed;
  candidate.eigendecompose_and_transform(expected_design_gram,
    expected_design_crossproduct, interleaved_vectors, interleaved_values,
    interleaved_transformed);
  candidate.ridge_predict_factorized(genotypes, true, fused_ridge_parameters,
    no_outcomes, false, actual_fused_predictions, actual_fused_coefficients,
    &fused_timings);
  const double fused_prediction_error = relative_error(
    actual_fused_predictions, expected_fused_predictions);
  const double fused_coefficient_error = relative_error(
    actual_fused_coefficients, expected_fused_coefficients);
  const double fused_tolerance = factorized_conformance_tolerance(
    candidate, 2e-11);
  if(fused_prediction_error > fused_tolerance ||
     fused_coefficient_error > fused_tolerance)
    throw std::runtime_error("fused ridge pipeline tolerance exceeded");
  if(!std::isfinite(fused_timings.upload_ms) ||
     !std::isfinite(fused_timings.crossproduct_ms) ||
     !std::isfinite(fused_timings.gram_ms) ||
     !std::isfinite(fused_timings.eigensolve_ms) ||
     !std::isfinite(fused_timings.transform_ms) ||
     !std::isfinite(fused_timings.ridge_ms) ||
     !std::isfinite(fused_timings.download_ms) ||
     fused_timings.upload_ms < 1.0 || fused_timings.crossproduct_ms < 1.0 ||
     fused_timings.gram_ms < 1.0 || fused_timings.eigensolve_ms < 1.0 ||
     fused_timings.transform_ms < 1.0 || fused_timings.ridge_ms < 1.0 ||
     fused_timings.download_ms < 1.0)
    throw std::runtime_error("fused ridge pipeline returned invalid timings");
  std::cout << "STEP1_BACKEND_TEST case=fused_ridge_pipeline"
            << " prediction_relative_error=" << fused_prediction_error
            << " coefficient_relative_error=" << fused_coefficient_error
            << " status=PASS\n";

  const Eigen::VectorXd empty_ridge_parameters =
    (Eigen::VectorXd(2) << 0.25, 2.0).finished();
  const Eigen::MatrixXd zero_feature_genotypes(0, 7);
  const Eigen::MatrixXd zero_feature_phenotypes =
    deterministic_matrix(7, 3, 0.14);
  candidate.compute_products_and_factorize_ridge(
    zero_feature_genotypes, zero_feature_phenotypes,
    Step1GramMode::selfadjoint_rank_update);
  candidate.ridge_predict_factorized(Eigen::MatrixXd(0, 5), true,
    empty_ridge_parameters, no_outcomes, false,
    state_predictions, state_coefficients);
  if(state_predictions.rows() != 5 || state_predictions.cols() != 6 ||
     state_coefficients.rows() != 0 || state_coefficients.cols() != 6 ||
     state_predictions.cwiseAbs().maxCoeff() != 0)
    throw std::runtime_error("zero-feature fused ridge returned incorrect output");

  const Eigen::MatrixXd zero_sample_genotypes(5, 0);
  const Eigen::MatrixXd zero_sample_phenotypes(0, 2);
  candidate.compute_products_and_factorize_ridge(
    zero_sample_genotypes, zero_sample_phenotypes,
    Step1GramMode::selfadjoint_rank_update);
  candidate.ridge_predict_factorized(deterministic_matrix(5, 4, -0.5), true,
    empty_ridge_parameters, no_outcomes, false,
    state_predictions, state_coefficients);
  if(state_predictions.rows() != 4 || state_predictions.cols() != 4 ||
     state_coefficients.rows() != 5 || state_coefficients.cols() != 4 ||
     state_predictions.cwiseAbs().maxCoeff() > 1e-14 ||
     state_coefficients.cwiseAbs().maxCoeff() > 1e-14)
    throw std::runtime_error("zero-sample fused ridge returned incorrect output");
  std::cout << "STEP1_BACKEND_TEST case=fused_ridge_empty_shapes status=PASS\n";

  Eigen::MatrixXd padded_phenotypes = deterministic_matrix(49, 5, -0.4);
  padded_phenotypes.middleRows(3, 43) = phenotypes;
  check_case(candidate, *oracle, genotypes, padded_phenotypes.middleRows(3, 43),
    Step1GramMode::full_product, "strided_phenotypes");

  Eigen::MatrixXd padded_genotypes = deterministic_matrix(23, 43, 0.2);
  padded_genotypes.middleRows(2, 17) = genotypes;
  check_case(candidate, *oracle, padded_genotypes.middleRows(2, 17), phenotypes,
    Step1GramMode::selfadjoint_rank_update, "strided_genotypes");

  check_case(candidate, *oracle,
    deterministic_matrix(1, 1, 0.7), deterministic_matrix(1, 1, -0.8),
    Step1GramMode::selfadjoint_rank_update, "scalar");
  check_case(candidate, *oracle,
    deterministic_matrix(31, 7, 0.3), deterministic_matrix(7, 3, -0.1),
    Step1GramMode::full_product, "more_blocks_than_samples");

  const Eigen::MatrixXd no_phenotypes(43, 0);
  check_case(candidate, *oracle, genotypes, no_phenotypes,
    Step1GramMode::selfadjoint_rank_update, "zero_phenotypes");
  const Eigen::MatrixXd no_samples_genotypes(17, 0);
  const Eigen::MatrixXd no_samples_phenotypes(0, 5);
  check_case(candidate, *oracle, no_samples_genotypes, no_samples_phenotypes,
    Step1GramMode::full_product, "zero_samples");
  const Eigen::MatrixXd no_blocks(0, 43);
  check_case(candidate, *oracle, no_blocks, phenotypes,
    Step1GramMode::full_product, "zero_blocks");

  Eigen::MatrixXd gram, crossproduct;
  Step1ComputeTimings timings;
  timings.upload_ms = timings.crossproduct_ms = timings.gram_ms = timings.download_ms = 1.0;
  candidate.compute_products(genotypes, phenotypes, gram, crossproduct,
    Step1GramMode::selfadjoint_rank_update, &timings);
  if(!std::isfinite(timings.upload_ms) || !std::isfinite(timings.crossproduct_ms) ||
     !std::isfinite(timings.gram_ms) || !std::isfinite(timings.download_ms) ||
     timings.upload_ms < 1.0 || timings.crossproduct_ms < 1.0 ||
     timings.gram_ms < 1.0 || timings.download_ms < 1.0)
    throw std::runtime_error("backend returned invalid or non-accumulating timings");
  std::cout << "STEP1_BACKEND_TEST case=timing_accumulation status=PASS\n";

  bool rejected_mismatch = false;
  try {
    candidate.compute_products(genotypes, deterministic_matrix(42, 5, 0.0),
      gram, crossproduct, Step1GramMode::full_product);
  } catch(const std::invalid_argument&) {
    rejected_mismatch = true;
  }
  if(!rejected_mismatch)
    throw std::runtime_error("backend accepted incompatible matrix dimensions");
  std::cout << "STEP1_BACKEND_TEST case=dimension_mismatch status=PASS\n";

  check_eigendecomposition(candidate, *oracle);
  check_streamed_design_operations(candidate);
}

void run_benchmark(Step1ComputeBackend& backend, const Options& options) {
  const Eigen::MatrixXd genotypes = deterministic_matrix(options.blocks, options.samples, 0.1);
  const Eigen::MatrixXd phenotypes = deterministic_matrix(options.samples, options.phenotypes, -0.2);
  Eigen::VectorXd ridge_parameters(5);
  ridge_parameters << 0.01, 0.25, 0.5, 0.75, 0.99;
  Eigen::MatrixXd predictions, coefficients;
  Step1ComputeTimings totals;
  double wall_ms = 0;

  const auto run_iteration = [&] (Step1ComputeTimings* timings) {
    const Clock::time_point start = Clock::now();
    backend.compute_products_and_factorize_ridge(genotypes, phenotypes,
      Step1GramMode::selfadjoint_rank_update, timings);
    backend.ridge_predict_factorized(genotypes, true, ridge_parameters,
      phenotypes, true, predictions, coefficients, timings);
    return std::chrono::duration<double, std::milli>(
      Clock::now() - start).count();
  };

  double warmup_wall_ms = 0;
  double first_warmup_wall_ms = 0;
  for(int repeat = 0; repeat < options.warmup_repeats; ++repeat) {
    Step1ComputeTimings ignored_timings;
    const double iteration_wall_ms = run_iteration(&ignored_timings);
    if(repeat == 0) first_warmup_wall_ms = iteration_wall_ms;
    warmup_wall_ms += iteration_wall_ms;
  }

  for(int repeat = 0; repeat < options.repeats; ++repeat) {
    Step1ComputeTimings timings;
    wall_ms += run_iteration(&timings);
    accumulate_timings(totals, timings);
  }

  const double divisor = options.repeats;
  const double steady_wall_ms = wall_ms / divisor;
  const double accounted_ms = total_timing_ms(totals) / divisor;
  std::cout << "STEP1_BACKEND_BENCHMARK backend=" << backend.name()
            << " blocks=" << options.blocks
            << " samples=" << options.samples
            << " phenotypes=" << options.phenotypes
            << " warmup_repeats=" << options.warmup_repeats
            << " first_warmup_wall_ms=" << first_warmup_wall_ms
            << " mean_warmup_wall_ms=" <<
              warmup_wall_ms / options.warmup_repeats
            << " repeats=" << options.repeats
            << " wall_ms=" << steady_wall_ms
            << " steady_wall_ms=" << steady_wall_ms
            << " accounted_ms=" << accounted_ms
            << " unaccounted_ms=" << steady_wall_ms - accounted_ms
            << " upload_ms=" << totals.upload_ms / divisor
            << " crossproduct_ms=" << totals.crossproduct_ms / divisor
            << " gram_ms=" << totals.gram_ms / divisor
            << " eigensolve_ms=" << totals.eigensolve_ms / divisor
            << " transform_ms=" << totals.transform_ms / divisor
            << " ridge_ms=" << totals.ridge_ms / divisor
            << " download_ms=" << totals.download_ms / divisor << "\n";
}

void run_nonlinear_benchmark(Step1ComputeBackend& backend,
  const Options& options) {

  const Eigen::MatrixXd design =
    deterministic_matrix(options.samples, options.blocks, 0.43);
  const Eigen::MatrixXd outcomes =
    deterministic_matrix(options.samples, options.phenotypes, -0.27);
  Eigen::VectorXd weights(options.samples);
  for(Eigen::Index row = 0; row < weights.size(); ++row)
    weights(row) = 0.1 + std::fmod(0.013 * (row + 1), 0.9);
  Eigen::VectorXd ridge_parameters(3);
  ridge_parameters << 0.05, 0.5, 5.0;
  const Eigen::VectorXd penalty_multipliers =
    Eigen::VectorXd::LinSpaced(options.blocks, 0.25, 1.75);
  const int solve_columns = std::min(options.samples, 2048);
  const Eigen::MatrixXd reusable_rhs =
    design.topRows(solve_columns).transpose();
  Eigen::VectorXi group_offsets(3), group_sizes(3);
  for(Eigen::Index group = 0; group < group_offsets.size(); ++group) {
    const int start = static_cast<int>(group * options.blocks /
      group_offsets.size());
    const int end = static_cast<int>((group + 1) * options.blocks /
      group_offsets.size());
    group_offsets(group) = start;
    group_sizes(group) = end - start;
  }

  Eigen::MatrixXd gram, crossproduct, predictions, coefficients,
    reusable_solution, grouped_predictions, crossproduct_only;
  Step1ComputeTimings totals;
  double wall_ms = 0;
  const Eigen::MatrixXd no_outcomes(0, 0);

  const auto run_iteration = [&] (Step1ComputeTimings* timings) {
    const Clock::time_point start = Clock::now();
    backend.compute_weighted_design_products(
      design, weights, outcomes, gram, crossproduct, timings);
    backend.compute_design_crossproduct(
      design, outcomes, crossproduct_only, timings);
    backend.diagonal_penalty_predict(
      gram, crossproduct, design, false, ridge_parameters,
      penalty_multipliers, no_outcomes, false,
      predictions, coefficients, timings);
    backend.factorize_diagonal_penalty(
      gram, ridge_parameters(1), penalty_multipliers, timings);
    backend.solve_factorized(reusable_rhs, reusable_solution, timings);
    backend.grouped_predict(design.topRows(solve_columns),
      reusable_solution.col(0), group_offsets, group_sizes,
      grouped_predictions, timings);
    backend.grouped_leave_one_out_predict_factorized(
      design.topRows(solve_columns), reusable_solution.col(0),
      outcomes.topRows(solve_columns).col(0), weights.head(solve_columns),
      group_offsets, group_sizes, grouped_predictions, timings);
    return std::chrono::duration<double, std::milli>(
      Clock::now() - start).count();
  };

  double warmup_wall_ms = 0;
  double first_warmup_wall_ms = 0;
  for(int repeat = 0; repeat < options.warmup_repeats; ++repeat) {
    Step1ComputeTimings ignored_timings;
    const double iteration_wall_ms = run_iteration(&ignored_timings);
    if(repeat == 0) first_warmup_wall_ms = iteration_wall_ms;
    warmup_wall_ms += iteration_wall_ms;
  }

  for(int repeat = 0; repeat < options.repeats; ++repeat) {
    Step1ComputeTimings timings;
    wall_ms += run_iteration(&timings);
    accumulate_timings(totals, timings);
  }

  const double divisor = options.repeats;
  const double steady_wall_ms = wall_ms / divisor;
  const double accounted_ms = total_timing_ms(totals) / divisor;
  std::cout << "STEP1_BACKEND_BENCHMARK_NONLINEAR backend=" << backend.name()
            << " features=" << options.blocks
            << " samples=" << options.samples
            << " outcomes=" << options.phenotypes
            << " parameters=" << ridge_parameters.size()
            << " groups=" << group_offsets.size()
            << " solve_columns=" << solve_columns
            << " warmup_repeats=" << options.warmup_repeats
            << " first_warmup_wall_ms=" << first_warmup_wall_ms
            << " mean_warmup_wall_ms=" <<
              warmup_wall_ms / options.warmup_repeats
            << " repeats=" << options.repeats
            << " wall_ms=" << steady_wall_ms
            << " steady_wall_ms=" << steady_wall_ms
            << " accounted_ms=" << accounted_ms
            << " unaccounted_ms=" << steady_wall_ms - accounted_ms
            << " upload_ms=" << totals.upload_ms / divisor
            << " crossproduct_ms=" << totals.crossproduct_ms / divisor
            << " gram_ms=" << totals.gram_ms / divisor
            << " ridge_ms=" << totals.ridge_ms / divisor
            << " download_ms=" << totals.download_ms / divisor << "\n";
}

void run_level1_benchmark(Step1ComputeBackend& backend,
  const Options& options) {

  std::vector<Eigen::MatrixXd> fold_designs;
  std::vector<Eigen::MatrixXd> fold_outcomes;
  fold_designs.reserve(options.folds);
  fold_outcomes.reserve(options.folds);
  int assigned_samples = 0;
  for(int fold = 0; fold < options.folds; ++fold) {
    const int fold_samples = options.samples / options.folds +
      (fold < options.samples % options.folds ? 1 : 0);
    fold_designs.push_back(deterministic_matrix(
      fold_samples, options.blocks, 0.19 + 0.07 * fold));
    fold_outcomes.push_back(deterministic_matrix(
      fold_samples, options.phenotypes, -0.31 - 0.05 * fold));
    assigned_samples += fold_samples;
  }
  if(assigned_samples != options.samples)
    throw std::runtime_error("Level 1 benchmark fold construction failed");

  const Eigen::VectorXd ridge_parameters = Eigen::VectorXd::LinSpaced(
    options.ridge_parameters, 0.01, 1.0);
  const Eigen::MatrixXd no_outcomes(0, 0);
  std::vector<Eigen::MatrixXd> fold_grams(options.folds);
  std::vector<Eigen::MatrixXd> fold_crossproducts(options.folds);
  Eigen::MatrixXd gram_sum, crossproduct_sum, training_gram,
    training_crossproduct, predictions, coefficients;
  Step1ComputeTimings totals;
  double wall_ms = 0;

  const auto run_iteration = [&] (Step1ComputeTimings* timings) {
    const Clock::time_point start = Clock::now();
    for(int phenotype = 0; phenotype < options.phenotypes; ++phenotype) {
      gram_sum.setZero(options.blocks, options.blocks);
      crossproduct_sum.setZero(options.blocks, 1);
      for(int fold = 0; fold < options.folds; ++fold) {
        backend.compute_design_products(
          fold_designs[fold], fold_outcomes[fold].col(phenotype),
          fold_grams[fold], fold_crossproducts[fold], timings);
        gram_sum += fold_grams[fold];
        crossproduct_sum += fold_crossproducts[fold];
      }
      for(int fold = 0; fold < options.folds; ++fold) {
        training_gram = gram_sum - fold_grams[fold];
        training_crossproduct =
          crossproduct_sum - fold_crossproducts[fold];
        backend.factorize_ridge_system(
          training_gram, training_crossproduct, timings);
        backend.ridge_predict_factorized(
          fold_designs[fold], false, ridge_parameters,
          no_outcomes, false, predictions, coefficients, timings);
      }
    }
    return std::chrono::duration<double, std::milli>(
      Clock::now() - start).count();
  };

  double warmup_wall_ms = 0;
  double first_warmup_wall_ms = 0;
  for(int repeat = 0; repeat < options.warmup_repeats; ++repeat) {
    Step1ComputeTimings ignored_timings;
    const double iteration_wall_ms = run_iteration(&ignored_timings);
    if(repeat == 0) first_warmup_wall_ms = iteration_wall_ms;
    warmup_wall_ms += iteration_wall_ms;
  }

  for(int repeat = 0; repeat < options.repeats; ++repeat) {
    Step1ComputeTimings timings;
    wall_ms += run_iteration(&timings);
    accumulate_timings(totals, timings);
  }

  const double divisor = options.repeats;
  const double steady_wall_ms = wall_ms / divisor;
  const double accounted_ms = total_timing_ms(totals) / divisor;
  std::cout << "STEP1_BACKEND_BENCHMARK_LEVEL1 backend=" << backend.name()
            << " features=" << options.blocks
            << " samples=" << options.samples
            << " phenotypes=" << options.phenotypes
            << " folds=" << options.folds
            << " ridge_parameters=" << options.ridge_parameters
            << " warmup_repeats=" << options.warmup_repeats
            << " first_warmup_wall_ms=" << first_warmup_wall_ms
            << " mean_warmup_wall_ms=" <<
              warmup_wall_ms / options.warmup_repeats
            << " repeats=" << options.repeats
            << " wall_ms=" << steady_wall_ms
            << " steady_wall_ms=" << steady_wall_ms
            << " accounted_ms=" << accounted_ms
            << " unaccounted_ms=" << steady_wall_ms - accounted_ms
            << " upload_ms=" << totals.upload_ms / divisor
            << " crossproduct_ms=" << totals.crossproduct_ms / divisor
            << " gram_ms=" << totals.gram_ms / divisor
            << " eigensolve_ms=" << totals.eigensolve_ms / divisor
            << " transform_ms=" << totals.transform_ms / divisor
            << " ridge_ms=" << totals.ridge_ms / divisor
            << " download_ms=" << totals.download_ms / divisor << "\n";
}

}

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::unique_ptr<Step1ComputeBackend> backend =
      make_step1_compute_backend(options.backend, options.device);
    std::cout << "STEP1_BACKEND_TEST backend=" << backend->name()
              << " description=\"" << backend->description() << "\"\n";
    run_conformance(*backend);
    if(options.benchmark) {
      run_benchmark(*backend, options);
      run_nonlinear_benchmark(*backend, options);
    }
    if(options.level1_benchmark)
      run_level1_benchmark(*backend, options);
    std::cout << "STEP1_BACKEND_TEST backend=" << backend->name() << " status=PASS\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "STEP1_BACKEND_TEST status=FAIL error=\"" << error.what() << "\"\n";
    return 1;
  }
}
