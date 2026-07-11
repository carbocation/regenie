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

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string backend = "cpu";
  int device = 0;
  bool benchmark = false;
  int blocks = 512;
  int samples = 20000;
  int phenotypes = 10;
  int repeats = 3;
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
    else if(argument == "--backend" && i + 1 < argc) options.backend = argv[++i];
    else if(argument == "--device" && i + 1 < argc)
      options.device = parse_positive(argv[++i], "--device", true);
    else if(argument == "--blocks" && i + 1 < argc)
      options.blocks = parse_positive(argv[++i], "--blocks");
    else if(argument == "--samples" && i + 1 < argc)
      options.samples = parse_positive(argv[++i], "--samples");
    else if(argument == "--phenotypes" && i + 1 < argc)
      options.phenotypes = parse_positive(argv[++i], "--phenotypes");
    else if(argument == "--repeats" && i + 1 < argc)
      options.repeats = parse_positive(argv[++i], "--repeats");
    else
      throw std::runtime_error("unknown or incomplete option: " + argument);
  }
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
  const double tolerance = 5e-12;
  if(gram_error > tolerance || crossproduct_error > tolerance || symmetry_error > tolerance) {
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

void run_conformance(Step1ComputeBackend& candidate) {
  std::unique_ptr<Step1ComputeBackend> oracle = make_cpu_step1_compute_backend();
  const Eigen::MatrixXd genotypes = deterministic_matrix(17, 43, 0.2);
  const Eigen::MatrixXd phenotypes = deterministic_matrix(43, 5, -0.4);

  check_case(candidate, *oracle, genotypes, phenotypes,
    Step1GramMode::full_product, "contiguous_full_product");
  check_case(candidate, *oracle, genotypes, phenotypes,
    Step1GramMode::selfadjoint_rank_update, "contiguous_rank_update");

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
}

void run_benchmark(Step1ComputeBackend& backend, const Options& options) {
  const Eigen::MatrixXd genotypes = deterministic_matrix(options.blocks, options.samples, 0.1);
  const Eigen::MatrixXd phenotypes = deterministic_matrix(options.samples, options.phenotypes, -0.2);
  Eigen::MatrixXd gram, crossproduct;
  Step1ComputeTimings totals;
  double wall_ms = 0;

  for(int repeat = 0; repeat < options.repeats; ++repeat) {
    Step1ComputeTimings timings;
    const Clock::time_point start = Clock::now();
    backend.compute_products(genotypes, phenotypes, gram, crossproduct,
      Step1GramMode::selfadjoint_rank_update, &timings);
    wall_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    totals.upload_ms += timings.upload_ms;
    totals.crossproduct_ms += timings.crossproduct_ms;
    totals.gram_ms += timings.gram_ms;
    totals.download_ms += timings.download_ms;
  }

  const double divisor = options.repeats;
  std::cout << "STEP1_BACKEND_BENCHMARK backend=" << backend.name()
            << " blocks=" << options.blocks
            << " samples=" << options.samples
            << " phenotypes=" << options.phenotypes
            << " repeats=" << options.repeats
            << " wall_ms=" << wall_ms / divisor
            << " upload_ms=" << totals.upload_ms / divisor
            << " crossproduct_ms=" << totals.crossproduct_ms / divisor
            << " gram_ms=" << totals.gram_ms / divisor
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
    if(options.benchmark) run_benchmark(*backend, options);
    std::cout << "STEP1_BACKEND_TEST backend=" << backend->name() << " status=PASS\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "STEP1_BACKEND_TEST status=FAIL error=\"" << error.what() << "\"\n";
    return 1;
  }
}
