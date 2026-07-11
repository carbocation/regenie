/*

   This file is part of the regenie software package.

   Copyright (c) 2020-2024 Joelle Mbatchou, Andrey Ziyatdinov & Jonathan Marchini

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

*/

#include "Step1_Compute.hpp"

#include <chrono>
#include <stdexcept>

#ifdef WITH_CUDA
std::unique_ptr<Step1ComputeBackend> make_cuda_step1_compute_backend(int device);
bool cuda_step1_compute_backend_available(int device, std::string& reason);
#endif

namespace {

using ComputeClock = std::chrono::steady_clock;

double elapsed_ms(const ComputeClock::time_point& start) {
  return std::chrono::duration<double, std::milli>(ComputeClock::now() - start).count();
}

}

class CpuStep1ComputeBackend : public Step1ComputeBackend {

  public:
    const char* name() const override {
      return "cpu";
    }

    std::string description() const override {
      return "Eigen CPU";
    }

    void compute_products(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1GramMode mode,
      Step1ComputeTimings* timings) override {

      if(genotypes.cols() != phenotypes.rows())
        throw std::invalid_argument(
          "Step 1 compute backend received incompatible genotype and phenotype matrices");

      if(mode == Step1GramMode::selfadjoint_rank_update) {
        ComputeClock::time_point start;
        if(timings) start = ComputeClock::now();
        gram.setZero(genotypes.rows(), genotypes.rows());
        gram.selfadjointView<Eigen::Lower>().rankUpdate(genotypes);
        gram.triangularView<Eigen::Upper>() = gram.transpose();
        if(timings) timings->gram_ms += elapsed_ms(start);

        if(timings) start = ComputeClock::now();
        crossproduct = genotypes * phenotypes;
        if(timings) timings->crossproduct_ms += elapsed_ms(start);
      } else {
        ComputeClock::time_point start;
        if(timings) start = ComputeClock::now();
        crossproduct = genotypes * phenotypes;
        if(timings) timings->crossproduct_ms += elapsed_ms(start);

        if(timings) start = ComputeClock::now();
        gram = genotypes * genotypes.transpose();
        if(timings) timings->gram_ms += elapsed_ms(start);
      }
    }
};

std::unique_ptr<Step1ComputeBackend> make_cpu_step1_compute_backend() {
  return std::unique_ptr<Step1ComputeBackend>(new CpuStep1ComputeBackend());
}

bool cuda_step1_compute_backend_compiled() {
#ifdef WITH_CUDA
  return true;
#else
  return false;
#endif
}

std::unique_ptr<Step1ComputeBackend> make_step1_compute_backend(
  const std::string& requested_backend,
  int device) {

  if(requested_backend == "cpu")
    return make_cpu_step1_compute_backend();

  if(requested_backend != "cuda" && requested_backend != "auto")
    throw std::runtime_error("unknown Step 1 compute backend '" + requested_backend +
      "' (expected cpu, cuda, or auto)");

#ifdef WITH_CUDA
  std::string reason;
  if(cuda_step1_compute_backend_available(device, reason))
    return make_cuda_step1_compute_backend(device);
  if(requested_backend == "cuda")
    throw std::runtime_error("CUDA Step 1 backend is unavailable: " + reason);
#else
  (void)device;
  if(requested_backend == "cuda")
    throw std::runtime_error(
      "CUDA Step 1 backend was requested, but this binary was built without REGENIE_WITH_CUDA");
#endif

  return make_cpu_step1_compute_backend();
}
