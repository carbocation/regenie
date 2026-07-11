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

#ifndef STEP1_COMPUTE_H
#define STEP1_COMPUTE_H

#include <memory>
#include <string>
#include <Eigen/Dense>

enum class Step1GramMode {
  full_product,
  selfadjoint_rank_update
};

struct Step1ComputeTimings {
  double upload_ms = 0;
  double crossproduct_ms = 0;
  double gram_ms = 0;
  double download_ms = 0;
};

class Step1ComputeBackend {

  public:
    virtual ~Step1ComputeBackend() {}

    virtual const char* name() const = 0;
    virtual std::string description() const = 0;

    virtual void compute_products(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& gram,
      Eigen::MatrixXd& crossproduct,
      Step1GramMode mode,
      Step1ComputeTimings* timings = nullptr) = 0;
};

std::unique_ptr<Step1ComputeBackend> make_cpu_step1_compute_backend();
std::unique_ptr<Step1ComputeBackend> make_step1_compute_backend(
  const std::string& requested_backend,
  int device);
bool cuda_step1_compute_backend_compiled();

#endif
