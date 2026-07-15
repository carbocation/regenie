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

class CpuStep1ComputeBackend : public Step1ComputeBackend {

  public:
    const char* name() const override {
      return "cpu";
    }

    void compute_gram(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      Eigen::MatrixXd& gram,
      Step1GramMode mode) override {

      if(mode == Step1GramMode::selfadjoint_rank_update) {
        gram.setZero(genotypes.rows(), genotypes.rows());
        gram.selfadjointView<Eigen::Lower>().rankUpdate(genotypes);
        gram.triangularView<Eigen::Upper>() = gram.transpose();
      } else {
        gram = genotypes * genotypes.transpose();
      }
    }

    void compute_crossproduct(
      const Eigen::Ref<const Eigen::MatrixXd>& genotypes,
      const Eigen::Ref<const Eigen::MatrixXd>& phenotypes,
      Eigen::MatrixXd& crossproduct) override {

      crossproduct = genotypes * phenotypes;
    }
};

std::unique_ptr<Step1ComputeBackend> make_cpu_step1_compute_backend() {
  return std::unique_ptr<Step1ComputeBackend>(new CpuStep1ComputeBackend());
}
