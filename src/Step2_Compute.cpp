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

#include "Step2_Compute.hpp"

#include <stdexcept>

#ifdef WITH_CUDA
std::unique_ptr<Step2ComputeBackend> make_cuda_step2_compute_backend(
  int device, bool automatic);
bool cuda_step2_compute_backend_available(int device, std::string& reason);
#endif

namespace {

class CpuStep2ComputeBackend : public Step2ComputeBackend {
 public:
  const char* name() const override { return "cpu"; }
  std::string description() const override {
    return "host Step 2 scoring";
  }
  bool ready() const override { return false; }
  void clear() override {}

  bool prepare_quantitative(
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic,
        Eigen::Dynamic>>&,
      bool, Step2ComputeTimings*) override {
    return false;
  }

  bool prepare_binary(
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const std::vector<Eigen::MatrixXd>&,
      const std::vector<Eigen::VectorXd>&,
      const Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, 1>>&,
      Step2ComputeTimings*) override {
    return false;
  }

  bool prepare_cox(
      const std::vector<Eigen::VectorXd>&,
      const std::vector<Eigen::MatrixXd>&,
      const std::vector<Eigen::MatrixXd>&,
      const std::vector<Eigen::VectorXd>&,
      const std::vector<Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::VectorXd>&,
      const Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, 1>>&,
      Step2ComputeTimings*) override {
    return false;
  }

  bool score_packed_block(
      const std::vector<std::vector<unsigned char>>& ,
      const std::vector<double>&,
      const std::vector<unsigned char>&,
      const std::vector<unsigned char>&,
      Eigen::Index, Eigen::MatrixXd&, Eigen::MatrixXd&,
      Step2ComputeTimings*) override {
    return false;
  }
};

}  // namespace

std::unique_ptr<Step2ComputeBackend> make_step2_compute_backend(
    const std::string& requested_backend, int device) {
  if(requested_backend == "cpu")
    return std::unique_ptr<Step2ComputeBackend>(
      new CpuStep2ComputeBackend());

  if(requested_backend != "cuda" && requested_backend != "auto")
    throw std::runtime_error("unknown Step 2 compute backend '" +
      requested_backend + "' (expected cpu, cuda, or auto)");

#ifdef WITH_CUDA
  std::string reason;
  if(cuda_step2_compute_backend_available(device, reason))
    return make_cuda_step2_compute_backend(device,
      requested_backend == "auto");
  if(requested_backend == "cuda")
    throw std::runtime_error("CUDA Step 2 backend is unavailable: " + reason);
#else
  (void)device;
  if(requested_backend == "cuda")
    throw std::runtime_error(
      "CUDA Step 2 backend was requested, but this binary was built without REGENIE_WITH_CUDA");
#endif

  return std::unique_ptr<Step2ComputeBackend>(new CpuStep2ComputeBackend());
}
