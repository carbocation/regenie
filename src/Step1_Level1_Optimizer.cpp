#include "Step1_Level1_Optimizer.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

const char* const kStep1Level1OptimizerVariable =
  "REGENIE_STEP1_LEVEL1_OPTIMIZER";

}

Step1Level1Optimizer parse_step1_level1_optimizer(const char* value) {
  if(!value || !*value || std::string(value) == "irls")
    return Step1Level1Optimizer::Irls;
  if(std::string(value) == "path-newton")
    return Step1Level1Optimizer::PathNewton;
  if(std::string(value) == "newton-cg")
    return Step1Level1Optimizer::NewtonCg;
  throw std::invalid_argument(
    std::string(kStep1Level1OptimizerVariable) +
    " must be one of: irls, path-newton, newton-cg");
}

Step1Level1Optimizer step1_level1_optimizer_from_environment() {
  return parse_step1_level1_optimizer(
    std::getenv(kStep1Level1OptimizerVariable));
}

const char* step1_level1_optimizer_name(
  Step1Level1Optimizer optimizer) {

  switch(optimizer) {
    case Step1Level1Optimizer::Irls:
      return "irls";
    case Step1Level1Optimizer::PathNewton:
      return "path-newton";
    case Step1Level1Optimizer::NewtonCg:
      return "newton-cg";
  }
  throw std::invalid_argument("Unknown Step 1 Level 1 optimizer");
}
