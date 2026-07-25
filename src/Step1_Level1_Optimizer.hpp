#ifndef STEP1_LEVEL1_OPTIMIZER_H
#define STEP1_LEVEL1_OPTIMIZER_H

enum class Step1Level1Optimizer {
  Irls,
  PathNewton,
  NewtonCg
};

Step1Level1Optimizer parse_step1_level1_optimizer(const char* value);
Step1Level1Optimizer step1_level1_optimizer_from_environment();
const char* step1_level1_optimizer_name(Step1Level1Optimizer optimizer);

#endif
