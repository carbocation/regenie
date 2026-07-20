// Microbenchmark for a possible packed-hardcall Step 2 CUDA path.
//
// This measures packed-block upload plus three resident-state kernels:
// quantitative numerator/denominator accumulation, a weighted binary-style
// projection, and a dual-projection Cox-style approximation. It is not a
// REGENIE end-to-end benchmark and does not include correction tests,
// formatting, or output.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kCovariates = 5;

void check_cuda(cudaError_t status, const char* operation) {
  if(status == cudaSuccess) return;
  std::fprintf(stderr, "%s failed: %s\n", operation,
    cudaGetErrorString(status));
  std::exit(1);
}

__inline__ __device__ double warp_sum(double value) {
  for(int offset = warpSize / 2; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__global__ void packed_score_kernel(const std::uint8_t* packed,
    const double* residuals, const std::uint8_t* observed,
    int samples, int packed_stride, int phenotypes,
    double* numerators, double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const std::uint8_t* variant_packed =
    packed + static_cast<std::size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    residuals + static_cast<std::size_t>(phenotype) * samples;
  const std::uint8_t* phenotype_observed =
    observed + static_cast<std::size_t>(phenotype) * samples;

  double numerator = 0;
  double denominator = 0;
  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const std::uint8_t code =
      (variant_packed[sample >> 2] >> (2 * (sample & 3))) & 3;
    const double genotype = code == 3 ? 0.0 : static_cast<double>(code);
    numerator += genotype * phenotype_residuals[sample];
    if(phenotype_observed[sample]) denominator += genotype * genotype;
  }

  numerator = warp_sum(numerator);
  denominator = warp_sum(denominator);
  __shared__ double numerator_warps[32];
  __shared__ double denominator_warps[32];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  if(lane == 0) {
    numerator_warps[warp] = numerator;
    denominator_warps[warp] = denominator;
  }
  __syncthreads();

  if(warp == 0) {
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    numerator = lane < warp_count ? numerator_warps[lane] : 0;
    denominator = lane < warp_count ? denominator_warps[lane] : 0;
    numerator = warp_sum(numerator);
    denominator = warp_sum(denominator);
    if(lane == 0) {
      const std::size_t output_index =
        static_cast<std::size_t>(variant) * phenotypes + phenotype;
      numerators[output_index] = numerator;
      denominators[output_index] = denominator;
    }
  }
}

__global__ void packed_weighted_score_kernel(const std::uint8_t* packed,
    const double* residuals, const double* weights, const double* designs,
    const double* design_residual_crossproducts, int samples,
    int packed_stride, int phenotypes, double* numerators,
    double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const std::uint8_t* variant_packed =
    packed + static_cast<std::size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    residuals + static_cast<std::size_t>(phenotype) * samples;
  const double* phenotype_weights =
    weights + static_cast<std::size_t>(phenotype) * samples;

  double accumulators[kCovariates + 2] = {};
  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const std::uint8_t code =
      (variant_packed[sample >> 2] >> (2 * (sample & 3))) & 3;
    const double genotype = code == 3 ? 0.0 : static_cast<double>(code);
    const double weighted_genotype = genotype * phenotype_weights[sample];
    accumulators[0] += weighted_genotype * phenotype_residuals[sample];
    accumulators[1] += weighted_genotype * weighted_genotype;
#pragma unroll
    for(int covariate = 0; covariate < kCovariates; ++covariate) {
      const std::size_t design_offset =
        (static_cast<std::size_t>(phenotype) * kCovariates + covariate) *
        samples;
      accumulators[covariate + 2] +=
        weighted_genotype * designs[design_offset + sample];
    }
  }

#pragma unroll
  for(int term = 0; term < kCovariates + 2; ++term)
    accumulators[term] = warp_sum(accumulators[term]);
  __shared__ double warp_terms[kCovariates + 2][32];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  if(lane == 0) {
#pragma unroll
    for(int term = 0; term < kCovariates + 2; ++term)
      warp_terms[term][warp] = accumulators[term];
  }
  __syncthreads();

  if(warp == 0) {
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
#pragma unroll
    for(int term = 0; term < kCovariates + 2; ++term) {
      accumulators[term] = lane < warp_count ? warp_terms[term][lane] : 0;
      accumulators[term] = warp_sum(accumulators[term]);
    }
    if(lane == 0) {
      double numerator = accumulators[0];
      double denominator = accumulators[1];
#pragma unroll
      for(int covariate = 0; covariate < kCovariates; ++covariate) {
        const double crossproduct = accumulators[covariate + 2];
        numerator -= crossproduct *
          design_residual_crossproducts[phenotype * kCovariates + covariate];
        denominator -= crossproduct * crossproduct;
      }
      const std::size_t output_index =
        static_cast<std::size_t>(variant) * phenotypes + phenotype;
      numerators[output_index] = numerator;
      denominators[output_index] = denominator;
    }
  }
}

__global__ void packed_cox_score_kernel(const std::uint8_t* packed,
    const double* residuals, const double* weighted_designs,
    const double* projections, const double* projection_scores,
    int samples, int packed_stride, int phenotypes,
    double* numerators, double* denominators) {
  const int variant = blockIdx.x;
  const int phenotype = blockIdx.y;
  const std::uint8_t* variant_packed =
    packed + static_cast<std::size_t>(variant) * packed_stride;
  const double* phenotype_residuals =
    residuals + static_cast<std::size_t>(phenotype) * samples;

  double accumulators[2 + 2 * kCovariates] = {};
  for(int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const std::uint8_t code =
      (variant_packed[sample >> 2] >> (2 * (sample & 3))) & 3;
    const double genotype = code == 3 ? 0.0 : static_cast<double>(code);
    accumulators[0] += genotype * phenotype_residuals[sample];
    accumulators[1] += genotype * genotype;
#pragma unroll
    for(int covariate = 0; covariate < kCovariates; ++covariate) {
      const std::size_t design_offset =
        (static_cast<std::size_t>(phenotype) * kCovariates + covariate) *
        samples + sample;
      accumulators[2 + covariate] +=
        genotype * weighted_designs[design_offset];
      accumulators[2 + kCovariates + covariate] +=
        genotype * projections[design_offset];
    }
  }

#pragma unroll
  for(int term = 0; term < 2 + 2 * kCovariates; ++term)
    accumulators[term] = warp_sum(accumulators[term]);
  __shared__ double warp_terms[2 + 2 * kCovariates][32];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  if(lane == 0) {
#pragma unroll
    for(int term = 0; term < 2 + 2 * kCovariates; ++term)
      warp_terms[term][warp] = accumulators[term];
  }
  __syncthreads();

  if(warp == 0) {
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
#pragma unroll
    for(int term = 0; term < 2 + 2 * kCovariates; ++term) {
      accumulators[term] = lane < warp_count ? warp_terms[term][lane] : 0;
      accumulators[term] = warp_sum(accumulators[term]);
    }
    if(lane == 0) {
      double numerator = accumulators[0];
      double denominator = accumulators[1];
#pragma unroll
      for(int covariate = 0; covariate < kCovariates; ++covariate) {
        const double coefficient = accumulators[2 + covariate];
        const double raw_cross =
          accumulators[2 + kCovariates + covariate];
        numerator -= coefficient *
          projection_scores[phenotype * kCovariates + covariate];
        // The generated projection Gram matrix is the identity.
        denominator += coefficient * coefficient -
          2 * coefficient * raw_cross;
      }
      const std::size_t output_index =
        static_cast<std::size_t>(variant) * phenotypes + phenotype;
      numerators[output_index] = numerator;
      denominators[output_index] = denominator;
    }
  }
}

std::uint64_t next_random(std::uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return state;
}

} // namespace

int main(int argc, char** argv) {
  const int samples = argc > 1 ? std::atoi(argv[1]) : 50000;
  const int variants = argc > 2 ? std::atoi(argv[2]) : 1000;
  const int phenotypes = argc > 3 ? std::atoi(argv[3]) : 32;
  const int repetitions = argc > 4 ? std::atoi(argv[4]) : 10;
  if(samples <= 0 || variants <= 0 || phenotypes <= 0 || repetitions <= 0) {
    std::fprintf(stderr,
      "usage: %s [samples variants phenotypes repetitions]\n", argv[0]);
    return 2;
  }

  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0),
    "cudaGetDeviceProperties");
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  const int packed_stride = (samples + 3) / 4;
  const std::size_t packed_bytes =
    static_cast<std::size_t>(variants) * packed_stride;
  const std::size_t phenotype_values =
    static_cast<std::size_t>(phenotypes) * samples;
  const std::size_t output_values =
    static_cast<std::size_t>(variants) * phenotypes;

  std::vector<std::uint8_t> packed(packed_bytes);
  std::vector<double> residuals(phenotype_values);
  std::vector<std::uint8_t> observed(phenotype_values);
  std::vector<double> weights(phenotype_values);
  const std::size_t design_values =
    phenotype_values * kCovariates;
  std::vector<double> designs(design_values);
  std::vector<double> projections(design_values);
  std::vector<double> small_crossproducts(
    static_cast<std::size_t>(phenotypes) * kCovariates);
  std::uint64_t random_state = 0x243f6a8885a308d3ULL;
  for(std::size_t i = 0; i < packed.size(); ++i)
    packed[i] = static_cast<std::uint8_t>(next_random(random_state) >> 56);
  for(std::size_t i = 0; i < phenotype_values; ++i) {
    const std::uint64_t bits = next_random(random_state);
    residuals[i] = (static_cast<double>((bits >> 32) & 0xffffu) - 32768.0) /
      32768.0;
    observed[i] = (bits & 0xffffu) >= 3277u;
    weights[i] = 0.25 + static_cast<double>(bits & 0xffffu) / 131072.0;
  }
  for(std::size_t i = 0; i < design_values; ++i) {
    const std::uint64_t bits = next_random(random_state);
    designs[i] =
      (static_cast<double>((bits >> 32) & 0xffffu) - 32768.0) / 327680.0;
    projections[i] =
      (static_cast<double>((bits >> 48) & 0xffffu) - 32768.0) / 327680.0;
  }
  for(std::size_t i = 0; i < small_crossproducts.size(); ++i)
    small_crossproducts[i] =
      (static_cast<double>(next_random(random_state) >> 48) - 32768.0) /
      327680.0;

  std::uint8_t* device_packed = nullptr;
  double* device_residuals = nullptr;
  std::uint8_t* device_observed = nullptr;
  double* device_weights = nullptr;
  double* device_designs = nullptr;
  double* device_projections = nullptr;
  double* device_small_crossproducts = nullptr;
  double* device_numerators = nullptr;
  double* device_denominators = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_packed),
    packed_bytes), "cudaMalloc packed");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_residuals),
    phenotype_values * sizeof(double)), "cudaMalloc residuals");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_observed),
    phenotype_values),
    "cudaMalloc observed");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_weights),
    phenotype_values * sizeof(double)), "cudaMalloc weights");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_designs),
    design_values * sizeof(double)), "cudaMalloc designs");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_projections),
    design_values * sizeof(double)), "cudaMalloc projections");
  check_cuda(cudaMalloc(
    reinterpret_cast<void**>(&device_small_crossproducts),
    small_crossproducts.size() * sizeof(double)),
    "cudaMalloc small crossproducts");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_numerators),
    output_values * sizeof(double)), "cudaMalloc numerators");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_denominators),
    output_values * sizeof(double)), "cudaMalloc denominators");
  check_cuda(cudaMemcpy(device_residuals, residuals.data(),
    phenotype_values * sizeof(double), cudaMemcpyHostToDevice),
    "copy residuals");
  check_cuda(cudaMemcpy(device_observed, observed.data(), phenotype_values,
    cudaMemcpyHostToDevice), "copy observed");
  check_cuda(cudaMemcpy(device_weights, weights.data(),
    phenotype_values * sizeof(double), cudaMemcpyHostToDevice),
    "copy weights");
  check_cuda(cudaMemcpy(device_designs, designs.data(),
    design_values * sizeof(double), cudaMemcpyHostToDevice),
    "copy designs");
  check_cuda(cudaMemcpy(device_projections, projections.data(),
    design_values * sizeof(double), cudaMemcpyHostToDevice),
    "copy projections");
  check_cuda(cudaMemcpy(device_small_crossproducts,
    small_crossproducts.data(),
    small_crossproducts.size() * sizeof(double), cudaMemcpyHostToDevice),
    "copy small crossproducts");

  cudaEvent_t start;
  cudaEvent_t stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  const dim3 grid(variants, phenotypes);
  const int threads = 256;

  check_cuda(cudaMemcpy(device_packed, packed.data(), packed_bytes,
    cudaMemcpyHostToDevice), "warmup packed copy");
  packed_score_kernel<<<grid, threads>>>(device_packed, device_residuals,
    device_observed, samples, packed_stride, phenotypes,
    device_numerators, device_denominators);
  check_cuda(cudaDeviceSynchronize(), "warmup kernel");

  float upload_total_ms = 0;
  float kernel_total_ms = 0;
  for(int repetition = 0; repetition < repetitions; ++repetition) {
    check_cuda(cudaEventRecord(start), "record upload start");
    check_cuda(cudaMemcpyAsync(device_packed, packed.data(), packed_bytes,
      cudaMemcpyHostToDevice), "copy packed block");
    check_cuda(cudaEventRecord(stop), "record upload stop");
    check_cuda(cudaEventSynchronize(stop), "sync upload stop");
    float elapsed = 0;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
      "measure upload");
    upload_total_ms += elapsed;

    check_cuda(cudaEventRecord(start), "record kernel start");
    packed_score_kernel<<<grid, threads>>>(device_packed, device_residuals,
      device_observed, samples, packed_stride, phenotypes,
      device_numerators, device_denominators);
    check_cuda(cudaEventRecord(stop), "record kernel stop");
    check_cuda(cudaEventSynchronize(stop), "sync kernel stop");
    check_cuda(cudaGetLastError(), "packed_score_kernel");
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
      "measure kernel");
    kernel_total_ms += elapsed;
  }

  std::vector<double> checksum_values(2);
  check_cuda(cudaMemcpy(&checksum_values[0], device_numerators,
    sizeof(double), cudaMemcpyDeviceToHost), "copy numerator checksum");
  check_cuda(cudaMemcpy(&checksum_values[1], device_denominators,
    sizeof(double), cudaMemcpyDeviceToHost), "copy denominator checksum");

  const double upload_ms = upload_total_ms / repetitions;
  const double kernel_ms = kernel_total_ms / repetitions;
  const double blocks_for_700k = 700000.0 / variants;
  std::printf("mode=quantitative gpu=%s compute_capability=%d.%d "
    "samples=%d variants=%d "
    "phenotypes=%d repetitions=%d packed_mib=%.3f upload_ms=%.3f "
    "kernel_ms=%.3f combined_ms=%.3f projected_700k_seconds=%.3f "
    "checksum_num=%.12g checksum_denom=%.12g\n",
    properties.name, properties.major, properties.minor, samples, variants,
    phenotypes, repetitions, packed_bytes / 1048576.0, upload_ms, kernel_ms,
    upload_ms + kernel_ms, blocks_for_700k * (upload_ms + kernel_ms) / 1000.0,
    checksum_values[0], checksum_values[1]);

  packed_weighted_score_kernel<<<grid, threads>>>(device_packed,
    device_residuals, device_weights, device_designs,
    device_small_crossproducts, samples, packed_stride, phenotypes,
    device_numerators, device_denominators);
  check_cuda(cudaDeviceSynchronize(), "warmup weighted kernel");
  float weighted_total_ms = 0;
  for(int repetition = 0; repetition < repetitions; ++repetition) {
    check_cuda(cudaEventRecord(start), "record weighted start");
    packed_weighted_score_kernel<<<grid, threads>>>(device_packed,
      device_residuals, device_weights, device_designs,
      device_small_crossproducts, samples, packed_stride, phenotypes,
      device_numerators, device_denominators);
    check_cuda(cudaEventRecord(stop), "record weighted stop");
    check_cuda(cudaEventSynchronize(stop), "sync weighted stop");
    check_cuda(cudaGetLastError(), "packed_weighted_score_kernel");
    float elapsed = 0;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
      "measure weighted kernel");
    weighted_total_ms += elapsed;
  }
  check_cuda(cudaMemcpy(&checksum_values[0], device_numerators,
    sizeof(double), cudaMemcpyDeviceToHost), "copy weighted numerator");
  check_cuda(cudaMemcpy(&checksum_values[1], device_denominators,
    sizeof(double), cudaMemcpyDeviceToHost), "copy weighted denominator");
  const double weighted_ms = weighted_total_ms / repetitions;
  std::printf("mode=weighted_projection gpu=%s compute_capability=%d.%d "
    "samples=%d variants=%d phenotypes=%d covariates=%d repetitions=%d "
    "packed_mib=%.3f upload_ms=%.3f kernel_ms=%.3f combined_ms=%.3f "
    "projected_700k_seconds=%.3f checksum_num=%.12g "
    "checksum_denom=%.12g\n",
    properties.name, properties.major, properties.minor, samples, variants,
    phenotypes, kCovariates, repetitions, packed_bytes / 1048576.0,
    upload_ms, weighted_ms, upload_ms + weighted_ms,
    blocks_for_700k * (upload_ms + weighted_ms) / 1000.0,
    checksum_values[0], checksum_values[1]);

  packed_cox_score_kernel<<<grid, threads>>>(device_packed,
    device_residuals, device_designs, device_projections,
    device_small_crossproducts, samples, packed_stride, phenotypes,
    device_numerators, device_denominators);
  check_cuda(cudaDeviceSynchronize(), "warmup Cox kernel");
  float cox_total_ms = 0;
  for(int repetition = 0; repetition < repetitions; ++repetition) {
    check_cuda(cudaEventRecord(start), "record Cox start");
    packed_cox_score_kernel<<<grid, threads>>>(device_packed,
      device_residuals, device_designs, device_projections,
      device_small_crossproducts, samples, packed_stride, phenotypes,
      device_numerators, device_denominators);
    check_cuda(cudaEventRecord(stop), "record Cox stop");
    check_cuda(cudaEventSynchronize(stop), "sync Cox stop");
    check_cuda(cudaGetLastError(), "packed_cox_score_kernel");
    float elapsed = 0;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
      "measure Cox kernel");
    cox_total_ms += elapsed;
  }
  check_cuda(cudaMemcpy(&checksum_values[0], device_numerators,
    sizeof(double), cudaMemcpyDeviceToHost), "copy Cox numerator");
  check_cuda(cudaMemcpy(&checksum_values[1], device_denominators,
    sizeof(double), cudaMemcpyDeviceToHost), "copy Cox denominator");
  const double cox_ms = cox_total_ms / repetitions;
  std::printf("mode=cox_projection gpu=%s compute_capability=%d.%d "
    "samples=%d variants=%d phenotypes=%d covariates=%d repetitions=%d "
    "packed_mib=%.3f upload_ms=%.3f kernel_ms=%.3f combined_ms=%.3f "
    "projected_700k_seconds=%.3f checksum_num=%.12g "
    "checksum_denom=%.12g\n",
    properties.name, properties.major, properties.minor, samples, variants,
    phenotypes, kCovariates, repetitions, packed_bytes / 1048576.0,
    upload_ms, cox_ms, upload_ms + cox_ms,
    blocks_for_700k * (upload_ms + cox_ms) / 1000.0,
    checksum_values[0], checksum_values[1]);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_packed);
  cudaFree(device_residuals);
  cudaFree(device_observed);
  cudaFree(device_weights);
  cudaFree(device_designs);
  cudaFree(device_projections);
  cudaFree(device_small_crossproducts);
  cudaFree(device_numerators);
  cudaFree(device_denominators);
  return 0;
}
