#!/usr/bin/env bash

# CUDA-first Step 1 phenotype-scaling benchmark. Large genotype data are
# generated once and reused for all phenotype counts and missingness regimes.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
regenie_binary="${REGENIE_BINARY:-$HOME/build/regenie-a100/regenie}"
plink2_binary="${PLINK2_BINARY:-$(command -v plink2 2>/dev/null)}"
device="${CUDA_DEVICE:-0}"
samples="${STEP1_SCALING_SAMPLES:-100000}"
variants="${STEP1_SCALING_VARIANTS:-500000}"
phenotype_counts_text="${STEP1_SCALING_PHENOTYPES:-1 4 8 32}"
regimes_text="${STEP1_SCALING_REGIMES:-overlap incident}"
lowmem_counts="${STEP1_SCALING_LOWMEM_COUNTS:-32}"
threads="${STEP1_SCALING_THREADS:-8}"
bsize="${STEP1_SCALING_BSIZE:-1000}"
seed="${STEP1_SCALING_SEED:-20260713}"
max_bed_gb="${STEP1_SCALING_MAX_BED_GB:-20}"
data_root="${STEP1_SCALING_DATA_ROOT:-$HOME/build/regenie-a100/step1-phenotype-scaling-data}"
result_root="${STEP1_SCALING_RESULT_ROOT:-$HOME/build/regenie-a100/step1-phenotype-scaling-results}"
run_id="${STEP1_SCALING_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
run_dir="$result_root/$run_id"

overall_status=0
if [ ! -x "$regenie_binary" ]; then
  echo "STEP1_PHENOTYPE_SCALING_ERROR: REGENIE binary is not executable: $regenie_binary" >&2
  exit 1
fi
if [ -z "$plink2_binary" ] || [ ! -x "$plink2_binary" ]; then
  echo "STEP1_PHENOTYPE_SCALING_ERROR: plink2 is not executable: $plink2_binary" >&2
  exit 1
fi

read -r -a phenotype_counts <<< "$phenotype_counts_text"
read -r -a regimes <<< "$regimes_text"
max_phenotypes=0
for phenotypes in "${phenotype_counts[@]}"; do
  if ! [[ "$phenotypes" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEP1_PHENOTYPE_SCALING_ERROR: invalid phenotype count: $phenotypes" >&2
    exit 1
  fi
  if (( phenotypes > max_phenotypes )); then
    max_phenotypes="$phenotypes"
  fi
done

mkdir -p "$data_root" "$run_dir" "$run_dir/lowmem"
bed_prefix="$data_root/n${samples}_m${variants}_human_s${seed}"
pgen_prefix="${bed_prefix}_pgen"

if [ ! -f "$pgen_prefix.pgen" ] || [ ! -f "$pgen_prefix.pvar" ] || \
   [ ! -f "$pgen_prefix.psam" ]; then
  if [ ! -f "$bed_prefix.bed" ] || [ ! -f "$bed_prefix.bim" ] || \
     [ ! -f "$bed_prefix.fam" ]; then
    python3 "$repo_dir/scripts/generate_step1_bed.py" \
      --prefix "$bed_prefix" \
      --samples "$samples" \
      --variants "$variants" \
      --phenotypes 1 \
      --chromosomes 22 \
      --chromosome-layout human \
      --trait-type qt \
      --missingness-profile none \
      --seed "$seed" \
      --max-bed-gb "$max_bed_gb"
    generation_status=$?
    if [ "$generation_status" -ne 0 ]; then
      exit "$generation_status"
    fi
  fi

  "$plink2_binary" \
    --bfile "$bed_prefix" \
    --make-pgen \
    --threads "$threads" \
    --out "$pgen_prefix"
  pgen_status=$?
  if [ "$pgen_status" -ne 0 ]; then
    exit "$pgen_status"
  fi
fi

for regime in "${regimes[@]}"; do
  case "$regime" in
    overlap)
      missingness_profile=none
      ;;
    incident)
      missingness_profile=incident
      ;;
    *)
      echo "STEP1_PHENOTYPE_SCALING_ERROR: unknown regime: $regime" >&2
      exit 1
      ;;
  esac
  phenotype_prefix="$data_root/bt_${regime}_n${samples}_p${max_phenotypes}_s${seed}"
  python3 "$repo_dir/scripts/generate_step1_phenotypes.py" \
    --prefix "$phenotype_prefix" \
    --samples "$samples" \
    --phenotypes "$max_phenotypes" \
    --trait-type bt \
    --missingness-profile "$missingness_profile" \
    --seed "$seed"
  phenotype_status=$?
  if [ "$phenotype_status" -ne 0 ]; then
    exit "$phenotype_status"
  fi
done

phenotype_list() {
  local count="$1"
  local result=""
  local phenotype
  for ((phenotype=1; phenotype<=count; ++phenotype)); do
    if [ -n "$result" ]; then
      result+=","
    fi
    result+="Y${phenotype}"
  done
  printf '%s' "$result"
}

run_case() {
  local regime="$1"
  local phenotypes="$2"
  local phenotype_prefix="$data_root/bt_${regime}_n${samples}_p${max_phenotypes}_s${seed}"
  local output_prefix="$run_dir/${regime}_p${phenotypes}"
  local log="$output_prefix.console.log"
  local selected_phenotypes
  selected_phenotypes="$(phenotype_list "$phenotypes")"
  local mode=in_memory
  local -a lowmem_args=()
  if [[ " $lowmem_counts " == *" $phenotypes "* ]]; then
    mode=lowmem
    lowmem_args=(
      --lowmem
      --lowmem-prefix "$run_dir/lowmem/${regime}_p${phenotypes}"
    )
  fi

  echo "STEP1_PHENOTYPE_SCALING_BEGIN regime=$regime phenotypes=$phenotypes mode=$mode"
  REGENIE_CUDA_DIRECT_GROUPED_UPLOAD="${REGENIE_CUDA_DIRECT_GROUPED_UPLOAD:-1}" \
  /usr/bin/time -f \
    "STEP1_PHENOTYPE_SCALING_TIME regime=$regime phenotypes=$phenotypes mode=$mode wall_s=%e max_rss_kb=%M" \
    "$regenie_binary" \
      --step 1 \
      --pgen "$pgen_prefix" \
      --covarFile "$phenotype_prefix.covar" \
      --phenoFile "$phenotype_prefix.pheno" \
      --phenoColList "$selected_phenotypes" \
      --bt \
      --bsize "$bsize" \
      --threads "$threads" \
      --seed "$seed" \
      --step1-profile \
      --compute-backend cuda \
      --gpu-device "$device" \
      "${lowmem_args[@]}" \
      --out "$output_prefix" \
      2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  echo "STEP1_PHENOTYPE_SCALING_EXIT regime=$regime phenotypes=$phenotypes status=$status"
  if [ "$status" -ne 0 ]; then
    overall_status=1
    return
  fi

  if [ "$phenotypes" -gt 1 ]; then
    local reference_prefix="$run_dir/${regime}_p1"
    python3 "$repo_dir/scripts/compare_numeric_files.py" \
      "${reference_prefix}_1.loco" \
      "${output_prefix}_1.loco" \
      --rtol 1e-7 \
      --atol 1e-9 \
      --output-significant-digits 6 \
      --engine numpy \
      --report-all
    local compare_status=$?
    echo "STEP1_PHENOTYPE_SCALING_COMPARE regime=$regime phenotypes=$phenotypes status=$compare_status"
    if [ "$compare_status" -ne 0 ]; then
      overall_status=1
    fi
  fi
}

echo "STEP1_PHENOTYPE_SCALING_CONFIG samples=$samples variants=$variants max_phenotypes=$max_phenotypes phenotype_counts=\"$phenotype_counts_text\" regimes=\"$regimes_text\" lowmem_counts=\"$lowmem_counts\" device=$device run_dir=$run_dir"
for regime in "${regimes[@]}"; do
  for phenotypes in "${phenotype_counts[@]}"; do
    run_case "$regime" "$phenotypes"
  done
done

echo
echo "STEP1_PHENOTYPE_SCALING_SUMMARY"
grep -H -E \
  '^STEP1_PROFILE version=|^STEP1_PROFILE scope=grouped_prediction|^STEP1_PROFILE_FINAL|^Elapsed time|^STEP1_PHENOTYPE_SCALING_TIME' \
  "$run_dir"/*.console.log
summary_status=$?
if [ "$summary_status" -ne 0 ]; then
  overall_status=1
fi

if [ "$overall_status" -eq 0 ]; then
  echo "STEP1_PHENOTYPE_SCALING status=PASS results=$run_dir"
else
  echo "STEP1_PHENOTYPE_SCALING status=FAIL results=$run_dir"
fi
exit "$overall_status"
