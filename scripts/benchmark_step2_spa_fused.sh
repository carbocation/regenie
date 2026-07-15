#!/usr/bin/env bash

# Matched dense-SPA benchmark for the fused CGF derivative evaluator. Verbose
# command output is retained in the result directory; stdout is a concise,
# machine-readable summary suitable for pasting into an issue or chat.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
regenie_binary="${REGENIE_BINARY:-$HOME/build/regenie-t4/regenie}"
pgen_prefix="${STEP2_SPA_PGEN_PREFIX:-}"
covar_file="${STEP2_SPA_COVAR_FILE:-}"
keep_file="${STEP2_SPA_KEEP_FILE:-}"
pheno_file="${STEP2_SPA_PHENO_FILE:-}"
extract_file="${STEP2_SPA_EXTRACT_FILE:-}"
pheno_column="${STEP2_SPA_PHENO_COLUMN:-}"
threads="${STEP2_SPA_THREADS:-8}"
bsize="${STEP2_SPA_BSIZE:-100}"
timeout_seconds="${STEP2_SPA_TIMEOUT_SECONDS:-120}"
result_root="${STEP2_SPA_RESULT_ROOT:-$HOME/build/regenie-t4/step2-spa-fused-ab}"
run_id="${STEP2_SPA_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
run_dir="$result_root/$run_id"
driver_log="$run_dir/driver.log"
numeric_log="$run_dir/numeric_compare.log"

mkdir -p "$run_dir"

run_case() {
  local label="$1"
  local fused_cgf="$2"
  local output_prefix="$run_dir/$label"
  local console_log="$output_prefix.console.log"
  local time_log="$output_prefix.time.log"

  /usr/bin/time \
    -f "T4_SPA_FUSED_TIME label=$label wall_s=%e max_rss_kb=%M" \
    -o "$time_log" \
    env REGENIE_SPA_FUSED_CGF="$fused_cgf" \
    timeout "$timeout_seconds" \
    "$regenie_binary" \
      --step 2 \
      --pgen "$pgen_prefix" \
      --covarFile "$covar_file" \
      --keep "$keep_file" \
      --phenoFile "$pheno_file" \
      --phenoColList "$pheno_column" \
      --extract "$extract_file" \
      --bt \
      --spa \
      --pThresh 0.999 \
      --minMAC 5 \
      --ignore-pred \
      --bsize "$bsize" \
      --threads "$threads" \
      --step2-profile \
      --out "$output_prefix" \
      > "$console_log" 2>&1
  local status=$?
  printf -v "${label}_status" '%s' "$status"
}

benchmark_body() {
  input_status=0
  for required_file in \
    "$pgen_prefix.pgen" \
    "$pgen_prefix.pvar" \
    "$pgen_prefix.psam" \
    "$covar_file" \
    "$keep_file" \
    "$pheno_file" \
    "$extract_file"
  do
    if [ ! -s "$required_file" ]; then
      echo "STEP2_SPA_FUSED_ERROR missing_input=$required_file"
      input_status=1
    fi
  done
  if [ ! -x "$regenie_binary" ]; then
    echo "STEP2_SPA_FUSED_ERROR missing_binary=$regenie_binary"
    input_status=1
  fi

  if [ -z "$pheno_column" ] && [ -s "$pheno_file" ]; then
    pheno_column="$(awk '
      NR == 1 {
        for(field = 3; field <= NF; ++field) {
          if($field != "TIME" && $field != "EVENT") {
            print $field
            exit
          }
        }
      }
    ' "$pheno_file")"
  fi
  if [ -z "$pheno_column" ]; then
    echo "STEP2_SPA_FUSED_ERROR missing_binary_phenotype_column=1"
    input_status=1
  fi

  legacy_r1_status=125
  fused_r1_status=125
  fused_r2_status=125
  legacy_r2_status=125
  exact_status=125
  numeric_status=125
  profile_status=125

  if [ "$input_status" -eq 0 ]; then
    run_case legacy_r1 0
    run_case fused_r1 1
    run_case fused_r2 1
    run_case legacy_r2 0
  fi

  legacy_output="$run_dir/legacy_r2_${pheno_column}.regenie"
  fused_output="$run_dir/fused_r2_${pheno_column}.regenie"
  if [ "$legacy_r2_status" -eq 0 ] && [ "$fused_r2_status" -eq 0 ] && \
     [ -f "$legacy_output" ] && [ -f "$fused_output" ]; then
    cmp -s "$legacy_output" "$fused_output"
    exact_status=$?
    python3 "$repo_dir/scripts/compare_numeric_files.py" \
      "$legacy_output" \
      "$fused_output" \
      --rtol 1e-7 \
      --atol 1e-9 \
      --output-significant-digits 6 \
      --engine numpy \
      --report-all \
      > "$numeric_log" 2>&1
    numeric_status=$?
  fi

  profile_status=0
  for label in legacy_r1 legacy_r2; do
    if ! grep -Eq \
      'scope=corrections .*spa_tests=200 .*spa_fast_tests=0 .*spa_fused_cgf_tests=0 spa_fused_cgf_evaluations=0 ' \
      "$run_dir/$label.log"
    then
      profile_status=1
    fi
  done
  for label in fused_r1 fused_r2; do
    if ! grep -Eq \
      'scope=corrections .*spa_tests=200 .*spa_fast_tests=0 .*spa_fused_cgf_tests=200 spa_fused_cgf_evaluations=[1-9][0-9]* ' \
      "$run_dir/$label.log"
    then
      profile_status=1
    fi
  done

  legacy_mean_wall_s="$(awk '
    {
      for(field = 1; field <= NF; ++field) {
        if($field ~ /^wall_s=/) {
          split($field, parts, "=")
          total += parts[2]
          ++count
        }
      }
    }
    END { if(count > 0) printf "%.6f", total / count }
  ' "$run_dir/legacy_r1.time.log" "$run_dir/legacy_r2.time.log" 2>/dev/null)"
  fused_mean_wall_s="$(awk '
    {
      for(field = 1; field <= NF; ++field) {
        if($field ~ /^wall_s=/) {
          split($field, parts, "=")
          total += parts[2]
          ++count
        }
      }
    }
    END { if(count > 0) printf "%.6f", total / count }
  ' "$run_dir/fused_r1.time.log" "$run_dir/fused_r2.time.log" 2>/dev/null)"
  speedup="$(awk \
    -v legacy="$legacy_mean_wall_s" \
    -v fused="$fused_mean_wall_s" '
    BEGIN {
      if(legacy > 0 && fused > 0) printf "%.6f", legacy / fused
    }
  ')"

  validation_status=0
  for required_status in \
    "$input_status" \
    "$legacy_r1_status" \
    "$fused_r1_status" \
    "$fused_r2_status" \
    "$legacy_r2_status" \
    "$numeric_status" \
    "$profile_status"
  do
    if [ "$required_status" -ne 0 ]; then
      validation_status=1
    fi
  done
  return 0
}

profile_summary() {
  local label="$1"
  local profile_log="$run_dir/$label.log"
  awk -v label="$label" '
    function field_value(prefix, field, parts) {
      for(field = 1; field <= NF; ++field) {
        if(index($field, prefix) == 1) {
          split($field, parts, "=")
          return parts[2]
        }
      }
      return ""
    }
    /^STEP2_PROFILE stage=variant_compute/ {
      variant_compute_ms = field_value("elapsed_ms=")
    }
    /^STEP2_PROFILE scope=corrections/ {
      spa_tests = field_value("spa_tests=")
      fast_tests = field_value("spa_fast_tests=")
      fused_tests = field_value("spa_fused_cgf_tests=")
      fused_evaluations = field_value("spa_fused_cgf_evaluations=")
      root_iterations = field_value("spa_root_iterations=")
      spa_thread_ms = field_value("spa_thread_ms=")
    }
    /^STEP2_PROFILE_FINAL/ {
      total_ms = field_value("total_ms=")
    }
    END {
      print "T4_SPA_FUSED_PROFILE label=" label \
        " total_ms=" total_ms \
        " variant_compute_ms=" variant_compute_ms \
        " spa_tests=" spa_tests \
        " fast_tests=" fast_tests \
        " fused_tests=" fused_tests \
        " fused_evaluations=" fused_evaluations \
        " root_iterations=" root_iterations \
        " spa_thread_ms=" spa_thread_ms
    }
  ' "$profile_log" 2>/dev/null
}

benchmark_body > "$driver_log" 2>&1
driver_status=$?

echo "T4_SPA_FUSED_RESULTS_BEGIN"
echo "T4_SPA_FUSED_CONFIG phenotype=$pheno_column threads=$threads variants_file=$extract_file"
for label in legacy_r1 fused_r1 fused_r2 legacy_r2; do
  if [ -f "$run_dir/$label.time.log" ]; then
    cat "$run_dir/$label.time.log"
  fi
  if [ -f "$run_dir/$label.log" ]; then
    profile_summary "$label"
  fi
done
if [ -f "$numeric_log" ]; then
  cat "$numeric_log"
fi
echo "T4_SPA_FUSED_COMPARE exact=$exact_status numeric=$numeric_status profile=$profile_status"
echo "T4_SPA_FUSED_PERFORMANCE legacy_mean_wall_s=$legacy_mean_wall_s fused_mean_wall_s=$fused_mean_wall_s speedup=$speedup"
echo "T4_SPA_FUSED_STATUS input=$input_status legacy_r1=$legacy_r1_status fused_r1=$fused_r1_status fused_r2=$fused_r2_status legacy_r2=$legacy_r2_status driver=$driver_status"
echo "T4_SPA_FUSED_VALIDATION_STATUS=$validation_status"
echo "T4_SPA_FUSED_RESULTS=$run_dir"
echo "T4_SPA_FUSED_RESULTS_END"

exit "$validation_status"
