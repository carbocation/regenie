#!/usr/bin/env bash

### REGENIE TEST SCRIPT 
# Functions used
help_msg="Update to most recent REGENIE version (using 'git pull') and re-compile the software."
fail_msg="Step 1 of REGENIE did not finish successfully."
err_msg="Uh oh, REGENIE did not build successfully. $help_msg"
print_err () { 
  echo "$err_msg"; exit 1 
}
print_simple_err () {
  echo "ERROR: ${1}"; exit 1 
}
print_custom_err () {
  echo "ERROR: ${1} $help_msg"; exit 1 
}
count_lines () {
  awk 'END { print NR }' "$1"
}


### READ OPTIONS
info_msg='
Usage: ./test_bash.sh OPTIONS
          --path  path to Regenie repository
'
REGENIE_PATH=$(pwd) # assume current wd

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --path) REGENIE_PATH="$2"; shift ;;
    -h|--help) echo "$info_msg"; exit 1;;
    *) echo -e "Unknown parameter passed: $1.\n$info_msg"; exit 1 ;;
  esac
  shift
done


# quick check src/example folders are present
if [ ! -d "${REGENIE_PATH}/src" ] || [ ! -d "${REGENIE_PATH}/example" ]; then
  print_simple_err "First input argument must be the directory where Regenie repo was cloned."
else
  cd $REGENIE_PATH
fi 

REGENIE_PATH=$(pwd)/  # use absolute path
mntpt=
regenie_bin=`ls regenie* | head -n 1`

if [ ! -f "$regenie_bin" ]; then
  print_simple_err "Regenie binary cannot be found. Specify the Regenie source path (--path) or compile the software first."
fi

# If compiling was done with Boost Iostream library, use gzipped files as input
if ./$regenie_bin --version | grep -q "gz"; then
  fsuf=.gz
  arg_gz="--gz"
fi


echo -e "==>Running step 1 of REGENIE"
# Prepare regenie command to run for Step 1
## with transposed phenotype file format
#  --tpheno-file ${mntpt}example/tphenotype_bin.txt${fsuf} \
#  --tpheno-indexCol 4 \
#  --tpheno-ignoreCols {1:3} \
basecmd="--step 1 \
  --bed ${mntpt}example/example \
  --exclude ${mntpt}example/snplist_rm.txt \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --bsize 100 \
  --bt $arg_gz"

rgcmd="$basecmd \
  --lowmem \
  --lowmem-prefix tmp_rg \
  --step1-profile \
  --out ${mntpt}test/fit_bin_out"

# run regenie
./$regenie_bin $rgcmd

## quick check that the correct files have been created
if [ ! -f "${REGENIE_PATH}test/fit_bin_out.log" ] || \
  [ ! -f "${REGENIE_PATH}test/fit_bin_out_pred.list" ] || \
  [ ! -f "${REGENIE_PATH}test/fit_bin_out_1.loco$fsuf" ] || \
  [ ! -f "${REGENIE_PATH}test/fit_bin_out_2.loco$fsuf" ]; then
  print_custom_err "$fail_msg"
elif [ "`grep \"0.4504\" ${REGENIE_PATH}test/fit_bin_out.log | grep \"min value\"`" = "" ]; then
  print_custom_err "$fail_msg"
elif [ "`grep -c '^STEP1_PROFILE stage=' ${REGENIE_PATH}test/fit_bin_out.log`" != "11" ]; then
  print_custom_err "Step 1 profiling output is incomplete."
elif ! grep -q '^STEP1_PROFILE version=9 backend=cpu mode=loocv ' ${REGENIE_PATH}test/fit_bin_out.log; then
  print_custom_err "Step 1 profiling header is missing."
# Scope records are path-dependent; require the scopes exercised by this
# LOOCV regression instead of assuming every optional scope is present.
elif ! grep -q '^STEP1_PROFILE scope=genotype_preprocess ' ${REGENIE_PATH}test/fit_bin_out.log; then
  print_custom_err "Step 1 genotype preprocessing profiling is missing."
elif ! grep -q '^STEP1_PROFILE scope=cv_matrices ' ${REGENIE_PATH}test/fit_bin_out.log; then
  print_custom_err "Step 1 CV matrix profiling is missing."
elif ! grep -q '^STEP1_PROFILE scope=level0_ridge ' ${REGENIE_PATH}test/fit_bin_out.log; then
  print_custom_err "Step 1 Level 0 ridge profiling is missing."
elif ! grep -q '^STEP1_PROFILE scope=prediction_output ' ${REGENIE_PATH}test/fit_bin_out.log; then
  print_custom_err "Step 1 prediction output profiling is missing."
elif ! grep -q '^STEP1_PROFILE_FINAL version=1 backend=cpu ' ${REGENIE_PATH}test/fit_bin_out.log; then
  print_custom_err "Step 1 final profiling output is missing."
fi

#### Run step 1 splitting across jobs for level 0
njobs=4
echo -e "==>Re-running step 1 splitting in $njobs jobs"
# pt1 - run regenie before l0
rgcmd="$basecmd \
  --split-l0 ${mntpt}test/fit_bin_parallel,$njobs \
  --out ${mntpt}test/fit_bin_l0"

./$regenie_bin $rgcmd
if [ ! -f "${REGENIE_PATH}test/fit_bin_parallel.master" ]; then
  print_custom_err "$fail_msg"
fi

# pt2 - run regenie for l0
nj=`seq 1 $njobs`
for job in $nj; do
  rgcmd="$basecmd \
    --run-l0 ${mntpt}test/fit_bin_parallel.master,$job \
    --out ${mntpt}test/fit_bin_l0"

  ./$regenie_bin $rgcmd
  if [ ! -f "${REGENIE_PATH}test/fit_bin_parallel_job${job}_l0_Y1" ]; then
    print_custom_err "$fail_msg"
  fi
done


# pt3 - run regenie for l1
rgcmd="$basecmd \
  --run-l1 ${mntpt}test/fit_bin_parallel.master \
  --out ${mntpt}test/fit_bin_l1"

./$regenie_bin $rgcmd

if [ ! -f "${REGENIE_PATH}test/fit_bin_l1_1.loco$fsuf" ]; then
  print_custom_err "$fail_msg"
elif ! cmp --silent \
  "${REGENIE_PATH}test/fit_bin_out_1.loco$fsuf" \
  "${REGENIE_PATH}test/fit_bin_l1_1.loco$fsuf" 
then
  print_custom_err "$fail_msg"
elif ! cmp --silent \
  "${REGENIE_PATH}test/fit_bin_out_2.loco$fsuf" \
  "${REGENIE_PATH}test/fit_bin_l1_2.loco$fsuf" 
then
  print_custom_err "$fail_msg"
fi



##########
##########
#### Step 2
i=1
echo -e "==>Running step 2 of REGENIE; test #$i"
rgcmd="--step 2 \
  --bgen ${mntpt}example/example.bgen \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --bsize 200 \
  --bt \
  --firth --approx \
  --pThresh 0.01 \
  --pred ${mntpt}test/fit_bin_out_pred.list \
  $arg_gz \
  --out ${mntpt}test/test_bin_out_firth"

# run regenie
./$regenie_bin $rgcmd

##  do this way so zcat works on OSX
if [ -f ${REGENIE_PATH}test/test_bin_out_firth_Y1.regenie.gz ]; then
  ( zcat < ${REGENIE_PATH}test/test_bin_out_firth_Y1.regenie.gz ) > ${REGENIE_PATH}test/test_bin_out_firth_Y1.regenie
fi

if [ "$(count_lines "${REGENIE_PATH}test/test_bin_out_firth_Y1.regenie")" != "1001" ]
then
  print_err
fi


(( i++ ))
echo -e "\n==>Running test #$i\n"
# PGEN binary traits with complete phenotype data
for file_type in bed pgen; do
  rgcmd="--step 2 \
    --${file_type} ${mntpt}example/example \
    --covarFile ${mntpt}example/covariates.txt${fsuf} \
    --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
    --remove ${mntpt}example/fid_iid_to_remove.txt \
    --bsize 200 \
    --bt \
    --ignore-pred \
    --out ${mntpt}test/test_bin_out_pgen_bt_complete_${file_type}"

  ./$regenie_bin $rgcmd
done

for phenotype in Y1 Y2; do
  if ! cmp --silent \
    ${REGENIE_PATH}test/test_bin_out_pgen_bt_complete_bed_${phenotype}.regenie \
    ${REGENIE_PATH}test/test_bin_out_pgen_bt_complete_pgen_${phenotype}.regenie
  then
    print_err
  fi
done


(( i++ ))
echo -e "\n==>Running test #$i\n"
# PGEN quantitative traits with complete phenotype data
rgcmd="--step 2 \
  --pgen ${mntpt}example/example \
  --covarFile ${mntpt}example/covariates.txt \
  --phenoFile ${mntpt}example/phenotype.txt \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --bsize 200 \
  --ignore-pred \
  --out ${mntpt}test/test_bin_out_pgen_qt_complete"

./$regenie_bin $rgcmd

for phenotype in Y1 Y2; do
  if [ "$(count_lines "${REGENIE_PATH}test/test_bin_out_pgen_qt_complete_${phenotype}.regenie")" != "1001" ]; then
    print_err
  fi
done


(( i++ ))
echo -e "\n==>Running test #$i\n"
# PGEN binary traits with phenotype-specific missingness
for file_type in bed pgen; do
  rgcmd="--step 2 \
    --${file_type} ${mntpt}example/example \
    --covarFile ${mntpt}example/covariates.txt${fsuf} \
    --phenoFile ${mntpt}example/phenotype_bin_wNA.txt \
    --remove ${mntpt}example/fid_iid_to_remove.txt \
    --bsize 200 \
    --bt \
    --ignore-pred \
    --out ${mntpt}test/test_bin_out_pgen_bt_missing_${file_type}"

  ./$regenie_bin $rgcmd
done

for phenotype in Y1 Y2; do
  if ! cmp --silent \
    ${REGENIE_PATH}test/test_bin_out_pgen_bt_missing_bed_${phenotype}.regenie \
    ${REGENIE_PATH}test/test_bin_out_pgen_bt_missing_pgen_${phenotype}.regenie
  then
    print_err
  fi
done


(( i++ ))
echo -e "\n==>Running test #$i\n"
# PGEN quantitative traits with phenotype-specific missingness
rgcmd="--step 2 \
  --pgen ${mntpt}example/example \
  --covarFile ${mntpt}example/covariates.txt \
  --phenoFile ${mntpt}example/phenotype_bin_wNA.txt \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --bsize 200 \
  --force-qt \
  --ignore-pred \
  --out ${mntpt}test/test_bin_out_pgen_qt_missing"

./$regenie_bin $rgcmd

for phenotype in Y1 Y2; do
  if [ "$(count_lines "${REGENIE_PATH}test/test_bin_out_pgen_qt_missing_${phenotype}.regenie")" != "1001" ]; then
    print_err
  fi
done


(( i++ ))
echo -e "\n==>Running test #$i\n"
# complete-sample BGEN dosage decoding in both allele orientations
for ref_mode in default ref_first; do
  ref_arg=""
  if [ "$ref_mode" = "ref_first" ]; then
    ref_arg="--ref-first"
  fi

  rgcmd="--step 2 \
    --bgen ${mntpt}example/example.bgen \
    --covarFile ${mntpt}example/covariates.txt${fsuf} \
    --phenoFile ${mntpt}example/phenotype.txt \
    --bsize 200 \
    --ignore-pred \
    $ref_arg \
    --out ${mntpt}test/test_bin_out_bgen_qt_${ref_mode}_bgen"
  ./$regenie_bin $rgcmd

  for phenotype in Y1 Y2; do
    if [ "$(count_lines "${REGENIE_PATH}test/test_bin_out_bgen_qt_${ref_mode}_bgen_${phenotype}.regenie")" != "1001" ]
    then
      print_err
    fi
  done
done


(( i++ ))
echo -e "\n==>Running test #$i\n"
# phenotype missingness retains the general BGEN decoder
rgcmd="--step 2 \
  --bgen ${mntpt}example/example.bgen \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin_wNA.txt \
  --bsize 200 \
  --force-qt \
  --ignore-pred \
  --out ${mntpt}test/test_bin_out_bgen_qt_missing"
./$regenie_bin $rgcmd

for phenotype in Y1 Y2; do
  if [ "$(count_lines "${REGENIE_PATH}test/test_bin_out_bgen_qt_missing_${phenotype}.regenie")" != "1001" ]
  then
    print_err
  fi
done


(( i++ ))
echo -e "\n==>Running test #$i\n"
# saddlepoint approximation
rgcmd="--step 2 \
  --bgen ${mntpt}example/example.bgen \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --bsize 200 \
  --bt \
  --spa \
  --pThresh 0.01 \
  --pred ${mntpt}test/fit_bin_out_pred.list \
  $arg_gz \
  --out ${mntpt}test/test_bin_out_spa"

# run regenie
./$regenie_bin $rgcmd

if [ -f ${REGENIE_PATH}test/test_bin_out_spa_Y1.regenie.gz ]; then
  ( zcat < ${REGENIE_PATH}test/test_bin_out_spa_Y1.regenie.gz ) > ${REGENIE_PATH}test/test_bin_out_spa_Y1.regenie
fi

if [ "$(count_lines "${REGENIE_PATH}test/test_bin_out_spa_Y1.regenie")" != "1001" ]; then
  print_err
elif ! grep -Eq '^Number of tests with SPA correction : [1-9][0-9]*$' ${REGENIE_PATH}test/test_bin_out_spa.log; then
  print_err
elif ! grep -Eq '^Number of failed tests : \(0/[1-9][0-9]*\)$' ${REGENIE_PATH}test/test_bin_out_spa.log; then
  print_err
fi


(( i++ ))
echo -e "\n==>Running test #$i\n"
# interaction tests
rgcmd="--step 2 \
  --bed ${mntpt}example/example \
  --covarFile ${mntpt}example/covariates_wBin.txt \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --bsize 200 \
  --force-qt \
  --ignore-pred \
  --covarColList V1,V5 \
  --catCovarList V5 \
  --interaction V5 \
  --out ${mntpt}test/test_bin_out_inter"

# run regenie
./$regenie_bin $rgcmd

if [ `grep "^1 1 .*ADD-INT" ${REGENIE_PATH}test/test_bin_out_inter_Y1.regenie | wc -l` != 5 ]; then
  print_err
fi


(( i++ ))
echo -e "\n==>Running test #$i\n"
# interaction tests
rgcmd="--step 2 \
  --bed ${mntpt}example/example --ref-first \
  --covarFile ${mntpt}example/covariates.txt \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --bsize 200 \
  --ignore-pred \
  --force-qt \
  --interaction-snp 1 \
  --out ${mntpt}test/test_bin_out_inter"

# run regenie
./$regenie_bin $rgcmd

rgcmd+="2 --interaction-file bgen,example/example.bgen --interaction-file-reffirst"
# run regenie
./$regenie_bin $rgcmd

if ! cmp --silent \
  ${REGENIE_PATH}test/test_bin_out_inter_Y1.regenie \
  ${REGENIE_PATH}test/test_bin_out_inter2_Y1.regenie 
then
  print_err
fi


(( i++ ))
echo -e "==>Running test #$i"
# Next test
basecmd="--step 2 \
  --bed ${mntpt}example/example_3chr \
  --ref-first \
  --covarFile ${mntpt}example/covariates_wBin.txt \
  --covarColList V{1:2},V4 \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --phenoColList Y2 \
  --bsize 100 \
  --test dominant \
  --force-qt \
  --ignore-pred"
rgcmd="$basecmd \
  --chrList 2,3 \
  --write-samples \
  --print-pheno \
  --out ${mntpt}test/test_out"

# run regenie
./$regenie_bin $rgcmd

# check files
if [ ! -f "${REGENIE_PATH}test/test_out_Y2.regenie.ids" -o -f "${REGENIE_PATH}test/test_out_Y1.regenie.ids" ]
then
  print_err
elif (( $(head -n 1 ${REGENIE_PATH}test/test_out_Y2.regenie.ids | cut -f1) != "Y2" )); then
  print_err
elif (( $(head -n 1 "${REGENIE_PATH}test/test_out_Y2.regenie.ids" | tr '\t' '\n' | wc -l) != 2 )); then
  print_err
elif (( `grep "mog_" "${REGENIE_PATH}test/test_out_Y2.regenie" | wc -l` > 0 )); then
  print_err
elif (( `grep "ADD" "${REGENIE_PATH}test/test_out_Y2.regenie" | wc -l` > 0 )); then
  print_err
elif [ "`cut -d ' ' -f1-5 ${REGENIE_PATH}test/test_out_Y2.regenie | sed '2q;d'`" != "`grep \"^2\" ${REGENIE_PATH}example/example_3chr.bim | head -n 1 | awk '{print $1,$4,$2,$5,$6}'`" ]; then
  print_err
fi


(( i++ ))
echo -e "==>Running test #$i"
# Next test
rgcmd="$basecmd \
  --force-qt \
  --catCovarList V4 \
  --extract ${mntpt}test/test_out.snplist \
  --out ${mntpt}test/test_out_extract"

awk '{if($1!=1) {print $2}}'  ${REGENIE_PATH}example/example_3chr.bim > ${REGENIE_PATH}test/test_out.snplist

# run regenie
./$regenie_bin $rgcmd

if ! cmp --silent \
  ${REGENIE_PATH}test/test_out_Y2.regenie \
  ${REGENIE_PATH}test/test_out_extract_Y2.regenie 
then
  print_err
elif (( `grep "n_cov = 3" "${REGENIE_PATH}test/test_out_extract.log" | wc -l` != 1 )); then
  print_err
fi

(( i++ ))
echo -e "==>Running test #$i"
# First command (V1)
rgcmd="--step 2 \
  --bed ${mntpt}example/example_3chr_masks \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --bsize 10 \
  --ignore-pred \
  --force-qt \
  --htp TEST \
  --out ${mntpt}test/test_out_masks_V1"
# run regenie
./$regenie_bin $rgcmd

# Second command (V2)
# build masks
awk '{print $4}' ${mntpt}example/example_3chr.setlist | tr ',' '\n' > ${REGENIE_PATH}test/tmp1.txt 
rgcmd="--step 2 \
  --ignore-pred \
  --bed ${mntpt}example/example_3chr \
  --extract ${mntpt}test/tmp1.txt \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --set-list ${mntpt}example/example_3chr.setlist \
  --anno-file ${mntpt}example/example_3chr.annotations \
  --mask-def ${mntpt}example/example_3chr.masks \
  --write-mask \
  --write-setlist ${mntpt}example/example_3chr.write_sets \
  --force-qt \
  --bsize 15 \
  --aaf-bins 0.2 \
  --chrList 1,3 \
  --htp TEST \
  --write-mask-snplist \
  --out ${mntpt}test/test_out_masks_V2"

# run regenie
./$regenie_bin $rgcmd 

head -n 3 ${REGENIE_PATH}test/test_out_masks_V2_Y1.regenie | tail -n 2 | cut -f1-3,6- > ${REGENIE_PATH}test/tmp1.txt
tail -n 1 ${REGENIE_PATH}test/test_out_masks_V2_Y1.regenie | cut -f1-3,6- >> ${REGENIE_PATH}test/tmp1.txt
cut -f1-3,6- ${REGENIE_PATH}test/test_out_masks_V1_Y1.regenie > ${REGENIE_PATH}test/tmp2.txt

if ! cmp --silent \
  ${REGENIE_PATH}test/tmp1.txt \
  ${REGENIE_PATH}test/tmp2.txt ; then
  print_err
elif [ ! -f ${REGENIE_PATH}test/test_out_masks_V2_masks.bed ]; then
  print_err
elif [ "$(hexdump -e \"%07_ax\ \"\ 16/1\ \"\ %02x\"\ \"\\n\"  -n 3 ${REGENIE_PATH}test/test_out_masks_V2_masks.bed | head -n 1 | awk '{print $2,$3,$4}' | tr ' ' ',')" != "6c,1b,01" ]; then
  print_err
elif [ "$(count_lines "${REGENIE_PATH}test/test_out_masks_V2_masks.bim"),$(count_lines "${REGENIE_PATH}test/test_out_masks_V2_masks.fam")" != "4,494" ]; then
  print_err
elif [ ! -f ${REGENIE_PATH}test/test_out_masks_V2_masks.snplist ]; then
  print_err
elif [ "$(awk -F, 'NR == 1 { print NF; exit }' "${REGENIE_PATH}test/test_out_masks_V2_tmp2.setlist")" != "2" ]; then
  print_err
fi


(( i++ ))
echo -e "==>Running test #$i"
# build masks
awk '{print $4}' ${mntpt}example/example_3chr.setlist | tr ',' '\n' > ${REGENIE_PATH}test/tmp1.txt 
rgcmd="--step 2 \
  --ignore-pred \
  --bed ${mntpt}example/example_3chr \
  --extract ${mntpt}test/tmp1.txt \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --set-list ${mntpt}example/example_3chr.setlist \
  --anno-file ${mntpt}example/example_3chr.annotations \
  --mask-def ${mntpt}example/example_3chr.masks \
  --mask-lovo SET1,M1,0.2 \
  --force-qt \
  --htp TEST \
  --out ${mntpt}test/test_out_masks_loo"

# run regenie
./$regenie_bin $rgcmd 

if [ ! -f ${REGENIE_PATH}test/test_out_masks_loo_Y1.regenie ]; then
  print_err
elif [ `cat ${REGENIE_PATH}test/test_out_masks_loo_Y1.regenie | wc -l` != 21 ]; then
  print_err
elif [ `grep "_mog" ${REGENIE_PATH}test/test_out_masks_loo_Y1.regenie | wc -l` != 18 ]; then
  print_err
fi


(( i++ ))
echo -e "==>Running test #$i"
# build masks using set domains
rgcmd="--step 2 \
  --ignore-pred \
  --bed ${mntpt}example/example_3chr \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --remove ${mntpt}example/fid_iid_to_remove.txt \
  --set-list ${mntpt}example/example_3chr.setlist \
  --anno-file ${mntpt}example/example_3chr.annotationsV2 \
  --mask-def ${mntpt}example/example_3chr.masks \
  --check-burden-files \
  --force-qt \
  --bsize 20 \
  --aaf-bins 0.2 \
  --out ${mntpt}test/test_out_masks_V3"

# run regenie
./$regenie_bin $rgcmd 

if ! [[ "`head -n 1 ${REGENIE_PATH}test/test_out_masks_V3_Y1.regenie`" =~ ^\#\#MASKS.* ]]
then
  print_err
elif [ `grep "SET2.*.M1" ${REGENIE_PATH}test/test_out_masks_V3_Y1.regenie | wc -l` != "4" ]
then
  print_err
elif [ `grep -e "->Detected 1" ${REGENIE_PATH}test/test_out_masks_V3_masks_report.txt | wc -l` != "4" ]
then
  print_err
fi

(( i++ ))
echo -e "==>Running test #$i"
# conditional analyses
rgcmd="${basecmd/_3chr/} \
  --condition-list ${mntpt}example/snplist_rm.txt \
  --sex-specific female \
  --out ${mntpt}test/test_out_cond"

# run regenie
./$regenie_bin $rgcmd

rgcmd="${basecmd/_3chr/} \
  --condition-list ${mntpt}example/snplist_rm.txt \
  --condition-file pgen,${mntpt}example/example \
  --out ${mntpt}test/test_out_cond2"

# run regenie
./$regenie_bin $rgcmd

if ! cmp --silent \
  ${REGENIE_PATH}test/test_out_cond_Y2.regenie \
  ${REGENIE_PATH}test/test_out_cond2_Y2.regenie 
then
  print_err
elif [ `grep "n_used = 6" ${REGENIE_PATH}test/test_out_cond*log | wc -l` != "2" ]; then
  print_err
fi


# with skat
(( i++ ))
echo -e "==>Running test #$i"
rgcmd="--step 2 \
  --ignore-pred \
  --bed ${mntpt}example/example_3chr \
  --covarFile ${mntpt}example/covariates.txt${fsuf} \
  --phenoFile ${mntpt}example/phenotype_bin.txt${fsuf} \
  --phenoCol Y1 \
  --set-list ${mntpt}example/example_3chr.setlist \
  --anno-file ${mntpt}example/example_3chr.annotations \
  --mask-def ${mntpt}example/example_3chr.masks \
  --vc-tests skat \
  --force-qt \
  --bsize 15 \
  --aaf-bins 0.2 \
  --write-mask-snplist \
  --out ${mntpt}test/test_out_vc"

# run regenie
./$regenie_bin $rgcmd 

if ! grep -q "all.*SKAT" ${REGENIE_PATH}test/test_out_vc_Y1.regenie  
then
  print_err
fi

##############
echo "SUCCESS: REGENIE build passed the tests!"
# file cleanup
rm -f ${REGENIE_PATH}test/fit_bin_* ${REGENIE_PATH}test/test_bin_out* ${REGENIE_PATH}test/test_out* ${REGENIE_PATH}test/tmp[12].txt
