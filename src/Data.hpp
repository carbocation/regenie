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

#ifndef DATA_H
#define DATA_H

class Step1ComputeBackend;

struct Step1GroupedPredictionProfile {
  uint64_t calls = 0;
  uint64_t design_uploads = 0;
  uint64_t design_upload_bytes = 0;
  double wall_ms = 0;
  double upload_ms = 0;
  double compute_ms = 0;
  double download_ms = 0;
  double host_materialization_ms = 0;
};

struct Step1Profile {
  uint64_t blocks = 0;
  uint64_t variants = 0;
  double total_ms = 0;
  double decode_ms = 0;
  double residualize_ms = 0;
  double gram_ms = 0;
  double gty_ms = 0;
  double backend_upload_ms = 0;
  double backend_download_ms = 0;
  double eigensolve_ms = 0;
  double association_ms = 0;
  double ridge_ms = 0;
  double backend_ridge_compute_ms = 0;
  double cv_wall_ms = 0;
  double cv_backend_compute_ms = 0;
  double cv_transfer_ms = 0;
  double cv_host_orchestration_ms = 0;
  double ridge_wall_ms = 0;
  double ridge_eigensolve_ms = 0;
  double ridge_transfer_ms = 0;
  double ridge_backend_compute_ms = 0;
  double ridge_host_orchestration_ms = 0;
  uint64_t ridge_cholesky_folds = 0;
  uint64_t ridge_batched_cholesky_blocks = 0;
  uint64_t ridge_eigendecomposition_folds = 0;
  uint64_t preprocess_backend_blocks = 0;
  uint64_t preprocess_fallback_blocks = 0;
  double preprocess_wall_ms = 0;
  double preprocess_backend_compute_ms = 0;
  double preprocess_upload_ms = 0;
  double preprocess_download_ms = 0;
  double preprocess_host_orchestration_ms = 0;
  double preprocess_data_setup_ms = 0;
  double preprocess_backend_wall_ms = 0;
  double preprocess_data_finalize_ms = 0;
  uint64_t preprocess_pinned_staging_upload_count = 0;
  uint64_t preprocess_pinned_staging_upload_bytes = 0;
  uint64_t preprocess_packed_hardcall_blocks = 0;
  uint64_t preprocess_packed_hardcall_upload_bytes = 0;
  double preprocess_packed_hardcall_expand_ms = 0;
  double preprocess_packed_hardcall_validation_ms = 0;
  double preprocess_packed_hardcall_allocation_ms = 0;
  double preprocess_packed_hardcall_host_prepare_ms = 0;
  double preprocess_packed_hardcall_backend_wall_ms = 0;
  uint64_t pgen_prefetched_blocks = 0;
  double pgen_prefetch_service_ms = 0;
  double pgen_prefetch_wait_ms = 0;
  double initialization_ms = 0;
  double level0_wall_ms = 0;
  double level1_prepare_ms = 0;
  double level1_fit_ms = 0;
  double output_ms = 0;
  uint64_t prediction_output_rows = 0;
  uint64_t prediction_output_values = 0;
  uint64_t prediction_output_threads = 0;
  double prediction_output_format_ms = 0;
  double prediction_output_write_ms = 0;
  Step1GroupedPredictionProfile grouped_prediction;
  double end_to_end_ms = 0;
};

struct Step2Profile {
  uint64_t chromosomes = 0;
  uint64_t blocks = 0;
  uint64_t variants = 0;
  uint64_t corrected_tests = 0;
  uint64_t failed_tests = 0;
  uint64_t logistic_firth_null_fits = 0;
  uint64_t cox_firth_null_fits = 0;
  double setup_ms = 0;
  double prediction_read_ms = 0;
  double null_model_ms = 0;
  double genotype_io_ms = 0;
  double variant_compute_ms = 0;
  double output_ms = 0;
  double logistic_firth_null_ms = 0;
  double cox_firth_null_ms = 0;
  double end_to_end_ms = 0;
};

struct Step2VariantComputeProfile {
  uint64_t variants = 0;
  uint64_t sparse_variants = 0;
  uint64_t unscaled_dense_qt_variants = 0;
  uint64_t shared_denom_dense_qt_variants = 0;
  double thread_work_ms = 0;
  double parse_thread_ms = 0;
  double preprocess_thread_ms = 0;
  double score_thread_ms = 0;
  double interaction_thread_ms = 0;
};

class Data {

  public:
    // class elements
    mstream sout;
    MeasureTime runtime;
    param params;
    in_files files;
    filter in_filters;
    std::vector<snp> snpinfo;
    phenodt pheno_data;
    geno_block Gblock;
    std::map<int, std::vector<int>> chr_map; // first=chr; second=[# SNPs analyzed, #blocks, # SNPs in file]
    ests m_ests;
    ridgel1 l1_ests;
    f_ests firth_est;
    // HLM
    HLM nullHLM; // for null model fitting of HLM
    remeta_sumstat_writer remeta_sumstats;

    std::string model_type, correction_type, test_string, wgr_string;

    uint32_t n_corrected = 0; // to keep track of how many SNPs require correction
    bool pval_converged = false; // keep track of whether SPA/Firth converged
    bool fastSPA; // use fast approx. for rare SNPs
    bool step2_bgen_fast_path_initialized = false;
    bool step2_bgen_fast_path_eligible = false;
    bool step2_bgen_lookup_path_enabled = false;

    std::vector < MatrixXb > masked_in_folds;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > predictions;

    uint32_t total_chrs_loco;
    Eigen::MatrixXd blup;
    Eigen::VectorXd denum_tstat;
    Eigen::MatrixXd res, stats, W_hat;
    Eigen::RowVectorXd p_sd_yres;
    Eigen::VectorXd scale_G; // keep track of sd(Y) (1xP) and sd(G) (M*1)
    MultiPhen mphen;
    Step1Profile step1_profile;
    Step1PgenReadProfile step1_pgen_read_profile;
    Step2Profile step2_profile;
    Step2PgenReadProfile step2_pgen_read_profile;
    Step2BgenParseProfile step2_bgen_parse_profile;
    Step2VariantComputeProfile step2_variant_compute_profile;
    std::unique_ptr<Step1ComputeBackend> step1_compute_backend;

    // function definitions
    void run();
    void run_step1();
    void run_step2();
    void print_step2_profile();

    void file_read_initialization();
    void residualize_genotypes();
    void scale_genotypes(bool);
    void get_block_size(int const&,int const&,int const&,int&);

    // step 1 
    void set_parallel_l0();
    void write_l0_master();
    void prep_parallel_l0();
    void prep_parallel_l1();
    void set_blocks();
    void set_folds();
    void setmem();
    void calc_cv_matrices(struct ridgel0*);
    void level_0_calculations();
    void print_step1_profile();
    void print_step1_final_profile();
    void prep_l1_models();
    void write_inputs(); 
    void exit_early();
    // output of step 1
    void output();
    void make_predictions(int const&,int const&);
    void make_predictions_loocv(int const&,int const&);
    void make_predictions_binary(int const&,int const&);
    void make_predictions_binary_loocv_full(int const&,int const&);
    void make_predictions_binary_loocv(int const&,int const&);
    void make_predictions_count(int const&,int const&);
    void make_predictions_count_loocv(int const&,int const&);
    void make_predictions_cox(int const&, int const&);
    void step1_grouped_predict(
      const Eigen::Ref<const Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::VectorXd>&,
      const Eigen::Ref<const Eigen::VectorXi>&,
      const Eigen::Ref<const Eigen::VectorXi>&,
      Eigen::MatrixXd&);
    void print_snp_betas(const Eigen::Ref<const Eigen::VectorXd>&);
    void write_predictions(int const&);
    std::string write_ID_header(std::vector<uint32_t>&);
    std::string write_chr_row(int const&,int const&,
      const Eigen::Ref<const Eigen::VectorXd>&,
      const std::vector<uint32_t>&);
    void rm_l0_files(int const& ph);

    // step 2 main functions
    void test_snps();
    void set_blocks_for_testing();
    void print_test_info();
    void set_nullreg_mat();
    void compute_res();
    void residualize_res();
    void compute_res_bin(int const&);
    void compute_res_count(int const&);
    void compute_res_cox(int const&);
    void setup_output(Files*,std::string&,std::vector<std::shared_ptr<Files>>&,std::vector<std::string>&);

    // step 2 using multithreading in eigen
    double check_pval(double const&,int const&,int const&,int const&);
    double run_firth_correction(int const&,int const&,int const&);
    void run_SPA_test(int const&);

    // step2 using multithreading in openmp
    void test_snps_fast();
    void analyze_block(int const&,int const&,tally*,std::vector<variant_block>&);
    void compute_tests_mt(int const&,std::vector<uint64>,std::vector<std::vector <uchar>>&,std::vector<uint32_t>,std::vector<uint32_t>&,std::vector<variant_block>&);
    void compute_tests_st(int const&,std::vector<uint64>,std::vector<std::vector <uchar>>&,std::vector<uint32_t>,std::vector<uint32_t>&,std::vector<variant_block>&);

    // step 2 with joint tests
    JTests jt;
    GenoMask bm;
    void test_joint();
    void set_groups_for_testing();
    void get_sum_stats(int const&,int const&,std::vector<variant_block>&);
    void readChunk(std::vector<uint64>&,int const&,std::vector<std::vector<uchar>>&,std::vector<uint32_t>&,std::vector<uint32_t>&,std::vector<variant_block>&);
    void getMask(int const&,int const&,std::vector<std::vector<uchar>>&,std::vector<uint32_t>&,std::vector<uint32_t>&,std::vector<variant_block>&);
    void getMask_loo(int const&,int const&,std::vector<std::vector<uchar>>&,std::vector<uint32_t>&,std::vector<uint32_t>&,std::vector<variant_block>&);

    // step 2 with multi-trait tests
    MTests mt;
    void test_multitrait();
    void analyze_block_multitrait(int const&,int const&,tally*,std::vector<variant_block>&);
    void compute_tests_mt_multitrait(int const&,std::vector<uint64>,std::vector<std::vector <uchar>>&,std::vector<uint32_t>,std::vector<uint32_t>&,std::vector<variant_block>&);
    void prep_multitrait(); 

    // step 2 with MultiPhen test
    /* MTests mt; */
    void test_multiphen();
    void analyze_block_multiphen(int const&,int const&,tally*,std::vector<variant_block>&);
    void compute_tests_mt_multiphen(int const&,std::vector<uint64>,std::vector<std::vector <uchar>>&,std::vector<uint32_t>,std::vector<uint32_t>&,std::vector<variant_block>&);
    void prep_multiphen(); 
    void set_multiphen();

    // for LD computation
    void ld_comp();
    void get_G_indices(Eigen::ArrayXi&,std::map<std::string,int>&);
    void write_snplist(ArrayXb&);
    // dosage-mode
    void compute_ld_dosages(Files*);
    void get_G_masks(SpMat&,ArrayXb&,std::map<std::string,int>&);
    void get_G_svs(int const&,int const&);
    void print_ld(MatrixXd&,Eigen::ArrayXi&,ArrayXb&,Files*);
    // hard-call mode
    void compute_ld_hardcalls(Files*);
    void get_G_svs(SpMat&,ArrayXb&,std::map<std::string,int>&);
    void get_G_masks_hc(SpMat&,ArrayXb&,std::map<std::string,int>&);
    void print_ld(SpMat&,Eigen::ArrayXi&,ArrayXb&,Files*);
    
    Data();
    ~Data();
};

// extra function
std::string get_fullpath(std::string);

#endif
