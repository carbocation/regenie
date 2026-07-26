#ifndef COXL2_H
#define COXL2_H
#include "Regenie.hpp"
#include "Step1_Cox_Path_Newton.hpp"
#include "Step1_Level1_Optimizer.hpp"

class Step1ComputeBackend;
struct Step1ComputeTimings;
class survival_data;

class cox_ridge_path {
    public:
        // coefficients
        Eigen::MatrixXd beta_mx;
        Eigen::MatrixXd eta_mx;
        
        Eigen::VectorXd lambda_vec;

        // objective value
        Eigen::VectorXd object_val;
        Eigen::VectorXd deviance;
        Eigen::VectorXd dev_ratio;
        Eigen::Array<bool, Eigen::Dynamic, 1> converge;
        Step1CoxPathNewtonStats path_newton_stats;
        cox_ridge_path(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask, const int& nlambda = 100, const double& lambda_min_max_ratio = -1, const Eigen::VectorXd& lambda = Eigen::VectorXd(), const int& max_iter = 100, const int& max_inner_iter = 30, const double& tolerance = 1e-6, const bool& verbose_fit = false, Step1ComputeBackend* compute_backend = nullptr, const bool& resident_design = false, Step1ComputeTimings* timings = nullptr, Step1Level1Optimizer optimizer = Step1Level1Optimizer::Irls);
        void fit(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask);
        
        // fitting info
        int niter, mxitnr;
        double tol;
        bool verbose;

    private:
        double _getCoxLambdaMax(const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& gradient);
        int _lambda_len;
        bool _user_define_lambda = false;
        Step1ComputeBackend* _compute_backend = nullptr;
        bool _resident_design = false;
        Step1ComputeTimings* _timings = nullptr;
        Step1Level1Optimizer _optimizer = Step1Level1Optimizer::Irls;
};

class cox_ridge {
    public:
        // coefficients
        Eigen::VectorXd beta;
        // prediction
        Eigen::VectorXd eta, eta_order;
        double lambda;
        bool converge;
        double dev_ratio;

        cox_ridge(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask, const double& lambda_val, const int& max_iter = 100, const int& max_inner_iter = 30, const double& tolerance = 1e-6, const bool& verbose_obj = false, const Eigen::VectorXd& beta_init = Eigen::VectorXd(), const double& null_deviance = -999, Step1ComputeBackend* compute_backend = nullptr, const bool& resident_design = false, Step1ComputeTimings* timings = nullptr);
        void fit(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask);
        void reset(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask, const double& lambda_val, const Eigen::VectorXd& beta_init = Eigen::VectorXd(), const double& null_deviance = -999);
        void coxGrad(const survival_data& survivalData);
        Eigen::VectorXd get_gradient();
        Eigen::VectorXd get_object_all();
        Eigen::VectorXd get_deviance_all();
        double get_object();
        double get_deviance();
        double get_null_deviance();

    private:
        // gradient
        Eigen::VectorXd _gradient, _diagHessian;
        int _niter, _mxitnr;
        double _tol;
        bool _verbose;
        Step1ComputeBackend* _compute_backend = nullptr;
        bool _resident_design = false;
        Step1ComputeTimings* _timings = nullptr;
        double _null_deviance = 0;
        uint64_t _weighted_solve_count = 0;
        // objective value
        Eigen::VectorXd _object;
        Eigen::VectorXd _deviance;
        
        friend class cox_ridge_path;
        void reset_from_path(
          const survival_data& survivalData,
          const Eigen::MatrixXd& Xmat,
          const Eigen::VectorXd& offset_val,
          const ArrayXb& mask,
          double lambda_val,
          const Eigen::VectorXd& beta_init,
          double null_deviance,
          double starting_deviance);
        bool exact_path_point(
          const survival_data& survivalData,
          const Eigen::MatrixXd& Xmat,
          Step1CoxPathNewtonPoint& point);
        void install_path_point(
          const Step1CoxPathNewtonPoint& point,
          double lambda_val);
        double _coxDevianceFromLoglik(
          const survival_data& survivalData,
          double loglik);
        double _coxDeviance(const survival_data& survivalData);
        double _coxLoglik(const survival_data& survivalData);
};

#endif
