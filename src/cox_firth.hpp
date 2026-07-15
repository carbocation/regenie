#ifndef COXFIRTH_H
#define COXFIRTH_H

class cox_firth {
    public:
		int p;
        // coefficients
        Eigen::VectorXd beta;
        // prediction
        Eigen::VectorXd eta, eta_order, residual;
        int iter;
        bool converge = false;

        // prepare for test
        Eigen::VectorXd exp_eta;
        Eigen::VectorXd mu;
		Eigen::VectorXd loglike;
		double loglik_val;
		Eigen::VectorXd first_der;
		Eigen::MatrixXd second_der;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrsd, qrsd_incl;
        double first_der_1;
		double second_der_1;

        cox_firth();
        void setup(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const int& cols_incl, const int& max_iter = 100, const int& max_inner_iter = 30, const double& tolerance = 1e-6, const double& stephalf_tol = 2.5e-4, const double& beta_tol = 1e-6, const double& max_step = 1, const bool& use_firth = true, const bool& verbose_obj = false, const Eigen::VectorXd& beta_init = Eigen::VectorXd());
        void cox_firth_likelihood(const survival_data& survivalData, const Eigen::MatrixXd& Xmat);
		void fit(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val);
        void cox_firth_likelihood_1(const survival_data& survivalData, const Eigen::VectorXd& g);
        void fit_1(const survival_data& survivalData, const Eigen::VectorXd& g, const Eigen::VectorXd& offset_val);

    private:
        void cox_firth_likelihood_compact(const survival_data& survivalData);
        void cox_firth_likelihood_1_compact(const survival_data& survivalData);
        void update_eta_order(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const Eigen::VectorXd& coefficients);
        void update_mu(const survival_data& survivalData);

        int _niter, _mxitnr, _cols_incl;
        double _tol, _stephalf_tol, _betatol;
		double _maxstep;
        bool _usefirth, _verbose;

        Eigen::MatrixXd _X_order;
        Eigen::VectorXd _offset_order;
        Eigen::VectorXd _w_exp_eta;
        Eigen::VectorXd _S0;
        Eigen::VectorXd _cumulative_hazard;
        Eigen::VectorXd _first_moment;
        Eigen::MatrixXd _second_moment;
        std::vector<double> _third_moment;
        std::vector<Eigen::MatrixXd> _firth_der;
};

#endif
