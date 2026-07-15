#include "Regenie.hpp"
#include "survival_data.hpp"
#include "cox_firth.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require_close(
    const std::string& label,
    double expected,
    double actual,
    double tolerance) {
    const double error = std::fabs(expected - actual);
    if (!std::isfinite(error) || error > tolerance) {
        throw std::runtime_error(
            label + " absolute error=" + std::to_string(error));
    }
}

void require_close(
    const std::string& label,
    const Eigen::VectorXd& expected,
    const Eigen::VectorXd& actual,
    double tolerance) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    const double error = expected.size() == 0 ? 0 :
        (expected - actual).cwiseAbs().maxCoeff();
    if (!std::isfinite(error) || error > tolerance) {
        throw std::runtime_error(
            label + " maximum absolute error=" + std::to_string(error));
    }
}

void require_close(
    const std::string& label,
    const Eigen::MatrixXd& expected,
    const Eigen::MatrixXd& actual,
    double tolerance) {
    if (expected.rows() != actual.rows() ||
        expected.cols() != actual.cols()) {
        throw std::runtime_error(label + " size mismatch");
    }
    const double error = expected.size() == 0 ? 0 :
        (expected - actual).cwiseAbs().maxCoeff();
    if (!std::isfinite(error) || error > tolerance) {
        throw std::runtime_error(
            label + " maximum absolute error=" + std::to_string(error));
    }
}

survival_data make_survival_data(int sample_count) {
    Eigen::VectorXd time(sample_count);
    Eigen::VectorXd status = Eigen::VectorXd::Zero(sample_count);
    ArrayXb mask = ArrayXb::Constant(sample_count, true);
    for (int row = 0; row < sample_count; ++row) {
        time(row) = 1 + row / 3;
        status(row) = (row % 4 == 0 || row % 11 == 0) ? 1 : 0;
        if (row == 7 || row == 53) mask(row) = false;
    }

    survival_data data;
    data.setup(time, status, mask);
    return data;
}

Eigen::MatrixXd make_design(int sample_count) {
    Eigen::MatrixXd design(sample_count, 4);
    for (int row = 0; row < sample_count; ++row) {
        design(row, 0) = row % 2;
        design(row, 1) = (row % 7 - 3) / 4.0;
        design(row, 2) = std::sin(0.07 * row);
        design(row, 3) = row % 5 == 0 ? 1 : 0;
    }
    return design;
}

cox_firth evaluate_general(
    const survival_data& data,
    const Eigen::MatrixXd& design,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta,
    int included_columns = -1,
    bool use_firth = true) {
    if (included_columns < 0) included_columns = design.cols();
    cox_firth model;
    model.setup(
        data, design, offset, included_columns, 100, 30, 1e-8, 0,
        1e-8, 25, use_firth, false, beta);
    model.cox_firth_likelihood(data, design);
    return model;
}

cox_firth evaluate_one_parameter(
    const survival_data& data,
    const Eigen::VectorXd& genotype,
    const Eigen::VectorXd& offset,
    double beta_value) {
    Eigen::VectorXd beta(1);
    beta << beta_value;
    cox_firth model;
    model.setup(
        data, genotype, offset, 1, 100, 30, 1e-8, 0,
        1e-8, 25, true, false, beta);
    model.cox_firth_likelihood_1(data, genotype);
    return model;
}

}  // namespace

int main() {
    try {
        const int sample_count = 160;
        const survival_data data = make_survival_data(sample_count);
        const Eigen::MatrixXd design = make_design(sample_count);
        Eigen::VectorXd offset(sample_count);
        Eigen::VectorXd genotype(sample_count);
        for (int row = 0; row < sample_count; ++row) {
            offset(row) = 0.04 * std::cos(0.03 * row);
            genotype(row) = row % 13 == 0 ? 2 : (row % 5 == 0 ? 1 : 0);
        }

        Eigen::VectorXd beta(4);
        beta << 0.03, -0.08, 0.05, 0.11;
        const cox_firth full =
            evaluate_general(data, design, offset, beta);
        require_close(
            "information symmetry", full.second_der,
            full.second_der.transpose(), 1e-10);

        const double epsilon = 1e-6;
        for (int coefficient = 0; coefficient < beta.size(); ++coefficient) {
            Eigen::VectorXd beta_plus = beta;
            Eigen::VectorXd beta_minus = beta;
            beta_plus(coefficient) += epsilon;
            beta_minus(coefficient) -= epsilon;
            const cox_firth plus =
                evaluate_general(data, design, offset, beta_plus);
            const cox_firth minus =
                evaluate_general(data, design, offset, beta_minus);
            const double numerical_score =
                (plus.loglik_val - minus.loglik_val) / (2 * epsilon);
            require_close(
                "penalized finite-difference score", numerical_score,
                full.first_der(coefficient), 1e-5);
        }

        Eigen::MatrixXd genotype_design(sample_count, 1);
        genotype_design.col(0) = genotype;
        Eigen::VectorXd genotype_beta(1);
        genotype_beta << 0.07;
        const cox_firth general_one = evaluate_general(
            data, genotype_design, offset, genotype_beta);
        const cox_firth specialized_one = evaluate_one_parameter(
            data, genotype, offset, genotype_beta(0));
        require_close(
            "one-parameter likelihood", general_one.loglik_val,
            specialized_one.loglik_val, 1e-10);
        require_close(
            "one-parameter score", general_one.first_der(0),
            specialized_one.first_der_1, 1e-10);
        require_close(
            "one-parameter information", general_one.second_der(0, 0),
            specialized_one.second_der_1, 1e-10);
        require_close(
            "one-parameter residual", general_one.residual,
            specialized_one.residual, 1e-10);

        const Eigen::MatrixXd nuisance_design = design.leftCols(3);
        cox_firth shared_null;
        shared_null.setup(
            data, nuisance_design, offset, 3, 200, 30, 1e-8, 0,
            1e-8, 25, true, false);
        shared_null.fit(data, nuisance_design, offset);
        if (!shared_null.converge) {
            throw std::runtime_error("shared null did not converge");
        }

        Eigen::VectorXd shared_beta = Eigen::VectorXd::Zero(4);
        shared_beta.head(3) = shared_null.beta;
        cox_firth cold_reduced;
        cold_reduced.setup(
            data, design, offset, 3, 200, 30, 1e-8, 0,
            1e-8, 25, true, false);
        cold_reduced.fit(data, design, offset);
        cox_firth warm_reduced;
        warm_reduced.setup(
            data, design, offset, 3, 200, 30, 1e-8, 0,
            1e-8, 25, true, false, shared_beta);
        warm_reduced.fit(data, design, offset);
        if (!cold_reduced.converge || !warm_reduced.converge) {
            throw std::runtime_error("reduced fit did not converge");
        }
        require_close(
            "reduced warm-start coefficients", cold_reduced.beta,
            warm_reduced.beta, 1e-7);
        require_close(
            "reduced warm-start likelihood", cold_reduced.loglike.tail(1)(0),
            warm_reduced.loglike.tail(1)(0), 1e-8);
        if (warm_reduced.iter > cold_reduced.iter) {
            throw std::runtime_error(
                "reduced warm start increased the iteration count");
        }

        cox_firth cold_full;
        cold_full.setup(
            data, design, offset, 4, 200, 30, 1e-8, 0,
            1e-8, 25, true, false, shared_beta);
        cold_full.fit(data, design, offset);
        const cox_firth initial_full =
            evaluate_general(data, design, offset, shared_beta);
        Eigen::VectorXd score_warm_beta = shared_beta;
        score_warm_beta += initial_full.qrsd.solve(initial_full.first_der);
        cox_firth warm_full;
        warm_full.setup(
            data, design, offset, 4, 200, 30, 1e-8, 0,
            1e-8, 25, true, false, score_warm_beta);
        warm_full.fit(data, design, offset);
        if (!cold_full.converge || !warm_full.converge) {
            throw std::runtime_error("full fit did not converge");
        }
        require_close(
            "full warm-start coefficients", cold_full.beta,
            warm_full.beta, 1e-7);
        require_close(
            "full warm-start likelihood", cold_full.loglike.tail(1)(0),
            warm_full.loglike.tail(1)(0), 1e-8);
        if (warm_full.iter > cold_full.iter) {
            throw std::runtime_error(
                "full warm start increased the iteration count");
        }

        cox_firth general_fit;
        general_fit.setup(
            data, genotype_design, offset, 1, 200, 30, 1e-8, 0,
            1e-8, 25, true, false, genotype_beta);
        general_fit.fit(data, genotype_design, offset);
        cox_firth specialized_fit;
        specialized_fit.setup(
            data, genotype, offset, 1, 200, 30, 1e-8, 0,
            1e-8, 25, true, false, genotype_beta);
        specialized_fit.fit_1(data, genotype, offset);
        if (!general_fit.converge || !specialized_fit.converge) {
            throw std::runtime_error("one-parameter fit did not converge");
        }
        require_close(
            "one-parameter fit coefficients", general_fit.beta,
            specialized_fit.beta, 1e-7);
        require_close(
            "one-parameter fit likelihood", general_fit.loglike.tail(1)(0),
            specialized_fit.loglike.tail(1)(0), 1e-8);

        std::cout << "COX_FIRTH_TEST status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "COX_FIRTH_TEST status=FAIL error=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
