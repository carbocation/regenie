#include "Regenie.hpp"
#include "survival_data.hpp"
#include "cox_firth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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
    bool compact,
    int included_columns = -1,
    bool use_firth = true) {
    setenv("REGENIE_COX_FIRTH_COMPACT", compact ? "1" : "0", 1);
    if (included_columns < 0) included_columns = design.cols();
    Eigen::VectorXd beta(4);
    beta << 0.03, -0.08, 0.05, 0.11;
    cox_firth model;
    model.setup(
        data, design, offset, included_columns, 40, 30, 1e-8, 2.5e-4,
        1e-8, 1, use_firth, false, beta);
    model.cox_firth_likelihood(data, design);
    return model;
}

cox_firth evaluate_one_parameter(
    const survival_data& data,
    const Eigen::VectorXd& genotype,
    const Eigen::VectorXd& offset,
    bool compact,
    bool use_firth = true) {
    setenv("REGENIE_COX_FIRTH_COMPACT", compact ? "1" : "0", 1);
    Eigen::VectorXd beta(1);
    beta << 0.07;
    cox_firth model;
    model.setup(
        data, genotype, offset, 1, 40, 30, 1e-8, 2.5e-4,
        1e-8, 1, use_firth, false, beta);
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

        const cox_firth general_legacy =
            evaluate_general(data, design, offset, false);
        const cox_firth general_compact =
            evaluate_general(data, design, offset, true);
        require_close(
            "general log likelihood", general_legacy.loglik_val,
            general_compact.loglik_val, 1e-9);
        require_close(
            "general score", general_legacy.first_der,
            general_compact.first_der, 1e-8);
        require_close(
            "general information", general_legacy.second_der,
            general_compact.second_der, 1e-9);
        require_close(
            "general residual", general_legacy.residual,
            general_compact.residual, 1e-10);

        const cox_firth reduced_legacy =
            evaluate_general(data, design, offset, false, 3);
        const cox_firth reduced_compact =
            evaluate_general(data, design, offset, true, 3);
        require_close(
            "reduced log likelihood", reduced_legacy.loglik_val,
            reduced_compact.loglik_val, 1e-9);
        require_close(
            "reduced score", reduced_legacy.first_der,
            reduced_compact.first_der, 1e-8);
        require_close(
            "reduced information", reduced_legacy.second_der,
            reduced_compact.second_der, 1e-9);

        const cox_firth unpenalized_legacy =
            evaluate_general(data, design, offset, false, 4, false);
        const cox_firth unpenalized_compact =
            evaluate_general(data, design, offset, true, 4, false);
        require_close(
            "unpenalized log likelihood", unpenalized_legacy.loglik_val,
            unpenalized_compact.loglik_val, 1e-9);
        require_close(
            "unpenalized score", unpenalized_legacy.first_der,
            unpenalized_compact.first_der, 1e-9);
        require_close(
            "unpenalized information", unpenalized_legacy.second_der,
            unpenalized_compact.second_der, 1e-9);

        const cox_firth one_legacy =
            evaluate_one_parameter(data, genotype, offset, false);
        const cox_firth one_compact =
            evaluate_one_parameter(data, genotype, offset, true);
        require_close(
            "one-parameter log likelihood", one_legacy.loglik_val,
            one_compact.loglik_val, 1e-9);
        require_close(
            "one-parameter score", one_legacy.first_der_1,
            one_compact.first_der_1, 1e-9);
        require_close(
            "one-parameter information", one_legacy.second_der_1,
            one_compact.second_der_1, 1e-9);
        require_close(
            "one-parameter residual", one_legacy.residual,
            one_compact.residual, 1e-10);

        cox_firth fit_legacy =
            evaluate_general(data, design, offset, false);
        cox_firth fit_compact =
            evaluate_general(data, design, offset, true);
        fit_legacy.fit(data, design, offset);
        fit_compact.fit(data, design, offset);
        if (fit_legacy.converge != fit_compact.converge) {
            throw std::runtime_error("general convergence mismatch");
        }
        require_close("fit beta", fit_legacy.beta, fit_compact.beta, 1e-7);
        require_close("fit eta", fit_legacy.eta, fit_compact.eta, 1e-7);
        require_close(
            "fit final log likelihood", fit_legacy.loglike.tail(1)(0),
            fit_compact.loglike.tail(1)(0), 1e-8);

        cox_firth one_fit_legacy =
            evaluate_one_parameter(data, genotype, offset, false);
        cox_firth one_fit_compact =
            evaluate_one_parameter(data, genotype, offset, true);
        one_fit_legacy.fit_1(data, genotype, offset);
        one_fit_compact.fit_1(data, genotype, offset);
        if (one_fit_legacy.converge != one_fit_compact.converge) {
            throw std::runtime_error("one-parameter convergence mismatch");
        }
        require_close(
            "one-parameter fit beta", one_fit_legacy.beta,
            one_fit_compact.beta, 1e-7);
        require_close(
            "one-parameter fit eta", one_fit_legacy.eta,
            one_fit_compact.eta, 1e-7);
        require_close(
            "one-parameter fit final log likelihood",
            one_fit_legacy.loglike.tail(1)(0),
            one_fit_compact.loglike.tail(1)(0), 1e-8);

        unsetenv("REGENIE_COX_FIRTH_COMPACT");
        std::cout << "COX_FIRTH_TEST status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "COX_FIRTH_TEST status=FAIL error=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
