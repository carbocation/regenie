#include "Regenie.hpp"
#include "survival_data.hpp"
#include "cox_firth.hpp"

using namespace Eigen;
using namespace std;

namespace {

bool compact_cox_firth_likelihood_enabled() {
    const char* value = std::getenv("REGENIE_COX_FIRTH_COMPACT");
    return value == nullptr || std::string(value) != "0";
}

bool direct_cox_firth_adjustment_enabled() {
    const char* value = std::getenv("REGENIE_COX_FIRTH_DIRECT_ADJUSTMENT");
    return value == nullptr || std::string(value) != "0";
}

bool consistent_reduced_cox_firth_adjustment_enabled() {
    const char* value =
        std::getenv("REGENIE_COX_FIRTH_CONSISTENT_REDUCED");
    return value != nullptr && std::string(value) != "0";
}

bool legacy_cox_firth_line_search_enabled() {
    const char* value =
        std::getenv("REGENIE_COX_FIRTH_LEGACY_LINE_SEARCH");
    return value != nullptr && std::string(value) != "0";
}

size_t third_moment_index(int first, int second, int third, int dimension) {
    if (first > second) std::swap(first, second);
    if (second > third) std::swap(second, third);
    if (first > second) std::swap(first, second);
    return (static_cast<size_t>(first) * dimension + second) * dimension +
        third;
}

}

cox_firth::cox_firth(){}

void cox_firth::setup(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const int& cols_incl, const int& max_iter, const int& max_inner_iter, const double& tolerance, const double& stephalf_tol, const double& beta_tol, const double& max_step, const bool& use_firth, const bool& verbose_obj, const Eigen::VectorXd& beta_init) {
	converge = false;
    p = Xmat.cols();
    
	_niter = max_iter;
    _mxitnr = max_inner_iter;
    _tol = tolerance;
    _stephalf_tol = stephalf_tol;
    _betatol = beta_tol;
	_maxstep = max_step;
	_usefirth = use_firth;
    _verbose = verbose_obj;
    _cols_incl = cols_incl;
    _compact_likelihood = compact_cox_firth_likelihood_enabled();
    _direct_adjustment = _compact_likelihood && _usefirth &&
        direct_cox_firth_adjustment_enabled();
    _consistent_reduced_adjustment = _direct_adjustment &&
        consistent_reduced_cox_firth_adjustment_enabled();
    likelihood_evaluations = 0;
    step_halving_evaluations = 0;
    line_search_exhaustions = 0;
    final_score_max = 0;
    loglike.resize(_niter + 1);
    first_der.resize(p);
    mu.resize(survivalData.n);
    residual.resize(survivalData.n);

    _X_order = survivalData.permute_mtx * Xmat;
    _offset_order = survivalData.keep_sample_order.select(
        survivalData.permute_mtx * offset_val, 0).matrix();
    _w_exp_eta.resize(survivalData.n);
    _S0.resize(survivalData.n_unique_time);
    _cumulative_hazard.resize(survivalData.n_unique_time);
    _first_moment.resize(p);
    _second_moment.resize(p, p);
    _third_moment.resize(
        _direct_adjustment ? static_cast<size_t>(p) * p * p : 0);
    _firth_der.resize(_usefirth ? p : 0);
    for (int i = 0; i < static_cast<int>(_firth_der.size()); ++i) {
        _firth_der[i].resize(p, p);
    }
    _leverage.resize(survivalData.n);

    beta = Eigen::VectorXd::Zero(p);
    if (beta_init.size() > 0) {
        beta.head(_cols_incl) = beta_init.head(_cols_incl);
        eta = Xmat * beta + offset_val;
    } else {
        eta = offset_val;
    }
    eta_order = _X_order * beta + _offset_order;
    eta_order = survivalData.keep_sample_order.select(eta_order, 0).matrix();
    if (p == 0) {
        _usefirth = false;
    }
}

void cox_firth::cox_firth_likelihood(const survival_data& survivalData, const Eigen::MatrixXd& Xmat) {
    ++likelihood_evaluations;
    if (_compact_likelihood) {
        cox_firth_likelihood_compact(survivalData);
    } else {
        cox_firth_likelihood_legacy(survivalData, Xmat);
    }
}

void cox_firth::update_eta_order(
    const survival_data& survivalData,
    const Eigen::MatrixXd& Xmat,
    const Eigen::VectorXd& offset_val,
    const Eigen::VectorXd& coefficients) {
    if (_compact_likelihood) {
        eta_order.noalias() = _X_order * coefficients;
        eta_order += _offset_order;
        eta_order = survivalData.keep_sample_order.select(
            eta_order, 0).matrix();
    } else {
        eta = Xmat * coefficients + offset_val;
        eta_order = survivalData.keep_sample_order.select(
            survivalData.permute_mtx * eta, 0).matrix();
    }
}

void cox_firth::update_mu(const survival_data& survivalData) {
    double cumulative_hazard = 0;
    for (unsigned int k = 0; k < survivalData.n_unique_time; ++k) {
        cumulative_hazard += survivalData.ww_k(k) / _S0(k);
        _cumulative_hazard(k) = cumulative_hazard;
    }

    for (unsigned int i = 0; i < survivalData.n; ++i) {
        const int risk_count = static_cast<int>(survivalData.rskcount(i));
        const double lambda0 = risk_count == 0 ?
            0 : _cumulative_hazard(risk_count - 1);
        mu(i) = lambda0 * _w_exp_eta(i);
    }
}

void cox_firth::cox_firth_likelihood_compact(
    const survival_data& survivalData) {
    exp_eta = eta_order.array().exp();
    _w_exp_eta = survivalData.w.array() * exp_eta.array();
    _S0.setZero();
    _first_moment.setZero();
    _second_moment.setZero();
    std::fill(_third_moment.begin(), _third_moment.end(), 0.0);
    second_der.setZero(p, p);
    for (int t = 0; t < static_cast<int>(_firth_der.size()); ++t) {
        _firth_der[t].setZero();
    }

    loglik_val = (survivalData.w.array() * eta_order.array() *
        (survivalData.status_order.array() == 1).cast<double>()).sum();

    double risk_sum = 0;
    int event_index = static_cast<int>(survivalData.n_unique_time) - 1;
    const int first_risk_row = survivalData.risk_set_start(0);
    for (int row = static_cast<int>(survivalData.n) - 1;
         row >= first_risk_row; --row) {
        const double row_risk = _w_exp_eta(row);
        risk_sum += row_risk;
        for (int j = 0; j < p; ++j) {
            const double weighted_x = row_risk * _X_order(row, j);
            _first_moment(j) += weighted_x;
            for (int k = 0; k < p; ++k) {
                const double weighted_xx = weighted_x * _X_order(row, k);
                _second_moment(j, k) += weighted_xx;
                if (_direct_adjustment && k >= j) {
                    for (int t = k; t < p; ++t) {
                        _third_moment[third_moment_index(j, k, t, p)] +=
                            weighted_xx * _X_order(row, t);
                    }
                }
            }
        }

        if (row != survivalData.risk_set_start(event_index)) {
            continue;
        }

        _S0(event_index) = risk_sum;
        const double event_weight = survivalData.ww_k(event_index);
        const double inverse_risk = 1.0 / risk_sum;
        const double inverse_risk2 = inverse_risk * inverse_risk;
        const double inverse_risk3 = inverse_risk2 * inverse_risk;
        loglik_val -= event_weight * std::log(risk_sum);

        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < p; ++k) {
                second_der(j, k) += event_weight *
                    (_second_moment(j, k) * inverse_risk -
                     _first_moment(j) * _first_moment(k) * inverse_risk2);
            }
        }

        if (_usefirth) {
            for (int t = 0; t < p; ++t) {
                for (int j = 0; j < p; ++j) {
                    for (int k = 0; k < p; ++k) {
                        _firth_der[t](j, k) += event_weight *
                            ((_direct_adjustment ?
                                  _third_moment[third_moment_index(
                                      t, j, k, p)] * inverse_risk : 0.0) +
                             (-_second_moment(j, k) * _first_moment(t) -
                              _second_moment(j, t) * _first_moment(k) -
                              _second_moment(t, j) * _first_moment(k)) *
                                 inverse_risk2 +
                             2 * _first_moment(j) * _first_moment(k) *
                                 _first_moment(t) * inverse_risk3);
                    }
                }
            }
        }

        --event_index;
    }

    update_mu(survivalData);
    if (p > 0) qrsd.compute(second_der);
    residual = survivalData.w.array() *
        (survivalData.status_order - mu).array();

    if (_cols_incl < p) {
        qrsd_incl.compute(second_der.block(0, 0, _cols_incl, _cols_incl));
        if (_usefirth) {
            loglik_val += 0.5 * qrsd.logAbsDeterminant();
            if (_direct_adjustment) {
                first_der =
                    _X_order.leftCols(_cols_incl).transpose() * residual;
                for (int t = 0; t < _cols_incl; ++t) {
                    if (_consistent_reduced_adjustment) {
                        first_der(t) +=
                            0.5 * qrsd.solve(_firth_der[t]).trace();
                    } else {
                        first_der(t) += 0.5 * qrsd_incl.solve(
                            _firth_der[t].block(
                                0, 0, _cols_incl, _cols_incl)).trace();
                    }
                }
            } else {
                _XtW = (_X_order.leftCols(_cols_incl).array().colwise() *
                    mu.array().sqrt()).transpose();
                _solved_XtW = qrsd_incl.solve(_XtW);
                _leverage = (_solved_XtW.array() * _XtW.array())
                    .colwise().sum().matrix().transpose();
                first_der = _X_order.leftCols(_cols_incl).transpose() *
                    survivalData.keep_sample_order.select(
                        residual + 0.5 * _leverage, 0);
                for (int t = 0; t < _cols_incl; ++t) {
                    first_der(t) += 0.5 * qrsd_incl.solve(
                        _firth_der[t].block(
                            0, 0, _cols_incl, _cols_incl)).trace();
                }
            }
        } else {
            first_der = _X_order.leftCols(_cols_incl).transpose() * residual;
        }
    } else {
        if (_usefirth) {
            loglik_val += 0.5 * qrsd.logAbsDeterminant();
            if (_direct_adjustment) {
                first_der = _X_order.transpose() * residual;
                for (int t = 0; t < p; ++t) {
                    first_der(t) +=
                        0.5 * qrsd.solve(_firth_der[t]).trace();
                }
            } else {
                _XtW = (_X_order.array().colwise() *
                    mu.array().sqrt()).transpose();
                _solved_XtW = qrsd.solve(_XtW);
                _leverage = (_solved_XtW.array() * _XtW.array())
                    .colwise().sum().matrix().transpose();
                first_der = _X_order.transpose() *
                    survivalData.keep_sample_order.select(
                        residual + 0.5 * _leverage, 0);
                for (int t = 0; t < p; ++t) {
                    first_der(t) +=
                        0.5 * qrsd.solve(_firth_der[t]).trace();
                }
            }
        } else {
            first_der = _X_order.transpose() * residual;
        }
    }
}

void cox_firth::cox_firth_likelihood_legacy(const survival_data& survivalData, const Eigen::MatrixXd& Xmat) {
    Eigen::VectorXd w_exp_eta, ww_rsk, S0;
    Eigen::VectorXd lambda0(survivalData.n);
    Eigen::MatrixXd S1, GammaX, XtW;
    Eigen::MatrixXd S2 = Eigen::MatrixXd::Zero(p, p);
    std::vector<Eigen::MatrixXd> firth_der;
    double log_terms_sum;
    second_der = Eigen::MatrixXd::Zero(p, p);

    if (_usefirth) {
        firth_der.resize(p);
        for(int i = 0; i < p; i++) {
            firth_der[i] = Eigen::MatrixXd::Zero(p, p);
        }
    }

    exp_eta = eta_order.array().exp();
    w_exp_eta = survivalData.w.array() * exp_eta.array();

    S0 = cumulativeSum_reverse2(survivalData.R.transpose() * w_exp_eta); // length K, risk set sum at each unique failure time
    log_terms_sum = (survivalData.ww_k.array() * S0.array().log()).sum();

    loglik_val = (survivalData.w.array() * eta_order.array() * (survivalData.status_order.array() == 1).cast<double>()).sum() - log_terms_sum;

    // double mean_eta = (eta.array() * survivalData.w_orig.array()).sum()/survivalData.w_orig.array().sum();
    // Eigen::VectorXd eta_center = eta_order.array() - mean_eta;
    // exp_eta = eta_center.array().exp();
    // w_exp_eta = survivalData.w.array() * exp_eta.array();
    // S0 = cumulativeSum_reverse2(survivalData.R.transpose() * w_exp_eta); // length K, risk 

    ww_rsk = cumulativeSum(survivalData.ww_k.array() / S0.array());
    for (unsigned int i = 0; i < survivalData.n; ++i) {
        if (survivalData.rskcount(i) == 0) {
            lambda0(i) = 0;
        } else {
            lambda0(i) = ww_rsk(int(survivalData.rskcount(i)) - 1);
        }
    }
    mu = lambda0.array() * w_exp_eta.array();

    S1 = survivalData.R.transpose() * ((survivalData.permute_mtx * Xmat).array().colwise() * w_exp_eta.array()).matrix(); // K by p

    GammaX = (survivalData.permute_mtx * Xmat).array().colwise() * w_exp_eta.array().sqrt(); // n by p
    for (int k = survivalData.n_unique_time - 1; k >= 0; --k) {
        if (k < survivalData.n_unique_time - 1) {
            S1.row(k) += S1.row(k+1);
        }

        std::vector<int> k_indices;
        for (SpMat::InnerIterator it(survivalData.R, k); it; ++it) {
            k_indices.push_back(it.index());
        }
        
        // for (int j = 0; j < survivalData.R.cols(); ++j) {
        //     if (survivalData.R(k, j) != 0) {
        //         k_indices.push_back(j);
        //     }
        // }

        S2 += GammaX(k_indices, all).transpose() * GammaX(k_indices, all);

        second_der = second_der + survivalData.ww_k(k) * (S2/S0(k) - S1.row(k).transpose() * S1.row(k)/(std::pow(S0(k), 2)));
        if (_usefirth) {
            for (int t = 0; t < p; ++t) {
                firth_der[t] += survivalData.ww_k(k) * ((-S2 * S1(k,t) - S2.col(t) * S1.row(k) - S2.row(t).transpose() * S1.row(k))/(std::pow(S0(k), 2)) + 2 * S1.row(k).transpose() * S1.row(k) * S1(k,t)/(std::pow(S0(k), 3)));
            }
        }
    }
    if (p > 0) qrsd.compute(second_der);
    residual = survivalData.w.array() * (survivalData.status_order - mu).array();
    if(_cols_incl < p) {
        qrsd_incl.compute(second_der.block(0,0,_cols_incl,_cols_incl)); // p-1 by p-1
        if (_usefirth) {
            loglik_val += 0.5 * qrsd.logAbsDeterminant();
            XtW = ((survivalData.permute_mtx * Xmat.leftCols(_cols_incl)).array().colwise() * mu.array().sqrt()).transpose(); // p-1 by n
            first_der = (survivalData.permute_mtx * Xmat.leftCols(_cols_incl)).transpose() * survivalData.keep_sample_order.select(residual + 0.5 * (qrsd_incl.solve(XtW).array() * XtW.array()).colwise().sum().matrix().transpose(), 0); // qrsd.solve(XtW) is p-1 by n
            for (int t = 0; t < _cols_incl; ++t) {
                first_der(t) = first_der(t) + 0.5 * qrsd_incl.solve(firth_der[t].block(0,0,_cols_incl,_cols_incl)).trace();
            }
        } else {
            first_der = (survivalData.permute_mtx * Xmat.leftCols(_cols_incl)).transpose() * residual;
        }
    } else {
        if (_usefirth) {
            loglik_val += 0.5 * qrsd.logAbsDeterminant();
            XtW = ((survivalData.permute_mtx * Xmat).array().colwise() * mu.array().sqrt()).transpose(); // p by n
            first_der = (survivalData.permute_mtx * Xmat).transpose() * survivalData.keep_sample_order.select(residual + 0.5 * (qrsd.solve(XtW).array() * XtW.array()).colwise().sum().matrix().transpose(), 0); // qrsd.solve(XtW) is p by n
            for (int t = 0; t < p; ++t) {
                first_der(t) = first_der(t) + 0.5 * qrsd.solve(firth_der[t]).trace();
            }
        } else {
            first_der = (survivalData.permute_mtx * Xmat).transpose() * residual;
        }
    }
}

void cox_firth::fit(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val) {
    Eigen::VectorXd steps, betanew;
    int ii;
    cox_firth_likelihood(survivalData, Xmat);
    loglike(0) = loglik_val;
    // std::cout << "start fitting:\n";
    // std::cout << "beta: " << beta << "\n";
    // std::cout << "loglik_val: " << loglik_val << "\n";
    // std::cout << "first_der: " << first_der << "\n";
    // std::cout << "second_der: " << second_der << "\n";
    iter = 0;
    if (p == 0 || _cols_incl == 0) {
        converge = true;
        residual = survivalData.permute_mtx.transpose() * residual;
        loglike.conservativeResize(iter+1);
        return;
    }
    betanew = beta;
    while (iter++ < _niter) {
        // std::cout << "iter: " << iter << "\n";
        ii = 0;
        if (_cols_incl < p) {
            steps = qrsd_incl.solve(first_der);
        } else{
            steps = qrsd.solve(first_der);
        }
        // std::cout << "steps: " << steps << "\n";
        for (int i = 0; i < steps.size(); ++i) {
            if (abs(steps(i)) >= _maxstep) {
                steps(i) = (steps(i) / fabs(steps(i))) * _maxstep;
            }
        }
        // std::cout << "adjusted steps: " << steps << "\n";
        betanew.head(_cols_incl) = beta.head(_cols_incl) + steps;
        // std::cout << "beta: " << betanew << "\n";
        update_eta_order(survivalData, Xmat, offset_val, betanew);
        cox_firth_likelihood(survivalData, Xmat);
        // std::cout << "loglik_val: " << loglik_val << "\n";
        // std::cout << "diff loglik_val: " << loglik_val - loglike(iter - 1) << "\n";
        if ((loglike(iter - 1) - loglik_val) > _stephalf_tol) { // step-halving
            // std::cout << "\nLoglikelihood decreases at iteration " << iter << ", start step-halving.\n";
            ii = 0;
            while ((loglike(iter - 1) - loglik_val) > _stephalf_tol) {
                ++ii;
                ++step_halving_evaluations;
                // std::cout << "inner iteration: " << ii << "\n";
                if (ii > _mxitnr) {
                    ++line_search_exhaustions;
                    // Historical behavior discarded the final halved
                    // candidate and retried the original rejected Newton
                    // step with an epsilon added to every coefficient.  That
                    // can recreate the same rejection cycle indefinitely.
                    // Keep the smallest evaluated candidate by default,
                    // matching the coxphf line-search behavior.  Retain an
                    // environment switch for numerical A/B validation.
                    if (legacy_cox_firth_line_search_enabled()) {
                        steps.array() += 1e-6;
                        betanew.head(_cols_incl) =
                            beta.head(_cols_incl) + steps;
                        update_eta_order(
                            survivalData, Xmat, offset_val, betanew);
                        cox_firth_likelihood(survivalData, Xmat);
                    }
                    break;
                    // throw std::runtime_error("inner loop: cannot correct step size");
                }
                betanew = (beta + betanew)/2;
                update_eta_order(survivalData, Xmat, offset_val, betanew);
                cox_firth_likelihood(survivalData, Xmat);
                if (_verbose) {
                    std::cout << "beta: " << betanew << "\n";
                    std::cout << "Iteration " << iter << " Halved, Objective: " << loglik_val << "\n";
                }
            }
        }
        loglike(iter) = loglik_val;
        // std::cout << "beta: " << betanew << "\n";
        // std::cout << "loglik_val: " << loglik_val << "\n";
        // std::cout << "loglik_val change: " << loglik_val - loglike(iter - 1) << "\n";
        // std::cout << "first_der max: " << first_der.array().abs().maxCoeff() << "\n";
        // std::cout << "beta change max: " << (beta - betanew).array().abs().maxCoeff() << "\n";
        if( first_der.array().abs().maxCoeff() < _tol || (ii <= 1 && (beta - betanew).array().abs().maxCoeff() < _betatol) ) {
            beta = betanew;
            converge = true;
            break;
        }
        beta = betanew;
    }
    final_score_max = first_der.size() == 0 ? 0 :
        first_der.array().abs().maxCoeff();
    if (_compact_likelihood) eta = Xmat * beta + offset_val;
    residual = survivalData.permute_mtx.transpose() * residual;
    loglike.conservativeResize(iter+1);
    // std::cout << "finish fitting\n";
}


void cox_firth::cox_firth_likelihood_1(
    const survival_data& survivalData,
    const Eigen::VectorXd& g) {
    ++likelihood_evaluations;
    if (_compact_likelihood) {
        cox_firth_likelihood_1_compact(survivalData);
    } else {
        cox_firth_likelihood_1_legacy(survivalData, g);
    }
}

void cox_firth::cox_firth_likelihood_1_compact(
    const survival_data& survivalData) {
    exp_eta = eta_order.array().exp();
    _w_exp_eta = survivalData.w.array() * exp_eta.array();
    _S0.setZero();

    loglik_val = (survivalData.w.array() * eta_order.array() *
        (survivalData.status_order.array() == 1).cast<double>()).sum();
    second_der_1 = 0;
    double firth_information_derivative = 0;
    double risk_sum = 0;
    double first_moment = 0;
    double second_moment = 0;
    double third_moment = 0;
    int event_index = static_cast<int>(survivalData.n_unique_time) - 1;
    const int first_risk_row = survivalData.risk_set_start(0);
    for (int row = static_cast<int>(survivalData.n) - 1;
         row >= first_risk_row; --row) {
        const double row_risk = _w_exp_eta(row);
        const double genotype = _X_order(row, 0);
        const double genotype2 = genotype * genotype;
        risk_sum += row_risk;
        first_moment += row_risk * genotype;
        second_moment += row_risk * genotype2;
        if (_usefirth) {
            third_moment += row_risk * genotype2 * genotype;
        }

        if (row != survivalData.risk_set_start(event_index)) {
            continue;
        }

        _S0(event_index) = risk_sum;
        const double event_weight = survivalData.ww_k(event_index);
        const double inverse_risk = 1.0 / risk_sum;
        const double mean = first_moment * inverse_risk;
        loglik_val -= event_weight * std::log(risk_sum);
        second_der_1 += event_weight *
            (second_moment * inverse_risk - mean * mean);
        if (_usefirth) {
            firth_information_derivative += event_weight *
                (third_moment * inverse_risk -
                 3 * second_moment * first_moment *
                     inverse_risk * inverse_risk +
                 2 * first_moment * first_moment * first_moment *
                     inverse_risk * inverse_risk * inverse_risk);
        }
        --event_index;
    }

    update_mu(survivalData);
    residual = survivalData.w.array() *
        (survivalData.status_order - mu).array();
    first_der_1 = _X_order.col(0).dot(residual);
    if (_usefirth) {
        loglik_val += 0.5 * std::log(std::fabs(second_der_1));
        first_der_1 +=
            0.5 * firth_information_derivative / second_der_1;
    }
}

void cox_firth::cox_firth_likelihood_1_legacy(const survival_data& survivalData, const Eigen::VectorXd& g) {
    Eigen::VectorXd w_exp_eta, ww_rsk;
    Eigen::VectorXd lambda0(survivalData.n);
    Eigen::VectorXd S0, S1, S2, S3;
    double log_terms_sum;

    exp_eta = eta_order.array().exp();
    w_exp_eta = survivalData.w.array() * exp_eta.array();

    S0 = cumulativeSum_reverse2(survivalData.R.transpose() * w_exp_eta); // length K, risk set sum at each unique failure time
    log_terms_sum = (survivalData.ww_k.array() * S0.array().log()).sum();

    loglik_val = (survivalData.w.array() * eta_order.array() * (survivalData.status_order.array() == 1).cast<double>()).sum() - log_terms_sum;

    ww_rsk = cumulativeSum(survivalData.ww_k.array() / S0.array());
    for (unsigned int i = 0; i < survivalData.n; ++i) {
        if (survivalData.rskcount(i) == 0) {
            lambda0(i) = 0;
        } else {
            lambda0(i) = ww_rsk(int(survivalData.rskcount(i)) - 1);
        }
    }
    mu = lambda0.array() * w_exp_eta.array();

    S1 = cumulativeSum_reverse2(survivalData.R.transpose() * ((survivalData.permute_mtx * g).array() * w_exp_eta.array()).matrix()); // K by 1

    S2 = cumulativeSum_reverse2(survivalData.R.transpose() * ((survivalData.permute_mtx * g.array().pow(2).matrix()).array() * w_exp_eta.array()).matrix()); // K by 1
    
    second_der_1 = (survivalData.ww_k.array() * (S2.array()/S0.array() - S1.array().pow(2)/S0.array().pow(2))).sum();

    residual = survivalData.w.array() * (survivalData.status_order - mu).array();

    if (_usefirth) {
        loglik_val += 0.5 * log(fabs(second_der_1));
        
        S3 = cumulativeSum_reverse2(survivalData.R.transpose() * ((survivalData.permute_mtx * g.array().pow(3).matrix()).array() * w_exp_eta.array()).matrix());
        
        first_der_1 = (survivalData.permute_mtx * g).dot(residual) + 0.5 * (survivalData.ww_k.array() * (S3.array()/S0.array() - 3 * S2.array() * S1.array()/S0.array().pow(2) + 2 * S1.array().pow(3)/S0.array().pow(3))).sum()/second_der_1;
    } else {
        first_der_1 = (survivalData.permute_mtx * g).dot(residual);
    }
}

void cox_firth::fit_1(const survival_data& survivalData, const Eigen::VectorXd& g, const Eigen::VectorXd& offset_val) {
    Eigen::VectorXd betanew;
    double steps;
    int ii = 0;
    cox_firth_likelihood_1(survivalData, g);
    // std::cout << "start fitting:\n";
    // std::cout << "beta: " << beta << "\n";
    // std::cout << "loglik_val: " << loglik_val << "\n";
    // std::cout << "first_der_1: " << first_der_1 << "\n";
    // std::cout << "second_der_1: " << second_der_1 << "\n";
    loglike(0) = loglik_val;
    iter = 0;
    while (iter++ < _niter) {
        // std::cout << "iter: " << iter << "\n";
        steps = first_der_1/second_der_1;
        // std::cout << "first der: " << first_der_1 << "\n";
        // std::cout << "second der: " << second_der_1 << "\n";
        // std::cout << "steps: " << steps << "\n";
        if (abs(steps) >= _maxstep) {
            steps = (steps / fabs(steps)) * _maxstep;
        }
        // std::cout << "adjusted steps: " << steps << "\n";
        betanew = beta.array() + steps;
        update_eta_order(survivalData, g, offset_val, betanew);
        cox_firth_likelihood_1(survivalData, g);
        // std::cout << "beta: " << betanew << "\n";
        // std::cout << "loglik_val: " << loglik_val << "\n";
        // std::cout << "diff loglik_val: " << loglik_val - loglike(iter - 1) << "\n";
        
        if ((loglike(iter - 1) - loglik_val) > _stephalf_tol) { // step-halving
            // std::cout << "\nLoglikelihood decreases at iteration " << iter << ", start step-halving.\n";
            ii = 0;
            while ((loglike(iter - 1) - loglik_val) > _stephalf_tol) {
                ++ii;
                ++step_halving_evaluations;
                // std::cout << "inner iteration: " << ii << "\n";
                if (ii > _mxitnr) {
                    ++line_search_exhaustions;
                    if (legacy_cox_firth_line_search_enabled()) {
                        steps += 1e-6;
                        betanew = beta.array() + steps;
                        update_eta_order(
                            survivalData, g, offset_val, betanew);
                        cox_firth_likelihood_1(survivalData, g);
                    }
                    break;
                    // throw std::runtime_error("inner loop: cannot correct step size");
                }
                betanew = (beta + betanew)/2;
                update_eta_order(survivalData, g, offset_val, betanew);
                cox_firth_likelihood_1(survivalData, g);
                if (_verbose) {
                    std::cout << "beta: " << betanew << "\n";
                    std::cout << "Iteration " << iter << " Halved, Objective: " << loglik_val << "\n";
                }
            }
        }
        loglike(iter) = loglik_val;
        // std::cout << "beta: " << betanew << "\n";
        // std::cout << "loglik_val: " << loglik_val << "\n";
        // std::cout << "first_der_1: " << first_der_1 << "\n";
        // std::cout << "second_der_1: " << second_der_1 << "\n";
        // std::cout << "first_der max: " << fabs(first_der_1) << "\n";
        // std::cout << "beta change max: " << (beta - betanew).array().abs().maxCoeff() << "\n";
        if (fabs(first_der_1) < _tol || (ii <= 1 && (beta - betanew).array().abs().maxCoeff() < _betatol)) {
            beta = betanew;
            converge = true;
            break;
        }
        beta = betanew;
    }
    final_score_max = std::fabs(first_der_1);
    if (_compact_likelihood) eta = g * beta + offset_val;
    residual = survivalData.permute_mtx.transpose() * residual;
    loglike.conservativeResize(iter+1);
    // std::cout << "finish fitting\n";
}
