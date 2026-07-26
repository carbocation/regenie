#include "Regenie.hpp"
#include "survival_data.hpp"
#include "cox_ridge.hpp"
#include "Step1_Compute.hpp"

using namespace Eigen;
using namespace std;

namespace {

Eigen::VectorXd step1_linear_prediction(
    Step1ComputeBackend* compute_backend,
    const Eigen::MatrixXd& design,
    const Eigen::VectorXd& coefficients,
    bool resident_design,
    Step1ComputeTimings* timings) {
    if(!compute_backend) return design * coefficients;
    if(resident_design) {
        Eigen::VectorXd prediction;
        compute_backend->predict_cached_design(
            coefficients, prediction, timings);
        return prediction;
    }
    Eigen::VectorXi group_offset(1), group_size(1);
    group_offset(0) = 0;
    group_size(0) = design.cols();
    Eigen::MatrixXd prediction;
    compute_backend->grouped_predict(
        design, coefficients, group_offset, group_size, prediction, timings);
    return prediction.col(0);
}

Eigen::VectorXd step1_design_crossproduct(
    Step1ComputeBackend* compute_backend,
    const Eigen::MatrixXd& design,
    const Eigen::VectorXd& values,
    bool resident_design,
    Step1ComputeTimings* timings) {
    if(!compute_backend) return design.transpose() * values;
    const Eigen::MatrixXd outcome = values;
    Eigen::MatrixXd crossproduct;
    if(resident_design)
        compute_backend->compute_cached_design_crossproduct(
            outcome, crossproduct, timings);
    else
        compute_backend->compute_design_crossproduct(
            design, outcome, crossproduct, timings);
    return crossproduct.col(0);
}

}

cox_ridge::cox_ridge(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask, const double& lambda_val, const int& max_iter, const int& max_inner_iter, const double& tolerance, const bool& verbose_obj, const Eigen::VectorXd& beta_init, const double& null_deviance, Step1ComputeBackend* compute_backend, const bool& resident_design, Step1ComputeTimings* timings) {
    converge = false;

    if (beta_init.size() > 0) {
        beta = beta_init;
    } else {
        beta = Eigen::VectorXd::Zero(Xmat.cols());
    }
    lambda = lambda_val;
    _niter = max_iter;
    _mxitnr = max_inner_iter;
    _tol = tolerance;
    _verbose = verbose_obj;
    _compute_backend = compute_backend;
    _resident_design = resident_design;
    _timings = timings;
    _object.resize(_niter + 1);
    _deviance.resize(_niter + 1);

    Eigen::VectorXd eta_unmasked = offset_val;
    if(!beta.isZero(0))
        eta_unmasked += step1_linear_prediction(
            _compute_backend, Xmat, beta, _resident_design, _timings);
    eta = mask.select(eta_unmasked, 0).matrix();
    eta_order = survivalData.permute_mtx * eta;
    
    if (null_deviance == -999) {
        _deviance(0) = _coxDeviance(survivalData);
    } else {
        _deviance(0) = null_deviance;
    }
    _null_deviance = _deviance(0);
    _object(0) = _deviance(0) + lambda * (beta.array().pow(2).sum())/2;
}

void cox_ridge::reset(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask, const double& lambda_val, const Eigen::VectorXd& beta_init, const double& null_deviance) {
    // Reset the object's state based on the provided parameters
    converge = false;
    _weighted_solve_count = 0;

    if (beta_init.size() > 0) {
        beta = beta_init;
    } else {
        beta = Eigen::VectorXd::Zero(Xmat.cols());
    }
    lambda = lambda_val;

    // Calculate eta, eta_order, or other necessary calculations
    Eigen::VectorXd eta_unmasked = offset_val;
    if(!beta.isZero(0))
        eta_unmasked += step1_linear_prediction(
            _compute_backend, Xmat, beta, _resident_design, _timings);
    eta = mask.select(eta_unmasked, 0).matrix();
    eta_order = survivalData.permute_mtx * eta;

    _deviance.resize(_niter + 1);
    _object.resize(_niter + 1);
    if (null_deviance == -999){
        _deviance(0) = _coxDeviance(survivalData);
    } else {
        _deviance(0) = null_deviance;
    }
    _null_deviance = _deviance(0);
    _object(0) = _deviance(0) + lambda * (beta.array().pow(2).sum())/2;
}

void cox_ridge::reset_from_path(
    const survival_data& survivalData,
    const Eigen::MatrixXd& Xmat,
    const Eigen::VectorXd& offset_val,
    const ArrayXb& mask,
    double lambda_val,
    const Eigen::VectorXd& beta_init,
    double null_deviance,
    double starting_deviance) {
    reset(
        survivalData, Xmat, offset_val, mask, lambda_val, beta_init,
        null_deviance);
    _deviance(0) = starting_deviance;
    _object(0) = starting_deviance +
        lambda * beta.squaredNorm() / 2;
}

void cox_ridge::coxGrad(const survival_data& survivalData) {
    double mean_eta = (eta.array() * survivalData.w_orig.array()).sum()/survivalData.w_orig.array().sum();
    Eigen::VectorXd eta_center = eta_order.array() - mean_eta;
    Eigen::VectorXd exp_eta = eta_center.array().exp();
    // Eigen::VectorXd exp_eta = eta_order.array().exp();
    Eigen::VectorXd rskden = cumulativeSum_reverse2(survivalData.w.array() * exp_eta.array());

    Eigen::VectorXd ww_rsk = survivalData.ww.array() / rskden.array();
    Eigen::VectorXd ww_rsk2 = survivalData.ww.array() / (rskden.array().pow(2));

    Eigen::VectorXd rskdeninv_n = cumulativeSum(survivalData.dd.array().cast<bool>().select(ww_rsk, 0));
    Eigen::VectorXd rskdeninv2_n = cumulativeSum(survivalData.dd.array().cast<bool>().select(ww_rsk2, 0));

    Eigen::VectorXd gradient_order = survivalData.w.array() * (survivalData.status_order.array() - exp_eta.array() * rskdeninv_n.array());

    Eigen::VectorXd diag_hessian_order = (survivalData.w.array() * exp_eta.array()).pow(2) * rskdeninv2_n.array() - 
        survivalData.w.array() * exp_eta.array() * rskdeninv_n.array();
    // change to original order
    _gradient = survivalData.permute_mtx.transpose() * gradient_order;
    _diagHessian = survivalData.permute_mtx.transpose() * diag_hessian_order;
}

bool cox_ridge::exact_path_point(
    const survival_data& survivalData,
    const Eigen::MatrixXd& Xmat,
    Step1CoxPathNewtonPoint& point) {
    coxGrad(survivalData);
    point.coefficients = beta;
    point.linear_predictor = eta;
    point.ordered_linear_predictor = eta_order;
    point.unpenalized_score = step1_design_crossproduct(
        _compute_backend, Xmat, _gradient, _resident_design, _timings);
    const double loglik = _coxLoglik(survivalData);
    point.negative_log_likelihood = -loglik;
    point.deviance = _coxDevianceFromLoglik(survivalData, loglik);
    return point.coefficients.allFinite() &&
        point.linear_predictor.allFinite() &&
        point.ordered_linear_predictor.allFinite() &&
        point.unpenalized_score.allFinite() &&
        std::isfinite(point.negative_log_likelihood) &&
        std::isfinite(point.deviance);
}

void cox_ridge::install_path_point(
    const Step1CoxPathNewtonPoint& point,
    double lambda_val) {
    beta = point.coefficients;
    eta = point.linear_predictor;
    eta_order = point.ordered_linear_predictor;
    lambda = lambda_val;
    converge = true;
    _deviance.resize(1);
    _object.resize(1);
    _deviance(0) = point.deviance;
    _object(0) = point.deviance +
        lambda * beta.squaredNorm() / 2;
    dev_ratio = 1 - point.deviance / _null_deviance;
}

double cox_ridge::_coxLoglik(const survival_data& survivalData) {
    Eigen::VectorXd rsk = cumulativeSum_reverse2(survivalData.w.array() * eta_order.array().exp());

    // take just the terms related to actual death times
    double log_terms_sum = (survivalData.ww.array() * (survivalData.keep_sample_order.select(rsk.array().log(), 0)) * (survivalData.dd.array() == 1).cast<double>()).sum();
    double loglik_val = (survivalData.w.array() * eta_order.array() * (survivalData.status_order.array() == 1).cast<double>()).sum() - log_terms_sum;
    return loglik_val;
}

double cox_ridge::_coxDevianceFromLoglik(
    const survival_data& survivalData,
    double loglik) {
    Eigen::VectorXd w_sub;
    if (survivalData.unique_time_indices.size() == survivalData.n_events) {
        // no tie
        w_sub = Eigen::VectorXd::Ones(survivalData.n_events);
        w_sub /= survivalData.neff;
    } else {
        // tie
        w_sub.resize(survivalData.unique_time_indices.size());
        int idx = 0;
        for (const auto& entry: survivalData.unique_time_indices) {
            const vector<int>& ties = entry.second;
            w_sub(idx) = static_cast<double>(ties.size())/survivalData.neff;
            ++idx;
        }
    }
    double lsat = -(w_sub.array() * (w_sub.array().log())).sum();
    return 2 * (lsat - loglik);
}

double cox_ridge::_coxDeviance(const survival_data& survivalData) {
    return _coxDevianceFromLoglik(
        survivalData, _coxLoglik(survivalData));
}

void cox_ridge::fit(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask) {
    int p = Xmat.cols();
    int ii = 0;
    Eigen::VectorXd beta_old;
    int break_pt = 1;
    for (int t = 1; t < _niter + 1; ++t) {
        ++break_pt;
        beta_old = beta;
        coxGrad(survivalData);
        Eigen::VectorXd z(survivalData.n);
        z = (_diagHessian.array() != 0).select(_gradient.array()/_diagHessian.array(), 0).matrix();
        z = mask.select(eta - offset_val, 0) - z;
        if (_compute_backend) {
            const Eigen::VectorXd weights = mask.select(
                (-_diagHessian.array()).max(0.0), 0.0).matrix();
            const Eigen::MatrixXd working_response = z;
            Eigen::MatrixXd gram, crossproduct, solution;
            Eigen::VectorXd ridge_parameter(1);
            ridge_parameter(0) = lambda;
            const Eigen::VectorXd penalty_multipliers = Eigen::VectorXd::Ones(p);
            const bool resident_solve = _resident_design &&
                _compute_backend->solve_cached_weighted_design(
                    weights, working_response, ridge_parameter,
                    penalty_multipliers, solution, _timings);
            if(!resident_solve) {
                if(_resident_design)
                    _compute_backend->compute_cached_weighted_design_products(
                        weights, working_response, gram, crossproduct, _timings);
                else
                    _compute_backend->compute_weighted_design_products(
                        Xmat, weights, working_response, gram, crossproduct,
                        _timings);
                _compute_backend->diagonal_penalty_solve(
                    gram, crossproduct, ridge_parameter, penalty_multipliers,
                    solution, _timings);
            }
            ++_weighted_solve_count;
            beta = solution.col(0);
            eta = mask.select(
                step1_linear_prediction(_compute_backend, Xmat, beta,
                    _resident_design, _timings) + offset_val,
                0).matrix();
        } else {
            for (unsigned int k = 0; k < p; ++k) {
                Eigen::VectorXd r = _diagHessian.array() * (z - eta + offset_val).array();
                eta = eta - mask.select(Xmat.col(k) * beta(k), 0).matrix();
                beta(k) = (r.dot(Xmat.col(k)) + beta(k) * (Xmat.col(k).array().pow(2) * _diagHessian.array()).sum()) /
                    ((Xmat.col(k).array().pow(2) * _diagHessian.array()).sum() - lambda);
                eta = eta + mask.select(Xmat.col(k) * beta(k), 0).matrix();
            }
        }
        eta_order = survivalData.permute_mtx * eta;
        _deviance(t) = _coxDeviance(survivalData);
        _object(t) = _deviance(t) + lambda * (beta.array().pow(2).sum())/2;
        const Eigen::VectorXd gradient_crossproduct =
            step1_design_crossproduct(_compute_backend, Xmat, _gradient,
                _resident_design, _timings);
        if (_verbose) {
            std::cout << "Iteration " << t << " objective: " << _object(t) << "; diff: " << _object(t) - _object(t-1) << "; rel diff: " << abs(_object(t) - _object(t - 1)) / (0.1 + abs(_object(t))) << "; score: " << (gradient_crossproduct - lambda * beta).cwiseAbs().maxCoeff() << "\n";
        }

        if ( (_deviance(t) - _deviance(t-1)) > _tol ) {
            std::cout << "\nDeviance increases at iteration " << t << ".\n";
            ii = 0;
            while ( (_deviance(t) - _deviance(t-1)) > _tol ) {
                ++ii;
                if (ii > _mxitnr) {
                    std::cout << "Convergence issue, inner loop: cannot correct step size\n";
                    return;
                    // throw std::runtime_error("inner loop: cannot correct step size");
                }
                beta = (beta + beta_old)/2;
                eta = mask.select(
                    step1_linear_prediction(_compute_backend, Xmat, beta,
                        _resident_design, _timings) + offset_val,
                    0).matrix();
                eta_order = survivalData.permute_mtx * eta;
                _deviance(t) = _coxDeviance(survivalData);
                _object(t) = _deviance(t) + lambda * (beta.array().pow(2).sum())/2;
                if (_verbose) {
                    std::cout << "Iteration " << t << " Halved, Objective: " << _object(t) << "; diff: " << _object(t) - _object(t-1) << ".\n";
                }
            }
        }

        if (abs(_object(t) - _object(t - 1)) / (0.1 + abs(_object(t))) < _tol ||
            (gradient_crossproduct - lambda * beta).cwiseAbs().maxCoeff() < _tol ) {
            converge = true;
            break;
        }
    }

    if (break_pt < (_niter + 1)) {
        _deviance.conservativeResize(break_pt);
        _object.conservativeResize(break_pt);
    }
    dev_ratio =
        1 - _deviance(_deviance.size() - 1) / _null_deviance;
}

Eigen::VectorXd cox_ridge::get_gradient() {
    return _gradient;
}

double cox_ridge::get_deviance() {
    return _deviance(_deviance.size() - 1);
}

double cox_ridge::get_null_deviance() {
    return _null_deviance;
}

double cox_ridge::get_object() {
    return _object(_object.size() - 1);
}

Eigen::VectorXd cox_ridge::get_object_all() {
    return _object;
}

Eigen::VectorXd cox_ridge::get_deviance_all() {
    return _deviance;
}


cox_ridge_path::cox_ridge_path(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask, const int& nlambda, const double& lambda_min_max_ratio, const Eigen::VectorXd& lambda, const int& max_iter, const int& max_inner_iter, const double& tolerance, const bool& verbose_fit, Step1ComputeBackend* compute_backend, const bool& resident_design, Step1ComputeTimings* timings, Step1Level1Optimizer optimizer) {
    _compute_backend = compute_backend;
    _resident_design = resident_design;
    _timings = timings;
    _optimizer = optimizer;
    int p = Xmat.cols();
    // set lambda_vec
    if (lambda.size() > 0) {
        _user_define_lambda = true;
        _lambda_len = lambda.size();
        if (lambda.minCoeff() < 0) { throw std::runtime_error("lambda must >= 0."); }
        lambda_vec = lambda;
        std::sort(lambda_vec.data(), lambda_vec.data() + lambda_vec.size(), std::greater<double>());
    } else {
        double lambda_min_ratio;
        if (lambda_min_max_ratio >= 1) { 
            throw std::runtime_error("lambda_min_max_ratio should be less than 1."); 
        } else if (lambda_min_max_ratio == -1) {
            if (survivalData.neff < p) {
                lambda_min_ratio = 1e-2;
            } else {
                lambda_min_ratio = 1e-4;
            }
        } else {
            lambda_min_ratio = lambda_min_max_ratio;
        }
        _lambda_len = nlambda;
        cox_ridge coxRidge_null_lamb0(survivalData, Xmat, offset_val, mask, 0, max_iter, max_inner_iter, tolerance);
        coxRidge_null_lamb0.coxGrad(survivalData);
        Eigen::VectorXd gradient = coxRidge_null_lamb0.get_gradient();
        double lambda_max = _getCoxLambdaMax(Xmat, gradient);
        // lambda_vec = (Eigen::seq(0, _lambda_len - 1) * log(lambda_min_ratio) + log(lambda_max)).exp();
        Eigen::VectorXd index(nlambda);
        for (int i = 0; i < nlambda; ++i) {
            if (i > 0) {
                index(i) = static_cast<double>(i)/(nlambda - 1);
            } else {
                index(i) = i;
            }
        }
        lambda_vec = (index.array() * log(lambda_min_ratio) + log(lambda_max)).exp();
    }
    beta_mx.resize(p, _lambda_len);
    eta_mx.resize(survivalData.n, _lambda_len);
    object_val.resize(_lambda_len);
    dev_ratio.resize(_lambda_len);
    deviance.resize(_lambda_len);
    converge.resize(_lambda_len);
    niter = max_iter;
    mxitnr = max_inner_iter;
    tol = tolerance;
    verbose = verbose_fit;
}

double cox_ridge_path::_getCoxLambdaMax(const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& gradient) {
    Eigen::VectorXd g = step1_design_crossproduct(
        _compute_backend, Xmat, gradient, _resident_design,
        _timings).array().abs();
    return g.maxCoeff() / 1e-3;
}

void cox_ridge_path::fit(const survival_data& survivalData, const Eigen::MatrixXd& Xmat, const Eigen::VectorXd& offset_val, const ArrayXb& mask) {
    int break_pt = 0;
    double cur_lambda = lambda_vec(0);
    cox_ridge coxRidge(survivalData, Xmat, offset_val, mask, cur_lambda,
        niter, mxitnr, tol, verbose, Eigen::VectorXd(), -999,
        _compute_backend, _resident_design, _timings);
    double nulldev_old = -999;
    Eigen::VectorXd beta_old(Xmat.cols());
    // Both experimental Level 1 modes request the same safeguarded Cox
    // continuation; Newton-CG remains specific to logistic Level 1.
    const bool use_path_newton =
        _optimizer != Step1Level1Optimizer::Irls &&
        _compute_backend && _resident_design;
    Step1CoxPathNewtonPoint exact_point;
    bool exact_point_ready = false;
    bool reusable_weighted_gram = false;
    path_newton_stats = Step1CoxPathNewtonStats();

    for (int k = 0; k < _lambda_len; ++k) {
        ++break_pt;
        bool completed_by_path_newton = false;
        if (k > 0) {
            cur_lambda = lambda_vec(k);
            if(use_path_newton && exact_point_ready &&
               reusable_weighted_gram) {
                const Eigen::VectorXd penalty_multipliers =
                    Eigen::VectorXd::Ones(Xmat.cols());
                const Step1CoxPathNewtonSolve solve =
                    [&] (const Eigen::VectorXd& score, double penalty,
                         Eigen::VectorXd& step) {
                        const Eigen::MatrixXd right_hand_side = score;
                        Eigen::VectorXd ridge_parameter(1);
                        ridge_parameter(0) = penalty;
                        Eigen::MatrixXd solution;
                        if(!_compute_backend->solve_cached_weighted_gram(
                             right_hand_side, ridge_parameter,
                             penalty_multipliers, solution, _timings) ||
                           solution.cols() != 1)
                            return false;
                        step = solution.col(0);
                        return true;
                    };
                const Step1CoxPathNewtonEvaluate evaluate =
                    [&] (const Eigen::VectorXd& coefficients,
                         Step1CoxPathNewtonPoint& point) {
                        cox_ridge candidate(
                            survivalData, Xmat, offset_val, mask, cur_lambda,
                            niter, mxitnr, tol, false, coefficients, nulldev_old,
                            _compute_backend, _resident_design, _timings);
                        return candidate.exact_path_point(
                            survivalData, Xmat, point);
                    };
                const Step1CoxPathNewtonResult path_result =
                    run_step1_cox_path_newton(
                        exact_point, cur_lambda, tol, solve, evaluate);
                Step1CoxPathNewtonStats result_stats = path_result.stats;
                if(!path_result.converged)
                    ++result_stats.fallbacks;
                accumulate_step1_cox_path_newton_stats(
                    path_newton_stats, result_stats);
                exact_point = path_result.accepted_point;
                exact_point_ready = true;
                if(path_result.converged) {
                    coxRidge.install_path_point(exact_point, cur_lambda);
                    completed_by_path_newton = true;
                } else {
                    coxRidge.reset_from_path(
                        survivalData, Xmat, offset_val, mask, cur_lambda,
                        exact_point.coefficients, nulldev_old,
                        exact_point.deviance);
                }
            } else {
                coxRidge.reset(
                    survivalData, Xmat, offset_val, mask, cur_lambda,
                    beta_old, nulldev_old);
            }
        }
        if (verbose) {
            std::cout << "lambda: " << cur_lambda << "\n";
        }
        if(!completed_by_path_newton) {
            coxRidge.fit(survivalData, Xmat, offset_val, mask);
            path_newton_stats.ordinary_weighted_solves +=
                coxRidge._weighted_solve_count;
            if(use_path_newton) {
                exact_point_ready = coxRidge.converge &&
                    coxRidge.exact_path_point(
                        survivalData, Xmat, exact_point);
                reusable_weighted_gram =
                    exact_point_ready &&
                    coxRidge._weighted_solve_count > 0;
            }
        }
        std::cout << "converge: " << coxRidge.converge << "\n";

        if (coxRidge.converge == false) {
            converge(k) = false;
            std::cout << "Warning: lambda " << cur_lambda << " failed to converge.\n";
        } else {
            converge(k) = true;
        }
        beta_old = coxRidge.beta;
        nulldev_old = coxRidge.get_null_deviance();
        beta_mx.col(k) = coxRidge.beta;
        eta_mx.col(k) = coxRidge.eta;
        deviance(k) = coxRidge.get_deviance();
        dev_ratio(k) = coxRidge.dev_ratio;
        object_val(k) = coxRidge.get_object();

        if (k > 4 && _user_define_lambda == false) {
            if (dev_ratio(k) > 0.99) { break; }
            if (k > 0 && (dev_ratio(k) - dev_ratio(k - 3)) <
                1e-3 * dev_ratio(k)) { break; }
        }
    }

    if (break_pt < _lambda_len) {
        beta_mx.conservativeResize(Xmat.cols(), break_pt);
        eta_mx.conservativeResize(survivalData.n, break_pt);
        deviance.conservativeResize(break_pt);
        dev_ratio.conservativeResize(break_pt);
        lambda_vec.conservativeResize(break_pt);
        object_val.conservativeResize(break_pt);
        converge.conservativeResize(break_pt);
    }
}
