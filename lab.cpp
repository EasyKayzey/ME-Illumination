//
// Created by erezm on 2021-01-31.
//

#include "lab.h"

pair<OArr, OArr> get_noisy_observables(const FGenome& epsilon, rng gen, FConstants& constants) {
    EMatrix& mu = get<0>(constants);
    EDMatrix& C = get<1>(constants);
    array<ECovector, DIM>& analysis_population = get<2>(constants);
    EVector& psi_i = get<3>(constants);
//    pair<HGenome, HGenome>& apx_bounds = get<4>(constants);

    vector<OArr> samples(N_REP);
    uniform_real_distribution<> UO(-obs_err, obs_err), UA(1 - amp_err, 1 + amp_err);
    //US(-sub_err * 2 * M_PI, sub_err * 2 * M_PI);

    vector<double> rda(N_REP * L);
    for (double& i : rda)
        i = UA(gen);

    vector<double> rdo(N_REP * N_OBS);
    for (double& i : rdo)
        i = UO(gen);

    auto diag_ret = diag_vec(mu, C);
    EVector lambda = diag_ret.second;
    EMatrix CP = diag_ret.first.first;
    EMatrix PdC = diag_ret.first.second;
    for (int run = 0; run < N_REP; ++run) {
        FGenome cur_epsilon = epsilon;
        for (int i = 0; i < L; ++i)
            cur_epsilon[i].first *= rda[run * L + i];
        vector<double> eps_inter = f_to_inter_t(cur_epsilon);
        OArr trans_prob = evolve_initial(eps_inter, CP, PdC, PdC * CP, lambda, psi_i, analysis_population);
        for (int i = 0; i < N_OBS; ++i)
            trans_prob[i] += rdo[run * N_OBS + i];
        samples[run] = trans_prob;
    }



    array<double, N_OBS> min_e{}, max_e{};
    min_e.fill(1);
    max_e.fill(0);
    for (int o = 0; o < N_OBS; ++o) {
        for (int i = 0; i < N_REP; ++i) {
            min_e[o] = min(min_e[o], samples[i][o]);
            max_e[o] = max(max_e[o], samples[i][o]);
        }
        min_e[o] = max(min_e[o], 0.);
        max_e[o] = min(max_e[o], 1.);
    }

    array<double, N_OBS> errors{};
    for (int i = 0; i < N_OBS; ++i)
        errors[i] = abs(max_e[i] - min_e[i]) / 2;

    OArr observables; // the following code was used for returning average, but was removed for bound center
//    for (int o = 0; o < N_OBS; ++o) {
//        observables[o] = 0;
//        for (int i = 0; i < N_REP; ++i)
//            observables[o] += samples[i][o];
//        observables[o] /= N_REP;
//    }
    for (int i = 0; i < N_OBS; ++i)
        observables[i] = (max_e[i] + min_e[i]) / 2;

    return {observables, errors};
}