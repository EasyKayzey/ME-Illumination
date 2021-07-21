//
// Not Created by erezm on 2020-05-1.
//

#include "illumination.h"

extern int ME_NE_MAX, ME_BS, ME_DE, ME_C0H_EXIT;
extern double P_HC, P_HM, S_HM;
extern int main_start_time, max_runtime_ME;
extern int max_seeds_ME;
extern int N_LINES, N_LBINARY, N_LSAMPLES;
extern double L_EXIT, L_BIAS;
extern double h_guess_err;
extern HGenome mu_true;

unordered_set<HGenome> global_seeds;

pair<HGenome, HGenome> run_illumination(FConstants& constants, const vector<double>& eps_inter, const OArr& obs_avg,
                                        const OArr& obs_err, rng& gen, function<int(const BArr&)>& get_grid_idx) {
//    EMatrix& mu = get<0>(constants);
    EDMatrix& C = get<1>(constants);
    array<ECovector, DIM>& anal_pop = get<2>(constants);
    EVector& psi_i = get<3>(constants);
    pair<HGenome, HGenome>& apx_bounds = get<4>(constants);

    function<OArr(const HGenome&)> get_obs = [&](const HGenome& h) -> OArr {
        EMatrix mu = gen_to_mat(h);
        auto diag_ret = diag_vec(mu, C);
        EVector lambda = diag_ret.second;
        EMatrix CP = diag_ret.first.first;
        EMatrix PdC = diag_ret.first.second;
        OArr trans_prob = evolve_initial(eps_inter, CP, PdC, PdC * CP, lambda, psi_i, anal_pop);
        return trans_prob;
    };
    function<double(const OArr&, const HGenome&)> get_cost = [&](const OArr& sim_obs, const HGenome& mu) -> double {
        double cost = 0;
        for (int i = 0; i < N_OBS; ++i) {
            double c = sim_obs[i], l = obs_avg[i], e = obs_err[i];
            if (abs(c - l) > e)
                return -1;
            double t = mu[i] - mu_true[i];
            cost += t * t;
        }
        return cost;
    };
    function<BArr(const OArr&)> get_beh = [&](const OArr& sim_obs) -> BArr {
        BArr beh;
        for (int i = 0; i < N_BEH; ++i)
            beh[i] = (sim_obs[i] - obs_avg[i] + obs_err[i]) / (2 * obs_err[i]);
        return beh;
    };
    auto MER = invert_ME(apx_bounds, get_obs, get_beh, get_grid_idx, get_cost, gen);
    return MER;
}

#define _LOG_ME true
#define _PARA_ME true
#define _USE_ALL_C0H true
#define _USE_GLOBAL_SEEDS true
#define _SEED_ALL_C0H_GLOBALLY true
#define _REVITALIZE_ON_CURIOSITY_DEATH true
pair<HGenome, HGenome> invert_ME(const pair<HGenome, HGenome>& apx_bounds, function<OArr(const HGenome&)>& get_obs,
                                 function<BArr(const OArr&)>& get_beh, function<int(const BArr&)>& get_grid_idx,
                                 function<double(const OArr&, const HGenome&)>& get_cost, rng& gen) {
#if _LOG_ME
    cout << "Starting MAP-Elites" << endl;
#endif
    time_t ME_start_time = time(nullptr);
    vector<MEPop> archive(N_GRID);
    vector<bool> archive_has(N_GRID, false);
    uniform_real_distribution<> U1(0, 1);
    normal_distribution<> NM(0, S_HM);
    int n_eval = 0;
    int C0H_in_arch = 0;
    int C0H_found = 0;
    double min_cost = 1e10;
    pair<HGenome, HGenome> C0H_bounds;
    for (int i = 0; i < N_H; ++i) {
        C0H_bounds.first[i] = 1e10;
        C0H_bounds.second[i] = -1e10;
    }


    vector<HGenome> init_pop_gen(ME_BS);
    vector<int> init_pop_loc(ME_BS);
    vector<MEPop> init_pop(ME_BS);
    vector<int> full_locs;
    vector<int> archive_curiosity_index(N_GRID, 0);
    vector<int> parent_of(ME_BS);

#if _USE_GLOBAL_SEEDS
    {
        int nc0s = 0;
        double stochastic_include = (double) max_seeds_ME / global_seeds.size();
#if _LOG_ME
        if (stochastic_include < 1)
            cout << "Stochastically evaluating " << (int) round(stochastic_include * 100) << "% of "
            << global_seeds.size() << " global seeds..." << endl;
        else
            cout << "Evaluating all " << global_seeds.size() << " global seeds..." << endl;
#endif
        vector<bool> seed_saves_using(global_seeds.size());
        vector<tuple<HGenome, OArr, double, BArr, int>> seed_saves(global_seeds.size());
#pragma omp parallel
#pragma omp single
        {
            int seed_i = 0;
            for (auto it = global_seeds.begin(); it != global_seeds.end(); ++it, ++seed_i) {
                bool is_using = U1(gen) < stochastic_include;
                seed_saves_using[seed_i] = is_using;
                if (is_using) {
#pragma omp task firstprivate(it) shared(seed_saves)
                    {
                        OArr obs = get_obs(*it);
                        double cost = get_cost(obs, *it);
                        int idx = -1;
                        BArr beh = {NAN};
                        if (cost >= 0) {
                            beh = get_beh(obs);
                            idx = get_grid_idx(beh);
                        }
                        seed_saves[seed_i] = {*it, obs, cost, beh, idx};
                    }
                }
            }
        }


        {
            cout << "Running lines:" << endl;

            vector<HGenome> line_starts(N_LINES);
            int N_HARDCODED_STARTS = N_H * 2;
            if (N_LINES < N_HARDCODED_STARTS) {
                cout << "ERROR: N_LINES is leq to N_HARDCODED_STARTS. This will be ignored." << endl;
                cerr << "ERROR: N_LINES is leq to N_HARDCODED_STARTS. This will be ignored." << endl;
            }
            for (int l = 0; l < N_HARDCODED_STARTS; ++l) {
                HGenome line_start = mu_true;
                line_start[l / 2] = line_start[l / 2] * (1 + ((l % 2) * 2 - 1) * h_guess_err);
                line_starts[l] = line_start;
            }

            for (int l = N_HARDCODED_STARTS; l < N_LINES; ++l) {
                HGenome line_start{};
                for (int i = 0; i < N_H; ++i)
                    line_start[i] = normalize(U1(gen), apx_bounds.first[i], apx_bounds.second[i]);
                line_starts[l] = line_start;
            }

            vector<double> bias_vals(N_LINES * N_LSAMPLES);
            for (int i = 0; i < N_LINES * N_LSAMPLES; ++i)
                bias_vals[i] = U1(gen);

            atomic_int line_exit_sum = 0;
            vector<vector<tuple<HGenome, OArr, double, BArr, int>>> line_seeds(N_LINES);
#pragma omp parallel for default(none) shared(N_LINES, N_LBINARY, N_LSAMPLES, L_BIAS, L_EXIT, get_obs, get_cost, get_beh, get_grid_idx, line_starts, apx_bounds, mu_true, line_exit_sum, line_seeds, bias_vals)
            for (int l = 0; l < N_LINES; ++l) {
                HGenome &line_start = line_starts[l], biased_endpoint;
                HGenome lower = mu_true, upper = line_start, mid;
                int bsi;
                for (bsi = 0; bsi < N_LBINARY && sqdist_HG(lower, upper) > L_EXIT; ++bsi) {
                    mid = bias_HG(lower, upper, 0.5);
                    OArr obs = get_obs(mid);
                    double cost = get_cost(obs, mid);
                    if (cost < 0) {
                        BArr beh{NAN};
                        line_seeds[l].emplace_back(mid, obs, cost, beh, -1);
                        upper = mid;
                    } else {
                        BArr beh = get_beh(obs);
                        int idx = get_grid_idx(beh);
                        line_seeds[l].emplace_back(mid, obs, cost, beh, idx);
                        lower = mid;
                    }
                }
                line_exit_sum += bsi;
                biased_endpoint = bias_HG(mu_true, mid, L_BIAS);
                for (int i = 0; i < N_LSAMPLES; ++i) {
                    HGenome cur_sam = bias_HG(mu_true, biased_endpoint, bias_vals[l * N_LSAMPLES + i]);
                    OArr obs = get_obs(cur_sam);
                    double cost = get_cost(obs, cur_sam);
                    if (cost < 0) {
                        BArr beh{NAN};
                        line_seeds[l].emplace_back(cur_sam, obs, cost, beh, -1);
                    } else {
                        BArr beh = get_beh(obs);
                        int idx = get_grid_idx(beh);
                        line_seeds[l].emplace_back(cur_sam, obs, cost, beh, idx);
                    }
                }
            }

            for (auto& t : line_seeds) {
                for (auto& tu : t) {
                    seed_saves.push_back(tu);
                    seed_saves_using.push_back(true);
                }
            }
            cout << "Average line binary search exit length is " << (double) line_exit_sum / N_LBINARY << endl;
        }

        cout << "Testing all evaluated seeds..." << endl;
        for (int i = 0; i < (int) seed_saves.size(); ++i) {
            if (seed_saves_using[i]) {
                auto &c_seed_s = seed_saves[i];
                HGenome c_seed = get<0>(c_seed_s);
                double cost = get<2>(c_seed_s);
                if (cost >= 0) {
#if _USE_ALL_C0H
                    for (int j = 0; j < N_H; ++j) {
                        C0H_bounds.first[j] = min(C0H_bounds.first[j], c_seed[j]);
                        C0H_bounds.second[j] = max(C0H_bounds.second[j], c_seed[j]);
                    }
#endif
                    BArr beh = get<3>(c_seed_s);
                    int idx = get<4>(c_seed_s);
                    if (!archive_has[idx]) {
                        archive[idx] = {cost, c_seed, beh};
                        archive_has[idx] = true;
                        full_locs.push_back(idx);
                        archive_curiosity_index[idx] = 2600;
                        ++C0H_in_arch;
                        ++C0H_found;
                        ++nc0s;
                    }
                }
            }
        }

        if (nc0s == 0) {
            cout << "PROBLEM: No cost-zero Hamiltonians found. This message is long so that it will be easily "
                 << "noticed. Returning a large value and setting main start time to be zero." << endl;
            cerr << "PROBLEM: No cost-zero Hamiltonians found. This message is long so that it will be easily "
                 << "noticed. Returning a large value and setting main start time to be zero." << endl;
            main_start_time = 0;
            HGenome low, high;
            low.fill(-100);
            high.fill(100);
            return {low, high};
        }
//            endpoint_O /= 10;
//            extern Oracle *oracle_ptr;
//            HGenome ret = get<0>(oracle_ptr->run_oracle());
//            generated_seeds.emplace(ret);
//            OArr obs = get_obs(ret);
//            double cost = get_cost(obs, ret);
//            if (cost != 0) {
//                cout << "Oracle failed to find C0H with endpoint " << endpoint_O << endl;
//                cerr << "Oracle failed to find C0H with endpoint " << endpoint_O << endl;
//                throw runtime_error("Second oracle failed.");
//            } else {
//#if _USE_ALL_C0H
//                for (int j = 0; j < N_H; ++j) {
//                    C0H_bounds.first[j] = min(C0H_bounds.first[j], ret[j]);
//                    C0H_bounds.second[j] = max(C0H_bounds.second[j], ret[j]);
//                }
//#endif
//                BArr beh = get_beh(obs);
//                int idx = get_grid_idx(beh);
//                if (!archive_has[idx]) {
//                    archive[idx] = {cost, ret, beh};
//                    archive_has[idx] = true;
//                    full_locs.push_back(idx);
//                    archive_curiosity_index[idx] = 3000;
//                    ++C0H_in_arch;
//                    ++C0H_found;
//                    ++nc0s;
//                }
//            }
//        }
        cout << "Used " << nc0s << " global seeds." << endl;
    }
#endif
//    while (n_eval < ME_NE_MAX && C0H_in_arch < ME_C0H_EXIT && time(nullptr) - start_time < max_runtime) { NOTE IF
//    YOU UNCOMMENT THIS THESE VALUES ARE NOT DEFINED IN TEST.CPP
    bool run_ME = true;
#if _REVITALIZE_ON_CURIOSITY_DEATH
    int n_c_deaths = 0;
#endif
    while (run_ME) {
        for (int i = 0; i < ME_BS; ++i) {
            if (U1(gen) < P_HC) {
                // crossover
                int p1 = curiosity_select(full_locs, archive_curiosity_index, archive_has, U1(gen));
                int p2 = curiosity_select(full_locs, archive_curiosity_index, archive_has, U1(gen));
                int gamma = U1(gen);
                for (int j = 0; j < N_H; ++j)
                    init_pop_gen[i][j] = gamma * get<1>(archive[p1])[j] + (1 - gamma) * get<1>(archive[p2])[j];
                parent_of[i] = p1;
            } else {
                // random variation
                int p1 = curiosity_select(full_locs, archive_curiosity_index, archive_has, U1(gen));
                init_pop_gen[i] = get<1>(archive[p1]);
                for (int j = 0; j < N_H; ++j)
                    init_pop_gen[i][j] += (U1(gen) < P_HM) * sqrt(abs(init_pop_gen[i][j])) * NM(gen);
                parent_of[i] = p1;
            }
        }
#if _PARA_ME
#pragma omp parallel for default(none) shared(ME_BS, init_pop, init_pop_gen, init_pop_loc, get_obs, get_beh, get_cost, get_grid_idx)
#endif
        for (int i = 0; i < ME_BS; ++i) {
            OArr obs = get_obs(init_pop_gen[i]);
            double c_cost = get_cost(obs, init_pop_gen[i]);
            if (c_cost < 0) {
                init_pop[i] = {c_cost, {}, {}};
            } else {
                init_pop[i] = {c_cost, init_pop_gen[i], get_beh(obs)};
                init_pop_loc[i] = get_grid_idx(get<2>(init_pop[i]));
            }
//            for (int j = 0; j < N_OBS; ++j)
//                cout << obs[j] << ' ';
//            cout << endl;
//            for (int j = 0; j < N_BEH; ++j)
//                cout << get<2>(init_pop[i])[j] << ' ';
//            cout << endl;
        }
        for (int i = 0; i < ME_BS; ++i) {
            if (get<0>(init_pop[i]) >= 0) {
                int idx = init_pop_loc[i];
                if (!archive_has[idx]) {
                    archive[idx] = init_pop[i];
                    full_locs.push_back(idx);
                    archive_has[idx] = true;
                    ++C0H_in_arch;
//#if _LOG_ME
//                    cout << "new @ " << idx << endl;
//#endif
                    archive_curiosity_index[idx] = 2200;
                    archive_curiosity_index[parent_of[i]] += 200;
                } else if (get<0>(init_pop[i]) < get<0>(archive[idx])) {
                    archive[idx] = init_pop[i];
//                    int cur_curi_val = archive_curiosity_index[idx];
//                    archive_curiosity_index[idx] = min(max(cur_curi_val, 2200), cur_curi_val + 20);
                    archive_curiosity_index[idx] += 6;
                    archive_curiosity_index[parent_of[i]] -= 7;
                } else {
//#if _LOG_ME
//                    cout << "discarded @ " << idx << endl;
//#endif
                    archive_curiosity_index[parent_of[i]] -= 6;
                }
                min_cost = min(min_cost, get<0>(init_pop[i]));
                ++C0H_found;
#if _USE_ALL_C0H
                for (int j = 0; j < N_H; ++j) {
                    C0H_bounds.first[j] = min(C0H_bounds.first[j], init_pop_gen[i][j]);
                    C0H_bounds.second[j] = max(C0H_bounds.second[j], init_pop_gen[i][j]);
                }
#endif
            } else {
//                cout << "rejected non-C0H with cost " << get<0>(init_pop[i]) << endl;
                archive_curiosity_index[parent_of[i]] += 2;
            }
        }
        n_eval += ME_BS;
        auto norm_res = normalize_curiosity(archive_curiosity_index);
#if _LOG_ME
        if (n_eval % 100000 == 0) {
            cout << "eval'd " << n_eval << ", filled " << full_locs.size() << '/' << ME_C0H_EXIT
//                << '/' << N_GRID
//                 << ", " << C0H_in_arch << " with C0H"
#if _USE_ALL_C0H
                 << "; found " << C0H_found << " C0H total"
#endif
//                 << "; min cost " << min_cost
                 << "; curiosity sum " << norm_res.second << "; non-zero curiosity num " << norm_res.first
                 << "; n deaths " << n_c_deaths
                 << endl;
        }
#endif

#if _REVITALIZE_ON_CURIOSITY_DEATH
        if (norm_res.second < N_GRID * 7) {
            for (int l : full_locs)
                archive_curiosity_index[l] = 1000;
            ++n_c_deaths;
        }
#endif

        run_ME = n_eval < ME_NE_MAX
#if _REVITALIZE_ON_CURIOSITY_DEATH == false
                && any_of(archive_curiosity_index.begin(), archive_curiosity_index.end(), [](int x){return x > 0;})
                && norm_res.second >= N_GRID
#endif
#define EXIT_ON_GRID_CONDITION false
#if EXIT_ON_GRID_CONDITION
                && C0H_in_arch < ME_C0H_EXIT
#endif
#undef EXIT_ON_GRID_CONDITION
                && time(nullptr) - ME_start_time < max_runtime_ME;
    }
    auto norm_res = normalize_curiosity(archive_curiosity_index);
#if _LOG_ME
    cout << "MAP_Elites done with " << n_eval << " evaluations in " << time(nullptr) - ME_start_time << "sec, "
        << C0H_in_arch << " cells filled with C0H, curiosity sum "
        << norm_res.second << ", nonzero curiosity num " << norm_res.first
#if _REVITALIZE_ON_CURIOSITY_DEATH
        << ", curiosity deaths " << n_c_deaths
#endif
        << endl;
//    if (C0H_in_arch < (ME_C0H_EXIT * 4) / 5)
//        cout << "CURIOSITY DEATH IN ABOVE RUN." << endl;
#endif
#if _USE_ALL_C0H == false
    for (int i = 0; i < N_GRID; ++i) {
        if (get<0>(archive[i]) == 0) {
            for (int j = 0; j < N_H; ++j) {
                C0H_bounds.first[j] = min(C0H_bounds.first[j], get<1>(archive[i])[j]);
                C0H_bounds.second[j] = max(C0H_bounds.second[j], get<1>(archive[i])[j]);
            }
        }
    }
#endif
#if _SEED_ALL_C0H_GLOBALLY
    for (int loc : full_locs)
        global_seeds.emplace(get<1>(archive[loc]));
#endif
#define UPDATE_C0H_EXIT_NUM true
#if UPDATE_C0H_EXIT_NUM
    ME_C0H_EXIT = max(C0H_in_arch, ME_C0H_EXIT);
#endif
#undef UPDATE_C0H_EXIT_NUM
#if UPDATE_ME_MUTATION_RATE
    extern double cur_gen_mut, dyn_mut_target;
    cur_gen_mut += ((C0H_found + 0.0) / n_eval) - dyn_mut_target;
#endif
    return C0H_bounds;
}

HGenome mat_to_gen(const EMatrix& matrix) {
    int c = 0;
    HGenome h;
    for (int i = 1; i < DIM; ++i)
        for (int j = 0; j < i; ++j)
            h[c++] = matrix(i, j).real();
    return h;
}

EMatrix gen_to_mat(const HGenome& genome) {
    int c = 0;
    EMatrix matrix = EMatrix::Zero();
    for (int i = 1; i < DIM; ++i)
        for (int j = 0; j < i; ++j)
            matrix(i, j) = genome[c++];
    return matrix + matrix.transpose();
}

int tournament_select(const array<MEPop, N_GRID>& archive, const vector<int>& full_locs, rng& gen, uniform_real_distribution<> U1) {
#define N_TOURN 15
    int best = -1;
    double best_cost = 1e10;
    for (int i = 0; i < N_TOURN; ++i) {
        int l = (int) (U1(gen) * full_locs.size());
        if (get<0>(archive[full_locs[l]]) < best_cost) {
            best = l;
            best_cost = get<0>(archive[full_locs[l]]);
        }
    }
    return full_locs[best];
#undef N_TOURN
}


int curiosity_select(const vector<int> &full_locs, const vector<int> &curiosity_index,
                     const vector<bool> &archive_has, double rand) {
    vector<double> p_sums(N_GRID + 1);
    p_sums[0] = 0;
    for (int i = 0; i < N_GRID; ++i)
        p_sums[i + 1] = p_sums[i] + curiosity_index[i] * archive_has[i];
    double rc = rand * p_sums[N_GRID];
    return lower_bound(&p_sums[0], &p_sums[N_GRID + 1], rc) - &p_sums[0] - 1;
}

pair<int, int> normalize_curiosity(vector<int> &curiosity_index) {
    int sum = 0, num = 0;
    for (int i = 0; i < N_GRID; ++i) {
        if (curiosity_index[i] < 0)
            curiosity_index[i] = 0;
        if (curiosity_index[i] > 0)
            ++num;
        sum += curiosity_index[i];
    }
    return {num, sum};
}

HGenome bias_HG(HGenome& t, HGenome& o, double bias) {
    HGenome c{};
    for (int i = 0; i < N_H; ++i)
        c[i] = bias * o[i] - (bias - 1) * t[i];
    return c;
}

double sqdist_HG(HGenome& a, HGenome& b) {
    double sqdist = 0;
    for (int i = 0; i < N_H; ++i)
        sqdist += (a[i] - b[i]) * (a[i] - b[i]);
    return sqdist;
}