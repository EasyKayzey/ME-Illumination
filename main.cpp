//
// Created by erezm on 2021-01-31.
//
#include "main.h"
#include "illumination.h"
#include "lab.h"
#include "cvt.h"

double T = 4000, DELTA_T, N_T_double = 2500;
int N_T;
double h_guess_err = 0.20;
double obs_err = 0.01, amp_err = 0.001;
int ME_NE_MAX = 3000000, ME_BS = 2000, ME_C0H_EXIT = 100;
double P_HC = .1, P_HM = .3, S_HM = 15e-3;
int N_LINES = 10000, N_LBINARY = 10, N_LSAMPLES = 10;
double L_EXIT = 1e-10, L_BIAS = 1.5;
int N_FP = 50, N_FGEN = 2, N_FTOURN = 3, GR_F = N_FP - 2, N_PRE_SEEDING = 1;
double S_FA = 0.05, S_FT = 0.1, A_MAX = 0.5, A_MIN = A_MAX * 0.5, P_FC = 0.6, B_FC = 0.3, P_FM = 0.3, BETA_FCOST = 0;
int main_start_time, max_runtime_ME = 900, max_runtime_GGA = 24*60*60;
int max_seeds_ME = ME_NE_MAX / 1.5;
double cost_multiplier = 1;
int seed;
int N_PFA = 2, N_PFS = 2;
#if UPDATE_ME_MUTATION_RATE
double cur_gen_mut = 0, dyn_mut_para = 0.005 / N_FP, dyn_mut_target = .4;
#endif
array<double, L> omega;
HGenome mu_true;

extern unordered_set<HGenome> global_seeds;

#define FIELD_TESTING false

#define _PARA_GGA false
#define _SAVE_ALL_POP true
int main(int argc, char** argv) {
    { // this will only work until 2038 so be careful
        time_t now;
        main_start_time = time(&now);
        assert(now == main_start_time);
        ptime();
    }
    hash<string> hash_string;
    seed = hash_string(to_string(main_start_time));
    seed = 1000;

    double init_S_HM = S_HM;

    if (argc > 1) {
        cout << "Setting " << (argc - 2) << " command-line argument values: ";
        for (int i = 1; i < argc; ++i)
            cout << argv[i] << " ";
        cout << endl;
#if FIELD_TESTING == false
        if (argc > 2) {
            A_MAX = stod(argv[2], nullptr);
            A_MIN = A_MAX * .5;
        }
        if (argc > 3) {
            T = stod(argv[3], nullptr);
            N_T_double = N_T_double * T / 10;
        }
        if (argc > 4)
            cost_multiplier = stod(argv[4], nullptr);
#endif
    }

    N_T = (int) round(N_T_double);
    DELTA_T = T / N_T;

    string message = "LOR";
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            message += '_';
            message += argv[i];
        }
    }

#if FIELD_TESTING
    vector<FGenome> population;
    {
        ifstream field_file(path + string(argv[2]) + ".txt");
        if (field_file.good()) {
            cout << "Using fields from " << argv[2] << ".txt" << endl;
            try {
                int al;
                field_file >> al;
                N_FP = al;
                population.resize(N_FP);
                FGenome temp_field{};
                for (int i = 0; i < N_FP; ++i) {
                    for (int j = 0; j < L; ++j)
                        field_file >> temp_field[j].first;
                    for (int j = 0; j < L; ++j)
                        field_file >> temp_field[j].second;
                    population[i] = temp_field;
                }
            } catch (...) {
                cout << "Reading fields failed..." << endl;
                exit(0);
            }
        } else {
            cout << "Reading fields failed..." << endl;
            exit(0);
        }
    }
    vector<FGenome> next_population(N_FP);
    vector<tuple<double, double, HGenome, HGenome>> cur_costs(N_FP);
#endif

    if (cost_multiplier < 0)
        message += "REV";

    string filename = string(path) + "MEG1DATA" + to_string(main_start_time) + (message == "#" ? "" : message);

    cout << "Message: " << message << endl;
    cout << "Seed: " << seed << endl;
    cout << "Filename: " << filename << endl;
    srand(seed);
    rng gen(seed);
    uniform_real_distribution<> U1(0, 1), UT(0, 2 * M_PI);
    normal_distribution<> NA(1, S_FA), NT(0, S_FT);

    EVector H0D;
    EMatrix mu;
    // h0, mu values
    {
        H0D << 0, 0.00820226918, 0.01558608386, 0.02215139847, 0.02789825857, 0.03282661861;
//    cout << "H_0 diagonal:" << endl << H0.toDenseMatrix() << endl;

        EMatrix mu_upper;
        mu_upper <<  0, 0.06116130402, -0.01272999623, 0.003275382148, -0.0009801377618, 0.0003314304586,
                     0, 0,              0.0834968862, -0.02150346764,   0.006457586337, -0.002195579355,
                     0, 0,              0,             0.09850502453,  -0.02960110398,   0.01003668145,
                     0, 0,              0,             0,               0.1093531798,   -0.0371225703,
                     0, 0,              0,             0,               0,               0.1171966409,
                     0, 0,              0,             0,               0,               0;

        mu = mu_upper + mu_upper.transpose();
//        cout << "mu:" << endl << mu.real() << endl;
    }
    mu_true = mat_to_gen(mu);

    for (int l = 1, c = 0; l < DIM; ++l)
        for (int m = 0; m < l; ++m)
            omega[c++] = (H0D(l) - H0D(m)).real() / HBAR;

    array<ECovector, DIM> analysis_population; // note this only works because N_OBS = N_TO * DIM
    for (int i = 0; i < DIM; ++i) {
        ECovector cur = ECovector::Zero();
        cur(i) = 1;
        analysis_population[i] = cur;
    }

    EVector psi_i = EVector::Zero();
    psi_i[0] = 1;
    cout << "psi_i:" << endl << psi_i.transpose() << endl;

    EDMatrix C = exp(H0D.array() * -1i * DELTA_T / 2 / HBAR).matrix().asDiagonal();

    pair<HGenome, HGenome> apx_bounds;
    { // randomized guess
        uniform_real_distribution<> down(1 - h_guess_err, 1), up(1, 1 + h_guess_err);
        HGenome min = mat_to_gen(mu);
        HGenome max = min;
        for (int i = 0; i < L; i++) {
            if (min[i] > 0) {
                min[i] *= 1 - h_guess_err;
                max[i] *= 1 + h_guess_err;
            } else if (min[i] < 0) {
                min[i] *= 1 + h_guess_err;
                max[i] *= 1 - h_guess_err;
            } else {
                max[i] = .001 * (up(gen) - 1);
                min[i] = .001 * (down(gen) - 1);
            }
        }
        apx_bounds = {min, max};
    }
//    { // hardcoded
//        EMatrix fullMax, fullMin;
//        HGenome genMax, genMin;
//        genMin = {0,0,0,0,0,0,0,0,0,0};
//        genMax = {0,0,0,0,0,0,0,0,0,0};
//        fullMin = gen_to_mat(genMin);
//        fullMax = gen_to_mat(genMax);
//        apx_bounds = {fullMin, fullMax};
//    }

    global_seeds.emplace(mu_true);
    vector<tuple<HGenome, vector<double>, double>> oracle_rets(N_LINES);
    bool use_file_seeds = argc > 1;
    if (use_file_seeds) {
        ifstream seed_file(path + string(argv[1]) + ".txt");
        if (seed_file.good()) {
            cout << "Using seeds from " << argv[1] << ".txt" << endl;
            try {
                int al;
                seed_file >> al;
                HGenome seed{};
                for (int i = 0; i < al; ++i) {
                    for (int j = 0; j < N_H; ++j)
                        seed_file >> seed[j];
                    global_seeds.emplace(seed);
                }
            } catch (...) {
                global_seeds.clear();
                cout << "Seed reading failed." << endl;
                cerr << "Seed reading failed." << endl;
                use_file_seeds = false;
            }
        } else {
            use_file_seeds = false;
        }
        seed_file.close();
        cout << "Post-read seed size: " << global_seeds.size() << endl;
    }
//    if (!use_file_oracles) {
//        for (int r = 0; r < N_ORACLE; ++r) {
//            if (N_ORACLE != 1)
//                cout << "Running oracle " << r << "..." << endl;
//                //        throw runtime_error("Non-zero N_ORACLE not implemented.");
//            else
//                cout << "Running oracle..." << endl;
//
//            for (int i = 0; i < N_H; ++i)
//                guess_h[i] = normalize(U1(gen), apx_bounds.first[i], apx_bounds.second[i]);
//
//            Oracle oracle{gen, mat_to_gen(mu), psi_i, analysis_population[DIM - 1], omega,
//                          analysis_population, H0D, guess_h};
//            oracle_ptr = &oracle;
//
//            oracle_rets[r] = oracle.run_oracle();
//            global_seeds.emplace(get<0>(oracle_rets[r]));
//            ptime();
//        }
//        write_seeds(filename + "_ORACLE.txt");
//    }

    vector<double> cvt_arr(N_GRID * N_BEH);
    bool use_file_cvt = true;
    if (use_file_cvt) {
        ifstream cvt_file(path + to_string(N_GRID) + ".cvt");
        if (cvt_file.good()) {
            cout << "Using CVT from " << N_GRID << ".cvt" << endl;
            try {
                int CVTL;
                cvt_file >> CVTL;
                if (CVTL != N_GRID * N_BEH) {
                    cout << "Incorrect CVT length at top of CVT file." << endl;
                    throw runtime_error("Incorrect CVT length at top of CVT file.");
                }
                for (int i = 0; i < CVTL; ++i)
                    cvt_file >> cvt_arr[i];
            } catch (...) {
                cout << "CVT reading failed, creating CVT:" << endl;
                use_file_cvt = false;
            }
        } else {
            use_file_cvt = false;
        }
        cvt_file.close();
    }
    if (!use_file_cvt) {
        cout << "Starting CVT..." << endl;
        int cvt_seed = 123456789, cvt_iter_out;
        double cvt_diff, cvt_energy;
        cvt(N_BEH, N_GRID, 1000, 0, 0, 10000, 1000, 50, &cvt_seed,
            cvt_arr.data(), &cvt_iter_out, &cvt_diff, &cvt_energy);
        ofstream cvt_file(path + to_string(N_GRID) + message + ".cvt");
        cvt_file << N_GRID * N_BEH << endl;
        cvt_file << setprecision(numeric_limits<long double>::digits10 + 1);
        for (double i : cvt_arr)
            cvt_file << i << ' ';
        cvt_file << endl;
    }

    function<int(const BArr&)> get_grid_idx = [&](const BArr& loc) {
        int temp[1];
        double loc_arr[N_BEH];
        copy(loc.begin(), loc.end(), loc_arr);
        find_closest(N_BEH, N_GRID, 1, loc_arr, cvt_arr.data(), temp);
        return temp[0];
    };
    cout << "CVT Complete." << endl;

#if FIELD_TESTING == false
    vector<FGenome> population(N_FP), next_population(N_FP);
    vector<tuple<double, double, HGenome, HGenome>> cur_costs(N_FP);
    for (int i = 0; i < N_FP; ++i) {
        FGenome epsilon;
        for (int j = 0; j < L; ++j) {
            epsilon[j].first = U1(gen) * (A_MAX - A_MIN) + A_MIN;
            epsilon[j].second = UT(gen);
        }
        for (int z : {6, 10, 11})
            epsilon[z].first = 0;
//        pair<FGenome, vector<double>> cur(epsilon, f_to_inter_t(epsilon));
        population[i] = epsilon;
    }
#endif

    int s = 0;
    vector<FGenome> best_fields, median_fields, min_field_val, max_field_val;
    vector<vector<double>> all_costs, all_dH;
    vector<HGenome> bst_minH, bst_maxH, med_minH, med_maxH, wst_minH, wst_maxH;
    FConstants constants(mu, C, analysis_population, psi_i, apx_bounds);
#if _SAVE_ALL_POP
    vector<vector<FGenome>> all_saves_pop;
    vector<vector<tuple<double, double, HGenome, HGenome>>> all_saves_costs;
#endif

    {
        ptime();
        cout << "Initializing MAP-Elites seeds with full runs, total of " << N_PRE_SEEDING << endl;

        int real_params_i[] = {ME_C0H_EXIT, max_runtime_ME, ME_NE_MAX, N_LINES};
        double real_params_d[] = {S_HM};
        ME_C0H_EXIT = N_GRID;
        N_LINES *= 2;
        max_runtime_ME *= .4;
//        ME_NE_MAX *= 2;
        for (int r = 0; r < N_PRE_SEEDING; ++r) {
            for (FGenome &f : population)
                get_f_cost(f, gen, constants, get_grid_idx);
            cout << "Finished pre-seeding run index " << r << endl;
            ptime();
        }
        ME_C0H_EXIT = real_params_i[0];
        max_runtime_ME = real_params_i[1];
        ME_NE_MAX = real_params_i[2];
        N_LINES = real_params_i[3];
        S_HM = real_params_d[0];
    }

    if (N_PRE_SEEDING > 0)
        write_seeds(filename + "_PRESEEDS.txt");

//#if _PARA_GGA
//#pragma omp parallel for default(shared)
//#endif
//    for (int i = GR_F; i < N_FP; ++i)
//        cur_costs[i] = get_f_cost(population[i], gen, constants, get_grid_idx);
    ptime();
    cout << "\n\nPre-calculations complete. Starting GA:\n" << endl;
    while (s < N_FGEN && time(nullptr) - main_start_time < max_runtime_GGA) {
#if _PARA_GGA
#pragma omp parallel for default(shared)
#endif
        for (int i = 0; i < N_FP; ++i)
            cur_costs[i] = get_f_cost(population[i], gen, constants, get_grid_idx);
#if FIELD_TESTING == false
        pairsort(cur_costs.data(), population.data(), N_FP, -1);
#endif
        double cost[N_FP];
        for (int i = 0; i < N_FP; ++i)
            cost[i] = get<0>(cur_costs[i]);

        for (int i = GR_F; i < N_FP; ++i)
            next_population[i] = population[i];

#if FIELD_TESTING == false
        for (int i = 0; i < GR_F; ++i) {
            FGenome child;

            // Crossover
            double gamma = U1(gen);
            int par1 = tournament_select_f(cost, N_FP, gen, U1);
            int par2 = tournament_select_f(cost, N_FP, gen, U1);
            if (gamma < P_FC) {
                for (int j = 0; j < L; ++j) {
                    double crv = U1(gen);
                    if (crv < B_FC)
                        child[j] = population[par1][j];
                    else
                        child[j] = population[par2][j];
                }
            } else
                child = population[par1];

            // Mutation
            for (int j = 0; j < L; ++j) {
                if (U1(gen) < P_FM)
                    child[j].first *= NA(gen);
                if (U1(gen) < P_FM)
                    child[j].second += NT(gen);
            }

            next_population[i] = child;
        }
        population = next_population;
#endif
        ptime();
        cout << endl << "Generation " << s << ", best cost " << cost[N_FP - 1] << ", med " << cost[N_FP / 2] << "\n\n";
        vector<double> costV(N_FP), dHV(N_FP);
        for (int i = 0; i < N_FP; ++i) {
            costV[i] = get<0>(cur_costs[i]);
            dHV[i] = get<1>(cur_costs[i]);
        }
        best_fields.push_back(population[N_FP - 1]);
        median_fields.push_back(population[N_FP / 2]);
        all_costs.push_back(costV);
        all_dH.push_back(dHV);
        bst_minH.push_back(get<2>(cur_costs[N_FP - 1]));
        bst_maxH.push_back(get<3>(cur_costs[N_FP - 1]));
        med_minH.push_back(get<2>(cur_costs[N_FP / 2]));
        med_maxH.push_back(get<3>(cur_costs[N_FP / 2]));
        wst_minH.push_back(get<2>(cur_costs[0]));
        wst_maxH.push_back(get<3>(cur_costs[0]));
#if _SAVE_ALL_POP
        all_saves_pop.push_back(population);
        all_saves_costs.push_back(cur_costs);
#endif
        FGenome min_field = population[0], max_field = population[0];
        for (FGenome& f : population) {
            for (int i = 0; i < L; ++i) {
                min_field[i].first = min(min_field[i].first, f[i].first);
                min_field[i].second = min(min_field[i].second, f[i].second);
                max_field[i].first = max(max_field[i].first, f[i].first);
                max_field[i].second = max(max_field[i].second, f[i].second);
            }
        }
        min_field_val.push_back(min_field);
        max_field_val.push_back(max_field);

#if UPDATE_ME_MUTATION_RATE
        cout << "Updated mutation parameter from " << S_HM;
        S_HM *= (1 + dyn_mut_para * cur_gen_mut);
        cout << " to " << S_HM << endl;
#endif

        ++s;

        ptime();
        cout << "Saving heartbeat..." << endl;
        time_t then;
        time(&then);
        {
#ifdef HAS_FILESYSTEM
            try {
                filesystem::copy(filename + "_HEARTBEAT.txt", filename + "_HEARTBEAT_OLD.txt",
                                 filesystem::copy_options::overwrite_existing);
            } catch (filesystem::filesystem_error& e) {
                if (s > 1) {
                    cout << "Heartbeat copy error:\n" << e.what() << endl;
                    cerr << "Heartbeat copy error:\n" << e.what() << endl;
                }
            }
#endif
            ofstream outfile(filename + "_HEARTBEAT.txt", ofstream::trunc);
            cout << "Filename is " << filename << "_HEARTBEAT.txt" << endl;

            int out_ints[] = {DIM, N_T, N_LINES, ME_NE_MAX, ME_BS, ME_C0H_EXIT, N_FP, N_FGEN, N_FTOURN, -1, GR_F,
                              main_start_time, max_runtime_ME, max_runtime_GGA, seed, N_LBINARY, N_LSAMPLES, L, s,
                              (int) global_seeds.size(), N_PRE_SEEDING, N_GRID, N_TO, N_OBS, N_PFA, max_seeds_ME, -1, -1,
                              -1, -1, -1, -1, -1, -1, -1};
            double out_doubles[] = {T, h_guess_err, -1, P_HC, P_HM, S_HM, S_FA, S_FT, A_MAX, P_FC, P_FM, BETA_FCOST,
                                    -1, HBAR, L_EXIT, L_BIAS, obs_err, amp_err, dyn_mut_para, dyn_mut_target,
                                    init_S_HM, B_FC, A_MIN, cost_multiplier, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

            if (message.empty())
                message = "#";
            outfile << "MEG3 " << 4 << ' ' << message;
            for (int o : out_ints)
                outfile << ' ' << o;
            for (double o : out_doubles)
                outfile << ' ' << o;
            outfile << ' ' << then - main_start_time << endl;

            outfile << "Preliminaries:" << endl;

            outfile << H0D.real().transpose() << endl;

            HGenome mugen = mat_to_gen(mu);
            for (double m : mugen)
                outfile << m << ' ';
            outfile << endl;

            outfile << psi_i.real().transpose() << endl;

            for (double w : omega)
                outfile << w << ' ';
            outfile << endl;

            outfile << "Fields:" << endl;

            print_arr(median_fields, outfile, [](pair<double, double> p){return p.first;});
            print_arr(median_fields, outfile, [](pair<double, double> p){return p.second;});
            print_arr(best_fields, outfile, [](pair<double, double> p){return p.first;});
            print_arr(best_fields, outfile, [](pair<double, double> p){return p.second;});
            print_arr(min_field_val, outfile, [](pair<double, double> p){return p.first;});
            print_arr(min_field_val, outfile, [](pair<double, double> p){return p.second;});
            print_arr(max_field_val, outfile, [](pair<double, double> p){return p.first;});
            print_arr(max_field_val, outfile, [](pair<double, double> p){return p.second;});

            outfile << "Arrays:" << endl;

            print_vec(all_costs, outfile, [](double x){return x;});
            print_vec(all_dH   , outfile, [](double x){return x;});
            print_arr(bst_minH, outfile, [](double x){return x;});
            print_arr(bst_maxH, outfile, [](double x){return x;});
            print_arr(med_minH, outfile, [](double x){return x;});
            print_arr(med_maxH, outfile, [](double x){return x;});
            print_arr(wst_minH, outfile, [](double x){return x;});
            print_arr(wst_maxH, outfile, [](double x){return x;});

            outfile << "Envelope:" << endl;

            for (int i = 0; i < N_T; ++i)
                outfile << envelope_funct((i + 0.5) * DELTA_T) << ' ';
            outfile << endl;

            cout << "Computing population graphs..." << endl;
            outfile << "Population graphs:" << endl;
            auto diag_ret = diag_vec(mu, C);
            EVector lambda = diag_ret.second;
            EMatrix CP = diag_ret.first.first;
            EMatrix PdC = diag_ret.first.second;
            vector<FGenome> *b_fields[] = {&median_fields, &best_fields};
            for (vector<FGenome>* farr : b_fields) {
                for (auto &field : *farr) {
                    auto t_field = f_to_inter_t(field);
                    vector<DArr> pg = gen_pop_graphs(t_field, CP, PdC, lambda, psi_i, analysis_population);
                    for (int i = 0; i < DIM; ++i) {
                        for (DArr &a : pg)
                            outfile << a[i] << ' ';
                        outfile << endl;
                    }
                }
            }

            outfile << "Final population:" << endl;
            print_arr(population, outfile, [](pair<double, double> p){return p.first;});
            print_arr(population, outfile, [](pair<double, double> p){return p.second;});

            outfile << "Oracle returns:" << endl;

            for (auto& t : oracle_rets) {
                for (double d : get<0>(t))
                    outfile << d << ' ';
                outfile << endl;
            }

            outfile << "Oracle costs:" << endl;

            for (auto& t : oracle_rets) {
                for (double d : get<1>(t))
                    outfile << d << ' ';
                outfile << endl;
            }

            outfile << "Oracle 's' values:" << endl;

            for (auto& t : oracle_rets)
                outfile << get<2>(t) << ' ';
            outfile << endl;

            if (!outfile.good())
                cerr << "Writing failed." << endl;

            outfile.close();
        }


        ifstream stop_file("stop_now");
        if (stop_file.good())
            break;
    }



//#pragma omp parallel for default(shared)

//    auto noisy_obs = get_noisy_observables(eps_inter, C, mu, gen, analysis_population, psi_i);
//
//    function<double(const OArr&)> get_g_cost = [&](const OArr& sim_obs) -> double {
//        double cost = 0;
//        for (int i = 0; i < N_OBS; ++i) {
//            double c = sim_obs[i], l = noisy_obs.first[i], e = noisy_obs.second[i];
//            if (abs(c - l) < e)
//                continue;
//            double t = (c - l) / l;
//            cost += t * t;
//        }
//        return cost;
//    };
//
//
//    pair<HGenome, HGenome> inverse_bounds = run_inversion(apx_bounds, eps_inter, gen, C, psi_i, noisy_obs.first,
//                                                          noisy_obs.second, analysis_population, get_grid_idx);

    ptime();
    cout << "Done! Moving computed data to _DATA file and saving EOR seeds..." << endl;
    time_t then;
    time(&then);
#ifdef HAS_FILESYSTEM
    try {
        filesystem::copy(filename + "_HEARTBEAT.txt", filename + "_DATA.txt");
        filesystem::remove(filename + "_HEARTBEAT.txt");
    } catch (filesystem::filesystem_error& e) {
        cout << "Error in _DATA copying:\n" << e.what() << endl;
        cerr << "Error in _DATA copying:\n" << e.what() << endl;
    }
#endif
    write_seeds(filename + "_EOR.txt");


#if _SAVE_ALL_POP
    {
        ptime();
        cout << "Saving full population file:" << endl;
        ofstream ffile(filename + "_FFD.txt");
        ffile << "FFD0 " << 0 << ' ' << message << ' ' << N_FP << ' ' << s << endl;
        vector<double> comparison_point = f_to_inter_t(all_saves_pop[0][0]);
        for (double d : comparison_point)
            ffile << d << ' ';
        ffile << endl;
        for (double d : mat_to_gen(mu))
            ffile << d << ' ';
        ffile << endl;
        for (auto& v : all_saves_pop)
            print_arr(v, ffile, [](pair<double, double> p){return p.first;});
        for (auto& v : all_saves_pop)
            print_arr(v, ffile, [](pair<double, double> p){return p.second;});
        for (auto &v : all_saves_costs) {
            for (auto &i : v) {
                for (auto &j : get<2>(i))
                    ffile << j << ' ';
                ffile << endl;
            }
        }
        for (auto &v : all_saves_costs) {
            for (auto &i : v) {
                for (auto &j : get<3>(i))
                    ffile << j << ' ';
                ffile << endl;
            }
        }
    }
#endif

    if (N_PFA > 0) {
        ptime();
        cout << "Running post-run sample field analyses:" << endl;
        ofstream hfile(filename + "_SFA.txt");
        hfile << "HEG2 " << 0 << ' ' << message << ' ' << N_PFA << ' ' << N_PFS << endl << endl;
        vector<FGenome> samples;
        if (N_PFS > 1) {
            for (int i = 0; i < N_PFS; ++i)
                samples.push_back(best_fields[(i * (best_fields.size() - 1)) / (N_PFS - 1)]);
            for (int i = 0; i < N_PFS; ++i)
                samples.push_back(median_fields[(i * (median_fields.size() - 1)) / (N_PFS - 1)]);
        } else {
            samples.push_back(best_fields[0]);
            samples.push_back(median_fields[0]);
        }
        cout << "Note that the first set of " << N_PFS << " fields are best, then the next set are median." << endl;
        hfile << "Note that the first set of " << N_PFS << " fields are best, then the next set are median." << endl;

        max_runtime_ME *= 2;
        dyn_mut_para = 0;

        int CMNM = ME_NE_MAX;
        ME_NE_MAX *= 2;
//        cout << "Running with current archive and S_HM = " << S_HM << ':' << endl;
//        hfile << "\n\nRunning with current archive and S_HM = " << S_HM << ':' << endl;

        S_HM = init_S_HM;
        cout << "Running with current archive and S_HM = " << S_HM << ':' << endl;
        hfile << "\nRunning with current archive and S_HM = " << S_HM << ':' << endl;
        for (int i = 0; i < samples.size(); ++i, gen.discard(i))
            write_sfa(get_f_cost(samples[i], gen, constants, get_grid_idx), hfile, i);

        ptime();
        cout << "Clearing archive..." << endl;

        for (int r = 0; r < N_PFA; ++r) {
            global_seeds.clear();
//            for (int ro = 0; ro < N_LINES; ++ro) {
//                if (N_LINES != 1)
//                    cout << "Running oracle " << ro << "..." << endl;
//                    //        throw runtime_error("Non-zero N_ORACLE not implemented.");
//                else
//                    cout << "Running oracle..." << endl;
//
//                for (int i = 0; i < N_H; ++i)
//                    guess_h[i] = normalize(U1(gen), apx_bounds.first[i], apx_bounds.second[i]);
//
//                Oracle oracle{gen, mat_to_gen(mu), psi_i, analysis_population[DIM - 1], omega, analysis_population, H0D, guess_h};
//                oracle_ptr = &oracle;
//
//                oracle_rets[ro] = oracle.run_oracle();
//                global_seeds.emplace(get<0>(oracle_rets[ro]));
//                ptime();
//            }

            N_LINES *= 2;

//            ME_NE_MAX = CMNM;
//            cout << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
//            hfile << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
//            for (int i = 0; i < N_PFS; ++i, gen.discard(i))
//                write_sfa(get_f_cost(samples[i], gen, constants, get_grid_idx), hfile, i);

            ME_NE_MAX = CMNM * 2;
            S_HM = init_S_HM / 2;
            cout << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
            hfile << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
            for (int i = 0; i < samples.size(); ++i, gen.discard(i))
                write_sfa(get_f_cost(samples[i], gen, constants, get_grid_idx), hfile, i);

//            ME_NE_MAX = CMNM;
//            cout << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
//            hfile << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
//            for (int i = 0; i < N_PFS; ++i, gen.discard(i))
//                write_sfa(get_f_cost(samples[i], gen, constants, get_grid_idx), hfile, i);

            ME_NE_MAX = CMNM;
            S_HM = init_S_HM * 2;
            cout << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
            hfile << "\n\nr =" << r << "; " << ME_NE_MAX << "; " << S_HM << endl;
            for (int i = 0; i < samples.size(); ++i, gen.discard(i))
                write_sfa(get_f_cost(samples[i], gen, constants, get_grid_idx), hfile, i);

            ME_NE_MAX = CMNM;
            S_HM = init_S_HM;
        }
    }

    ptime();
    cout << "Exiting..." << endl;

    return 0;
}

double envelope_funct(double t) {
    static_assert(N_TO == 2, "The current envelope function is a double bell curve...\n");
    return exp(-30 * (2 * t / T - .5) * (2 * t / T - .5)) + exp(-30 * ((2 * t - T) / T - .5) * ((2 * t - T) / T - .5));
}


pair<pair<EMatrix, EMatrix>, EVector> diag_vec(const EMatrix& mu, const EDMatrix& C) {
    ComplexSchur<EMatrix> schur;
    schur.compute(mu, true);
    if (schur.info() != Success) {
        cout << "Schur computation failed." << endl;
        exit(1);
    }

//    cout << "C: " << endl << C.toDenseMatrix() << endl;

    EVector lambda;
    lambda = schur.matrixT().diagonal();
    EMatrix P;
    P = schur.matrixU();
//    cout << "Lambda diagonal: " << endl << lambda.transpose() << endl;
//    cout << "P: " << endl << P << endl;
//    cout << "P Lambda P^d:" << endl << P * lambda.asDiagonal() * P.adjoint() << endl;

    EMatrix CP, PdC;
    CP = C * P;
    PdC = P.adjoint() * C;
    return {{CP, PdC}, lambda};
}

OArr evolve_initial(const vector<double>& epsilon, const EMatrix& CP, const EMatrix& PdC, const EMatrix& PdCCP,
                    const EVector& lambda, const EVector& psi_i, const array<ECovector, DIM>& anal_pop) {
    vector<EDMatrix, aligned_allocator<EDMatrix>> E(N_T);
    for (int i = 0; i < N_T; ++i)
        E[i] = exp(1i * DELTA_T / HBAR * lambda.array() * epsilon[i]).matrix().asDiagonal();

    vector<EVector, aligned_allocator<EVector>> it(N_T + 1);
    it[0] = PdC * psi_i;
    for (int i = 1; i < N_T; ++i)
        it[i] = PdCCP * (E[i - 1] * it[i - 1]);
    it[N_T] = CP * (E[N_T - 1] * it[N_T - 1]);

    OArr samples{};
    for (int i = 0; i < N_TO; ++i)
        for (int j = 0; j < DIM; ++j)
            samples[i * DIM + j] = (anal_pop[j] * it[(i + 1) * N_T / N_TO]).squaredNorm();
    return samples;
}

double normalize(double rand, double min, double max) {
    return rand * (max - min) + min;
}

complex<double> get_only_element(Matrix<complex<double>, -1, -1> scalar) {
    if (scalar.rows() > 1 || scalar.cols() > 1) {
        cout << scalar << endl;
        throw runtime_error("Tried to get single element from matrix, see cout for matrix");
    }
    return scalar(0, 0);
}

pair<int, int> calc_loc(int u_i) { // I could binary search this but I'm too lazy
    for (int i = DIM - 1; i > 0; --i)
        if ((i * (i - 1)) / 2 <= u_i)
            return {i, u_i - ((i * (i - 1)) / 2)};
    throw runtime_error("calc_loc failed");
}

vector<DArr> gen_pop_graphs(const vector<double>& eps_inter, const EMatrix& CP, const EMatrix& PdC,
                            const EVector& lambda, const EVector& psi_i, const array<ECovector, DIM>& anal_pop) {
    vector<EMatrix, aligned_allocator<EMatrix>> E(N_T);
    for (int i = 0; i < N_T; ++i)
        E[i] = exp(1i * DELTA_T / HBAR * lambda.array() * eps_inter[i]).matrix().asDiagonal();

    vector<EVector, aligned_allocator<EVector>> it(N_T + 1);
    it[0] = psi_i;
    for (int i = 1; i <= N_T; ++i)
        it[i] = CP * (E[i - 1] * (PdC * it[i - 1]));

    vector<DArr> pops(N_T + 1);
    for (int i = 0; i <= N_T; ++i)
        for (int o = 0; o < DIM; ++o)
            pops[i][o] = (anal_pop[o] * it[i]).squaredNorm();
    return pops;
}


vector<double> f_to_inter_t(const FGenome& epsilon) {
    vector<double> eps_inter(N_T);
    for (int i = 0; i < N_T; ++i) {
        eps_inter[i] = 0;
        double t_i = (i + 0.5) * DELTA_T;
        int c = 0;
        for (int l = 1; l < DIM; ++l) {
            for (int m = 0; m < l; ++m) {
                eps_inter[i] += epsilon[c].first * sin(omega[c] * t_i + epsilon[c].second);
                ++c;
            }
        }
        eps_inter[i] *= envelope_funct(t_i);
    }
    return eps_inter;
}

tuple<double, double, HGenome, HGenome> get_f_cost(const FGenome& epsilon, rng& gen, FConstants& constants,
                                                   function<int(const BArr&)>& get_grid_idx) {
//    EMatrix& mu = get<0>(constants);
//    EDMatrix& C = get<1>(constants);
//    array<ECovector, DIM>& anal_pop = get<2>(constants);
//    EVector& psi_i = get<3>(constants);
//    pair<HGenome, HGenome>& apx_bounds = get<4>(constants);
    vector<double> eps_inter = f_to_inter_t(epsilon);

    auto noisy_obs = get_noisy_observables(epsilon, gen, constants);

    function<double(const OArr&)> get_g_cost = [&](const OArr& sim_obs) -> double {
        double cost = 0;
        for (int i = 0; i < N_OBS; ++i) {
            double c = sim_obs[i], l = noisy_obs.first[i], e = noisy_obs.second[i];
            if (abs(c - l) < e)
                continue;
            double t = (c - l) / l;
            cost += t * t;
        }
        return cost;
    };

    pair<HGenome, HGenome> inverse_bounds = run_illumination(constants, eps_inter, noisy_obs.first,
                                                             noisy_obs.second, gen, get_grid_idx);
    HGenome& minH = inverse_bounds.first;
    HGenome& maxH = inverse_bounds.second;

    double flu = 0;
    for (double d : eps_inter)
        flu += d * d;
    flu /= N_T;

    HGenome true_vals = mu_true;

    double dH = 0;
    for (int i = 0; i < N_H; ++i)
        if (maxH[i] + minH[i] != 0)
            dH += abs((maxH[i] - minH[i]) / (2 * true_vals[i]));

    cout << "dH equals " << dH << endl << endl;
    return {cost_multiplier * (dH + BETA_FCOST * flu), dH, minH, maxH};
}

void pairsort(tuple<double, double, HGenome, HGenome> *a, FGenome *b, int n, double p) {
    pair<tuple<double, double, HGenome, HGenome>, FGenome> pairs[n];

    // Storing the respective array elements in pairs.
    for (int i = 0; i < n; ++i)
    {
        pairs[i].first = a[i];
        get<0>(pairs[i].first) *= p;
        pairs[i].second = b[i];
    }

    // Sorting the pair array.
    sort(pairs, pairs + n,
         [](pair<tuple<double, double, HGenome, HGenome>, FGenome>& a,
                 pair<tuple<double, double, HGenome, HGenome>, FGenome>& b) {
        return get<0>(a.first) < get<0>(b.first);
    });

    // Modifying original arrays
    for (int i = 0; i < n; ++i)
    {
        get<0>(pairs[i].first) /= p;
        a[i] = pairs[i].first;
        b[i] = pairs[i].second;
    }
}

int tournament_select_f(const double* costs, int n, rng& gen, uniform_real_distribution<> U1) {
    int best = -1;
    double best_cost = 1e20;
    for (int i = 0; i < N_FTOURN; ++i) {
        int l = (int) (U1(gen) * n);
        if (costs[l] < best_cost) {
            best = l;
            best_cost = costs[l];
        }
    }
    return best;
}

template <class T, class F> void print_vec(vector<vector<T>> vec, ofstream& outfile, F lambda) {
    for (vector<T> i : vec) {
        for (T j : i)
            outfile << lambda(j) << ' ';
        outfile << endl;
    }
}

template <class T, class F, size_t N> void print_arr(vector<array<T, N>> vec, ofstream& outfile, F lambda) {
    for (array<T, N> i : vec) {
        for (T j : i)
            outfile << lambda(j) << ' ';
        outfile << endl;
    }
}

void ptime() {
    time_t t = time(nullptr);
    cout << "Unix time " << (long) t << ", runtime " << (long) (t - main_start_time) << ", date " << ctime(&t);
}

void write_sfa(tuple<double, double, HGenome, HGenome> cost, ofstream& file, int n) {
    file << "Field " << n << ':' << endl;
    file << "Cost: " << get<1>(cost) << endl;
    file << "Min:\n" << gen_to_mat(get<2>(cost)).real() << endl;
    file << "Max:\n" << gen_to_mat(get<3>(cost)).real() << endl << endl;
}

void write_seeds(string filename) {
    ofstream preseeding_file(filename);
    preseeding_file << setprecision(numeric_limits<long double>::digits10 + 1);
    preseeding_file << global_seeds.size() << endl;
    for (auto& cs : global_seeds) {
        for (double d : cs)
            preseeding_file << d << ' ';
        preseeding_file << endl;
    }
    if (preseeding_file.good())
        cout << "Pre-seeds saved successfully to " << filename << endl;
    else
        cout << "Unknown error saving pre-seeds to " << filename << endl;
    preseeding_file.close();
}
