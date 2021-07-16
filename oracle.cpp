//
// Created by erezm on 2021-02-13.
//

#include "oracle.h"

extern int OR_NT;
extern double OR_T, OR_DT;
extern int seed, main_start_time;
extern double DELTA_S, endpoint_O;
extern int max_gen_O;

#define _LOG_ORACLE true
#define _SAVE_ORACLE false
#define _ORACLE_EXIT_BOUNDARY true
#define _PARALLEL_ORACLE true
class Oracle {
public:
    Oracle(rng& gen, const HGenome &mu_t, const EVector &psi_i, const ECovector &psi_f, const array<double, L> &omega,
           const array<ECovector, DIM> &anal_pop, const EVector &H0D, const HGenome &guess_h) :
           gen(gen), mu_t(mu_t), psi_i(psi_i), psi_f(psi_f), omega(omega), anal_pop(anal_pop), guess_h(guess_h) {
#if _LOG_ORACLE
        cout << "Oracle: creating fields..." << endl;
#endif
        C = exp(H0D.array() * -1i * OR_DT / 2 / HBAR).matrix().asDiagonal();
        create_fields();
#if _LOG_ORACLE
        cout << "Oracle: calculating observables..." << endl;
#endif
        true_obs = create_obs(mu_t);
#if _LOG_ORACLE
        cout << "Oracle created successfully." << endl;
#endif
    }

    void create_fields() {
        array<array<pair<double, double>, L>, N_OF> epsilons;
        for (int i = 0; i < N_OF; ++i) {
            for (int j = 0; j < L; ++j) {
                epsilons[i][j].first = U1(gen);
                epsilons[i][j].second = U1(gen) * 2 * M_PI;
            }
        }
#if _PARALLEL_ORACLE
#pragma omp parallel for default(shared)
#endif
        for (int field = 0; field < N_OF; ++field) {
            vector<double> eps_inter(OR_NT), eps_point(OR_NT + 1);
            for (int i = 0; i < OR_NT; ++i) {
                eps_inter[i] = 0;
                double t_i = (i + 0.5) * OR_DT;
                int c = 0;
                for (int l = 1; l < DIM; ++l) {
                    for (int m = 0; m < l; ++m) {
                        eps_inter[i] += epsilons[field][c].first * sin(omega[c] * t_i + epsilons[field][c].second);
                        ++c;
                    }
                }
                eps_inter[i] *= envelope_funct_OR(t_i);
            }
            eps_inters[field] = eps_inter;
            for (int i = 0; i <= OR_NT; ++i) {
                eps_point[i] = 0;
                double t_i = i * OR_DT;
                int c = 0;
                for (int l = 1; l < DIM; ++l) {
                    for (int m = 0; m < l; ++m) {
                        eps_point[i] += epsilons[field][c].first * sin(omega[c] * t_i + epsilons[field][c].second);
                        ++c;
                    }
                    eps_point[i] *= envelope_funct_OR(t_i);
                }
            }
            eps_points[field] = eps_point;
        }
    }

    array<double, N_OF> create_obs(const HGenome &mu) {
        auto diag = diag_vec(gen_to_mat(mu), C);
        EVector lambda = diag.second;
        EMatrix CP = diag.first.first, PdC = diag.first.second;

        array<double, N_OF> obs{};
#if _PARALLEL_ORACLE
#pragma omp parallel for default(shared)
#endif
        for (int field = 0; field < N_OF; ++field) {
            vector<EMatrix, aligned_allocator<EMatrix>> E(OR_NT);
            for (int i = 0; i < OR_NT; ++i)
                E[i] = exp(1i * OR_DT / HBAR * lambda.array() * eps_inters[field][i]).matrix().asDiagonal();

            vector<EVector, aligned_allocator<EVector>> it(OR_NT + 1);
            it[0] = psi_i;
            for (int i = 1; i <= OR_NT; ++i)
                it[i] = CP * (E[i - 1] * (PdC * it[i - 1]));

            obs[field] = (psi_f * it[OR_NT]).squaredNorm();
        }
        return obs;
    }

    tuple<HGenome, vector<double>, double> run_oracle() {
//        time_t start_time = time(nullptr);

        vector<double> costs;
#if _SAVE_ORACLE
        vector<HGenome> gens;
        gens.push_back(guess_h);
        vector<array<double, N_OF>> obses;
#endif
        HGenome h = guess_h, ph;
        double s = 0;
        bool exit = false;
        int gen_g = 0;
        array<vector<EMatrix, aligned_allocator<EMatrix>>, N_OF> Es;
        array<vector<EVector, aligned_allocator<EVector>>, N_OF> evo_fields;
        array<vector<ECovector, aligned_allocator<ECovector>>, N_OF> qs;
        array<double, N_OF> cur_obs{};
        array<HGenome, 4> grads{};
        while (!exit) {
            ++gen_g;
            ph = h;
            for (int k = 0; k < 3; ++k) {
                auto diag = diag_vec(gen_to_mat(h), C);
                EVector lambda = diag.second;
                EMatrix CP = diag.first.first, PdC = diag.first.second;

                cur_obs = create_obs(h);

#if _PARALLEL_ORACLE
#pragma omp parallel for default(shared)
#endif
                for (int field = 0; field < N_OF; ++field) {
                    vector<EMatrix, aligned_allocator<EMatrix>> E(OR_NT);
                    for (int i = 0; i < OR_NT; ++i)
                        E[i] = exp(1i * OR_DT / HBAR * lambda.array() * eps_inters[field][i]).matrix().asDiagonal();
                    Es[field] = E;
                }


#if _PARALLEL_ORACLE
#pragma omp parallel for default(shared)
#endif
                for (int field = 0; field < N_OF; ++field) {
                    vector<EVector, aligned_allocator<EVector>> it(OR_NT + 1);
                    it[0] = psi_i;
                    for (int i = 1; i <= OR_NT; ++i)
                        it[i] = CP * (Es[field][i - 1] * (PdC * it[i - 1]));
                    evo_fields[field] = it;
                }

#if _PARALLEL_ORACLE
#pragma omp parallel for default(shared)
#endif
                for (int field = 0; field < N_OF; ++field) {
                    vector<ECovector, aligned_allocator<ECovector>> ft(OR_NT + 1); // UT[i] = U(T, t_i)
                    ft[OR_NT] = psi_f;
                    for (int i = OR_NT - 1; i >= 0; --i)
                        ft[i] = ((ft[i + 1] * CP) * Es[field][i]) * PdC;

                    complex<double> qp1 = get_only_element(psi_i.adjoint() * ft[0].adjoint());
                    vector<ECovector, aligned_allocator<ECovector>> qt(OR_NT + 1); // UT[i] = U(T, t_i)
                    for (int i = 0; i <= OR_NT; ++i)
                        qt[i] = qp1 * ft[i];
                    qs[field] = qt;
                }

#if _PARALLEL_ORACLE
#pragma omp parallel for default(shared)
#endif
                for (int u_i = 0; u_i < N_H; ++u_i) {
                    array<double, N_OF> dpif{};
                    vector<complex<double>> intvals(OR_NT + 1);
                    auto ij = calc_loc(u_i);
                    grads[k][u_i] = 0;
                    for (int i = 0; i < N_OF; ++i) {
                        for (int j = 0; j <= OR_NT; ++j)
                            intvals[j] = eps_points[i][j] * (qs[i][j](ij.first) * evo_fields[i][j](ij.second)
                                                             + qs[i][j](ij.second) * evo_fields[i][j](ij.first));
                        dpif[i] = -2 / HBAR * get_integral(intvals, OR_NT).imag();
                    }
                    for (int i = 0; i < N_OF; ++i)
                        grads[k][u_i] += -1 * (true_obs[i] - cur_obs[i]) * dpif[i];
                }
                if (k == 0) {
                    for (int i = 0; i < N_H; ++i)
                        h[i] = ph[i] - DELTA_S / 2 * grads[0][i];
                } else if (k == 1) {
                    for (int i = 0; i < N_H; ++i)
                        h[i] = ph[i] - DELTA_S / 2 * grads[1][i];
                } else if (k == 2) {
                    for (int i = 0; i < N_H; ++i)
                        h[i] = ph[i] - DELTA_S * grads[2][i];
                } else if (k == 3) {
                    for (int i = 0; i < N_H; ++i)
                        h[i] = ph[i] - DELTA_S / 6 * (grads[0][i] + 2 * grads[1][i] + 2 * grads[2][i] + grads[3][i]);
                }
            }
            s += DELTA_S;
#define SEED_ALL_ORACLE_ATTEMPTS true
#if SEED_ALL_ORACLE_ATTEMPTS
            extern unordered_set<HGenome> global_seeds;
            global_seeds.emplace(h);
#endif
#undef SEED_ALL_ORACLE_ATTEMPTS
#if _LOG_ORACLE
            double cost = 0;
            for (int field = 0; field < N_OF; ++field)
                cost += (true_obs[field] - cur_obs[field]) * (true_obs[field] - cur_obs[field]);
            exit = gen_g > max_gen_O || cost < endpoint_O;
            if (gen_g % 50 == 0) {
                printf("s: %.5f,  gen: %5d,  cost: %2.10f\n", s, gen_g, cost);
                if (gen_g % 5 == 0) {
                    printf("Genome: ");
                    for (int i = 0; i < N_H; ++i)
                        printf("%.5f  ", h[i]);
                    printf("\n");
                    printf("Grad: ");
                    for (int i = 0; i < N_H; ++i)
                        printf("%.7f  ", grads[0][i] + 2 * grads[1][i] + 2 * grads[2][i] + grads[3][i]);
                    printf("\n\n");
                }
            }
#endif
            costs.push_back(cost);
#if _SAVE_ORACLE
            gens.push_back(h);
            obses.push_back(cur_obs);
#endif
        }

#if _SAVE_ORACLE
#if _LOG_ORACLE
        cout << "Finished oracle. Saving..." << endl;
#endif
        ofstream outfile(string(path) + "MEI1DATA" + to_string(start_time) + "ORACLE.txt");
#define ns << ' ' <<
        outfile << "MEIDMF" ns 4 ns "gradientrk" ns ORACLE_NT ns T ns DELTA_S ns N_OF ns seed ns main_start_time << endl;
#undef ns
        for (double d : true_obs)
            outfile << d << ' ';
        outfile << endl;
        for (double d : mu_t)
            outfile << d << ' ';
        outfile << endl;
        for (double d : costs)
            outfile << d << ' ';
        outfile << endl;
        for (auto v : gens) {
            for (double d : v)
                outfile << d << ' ';
            outfile << endl;
        }
        for (auto v : obses) {
            for (double d : v)
                outfile << d << ' ';
            outfile << endl;
        }
        for (double d : eps_points[0])
            outfile << d << ' ';
        outfile << endl;

        vector<DArr> guess_pop = gen_pop_graphs(eps_inters[0], C, gen_to_mat(guess_h), psi_i, anal_pop),
                     final_pop = gen_pop_graphs(eps_inters[0], C, gen_to_mat(h), psi_i, anal_pop),
                     true_pop = gen_pop_graphs(eps_inters[0], C, gen_to_mat(mu_t), psi_i, anal_pop);
        for (int o = 0; o < DIM; ++o) {
            for (auto a : true_pop)
                outfile << a[o] << ' ';
            outfile << endl;
        }
        for (int o = 0; o < DIM; ++o) {
            for (auto a : guess_pop)
                outfile << a[o] << ' ';
            outfile << endl;
        }
        for (int o = 0; o < DIM; ++o) {
            for (auto a : final_pop)
                outfile << a[o] << ' ';
            outfile << endl;
        }
#endif
        return {h, costs, s};
    }

private:
    static complex<double> get_integral (vector<complex<double>> &vals, int i_f) {
        complex<double> sum = 0;
        for (int i = 1; i < i_f; ++i) {
            sum += vals[i];
        }
        sum += (vals[0] + vals[i_f]) / 2.0;
        sum *= OR_DT;
        return sum;
    }

    rng &gen;
    array<double, N_OF> true_obs;
    array<vector<double>, N_OF> eps_inters, eps_points;
    uniform_real_distribution<> U1{0, 1};
    const HGenome &mu_t;
    const EVector &psi_i;
    const ECovector &psi_f;
    const array<double, L> &omega;
    const array<ECovector, DIM> &anal_pop;
    EDMatrix C;
    const HGenome &guess_h;
};