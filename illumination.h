//
// Created by erezm on 2021-01-31.
//

#ifndef ME_INVERSION_INVERSION_H
#define ME_INVERSION_INVERSION_H

#include "main.h"

#define UPDATE_ME_MUTATION_RATE true

const int N_H = (DIM * (DIM - 1)) / 2;
const int N_BEH = N_OBS;
const int N_GRID = 10000;

typedef array<double, N_H> HGenome;
typedef array<double, N_BEH> BArr;
typedef tuple<double, HGenome, BArr> MEPop;
// this typedef should be in main
typedef tuple<EMatrix&, EDMatrix&, array<ECovector, DIM>&, EVector&, pair<HGenome, HGenome>&> FConstants;

pair<HGenome, HGenome> run_illumination(FConstants& constants, const vector<double>& eps_inter, const OArr& obs_avg,
                                        const OArr& obs_err, rng& gen, function<int(const BArr&)>& get_grid_idx);

pair<HGenome, HGenome> invert_ME(const pair<HGenome, HGenome>& apx_bounds, function<OArr(const HGenome&)>& get_obs,
                                 function<BArr(const OArr&)>& get_beh, function<int(const BArr&)>& get_grid_idx,
                                 function<double(const OArr&, const HGenome&)>& get_cost, rng& gen);

HGenome mat_to_gen(const EMatrix& matrix);
EMatrix gen_to_mat(const HGenome& genome);

int tournament_select(const array<MEPop, N_GRID>& archive, const vector<int>& full_locs, rng& gen, uniform_real_distribution<> U1);

int curiosity_select(const vector<int> &full_locs, const vector<int> &curiosity_index,
                     const vector<bool> &archive_has, double rand);

pair<int, int> normalize_curiosity(vector<int> &curiosity_index);

// these methods should be in main
tuple<double, double, HGenome, HGenome> get_f_cost(const FGenome& epsilon, rng& gen, FConstants& constants,
                                                   function<int(const BArr&)>& get_grid_idx);

void pairsort(tuple<double, double, HGenome, HGenome> *a, FGenome *b, int n, double p);

void write_sfa(tuple<double, double, HGenome, HGenome> cost, ofstream& file, int n);

HGenome bias_HG(HGenome& t, HGenome& o, double bias);

double sqdist_HG(HGenome& a, HGenome& b);

#endif //ME_ILLUMINATION_H
