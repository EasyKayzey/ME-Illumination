//
// Created by erezm on 2021-01-31.
//

#include "main.h"
#include "lab.h"
#include "illumination.h"
#include "cvt.h"


//extern int N_T;
//extern double T, DELTA_T;
// extern vector<HGenome> global_seeds;

// int maina(int argc, char** argv) {

//     vector<double> cvt_arr(N_GRID * N_BEH);
//     int cvt_seed = 8008315, cvt_iter_out;
//     double cvt_diff, cvt_energy;
//     cvt(N_BEH, N_GRID, 1000, 0, 0, 10000, 1000, 50, &cvt_seed, cvt_arr.data(), &cvt_iter_out, &cvt_diff, &cvt_energy);

//     function<vector<int>(const vector<BArr>&)> get_grid_idx = [&](const vector<BArr>& loc) {
//         vector<int> vals(loc.size());
//         find_closest(N_BEH, N_GRID, loc.size(), (double*) loc.data(), cvt_arr.data(), vals.data());
//         return vals;
//     };

//     cout << "CVT done" << endl;
//     rng r;
//     uniform_real_distribution<> U{0, 1};
//     int m = pow(3, N_OBS+3);
//     vector<BArr> ts(m);
//     array<int, N_GRID> vs{};
//     vs.fill(0);
//     for (auto& t : ts) {
//         for (double& d : t)
//             d = U(r);
//     }
//     vector<int> locs = get_grid_idx(ts);
//     for (int i : locs)
//         vs[i]++;

//     int mi = 100000000, ma = -1;
//     for (int i = 0; i < N_GRID; ++i) {
//         mi = min(vs[i], mi);
//         ma = max(vs[i], ma);
//         cout << i << ": " << vs[i] << endl;
//     }
//     cout << mi << " is min, max is " << ma << endl;

// }