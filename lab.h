//
// Created by erezm on 2021-01-31.
//

#ifndef ME_INVERSION_LAB_H
#define ME_INVERSION_LAB_H

#include "main.h"
#include "illumination.h"

extern double obs_err, amp_err;
const int N_REP = 100;

pair<OArr, OArr> get_noisy_observables(const FGenome& epsilon, rng gen, FConstants& constants);

#endif //ME_INVERSION_LAB_H
