//
// Created by erezm on 2020-12-12.
//

#ifndef ME_EX_MAIN_H
#define ME_EX_MAIN_H

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <ctime>
#include <complex>
#include <functional>
#include <atomic>
#include <chrono>
#include <unordered_set>
#include <iomanip>
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "omp.h"

#if !(defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64))
#include <filesystem>
#define HAS_FILESYSTEM 1
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define path "C:\\EZKZ\\ME-G1\\"
#elif defined(linux) || defined(_linux_) || defined(unix) || defined(_unix_) || defined(__unix__)
#define path "./"
#endif

using namespace std;
using namespace Eigen;

const int DIM = 6;
const int L = (DIM * (DIM - 1)) / 2;
const double HBAR = 1;
const int N_TO = 2;
const int N_OBS = DIM * N_TO;

typedef Matrix<complex<double>, DIM, DIM> EMatrix;
typedef DiagonalMatrix<complex<double>, DIM> EDMatrix;
typedef Matrix<complex<double>, DIM, 1> EVector;
typedef Matrix<complex<double>, 1, DIM> ECovector;
typedef Matrix<complex<double>, Dynamic, 1> TVector;
typedef Matrix<double, Dynamic, 1> RTVector;
typedef array<double, N_OBS> OArr;
typedef array<double, DIM> DArr;
typedef array<pair<double, double>, L> FGenome;
typedef mt19937 rng;

int main(int argc, char** argv);

double envelope_funct(double t);
double envelope_funct_OR(double t);

pair<pair<EMatrix, EMatrix>, EVector> diag_vec(const EMatrix& mu, const EDMatrix& C);

OArr evolve_initial(const vector<double>& epsilon, const EMatrix& CP, const EMatrix& PdC, const EMatrix& PdCCP,
                    const EVector& lambda, const EVector& psi_i, const array<ECovector, DIM>& anal_pop);

double normalize(double rand, double min, double max);

complex<double> get_only_element(Matrix<complex<double>, -1, -1> scalar);

pair<int, int> calc_loc(int u_i);

vector<DArr> gen_pop_graphs(const vector<double>& eps_inter, const EMatrix& CP, const EMatrix& PdC,
                            const EVector& lambda, const EVector& psi_i, const array<ECovector, DIM>& anal_pop);

vector<double> f_to_inter_t(const FGenome& epsilon);

int tournament_select_f(const double* costs, int n, rng& gen, uniform_real_distribution<> U1);

template <class T, class F> void print_vec(vector<vector<T>> vec, ofstream& outfile, F lambda);
template <class T, class F, size_t N> void print_arr(vector<array<T, N>> vec, ofstream& outfile, F lambda);

void write_seeds(string filename);

void ptime();

// Some of the methods are actually in illumination.h because they need illumination typedefs
// and if I try to include illumination.h clion breaks...

// Here's a hash function for arrays:
template<typename T, size_t N>
struct hash<array<T, N>>
{
    typedef array<T, N> argument_type;
    typedef size_t result_type;

    result_type operator()(const argument_type& a) const
    {
        hash<T> hasher;
        result_type h = 0;
        for (result_type i = 0; i < N; ++i)
        {
            h = h * 31 + hasher(a[i]);
        }
        return h;
    }
};

#endif
