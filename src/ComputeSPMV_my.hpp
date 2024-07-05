
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTESPMV_MY_HPP
#define COMPUTESPMV_MY_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"

int ComputeSPMV_my( const SparseMatrix & A, Vector  & x, Vector & y);

#endif  // COMPUTESPMV_REF_HPP
