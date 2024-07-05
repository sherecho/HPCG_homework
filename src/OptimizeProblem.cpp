
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

/*!
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
local_int_t binary_search(
  local_int_t * row_ptr,                      
  local_int_t num, 
  local_int_t end) 
{
  local_int_t l, r, h, t = 0;
  l = 0, r = end;
  while (l <= r) {
    h = (l + r) >> 1;
    if (row_ptr[h] >= num) {
      r = h - 1;
    } else {
      l = h + 1;
      t = h;
    }
  }
  return t;
}
 void albus_balance(local_int_t rows, 
                   local_int_t nonzeronums,
                   local_int_t * row_ptr, 
                   local_int_t* start,
                   local_int_t* end,
                   local_int_t* start1,
                   local_int_t* end1, double * mid_ans,
                   std::size_t thread_nums) {
  std::size_t tmp;
  start[0] = 0;
  start1[0] = 0;
  end[thread_nums - 1] = rows;
  end1[thread_nums - 1] = 0;
  std::size_t tt = nonzeronums / thread_nums;
  
  for (std::size_t i = 1; i < thread_nums; i++) {
    tmp = tt * i;
    start[i] = binary_search(row_ptr, tmp, rows);
    start1[i] = tmp - row_ptr[start[i]];
    end[i - 1] = start[i];
    end1[i - 1] = start1[i];
  } 
}

void subopt(const SparseMatrix & A, albusdata* data1){

   data1->row_ptr[0]=0;
   int cur_nnz=0;
   for (local_int_t i=0; i< A.localNumberOfRows; i++)  {
    cur_nnz+=A.nonzerosInRow[i];
    data1->row_ptr[i+1]=cur_nnz;
   }
   int m=0;
   for(int i=0;i<A.localNumberOfRows;i++){
       const double * const cur_vals = A.matrixValues[i];
       const local_int_t * const cur_inds = A.mtxIndL[i];
       const int cur_nnz = A.nonzerosInRow[i];
      for(int j=0;j<cur_nnz;j++){
        data1->value[m]=cur_vals[j];
        data1->col_index[m]=cur_inds[j];
        m++;
      }
   }
   albus_balance(A.localNumberOfRows, A.localNumberOfNonzeros, data1->row_ptr, data1->start,data1->end,data1->start1,data1->end1,data1->result_mid,data1->thread_nums);
}
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints
   
   //SPMVCSR * tmppointer=new SPMVCSR(A);  
    albusdata* data1=new albusdata;
    data1->row_ptr=new local_int_t[A.localNumberOfRows+1];
    data1->col_index=new local_int_t[A.localNumberOfNonzeros];
    data1->value=new alignas(64) double[A.localNumberOfNonzeros];
    data1->thread_nums= 12;//std::thread::hardware_concurrency();
    data1->start=new local_int_t[data1->thread_nums];
    data1->end=new local_int_t[data1->thread_nums];
    data1->start1=new local_int_t[data1->thread_nums];
    data1->end1=new local_int_t[data1->thread_nums];
    data1->result_mid=new alignas(64) double[data1->thread_nums*2];
    memset(data1->result_mid, 0.0, data1->thread_nums*2* sizeof(double));
    subopt( A,data1);
    A.optimizationData=(void* )data1;

#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  // form in-place prefix scan
  local_int_t old=counters[0], old0;
  for (local_int_t i=1; i < totalColors; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;

  // translate `colors' into a permutation
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
    colors[i] = counters[colors[i]]++;
#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
