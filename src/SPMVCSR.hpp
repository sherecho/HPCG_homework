#ifndef SPMVCSR_HPP
#define SPMVCSR_HPP
#include <vector>
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include <ostream>
#include "hpcg.hpp"
#include<cstring>
#include<thread>

#endif
struct albusstruct {
   local_int_t * row_ptr;
   local_int_t * col_index;
   double * value;
   local_int_t * start;
   local_int_t * start1;
   local_int_t * end;
   local_int_t * end1;
   std::size_t thread_nums ;
   double *result_mid;
};
typedef struct albusstruct albusdata;

class SPMVCSR {
  private:
  std::vector<local_int_t> row_ptr;
  std::vector<local_int_t> col_index;
  std::vector<double> values;
  local_int_t rows;
  local_int_t cols;
  local_int_t nonzeros;
  public:
  SPMVCSR(const SparseMatrix & A){
    rows=A.localNumberOfRows;
    cols=A.localNumberOfColumns;
    nonzeros=A.localNumberOfNonzeros;
    row_ptr.resize(rows + 1);
    row_ptr[0] = 0;
    row_ptr[rows] = nonzeros;
    col_index.resize(nonzeros);   
    values.resize(nonzeros);
// #ifdef HPCG_DEBUG
//   HPCG_fout << "Initial spmvcsr START"<< std::endl;
//   HPCG_fout << "NONZ:"<<nonzeros<< std::endl;
//   HPCG_fout << "rows:"<<rows<< std::endl;
//   HPCG_fout << "cols:"<<cols<< std::endl;
// #endif
    //转换为CSR格式
    int nnv=0;
    for (local_int_t i=0; i< rows; i++)  {
    row_ptr[i]=nnv;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];
    for (int j=0; j<cur_nnz; j++){  
        if(cur_vals[j]!=0){
            values[nnv]=cur_vals[j];
            col_index[nnv]=cur_inds[j];
            nnv++;
        }
    }  
    }
  }
  local_int_t getRows() const { return rows; }
  local_int_t getCols() const { return cols; }
  local_int_t  getNonzeros() const { return nonzeros; }
  const std::vector<local_int_t> &getRowPtr() const { return row_ptr; }
  const std::vector<local_int_t> &getColIndex() const { return col_index; }
  const std::vector<double> &getValues() const { return values; }
}; 