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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */

#include <immintrin.h>

#include "hpcg.hpp"
#include "ComputeSPMV_my.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include<vector>
#include<thread>
#include"OptimizeProblem.hpp"
#include "mytimer.hpp"
/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
// albus balance

void thread_block1(std::size_t thread_id, local_int_t start, local_int_t end,
                   local_int_t start2,  local_int_t end2,
                   local_int_t*  row_ptr,
                   local_int_t * col_index,
                   double * values,
                   double * mtx_ans,
                   double * mid_ans,
                   const double * const vectorx) {

  std::size_t start1, end1, num, Thread, i, j;
  double sum;
  switch (start < end) {
  case true: {
    mtx_ans[start] = 0.0;
    mtx_ans[end] = 0.0;
    start1 = row_ptr[start] + start2;
    start++;
    end1 = row_ptr[start];
    num = end1 - start1;
    Thread = thread_id << 1;
    sum = 0.0;
    //#pragma unroll(16)
    for (j = start1; j < end1; j++) {
      sum += values[j] * vectorx[col_index[j]];
    }
    mid_ans[Thread] = sum;
    start1 = end1;
    for (i = start; i < end; ++i) {
      end1 = row_ptr[i + 1];
      sum = 0.0;
      #pragma simd
      for (j = start1; j < end1; j++) {
        sum += values[j] * vectorx[col_index[j]];
      }
      mtx_ans[i] = sum;
      start1 = end1;
      
     /* 
      //SIMD
      double sum = 0.0;
       __m512d sum_vec = _mm512_setzero_pd(); // 初始化一个全零的AVX512向量
      int n=(end1-start1)/8;
      int st=start1;
      int en= start1+8*n;
      for (auto j = st; j < en; j += 8) {            
        // Load data1->value[j:j+7] into an AVX512 vector
        __m512d val_vec = _mm512_load_pd(&values[j]);      
        // Load data1->col_index[j:j+7] into an AVX512 vector
        __m512i col_idx_vec = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)&col_index[j]));
        // Gather xv[data1->col_index[j:j+7]] into an AVX512 vector
        __m512d x_vec = _mm512_i64gather_pd(col_idx_vec, vectorx, sizeof(double));
        // Perform the dot product using FMA (Fused Multiply-Add)    
         sum_vec = _mm512_fmadd_pd(val_vec, x_vec, sum_vec);        
      }     
      double sum_array[8];
      _mm512_store_pd(sum_array, sum_vec);
      for (int k = 0; k < 8; ++k) {
           sum += sum_array[k];
      }
    //HPCG_fout<<"simd2<<"<<std::endl;  
    //处理剩余的
    for(auto j =en; j < end1; j ++){
        sum += values[j] * vectorx[col_index[j]];
    }
    mtx_ans[i] = sum;
    start1 = end1;*/
    }
    start1 = row_ptr[end];
    end1 = start1 + end2;
    sum = 0.0;
    //#pragma unroll(16)
    for (j = start1; j < end1; j++) {
      sum += values[j] * vectorx[col_index[j]];
    }
    mid_ans[Thread | 1] = sum;
    return;
  }
  default: {
    mtx_ans[start] = 0.0;
    sum = 0.0;
    Thread = thread_id << 1;
    start1 = row_ptr[start] + start2;
    end1 = row_ptr[end] + end2;
    //#pragma unroll(8)
    for (j = start1; j < end1; j++) {
      sum += values[j] * vectorx[col_index[j]];
    }
    mid_ans[Thread] = sum;
    mid_ans[Thread | 1] = 0.0;
    return;
  }
  }
}


int ComputeSPMV_my( const SparseMatrix & A, Vector & x, Vector & y) {
   assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
   assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  albusdata* data1=( albusdata*)A.optimizationData;
  //HPCG_fout<<"simd<<"<<std::endl;
      //register  blocking//////////////////////////////
       local_int_t * col=data1->col_index;
       double * value=data1->value;
       #pragma omp parallel for schedule(static)
       for (auto i = 0; i < nrow; i++) {
            register double sum0 = 0.0;
            register double sum1 = 0.0;
            register double sum2 = 0.0;
            register double sum3 = 0.0;
            int j;
            // 使用寄存器阻塞处理多个元素
            int n=(data1->row_ptr[i + 1] - data1->row_ptr[i])/4;
            int end=4*n+data1->row_ptr[i];           
            #pragma vector always
            for (j = data1->row_ptr[i]; j <end ;j+=4) {
                sum0 +=value[j] * xv[col[j]];
                sum1 += value[j+1] * xv[col[j+1]];
                sum2 += value[j+2] * xv[col[j+2]];
                sum3 += value[j+3] * xv[col[j+3]];
            }

            // 处理剩余的元素
            for (; j < data1->row_ptr[i + 1]; j++) {
                sum0 += data1->value[j] * xv[data1->col_index[j]];
            }

            yv[i] = sum0 + sum1+sum2+sum3;
        }
  //单纯划分行的负载均衡问题
  /*
   #pragma omp for schedule(static) 	  
    for (local_int_t j=0; j< nrow; j++)  {
          double sum = 0.0;
          const double * const cur_vals = A.matrixValues[j];
          const local_int_t * const cur_inds = A.mtxIndL[j];
          const int cur_nnz = A.nonzerosInRow[j];
          for (int k=0; k< cur_nnz; k++){
              sum += cur_vals[k]*xv[cur_inds[k]];
          }     
          yv[j] = sum;
    }     
    */ 
   /*
  //行负载均衡+CSR版本的测试
    //double t00 = mytimer();
    //local_int_t * col=data1->col_index;
    #pragma omp parallel for schedule(static) 
    //#pragma omp parallel for
    for (auto i = 0; i < nrow; i++) {
      double sum = 0.0;
       #pragma vector always
       for (auto j = data1->row_ptr[i]; j < data1->row_ptr[i + 1]; j++) {  
        sum+= data1->value[j]*xv[data1->col_index[j]];
       }
       yv[i]=sum;   
    }  */
    
    //HPCG_fout<<"NoSIMD Time:"<< mytimer() - t00<<std::endl;  
   
    //double t0 = mytimer();
    /*
    #pragma omp parallel for schedule(static)
    for (auto i = 0; i < nrow; i++) {
    double sum = 0.0;
    __m512d sum_vec = _mm512_setzero_pd(); // 初始化一个全零的AVX512向量

    int n=(data1->row_ptr[i + 1]-data1->row_ptr[i])/8;
    int start=data1->row_ptr[i];
    int end= data1->row_ptr[i]+8*n;
    for (auto j = start; j < end; j += 8) {            
        // Load data1->value[j:j+7] into an AVX512 vector
        __m512d val_vec = _mm512_load_pd(&data1->value[j]);      
        // Load data1->col_index[j:j+7] into an AVX512 vector
        __m512i col_idx_vec = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)&data1->col_index[j]));
        // Gather xv[data1->col_index[j:j+7]] into an AVX512 vector
        __m512d x_vec = _mm512_i64gather_pd(col_idx_vec, xv, sizeof(double));
        // Perform the dot product using FMA (Fused Multiply-Add)    
         sum_vec = _mm512_fmadd_pd(val_vec, x_vec, sum_vec);        
      }     
        double sum_array[8];
       _mm512_store_pd(sum_array, sum_vec);
        for (int k = 0; k < 8; ++k) {
           sum += sum_array[k];
      } 
    //处理剩余的
    start=data1->row_ptr[i]+8*n;
    end =data1->row_ptr[i + 1];
    for(auto j =start; j < end; j ++){
        sum+= data1->value[j]*xv[data1->col_index[j]];
    }
    yv[i] = sum;
    }*/
    //HPCG_fout<<"SIMDime:"<< mytimer() - t0<<std::endl;   
    //负载均衡版本
    /*
    double t000 = mytimer();
    register int i;
    double * result=new double[nrow + 1];
    memset(result, 0.0, (nrow + 1)* sizeof(double));
    //omp_set_num_threads(4);
    //data1->thread_nums=omp_get_num_threads();
       #pragma omp parallel private(i)
       {
        #pragma omp for schedule(static) nowait
		    for(i=0;i<data1->thread_nums;++i) {
             
            //thread_block1(i, data1->start[i], data1->end[i], data1->start1[i], data1->end1[i], data1->row_ptr, data1->col_index,
            //data1->value, result, data1->result_mid, xv);
            //////////////////////
            local_int_t start =data1->start[i];
            local_int_t end =data1->end[i];
            local_int_t start2=data1->start1[i];
            local_int_t end2=data1->end1[i];
            local_int_t * row_ptr=data1->row_ptr;
            std::size_t start1, end1, Thread, k, j;
            double sum;
            
            if (start < end) {
            result[start] = 0.0;
            result[end] = 0.0;
            start1 = row_ptr[start] + start2;
            start++;
            end1 = row_ptr[start];
            //num = end1 - start1;
            Thread = i<< 1;
            sum = 0.0;
            #pragma vector always         
            for (j = start1; j < end1; j++) {
              sum += data1->value[j] * xv[data1->col_index[j]];
            }
            data1->result_mid[Thread] = sum;
            start1 = end1;
            
            for (k = start; k< end; ++k) {
              end1 = row_ptr[k + 1];
              sum = 0.0;
              #pragma vector always
              for (j = start1; j < end1; j++) {        
                sum += data1->value[j] * xv[data1->col_index[j]];
              }
              result[k] = sum;
              start1 = end1;
            }
            start1 = row_ptr[end];
            end1 = start1 + end2;
            sum = 0.0;
            //#pragma unroll(16)
            #pragma vector always
            for (j = start1; j < end1; j++) {            
               sum += data1->value[j] * xv[data1->col_index[j]];
               }
               data1->result_mid[Thread | 1] = sum;

             //HPCG_fout<<"6:"<<std::endl;
             }
             else {
               result[start] = 0.0;
               sum = 0.0;
               Thread = i << 1;
               start1 = row_ptr[start] + start2;
               end1 = row_ptr[end] + end2;
               //#pragma unroll(8)
               
               for (j = start1; j < end1; j++) {
                 
                 sum += data1->value[j] * xv[data1->col_index[j]];
               }
               data1->result_mid[Thread] = sum;
               data1->result_mid[Thread | 1] = 0.0;
               //HPCG_fout<<"7:"<<std::endl;
             }
           
         }
    }
    HPCG_fout<<"ALBcacT:"<< mytimer() - t000<<std::endl;    
   //结果整合
   result[0] = data1->result_mid[0];
   std::size_t sub;
   for (std::size_t i = 1; i < data1->thread_nums; ++i) {
    sub = i << 1;
    std::size_t tmp1 = data1->start[i];
    std::size_t tmp2 = data1->end[i - 1];
    if (tmp1 == tmp2) {
      result[tmp1] += (data1->result_mid[sub - 1] + data1->result_mid[sub]);
    } else {
      result[tmp1] += data1->result_mid[sub];
      result[tmp2] += data1->result_mid[sub - 1];
    }
   }
   #pragma vector always
   for (local_int_t i=0; i< nrow; i++)  {
     yv[i] = result[i];
   }

 HPCG_fout<<"ALBUStime:"<< mytimer() - t000<<std::endl;  */   

  return 0;
}
