//
// Created by lxu on 23-6-23.
//

#include <iostream>
#include <vector>
#include "operator-chain.h"
#include "to-gurobi.h"
#include "dse.h"

int main() {

  DAT::Dim dim_n("n"), dim_l("l"), dim_q("q"), dim_m("m"), dim_k("k"), dim_p("p"), dim_d("d");
  dim_n.setSize(512);
  dim_l.setSize(512);
  dim_k.setSize(512);
  dim_p.setSize(512);
  dim_q.setSize(64);
  dim_d.setSize(64);
  dim_m.setSize(512);
  DAT::Tensor2D mat_i1("I1", &dim_n, &dim_l), mat_wq("Wq", &dim_l, &dim_q);
  DAT::Tensor2D mat_i2("I2", &dim_m, &dim_k), mat_wk("Wk", &dim_k, &dim_q);
  DAT::Tensor2D mat_q("Q", &dim_n, &dim_q), mat_k("K", &dim_m, &dim_q);
  DAT::Tensor2D mat_i3("I3", &dim_m, &dim_p), mat_wv("Wv", &dim_p, &dim_d);
  DAT::Tensor2D mat_s("S", &dim_n, &dim_m), mat_v("V", &dim_m, &dim_d);
  DAT::Tensor2D mat_a("A", &dim_n, &dim_d);
  DAT::MatrixMul mul_q("mul_q", &mat_i1, &mat_wq, &mat_q), mul_v("mul_v", &mat_i3, &mat_wv, &mat_v);
  DAT::MatrixMul mul_k("mul_k", &mat_i2, &mat_wk, &mat_k), mul_s("mul_s", &mat_q, &mat_k, &mat_s);
  DAT::MatrixMul mul_a("mul_a", &mat_s, &mat_v, &mat_a);
  DAT::OperatorNode m_q(&mul_q), m_v(&mul_v), m_k(&mul_k), m_s(&mul_s), m_a(&mul_a);
  DAT::OperatorChain op_chain[5];
  op_chain[0].addOperator(&mul_q);
  op_chain[1].addOperator(&mul_k);
  op_chain[2].addOperator(&mul_v);
  op_chain[3].addOperator(&mul_s);
  op_chain[4].addOperator(&mul_a);
  for (int i = 0; i < 5; ++i) {
    op_chain[i].updateTree();
  }

  long mem_size = 102400;
  long total_access_volume = 0;
  std::vector<long> results = {327680, 327680, 327680, 327680, 327680};
  for (int i = 0; i < 5; ++i) {
    DAT::OperatorChain mul_chain = op_chain[i];
    size_t dims_num = mul_chain.getDims().size();
    double best_obj = MAXFLOAT;
    std::vector<DAT::Dim *> best_dim_order;
    std::vector<long> record;
    record.reserve(dims_num - 1);
    std::vector<DAT::Dim *> best_root_dims_order;
    std::vector<std::vector<int> > best_free_dims_offsets;
    traversalDimOrder(&mul_chain, mem_size, best_obj, best_root_dims_order, best_free_dims_offsets);

    mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
    DAT::optimizeBlockSize(&mul_chain, mem_size);
    mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
    if (mul_chain.getMemAccessVolume() != results[i]) {
      std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
      std::cout << "failed at i: " << i << std::endl;
      std::cout << "result: " << results[i] << "  mem access: " << mul_chain.getMemAccessVolume() << std::endl;
      return 1;
    }
    total_access_volume += mul_chain.getMemAccessVolume();
  }

  if (total_access_volume != 1638400) {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout << total_access_volume << std::endl;
    return 1;
  }

  mem_size = 10240;
  total_access_volume = 0;
  results = {425984, 425984, 425984, 425984, 425984};
  for (int i = 0; i < 5; ++i) {
    DAT::OperatorChain mul_chain = op_chain[i];
    double best_obj = MAXFLOAT;
    std::vector<DAT::Dim *> best_root_dims_order;
    std::vector<std::vector<int> > best_free_dims_offsets;
    traversalDimOrder(&mul_chain, mem_size, best_obj, best_root_dims_order, best_free_dims_offsets);

    mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
    DAT::optimizeBlockSize(&mul_chain, mem_size);
    mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
    if (mul_chain.getMemAccessVolume() != results[i]) {
      std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
      std::cout << "failed at i: " << i << std::endl;
      std::cout << "result: " << results[i] << "  mem access: " << mul_chain.getMemAccessVolume() << std::endl;
      return 1;
    }
    total_access_volume += mul_chain.getMemAccessVolume() ;
  }

  if (total_access_volume != 2129920) {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout << total_access_volume << std::endl;
    return 1;
  }

  return 0;
}
