//
// Created by lxu on 23-6-23.
//
#include <iostream>
#include <vector>
#include "operator-chain.h"
#include "to-gurobi.h"
#include "dse.h"

int main() {

  DAT::Dim dim_n("n"), dim_l("l"), dim_q("q"), dim_m("m");
  dim_n.setSize(512);
  dim_l.setSize(512);
  dim_q.setSize(64);
  dim_m.setSize(512);
  DAT::Tensor2D mat_i("I", &dim_n, &dim_l), mat_wq("Wq", &dim_l, &dim_q);
  DAT::Tensor2D mat_q("Q", &dim_n, &dim_q), mat_k("K", &dim_m, &dim_q),
      mat_s("S", &dim_n, &dim_m);
  DAT::MatrixMul mul_q("mul_q", &mat_i, &mat_wq, &mat_q), mul_s("mul_s", &mat_q, &mat_k, &mat_s);
  DAT::OperatorNode m_q(&mul_q), m_s(&mul_s);
  dim_n.setBlockSize(&mul_q, 16);
  dim_l.setBlockSize(&mul_q, 16);
  dim_q.setBlockSize(&mul_q, 16);
  dim_n.setBlockSize(&mul_s, 16);
  dim_q.setBlockSize(&mul_s, 16);
  dim_m.setBlockSize(&mul_s, 16);
  mat_q.setFuse();
  DAT::OperatorChain mul_chain;
  mul_chain.addOperator(&mul_q, &mul_s);
  DAT::updateOperatorTreeRelationship({&mat_q, &mat_s, &mat_i, &mat_k, &mat_wq});
  mul_chain.addInternalTensor(&mat_q);
  mul_chain.updateTree();
  mul_chain.setDimsOrder({&dim_l, &dim_q, &dim_m, &dim_n});

  long mem_size = 10240;
  double opt_obj = DAT::optimizeBlockSize(&mul_chain, mem_size);
  if (opt_obj != 524288 * 2 ) {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout << opt_obj << std::endl;
    return 1;
  }

  double best_obj = MAXFLOAT;
  std::vector<DAT::Dim *> best_root_dims_order;
  std::vector<std::vector<int> > best_free_dims_offsets;
  traversalDimOrder(&mul_chain, mem_size, best_obj, best_root_dims_order, best_free_dims_offsets);

  mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
  DAT::optimizeBlockSize(&mul_chain, mem_size);
  mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
  if (mul_chain.getMemAccessVolume() != 786432) {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout << mul_chain.getMemAccessVolume() << " should be 786432" << std::endl;
    return 1;
  }

  return 0;
}
