//
// Created by lxu on 23-6-23.
//
#include <iostream>
#include <vector>
#include "operator-chain.h"
#include "to-gurobi.h"
#include "dse.h"

int main(int argc, char *argv[]) {

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
  DAT::Tensor2D mat_s("S", &dim_n, &dim_m), mat_v("V", &dim_m, &dim_d),
      mat_a("A", &dim_n, &dim_d);
  DAT::MatrixMul mul_q("mul_q", &mat_i1, &mat_wq, &mat_q),
      mul_v("mul_v", &mat_i3, &mat_wv, &mat_v);
  DAT::MatrixMul mul_k("mul_k", &mat_i2, &mat_wk, &mat_k),
      mul_s("mul_s", &mat_q, &mat_k, &mat_s);
  DAT::MatrixMul mul_a("mul_a", &mat_s, &mat_v, &mat_a);
  DAT::OperatorNode m_q(&mul_q), m_v(&mul_v), m_k(&mul_k), m_s(&mul_s), m_a(&mul_a);
  for (auto d : mul_q.getDims()) {
    d->setBlockSize(&mul_q, 16);
  }
  for (auto d : mul_k.getDims()) {
    d->setBlockSize(&mul_k, 16);
  }
  for (auto d : mul_v.getDims()) {
    d->setBlockSize(&mul_v, 16);
  }
  for (auto d : mul_s.getDims()) {
    d->setBlockSize(&mul_s, 16);
  }
  for (auto d : mul_a.getDims()) {
    d->setBlockSize(&mul_a, 16);
  }
  mat_q.setFuse();
  mat_k.setFuse();
  mat_s.setFuse();
  mat_v.setFuse();
  DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
  long operator_chain_num =
      DAT::createToOperatorChain(op_chain, {&mul_q, &mul_k, &mul_s, &mul_v, &mul_a},
                                 {&mat_q, &mat_s, &mat_i1, &mat_i2, &mat_i3, &mat_k, &mat_v,
                                      &mat_wk, &mat_wv, &mat_a, &mat_wq});
  assert(operator_chain_num == 1);
  DAT::OperatorChain mul_chain = *op_chain[0];
  mul_chain.setDimsOrder({&dim_m, &dim_n, &dim_l, &dim_q, &dim_k, &dim_d, &dim_p});

  long mem_size = 102400;
  double opt_obj = DAT::optimizeBlockSize(&mul_chain, mem_size);
  if (opt_obj != MAXFLOAT) {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    return 1;
  }

  double best_obj = MAXFLOAT;
  std::vector<DAT::Dim *> best_root_dims_order;
  std::vector<std::vector<int> > best_free_dims_offsets;
  traversalDimOrder(&mul_chain, mem_size, best_obj, best_root_dims_order,
                    best_free_dims_offsets);

  mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
  DAT::optimizeBlockSize(&mul_chain, mem_size);
  mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);

  if (mul_chain.getMemAccessVolume() != 917504) {
    std::cout << mul_chain.getMemAccessVolume() << std::endl;
    std::cout << mul_chain.getMemFootprint() << std::endl;
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    return 1;
  }

  mem_size = 1024 * 64;
  best_obj = MAXFLOAT;
  traversalDimOrder(&mul_chain, mem_size, best_obj, best_root_dims_order,
                    best_free_dims_offsets);

  mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);
  DAT::optimizeBlockSize(&mul_chain, mem_size);
  mul_chain.setDimsOrder(best_root_dims_order, best_free_dims_offsets);

  if (mul_chain.getMemAccessVolume() != 1015808) {
    std::cout << mul_chain.getMemAccessVolume() << std::endl;
    std::cout << mul_chain.getMemFootprint() << std::endl;
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    return 1;
  }
  return 0;
}
