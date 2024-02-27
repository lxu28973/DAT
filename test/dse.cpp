//
// Created by lxu on 23-6-24.
//
#include <iostream>
#include <vector>
#include "operator-chain.h"
#include "to-gurobi.h"
#include "dse.h"

int main(int argc, char *argv[]) {

  DAT::Options::parse(argc, argv);

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
  DAT::MatrixMul mul_a("mul_s", &mat_s, &mat_v, &mat_a);
  DAT::OperatorNode m_q(&mul_q), m_v(&mul_v), m_k(&mul_k), m_s(&mul_s), m_a(&mul_a);

  mat_i1.unsetFuse();
  mat_i2.unsetFuse();
  mat_i3.unsetFuse();
  mat_wq.unsetFuse();
  mat_wk.unsetFuse();
  mat_wv.unsetFuse();
  mat_q.unsetFuse();
  mat_k.unsetFuse();
  mat_v.unsetFuse();
  mat_s.unsetFuse();
  mat_a.unsetFuse();

  std::set<DAT::TensorOperator *>
      non_add_to_operator_chain = {&mul_q, &mul_k, &mul_v, &mul_s, &mul_a};
  std::set<DAT::TensorOperator *>
      add_to_operator_chain;
  std::set<DAT::Tensor *> tensors =
      {&mat_i1, &mat_i2, &mat_i3, &mat_wq, &mat_wk, &mat_wv, &mat_q, &mat_k, &mat_v, &mat_s,
       &mat_a};

  long mem_size = 1024 * 64;
  DAT::OperatorChain *op_chain[16] = {nullptr};
  long operator_chain_num = DAT::traversalFused(non_add_to_operator_chain, tensors, mem_size, op_chain);
  for (long i = 0; i < operator_chain_num; ++i) {
    op_chain[i]->setTensorsIsExternal();
  }

  long total_access_volume = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    mul_chain->setInternalTensorsFuse();
    mul_chain->setExternalTensorsFusePattern(mul_chain->getExternalTensorsFusePattern());
    mul_chain->setOrder(mul_chain->getOperatorTree().getOrderInfos());
    DAT::optimizeBlockSize(mul_chain, mem_size);
    mul_chain->setOrder(mul_chain->getOperatorTree().getOrderInfos());
    total_access_volume += mul_chain->getMemAccessVolume();
  }

  if (total_access_volume != 917504) {
    std::cout << total_access_volume << std::endl;
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    return 1;
  }

  for(long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
  }

  return 0;
}
