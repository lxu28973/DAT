#include <iostream>
#include <vector>
#include "src/operator-chain.h"
#include "src/to-gurobi.h"
#include "src/dse.h"
#include "src/options.h"

int main(int argc, char *argv[]) {

  DAT::Options::parse(argc, argv);

  DAT::Dim dim_n("n"), dim_l("l"), dim_q("q"), dim_m("m"), dim_k("k"), dim_p("p"), dim_d("d");
  DAT::Dim dim_bs("bsc"), dim_hs("hsc");

  long h_size = DAT::Options::head_num;
  long d_size = DAT::Options::hid_size;
  long n_size = DAT::Options::seq_length;
  long mem_size = DAT::Options::mem_size;
  long batch_size = DAT::Options::batch_size;
  long batch_blocksize = DAT::Options::batch_blocksize;
  long head_blocksize = DAT::Options::head_blocksize;
  long dh_size = d_size * h_size;
  dim_n.setSize(n_size);
  dim_m.setSize(n_size);
  dim_l.setSize(dh_size);
  dim_k.setSize(dh_size);
  dim_p.setSize(dh_size);
  dim_q.setSize(d_size);
  dim_d.setSize(d_size);
  dim_bs.setSize(batch_size);
  dim_hs.setSize(h_size);

  DAT::Tensor3D mat_i1("I1", &dim_n, &dim_l, &dim_bs), mat_wq("Wq", &dim_l, &dim_q, &dim_hs);
  DAT::Tensor3D mat_i2("I2", &dim_m, &dim_k, &dim_bs), mat_wk("Wk", &dim_k, &dim_q, &dim_hs);
  DAT::Tensor3D mat_i3("I3", &dim_m, &dim_p, &dim_bs), mat_wv("Wv", &dim_p, &dim_d, &dim_hs);
  DAT::Tensor4D mat_q("Q", &dim_n, &dim_q, &dim_bs, &dim_hs),
      mat_k("K", &dim_m, &dim_q, &dim_bs, &dim_hs);
  DAT::Tensor4D mat_s("S", &dim_n, &dim_m, &dim_bs, &dim_hs),
      mat_v("V", &dim_m, &dim_d, &dim_bs, &dim_hs);
  DAT::Tensor4D mat_a("A", &dim_n, &dim_d, &dim_bs, &dim_hs);
  DAT::MatrixMul mul_q("mul_q", &mat_i1, &mat_wq, &mat_q), mul_v("mul_v", &mat_i3, &mat_wv, &mat_v);
  DAT::MatrixMul mul_k("mul_k", &mat_i2, &mat_wk, &mat_k), mul_s("mul_s", &mat_q, &mat_k, &mat_s);
  DAT::MatrixMul mul_a("mul_a", &mat_s, &mat_v, &mat_a);
  mul_q.isBatchDependent(true);
  mul_k.isBatchDependent(true);
  mul_v.isBatchDependent(true);
  mul_s.isBatchDependent(false);
  mul_a.isBatchDependent(false);
  mul_q.isHeadDependent(true);
  mul_k.isHeadDependent(true);
  mul_v.isHeadDependent(true);
  mul_s.isHeadDependent(false);
  mul_a.isHeadDependent(false);
  DAT::OperatorNode m_q(&mul_q), m_v(&mul_v), m_k(&mul_k), m_s(&mul_s), m_a(&mul_a);

  dim_bs.setBlockSize(&mul_q, batch_blocksize);
  dim_hs.setBlockSize(&mul_q, head_blocksize);
  dim_bs.setBlockSize(&mul_k, batch_blocksize);
  dim_hs.setBlockSize(&mul_k, head_blocksize);
  dim_bs.setBlockSize(&mul_v, batch_blocksize);
  dim_hs.setBlockSize(&mul_v, head_blocksize);
  dim_bs.setBlockSize(&mul_s, batch_blocksize);
  dim_hs.setBlockSize(&mul_s, head_blocksize);
  dim_bs.setBlockSize(&mul_a, batch_blocksize);
  dim_hs.setBlockSize(&mul_a, head_blocksize);

  std::set<DAT::TensorOperator *>
      non_add_to_operator_chain = {&mul_q, &mul_k, &mul_v, &mul_s, &mul_a};
  std::set<DAT::TensorOperator *>
      add_to_operator_chain;
  std::set<DAT::Tensor *> tensors =
      {&mat_i1, &mat_i2, &mat_i3, &mat_wq, &mat_wk, &mat_wv, &mat_q, &mat_k, &mat_v, &mat_s,
       &mat_a};

  DAT::LogRecord6 log_record;
  if (DAT::Options::save_log_file >= 1) {
    log_record.id = 1;
    log_record.defineAlgroithm(non_add_to_operator_chain);
    log_record.set_ndh(n_size, d_size, h_size);
    log_record.mem_size = mem_size;
  }

  DAT::OperatorChain *op_chain[16];
  long operator_chain_num =
      DAT::traversalFused(non_add_to_operator_chain, tensors, mem_size, op_chain,
                          DAT::Options::print_to_screen);
  for (long i = 0; i < operator_chain_num; ++i) {
    op_chain[i]->setTensorsIsExternal();
  }
  if (DAT::Options::save_log_file >= 1) {
    std::set<DAT::Tensor *> non_io_tensors;
    for (auto t :tensors) {
      if (!t->isIO()) {
        non_io_tensors.insert(t);
      }
    }
    log_record.recordFuseStatus(non_io_tensors);
  }
  long total_access_volume = 0;
  long total_compute_time = 0;
  long mem_footprint = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    mul_chain->setInternalTensorsFuse();
    mul_chain->setExternalTensorsFusePattern(mul_chain->getExternalTensorsFusePattern());
    mul_chain->setOrder(mul_chain->getOperatorTree().getOrderInfos());
    DAT::optimizeBlockSize(mul_chain, mem_size);
    mul_chain->setOrder(mul_chain->getOperatorTree().getOrderInfos());
    if (DAT::Options::save_log_file >= 1) {
      log_record.recordSubFuseStatus(mul_chain, mul_chain->getExternalTensors());
      log_record.recordOrder(mul_chain, mul_chain->getOperatorTree().getOrderInfos());
      log_record.recordDimBlocksizes(mul_chain);
      log_record.recordOpChainMemAccessVolume(mul_chain);
    }
    std::cout << mul_chain->toString();
    std::cout << "access times: " << mul_chain->getMemAccessTimes() << std::endl;
    std::cout << "access volume: " << mul_chain->getMemAccessVolume() << std::endl;
    std::cout << "SRAM footprint: " << mul_chain->getMemFootprint() << std::endl;
    total_access_volume += mul_chain->getMemAccessVolume();
    total_compute_time += mul_chain->compute_time();
    mem_footprint = std::max(mem_footprint, mul_chain->getMemFootprint());
  }

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "total access volume: " << total_access_volume << std::endl;

  if (DAT::Options::save_log_file >= 1) {
    log_record.mem_access_volume = total_access_volume;
    log_record.compute_time = total_compute_time;
    log_record.mem_footprint = mem_footprint;
    log_record.writeToFile(DAT::Options::log_directory + "/log1.csv");
  }

  for (long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
  }

  return 0;
}
