//
// Created by Lei Xu on 2023/7/18.
//
#include "operator-chain.h"

int main() {

  DAT::Dim dim_n("n"), dim_l("l"), dim_q("q"), dim_m("m"), dim_b("bsc"), dim_h("hsc");
  dim_n.setSize(512);
  dim_l.setSize(512);
  dim_q.setSize(64);
  dim_m.setSize(512);
  dim_b.setSize(8);
  dim_h.setSize(16);
  DAT::Tensor3D mat_i("I", &dim_n, &dim_l, &dim_b), mat_wq("Wq", &dim_l, &dim_q, &dim_h);
  DAT::Tensor4D mat_q("Q", &dim_n, &dim_q, &dim_b, &dim_h), mat_k("K", &dim_m, &dim_q, &dim_b, &dim_h),
      mat_s("S", &dim_n, &dim_m, &dim_b, &dim_h);
  DAT::MatrixMul mul_q(&mat_i, &mat_wq, &mat_q), mul_s(&mat_q, &mat_k, &mat_s);
  DAT::OperatorNode m_q(&mul_q), m_s(&mul_s);
  dim_n.setBlockSize(&mul_q, 16);
  dim_l.setBlockSize(&mul_q, 16);
  dim_q.setBlockSize(&mul_q, 16);
  dim_n.setBlockSize(&mul_s, 16);
  dim_q.setBlockSize(&mul_s, 16);
  dim_m.setBlockSize(&mul_s, 16);
  dim_b.setBlockSize(&mul_q, 2);
  dim_h.setBlockSize(&mul_q, 4);
  dim_b.setBlockSize(&mul_s, 2);
  dim_h.setBlockSize(&mul_s, 4);
  mat_q.setFuse();
  DAT::OperatorChain mul_chain;
  mul_chain.addOperator(&mul_q, &mul_s);
  DAT::updateOperatorTreeRelationship({&mat_q, &mat_s, &mat_i, &mat_k, &mat_wq});
  mul_chain.updateTree();
  mul_chain.setDimsOrder({&dim_l, &dim_q, &dim_m, &dim_n, &dim_b, &dim_h});
  if (mul_chain.getMemAccessTimes() == 229376 && mul_chain.getMemAccessVolume() == 268435456
      && mul_chain.getMemFootprint() == 13824) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }
  mul_chain.setDimsOrder({&dim_q, &dim_l, &dim_m, &dim_n, &dim_b, &dim_h});
  if (mul_chain.getMemAccessTimes() == 180224 && mul_chain.getMemAccessVolume() == 243269632
      && mul_chain.getMemFootprint() == 13824) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }
  mul_chain.setDimsOrder({&dim_m, &dim_l, &dim_q, &dim_n, &dim_b, &dim_h});
  if (mul_chain.getMemAccessTimes() == 327680 && mul_chain.getMemAccessVolume() == 469762048
      && mul_chain.getMemFootprint() == 6144) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }
  mul_chain.setDimsOrder({&dim_n, &dim_l, &dim_q, &dim_m, &dim_b, &dim_h});
  if (mul_chain.getMemAccessTimes() == 200704 && mul_chain.getMemAccessVolume() == 274726912
      && mul_chain.getMemFootprint() == 267776) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }

  return 0;

}
