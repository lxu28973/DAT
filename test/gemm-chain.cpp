#include "operator-chain.h"

int main() {

  DAT::Dim dim_n("n"), dim_l("l"), dim_q("q"), dim_m("m");
  dim_n.setSize(512);
  dim_l.setSize(512);
  dim_q.setSize(64);
  dim_m.setSize(512);
  DAT::Tensor2D mat_i("I", &dim_n, &dim_l), mat_wq("Wq", &dim_l, &dim_q);
  DAT::Tensor2D mat_q("Q", &dim_n, &dim_q), mat_k("K", &dim_m, &dim_q),
      mat_s("S", &dim_n, &dim_m);
  DAT::MatrixMul mul_q(&mat_i, &mat_wq, &mat_q), mul_s(&mat_q, &mat_k, &mat_s);
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
  mul_chain.updateTree();
  mul_chain.setDimsOrder({&dim_l, &dim_q, &dim_m, &dim_n});
  if (mul_chain.getMemAccessTimes() == 14336 && mul_chain.getMemAccessVolume() == 3407872
      && mul_chain.getMemFootprint() == 2048) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }
  mul_chain.setDimsOrder({&dim_q, &dim_l, &dim_m, &dim_n});
  if (mul_chain.getMemAccessTimes() == 11264 && mul_chain.getMemAccessVolume() == 2621440
      && mul_chain.getMemFootprint() == 2048) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }
  mul_chain.setDimsOrder({&dim_m, &dim_l, &dim_q, &dim_n});
  if (mul_chain.getMemAccessTimes() == 20480 && mul_chain.getMemAccessVolume() == 4980736
      && mul_chain.getMemFootprint() == 768) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }
  mul_chain.setDimsOrder({&dim_n, &dim_l, &dim_q, &dim_m});
  if (mul_chain.getMemAccessTimes() == 12544 && mul_chain.getMemAccessVolume() == 2949120
      && mul_chain.getMemFootprint() == 33792) {
  } else {
    std::cerr << "failed:" << __FILE__ << ":" << __LINE__ << std::endl;
    std::cout <<  mul_chain.getMemAccessTimes() << " " << mul_chain.getMemAccessVolume() << " " <<  mul_chain.getMemFootprint();
    return 1;
  }

  return 0;

}
