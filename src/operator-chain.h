//
// Created by Lei Xu on 2023/7/13.
//

#ifndef MMCHAIN_ANALYSIS_SRC_OPERATOR_CHAIN_H
#define MMCHAIN_ANALYSIS_SRC_OPERATOR_CHAIN_H

#include "tensor-operator.h"
#include "defines.h"
#include <bitset>
#include <utility>
#include "operator-tree.h"

namespace DAT {

void updateOperatorTreeRelationship(const std::set<DAT::Tensor *> &tensors) {
  for (auto t : tensors) {
    for (auto to : t->getRelatedOperator()) {
      to.first->getLinkedNode()->clearParent();
      to.first->getLinkedNode()->clearChildren();
    }
  }
  for (auto t : tensors) {
    if ((!t->isIO()) && t->isFused()) {
      auto related_ops = t->getRelatedOperator();
      OperatorNode *parent = nullptr;
      OperatorNode *child = nullptr;
      for (auto op : related_ops) {
        if (op.second == "output" && child == nullptr) {
          child = op.first->getLinkedNode();
        } else if (op.second == "input" && parent == nullptr) {
          parent = op.first->getLinkedNode();
        } else {
          assert(0);
        }
      }
      assert(child != nullptr);
      assert(parent != nullptr);
      parent->addChild(child);
      child->setParent(parent);
    }
  }
}

class OperatorChain {
public:
  void setOrder(std::map<OperatorNode *, OrderInfo> o_infos) {
    tree.setOrder(std::move(o_infos));
    analysisEnvTensors();
    analyzeMemFootprint();
  }
  void recordOrder(std::map<OperatorNode *, OrderInfo> o_infos) {
    tree.setOrder(std::move(o_infos));
  }
  void setDimsOrder(const std::vector<Dim *> &d_init_order,
                    std::vector<std::vector<int> > offsets) {
    for (auto d : d_init_order) {
      assert(tree.getRoot()->getOperator()->getDims().count(d));
    }
    tree.setDimsOrder(d_init_order, std::move(offsets));
    analysisEnvTensors();
    analyzeMemFootprint();
  }
  void setDimsOrder(const std::vector<Dim *> &d_order) {
    assert(d_order.size() == dims.size()
               || ((std::cout << "d_order: " << d_order.size() << ", dims: " << dims.size()
                              << std::endl) && false));
    for (auto node : tree.getNodes()) {
      std::vector<Dim *> node_d_order;
      for (auto d : d_order) {
        if (node->getOperator()->getDims().count(d)) {
          node_d_order.push_back(d);
        }
      }
      node->setDimsOrder(node_d_order);
    }
    analysisEnvTensors();
    analyzeMemFootprint();
  }
  [[nodiscard]] OperatorTree getOperatorTree() const {
    return tree;
  }
  void addOperator() {
  }
  template<class T, class... Args>
  void addOperator(T to, Args... tos) {
    for (auto op : operators) {
      assert(op != to);
    }
    operators.insert(to);
    auto ts = to->getTensors();
    tensors.insert(ts.begin(), ts.end());
    for (auto d : to->getDims()) {
      dims.insert(d);
    }
    tree.setRoot(to->getLinkedNode());
    addOperator(tos...);
  }
  std::set<Dim *> getDims() {
    return dims;
  }

  Dim *getDim(const std::string &n) {
    for (auto dim : dims) {
      if (dim->getName() == n) {
        return dim;
      }
    }
    return nullptr;
  }
  bool hasOperator(TensorOperator * op) {
    return operators.count(op);
  }
  std::set<TensorOperator *> getOperators() {
    return operators;
  }
  [[nodiscard]] long getMemAccessTimes() const {
    return mem_access_times;
  }
  [[nodiscard]] long getMemAccessVolume() const {
    return mem_access_volume;
  }
  [[nodiscard]] long getMemFootprint() const {
    return mem_footprint;
  }
  [[nodiscard]] long getExternalTensorsFusePattern() const {
    assert(external_tensors_fuse_pattern >= 0);
    return external_tensors_fuse_pattern;
  }
  [[nodiscard]] std::map<Tensor *, bool> getExternalTensorsFuseStatus() const {
    assert(!external_tensor_fuse_status.empty());
    return external_tensor_fuse_status;
  }
  void recordExternalTensorsFuseInfo(long fp) {
    assert(fp >= 0);
    external_tensors_fuse_pattern = fp;
    external_tensor_fuse_status.clear();
    std::bitset<MAX_TENSOR_NUM> sub_fuse_or_not(fp);
    long sf_i = 0;
    for (auto t : external_tensors) {
      if (sub_fuse_or_not[sf_i]) {
        external_tensor_fuse_status[t] = true;
      } else {
        external_tensor_fuse_status[t] = false;
      }
      ++sf_i;
    }
  }
  void setInternalTensorsFuse() {
    for (auto t : internal_tensors) {
      t->setFuse();
    }
  }
  void setExternalTensorsFusePattern(long fp) {
    recordExternalTensorsFuseInfo(fp);
    std::bitset<MAX_TENSOR_NUM> sub_fuse_or_not(fp);
    long sf_i = 0;
    for (auto t : external_tensors) {
      if (sub_fuse_or_not[sf_i]) {
        t->setFuse();
      } else {
        t->unsetFuse();
      }
      ++sf_i;
    }
  }
  std::string toString() {
    std::string str;
    for (auto op : operators) {
      str += op->toString();
      str += "\n";
    }
    return str;
  }
  bool hasTensor(Tensor *t) {
    for (auto op : operators) {
      if (op->hasTensor(t)) {
        return true;
      }
    }
    return false;
  }
  long getOpsNum() {
    long ops = 0;
    for (auto op : operators) {
      ops += op->getOpsNum();
    }
    return ops;
  }
  bool hasExternalTensor(Tensor *t) {
    return std::any_of(external_tensors.begin(), external_tensors.end(),
                       [t](auto e) { return e == t; });
  }
  void addInternalTensor(Tensor *t) {
    internal_tensors.insert(t);
    t->isExternal(false);
  }
  void addExternalTensor(Tensor *t) {
    external_tensors.insert(t);
    t->isExternal(true);
  }
  void removeInternalTensor(Tensor *t) {
    internal_tensors.erase(t);
  }
  void removeExternalTensor(Tensor *t) {
    external_tensors.erase(t);
  }
  std::set<Tensor *> getInternalTensors() {
    return internal_tensors;
  }
  std::set<Tensor *> getExternalTensors() {
    return external_tensors;
  }
  void updateTree() {
    tree.updateNodes();
  }
  void setTensorsIsExternal() {
    for (auto t : internal_tensors) {
      t->isExternal(false);
    }
    for (auto t : external_tensors) {
      t->isExternal(true);
    }
  }
  long compute_time() {
    long t = 0;
    for (auto to : operators) {
      t += to->compute_time();
    }
    return t;
  }

  std::vector<std::set<TensorOperator *> > getOperatorGroups() {
    return operator_groups;
  }


protected:
  void checkOperatorChainValid() {
    std::set<Tensor *> io_tensors;
    for (auto op : operators) {
      for (auto input : op->getInputTensors()) {
        if (!input->isFused()) {
          assert(io_tensors.count(input) == 0
                     && "same infusing tensor instance is used in two operators. need to use two infusing instance for two operators");
        }
      }
      for (auto output : op->getOutputTensors()) {
        if (!output->isFused()) {
          assert(io_tensors.count(output) == 0
                     && "same infusing tensor instance is used in two operators. need to use two infusing instance for two operators");
        }
      }
    }
  }

  void analysisEnvTensors() {
    auto op_nodes = tree.getNodes();
    auto order_infos = tree.getOrderInfos();
    for (auto node : op_nodes) {
      node->getOperator()->clearEnvTensors();
    }
    for (auto node : op_nodes) {
      auto exec_rank_this = order_infos[node].execute_rank;
      auto exec_rank_parent = order_infos[node->getParent()].execute_rank;
      for (auto op : op_nodes) {
        if (op != node && op != node->getParent()) {
          auto exec_rank = order_infos[op].execute_rank;
          // the ">=, <=" is desired when allow multi-operators execute together
          if (exec_rank > exec_rank_parent && exec_rank < exec_rank_this) {
            op->getOperator()->addEnvTensor(node->getOperator()->getOutputTensors());
          }
        }
      }
    }
  }

  void clearMemAccess() {
    for (auto op : operators) {
      op->clearMemAccess();
    }
  }

  void clearMemFootprint() {
    for (auto op : operators) {
      op->clearMemFootprint();
    }
  }

  void analyzeMemAccess() {
    checkOperatorChainValid();
    clearMemAccess();
    long total_access_times = 0;
    long total_access_volume = 0;
    for (auto op : operators) {
      op->analyzeMemAccess(op->getLinkedNode()->getDimsOrder());
      total_access_times += op->getAccessTimes();
      total_access_volume += op->getAccessVolume();
    }
    mem_access_times = total_access_times;
    mem_access_volume = total_access_volume;
  }

  void analyzeMemFootprint() {
    clearMemFootprint();
    analyzeMemAccess();
    long total_mem_footprint = 0;
    std::string total_mem_footprint_str = "max: ";
    for (auto t : tensors) {
      t->clearExpandDims();
    }
    for (auto op : operators) {
      op->analyzeMemFootprint(op->getLinkedNode()->getDimsOrder());
    }

    buildOperatorGroup();

    for (const auto& og : operator_groups) {
      long og_mem_footprint = 0;
      std::string og_mem_footprint_str = "0";
      std::vector<Tensor *> op_tensors;
      for (auto op : og) {
        op_tensors.reserve(op_tensors.size() + op->getInputTensors().size() + op->getOutputTensors().size());
        auto a = op->getInputTensors();
        auto b = op->getOutputTensors();
        auto c = op->getEnvTensors();
        op_tensors.insert(op_tensors.end(), a.begin(), a.end());
        op_tensors.insert(op_tensors.end(), b.begin(), b.end());
        op_tensors.insert(op_tensors.end(), c.begin(), c.end());
      }
      std::set<Tensor *> tensor_sets(op_tensors.begin(), op_tensors.end());
      for (auto t : tensor_sets) {
        if (t->isFused()) {
          long max_footprint = 0;
          std::string max_footprint_str;
          for (auto tf : t->getMemFootprint()) {
            if (tf.second > max_footprint) {
              max_footprint = tf.second;
              max_footprint_str = t->getMemFootprintStr()[tf.first];
            }
          }
          for (auto pt : t->getParTensors()) {
            for (auto ptf : pt->getMemFootprint()) {
              if (ptf.second > max_footprint) {
                max_footprint = ptf.second;
                max_footprint_str = pt->getMemFootprintStr()[ptf.first];
              }
            }
          }
          og_mem_footprint += max_footprint;
          og_mem_footprint_str += " + " + max_footprint_str;
        } else {
          for (const auto& ro : t->getRelatedOperator()) {
            if (og.count(ro.first)) {
              og_mem_footprint += t->getMemFootprint()[ro.first];
              og_mem_footprint_str += " + " + t->getMemFootprintStr()[ro.first];
              break;
            }
          }
        }
      }
      total_mem_footprint = std::max(total_mem_footprint, og_mem_footprint);
      total_mem_footprint_str += og_mem_footprint_str + ", ";
    }
    mem_footprint = total_mem_footprint;
    mem_footprint_str = total_mem_footprint_str;
  }

  void buildOperatorGroup() {
    operator_groups.clear();
    std::set<TensorOperator *> og;
    og.insert(tree.getRoot()->getOperator());
    operator_groups.push_back(og);
    tree.getRoot()->getOperator()->setGroupInd(0);
    for (auto op_node : tree.getNodes()) {
      if (!op_node->isRoot()) {
        TensorOperator *op = op_node->getOperator();
        TensorOperator *op_p = op_node->getParent()->getOperator();
        if (op_p->isReduceDimExpended(op->getOutputTensor(0))) {
          operator_groups.at(op_p->getGroupInd()).insert(op);
          op->setGroupInd(op_p->getGroupInd());
        } else {
          std::set<TensorOperator *> new_og;
          new_og.insert(op);
          op->setGroupInd(operator_groups.size());
          operator_groups.push_back(new_og);
        }
      }
    }
  }

protected:
  std::set<TensorOperator *> operators;
  std::set<Dim *> dims;
  OperatorTree tree;
  std::set<Tensor *> tensors;
  std::set<Tensor *> internal_tensors;
  std::set<Tensor *> external_tensors;
  std::vector<std::set<TensorOperator *> > operator_groups;

  long mem_access_times;
  long mem_access_volume;
  long mem_footprint;
  std::string mem_footprint_str;
  long external_tensors_fuse_pattern{-1};
  std::map<Tensor *, bool> external_tensor_fuse_status;
};

}

#endif //MMCHAIN_ANALYSIS_SRC_OPERATOR_CHAIN_H
