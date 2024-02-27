//
// Created by Lei Xu on 2023/7/24.
//

#ifndef MMCHAIN_ANALYSIS_SRC_OPERATOR_TREE_H
#define MMCHAIN_ANALYSIS_SRC_OPERATOR_TREE_H

#include <queue>
#include <utility>
#include "tensor-operator.h"

namespace DAT {

struct OrderInfo {
  OrderInfo() = default;
  explicit OrderInfo(OperatorNode *op_node) : operator_node(op_node) {}

  OperatorNode *operator_node{};
  std::vector<Dim *> dims_order{};
  std::vector<Dim *> dims_order_constraint{};
  std::set<Dim *> free_dims{};

  int execute_rank{-1};
};

class OperatorNode {
public:
  explicit OperatorNode(TensorOperator *to) : tensor_operator(to) {
    to->setLinkedNode(this);
    order_info.free_dims = (tensor_operator->getDims());
  }
  [[nodiscard]] OperatorNode *getParent() const {
    return parent;
  }
  void setParent(OperatorNode *op_node) {
    parent = op_node;
  }
  [[nodiscard]] OperatorNode *getChild(int i) const {
    return children[i];
  }
  void addChild(OperatorNode *op_node) {
    children.push_back(op_node);
  }
  void setChild(int i, OperatorNode *op_node) {
    children[i] = op_node;
  }
  [[nodiscard]] std::vector<OperatorNode *> getChildren() const {
    return children;
  }
  bool isRoot() {
    return parent == nullptr;
  }
  bool isLeaf() {
    return children[0] == nullptr && children[1] == nullptr;
  }
  [[nodiscard]] int getExecuteRank() const {
    return order_info.execute_rank;
  }
  void setExecuteRank(int r) {
    order_info.execute_rank = r;
  }
  TensorOperator *getOperator() {
    return tensor_operator;
  }
  [[nodiscard]] std::set<Dim *> getFreeDims() const {
    return order_info.free_dims;
  }
  [[nodiscard]] std::vector<Dim *> getDimsOrderConstraint() const {
    return order_info.dims_order_constraint;
  }
  void setDimsOrderConstraint(const std::vector<Dim *> &doc) {
    order_info.dims_order_constraint = doc;
    resetFreeDims();
    for (auto d : doc) {
      order_info.free_dims.erase(d);
    }
  }
  void resetFreeDims() {
    order_info.free_dims = (tensor_operator->getDims());
  }
  void setChildrenDimsOrderConstraint() {
    for (auto child : children) {
      if (child) {
        std::vector<Dim *> dims_order_c;
        for (auto d : order_info.dims_order) {
          if (child->getOperator()->getDims().count(d)) {
            dims_order_c.push_back(d);
          }
        }
        child->setDimsOrderConstraint(dims_order_c);
      }
    }
  }
  void setParentDimsOrderConstraint() {
    if (parent) {
      std::vector<Dim *> dims_order_c;
      for (auto d : order_info.dims_order) {
        if (parent->getOperator()->getDims().count(d)) {
          dims_order_c.push_back(d);
        }
      }
      parent->setDimsOrderConstraint(dims_order_c);
    }
  }
  [[nodiscard]] std::vector<Dim *> getDimsOrder() const {
    return order_info.dims_order;
  }
  void setDimsOrder(std::vector<int> offsets) {
    assert(order_info.free_dims.size() == offsets.size());
    order_info.dims_order = order_info.dims_order_constraint;
    int i = 0;
    for (auto d : order_info.free_dims) {
      order_info.dims_order.insert(order_info.dims_order.begin() + offsets[i], d);
      i++;
    }
  }
  void setDimsOrder(const std::vector<Dim *> &dimo) {
    setDimsOrderConstraint(dimo);
    setDimsOrder(std::vector<int>());
  }
  void setDimsOrder(int offset) {
    assert(order_info.free_dims.size() == 1);
    order_info.dims_order = order_info.dims_order_constraint;
    for (auto d : order_info.free_dims) {
      order_info.dims_order.insert(order_info.dims_order.begin() + offset, d);
    }
  }
  OrderInfo getOrderInfo() {
    return order_info;
  }
  void setOrderInfo(OrderInfo o_info) {
    order_info = std::move(o_info);
  }
  void setDimsInfo(const OrderInfo& o_info) {
    order_info.dims_order = o_info.dims_order;
    order_info.dims_order_constraint = o_info.dims_order_constraint;
    order_info.free_dims = o_info.free_dims;
  }
  void setRankInfo(const OrderInfo& o_info) {
    order_info.execute_rank = o_info.execute_rank;
  }
  void clearParent() {
    parent = nullptr;
  }
  void clearChildren() {
    for (auto &c : children) {
      c = nullptr;
    }
    children.clear();
  }

private:
  TensorOperator *tensor_operator;
  OrderInfo order_info{this};
  OperatorNode *parent{nullptr};
  std::vector<OperatorNode *> children;
};

class OperatorTree {
public:
  [[nodiscard]] OperatorNode *getRoot() const {
    return root;
  }
  void setRoot(OperatorNode *r) {
    assert(r != nullptr);
    root = r;
  }
  void updateRoot() {
    while (root->getParent()) {
      root = root->getParent();
    }
  }
  void updateNodes() {
    if (!root->isRoot()) {
      updateRoot();
    }
    nodes = breadthFirstSort(root);
    root->getOperator();
  }
  [[nodiscard]] std::vector<OperatorNode *> getNodes() const {
    return nodes;
  }
  void setDimsOrder(const std::vector<Dim *> &init_dims_order_c,
                    std::vector<std::vector<int>> offsets) {
    assert(offsets.size() == nodes.size());
    root->setDimsOrderConstraint(init_dims_order_c);
    int i = 0;
    for (const auto &node : nodes) {
      node->setDimsOrder(offsets[i]);
      node->setChildrenDimsOrderConstraint();
      i++;
    }
    updateOrderInfos();
  }
  void setOrder(std::map<OperatorNode *, OrderInfo> o_infos) {
    assert(o_infos.size() >= nodes.size());
    for (const auto &node : nodes) {
      node->setOrderInfo(o_infos[node]);
    }
    order_infos = o_infos;
  }
  void setDimsInfos(std::map<OperatorNode *, OrderInfo> o_infos) {
    assert(o_infos.size() == nodes.size());
    for (const auto &node : nodes) {
      node->setDimsInfo(o_infos[node]);
    }
    for (const auto& o_i : o_infos) {
      order_infos[o_i.first].dims_order = o_i.second.dims_order;
      order_infos[o_i.first].dims_order_constraint = o_i.second.dims_order_constraint;
      order_infos[o_i.first].free_dims = o_i.second.free_dims;
    }
  }
  void setRanksInfos(std::map<OperatorNode *, OrderInfo> o_infos) {
    assert(o_infos.size() == nodes.size());
    for (const auto &node : nodes) {
      node->setRankInfo(o_infos[node]);
    }
    for (const auto& o_i : o_infos) {
      order_infos[o_i.first].execute_rank = o_i.second.execute_rank;
    }
  }
  std::map<OperatorNode *, OrderInfo> getOrderInfos() {
    return order_infos;
  }
  void setOrderInfos(std::map<OperatorNode *, OrderInfo> o_infos) {
    order_infos = std::move(o_infos);
  }

  static std::vector<OperatorNode *> breadthFirstSort(OperatorNode *r) {
    std::vector<OperatorNode *> result;
    if (r == nullptr) {
      return result;
    }
    std::queue<OperatorNode *> queue;
    queue.push(r);
    while (!queue.empty()) {
      OperatorNode *node = queue.front();
      queue.pop();
      result.push_back(node);
      for (const auto &child : node->getChildren()) {
        if (child) {
          queue.push(child);
        }
      }
    }
    return result;
  }

  std::string toStringExecuteOrder(const std::string &prefix, const OperatorNode *op_node, bool isFirst) {
    std::string ret;
    if (op_node != nullptr) {
      ret += prefix;
      ret += (isFirst ? "├──" : "└──");
      // print the value of the node
      ret += std::to_string(op_node->getExecuteRank());
      ret += "\n";
      // enter the next tree level - left and right branch
      size_t children_num = op_node->getChildren().size();
      if (children_num > 1)
        ret += toStringExecuteOrder(prefix + (isFirst ? "│   " : "    "), op_node->getChild(0), true);
      else if (children_num > 0)
        ret += toStringExecuteOrder(prefix + (isFirst ? "│   " : "    "), op_node->getChild(0), false);
      for (int i = 1; i < children_num; ++i) {
        ret += toStringExecuteOrder(prefix + (isFirst ? "│   " : "    "), op_node->getChild(i), false);
      }
    }
    return ret;
  }

  std::string toStringExecuteOrder(const OperatorNode *op_node) {
    return toStringExecuteOrder("", op_node, false);
  }

  std::string toStringExecuteOrder() {
    return toStringExecuteOrder(root);
  }

private:
  void updateOrderInfos() {
    order_infos.clear();
    for (const auto &node : nodes) {
      order_infos[node] = (node->getOrderInfo());
    }
  }

  OperatorNode *root{nullptr};
  std::vector<OperatorNode *> nodes;
  std::map<OperatorNode *, OrderInfo> order_infos;
};

}

#endif //MMCHAIN_ANALYSIS_SRC_OPERATOR_TREE_H
