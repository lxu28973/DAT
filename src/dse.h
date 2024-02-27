//
// Created by lxu on 23-6-22.
//

#ifndef MMCHAIN_ANALYSIS_SRC_DSE_H
#define MMCHAIN_ANALYSIS_SRC_DSE_H

#include <bitset>
#include <fstream>
#include "operator-chain.h"
#include "to-gurobi.h"
#include <boost/filesystem.hpp>
#include <utility>
#include "options.h"
#include <algorithm>
#include <random>
#include "defines.h"
#include "operator-tree.h"
#include "util.h"

namespace DAT {

// Initialize random number generator
auto rd = std::random_device{};
auto rng = std::default_random_engine{rd()};

void checkTree(const OperatorTree &t) {
  for (auto r : t.getRoot()->getChildren()) {
    r->getOperator();
  }
  for (auto n : t.getNodes()) {
    n->getOperator();
  }
}

class LogRecord {
public:
  long id = 0;
  // algorithm
  std::set<TensorOperator *> tensor_operators;
  std::map<std::string, long> dims_size;
  long seq_len = 0;
  long hid_size = 0;
  long head_num = 0;
  // constraint
  long mem_size = 0;

  // result
  long mem_access_volume = -1;
  long mem_footprint{};
  long compute_time = -1;

  virtual std::string title_line() = 0;
  virtual std::string content() = 0;

  long getTotalOps() {
    long ops = 0;
    for (auto to : tensor_operators) {
      ops += to->getOpsNum();
    }
    return ops;
  }

  void set_ndh(long n_size, long d_size, long h_size) {
    this->seq_len = n_size;
    this->hid_size = d_size;
    this->head_num = h_size;
  }

  void defineAlgroithm(const std::set<TensorOperator *> &tos) {
    tensor_operators = tos;
    for (auto to : tos) {
      for (auto d : to->getDims()) {
        dims_size[d->getName()] = d->getSize();
      }
    }
  }

  void writeToFile(const std::string &file_name) {
    std::ofstream outfile;
    bool file_exists = boost::filesystem::exists(file_name);
    outfile.open(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
    if (!file_exists) {
      outfile << title_line() << std::endl;
      outfile << content() << std::endl;
    } else {
      outfile << content() << std::endl;
    }
  }
};

class LogRecord1 : public LogRecord {
  std::string title_line() override {
    return "id,dim size,mem size,ops num,mem access volume";
  }
  std::string content() override {
    std::string str = std::to_string(id) + ",";
    for (const auto &s : dims_size) {
      str += "(" + s.first + ":" + std::to_string(s.second) + ")";
    }
    str += "," + std::to_string(mem_size) + "," + std::to_string(getTotalOps()) + ","
        + std::to_string(mem_access_volume);
    return str;
  }

};

class LogRecord2 : public LogRecord {
public:
  // best order
  // schedule
  long top_fuse_pattern = -1;
  size_t fuse_num = -1;
  std::map<std::string, bool> fuse_status;

  void recordTopFusePattern(long f) {
    top_fuse_pattern = f;
  }

  void recordFuseStatus(const std::set<Tensor *> &ts) {
    fuse_num = 0;
    size_t i = 0;
    for (auto t : ts) {
      fuse_status[t->getName()] = t->isFused();
      if (t->isFused()) fuse_num++;
      i++;
    }
  }

  std::string title_line() override {
    return "id,seq length,hid size,head num,dim size,mem size,top fuse pattern,fuse num,fuse status,mem footprint,ops num,mem access volume, compute time";
  }
  std::string content() override {
    std::string str =
        std::to_string(id) + "," + std::to_string(seq_len) + "," + std::to_string(hid_size) + ","
            + std::to_string(head_num) + ",";
    for (const auto &s : dims_size) {
      str += "(" + s.first + ":" + std::to_string(s.second) + ")";
    }
    str += "," + std::to_string(mem_size) + "," + std::to_string(top_fuse_pattern) + ","
        + std::to_string(fuse_num) + ",";
    for (const auto &fs : fuse_status) {
      str += "(" + fs.first + ":" + std::to_string(fs.second) + ")";
    }
    str += "," + std::to_string(mem_footprint) + "," + std::to_string(getTotalOps()) + ","
        + std::to_string(mem_access_volume) + "," + std::to_string(compute_time);
    return str;
  }

};

class LogRecord3 : public LogRecord {
public:
  // schedule
  std::vector<std::vector<Dim *> > dims_orders;
  std::map<std::string, bool> fuse_status;

  void recordFuseStatus(const std::set<Tensor *> &ts) {
    for (auto t : ts) {
      fuse_status[t->getName()] = t->isFused();
    }
  }

  void recordDimOrder(const std::vector<Dim *> &d_o) {
    dims_orders.push_back(d_o);
  }

  std::string title_line() override {
    return "id,dim size,mem size,extern fuse status,dim order,mem footprint,ops num,mem access volume, compute time";
  }
  std::string content() override {
    std::string str = std::to_string(id) + ",";
    for (const auto &s : dims_size) {
      str += "(" + s.first + ":" + std::to_string(s.second) + ")";
    }
    str += "," + std::to_string(mem_size) + ",";
    for (const auto &fs : fuse_status) {
      str += "(" + fs.first + ":" + std::to_string(fs.second) + ")";
    }
    str += ",";
    for (const auto &dims_order : dims_orders) {
      str += "[";
      for (auto d : dims_order) {
        str += "(" + d->getName() + ")";
      }
      str += "]";
    }
    str += "," + std::to_string(mem_footprint) + "," + std::to_string(getTotalOps()) + ","
        + std::to_string(mem_access_volume) + "," + std::to_string(compute_time);
    return str;
  }

};

class LogRecord4 : public LogRecord {
public:
  std::map<OperatorChain *, std::map<std::string, long> > dim_blocksize;
  size_t fuse_num = -1;
  std::map<std::string, bool> fuse_status;

  void recordFusePattern(const std::set<Tensor *> &ts) {
    fuse_num = 0;
    for (auto t : ts) {
      fuse_status[t->getName()] = t->isFused();
      if (t->isFused()) fuse_num++;
    }
  }
  void recordDimBlocksize(OperatorChain *oc) {
    std::map<std::string, long> blocksize;
    for (auto op : oc->getOperators())
      for (auto d : op->getDims()) {
        blocksize[op->getName() + "_" + d->getName()] = d->getBlockSize(op);
      }
    dim_blocksize[oc] = blocksize;
  }
  std::string title_line() override {
    return "id,seq length,hid size,head num,dim size,mem size,dim block size,fuse num,fuse status,mem footprint,ops num,mem access volume, compute time";
  }
  std::string content() override {
    std::string str =
        std::to_string(id) + "," + std::to_string(seq_len) + "," + std::to_string(hid_size) + ","
            + std::to_string(head_num) + ",";
    for (const auto &s : dims_size) {
      str += "(" + s.first + ":" + std::to_string(s.second) + ")";
    }
    str += "," + std::to_string(mem_size) + ",";
    for (const auto &to : dim_blocksize) {
      str += "[";
      for (const auto &ds : to.second) {
        str += "(" + ds.first + ":" + std::to_string(ds.second) + ")";
      }
      str += "]";
    }
    str += "," + std::to_string(fuse_num) + ",";
    for (const auto &fs : fuse_status) {
      str += "(" + fs.first + ":" + std::to_string(fs.second) + ")";
    }
    str += "," + std::to_string(mem_footprint) + "," + std::to_string(getTotalOps()) + ","
        + std::to_string(mem_access_volume) + "," + std::to_string(compute_time);
    return str;
  }
};

class LogRecord5 : public LogRecord {
public:
  std::string execute_tree;
  std::vector<int> execute_ranks;

  std::map<std::string, bool> fuse_status;

  void recordFuseStatus(const std::set<Tensor *> &ts) {
    for (auto t : ts) {
      fuse_status[t->getName()] = t->isFused();
    }
  }

  void setExecuteTree(std::string execute_order) {
    execute_tree = std::move(execute_order);
  }
  void recordExecuteOrder(const OperatorTree &op_tree) {
    execute_ranks.clear();
    for (auto op_node : op_tree.getNodes()) {
      execute_ranks.push_back(op_node->getExecuteRank());
    }
  }

  std::string title_line() override {
    return "id,dim size,mem size,extern fuse status,execute ranks,mem footprint,ops num,mem access volume, compute time";
  }
  std::string content() override {
    std::string str;
    str += std::to_string(id) + ",";
    for (const auto &s : dims_size) {
      str += "(" + s.first + ":" + std::to_string(s.second) + ")";
    }
    str += "," + std::to_string(mem_size) + ",";
    for (const auto &fs : fuse_status) {
      str += "(" + fs.first + ":" + std::to_string(fs.second) + ")";
    }
    str += ",";
    for (const auto &ex : execute_ranks) {
      str += "(";
      str += std::to_string(ex);
      str += ")";
    }
    str += "," + std::to_string(mem_footprint) + "," + std::to_string(getTotalOps()) + ","
        + std::to_string(mem_access_volume) + "," + std::to_string(compute_time);
    return str;
  }
  std::string contentWithTree() {
    std::string str = content();
    str += "\n" + execute_tree;
    return str;
  }
  void writeExecTree(const std::string &file_name) {
    std::ofstream outfile;
    bool file_exists = boost::filesystem::exists(file_name);
    outfile.open(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
    if (!file_exists) {
      outfile << title_line() << std::endl;
      outfile << contentWithTree() << std::endl;
    } else {
      outfile << contentWithTree() << std::endl;
    }
  }
};

class LogRecord6 : public LogRecord {
public:
  size_t fuse_num = -1;
  long fuse_pattern = -1;
  std::map<std::string, bool> fuse_status;
  size_t sub_fuse_num = -1;
  std::map<OperatorChain *, std::map<std::string, bool> > sub_fuse_status_map;
  std::map<OperatorChain *, std::vector<int> > execute_ranks_map;
  std::map<OperatorChain *, std::vector<std::map<OperatorNode*, std::vector<Dim *> > > > dims_orders_map;
  std::map<OperatorChain *, std::map<std::string, long> > dims_blocksizes_map;
  std::map<OperatorChain *, std::map<std::string, std::set<Dim *> > > dims_expand_map;
  std::map<OperatorChain *, std::map<std::string, std::set<std::string> > > operator_dims_expand_map;
  std::map<TensorOperator *, std::map<std::string, long> > operator_dims_blocksizes_map;
  std::map<OperatorChain *, long> mem_access_volume_map;

  void recordFusePattern(long f) {
    fuse_pattern = f;
  }

  void recordFuseStatus(const std::set<Tensor *> &ts) {
    fuse_num = 0;
    for (auto t : ts) {
      fuse_status[t->getName()] = t->isFused();
      if (t->isFused()) fuse_num++;
    }
  }
  void recordSubFuseStatus(OperatorChain *oc, const std::set<Tensor *> &ts) {
    std::map<std::string, bool> tmp_map;
    for (auto t : ts) {
      tmp_map[t->getName()] = t->isFused();
    }
    sub_fuse_status_map[oc] = (tmp_map);
    updateSubFuseNum();
  }
  void updateSubFuseNum() {
    sub_fuse_num = 0;
    for (const auto& map : sub_fuse_status_map){
      for (const auto& t : map.second) {
        if (t.second) sub_fuse_num++;
      }
    }
  }
  void recordDimBlocksizes(OperatorChain *oc) {
    std::map<std::string, long> blocksizes;
    std::map<std::string,  std::set<Dim *> > expands;
    std::map<std::string,  std::set<std::string> > operator_expands;
    for (auto op : oc->getOperators()) {
      for (auto d : op->getDims()) {
        blocksizes[op->getName() + "_" + d->getName()] = d->getBlockSize(op);
      }
      for (auto t : op->getTensors()) {
        expands[op->getName() + "_" + t->getName()] = t->getExpandDims();
        for (auto d : t->getExpandDims()) {
          if (op->getBatchDims().count(d)) {
            operator_expands[op->getName() + "_" + t->getName()].insert("B");
          }
          if (op->getReduceDims().count(d)) {
            operator_expands[op->getName() + "_" + t->getName()].insert("K");
          }
          if (op->getShapeDims(0).count(d)) {
            operator_expands[op->getName() + "_" + t->getName()].insert("M");
          }
          if (op->getShapeDims(1).count(d)) {
            operator_expands[op->getName() + "_" + t->getName()].insert("N");
          }
        }
      }
      std::map<std::string, long> operator_block_size;
      long batch_blocksize = 1;
      for (auto d : op->getBatchDims()) {
        batch_blocksize *= d->getBlockSize(op);
      }
      long reduce_blocksize = 1;
      for (auto d : op->getReduceDims()) {
        reduce_blocksize *= d->getBlockSize(op);
      }
      long shape_blocksize0 = 1;
      for (auto d : op->getShapeDims(0)) {
        shape_blocksize0 *= d->getBlockSize(op);
      }
      long shape_blocksize1 = 1;
      for (auto d : op->getShapeDims(1)) {
        shape_blocksize1 *= d->getBlockSize(op);
      }
      operator_dims_blocksizes_map[op] =
          {{"B", batch_blocksize}, {"K", reduce_blocksize}, {"M", shape_blocksize0},
           {"N", shape_blocksize1}};
    }
    dims_blocksizes_map[oc] = blocksizes;
    dims_expand_map[oc] = expands;
    operator_dims_expand_map[oc] = operator_expands;
  }
  void recordOrder(OperatorChain *oc, const std::map<OperatorNode *, OrderInfo> &o_infos) {
    execute_ranks_map[oc].clear();
    dims_orders_map[oc].clear();
    for (const auto &o_info : o_infos) {
      if (oc->hasOperator(o_info.first->getOperator())) {
        execute_ranks_map[oc].push_back(o_info.second.execute_rank);
        dims_orders_map[oc].push_back({{o_info.first,o_info.second.dims_order}});
      }
    }
  }
  void recordOpChainMemAccessVolume(OperatorChain *oc) {
    mem_access_volume_map[oc] = oc->getMemAccessVolume();
  }

  std::string title_line() override {
    return "id,seq length,hid size,head num,dim size,mem size,fuse pattern,fuse num,fuse status,sub fuse num,sub fuse statuses,execute ranks,dim orders,dim block sizes,expand dims,change dim,operator expand dims,operator station tensor,operator dim block sizes,mem footprint,ops num,op chain mem access volume,mem access volume, compute time";
  }
  std::string content() override {
    std::string str =
        std::to_string(id) + "," + std::to_string(seq_len) + "," + std::to_string(hid_size) + ","
            + std::to_string(head_num) + ",";
    for (const auto &s : dims_size) {
      str += "(" + s.first + ":" + std::to_string(s.second) + ")";
    }
    str += "," + std::to_string(mem_size) + ",";
    str += std::to_string(fuse_pattern) + ",";
    str += std::to_string(fuse_num) + ",";
    for (const auto &fs : fuse_status) {
      str += "(" + fs.first + ":" + std::to_string(fs.second) + ")";
    }
    str += ",";
    str += std::to_string(sub_fuse_num);
    str += ",";
    for (const auto &sfs : sub_fuse_status_map) {
      str += "[";
      for (const auto &fs : sfs.second) {
        str += "(" + fs.first + ":" + std::to_string(fs.second) + ")";
      }
      str += "]";
    }
    str += ",";
    for (const auto &exrs : execute_ranks_map) {
      str += "[";
      for (const auto &ex : exrs.second) {
        str += "(" + std::to_string(ex) + ")";
      }
      str += "]";
    }
    str += ",";
    for (const auto &dos : dims_orders_map) {
      str += "{";
      for (const auto &dims_order : dos.second) {
        str += "[";
        for (const auto& ds : dims_order) {
          for (const auto& d : ds.second) {
            str += "(" + d->getName() + ")";
          }
        }
        str += "]";
      }
      str += "}";
    }
    str += ",";
    for (const auto &to : dims_blocksizes_map) {
      str += "[";
      for (const auto &ds : to.second) {
        str += "(" + ds.first + ":" + std::to_string(ds.second) + ")";
      }
      str += "]";
    }
    str += ",";
    for (const auto &to : dims_expand_map) {
      str += "{";
      for (const auto &ds : to.second) {
        str += "[" + ds.first + ":";
        for (const auto &d : ds.second) {
          str += "(" + d->getName() + ")";
        }
        str += "]";
      }
      str += "}";
    }
    str += ",";
    for (const auto &dm : dims_orders_map) {
      for (auto to : dm.first->getOperators()) {
        str += "[" + to->getName();
        for (const auto &dims_order : dm.second) {
          for (const auto& ds : dims_order) {
            if (ds.first->getOperator() == to) {
              for (auto d : ds.second) {
                if (to->hasDim(d)) {
                  str += "(" + d->getName() + ")";
                  break;
                }
              }
            }
          }
        }
        str += "]";
      }
    }
    str += ",";
    for (const auto &to : operator_dims_expand_map) {
      str += "{";
      for (const auto &ds : to.second) {
        str += "[" + ds.first + ":";
        for (const auto &d : ds.second) {
          str += "(" + d + ")";
        }
        str += "]";
      }
      str += "}";
    }
    str += ",";
    for (const auto &dm : dims_orders_map) {
      for (auto to : dm.first->getOperators()) {
        str += "[" + to->getName();
        for (const auto &dims_order : dm.second) {
          for (const auto& ds : dims_order) {
            if (ds.first->getOperator() == to) {
              for (auto d : ds.second) {
                if (to->hasDim(d)) {
                  for (auto t : to->getTensors() ){
                    if (!t->hasDim(d)) {
                      str += "(" + t->getName() + ")";
                    }
                  }
                  break;
                }
              }
            }
          }
        }
        str += "]";
      }
    }
    str += ",";
    for (const auto &to : operator_dims_blocksizes_map) {
      str += "[" + to.first->getName();
      for (const auto &ds : to.second) {
        str += "(" + ds.first + ":" + std::to_string(ds.second) + ")";
      }
      str += "]";
    }
    str += "," + std::to_string(mem_footprint) + "," + std::to_string(getTotalOps()) + ",";
    for (const auto &mavolume : mem_access_volume_map) {
      str += "(" + std::to_string(mavolume.second) + ")";
    }
    str += "," + std::to_string(mem_access_volume) + "," + std::to_string(compute_time);
    return str;
  }

};

std::vector<long> findFactors(long number) {
  std::vector<long> factors;

  for (int i = 1; i <= number; ++i) {
    if (number % i == 0) {
      factors.push_back(i);
    }
  }

  return factors;
}

double optimizeBlockSize(OperatorChain *op_chain,
                         long mem_constraint,
                         bool print_info = false,
                         const std::string &lp_file = "") {
  bool traversal_batch_blocksize = false;
  bool traversal_head_blocksize = false;
  bool batch_independent = false;
  bool head_independent = false;
  if (op_chain->getDim("bsc")) {
    traversal_batch_blocksize = true;
    batch_independent = true;
    for (auto op : op_chain->getOperators()) {
      traversal_batch_blocksize &= op->isBatchDependent();
      batch_independent &= !op->isBatchDependent();
    }
  }
  if (op_chain->getDim("hsc")) {
    traversal_head_blocksize = true;
    head_independent = true;
    for (auto op : op_chain->getOperators()) {
      traversal_head_blocksize &= op->isBatchDependent();
      head_independent &= !op->isHeadDependent();
    }
  }
  double best_obj = MAXFLOAT;
  long best_batch = Options::batch_blocksize;
  long best_head = Options::head_blocksize;
  if (traversal_batch_blocksize) {
    std::vector<long> batch_blocksizes;
    batch_blocksizes = findFactors(op_chain->getDim("bsc")->getSize());
    for (auto bs : batch_blocksizes) {
      for (auto op : op_chain->getOperators()) {
        op->getDim("bsc")->setBlockSize(op, bs);
      }
      if (traversal_head_blocksize) {
        std::vector<long> head_blocksizes;
        head_blocksizes = findFactors(op_chain->getDim("hsc")->getSize());
        for (auto hs : head_blocksizes) {
          for (auto op : op_chain->getOperators()) {
            op->getDim("hsc")->setBlockSize(op, hs);
          }
          double obj = mipBlockSize(op_chain, mem_constraint, print_info, lp_file);
          if (obj < best_obj) {
            best_obj = obj;
            best_batch = bs;
            best_head = hs;
          }
        }
      } else {
        double obj = mipBlockSize(op_chain, mem_constraint, print_info, lp_file);
        if (obj < best_obj) {
          best_obj = obj;
          best_batch = bs;
        }
      }
    }
  }
  if (batch_independent) {
    best_batch = 1;
  }
  if (head_independent) {
    best_head = 1;
  }

  if (op_chain->getDim("bsc")) {
    for (auto op : op_chain->getOperators()) {
      op->getDim("bsc")->setBlockSize(op, best_batch);
    }
  }
  if (op_chain->getDim("hsc")) {
    for (auto op : op_chain->getOperators()) {
      op->getDim("hsc")->setBlockSize(op, best_head);
    }
  }

  return mipBlockSize(op_chain, mem_constraint, print_info, lp_file);
}

std::vector<Dim *> randomExchangeTwoDims(const std::vector<Dim *> &dims_order) {
  std::vector<Dim *> ret_dims_order(dims_order.begin(), dims_order.end());
  // Generate random indices
  std::uniform_int_distribution<size_t> distribution(0, ret_dims_order.size() - 1);
  size_t index1 = distribution(rng);
  size_t index2 = distribution(rng);
  // Ensure the two indices are different
  while (index1 == index2) {
    index2 = distribution(rng);
  }
  // Swap elements at the random indices
  std::swap(ret_dims_order[index1], ret_dims_order[index2]);
  return ret_dims_order;
}

template<typename T>
std::vector<int> findTopKIndices(const std::vector<T> &input, int k) {
  std::vector<int> indices(input.size());
  for (int i = 0; i < input.size(); ++i) {
    indices[i] = i; // Initialize the indices vector with 0 to N-1
  }
  // Custom comparator to compare elements based on their values
  auto compareElements = [&](int a, int b) {
    return input[a] < input[b];
  };
  // Partially sort the indices vector so that the kth element is in its correct position
  std::nth_element(indices.begin(), indices.begin() + k, indices.end(), compareElements);
  // Create a vector to store the top k indices
  std::vector<int> topIndices(indices.begin(), indices.begin() + k);
  return topIndices;
}

OperatorTree selectOneNodeAndChangeDimsOrder(OperatorTree op_tree, OperatorNode *start_node) {
  std::uniform_int_distribution<> dis;

  start_node->setDimsOrderConstraint(op_tree.getOrderInfos()[start_node].dims_order_constraint);
  const std::vector<OperatorNode *> op_nodes = OperatorTree::breadthFirstSort(start_node);
  std::vector<size_t> constraint_dims_num(op_nodes.size());
  std::vector<std::vector<int> > free_dims_offsets(op_nodes.size());
  // init free_dims_offsets and constraint_dims_num
  for (int ii = 0; ii < op_nodes.size(); ++ii) {
    constraint_dims_num[ii] = op_nodes[ii]->getDimsOrderConstraint().size();
    for (int iii = 0; iii < op_nodes[ii]->getFreeDims().size(); ++iii) {
      free_dims_offsets[ii].push_back(dis(rng) % (constraint_dims_num[ii] + iii + 1));
    }
    op_nodes[ii]->setDimsOrder(free_dims_offsets[ii]);
    op_nodes[ii]->setChildrenDimsOrderConstraint();
  }
  std::map<OperatorNode *, OrderInfo> new_orders = op_tree.getOrderInfos();
  for (auto node : op_nodes) {
    new_orders[node] = node->getOrderInfo();
  }
  op_tree.setDimsInfos(new_orders);

  return op_tree;

}

OperatorTree generateRandomTreeOrder(OperatorTree op_tree) {
  OperatorNode *op_root = op_tree.getRoot();
  std::set<Dim *> root_dims = op_root->getOperator()->getDims();
  std::vector<Dim *> root_dims_vec(root_dims.begin(), root_dims.end());
  root_dims_vec.pop_back();

  std::vector<Dim *> root_dims_order = root_dims_vec;
  std::shuffle(root_dims_order.begin(), root_dims_order.end(), rng);

  op_root->setDimsOrderConstraint(root_dims_order);
  op_tree = selectOneNodeAndChangeDimsOrder(op_tree, op_root);

  return op_tree;

}

// DO NOT USE THIS, because setOrder is also time-consuming
//bool isInferior(OperatorChain *op_chain, OperatorTree tree1, OperatorTree tree2) {
//  bool is_inferior = true;
//  std::string mem1, mem2;
//  std::string access1, access2;
//  op_chain->setOrder(tree1.getOrderInfos());
//  mem1 = op_chain->getMemFootprint();
//  access1 = op_chain->getMemAccessVolume();
//  op_chain->setOrder(tree2.getOrderInfos());
//  mem2 = op_chain->getMemFootprint();
//  access2 = op_chain->getMemAccessVolume();
//  is_inferior &= (mem1 >= mem2);
//  is_inferior &= (access1 >= access2);
//  return is_inferior;
//}

bool geneticDimOrder(OperatorChain *op_chain,
                     long mem_size,
                     double &best_obj,
                     OperatorTree &best_dims_order_tree,
                     size_t generations) {
  bool infeasible = true;
  std::uniform_int_distribution<> dis;

  OperatorTree op_tree = op_chain->getOperatorTree();

  // init
  const size_t k = 5;
  OperatorTree top_k_dims_orders[k];
  double top_k_objs[k] = {MAXFLOAT};
  for (int i = 0; i < k; ++i) {
    top_k_dims_orders[i] = generateRandomTreeOrder(op_tree);
    op_chain->setOrder(top_k_dims_orders[i].getOrderInfos());
    top_k_objs[i] = optimizeBlockSize(op_chain, mem_size);
    if (Options::save_log_file >= 3) {
      auto dims_orders_candidate = top_k_dims_orders[i];
      auto obj_candidate = top_k_objs[i];
      op_chain->setOrder(dims_orders_candidate.getOrderInfos());
      LogRecord3 log_record;
      log_record.id = 3;
      log_record.defineAlgroithm(op_chain->getOperators());
      log_record.mem_size = mem_size;
      log_record.recordFuseStatus(op_chain->getExternalTensors());
      for (auto op : dims_orders_candidate.getNodes()) {
        log_record.recordDimOrder(op->getDimsOrder());
      }
      log_record.mem_footprint = op_chain->getMemFootprint();
      if (obj_candidate == MAXFLOAT) {
        log_record.mem_access_volume = -1;
        log_record.compute_time = -1;
      } else {
        log_record.mem_access_volume = op_chain->getMemAccessVolume();
        log_record.compute_time = op_chain->compute_time();
      }
      log_record.writeToFile(Options::log_directory + "/log3.csv");
    }
  }
  // genetic
  for (int i = 0; i < generations; ++i) {
    // generate candidates
    std::vector<OperatorTree> dims_orders_candidates;
    std::vector<double> obj_candidates;
    for (int ii = 0; ii < k; ++ii) {
      dims_orders_candidates.push_back(top_k_dims_orders[ii]);
      obj_candidates.push_back(top_k_objs[ii]);
    }
    for (auto &top_k_dims_order : top_k_dims_orders) {
      OperatorTree order_candidate = generateRandomTreeOrder(top_k_dims_order);
//      if (isInferior(op_chain, order_candidate, top_k_dims_order)) {
//        obj_candidates.push_back(INFINITY);
//      } else {
      op_chain->setOrder(order_candidate.getOrderInfos());
      obj_candidates.push_back(optimizeBlockSize(op_chain, mem_size));
//      }
      dims_orders_candidates.push_back(order_candidate);
      if (Options::save_log_file >= 3) {
        auto dims_orders_candidate = dims_orders_candidates.back();
        auto obj_candidate = obj_candidates.back();
        op_chain->setOrder(dims_orders_candidate.getOrderInfos());
        LogRecord3 log_record;
        log_record.id = 3;
        log_record.defineAlgroithm(op_chain->getOperators());
        log_record.mem_size = mem_size;
        log_record.recordFuseStatus(op_chain->getExternalTensors());
        for (auto op : dims_orders_candidate.getNodes()) {
          log_record.recordDimOrder(op->getDimsOrder());
        }
        log_record.mem_footprint = op_chain->getMemFootprint();
        if (obj_candidate == MAXFLOAT) {
          log_record.mem_access_volume = -1;
          log_record.compute_time = -1;
        } else {
          log_record.mem_access_volume = op_chain->getMemAccessVolume();
          log_record.compute_time = op_chain->compute_time();
        }
        log_record.writeToFile(Options::log_directory + "/log3.csv");
      }
      for (int c = 0; c < 4; ++c) {
        int select_node_index = dis(rng) % (top_k_dims_order.getNodes().size());
        order_candidate = selectOneNodeAndChangeDimsOrder(top_k_dims_order,
                                                          top_k_dims_order.getNodes()[select_node_index]);
        op_chain->setOrder(order_candidate.getOrderInfos());
        obj_candidates.push_back(optimizeBlockSize(op_chain, mem_size));
        dims_orders_candidates.push_back(order_candidate);
        if (Options::save_log_file >= 3) {
          auto dims_orders_candidate = dims_orders_candidates.back();
          auto obj_candidate = obj_candidates.back();
          op_chain->setOrder(dims_orders_candidate.getOrderInfos());
          LogRecord3 log_record;
          log_record.id = 3;
          log_record.defineAlgroithm(op_chain->getOperators());
          log_record.mem_size = mem_size;
          log_record.recordFuseStatus(op_chain->getExternalTensors());
          for (auto op : dims_orders_candidate.getNodes()) {
            log_record.recordDimOrder(op->getDimsOrder());
          }
          log_record.mem_footprint = op_chain->getMemFootprint();
          if (obj_candidate == MAXFLOAT) {
            log_record.mem_access_volume = -1;
            log_record.compute_time = -1;
          } else {
            log_record.mem_access_volume = op_chain->getMemAccessVolume();
            log_record.compute_time = op_chain->compute_time();
          }
          log_record.writeToFile(Options::log_directory + "/log3.csv");
        }
      }
    }
    assert(dims_orders_candidates.size() == obj_candidates.size());
    // find the top k and update
    auto top_k_indices = findTopKIndices(obj_candidates, k);
    for (int ii = 0; ii < k; ++ii) {
      auto top_k_index = top_k_indices[ii];
      top_k_dims_orders[ii] = dims_orders_candidates[top_k_index];
      top_k_objs[ii] = obj_candidates[top_k_index];
    }
  }
  for (int i = 0; i < k; ++i) {
    if (top_k_objs[i] < best_obj) {
      best_obj = top_k_objs[i];
      best_dims_order_tree = top_k_dims_orders[i];
      infeasible = false;
    }
  }
  return infeasible;
}

bool randomDimOrder(OperatorChain *op_chain,
                    long mem_size,
                    double &best_obj,
                    OperatorTree &best_dims_order_tree,
                    size_t times) {
  bool infeasible = true;
  OperatorTree op_tree = op_chain->getOperatorTree();

  for (size_t n = 0; n < times; ++n) {
    op_tree = generateRandomTreeOrder(op_tree);
    op_chain->setOrder(op_tree.getOrderInfos());
    double obj = DAT::optimizeBlockSize(op_chain, mem_size);
    if (obj < best_obj) {
      best_obj = obj;
      best_dims_order_tree = op_tree;
      infeasible = false;
    }
    if (Options::save_log_file >= 3) {
      op_chain->setOrder(op_tree.getOrderInfos());
      LogRecord3 log_record;
      log_record.id = 3;
      log_record.defineAlgroithm(op_chain->getOperators());
      log_record.mem_size = mem_size;
      log_record.recordFuseStatus(op_chain->getExternalTensors());
      for (auto op : op_tree.getNodes()) {
        log_record.recordDimOrder(op->getDimsOrder());
      }
      log_record.mem_footprint = op_chain->getMemFootprint();
      if (obj == MAXFLOAT) {
        log_record.mem_access_volume = -1;
        log_record.compute_time = -1;
      } else {
        log_record.mem_access_volume = op_chain->getMemAccessVolume();
        log_record.compute_time = op_chain->compute_time();
      }
      log_record.writeToFile(Options::log_directory + "/log3.csv");
    }
  }

  return infeasible;
}

bool traversalDimOrder(OperatorChain *op_chain,
                       long mem_size,
                       double &best_obj,
                       std::vector<Dim *> &best_root_dims_order_constraint,
                       std::vector<std::vector<int> > &best_free_dims_offsets) {
  bool infeasible = true;
  const OperatorTree op_tree = op_chain->getOperatorTree();
  OperatorNode *op_root = op_tree.getRoot();

  std::set<Dim *> root_dims = op_root->getOperator()->getDims();
  size_t root_dims_num = root_dims.size();
  assert(root_dims_num >= 1);
  std::vector<Dim *> root_dims_vec(root_dims.begin(), root_dims.end());
  root_dims_vec.pop_back();
  std::vector<int> root_dim_order_indices(root_dims_num - 1, 0);
  // Use std::iota to fill the vector with continuous numbers starting from 0
  std::iota(root_dim_order_indices.begin(), root_dim_order_indices.end(), 0);

  do {
    std::vector<Dim *> root_dims_order_constraint;
    root_dims_order_constraint.reserve(root_dim_order_indices.size());
    for (auto ind : root_dim_order_indices) {
      root_dims_order_constraint.push_back(root_dims_vec.at(ind));
    }
    op_root->setDimsOrderConstraint(root_dims_order_constraint);

    const std::vector<OperatorNode *> op_nodes = op_tree.getNodes();
    size_t op_nodes_size = op_nodes.size();
    std::vector<size_t> constraint_dims_num(op_nodes_size, 0);
    std::vector<std::vector<int> > free_dims_offsets(op_nodes_size);
    // init free_dims_offsets and constraint_dims_num
    for (int i = 0; i < op_nodes_size; ++i) {
      constraint_dims_num.at(i) = op_nodes.at(i)->getDimsOrderConstraint().size();
      free_dims_offsets.at(i).clear();
      for (auto d : op_nodes.at(i)->getFreeDims()) {
        free_dims_offsets.at(i).push_back(0);
      }
      op_nodes.at(i)->setDimsOrder(free_dims_offsets.at(i));
      op_nodes.at(i)->setChildrenDimsOrderConstraint();
    }

    int nodes_index = 0;
    std::vector<int> offsets_index(op_nodes_size + 1, 0);
    while (nodes_index >= 0 && offsets_index.at(nodes_index) >= 0) {
      if (nodes_index < op_nodes_size) {
        if (offsets_index.at(nodes_index) < op_nodes.at(nodes_index)->getFreeDims().size()) {
          offsets_index.at(nodes_index)++;
        } else {
          nodes_index++;
        }
      } else {
        op_chain->setDimsOrder(root_dims_order_constraint, free_dims_offsets);
        double obj = DAT::optimizeBlockSize(op_chain, mem_size);
        if (obj < best_obj) {
          best_obj = obj;
          best_root_dims_order_constraint = root_dims_order_constraint;
          best_free_dims_offsets = free_dims_offsets;
          infeasible = false;
        }
        if (Options::save_log_file >= 3) {
          op_chain->setDimsOrder(root_dims_order_constraint, free_dims_offsets);
          LogRecord3 log_record;
          log_record.id = 3;
          log_record.defineAlgroithm(op_chain->getOperators());
          log_record.mem_size = mem_size;
          log_record.recordFuseStatus(op_chain->getExternalTensors());
          for (auto op : op_nodes) {
            log_record.recordDimOrder(op->getDimsOrder());
          }
          log_record.mem_footprint = op_chain->getMemFootprint();
          if (obj == MAXFLOAT) {
            log_record.mem_access_volume = -1;
            log_record.compute_time = -1;
          } else {
            log_record.mem_access_volume = op_chain->getMemAccessVolume();
            log_record.compute_time = op_chain->compute_time();
          }
          log_record.writeToFile(Options::log_directory + "/log3.csv");
        }

        nodes_index--;
        offsets_index.at(nodes_index)--;
        while (nodes_index >= 0
            && free_dims_offsets.at(nodes_index).at(offsets_index.at(nodes_index))
                >= constraint_dims_num.at(nodes_index) + offsets_index.at(nodes_index)) {
          if (offsets_index.at(nodes_index) > 0) {
            free_dims_offsets.at(nodes_index).at(offsets_index.at(nodes_index)) = 0;
            offsets_index.at(nodes_index)--;
          } else {
            free_dims_offsets.at(nodes_index).at(offsets_index.at(nodes_index)) = 0;
            nodes_index--;
            if (nodes_index >= 0)
              offsets_index.at(nodes_index)--;
          }
        }
        if (nodes_index >= 0)
          free_dims_offsets.at(nodes_index).at(offsets_index.at(nodes_index)) += 1;
      }
    }
  } while (std::next_permutation(root_dim_order_indices.begin(), root_dim_order_indices.end()));

  return infeasible;
}

bool traversalDimOrder(OperatorChain *op_chain,
                       long mem_size,
                       double &best_obj,
                       OperatorTree &best_dims_order_tree) {
  std::vector<DAT::Dim *> best_root_dims_order_constraint;
  std::vector<std::vector<int> > best_free_dims_offsets;
  bool infeasible = traversalDimOrder(op_chain, mem_size, best_obj, best_root_dims_order_constraint,
                                      best_free_dims_offsets);
  if (!infeasible) {
    op_chain->setDimsOrder(best_root_dims_order_constraint, best_free_dims_offsets);
    best_dims_order_tree = op_chain->getOperatorTree();
  }

  return infeasible;
}

bool randomExecuteOrder(OperatorChain *op_chain,
                            long mem_size,
                            double &best_obj,
                            OperatorTree &best_order_tree,
                            OperatorTree op_tree,
                            const std::set<OperatorNode *> &current_options,
                            int rank) {
  bool infeasible = true;
  std::set<OperatorNode *> new_options = current_options;
  while (!new_options.empty()) {
    std::uniform_int_distribution<> dist(0, new_options.size() - 1);
    int randomIndex = dist(rng);
    auto it = new_options.begin();
    std::advance(it, randomIndex);
    auto select_node = *it;
    auto order_info = op_tree.getOrderInfos();
    order_info[select_node].execute_rank = rank;
    op_tree.setOrderInfos(order_info);
    new_options.erase(select_node);
    for (auto c : select_node->getChildren()) {
      if (c) {
        new_options.insert(c);
      }
    }
  }
    {
      bool sub_infeasible = true;
      double sub_best_obj = MAXFLOAT;
      OperatorTree best_dims_order_tree;
      op_chain->setOrder(op_tree.getOrderInfos());
          sub_infeasible =
              DAT::randomDimOrder(op_chain, mem_size, sub_best_obj, best_dims_order_tree, 1);
      if (Options::save_log_file >= 3) {
        LogRecord5 log_record;
        log_record.id = 5;
        log_record.defineAlgroithm(op_chain->getOperators());
        log_record.recordFuseStatus(op_chain->getExternalTensors());
        log_record.recordExecuteOrder(op_tree);
        log_record.setExecuteTree(op_tree.toStringExecuteOrder());
        log_record.mem_size = mem_size;
        if (sub_infeasible) {
          log_record.mem_access_volume = -1;
          log_record.compute_time = -1;
        } else {
          op_chain->setOrder(best_dims_order_tree.getOrderInfos());
          DAT::optimizeBlockSize(op_chain, mem_size);
          op_chain->setOrder(best_dims_order_tree.getOrderInfos());
          log_record.mem_access_volume = op_chain->getMemAccessVolume();
          log_record.compute_time = op_chain->compute_time();
          log_record.mem_footprint = op_chain->getMemFootprint();
        }
        log_record.writeToFile(Options::log_directory + "/log3exec.csv");
        log_record.writeExecTree(Options::log_directory + "/log3tree.csv");
      }
      if (!sub_infeasible) {
        infeasible = false;
        if (sub_best_obj < best_obj) {
          best_obj = sub_best_obj;
          best_order_tree = best_dims_order_tree;
        }
      }
    }
  return infeasible;
}

bool travsersalExecuteOrder(OperatorChain *op_chain,
                            long mem_size,
                            double &best_obj,
                            OperatorTree &best_order_tree,
                            OperatorTree op_tree,
                            const std::set<OperatorNode *> &current_options,
                            int rank) {
  bool infeasible = true;
  for (auto select_node : current_options) {
    auto order_info = op_tree.getOrderInfos();
    order_info[select_node].execute_rank = rank;
    op_tree.setOrderInfos(order_info);
    std::set<OperatorNode *> new_options = current_options;
    new_options.erase(select_node);
    for (auto c : select_node->getChildren()) {
      if (c) {
        new_options.insert(c);
      }
    }
    if (!new_options.empty()) {
      infeasible = travsersalExecuteOrder(op_chain, mem_size, best_obj, best_order_tree,
                                          op_tree, new_options, rank + 1) && infeasible;
    } else {
      bool sub_infeasible = true;
      double sub_best_obj = MAXFLOAT;
      OperatorTree best_dims_order_tree;
      op_chain->setOrder(op_tree.getOrderInfos());
      if (op_chain->getDims().size() <= 7) {
        sub_infeasible = traversalDimOrder(op_chain, mem_size, sub_best_obj, best_dims_order_tree);
      } else {
        if (Options::dim_order_opt == "traversal") {
          sub_infeasible =
              traversalDimOrder(op_chain, mem_size, sub_best_obj, best_dims_order_tree);
        } else if (DAT::Options::dim_order_opt == "random") {
          sub_infeasible =
              DAT::randomDimOrder(op_chain, mem_size, sub_best_obj, best_dims_order_tree, 6000);
        } else if (DAT::Options::dim_order_opt == "genetic") {
          sub_infeasible =
              DAT::geneticDimOrder(op_chain, mem_size, sub_best_obj, best_dims_order_tree, 200);
        } else {
          std::cout << "warning: dim_order_opt is unrecognizable, using traversal instead"
                    << std::endl;
          sub_infeasible =
              traversalDimOrder(op_chain, mem_size, sub_best_obj, best_dims_order_tree);
        }
      }
      if (Options::save_log_file >= 3) {
        LogRecord5 log_record;
        log_record.id = 5;
        log_record.defineAlgroithm(op_chain->getOperators());
        log_record.recordFuseStatus(op_chain->getExternalTensors());
        log_record.recordExecuteOrder(op_tree);
        log_record.setExecuteTree(op_tree.toStringExecuteOrder());
        log_record.mem_size = mem_size;
        if (sub_infeasible) {
          log_record.mem_access_volume = -1;
          log_record.compute_time = -1;
        } else {
          op_chain->setOrder(best_dims_order_tree.getOrderInfos());
          DAT::optimizeBlockSize(op_chain, mem_size);
          op_chain->setOrder(best_dims_order_tree.getOrderInfos());
//          assert(op_chain->getMemAccessVolume() == std::round(sub_best_obj) || printAndReturnFalse(
//              std::to_string(op_chain->getMemAccessVolume()) + " "
//                  + std::to_string(std::round(sub_best_obj))));
          log_record.mem_access_volume = op_chain->getMemAccessVolume();
          log_record.compute_time = op_chain->compute_time();
          log_record.mem_footprint = op_chain->getMemFootprint();
        }
        log_record.writeToFile(Options::log_directory + "/log3exec.csv");
        log_record.writeExecTree(Options::log_directory + "/log3tree.csv");
      }
      if (!sub_infeasible) {
        infeasible = false;
        if (sub_best_obj < best_obj) {
          best_obj = sub_best_obj;
          best_order_tree = best_dims_order_tree;
        }
      }
    }
  }
  return infeasible;
}

bool randomOrder(OperatorChain *op_chain,
                   long mem_size,
                   double &best_obj,
                   OperatorTree &best_order_tree) {
  bool infeasible = true;
  OperatorTree op_tree = op_chain->getOperatorTree();
  std::set<OperatorNode *> init_option;
  init_option.insert(op_tree.getRoot());
  infeasible = randomExecuteOrder(op_chain, mem_size, best_obj, best_order_tree, op_tree,
                                      init_option, 0);
  if (!infeasible)
    op_chain->setOrder(best_order_tree.getOrderInfos());
  return infeasible;
}

bool optimizeOrder(OperatorChain *op_chain,
                   long mem_size,
                   double &best_obj,
                   OperatorTree &best_order_tree) {
  bool infeasible = true;
  OperatorTree op_tree = op_chain->getOperatorTree();
  std::set<OperatorNode *> init_option;
  init_option.insert(op_tree.getRoot());
  infeasible = travsersalExecuteOrder(op_chain, mem_size, best_obj, best_order_tree, op_tree,
                                      init_option, 0);
  if (!infeasible)
    op_chain->setOrder(best_order_tree.getOrderInfos());
  return infeasible;
}

long createToOperatorChain(OperatorChain *op_chain[],
                           std::set<DAT::TensorOperator *> non_add_to_operator_chain,
                           const std::set<DAT::Tensor *> &tensors) {

  std::set<DAT::TensorOperator *> add_to_operator_chain;
  long operator_chain_num = 0;
  for (auto t : tensors) {
    if ((!t->isIO()) && t->isFused()) {
      for (auto to : non_add_to_operator_chain) {
        if (to->hasTensor(t)) {
          bool add_to_exist_operator_chain = false;
          for (long n = 0; n < operator_chain_num; ++n) {
            if (op_chain[n]->hasTensor(t)) {
              add_to_exist_operator_chain = true;
              op_chain[n]->addOperator(to);
            }
          }
          if (!add_to_exist_operator_chain) {
            op_chain[operator_chain_num] = new DAT::OperatorChain;
            op_chain[operator_chain_num]->addOperator(to);
            operator_chain_num++;
          }
          add_to_operator_chain.insert(to);
        }
      }
      for (auto rop : add_to_operator_chain) {
        non_add_to_operator_chain.erase(rop);
      }
    }
  }
  for (auto to : non_add_to_operator_chain) {
    op_chain[operator_chain_num] = new DAT::OperatorChain;
    op_chain[operator_chain_num]->addOperator(to);
    operator_chain_num++;
  }

  for (auto t : tensors) {
    if ((!t->isIO()) && t->isFused()) {
      for (long n = 0; n < operator_chain_num; ++n) {
        if (op_chain[n]->hasTensor(t)) {
          op_chain[n]->addInternalTensor(t);
        }
      }
    } else {
      for (long n = 0; n < operator_chain_num; ++n) {
        if (op_chain[n]->hasTensor(t)) {
          op_chain[n]->addExternalTensor(t);
        }
      }
    }
  }

  updateOperatorTreeRelationship(tensors);
  for (long n = 0; n < operator_chain_num; ++n) {
    op_chain[n]->updateTree();
  }

  return operator_chain_num;

}

long randomFused(const std::set<DAT::TensorOperator *> &non_add_to_operator_chain,
                 const std::set<DAT::Tensor *> &tensors,
                 long mem_size,
                 DAT::OperatorChain *best_op_chain[],
                 bool print_log = false) {
  long best_operator_chain_num = 0;
  long best_fuse_pattern = 0;
  long best_total_access_volume = LONG_MAX;

  std::set<Tensor *> non_io_tensors;
  for (auto t : tensors) {
    if (!t->isIO()) {
      non_io_tensors.insert(t);
    } else {
      t->unsetFuse();
    }
  }
  long situation_num = pow(2, non_io_tensors.size());

  std::uniform_int_distribution<std::mt19937::result_type> dist_f(0, situation_num);
  {
  long f = dist_f(rng);
  assert(tensors.size() <= MAX_TENSOR_NUM);
  DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
  std::bitset<MAX_TENSOR_NUM> fuse_or_not(f);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  LogRecord6 log_record;
  if (Options::save_log_file >= 2) {
    log_record.id = 2;
    log_record.defineAlgroithm(non_add_to_operator_chain);
    log_record.mem_size = mem_size;
    log_record.recordFusePattern(f);
    log_record.recordFuseStatus(non_io_tensors);
  }
  if (print_log)
    std::cout << "----------------situation " << f << "-----------------" << std::endl;
  long operator_chain_num =
      DAT::createToOperatorChain(op_chain, non_add_to_operator_chain, tensors);
  long total_access_volume = 0;
  long mem_footprint = 0;
  long infeasible = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    const size_t dims_num = mul_chain->getDims().size();
    const std::set<Tensor *> external_tensors = mul_chain->getExternalTensors();
    const size_t external_tensors_num = external_tensors.size();
    long sub_best_fuse_pattern = 0;
    OperatorTree sub_best_order_tree = mul_chain->getOperatorTree();
    long sub_best_access_volume = LONG_MAX;
    long sub_mem_footprint = 0;
    long sub_situation_num = pow(2, external_tensors_num);
    bool sub_infeasible = true;
    std::uniform_int_distribution<std::mt19937::result_type> dist_sf(0, sub_situation_num);
    long sf = dist_sf(rng);
    mul_chain->setExternalTensorsFusePattern(sf);
    LogRecord2 sub_log_record;
    if (Options::save_log_file >= 2) {
      sub_log_record.id = 2;
      sub_log_record.defineAlgroithm(mul_chain->getOperators());
      sub_log_record.mem_size = mem_size;
      sub_log_record.recordTopFusePattern(f);
      sub_log_record.recordFuseStatus(mul_chain->getExternalTensors());
    }
    if (print_log)
      std::cout << "----------------sub situation " << sf << "-----------------" << std::endl;
    double best_obj = MAXFLOAT;
    OperatorTree best_op_tree = mul_chain->getOperatorTree();
    bool sf_infeasible = true;
    sf_infeasible = randomOrder(mul_chain, mem_size, best_obj, best_op_tree);

    if (!sf_infeasible) {
      sub_infeasible = false;
      mul_chain->setOrder(best_op_tree.getOrderInfos());
      DAT::optimizeBlockSize(mul_chain, mem_size);
      mul_chain->setOrder(best_op_tree.getOrderInfos());
      long mem_access_volume = mul_chain->getMemAccessVolume();
      if (mem_access_volume < sub_best_access_volume) {
        sub_best_fuse_pattern = sf;
        sub_best_order_tree = best_op_tree;
        sub_best_access_volume = mem_access_volume;
        sub_mem_footprint = mul_chain->getMemFootprint();
      }
      if (Options::save_log_file >= 2) {
        sub_log_record.mem_access_volume = mem_access_volume;
        sub_log_record.compute_time = mul_chain->compute_time();
        sub_log_record.mem_footprint = mul_chain->getMemFootprint();
      }
    } else {
      if (Options::save_log_file >= 2) {
        sub_log_record.mem_access_volume = -1;
        sub_log_record.compute_time = -1;
      }
    }
    if (Options::save_log_file >= 2) {
      sub_log_record.writeToFile(Options::log_directory + "/log2s.csv");
    }
    if (!sub_infeasible) {
      if (print_log) {
        std::cout << mul_chain->toString();
        std::cout << "access volume: " << sub_best_access_volume << std::endl;
        std::cout << "SRAM footprint: " << sub_mem_footprint << std::endl;
      }
      total_access_volume += sub_best_access_volume;
      mem_footprint = std::max(mem_footprint, sub_mem_footprint);
      mul_chain->recordExternalTensorsFuseInfo(sub_best_fuse_pattern);
      mul_chain->recordOrder(sub_best_order_tree.getOrderInfos());
    } else {
      if (print_log) {
        std::cerr << "operators: " << mul_chain->toString() << std::endl;
      }
    }
    infeasible += sub_infeasible;
  }
  if (!infeasible) {
    if (total_access_volume < best_total_access_volume) {
      for (long i = 0; i < best_operator_chain_num; ++i) {
        delete best_op_chain[i];
        best_op_chain[i] = nullptr;
      }
      best_total_access_volume = total_access_volume;
      best_operator_chain_num = operator_chain_num;
      best_fuse_pattern = f;
      for (long i = 0; i < operator_chain_num; ++i) {
        best_op_chain[i] = new OperatorChain;
        *(best_op_chain[i]) = *(op_chain[i]);
      }
    }
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = total_access_volume;
      log_record.mem_footprint = mem_footprint;
    }
    if (print_log) {
      std::cout << "total access volume: " << total_access_volume << std::endl;
      std::cout << "best total access volume: " << best_total_access_volume << std::endl;
    }
  } else {
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = -1;
    }
    if (print_log)
      std::cerr << "Can not use this fuse pattern with mem_size: " << mem_size << std::endl;
  }

  if (Options::save_log_file >= 2) {
    long total_compute_time = 0;
    for (long i = 0; i < operator_chain_num; ++i) {
      op_chain[i]->setInternalTensorsFuse();
      op_chain[i]->setExternalTensorsFusePattern(op_chain[i]->getExternalTensorsFusePattern());
      op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
      DAT::optimizeBlockSize(op_chain[i], mem_size);
      op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordSubFuseStatus(op_chain[i], op_chain[i]->getExternalTensors());
      log_record.recordOrder(op_chain[i], op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordDimBlocksizes(op_chain[i]);
      log_record.recordOpChainMemAccessVolume(op_chain[i]);
      total_compute_time += op_chain[i]->compute_time();
    }
    log_record.compute_time = total_compute_time;
    log_record.writeToFile(Options::log_directory + "/log2.csv");
  }
  for (long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
    op_chain[i] = nullptr;
  }

}

  std::bitset<MAX_TENSOR_NUM> fuse_or_not(best_fuse_pattern);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  return best_operator_chain_num;

}

long traversalFused(const std::set<DAT::TensorOperator *> &non_add_to_operator_chain,
                    const std::set<DAT::Tensor *> &tensors,
                    long mem_size,
                    DAT::OperatorChain *best_op_chain[],
                    bool print_log = false) {
  long best_operator_chain_num = 0;
  long best_fuse_pattern = 0;
  long best_total_access_volume = LONG_MAX;

  std::set<Tensor *> non_io_tensors;
  for (auto t : tensors) {
    if (!t->isIO()) {
      non_io_tensors.insert(t);
    } else {
      t->unsetFuse();
    }
  }
  long situation_num = pow(2, non_io_tensors.size());

  for (long f = 0; f < situation_num; f++) {
    assert(tensors.size() <= MAX_TENSOR_NUM);
    DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
    std::bitset<MAX_TENSOR_NUM> fuse_or_not(f);
    long f_i = 0;
    for (auto t : non_io_tensors) {
      if (fuse_or_not[f_i]) {
        t->setFuse();
      } else {
        t->unsetFuse();
      }
      ++f_i;
    }
    LogRecord6 log_record;
    if (Options::save_log_file >= 2) {
      log_record.id = 2;
      log_record.defineAlgroithm(non_add_to_operator_chain);
      log_record.mem_size = mem_size;
      log_record.recordFusePattern(f);
      log_record.recordFuseStatus(non_io_tensors);
    }
    if (print_log)
      std::cout << "----------------situation " << f << "-----------------" << std::endl;
    long operator_chain_num =
        DAT::createToOperatorChain(op_chain, non_add_to_operator_chain, tensors);
    long total_access_volume = 0;
    long mem_footprint = 0;
    long infeasible = 0;
    for (long i = 0; i < operator_chain_num; ++i) {
      DAT::OperatorChain *mul_chain = op_chain[i];
      const size_t dims_num = mul_chain->getDims().size();
      const std::set<Tensor *> external_tensors = mul_chain->getExternalTensors();
      const size_t external_tensors_num = external_tensors.size();
      long sub_best_fuse_pattern = 0;
      OperatorTree sub_best_order_tree = mul_chain->getOperatorTree();
      long sub_best_access_volume = LONG_MAX;
      long sub_mem_footprint = 0;
      long sub_situation_num = pow(2, external_tensors_num);
      bool sub_infeasible = true;
      for (long sf = 0; sf < sub_situation_num; sf++) {
        mul_chain->setExternalTensorsFusePattern(sf);
        LogRecord2 sub_log_record;
        if (Options::save_log_file >= 2) {
          sub_log_record.id = 2;
          sub_log_record.defineAlgroithm(mul_chain->getOperators());
          sub_log_record.mem_size = mem_size;
          sub_log_record.recordTopFusePattern(f);
          sub_log_record.recordFuseStatus(mul_chain->getExternalTensors());
        }
        if (print_log)
          std::cout << "----------------sub situation " << sf << "-----------------" << std::endl;
        double best_obj = MAXFLOAT;
        OperatorTree best_op_tree = mul_chain->getOperatorTree();
        bool sf_infeasible = true;
        sf_infeasible = optimizeOrder(mul_chain, mem_size, best_obj, best_op_tree);
        if (!sf_infeasible) {
          sub_infeasible = false;
          mul_chain->setOrder(best_op_tree.getOrderInfos());
          DAT::optimizeBlockSize(mul_chain, mem_size);
          mul_chain->setOrder(best_op_tree.getOrderInfos());
          long mem_access_volume = mul_chain->getMemAccessVolume();
          if (mem_access_volume < sub_best_access_volume) {
            sub_best_fuse_pattern = sf;
            sub_best_order_tree = best_op_tree;
            sub_best_access_volume = mem_access_volume;
            sub_mem_footprint = mul_chain->getMemFootprint();
          }
          if (Options::save_log_file >= 2) {
            sub_log_record.mem_access_volume = mem_access_volume;
            sub_log_record.compute_time = mul_chain->compute_time();
            sub_log_record.mem_footprint = mul_chain->getMemFootprint();
          }
        } else {
          if (Options::save_log_file >= 2) {
            sub_log_record.mem_access_volume = -1;
            sub_log_record.compute_time = -1;
          }
        }
        if (Options::save_log_file >= 2) {
          sub_log_record.writeToFile(Options::log_directory + "/log2s.csv");
        }
      }
      if (!sub_infeasible) {
        if (print_log) {
          std::cout << mul_chain->toString();
          std::cout << "access volume: " << sub_best_access_volume << std::endl;
          std::cout << "SRAM footprint: " << sub_mem_footprint << std::endl;
        }
        total_access_volume += sub_best_access_volume;
        mem_footprint = std::max(mem_footprint, sub_mem_footprint);
        mul_chain->recordExternalTensorsFuseInfo(sub_best_fuse_pattern);
        mul_chain->recordOrder(sub_best_order_tree.getOrderInfos());
      } else {
        if (print_log) {
          std::cerr << "operators: " << mul_chain->toString() << std::endl;
        }
      }
      infeasible += sub_infeasible;
    }
    if (!infeasible) {
      if (total_access_volume < best_total_access_volume) {
        for (long i = 0; i < best_operator_chain_num; ++i) {
          delete best_op_chain[i];
          best_op_chain[i] = nullptr;
        }
        best_total_access_volume = total_access_volume;
        best_operator_chain_num = operator_chain_num;
        best_fuse_pattern = f;
        for (long i = 0; i < operator_chain_num; ++i) {
          best_op_chain[i] = new OperatorChain;
          *(best_op_chain[i]) = *(op_chain[i]);
        }
      }
      if (Options::save_log_file >= 2) {
        log_record.mem_access_volume = total_access_volume;
        log_record.mem_footprint = mem_footprint;
      }
      if (print_log) {
        std::cout << "total access volume: " << total_access_volume << std::endl;
        std::cout << "best total access volume: " << best_total_access_volume << std::endl;
      }
    } else {
      if (Options::save_log_file >= 2) {
        log_record.mem_access_volume = -1;
      }
      if (print_log)
        std::cerr << "Can not use this fuse pattern with mem_size: " << mem_size << std::endl;
    }

    if (Options::save_log_file >= 2) {
      long total_compute_time = 0;
      for (long i = 0; i < operator_chain_num; ++i) {
        op_chain[i]->setInternalTensorsFuse();
        op_chain[i]->setExternalTensorsFusePattern(op_chain[i]->getExternalTensorsFusePattern());
        op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
        DAT::optimizeBlockSize(op_chain[i], mem_size);
        op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
        log_record.recordSubFuseStatus(op_chain[i], op_chain[i]->getExternalTensors());
        log_record.recordOrder(op_chain[i], op_chain[i]->getOperatorTree().getOrderInfos());
        log_record.recordDimBlocksizes(op_chain[i]);
        log_record.recordOpChainMemAccessVolume(op_chain[i]);
        total_compute_time += op_chain[i]->compute_time();
      }
      log_record.compute_time = total_compute_time;
      log_record.writeToFile(Options::log_directory + "/log2.csv");
    }
    for (long i = 0; i < operator_chain_num; ++i) {
      delete op_chain[i];
      op_chain[i] = nullptr;
    }
  }

  std::bitset<MAX_TENSOR_NUM> fuse_or_not(best_fuse_pattern);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  return best_operator_chain_num;
}

long flatDSE(const std::set<DAT::TensorOperator *> &non_add_to_operator_chain,
             const std::set<DAT::Tensor *> &tensors,
             long mem_size,
             DAT::OperatorChain *best_op_chain[],
             std::map<OperatorNode *, OrderInfo> o_infos,
             bool print_log = false) {
  long best_operator_chain_num = 0;
  long best_total_access_volume = LONG_MAX;

  std::set<Tensor *> non_io_tensors;
  for (auto t : tensors) {
    if (!t->isIO()) {
      non_io_tensors.insert(t);
    } else {
      t->unsetFuse();
    }
  }
  long situation_num = pow(2, non_io_tensors.size());

  std::bitset<MAX_TENSOR_NUM> bit_tmp;
  long tmp = 0;
  for (auto t : non_io_tensors) {
    if (t->isFused()) {
      bit_tmp[tmp] = true;
    } else {
      bit_tmp[tmp] = false;
    }
    ++tmp;
  }

  assert(tensors.size() <= MAX_TENSOR_NUM);
  DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
  LogRecord6 log_record;
  if (Options::save_log_file >= 2) {
    log_record.id = 2;
    log_record.defineAlgroithm(non_add_to_operator_chain);
    log_record.mem_size = mem_size;
    log_record.recordFuseStatus(non_io_tensors);
  }
  if (print_log)
    std::cout << "----------------situation " << -1 << "-----------------" << std::endl;
  long operator_chain_num =
      DAT::createToOperatorChain(op_chain, non_add_to_operator_chain, tensors);
  long total_access_volume = 0;
  long mem_footprint = 0;
  long infeasible = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    const size_t dims_num = mul_chain->getDims().size();
    const std::set<Tensor *> external_tensors = mul_chain->getExternalTensors();
    const size_t external_tensors_num = external_tensors.size();
    long sub_best_fuse_pattern = 0;
    OperatorTree sub_best_order_tree = mul_chain->getOperatorTree();
    long sub_best_access_volume = LONG_MAX;
    long sub_mem_footprint = 0;
    long sub_situation_num = 1;
    bool sub_infeasible = true;
    for (long sf = 0; sf < sub_situation_num; sf++) {
      mul_chain->setExternalTensorsFusePattern(sf);
      LogRecord2 sub_log_record;
      if (Options::save_log_file >= 2) {
        sub_log_record.id = 2;
        sub_log_record.defineAlgroithm(mul_chain->getOperators());
        sub_log_record.mem_size = mem_size;
        sub_log_record.recordFuseStatus(mul_chain->getExternalTensors());
      }
      if (print_log)
        std::cout << "----------------sub situation " << sf << "-----------------" << std::endl;
      double best_obj = MAXFLOAT;
      OperatorTree best_op_tree = mul_chain->getOperatorTree();
      bool sf_infeasible = true;
      if (mul_chain->getOperators().size() > 1) {
        mul_chain->setOrder(o_infos);
        best_obj = optimizeBlockSize(mul_chain, mem_size);
        best_op_tree = mul_chain->getOperatorTree();
        sf_infeasible = (best_obj == MAXFLOAT);
      } else {
        sf_infeasible = optimizeOrder(mul_chain, mem_size, best_obj, best_op_tree);
      }
      if (!sf_infeasible) {
        sub_infeasible = false;
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        DAT::optimizeBlockSize(mul_chain, mem_size);
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        long mem_access_volume = mul_chain->getMemAccessVolume();
        if (mem_access_volume < sub_best_access_volume) {
          sub_best_fuse_pattern = sf;
          sub_best_order_tree = best_op_tree;
          sub_best_access_volume = mem_access_volume;
          sub_mem_footprint = mul_chain->getMemFootprint();
        }
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = mem_access_volume;
          sub_log_record.mem_footprint = mul_chain->getMemFootprint();
        }
      } else {
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = -1;
        }
      }
      if (Options::save_log_file >= 2) {
        sub_log_record.writeToFile(Options::log_directory + "/log2s.csv");
      }
    }
    if (!sub_infeasible) {
      if (print_log) {
        std::cout << mul_chain->toString();
        std::cout << "access volume: " << sub_best_access_volume << std::endl;
        std::cout << "SRAM footprint: " << sub_mem_footprint << std::endl;
      }
      total_access_volume += sub_best_access_volume;
      mem_footprint = std::max(mem_footprint, sub_mem_footprint);
      mul_chain->recordExternalTensorsFuseInfo(sub_best_fuse_pattern);
      mul_chain->recordOrder(sub_best_order_tree.getOrderInfos());
    } else {
      if (print_log) {
        std::cerr << "operators: " << mul_chain->toString() << std::endl;
      }
    }
    infeasible += sub_infeasible;
  }
  if (!infeasible) {
    if (total_access_volume < best_total_access_volume) {
      for (long i = 0; i < best_operator_chain_num; ++i) {
        delete best_op_chain[i];
        best_op_chain[i] = nullptr;
      }
      best_total_access_volume = total_access_volume;
      best_operator_chain_num = operator_chain_num;
      for (long i = 0; i < operator_chain_num; ++i) {
        best_op_chain[i] = new OperatorChain;
        *(best_op_chain[i]) = *(op_chain[i]);
      }
    }
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = total_access_volume;
      log_record.mem_footprint = mem_footprint;
    }
    if (print_log) {
      std::cout << "total access volume: " << total_access_volume << std::endl;
      std::cout << "best total access volume: " << best_total_access_volume << std::endl;
    }
  } else {
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = -1;
    }
    if (print_log)
      std::cerr << "Can not use this fuse pattern with mem_size: " << mem_size << std::endl;
  }

  if (Options::save_log_file >= 2) {
    for (long i = 0; i < operator_chain_num; ++i) {
      op_chain[i]->setInternalTensorsFuse();
      op_chain[i]->setExternalTensorsFusePattern(op_chain[i]->getExternalTensorsFusePattern());
      op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
      DAT::optimizeBlockSize(op_chain[i], mem_size);
      log_record.recordSubFuseStatus(op_chain[i], op_chain[i]->getExternalTensors());
      log_record.recordOrder(op_chain[i], op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordDimBlocksizes(op_chain[i]);
      log_record.recordOpChainMemAccessVolume(op_chain[i]);
    }
    log_record.writeToFile(Options::log_directory + "/log2.csv");
  }
  for (long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
    op_chain[i] = nullptr;
  }

  std::bitset<MAX_TENSOR_NUM> fuse_or_not(bit_tmp);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  return best_operator_chain_num;
}

long chimeraDSE(const std::set<DAT::TensorOperator *> &non_add_to_operator_chain,
                const std::set<DAT::Tensor *> &tensors,
                long mem_size,
                DAT::OperatorChain *best_op_chain[],
                bool print_log = false) {
  long best_operator_chain_num = 0;
  long best_total_access_volume = LONG_MAX;

  std::set<Tensor *> non_io_tensors;
  for (auto t : tensors) {
    if (!t->isIO()) {
      non_io_tensors.insert(t);
    } else {
      t->unsetFuse();
    }
  }
  long situation_num = pow(2, non_io_tensors.size());

  std::bitset<MAX_TENSOR_NUM> bit_tmp;
  long tmp = 0;
  for (auto t : non_io_tensors) {
    if (t->isFused()) {
      bit_tmp[tmp] = true;
    } else {
      bit_tmp[tmp] = false;
    }
    ++tmp;
  }

  assert(tensors.size() <= MAX_TENSOR_NUM);
  DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
  LogRecord6 log_record;
  if (Options::save_log_file >= 2) {
    log_record.id = 2;
    log_record.defineAlgroithm(non_add_to_operator_chain);
    log_record.mem_size = mem_size;
    log_record.recordFuseStatus(non_io_tensors);
  }
  if (print_log)
    std::cout << "----------------situation " << -1 << "-----------------" << std::endl;
  long operator_chain_num =
      DAT::createToOperatorChain(op_chain, non_add_to_operator_chain, tensors);
  long total_access_volume = 0;
  long mem_footprint = 0;
  long infeasible = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    const size_t dims_num = mul_chain->getDims().size();
    const std::set<Tensor *> external_tensors = mul_chain->getExternalTensors();
    const size_t external_tensors_num = external_tensors.size();
    long sub_best_fuse_pattern = 0;
    OperatorTree sub_best_order_tree = mul_chain->getOperatorTree();
    long sub_best_access_volume = LONG_MAX;
    long sub_mem_footprint = 0;
    long sub_situation_num = 1;
    bool sub_infeasible = true;
    for (long sf = 0; sf < sub_situation_num; sf++) {
      mul_chain->setExternalTensorsFusePattern(sf);
      LogRecord2 sub_log_record;
      if (Options::save_log_file >= 2) {
        sub_log_record.id = 2;
        sub_log_record.defineAlgroithm(mul_chain->getOperators());
        sub_log_record.mem_size = mem_size;
        sub_log_record.recordFuseStatus(mul_chain->getExternalTensors());
      }
      if (print_log)
        std::cout << "----------------sub situation " << sf << "-----------------" << std::endl;
      double best_obj = MAXFLOAT;
      OperatorTree best_op_tree = mul_chain->getOperatorTree();
      bool sf_infeasible = true;
      sf_infeasible = optimizeOrder(mul_chain, mem_size, best_obj, best_op_tree);
      if (!sf_infeasible) {
        sub_infeasible = false;
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        DAT::optimizeBlockSize(mul_chain, mem_size);
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        long mem_access_volume = mul_chain->getMemAccessVolume();
        if (mem_access_volume < sub_best_access_volume) {
          sub_best_fuse_pattern = sf;
          sub_best_order_tree = best_op_tree;
          sub_best_access_volume = mem_access_volume;
          sub_mem_footprint = mul_chain->getMemFootprint();
        }
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = mem_access_volume;
          sub_log_record.mem_footprint = mul_chain->getMemFootprint();
        }
      } else {
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = -1;
        }
      }
      if (Options::save_log_file >= 2) {
        sub_log_record.writeToFile(Options::log_directory + "/log2s.csv");
      }
    }
    if (!sub_infeasible) {
      if (print_log) {
        std::cout << mul_chain->toString();
        std::cout << "access volume: " << sub_best_access_volume << std::endl;
        std::cout << "SRAM footprint: " << sub_mem_footprint << std::endl;
      }
      total_access_volume += sub_best_access_volume;
      mem_footprint = std::max(mem_footprint, sub_mem_footprint);
      mul_chain->recordExternalTensorsFuseInfo(sub_best_fuse_pattern);
      mul_chain->recordOrder(sub_best_order_tree.getOrderInfos());
    } else {
      if (print_log) {
        std::cerr << "operators: " << mul_chain->toString() << std::endl;
      }
    }
    infeasible += sub_infeasible;
  }
  if (!infeasible) {
    if (total_access_volume < best_total_access_volume) {
      for (long i = 0; i < best_operator_chain_num; ++i) {
        delete best_op_chain[i];
        best_op_chain[i] = nullptr;
      }
      best_total_access_volume = total_access_volume;
      best_operator_chain_num = operator_chain_num;
      for (long i = 0; i < operator_chain_num; ++i) {
        best_op_chain[i] = new OperatorChain;
        *(best_op_chain[i]) = *(op_chain[i]);
      }
    }
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = total_access_volume;
      log_record.mem_footprint = mem_footprint;
    }
    if (print_log) {
      std::cout << "total access volume: " << total_access_volume << std::endl;
      std::cout << "best total access volume: " << best_total_access_volume << std::endl;
    }
  } else {
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = -1;
    }
    if (print_log)
      std::cerr << "Can not use this fuse pattern with mem_size: " << mem_size << std::endl;
  }

  if (Options::save_log_file >= 2) {
    for (long i = 0; i < operator_chain_num; ++i) {
      op_chain[i]->setInternalTensorsFuse();
      op_chain[i]->setExternalTensorsFusePattern(op_chain[i]->getExternalTensorsFusePattern());
      op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
      DAT::optimizeBlockSize(op_chain[i], mem_size);
      log_record.recordSubFuseStatus(op_chain[i], op_chain[i]->getExternalTensors());
      log_record.recordOrder(op_chain[i], op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordDimBlocksizes(op_chain[i]);
      log_record.recordOpChainMemAccessVolume(op_chain[i]);
    }
    log_record.writeToFile(Options::log_directory + "/log2.csv");
  }
  for (long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
    op_chain[i] = nullptr;
  }

  std::bitset<MAX_TENSOR_NUM> fuse_or_not(bit_tmp);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  return best_operator_chain_num;
}

long baselineDSE(const std::set<DAT::TensorOperator *> &non_add_to_operator_chain,
                 const std::set<DAT::Tensor *> &tensors,
                 long mem_size,
                 DAT::OperatorChain *best_op_chain[],
                 bool print_log = false) {
  long best_operator_chain_num = 0;
  long best_total_access_volume = LONG_MAX;

  std::set<Tensor *> non_io_tensors;
  for (auto t : tensors) {
    if (!t->isIO()) {
      non_io_tensors.insert(t);
    } else {
      t->unsetFuse();
    }
  }
  long situation_num = pow(2, non_io_tensors.size());

  std::bitset<MAX_TENSOR_NUM> bit_tmp;
  long tmp = 0;
  for (auto t : non_io_tensors) {
    if (t->isFused()) {
      bit_tmp[tmp] = true;
    } else {
      bit_tmp[tmp] = false;
    }
    ++tmp;
  }

  assert(tensors.size() <= MAX_TENSOR_NUM);
  DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
  LogRecord6 log_record;
  if (Options::save_log_file >= 2) {
    log_record.id = 2;
    log_record.defineAlgroithm(non_add_to_operator_chain);
    log_record.mem_size = mem_size;
    log_record.recordFuseStatus(non_io_tensors);
  }
  if (print_log)
    std::cout << "----------------situation " << -1 << "-----------------" << std::endl;
  long operator_chain_num =
      DAT::createToOperatorChain(op_chain, non_add_to_operator_chain, tensors);
  long total_access_volume = 0;
  long mem_footprint = 0;
  long infeasible = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    const size_t dims_num = mul_chain->getDims().size();
    const std::set<Tensor *> external_tensors = mul_chain->getExternalTensors();
    const size_t external_tensors_num = external_tensors.size();
    long sub_best_fuse_pattern = 0;
    OperatorTree sub_best_order_tree = mul_chain->getOperatorTree();
    long sub_best_access_volume = LONG_MAX;
    long sub_mem_footprint = 0;
    long sub_situation_num = 1;
    bool sub_infeasible = true;
    for (long sf = 0; sf < sub_situation_num; sf++) {
      mul_chain->setExternalTensorsFusePattern(sf);
      LogRecord2 sub_log_record;
      if (Options::save_log_file >= 2) {
        sub_log_record.id = 2;
        sub_log_record.defineAlgroithm(mul_chain->getOperators());
        sub_log_record.mem_size = mem_size;
        sub_log_record.recordFuseStatus(mul_chain->getExternalTensors());
      }
      if (print_log)
        std::cout << "----------------sub situation " << sf << "-----------------" << std::endl;
      double best_obj = MAXFLOAT;
      OperatorTree best_op_tree = mul_chain->getOperatorTree();
      bool sf_infeasible = true;
      sf_infeasible = optimizeOrder(mul_chain, mem_size, best_obj, best_op_tree);
      if (!sf_infeasible) {
        sub_infeasible = false;
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        DAT::optimizeBlockSize(mul_chain, mem_size);
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        long mem_access_volume = mul_chain->getMemAccessVolume();
        if (mem_access_volume < sub_best_access_volume) {
          sub_best_fuse_pattern = sf;
          sub_best_order_tree = best_op_tree;
          sub_best_access_volume = mem_access_volume;
          sub_mem_footprint = mul_chain->getMemFootprint();
        }
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = mem_access_volume;
          sub_log_record.mem_footprint = mul_chain->getMemFootprint();
        }
      } else {
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = -1;
        }
      }
      if (Options::save_log_file >= 2) {
        sub_log_record.writeToFile(Options::log_directory + "/log2s.csv");
      }
    }
    if (!sub_infeasible) {
      if (print_log) {
        std::cout << mul_chain->toString();
        std::cout << "access volume: " << sub_best_access_volume << std::endl;
        std::cout << "SRAM footprint: " << sub_mem_footprint << std::endl;
      }
      total_access_volume += sub_best_access_volume;
      mem_footprint = std::max(mem_footprint, sub_mem_footprint);
      mul_chain->recordExternalTensorsFuseInfo(sub_best_fuse_pattern);
      mul_chain->recordOrder(sub_best_order_tree.getOrderInfos());
    } else {
      if (print_log) {
        std::cerr << "operators: " << mul_chain->toString() << std::endl;
      }
    }
    infeasible += sub_infeasible;
  }
  if (!infeasible) {
    if (total_access_volume < best_total_access_volume) {
      for (long i = 0; i < best_operator_chain_num; ++i) {
        delete best_op_chain[i];
        best_op_chain[i] = nullptr;
      }
      best_total_access_volume = total_access_volume;
      best_operator_chain_num = operator_chain_num;
      for (long i = 0; i < operator_chain_num; ++i) {
        best_op_chain[i] = new OperatorChain;
        *(best_op_chain[i]) = *(op_chain[i]);
      }
    }
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = total_access_volume;
      log_record.mem_footprint = mem_footprint;
    }
    if (print_log) {
      std::cout << "total access volume: " << total_access_volume << std::endl;
      std::cout << "best total access volume: " << best_total_access_volume << std::endl;
    }
  } else {
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = -1;
    }
    if (print_log)
      std::cerr << "Can not use this fuse pattern with mem_size: " << mem_size << std::endl;
  }

  if (Options::save_log_file >= 2) {
    for (long i = 0; i < operator_chain_num; ++i) {
      op_chain[i]->setInternalTensorsFuse();
      op_chain[i]->setExternalTensorsFusePattern(op_chain[i]->getExternalTensorsFusePattern());
      op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
      DAT::optimizeBlockSize(op_chain[i], mem_size);
      log_record.recordSubFuseStatus(op_chain[i], op_chain[i]->getExternalTensors());
      log_record.recordOrder(op_chain[i], op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordDimBlocksizes(op_chain[i]);
      log_record.recordOpChainMemAccessVolume(op_chain[i]);
    }
    log_record.writeToFile(Options::log_directory + "/log2.csv");
  }
  for (long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
    op_chain[i] = nullptr;
  }

  std::bitset<MAX_TENSOR_NUM> fuse_or_not(bit_tmp);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  return best_operator_chain_num;
}

long nonDSE(const std::set<DAT::TensorOperator *> &non_add_to_operator_chain,
            const std::set<DAT::Tensor *> &tensors,
            long mem_size,
            DAT::OperatorChain *best_op_chain[],
            std::map<OperatorNode *, OrderInfo> o_infos,
            bool print_log = false) {
  long best_operator_chain_num = 0;
  long best_total_access_volume = LONG_MAX;

  std::set<Tensor *> non_io_tensors;
  for (auto t : tensors) {
    if (!t->isIO()) {
      non_io_tensors.insert(t);
    } else {
      t->unsetFuse();
    }
  }
  long situation_num = pow(2, non_io_tensors.size());

  std::bitset<MAX_TENSOR_NUM> bit_tmp;
  long tmp = 0;
  for (auto t : non_io_tensors) {
    if (t->isFused()) {
      bit_tmp[tmp] = true;
    } else {
      bit_tmp[tmp] = false;
    }
    ++tmp;
  }

  assert(tensors.size() <= MAX_TENSOR_NUM);
  DAT::OperatorChain *op_chain[MAX_TENSOR_NUM] = {nullptr};
  LogRecord6 log_record;
  if (Options::save_log_file >= 2) {
    log_record.id = 2;
    log_record.defineAlgroithm(non_add_to_operator_chain);
    log_record.mem_size = mem_size;
    log_record.recordFuseStatus(non_io_tensors);
  }
  if (print_log)
    std::cout << "----------------situation " << -1 << "-----------------" << std::endl;
  long operator_chain_num =
      DAT::createToOperatorChain(op_chain, non_add_to_operator_chain, tensors);
  long total_access_volume = 0;
  long mem_footprint = 0;
  long infeasible = 0;
  for (long i = 0; i < operator_chain_num; ++i) {
    DAT::OperatorChain *mul_chain = op_chain[i];
    const size_t dims_num = mul_chain->getDims().size();
    const std::set<Tensor *> external_tensors = mul_chain->getExternalTensors();
    const size_t external_tensors_num = external_tensors.size();
    long sub_best_fuse_pattern = 0;
    OperatorTree sub_best_order_tree = mul_chain->getOperatorTree();
    long sub_best_access_volume = LONG_MAX;
    long sub_mem_footprint = 0;
    long sub_situation_num = 1;
    bool sub_infeasible = true;
    for (long sf = 0; sf < sub_situation_num; sf++) {
      mul_chain->setExternalTensorsFusePattern(sf);
      LogRecord2 sub_log_record;
      if (Options::save_log_file >= 2) {
        sub_log_record.id = 2;
        sub_log_record.defineAlgroithm(mul_chain->getOperators());
        sub_log_record.mem_size = mem_size;
        sub_log_record.recordFuseStatus(mul_chain->getExternalTensors());
      }
      if (print_log)
        std::cout << "----------------sub situation " << sf << "-----------------" << std::endl;
      double best_obj = MAXFLOAT;
      OperatorTree best_op_tree = mul_chain->getOperatorTree();
      bool sf_infeasible = true;
      mul_chain->setOrder(o_infos);
      best_obj = mul_chain->getMemAccessVolume();
      best_op_tree = mul_chain->getOperatorTree();
      sf_infeasible = (best_obj == MAXFLOAT);
      if (!sf_infeasible) {
        sub_infeasible = false;
        mul_chain->setOrder(best_op_tree.getOrderInfos());
        long mem_access_volume = mul_chain->getMemAccessVolume();
        if (mem_access_volume < sub_best_access_volume) {
          sub_best_fuse_pattern = sf;
          sub_best_order_tree = best_op_tree;
          sub_best_access_volume = mem_access_volume;
          sub_mem_footprint = mul_chain->getMemFootprint();
        }
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = mem_access_volume;
          sub_log_record.mem_footprint = mul_chain->getMemFootprint();
        }
      } else {
        if (Options::save_log_file >= 2) {
          sub_log_record.mem_access_volume = -1;
        }
      }
      if (Options::save_log_file >= 2) {
        sub_log_record.writeToFile(Options::log_directory + "/log2s.csv");
      }
    }
    if (!sub_infeasible) {
      if (print_log) {
        std::cout << mul_chain->toString();
        std::cout << "access volume: " << sub_best_access_volume << std::endl;
        std::cout << "SRAM footprint: " << sub_mem_footprint << std::endl;
      }
      total_access_volume += sub_best_access_volume;
      mem_footprint = std::max(mem_footprint, sub_mem_footprint);
      mul_chain->recordExternalTensorsFuseInfo(sub_best_fuse_pattern);
      mul_chain->recordOrder(sub_best_order_tree.getOrderInfos());
    } else {
      if (print_log) {
        std::cerr << "operators: " << mul_chain->toString() << std::endl;
      }
    }
    infeasible += sub_infeasible;
  }
  if (!infeasible) {
    if (total_access_volume < best_total_access_volume) {
      for (long i = 0; i < best_operator_chain_num; ++i) {
        delete best_op_chain[i];
        best_op_chain[i] = nullptr;
      }
      best_total_access_volume = total_access_volume;
      best_operator_chain_num = operator_chain_num;
      for (long i = 0; i < operator_chain_num; ++i) {
        best_op_chain[i] = new OperatorChain;
        *(best_op_chain[i]) = *(op_chain[i]);
      }
    }
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = total_access_volume;
      log_record.mem_footprint = mem_footprint;
    }
    if (print_log) {
      std::cout << "total access volume: " << total_access_volume << std::endl;
      std::cout << "best total access volume: " << best_total_access_volume << std::endl;
    }
  } else {
    if (Options::save_log_file >= 2) {
      log_record.mem_access_volume = -1;
    }
    if (print_log)
      std::cerr << "Can not use this fuse pattern with mem_size: " << mem_size << std::endl;
  }

  if (Options::save_log_file >= 2) {
    for (long i = 0; i < operator_chain_num; ++i) {
      op_chain[i]->setInternalTensorsFuse();
      op_chain[i]->setExternalTensorsFusePattern(op_chain[i]->getExternalTensorsFusePattern());
      op_chain[i]->setOrder(op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordSubFuseStatus(op_chain[i], op_chain[i]->getExternalTensors());
      log_record.recordOrder(op_chain[i], op_chain[i]->getOperatorTree().getOrderInfos());
      log_record.recordDimBlocksizes(op_chain[i]);
      log_record.recordOpChainMemAccessVolume(op_chain[i]);
    }
    log_record.writeToFile(Options::log_directory + "/log2.csv");
  }
  for (long i = 0; i < operator_chain_num; ++i) {
    delete op_chain[i];
    op_chain[i] = nullptr;
  }

  std::bitset<MAX_TENSOR_NUM> fuse_or_not(bit_tmp);
  long f_i = 0;
  for (auto t : non_io_tensors) {
    if (fuse_or_not[f_i]) {
      t->setFuse();
    } else {
      t->unsetFuse();
    }
    ++f_i;
  }
  return best_operator_chain_num;
}

}
#endif //MMCHAIN_ANALYSIS_SRC_DSE_H
