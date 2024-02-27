//
// Created by lxu on 23-6-10.
//

#ifndef MMCHAIN_ANALYSIS_SRC_DIM_H
#define MMCHAIN_ANALYSIS_SRC_DIM_H

#include <cassert>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include "options.h"

namespace DAT {

class TensorOperator;
class Tensor;

class Dim {
public:
  Dim() : size_set(false) {};
  Dim(std::string n, long s) : name(std::move(n)), size(s), size_set(true) {};
  explicit Dim(std::string n) : name(std::move(n)), size_set(false) {};
  explicit Dim(long s) : size_set(true), size(s) {};
  [[nodiscard]] long getSize() const {
    assert(size_set);
    return size;
  }
  [[nodiscard]] long getBlocks(TensorOperator *to) const {
    assert(size_set);
    return tensor_blocks.at(to);
  }
  [[nodiscard]] long getBlockSize(TensorOperator *to) const {
    assert(size_set);
    return tensor_block_sizes.at(to);
  }
  std::string getName() {
    return name;
  }
  void setSize(long s) {
    size_set = true;
    size = s;
  }
  void setBlocks(TensorOperator *to, long b) {
    assert(size_set);
    assert(size % b == 0);
    tensor_blocks[to] = b;
    tensor_block_sizes[to] = size / b;
  }
  void setBlockSize(TensorOperator *to, long bs) {
    assert(size_set);
    assert(size % bs == 0);
    tensor_block_sizes[to] = bs;
    tensor_blocks[to] = size / bs;
  }
  void setName(std::string n) {
    name = std::move(n);
  }
  void addRelatedTensor(Tensor *t, long i) {
    related_tensor[t] = i;
  }
  std::string toString(TensorOperator *to) {
    return name + "(" + std::to_string(size) + "," + std::to_string(tensor_blocks.at(to)) + ","
        + std::to_string(tensor_block_sizes.at(to)) + ")";
  }
  bool equal(Dim *d) const {
    return d->size == this->size && d->tensor_block_sizes == this->tensor_block_sizes && d->tensor_blocks == this->tensor_blocks;
  }
  void addParDim(Dim *d) {
    par_dims.insert(d);
    d->par_dims.insert(this);
  }
  void removeParDim(Dim *d) {
    par_dims.erase(d);
    d->par_dims.erase(d);
  }
  std::set<Dim *> getParDims() {
    return par_dims;
  }
  void clear() {
    size_set = false;
    size = 0;
    tensor_blocks.clear();
    tensor_block_sizes.clear();
    related_tensor.clear();
    par_dims.clear();
  }

private:
  bool size_set{false};
  long size{0};
  std::string name;
  std::map<Tensor *, long> related_tensor;
  std::map<TensorOperator *, long> tensor_block_sizes;
  std::map<TensorOperator *, long> tensor_blocks;
  std::set<Dim *> par_dims;

};

} // DAT

#endif //MMCHAIN_ANALYSIS_SRC_DIM_H
