//
// Created by Lei Xu on 2023/7/13.
//

#ifndef MMCHAIN_ANALYSIS_SRC_TENSOR_H
#define MMCHAIN_ANALYSIS_SRC_TENSOR_H

#include "dim.h"

namespace DAT {

class Tensor {
public:
  std::string getName() {
    return name;
  }
  [[nodiscard]] bool isFused() const {
    return fuse;
  }
  Dim *getDim(long i) {
    return dims[i];
  }
  Dim *getDim(const std::string &n) {
    for (auto dim : dims) {
      if (dim->getName() == n) {
        return dim;
      }
    }
    return nullptr;
  }
  std::vector<Dim *> getDims() {
    return dims;
  }
  bool hasDim(const std::string &n) {
    for (auto dim : dims) {
      if (dim->getName() == n) {
        return true;
      }
    }
    return false;
  }
  bool hasDim(Dim *d) {
    return std::any_of(dims.begin(), dims.end(), [d](Dim *dim) { return dim == d; });
  }
  bool hasParDim(Dim *d) {
    for (auto dim : dims) {
      if (dim->getParDims().count(d)) {
        return true;
      }
    }
    return false;
  }
  void setName(std::string n) {
    name = std::move(n);
  }
  void setFuse() {
    fuse = true;
  }
  void unsetFuse() {
    fuse = false;
  }
  void setDim(long i, Dim *d) {
    assert(i < dims.size());
    dims[i] = d;
    d->addRelatedTensor(this, i);
  }
  void addRelatedOperator(TensorOperator *to, const std::string &io) {
    assert(io == "input" || io == "output");
    related_operators[to] = io;
  }
  bool isIO() {
    bool as_input = false;
    bool as_output = false;
    for (const auto &ro : related_operators) {
      if (ro.second == "input") {
        as_input = true;
      } else if (ro.second == "output") {
        as_output = true;
      }
    }
    assert(as_input || as_output);
    return as_input ^ as_output;
  }
  std::map<TensorOperator *, std::string> getRelatedOperator() {
    return related_operators;
  }
  void updateAccessTimesStr(TensorOperator *to, std::string times) {
    access_times_str[to] = std::move(times);
  }
  void updateAccessTimes(TensorOperator *to, long times) {
    access_times[to] = times;
  }
  void updateMemFootprintStr(TensorOperator *to, std::string fp) {
    mem_footprint_str[to] = std::move(fp);
  }
  void updateMemFootprint(TensorOperator *to, long fp) {
    mem_footprint[to] = fp;
  }
  std::map<TensorOperator *, std::string> getAccessTimesStr() {
    return access_times_str;
  }
  std::map<TensorOperator *, std::string> getAccessVolumeStr(TensorOperator *to) {
    std::map<TensorOperator *, std::string> access_volume_str;
    for (const auto &at_s : access_times_str) {
      std::string str = at_s.second;
      str += " * " + getBlockSizeStr(to);
      access_volume_str[at_s.first] = str;
    }
    return access_volume_str;
  }
  std::map<TensorOperator *, long> getAccessTimes() {
    return access_times;
  }
  std::map<TensorOperator *, long> getAccessVolume(TensorOperator *to) {
    std::map<TensorOperator *, long> access_volume;
    for (auto at : access_times) {
      access_volume[at.first] = at.second * getBlockSize(to);
    }
    return access_volume;
  }
  std::map<TensorOperator *, std::string> getMemFootprintStr() {
    return mem_footprint_str;
  }
  std::map<TensorOperator *, long> getMemFootprint() {
    return mem_footprint;
  }
  void isReused(bool reuse) {
    is_reused = reuse;
  }
  [[nodiscard]] bool isReused() const {
    return is_reused;
  }
  void isExternal(bool external) {
    is_external = external;
  }
  [[nodiscard]] bool isExternal() const {
    return is_external;
  }
  bool equal(Tensor *t) {
    if (t->dims.size() == this->dims.size()) {
      for (long i = 0; i < this->dims.size(); i++) {
        if (!(t->dims[i]->equal(this->dims[i]))) {
          return false;
        }
      }
    } else {
      return false;
    }
    return true;
  }
  virtual long getBlockSize(TensorOperator *to) {
    long ret = 1;
    for (auto &dim : dims) {
      ret *= dim->getBlockSize(to);
    }
    return ret;
  }
  virtual std::string getBlockSizeStr(TensorOperator *to) {
    std::string str;
    for (auto d = dims.begin(); d != dims.end(); d++) {
      if (d != dims.begin()) {
        str += " * ";
      }
      str += (*d)->getName() + "_blocksize(" + std::to_string((*d)->getBlockSize(to)) + ")";
    }
    return str;
  }
  virtual long getBlocks(TensorOperator *to) {
    long ret = 1;
    for (auto &dim : dims) {
      ret *= dim->getBlocks(to);
    }
    return ret;
  }
  virtual long getSize() {
    long ret = 1;
    for (auto &dim : dims) {
      ret *= dim->getSize();
    }
    return ret;
  }
  void addParTensor(Tensor *t) {
    assert(t->equal(this));
    par_tensors.insert(t);
    t->par_tensors.insert(this);
    for (long i = 0; i < dims.size(); i++) {
      dims[i]->addParDim(t->dims[i]);
    }
  }
  void removeParTensor(Tensor *t) {
    par_tensors.erase(t);
    t->par_tensors.erase(t);
    for (long i = 0; i < dims.size(); i++) {
      dims[i]->removeParDim(t->dims[i]);
    }
  }
  std::set<Tensor *> getParTensors() {
    return par_tensors;
  }
  void clear() {
    for (auto d : dims) {
      d->clear();
    }
    dims.clear();
    fuse = false;
    related_operators.clear();
    access_times_str.clear();
    access_times.clear();
    mem_footprint_str.clear();
    mem_footprint.clear();
    is_reused = false;
    is_external = false;
    par_tensors.clear();
    expand_dims.clear();
  }
  void clearMemAccess() {
    access_times_str.clear();
    access_times.clear();
    is_reused = false;
  }
  void clearMemFootprint() {
    mem_footprint_str.clear();
    mem_footprint.clear();
    expand_dims.clear();
  }
  virtual std::string toString(TensorOperator *to) {
    std::string str;
    str = name;
    for (auto d : dims) {
      str += ("[" + d->toString(to) + "]");
    }
    return str;
  }
  void addExpandDims(Dim *d) {
    expand_dims.insert(d);
  }
  std::set<Dim *> getExpandDims() {
    return expand_dims;
  }
  bool hasExpandDim(Dim *d) {
    return expand_dims.count(d);
  }
  void clearExpandDims() {
    expand_dims.clear();
  }

protected:
  std::vector<Dim *> dims;
  std::set<Dim *> expand_dims;
  bool fuse{false};
  std::string name;
  std::map<TensorOperator *, std::string> related_operators;
  std::map<TensorOperator *, std::string> access_times_str;
  std::map<TensorOperator *, long> access_times;
  std::map<TensorOperator *, std::string> mem_footprint_str;
  std::map<TensorOperator *, long> mem_footprint;
  bool is_reused{false};
  bool is_external{false};
  std::set<Tensor *> par_tensors;
};

class Tensor2D : public Tensor {
public:
  Tensor2D() {
    dims.resize(2);
  }
  explicit Tensor2D(std::string n) {
    dims.resize(2);
    setName(std::move(n));
  }
  Tensor2D(Dim *d1, Dim *d2) {
    dims.resize(2);
    setDim(0, d1);
    setDim(1, d2);
  }
  Tensor2D(std::string n, Dim *d1, Dim *d2) {
    dims.resize(2);
    setDim(0, d1);
    setDim(1, d2);
    setName(std::move(n));
  }
};

class BatchTensor2D : public Tensor2D {
public:
  BatchTensor2D() {
    dims.resize(2);
  }
  explicit BatchTensor2D(std::string n) {
    dims.resize(2);
    setName(std::move(n));
  }
  BatchTensor2D(Dim *d1, Dim *d2) {
    dims.resize(2);
    setDim(0, d1);
    setDim(1, d2);
  }
  BatchTensor2D(std::string n, Dim *d1, Dim *d2) {
    dims.resize(2);
    setDim(0, d1);
    setDim(1, d2);
    setName(std::move(n));
  }
  BatchTensor2D(std::string n, long bs, Dim *d1, Dim *d2) {
    dims.resize(2);
    setDim(0, d1);
    setDim(1, d2);
    setName(std::move(n));
    setBatchSize(bs);
  }
  std::string toString(TensorOperator *to) override {
    return Tensor2D::toString(to) + "(" + std::to_string(batch_size) + ")";
  }
  void setBatchSize(long bs) {
    batch_size = bs;
  }
  [[nodiscard]] long getBatchSize() const {
    return batch_size;
  }
  long getBlockSize(TensorOperator *to) override {
    return batch_size * Tensor2D::getBlockSize(to);
  }
  std::string getBlockSizeStr(TensorOperator *to) override {
    return std::to_string(batch_size) + " * " + Tensor2D::getBlockSizeStr(to);
  }

protected:
  long batch_size = 1;
};

class Tensor3D : public Tensor {
public:
  Tensor3D() {
    dims.resize(3);
  }
  explicit Tensor3D(std::string n) {
    dims.resize(3);
    setName(std::move(n));
  }
  Tensor3D(Dim *d1, Dim *d2, Dim *d3) {
    dims.resize(3);
    setDim(0, d1);
    setDim(1, d2);
    setDim(2, d3);
  }
  Tensor3D(std::string n, Dim *d1, Dim *d2, Dim *d3) {
    dims.resize(3);
    setDim(0, d1);
    setDim(1, d2);
    setDim(2, d3);
    setName(std::move(n));
  }
};

class Tensor4D : public Tensor {
public:
  Tensor4D() {
    dims.resize(4);
  }
  explicit Tensor4D(std::string n) {
    dims.resize(4);
    setName(std::move(n));
  }
  Tensor4D(Dim *d1, Dim *d2, Dim *d3, Dim *d4) {
    dims.resize(4);
    setDim(0, d1);
    setDim(1, d2);
    setDim(2, d3);
    setDim(3, d4);
  }
  Tensor4D(std::string n, Dim *d1, Dim *d2, Dim *d3, Dim *d4) {
    dims.resize(4);
    setDim(0, d1);
    setDim(1, d2);
    setDim(2, d3);
    setDim(3, d4);
    setName(std::move(n));
  }
};

}

#endif //MMCHAIN_ANALYSIS_SRC_TENSOR_H
