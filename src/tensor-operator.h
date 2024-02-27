//
// Created by Lei Xu on 2023/7/13.
//

#ifndef MMCHAIN_ANALYSIS_SRC_TENSOR_OPERATOR_H
#define MMCHAIN_ANALYSIS_SRC_TENSOR_OPERATOR_H

#include <utility>

#include "tensor.h"

namespace DAT {

class OperatorNode;

class TensorOperator {
public:
  Tensor *getInputTensor(long i) {
    assert(i < inputs.size());
    return inputs[i];
  };
  Tensor *getOutputTensor(long i) {
    assert(i < outputs.size());
    return outputs[i];
  };
  Tensor *getInputTensor(const std::string &n) {
    for (auto tensor : inputs) {
      if (tensor->getName() == n) {
        return tensor;
      }
    }
    return nullptr;
  }
  Tensor *getOutputTensor(const std::string &n) {
    for (auto tensor : outputs) {
      if (tensor->getName() == n) {
        return tensor;
      }
    }
    return nullptr;
  }
  std::vector<Tensor *> getInputTensors() {
    return inputs;
  }
  std::vector<Tensor *> getOutputTensors() {
    return outputs;
  }
  std::set<Tensor *> getTensors() {
    std::set<Tensor *> ret(inputs.begin(), inputs.end());
    ret.insert(outputs.begin(), outputs.end());
    return ret;
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
  bool hasDim(Dim *d) {
    return dims.count(d);
  }
  bool hasActualDim(Dim *d) {
    return actual_dims.count(d);
  }
  bool hasParDim(Dim *d) {
    for (auto dim : dims) {
      if (dim->getParDims().count(d)) {
        return true;
      }
    }
    return false;
  }
  bool hasTensor(Tensor *t) {
    return std::any_of(inputs.begin(), inputs.end(), [t](auto i) { return i == t; })
        || std::any_of(outputs.begin(), outputs.end(), [t](auto o) { return o == t; });
  }
  void setInputTensor(long i, Tensor *t) {
    assert(i < inputs.size());
    inputs[i] = t;
    for (auto d : t->getDims()) {
      dims.insert(d);
      d->setBlockSize(this, 1);
    }
    t->addRelatedOperator(this, "input");
  };
  void setOutputTensor(long i, Tensor *t) {
    assert(i < outputs.size());
    outputs[i] = t;
    for (auto d : t->getDims()) {
      dims.insert(d);
      d->setBlockSize(this, 1);
    }
    t->addRelatedOperator(this, "output");
  };
  [[nodiscard]] long getReuseTensorNum() const {
    long num_reuse_tensors = 0;
    for (auto i : inputs) {
      if (i->isReused())
        num_reuse_tensors += 1;
    }
    for (auto o : outputs) {
      if (o->isReused())
        num_reuse_tensors += 1;
    }
    return num_reuse_tensors;
  }
  [[nodiscard]] long getAccessTimes() const {
    return access_times;
  }
  [[nodiscard]] long getAccessVolume() const {
    return access_volume;
  }
  virtual void analyzeMemAccess(const std::vector<Dim *> &dims_order) = 0;
  virtual void analyzeMemFootprint(const std::vector<Dim *> &dims_order) = 0;
  virtual bool isReduceDimExpended(Tensor *t) = 0;
  void clear() {
    for (auto t : inputs) {
      t->clear();
    }
    for (auto t : outputs) {
      t->clear();
    }
    for (auto d : dims) {
      d->clear();
    }
    inputs.clear();
    outputs.clear();
    dims.clear();
    actual_dims.clear();
    access_times = 0;
    access_volume = 0;
  }
  void clearMemAccess() {
    access_times = 0;
    access_volume = 0;
    for (auto t : inputs) {
      t->clearMemAccess();
    }
    for (auto t : outputs) {
      t->clearMemAccess();
    }
  }
  void clearMemFootprint() {
    for (auto t : inputs) {
      t->clearMemFootprint();
    }
    for (auto t : outputs) {
      t->clearMemFootprint();
    }
  }
  virtual std::string toString() = 0;
  virtual long getOpsNum() = 0;
  void setLinkedNode(OperatorNode *on) {
    linked_node = on;
  }
  void clearLinkedNode() {
    linked_node = nullptr;
  }
  OperatorNode *getLinkedNode() {
    return linked_node;
  }
  std::set<Tensor *> getEnvTensors() {
    return env_tensors;
  }
  void setEnvTensors(std::set<Tensor *> env_ts) {
    env_tensors = std::move(env_ts);
  }
  void clearEnvTensors() {
    env_tensors.clear();
  }
  void addEnvTensor(Tensor *t) {
    env_tensors.insert(t);
  }
  void addEnvTensor(std::set<Tensor *> ts) {
    env_tensors.insert(ts.begin(), ts.end());
  }
  void addEnvTensor(std::vector<Tensor *> ts) {
    env_tensors.insert(ts.begin(), ts.end());
  }
  void removeEnvTensor(Tensor *t) {
    env_tensors.erase(t);
  }
  [[nodiscard]] bool is_with_bias() const {
    return with_bias;
  }
  void is_with_bias(bool w) {
    with_bias = w;
  }
  virtual long compute_time() {
    return INT64_MIN;
  };
  std::string getName() {
    assert(name_set);
    return name;
  }
  void setName(std::string n) {
    name = std::move(n);
    name_set = true;
  }
  int getGroupInd() {
    return group_ind;
  }
  void setGroupInd(int i) {
    group_ind = i;
  }
  void isBatchDependent(bool bd) {
    batch_dependent = bd;
  }
  bool isBatchDependent() {
    assert(batch_dependent >= 0);
    return batch_dependent;
  }
  void isHeadDependent(bool hd) {
    head_dependent = hd;
  }
  bool isHeadDependent() {
    assert(head_dependent >= 0);
    return head_dependent;
  }
  virtual std::set<Dim *> getBatchDims() = 0;
  virtual std::set<Dim *> getReduceDims() = 0;
  virtual std::set<Dim *> getShapeDims(int i) = 0;

protected:
  std::string name;
  bool name_set{false};
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  std::set<Dim *> dims;
  int group_ind{-1};
  bool with_bias{false};
  int batch_dependent{-1};
  int head_dependent{-1};
  OperatorNode *linked_node{nullptr};
  std::set<Tensor *> env_tensors;
  std::set<Dim *> actual_dims;
  long access_times{};
  long access_volume{};

};

class MatrixMul : public TensorOperator {
public:
  MatrixMul(std::string n, Tensor *input1, Tensor *input2, Tensor *output) {
    setName(std::move(n));
    inputs.resize(2);
    outputs.resize(1);
    setInputTensor(0, input1);
    setInputTensor(1, input2);
    setOutputTensor(0, output);
  }
  MatrixMul(Tensor *input1, Tensor *input2, Tensor *output) {
    inputs.resize(2);
    outputs.resize(1);
    setInputTensor(0, input1);
    setInputTensor(1, input2);
    setOutputTensor(0, output);
  }
  MatrixMul() {
    inputs.resize(2);
    outputs.resize(1);
  }
  Tensor *getOutput() {
    assert(outputs[0] != nullptr);
    return outputs[0];
  }
  std::string toString() override {
    std::string str;
    str = outputs[0]->toString(this) + " = " + inputs[0]->toString(this) + " * " + inputs[1]->toString(this);
    if (with_bias) str += " + bias";
    return str;
  }
  long getOpsNum() override {
    long ops = 1;
    for (auto d : dims) {
      ops *= d->getSize();
    }
    return ops;
  }
  long compute_time() override {
    double t = 0;
    Dim *time_d;
    for (auto d : dims) {
      if (!outputs[0]->hasDim(d))
        time_d = d;
    }
    long reduce_d_block_size = time_d->getBlockSize(this);
    t = std::ceil(outputs[0]->getBlockSize(this) / static_cast<double>(Options::compute_power))
        * getOpsNum() / outputs[0]->getBlockSize(this)
        * (reduce_d_block_size + 2.0) / reduce_d_block_size;

    for (auto d : dims) {
      if (!inputs[0]->hasDim(d))
        time_d = d;
    }
    reduce_d_block_size = time_d->getBlockSize(this);
    t = std::min(t,
                 std::ceil(inputs[0]->getBlockSize(this) / static_cast<double>(Options::compute_power))
                     * getOpsNum() / inputs[0]->getBlockSize(this)
                     * (reduce_d_block_size + 2.0) / reduce_d_block_size);

    for (auto d : dims) {
      if (!inputs[1]->hasDim(d))
        time_d = d;
    }
    reduce_d_block_size = time_d->getBlockSize(this);
    t = std::min(t,
                 std::ceil(inputs[1]->getBlockSize(this) / static_cast<double>(Options::compute_power))
                     * getOpsNum() / inputs[1]->getBlockSize(this)
                     * (reduce_d_block_size + 2.0) / reduce_d_block_size);

    return std::ceil(t);
  }
  bool isReduceDimExpended(Tensor *t) override {
    for (auto d : dims) {
      if (!outputs[0]->hasDim(d)) {
        return t->hasExpandDim(d);
      }
    }
    return true;
  }
  std::set<Dim *> getBatchDims() override {
    std::set<Dim *> batch_dims;
    for (auto d : dims) {
      if (inputs[0]->hasDim(d) && inputs[1]->hasDim(d) && outputs[0]->hasDim(d)) {
        batch_dims.insert(d);
      }
    }
    return batch_dims;
  }
  std::set<Dim *> getReduceDims() override {
    std::set<Dim *> reduce_dims;
    for (auto d : dims) {
      if (!outputs[0]->hasDim(d)) {
        reduce_dims.insert(d);
      }
    }
    return reduce_dims;
  }
  std::set<Dim *> getShapeDims(int i) override {
    std::set<Dim *> shape_dims;
    for (auto d : dims) {
      if (inputs[0]->hasDim(d) && inputs[1]->hasDim(d) && outputs[0]->hasDim(d)) {
      } else if (inputs[i]->hasDim(d) && outputs[0]->hasDim(d)) {
        shape_dims.insert(d);
      }
    }
    return shape_dims;
  }
protected:
  void analyzeMemAccess(const std::vector<Dim *> &dims_order) override {
    long op_access_times = 0;
    long op_access_volume = 0;
    for (auto input : inputs) {
      if (input->isFused()) {
        if (input->isIO() || input->isExternal()) {
          long access_times = input->getBlocks(this);
          input->updateAccessTimesStr(this, "fused io(" + std::to_string(access_times) + ")");
          input->updateAccessTimes(this, access_times);
          op_access_times += access_times;
          op_access_volume += access_times * input->getBlockSize(this);
        } else {
          input->updateAccessTimesStr(this, "fused");
          input->updateAccessTimes(this, 0);
        }
        input->isReused(true);
      } else {
        input->isReused(false);
        std::string access_times_str = "1";
        long access_times = 1;
        bool access_tensor = false;
        for (auto d : dims_order) {
          if (this->hasDim(d)) {
            if (input->hasDim(d)) {
              access_tensor = true;
            }
            if (access_tensor) {
              access_times *= d->getBlocks(this);
              access_times_str.append(
                  " * " + d->getName() + "(" + std::to_string(d->getBlocks(this)) + ")");
            } else {
              input->isReused(true);
            }
          }
        }
        input->updateAccessTimesStr(this, access_times_str);
        input->updateAccessTimes(this, access_times);
        op_access_times += access_times;
        op_access_volume += access_times * input->getBlockSize(this);
      }
    }
    for (auto output : outputs) {
      if (output->isFused()) {
        if (is_with_bias())
          op_access_volume += output->getSize();
        if (output->isIO() || output->isExternal()) {
          long access_times = output->getBlocks(this);
          output->updateAccessTimesStr(this, "fused io(" + std::to_string(access_times) + ")");
          output->updateAccessTimes(this, access_times);
          op_access_times += access_times;
          op_access_volume += access_times * output->getBlockSize(this);
        } else {
          output->updateAccessTimesStr(this, "fused");
          output->updateAccessTimes(this, 0);
        }
        output->isReused(true);
      } else {
        if (!is_with_bias())
          op_access_volume -= output->getSize();
        output->isReused(false);
        std::string access_times_str;
        long access_times = 1;
        bool access_tensor = false;
        for (auto d : dims_order) {
          if (this->hasDim(d)) {
            if (output->hasDim(d)) {
              access_tensor = true;
            }
            if (access_tensor) {
              access_times *= d->getBlocks(this);
              access_times_str.append(
                  " * " + d->getName() + "(" + std::to_string(d->getBlocks(this)) + ")");
            } else {
              output->isReused(true);
            }
          }
        }
        access_times *= 2;
        // This is OK for mip, because this only mismatch a const, which would not affect result.
        access_times_str = "2" + access_times_str;
        output->updateAccessTimesStr(this, access_times_str);
        output->updateAccessTimes(this, access_times);
        op_access_times += access_times;
        op_access_volume += access_times * output->getBlockSize(this);
      }
    }
    access_times = op_access_times;
    access_volume = op_access_volume;
  }

  void analyzeMemFootprint(const std::vector<Dim *> &dims_order) override {
    std::vector<Tensor *> tensors;
    tensors.reserve(inputs.size() + outputs.size());
    auto a = inputs;
    auto b = outputs;
    tensors.insert(tensors.end(), a.begin(), a.end());
    tensors.insert(tensors.end(), b.begin(), b.end());
    for (auto t : tensors) {
      if (t->isFused()) {
        long tensor_footprint = t->getBlockSize(this);
        std::string tensor_footprint_str = t->getBlockSizeStr(this);
        bool tmp = true;
        std::vector<Dim *> reverse_dims_order = dims_order;
        std::reverse(reverse_dims_order.begin(), reverse_dims_order.end());
        for (auto d : reverse_dims_order) {
          if (this->hasDim(d)) {
            if (!t->hasDim(d)) {
              tmp = false;
            }
            if (!tmp && t->hasDim(d)) {
              tensor_footprint *= d->getBlocks(this);
              tensor_footprint_str.append(
                  " * " + d->getName() + "(" + std::to_string(d->getBlocks(this)) + ")");
              t->addExpandDims(d);
            }
          }
        }
        t->updateMemFootprintStr(this, tensor_footprint_str);
        t->updateMemFootprint(this, tensor_footprint);
      } else {
        if (t->isReused()) {
          t->updateMemFootprintStr(this, t->getBlockSizeStr(this));
          t->updateMemFootprint(this, t->getBlockSize(this));
        } else {
          if (Options::store_whole_block) {
            t->updateMemFootprintStr(this, t->getBlockSizeStr(this));
            t->updateMemFootprint(this, t->getBlockSize(this));
          } else {
            assert(0);
          }
        }
      }
    }
  }
};

// FIXME: BatchTensor and BatchMatrixMul are not maintained
class BatchMatrixMul : public MatrixMul {
public:
  BatchMatrixMul(BatchTensor2D *input1, BatchTensor2D *input2, BatchTensor2D *output) {
    inputs.resize(2);
    outputs.resize(1);
    setInputTensor(0, input1);
    setInputTensor(1, input2);
    setOutputTensor(0, output);
  }
  BatchMatrixMul() {
    inputs.resize(2);
    outputs.resize(1);
  }
  void setInputTensor(long i, BatchTensor2D *t) {
    assert(i < inputs.size());
    inputs[i] = t;
    for (auto d : t->getDims()) {
      dims.insert(d);
      d->setBlockSize(this, 1);
    }
    t->addRelatedOperator(this, "input");
  };
  void setOutputTensor(long i, BatchTensor2D *t) {
    assert(i < outputs.size());
    outputs[i] = t;
    for (auto d : t->getDims()) {
      dims.insert(d);
      d->setBlockSize(this, 1);
    }
    t->addRelatedOperator(this, "output");
  };
  void setPar(long p) {
    par = p;
  }
  [[nodiscard]] long getPar() const {
    return par;
  }
  std::string toString() override {
    std::string str;
    str = "parallel(" + std::to_string(par) + ") " + outputs[0]->toString(this) + " = "
        + inputs[0]->toString(this) + " * " + inputs[1]->toString(this);
    return str;
  }
  long getOpsNum() override {
    long ops = par * MatrixMul::getOpsNum();
    return ops;
  }

  void analyzeMemFootprint(const std::vector<Dim *> &dims_order) override {
    long num_reuse_tensors = this->getReuseTensorNum();
    std::vector<BatchTensor2D *> tensors;
    tensors.reserve(inputs.size() + outputs.size());
    for (auto i : inputs) {
      tensors.insert(tensors.end(), dynamic_cast<BatchTensor2D *>(i));
    }
    for (auto o : outputs) {
      tensors.insert(tensors.end(), dynamic_cast<BatchTensor2D *>(o));
    }
    for (auto t : tensors) {
      if (t->isFused()) {
        long tensor_footprint = t->getBlockSize(this);
        std::string tensor_footprint_str = t->getBlockSizeStr(this);
        bool tmp = true;
        std::vector<Dim *> reverse_dims_order = dims_order;
        std::reverse(reverse_dims_order.begin(), reverse_dims_order.end());
        for (auto d : reverse_dims_order) {
          if (this->hasDim(d)) {
            if (!t->hasDim(d)) {
              tmp = false;
            }
            if (!tmp && t->hasDim(d)) {
              tensor_footprint *= d->getBlocks(this);
              tensor_footprint_str.append(
                  " * " + d->getName() + "(" + std::to_string(d->getBlocks(this)) + ")");
            }
          }
        }
        t->updateMemFootprintStr(this, tensor_footprint_str);
        t->updateMemFootprint(this, tensor_footprint);
      } else {
        if (t->isReused()) {
          t->updateMemFootprintStr(this, t->getBlockSizeStr(this));
          t->updateMemFootprint(this, t->getBlockSize(this));
        } else {
          if (Options::store_whole_block) {
            t->updateMemFootprintStr(this, t->getBlockSizeStr(this));
            t->updateMemFootprint(this, t->getBlockSize(this));
          } else {
            assert(0);
          }
        }
      }
    }
  }

protected:
  long par{};
};

}

#endif //MMCHAIN_ANALYSIS_SRC_TENSOR_OPERATOR_H
