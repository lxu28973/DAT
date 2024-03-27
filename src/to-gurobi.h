//
// Created by lxu on 23-6-21.
//

#ifndef MMCHAIN_ANALYSIS_SRC_TO_GUROBI_H
#define MMCHAIN_ANALYSIS_SRC_TO_GUROBI_H

#include "operator-chain.h"
#include <cfloat>
#include <limits>
#include <sstream>
#include "gurobi_c++.h"

namespace DAT {

bool isInteger(const std::string &str) {
  if (str.empty()) {
    return false;
  }
  return std::all_of(str.begin(), str.end(), [](char c) { return std::isdigit(c); });
}

std::string removeParenthesesInfo(const std::string &str) {
  std::string result;
  bool withinParentheses = false;

  for (char c : str) {
    if (c == '(') {
      withinParentheses = true;
      continue;
    } else if (c == ')') {
      withinParentheses = false;
      continue;
    }

    if (!withinParentheses)
      result += c;
  }

  return result;
}

std::vector<std::string> convertStringToVector(const std::string &str) {
  std::vector<std::string> result;
  std::istringstream iss(str);
  std::string token;

  while (std::getline(iss, token, '*')) {
    // Trim leading and trailing whitespaces from each token
    size_t start = token.find_first_not_of(' ');
    size_t end = token.find_last_not_of(' ');
    std::string trimmedToken = token.substr(start, end - start + 1);

    result.push_back(trimmedToken);
  }

  return result;
}

std::string removeBlocksizeSuffix(const std::string &str) {
  if (str.length() >= 10 && str.substr(str.length() - 10) == "_blocksize") {
    return str.substr(0, str.length() - 10);
  }
  return str;
}

std::vector<std::string> changeVarToConst(const std::vector<std::string> &mul_strs,
                                          const std::string &dim_name,
                                          TensorOperator *to,
                                          Tensor *t) {
  std::vector<std::string> ret_strs;
  for (const std::string &str : mul_strs) {
    if (str.compare(0, dim_name.length(), dim_name) == 0) {
      if (str.length() >= 10 && str.substr(str.length() - 10) == "_blocksize") {
        ret_strs.push_back(std::to_string(t->getDim(dim_name)->getBlockSize(to)));
      } else {
        ret_strs.push_back(std::to_string(t->getDim(dim_name)->getBlocks(to)));
      }
    } else {
      ret_strs.push_back(str);
    }
  }
  return ret_strs;
}

GRBQuadExpr convertToLocalGRBQuad(const std::vector<std::string> &mul_strs,
                                  TensorOperator *to,
                                  Tensor *t,
                                  std::map<std::string, GRBVar> grb_vars_map) {

  GRBQuadExpr local_obj;
  GRBVar mul_vars[2];
  long mul_vars_index = 0;
  double const_base = 1;
  std::set<std::string> withBlocksize;
  std::set<std::string> withoutBlocksize;
  for (const std::string &str : mul_strs) {
    if (isInteger(str)) {
      const_base *= std::stoi(str);
    } else if (str.length() >= 10 && str.substr(str.length() - 10) == "_blocksize") {
      withBlocksize.insert(removeBlocksizeSuffix(str));
    } else {
      withoutBlocksize.insert(str);
    }
  }
  for (const auto &d : withoutBlocksize) {
    if (withBlocksize.count(d)) {
      const_base = const_base * t->getDim(d)->getSize();
      withBlocksize.erase(d);
    } else {
      assert(mul_vars_index < 2 && "should not exceed quad");
      mul_vars[mul_vars_index] = grb_vars_map.at(to->getName() + '_' + d + "_bn");
      mul_vars_index++;
    }
  }
  for (const auto &dbs : withBlocksize) {
    assert(mul_vars_index < 2 && "should not exceed quad");
    mul_vars[mul_vars_index] = grb_vars_map.at(to->getName() + '_' + dbs + "_bs");
    mul_vars_index++;
  }
  if (mul_vars_index == 2) {
    local_obj = mul_vars[0] * mul_vars[1];
  } else if (mul_vars_index == 1) {
    local_obj = mul_vars[0];
  } else {
    local_obj = 1;
  }
  local_obj = const_base * local_obj;

  return local_obj;
}

// The return obj may not equal to the mem access. They may differ by a constant.
double mipBlockSize(OperatorChain *op_chain,
                         long mem_constraint,
                         bool print_info = false,
                         const std::string &lp_file = "") {
  double ret_obj = -1;
  GRBVar *grbvars = nullptr;
  const std::set<Dim *> op_chain_dims = op_chain->getDims();
  const std::set<TensorOperator *> op_chain_ops = op_chain->getOperators();
  auto dim_num = op_chain_dims.size();
  auto op_num = op_chain_ops.size();
  grbvars = new GRBVar[op_num * dim_num * 2];
  try {

    // Create an environment
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "");
    if (!print_info)
      env.set("OutputFlag", "0");
    env.start();

    // Create an empty model
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_NonConvex, 2);

    // Create variables
    std::map<std::string, GRBVar> grb_vars_map;
    long var_i = 0;
    for (auto op : op_chain_ops)
    for (auto d : op->getDims()) {
      std::string grbvar_name = op->getName() + '_' + d->getName();
      std::string grbvar_bn_name = grbvar_name + "_bn";
      std::string grbvar_bs_name = grbvar_name + "_bs";
      grbvars[var_i] = model.addVar(1.0, d->getSize(), 0.0, GRB_INTEGER, grbvar_bn_name);
      grb_vars_map[grbvar_bn_name] = grbvars[var_i];
      grbvars[var_i + 1] = model.addVar(1.0, d->getSize(), 0.0, GRB_INTEGER, grbvar_bs_name);
      grb_vars_map[grbvar_bs_name] = grbvars[var_i + 1];
      model.addQConstr(grbvars[var_i] * grbvars[var_i + 1] == d->getSize(), grbvar_name + "_c");
      var_i += 2;
    }

    // Set objective
    GRBQuadExpr obj = 0;
    for (auto op : op_chain->getOperators()) {
      std::vector<Tensor *> tensors = op->getInputTensors();
      std::vector<Tensor *> outputs = op->getOutputTensors();
      tensors.insert(tensors.end(), outputs.begin(), outputs.end());
      for (auto t : tensors) {
        if (!t->isFused()) {
          long const_base;
          std::string access_volume_str = removeParenthesesInfo(t->getAccessVolumeStr(op)[op]);
          std::vector<std::string> mul_strs = convertStringToVector(access_volume_str);
          if (mul_strs.front() == "1") {
            const_base = 1;
          } else if (mul_strs.front() == "2") {
            const_base = 2;
          } else {
            assert(false && "access volume should begin with \"1 *\" or \"2 *\"");
          }
          mul_strs.erase(mul_strs.begin());
          GRBQuadExpr local_obj = const_base * convertToLocalGRBQuad(mul_strs, op, t, grb_vars_map);
          obj = obj + local_obj;
        } else if (t->isIO() || t->isExternal()) {
          obj = obj + t->getSize();
        }
      }
    }
    model.setObjective(obj, GRB_MINIMIZE);

    // Add constraint
    auto addGRBConstConstraint = [&](TensorOperator *op, const std::string &dim_name) {
      std::string grbvar_bn_name = op->getName() + "_" + dim_name + "_bn";
      std::string grbvar_bs_name = op->getName() + "_" + dim_name + "_bs";

      model.addConstr(grb_vars_map.at(grbvar_bn_name) == op_chain->getDim(dim_name)->getBlocks(op),
                      grbvar_bn_name);
      model.addConstr(grb_vars_map.at(grbvar_bs_name) == op_chain->getDim(dim_name)->getBlockSize(op),
                      grbvar_bs_name);
    };
    for (auto op : op_chain_ops) {
      if (op->getDim("bsc")) {
        addGRBConstConstraint(op, "bsc");
      }
      if (op->getDim("hsc")) {
        addGRBConstConstraint(op, "hsc");
      }
    }
    for (auto op : op_chain->getOperators()) {
      if (Options::enable_compute_utilization_constraint) {
        for (auto d : op->getDims()) {
          std::string name = d->getName();
          if (name == "bsc" || name == "hsc") {
          } else {
            std::string bs_name = op->getName() + "_" + name + "_bs";
            model.addConstr(grb_vars_map.at(bs_name) >= 16);
          }
        }
        GRBQuadExpr compute_util_constraint = 0;
        for (auto t : op->getTensors()) {
          GRBQuadExpr tensor_compute_util_constraint = 1;
          std::vector<std::string> mul_names;
          double mul_const = 1;
          for (auto d : t->getDims()) {
            std::string name = d->getName();
            if (name == "bsc" || name == "hsc") {
              mul_const *= d->getBlockSize(op);
            } else {
              std::string bs_name = op->getName() + "_" + name + "_bs";
              mul_names.push_back(bs_name);
            }
          }
          if (mul_names.size() == 2) {
            tensor_compute_util_constraint =
                mul_const * grb_vars_map.at(mul_names[0]) * grb_vars_map.at(mul_names[1]);
          } else {
            assert(0);
          }
          compute_util_constraint = compute_util_constraint + tensor_compute_util_constraint;
        }
        assert(op->getTensors().size() == 3);
        model.addQConstr(compute_util_constraint >= Options::compute_power);
      }
    }
    for (const auto& og : op_chain->getOperatorGroups()) {
      GRBQuadExpr local_constraint = 0;
      std::vector<Tensor *> op_tensors;
      for (auto op : og) {
        std::vector<Tensor *> inputs = op->getInputTensors();
        std::vector<Tensor *> outputs = op->getOutputTensors();
        std::set<Tensor *> envs = op->getEnvTensors();
        op_tensors.insert(op_tensors.end(), inputs.begin(), inputs.end());
        op_tensors.insert(op_tensors.end(), outputs.begin(), outputs.end());
        op_tensors.insert(op_tensors.end(), envs.begin(), envs.end());
      }
      std::set<Tensor *> tensor_sets(op_tensors.begin(), op_tensors.end());
      for (auto t : tensor_sets) {
        std::vector<std::string> mul_strs;
        bool added = false;
        for (const auto &mfs : t->getMemFootprintStr()) {
          std::vector<std::string>
              new_mul_strs = convertStringToVector(removeParenthesesInfo(mfs.second));
          if (new_mul_strs.size() > mul_strs.size()) {
            mul_strs = new_mul_strs;
          }
        }
        for (auto op_m : t->getRelatedOperator()) {
          if (og.count(op_m.first)) {
            if (op_m.first->getDim("bsc")) {
              mul_strs = changeVarToConst(mul_strs, "bsc", op_m.first, t);
            }
            if (op_m.first->getDim("hsc")) {
              mul_strs = changeVarToConst(mul_strs, "hsc", op_m.first, t);
            }
            if (!added) {
              local_constraint = local_constraint + convertToLocalGRBQuad(mul_strs, op_m.first, t, grb_vars_map);
              added = true;
            }
          }
        }
      }
      model.addQConstr(local_constraint <= mem_constraint, "_c");
    }
    for (auto t : op_chain->getInternalTensors()) {
      std::set<Dim *> expand_dims = t->getExpandDims();
      for (auto d : t->getDims()) {
        if (!expand_dims.count(d)) {
          std::vector<TensorOperator *> asInput;
          std::vector<TensorOperator *> asOutput;
          for(const auto& to : t->getRelatedOperator()) {
            if (to.second == "input") {
              asInput.push_back(to.first);
            } else if (to.second == "output") {
              asOutput.push_back(to.first);
            }
          }
          assert(asInput.size() == 1);
          for (auto op1 : asInput)
            for (auto op2 : asOutput) {
              model.addConstr(grb_vars_map.at(op1->getName() + '_' + d->getName() + "_bs") == grb_vars_map.at(op2->getName() + '_' + d->getName() + "_bs"));
            }
        }
      }
    }

    if (!lp_file.empty()) {
      model.write(lp_file);
    }
    // Optimize model
    model.optimize();

    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      for (auto op : op_chain_ops)
      for (auto d : op->getDims()) {
        std::string grbvar_name = op->getName() + "_" + d->getName();
        std::string grbvar_bn_name = grbvar_name + "_bn";
        std::string grbvar_bs_name = grbvar_name + "_bs";
        d->setBlockSize(op,
            std::round(grb_vars_map.at(grbvar_bs_name).get(GRB_DoubleAttr_X)));
      }
      if (print_info) {
        for (long i = 0; i < dim_num * 2; i++) {
          std::cout << grbvars[i].get(GRB_StringAttr_VarName) << " "
                    << grbvars[i].get(GRB_DoubleAttr_X) << std::endl;
        }

        std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
      }
    } else if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
      if (print_info) {
        std::cout << "This is infeasible for mem size " << mem_constraint << ": "
                  << op_chain->toString() << std::endl;
      }
      delete[]grbvars;
      return FLT_MAX;
    } else {
      std::cerr << "Get non optimal results, please check. mem_size: " << mem_constraint << ": "
                << op_chain->toString() << std::endl;
    }

    ret_obj = model.get(GRB_DoubleAttr_ObjVal);

    delete[]grbvars;

    return ret_obj;

  } catch (const GRBException &e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
    return FLT_MAX;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
    return FLT_MAX;
  }

}

}

#endif //MMCHAIN_ANALYSIS_SRC_TO_GUROBI_H
