//===- FuncEmissionPatterns.cpp - Func Dialect Emission Patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the func dialect.
//
//===----------------------------------------------------------------------===//

#include "FuncEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir::func;
using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit a func.func function. Users of the function arguments request an
/// expression to be inlined and we simply return the name of the argument.This
/// name has to be passed to this emission pattern via an array of strings
/// attribute called 'argNames' because the emitter cannot do any name uniquing
/// as it just emits the IR statement by statement. However, relying on an
/// attribute for the argument names also has the advantage that the names
/// can be preserved during a lowering pipeline and upstream passes have more
/// control on how the arguments should be named (e.g. when they create a
/// function and have some context to assign better names).
struct FuncEmitter : OpEmissionPattern<FuncOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value) override {
    if (!value.isa<BlockArgument>())
      return {};

    if (auto funcOp = value.getParentRegion()->getParentOfType<FuncOp>()) {
      if (auto argNames = funcOp->getAttrOfType<ArrayAttr>("argNames")) {
        if (argNames.size() == funcOp.getNumArguments() &&
            llvm::all_of(argNames,
                         [](Attribute arg) { return arg.isa<StringAttr>(); }))
          return Precedence::VAR;
      }
    }

    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto func = value.getParentRegion()->getParentOfType<FuncOp>();
    for (size_t i = 0, e = func.getNumArguments(); i < e; ++i) {
      if (func.getArgument(i) == value) {
        p << func->getAttr("argNames")
                 .cast<ArrayAttr>()[i]
                 .cast<StringAttr>()
                 .getValue();
        return;
      }
    }
  }

  bool matchStatement(Operation *op) override {
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      if (funcOp.getFunctionType().getNumResults() > 1)
        return false;

      if (funcOp.getNumArguments() == 0)
        return true;

      if (auto argNames = funcOp->getAttrOfType<ArrayAttr>("argNames"))
        return argNames.size() == funcOp.getNumArguments() &&
               llvm::all_of(argNames, [](Attribute arg) {
                 return arg.isa<StringAttr>();
               });
    }

    return false;
  }

  void emitStatement(FuncOp func, EmissionPrinter &p) override {
    // Emit a newline at the start to ensure an empty line before the function
    // for better readability.
    p << "\n";

    // Emit return type.
    if (func.getFunctionType().getNumResults() == 0)
      p << "void";
    else
      p.emitType(func.getFunctionType().getResult(0));

    p << " " << func.getSymName() << "(";

    // Emit argument list.
    for (size_t i = 0, e = func.getFunctionType().getNumInputs(); i < e; ++i) {
      if (i > 0)
        p << ", ";
      p.emitType(func.getFunctionType().getInput(i));
      p << " "
        << func->getAttr("argNames")
               .cast<ArrayAttr>()[i]
               .cast<StringAttr>()
               .getValue();
    }

    p << ")";

    // Emit body when present.
    if (func.isDeclaration()) {
      p << ";\n";
    } else {
      p << " ";
      p.emitRegion(func.getRegion());
    }
  }
};

/// Emit a func.call operation. Only zero or one result values are allowed.
/// If it has no result, it is treated as a statement, otherwise as an
/// expression that will always be inlined. That means, an emission preparation
/// pass has to insert a VariableOp to bind the call result to such that
/// reordering of the call cannot lead to incorrectness due to interference of
/// side-effects.
class CallEmitter : public OpEmissionPattern<CallOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (auto callOp = value.getDefiningOp<CallOp>()) {
      if (callOp->getNumResults() > 1)
        return {};

      return Precedence::FUNCTION_CALL;
    }
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCall(value.getDefiningOp<CallOp>(), p);
  }

  void emitStatement(CallOp op, EmissionPrinter &p) override {
    // If the call returns values, then it is treated as an expression rather
    // than a statement.
    if (op.getNumResults() > 0)
      return;

    printCall(op, p);
    p << ";\n";
  }

private:
  void printCall(CallOp op, EmissionPrinter &p) {
    p << op.getCallee() << "(";
    llvm::interleaveComma(op.getOperands(), p, [&](auto arg) {
      p.getInlinable(arg).emitWithParensOnLowerPrecedence(Precedence::COMMA);
    });
    p << ")";
  }
};

/// Emit a func.call_indirect operation. Only zero or one result values are
/// allowed. If it has no result, it is treated as a statement, otherwise as an
/// expression that will always be inlined. That means, an emission preparation
/// pass has to insert a VariableOp to bind the call result to such that
/// reordering of the call cannot lead to incorrectness due to interference of
/// side-effects.
class CallIndirectEmitter : public OpEmissionPattern<CallIndirectOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (auto callOp = value.getDefiningOp<CallIndirectOp>()) {
      if (callOp->getNumResults() > 1)
        return {};

      return Precedence::FUNCTION_CALL;
    }

    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCall(value.getDefiningOp<CallIndirectOp>(), p);
  }

  void emitStatement(CallIndirectOp op, EmissionPrinter &p) override {
    // If the call returns values, then it is treated as an expression rather
    // than a statement.
    if (op.getNumResults() > 0)
      return;

    printCall(op, p);
    p << ";\n";
  }

private:
  void printCall(CallIndirectOp op, EmissionPrinter &p) {
    p.getInlinable(op.getCallee())
        .emitWithParensOnLowerPrecedence(Precedence::FUNCTION_CALL);
    p << "(";
    llvm::interleaveComma(op.getCalleeOperands(), p, [&](auto arg) {
      p.getInlinable(arg).emitWithParensOnLowerPrecedence(Precedence::COMMA);
    });
    p << ")";
  }
};

/// Emit a func.return operation.
struct ReturnEmitter : OpEmissionPattern<ReturnOp> {
  using OpEmissionPattern::OpEmissionPattern;

  bool matchStatement(Operation *op) override {
    return isa<ReturnOp>(op) && cast<ReturnOp>(op)->getNumOperands() <= 1;
  }

  void emitStatement(ReturnOp op, EmissionPrinter &p) override {
    p << "return";
    if (op->getNumOperands() == 1) {
      p << " ";
      p.getInlinable(op.getOperand(0)).emit();
    }
    p << ";\n";
  }
};

/// Emit a func.constant operation.
struct FuncConstantEmitter : OpEmissionPattern<ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ConstantOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp<ConstantOp>().getValue();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateFuncOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<FuncEmitter, FuncConstantEmitter, ReturnEmitter, CallEmitter,
               CallIndirectEmitter>(context);
}
