//===- EmitOMIR.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EmitOMIR pass.
//
//===----------------------------------------------------------------------===//

#include "AnnotationDetails.h"
#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/CircuitNamespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "omir"

using namespace circt;
using namespace firrtl;
using mlir::LocationAttr;
using mlir::UnitAttr;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Information concerning a tracker in the IR.
struct Tracker {
  /// The unique ID of this tracker.
  IntegerAttr id;
  /// The operation onto which this tracker was annotated.
  Operation *op;
  /// If this tracker is non-local, this is the corresponding anchor.
  NonLocalAnchor nla;
};

class EmitOMIRPass : public EmitOMIRBase<EmitOMIRPass> {
public:
  using EmitOMIRBase::outputFilename;

private:
  void runOnOperation() override;

  void emitSourceInfo(Location input, SmallString<64> &into);
  void emitOMNode(Attribute node, llvm::json::OStream &jsonStream);
  void emitOMField(Identifier fieldName, DictionaryAttr field,
                   llvm::json::OStream &jsonStream);
  void emitValue(Attribute node, llvm::json::OStream &jsonStream);
  void emitTrackedTarget(DictionaryAttr node, llvm::json::OStream &jsonStream);

  SmallString<8> addSymbol(FlatSymbolRefAttr symbol) {
    unsigned id;
    auto it = symbolIndices.find(symbol);
    if (it != symbolIndices.end()) {
      id = it->second;
    } else {
      id = symbols.size();
      symbols.push_back(symbol);
      symbolIndices.insert({symbol, id});
    }
    SmallString<8> str;
    ("{{" + Twine(id) + "}}").toVector(str);
    return str;
  }
  SmallString<8> addSymbol(StringAttr symbolName) {
    return addSymbol(FlatSymbolRefAttr::get(symbolName));
  }
  SmallString<8> addSymbol(Operation *op) {
    return addSymbol(SymbolTable::getSymbolName(op));
  }

  /// Whether any errors have occurred in the current `runOnOperation`.
  bool anyFailures;
  /// A symbol table for the current operation.
  SymbolTable *symtbl;
  /// OMIR target trackers gathered in the current operation, by tracker ID.
  DenseMap<Attribute, Tracker> trackers;
  /// The list of symbols to be interpolated in the verbatim JSON. This gets
  /// populated as the JSON is constructed and module and instance names are
  /// collected.
  SmallVector<Attribute> symbols;
  SmallDenseMap<Attribute, unsigned> symbolIndices;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void EmitOMIRPass::runOnOperation() {
  MLIRContext *context = &getContext();
  anyFailures = false;
  symtbl = nullptr;
  trackers.clear();
  symbols.clear();
  symbolIndices.clear();
  CircuitOp circuitOp = getOperation();

  // Gather the relevant annotations from the circuit. On the one hand these are
  // all the actual `OMIRAnnotation`s that need processing and emission, as well
  // as an optional `OMIRFileAnnotation` that overrides the default OMIR output
  // file.
  SmallVector<ArrayRef<Attribute>> annoNodes;
  Optional<StringRef> outputFilename = {};

  AnnotationSet::removeAnnotations(circuitOp, [&](Annotation anno) {
    if (anno.isClass(omirFileAnnoClass)) {
      auto pathAttr = anno.getMember<StringAttr>("filename");
      if (!pathAttr) {
        circuitOp.emitError(omirFileAnnoClass)
            << " annotation missing `filename` string attribute";
        anyFailures = true;
        return true;
      }
      LLVM_DEBUG(llvm::dbgs() << "- OMIR path: " << pathAttr << "\n");
      outputFilename = pathAttr.getValue();
      return true;
    }
    if (anno.isClass(omirAnnoClass)) {
      auto nodesAttr = anno.getMember<ArrayAttr>("nodes");
      if (!nodesAttr) {
        circuitOp.emitError(omirAnnoClass)
            << " annotation missing `nodes` array attribute";
        anyFailures = true;
        return true;
      }
      LLVM_DEBUG(llvm::dbgs() << "- OMIR: " << nodesAttr << "\n");
      annoNodes.push_back(nodesAttr.getValue());
      return true;
    }
    return false;
  });
  if (anyFailures)
    return signalPassFailure();

  // Traverse the IR and collect all tracker annotations that were previously
  // scattered into the circuit.
  SymbolTable currentSymtbl(circuitOp);
  symtbl = &currentSymtbl;
  circuitOp.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      if (!anno.isClass(omirTrackerAnnoClass))
        return false;
      Tracker tracker;
      tracker.op = op;
      tracker.id = anno.getMember<IntegerAttr>("id");
      if (!tracker.id) {
        op->emitError(omirTrackerAnnoClass)
            << " annotation missing `id` integer attribute";
        anyFailures = true;
        return true;
      }
      if (auto nlaSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
        tracker.nla =
            dyn_cast_or_null<NonLocalAnchor>(symtbl->lookup(nlaSym.getAttr()));
      trackers.insert({tracker.id, tracker});
      return true;
    });
  });

  // If an OMIR output filename has been specified as a pass parameter, override
  // whatever the annotations have configured. If neither are specified we just
  // bail.
  if (!this->outputFilename.empty())
    outputFilename = this->outputFilename;
  if (!outputFilename) {
    LLVM_DEBUG(llvm::dbgs() << "Not emitting OMIR because no annotation or "
                               "pass parameter specified an output file\n");
    markAllAnalysesPreserved();
    return;
  }

  // Build the output JSON.
  std::string jsonBuffer;
  llvm::raw_string_ostream jsonOs(jsonBuffer);
  llvm::json::OStream json(jsonOs, 2);
  json.array([&] {
    for (auto nodes : annoNodes) {
      for (auto node : nodes) {
        emitOMNode(node, json);
        if (anyFailures)
          return;
      }
    }
  });
  if (anyFailures)
    return signalPassFailure();

  // Emit the OMIR JSON as a verbatim op.
  auto builder = OpBuilder(circuitOp);
  builder.setInsertionPointAfter(circuitOp);
  auto verbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), jsonBuffer);
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, *outputFilename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  verbatimOp.symbolsAttr(ArrayAttr::get(context, symbols));
}

/// Emit a source locator into a string, for inclusion in the `info` field of
/// `OMNode` and `OMField`.
void EmitOMIRPass::emitSourceInfo(Location input, SmallString<64> &into) {
  into.clear();
  input->walk([&](Location loc) {
    if (FileLineColLoc fileLoc = loc.dyn_cast<FileLineColLoc>()) {
      into.append(into.empty() ? "@[" : " ");
      auto twine = Twine(fileLoc.getFilename()) + " " +
                   Twine(fileLoc.getLine()) + ":" + Twine(fileLoc.getColumn());
      twine.toVector(into);
    }
    return WalkResult::advance();
  });
  if (!into.empty())
    into.append("]");
}

/// Emit an entire `OMNode` as JSON.
void EmitOMIRPass::emitOMNode(Attribute node, llvm::json::OStream &jsonStream) {
  auto dict = node.dyn_cast<DictionaryAttr>();
  if (!dict) {
    getOperation()
            .emitError("OMNode must be a dictionary")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return;
  }

  // Extract the `info` field and serialize the location.
  SmallString<64> info;
  if (auto infoAttr = dict.getAs<LocationAttr>("info"))
    emitSourceInfo(infoAttr, info);
  if (anyFailures)
    return;

  // Extract the `id` field.
  auto idAttr = dict.getAs<StringAttr>("id");
  if (!idAttr) {
    getOperation()
            .emitError("OMNode missing `id` string field")
            .attachNote(getOperation().getLoc())
        << dict;
    anyFailures = true;
    return;
  }

  // Extract and order the fields of this node.
  SmallVector<std::tuple<unsigned, Identifier, DictionaryAttr>> orderedFields;
  if (auto fieldsDict = dict.getAs<DictionaryAttr>("fields")) {
    for (auto nameAndField : fieldsDict.getValue()) {
      auto fieldDict = nameAndField.second.dyn_cast<DictionaryAttr>();
      if (!fieldDict) {
        getOperation()
                .emitError("OMField must be a dictionary")
                .attachNote(getOperation().getLoc())
            << nameAndField.second;
        anyFailures = true;
        return;
      }

      unsigned index = 0;
      if (auto indexAttr = fieldDict.getAs<IntegerAttr>("index"))
        index = indexAttr.getValue().getLimitedValue();

      orderedFields.push_back({index, nameAndField.first, fieldDict});
    }
    llvm::sort(orderedFields,
               [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });
  }

  jsonStream.object([&] {
    jsonStream.attribute("info", info);
    jsonStream.attribute("id", idAttr.getValue());
    jsonStream.attributeArray("fields", [&] {
      for (auto &orderedField : orderedFields) {
        emitOMField(std::get<1>(orderedField), std::get<2>(orderedField),
                    jsonStream);
        if (anyFailures)
          return;
      }
    });
  });
}

/// Emit a single `OMField` as JSON. This expects the field's name to be
/// provided from the outside, for example as the field name that this attribute
/// has in the surrounding dictionary.
void EmitOMIRPass::emitOMField(Identifier fieldName, DictionaryAttr field,
                               llvm::json::OStream &jsonStream) {
  // Extract the `info` field and serialize the location.
  auto infoAttr = field.getAs<LocationAttr>("info");
  SmallString<64> info;
  if (infoAttr)
    emitSourceInfo(infoAttr, info);
  if (anyFailures)
    return;

  jsonStream.object([&] {
    jsonStream.attribute("info", info);
    jsonStream.attribute("name", fieldName.strref());
    jsonStream.attributeBegin("value");
    emitValue(field.get("value"), jsonStream);
    jsonStream.attributeEnd();
  });
}

void EmitOMIRPass::emitValue(Attribute node, llvm::json::OStream &jsonStream) {
  // Handle the null case.
  if (!node || node.isa<UnitAttr>())
    return jsonStream.value(nullptr);

  // Handle the trivial cases where the OMIR serialization simply uses the
  // builtin JSON types.
  if (auto attr = node.dyn_cast<BoolAttr>())
    return jsonStream.value(attr.getValue()); // OMBoolean
  if (auto attr = node.dyn_cast<IntegerAttr>()) {
    // CAVEAT: We expect these integers to come from an OMIR file that is
    // initially read in from JSON, where they are i32 or i64, so this should
    // yield a valid value. However, a user could cook up an arbitrary precision
    // integer attr in MLIR input and then subtly break the JSON spec.
    SmallString<16> val;
    attr.getValue().toStringSigned(val);
    return jsonStream.rawValue(val); // OMInt
  }
  if (auto attr = node.dyn_cast<FloatAttr>()) {
    // CAVEAT: We expect these floats to come from an OMIR file that is
    // initially read in from JSON, where they are f32 or f64, so this should
    // yield a valid value. However, a user could cook up an arbitrary precision
    // float attr in MLIR input and then subtly break the JSON spec.
    SmallString<16> val;
    attr.getValue().toString(val);
    return jsonStream.rawValue(val); // OMDouble
  }

  // Handle aggregate types.
  if (auto attr = node.dyn_cast<ArrayAttr>()) {
    jsonStream.array([&] {
      for (auto element : attr.getValue()) {
        emitValue(element, jsonStream);
        if (anyFailures)
          return;
      }
    });
    return;
  }
  if (auto attr = node.dyn_cast<DictionaryAttr>()) {
    // Handle targets that have a corresponding tracker annotation in the IR.
    if (attr.getAs<UnitAttr>("omir.tracker"))
      return emitTrackedTarget(attr, jsonStream);

    // Handle regular dictionaries.
    jsonStream.object([&] {
      for (auto field : attr.getValue()) {
        jsonStream.attributeBegin(field.first);
        emitValue(field.second, jsonStream);
        jsonStream.attributeEnd();
        if (anyFailures)
          return;
      }
    });
    return;
  }

  // The remaining types are all simple string-encoded pass-through cases.
  if (auto attr = node.dyn_cast<StringAttr>()) {
    StringRef val = attr.getValue();
    if (isOMIRStringEncodedPassthrough(val.split(":").first))
      return jsonStream.value(val);
  }

  // If we get here, we don't know how to serialize the given MLIR attribute as
  // a OMIR value.
  jsonStream.value("<unsupported value>");
  getOperation().emitError("unsupported attribute for OMIR serialization: `")
      << node << "`";
  anyFailures = true;
}

void EmitOMIRPass::emitTrackedTarget(DictionaryAttr node,
                                     llvm::json::OStream &jsonStream) {
  // Extract the `id` field.
  auto idAttr = node.getAs<IntegerAttr>("id");
  if (!idAttr) {
    getOperation()
            .emitError("tracked OMIR target missing `id` string field")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }

  // Extract the `type` field.
  auto typeAttr = node.getAs<StringAttr>("type");
  if (!typeAttr) {
    getOperation()
            .emitError("tracked OMIR target missing `type` string field")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  StringRef type = typeAttr.getValue();

  // Find the tracker for this target, and handle the case where the tracker has
  // been deleted.
  auto trackerIt = trackers.find(idAttr);
  if (trackerIt == trackers.end()) {
    // Some of the target types indicate removal of the target through an
    // `OMDeleted` node.
    if (type == "OMReferenceTarget" || type == "OMMemberReferenceTarget" ||
        type == "OMMemberInstanceTarget")
      return jsonStream.value("OMDeleted");

    // The remaining types produce an error upon removal of the target.
    auto diag = getOperation().emitError("tracked OMIR target of type `")
                << type << "` was deleted";
    diag.attachNote(getOperation().getLoc())
        << type << " should never be deleted";
    diag.attachNote(getOperation().getLoc()) << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  auto tracker = trackerIt->second;

  // Serialize the target circuit first.
  SmallString<64> target(type);
  target.append(":~");
  target.append(getOperation().name());
  target.push_back('|');

  // Serialize the local or non-local module/instance hierarchy path.
  if (tracker.nla) {
    bool notFirst = false;
    StringAttr instName;
    for (auto modAndName : llvm::zip(tracker.nla.modpath().getValue(),
                                     tracker.nla.namepath().getValue())) {
      auto symAttr = std::get<0>(modAndName).cast<FlatSymbolRefAttr>();
      auto nameAttr = std::get<1>(modAndName).cast<StringAttr>();
      Operation *module = symtbl->lookup(symAttr.getValue());
      assert(module);
      if (notFirst)
        target.push_back('/');
      notFirst = true;
      if (instName) {
        // TODO: This should *really* drop a symbol to represent the instance
        // name. See below.
        target.append(instName.getValue());
        target.push_back(':');
      }
      target.append(addSymbol(module));
      instName = nameAttr;

      // Find an instance with the given name in this module.
      module->walk([&](InstanceOp instOp) {
        if (instOp.nameAttr() == nameAttr) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Marking NLA-participating instance " << nameAttr
                     << " in module " << symAttr << " as dont-touch\n");
          AnnotationSet::addDontTouch(instOp);
        }
      });
    }
  } else {
    FModuleOp module = dyn_cast<FModuleOp>(tracker.op);
    if (!module)
      module = tracker.op->getParentOfType<FModuleOp>();
    assert(module);
    target.append(addSymbol(module));
  }

  // Serialize any potential component *inside* the module that this target may
  // specifically refer to.
  StringRef componentName;
  if (isa<WireOp, RegOp, RegResetOp, InstanceOp, NodeOp>(tracker.op)) {
    AnnotationSet::addDontTouch(tracker.op);
    LLVM_DEBUG(llvm::dbgs()
               << "Marking OMIR-targeted " << tracker.op << " as dont-touch\n");
    // TODO: This should *really* drop a symbol placeholder into the JSON. But
    // we currently don't have any symbols for these FIRRTL ops. May be solved
    // through NLAs.
    componentName = tracker.op->getAttrOfType<StringAttr>("name").getValue();
  } else if (!isa<FModuleOp>(tracker.op)) {
    tracker.op->emitError("invalid target for `") << type << "` OMIR";
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  if (!componentName.empty()) {
    target.push_back('>');
    target.append(componentName);
  }

  jsonStream.value(target);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
circt::firrtl::createEmitOMIRPass(StringRef outputFilename) {
  auto pass = std::make_unique<EmitOMIRPass>();
  if (!outputFilename.empty())
    pass->outputFilename = outputFilename.str();
  return pass;
}