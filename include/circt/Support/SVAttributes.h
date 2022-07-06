#ifndef CIRCT_SUPPORT_SV_ATTRIBUTES_H
#define CIRCT_SUPPORT_SV_ATTRIBUTES_H
#include "circt/Support/LLVM.h"

namespace circt {

/// Helper functions to handle SV attributes.
bool hasSVAttributes(mlir::Operation *op);
mlir::ArrayAttr getSVAttributes(mlir::Operation *op);
void setSVAttributes(mlir::Operation *op, mlir::Attribute);

} // namespace circt

#endif // CIRCT_SUPPORT_SV_ATTRIBUTES_H
