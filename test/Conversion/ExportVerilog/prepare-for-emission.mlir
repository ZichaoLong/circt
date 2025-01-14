// RUN: circt-opt %s -test-prepare-for-emission --split-input-file -verify-diagnostics | FileCheck %s

// CHECK: @namehint_variadic
hw.module @namehint_variadic(%a: i3) -> (b: i3) {
  // CHECK-NEXT: %0 = comb.add %a, %a : i3
  // CHECK-NEXT: %1 = comb.add %a, %0 {sv.namehint = "bar"} : i3
  // CHECK-NEXT: hw.output %1
  %0 = comb.add %a, %a, %a { sv.namehint = "bar" } : i3
  hw.output %0 : i3
}

// CHECK-LABEL:  hw.module @outOfOrderInoutOperations
hw.module @outOfOrderInoutOperations(%a: i4) -> (c: i4) {
  // CHECK: %wire = sv.wire
  // CHECK-NEXT: %0 = sv.array_index_inout %wire[%false]
  // CHECK-NEXT: %1 = sv.array_index_inout %0[%false]
  // CHECK-NEXT: %2 = sv.array_index_inout %1[%false]
  // CHECK-NEXT: %3 = sv.read_inout %2
  %false = hw.constant false
  %0 = sv.read_inout %3 : !hw.inout<i4>
  %3 = sv.array_index_inout %2[%false] : !hw.inout<array<1xi4>>, i1
  %2 = sv.array_index_inout %1[%false] : !hw.inout<array<1xarray<1xi4>>>, i1
  %1 = sv.array_index_inout %wire[%false] : !hw.inout<array<1xarray<1xarray<1xi4>>>>, i1
  %wire = sv.wire  : !hw.inout<array<1xarray<1xarray<1xi4>>>>
  hw.output %0: i4
}

// -----

module {
  // CHECK-LABEL:  hw.module @SpillTemporaryInProceduralRegion
  hw.module @SpillTemporaryInProceduralRegion(%a: i4, %b: i4, %fd: i32) -> () {
    // CHECK-NEXT: %r = sv.reg
    // CHECK-NEXT: sv.initial {
    // CHECK-NEXT:   %0 = sv.logic
    // CHECK-NEXT:   %1 = comb.add %a, %b
    // CHECK-NEXT:   sv.bpassign %0, %1
    // CHECK-NEXT:   %2 = sv.read_inout %0
    // CHECK-NEXT:   %3 = comb.extract %2 from 3
    // CHECK-NEXT:   sv.passign %r, %3
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output
    %r = sv.reg : !hw.inout<i1>
    sv.initial {
      %0 = comb.add %a, %b : i4
      %1 = comb.extract %0 from 3 : (i4) -> i1
      sv.passign %r, %1: i1
    }
  }
}

// -----

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  // CHECK: @test_hoist
  hw.module @test_hoist(%a: i3) -> () {
    // CHECK-NEXT: %reg = sv.reg
    %reg = sv.reg : !hw.inout<i3>
    // CHECK-NEXT: %0 = comb.add
    // CHECK-NEXT: sv.initial
    sv.initial {
      %0 = comb.add %a, %a : i3
      sv.passign %reg, %0 : i3
    }
  }

  // CHECK-LABEL:  hw.module @SpillTemporary
  hw.module @SpillTemporary(%a: i4, %b: i4) -> (c: i1) {
    // CHECK-NEXT:  %0 = comb.add %a, %b
    // CHECK-NEXT:  %[[GEN:.+]] = sv.wire
    // CHECK-NEXT:  sv.assign %1, %[[GEN:.+]]
    // CHECK-NEXT:  %2 = sv.read_inout %1
    // CHECK-NEXT:  %3 = comb.extract %2 from 3
    // CHECK-NEXT:  hw.output %3
    %0 = comb.add %a, %b : i4
    %1 = comb.extract %0 from 3 : (i4) -> i1
    hw.output %1 : i1
  }

  // CHECK-LABEL:  hw.module @SpillTemporaryInProceduralRegion
  hw.module @SpillTemporaryInProceduralRegion(%a: i4, %b: i4, %fd: i32) -> () {
    // CHECK-NEXT: %r = sv.reg
    // CHECK-NEXT: %[[VAL:.+]] = comb.add %a, %b
    // CHECK-NEXT: %[[GEN:.+]] = sv.wire
    // CHECK-NEXT: sv.assign %[[GEN]], %[[VAL]]
    // CHECK-NEXT: %2 = sv.read_inout %[[GEN]]
    // CHECK-NEXT: %3 = comb.extract %2 from 3
    // CHECK-NEXT: sv.initial {
    // CHECK-NEXT:   sv.passign %r, %3
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output
    %r = sv.reg : !hw.inout<i1>
    sv.initial {
      %0 = comb.add %a, %b : i4
      %1 = comb.extract %0 from 3 : (i4) -> i1
      sv.passign %r, %1: i1
    }
  }

  // CHECK-LABEL: @SpillTemporaryWireForMultipleUseExpression
  hw.module @SpillTemporaryWireForMultipleUseExpression(%a: i4, %b: i4) -> (c: i4, d: i4) {
    // CHECK-NEXT: %[[VAL:.+]] = comb.add %a, %b
    // CHECK-NEXT: %[[GEN:bar]] = sv.wire
    // CHECK-NEXT: sv.assign %[[GEN]], %[[VAL]]
    // CHECK-NEXT: %1 = sv.read_inout %[[GEN]]
    // CHECK-NEXT: %2 = sv.read_inout %[[GEN]]
    // CHECK-NEXT: hw.output %2, %1
    %0 = comb.add %a, %b {sv.namehint = "bar"}: i4
    hw.output %0, %0 : i4, i4
  }
}

// -----

module attributes {circt.loweringOptions = "disallowExpressionInliningInPorts"} {
 hw.module.extern @MyExtModule(%in: i8)
 // CHECK-LABEL: @MoveInstances
 hw.module @MoveInstances(%a_in: i8) -> (){
  // CHECK-NEXT: %_xyz3_in = sv.wire
  // CHECK-NEXT: %0 = comb.add %a_in, %a_in
  // CHECK-NEXT: %1 = sv.read_inout %_xyz3_in
  // CHECK-NEXT: sv.assign %_xyz3_in, %0
  // CHECK-NEXT: hw.instance "xyz3" @MyExtModule(in: %1: i8) -> ()
  %0 = comb.add %a_in, %a_in : i8
  hw.instance "xyz3" @MyExtModule(in: %0: i8) -> ()
 }
}

// -----
module attributes {circt.loweringOptions =
                  "wireSpillingHeuristic=spillLargeTermsWithNamehints,wireSpillingNamehintTermLimit=3"} {
  // CHECK-LABEL: namehints
  hw.module @namehints(%a: i8) -> (b: i8) {
    // "foo" should not be spilled because the term size is 2.
    // CHECK-NOT: %foo = sv.wire
    %0 = comb.add %a, %a {sv.namehint = "foo" } : i8
    // "bar" should be spilled because the term size is 3.
    // CHECK: %bar = sv.wire
    %1 = comb.add %a, %a, %a {sv.namehint = "bar" } : i8
    %2 = comb.add %0, %1 : i8
    hw.output %2 : i8
  }
}
