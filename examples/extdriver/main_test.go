package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

// TestVendorNumbersInReservedRanges asserts the claimed OpCode and VerdictKind
// sit inside the reserved vendor ranges — the same range check main prints, made
// a failing assertion. A driver that drifts its constant out of range is a
// genuine ABI-contract break, so this guards it at test time.
func TestVendorNumbersInReservedRanges(t *testing.T) {
	if uint32(vendorOp) < abi.OpsVendor.Lo || uint32(vendorOp) >= abi.OpsVendor.Hi {
		t.Errorf("vendorOp %d outside OpsVendor range [%d,%d)", vendorOp, abi.OpsVendor.Lo, abi.OpsVendor.Hi)
	}
	if uint32(vendorVerdict) < abi.VerdictsVendor.Lo || uint32(vendorVerdict) >= abi.VerdictsVendor.Hi {
		t.Errorf("vendorVerdict %d outside VerdictsVendor range [%d,%d)", vendorVerdict, abi.VerdictsVendor.Lo, abi.VerdictsVendor.Hi)
	}
}

// TestVendorVerdictRegisteredInLattice asserts the init-time RegisterVerdictKind
// landed: the frozen fold lattice reports the fold-rank we registered. A
// registration that silently failed would read back a different rank.
func TestVendorVerdictRegisteredInLattice(t *testing.T) {
	if got := abi.FoldRank(vendorVerdict); got != vendorVerdictFoldRank {
		t.Errorf("FoldRank(%d) = %d, want %d", vendorVerdict, got, vendorVerdictFoldRank)
	}
}

// TestDenyToolDeniesNamedTool exercises the driver's core decision: the named
// tool is denied with the vendor verdict kind and the POLICY_BLOCK reason.
func TestDenyToolDeniesNamedTool(t *testing.T) {
	v := denyToolAdj.Adjudicate(context.Background(), &abi.ToolCall{Op: vendorOp, Tool: deniedTool})
	if v.Kind != vendorVerdict {
		t.Errorf("Adjudicate(%q).Kind = %d, want vendorVerdict %d", deniedTool, v.Kind, vendorVerdict)
	}
	if name := abi.ReasonName(v.Reason); name != "POLICY_BLOCK" {
		t.Errorf("Adjudicate(%q).Reason = %s, want POLICY_BLOCK", deniedTool, name)
	}
}

// TestDenyToolDefersOnOtherTools asserts the fold identity: any tool the driver
// does not target returns VerdictDefer, so an unaware caller is not blocked by it.
func TestDenyToolDefersOnOtherTools(t *testing.T) {
	v := denyToolAdj.Adjudicate(context.Background(), &abi.ToolCall{Tool: "read_file"})
	if v.Kind != abi.VerdictDefer {
		t.Errorf("Adjudicate(%q).Kind = %d, want VerdictDefer %d", "read_file", v.Kind, abi.VerdictDefer)
	}
}

// TestVendorEchoInvokeReturnsOK checks the claimed Op is a real, invokable entry:
// its trivial pass-through returns an OK result bound to the call, with an Allow
// verdict.
func TestVendorEchoInvokeReturnsOK(t *testing.T) {
	call := &abi.ToolCall{Op: vendorOp, Tool: "anything"}
	res, v := vendorEcho{}.Invoke(context.Background(), nil, call)
	if res == nil || res.Call != call || res.Status != abi.StatusOK {
		t.Errorf("Invoke result = %+v, want OK result bound to the call", res)
	}
	if v.Kind != abi.VerdictAllow {
		t.Errorf("Invoke verdict = %d, want VerdictAllow %d", v.Kind, abi.VerdictAllow)
	}
}
