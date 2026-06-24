// Command extdriver is the worked OUT-OF-TREE driver for fak's ABI.
//
// It lives in its own module (see go.mod) and imports ONLY
// github.com/anthony-chaudhary/fak/pkg/abi — never internal/abi, which Go's
// internal/ rule would forbid from here. It claims BOTH vendor numbers
// end-to-end:
//
//   - an OpCode in the OpsVendor range [1<<16, 1<<17), via abi.RegisterOp
//   - a VerdictKind in the VerdictsVendor range [1024, 1<<16), via
//     abi.RegisterVerdictKind
//
// and registers one real driver — an Adjudicator that DENIES a named tool. main
// then exercises the whole round-trip and exits NON-ZERO if any leg fails, so
// running it is a witness that the ABI is importable and usable from outside the
// module, not merely that it compiles.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

// --- the claimed vendor numbers (drawn from the reserved vendor ranges) -------

// vendorOp is an OpCode inside OpsVendor [1<<16, 1<<17).
const vendorOp abi.OpCode = (1 << 16) + 0x2A // 65578

// vendorVerdict is a VerdictKind inside VerdictsVendor [1024, 1<<16). It rides
// the same fold lattice as the core kinds; we give it a high fold-rank so it
// wins a fold (like a deny) and FallbackDeny so an unaware worker fails closed.
const vendorVerdict abi.VerdictKind = 1024 + 0x07 // 1031

const (
	// FoldRank is the position in the restrictiveness lattice (higher == more
	// restrictive / wins the fold). 200 sits above the core VerdictDeny(100), so
	// this vendor verdict is the most restrictive kind in the chain.
	vendorVerdictFoldRank = 200
	deniedTool            = "refund_payment"
)

// --- the registered Op (claims the OpsVendor number) --------------------------

type vendorEcho struct{}

func (vendorEcho) Code() abi.OpCode { return vendorOp }

// Invoke is a trivial pass-through so the Op is a real, invokable entry. The
// demo does not route a call through the kernel's Op table; registering it is
// what claims the opcode (RegisterOp panics on a clash, so a successful
// registration IS the proof the number is ours).
func (vendorEcho) Invoke(_ context.Context, _ abi.Kernel, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	return &abi.Result{Call: c, Status: abi.StatusOK}, abi.Verdict{Kind: abi.VerdictAllow}
}

// --- the registered Adjudicator (a real driver) -------------------------------

// denyTool denies one named tool with the vendor verdict kind and a core reason,
// and Defers on everything else (the fold identity for an Adjudicator).
type denyTool struct{ tool string }

func (d denyTool) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c != nil && c.Tool == d.tool {
		return abi.Verdict{
			Kind:   vendorVerdict,
			Reason: abi.ReasonPolicyBlock,
			By:     "extdriver/denyTool",
		}
	}
	return abi.Verdict{Kind: abi.VerdictDefer}
}

func (denyTool) Caps() []abi.Capability { return nil }

// denyToolAdj is registered in init so it is live before main runs.
var denyToolAdj = denyTool{tool: deniedTool}

func init() {
	// Claim the OpsVendor opcode. Panics on a clash — a clean return is proof
	// the number is ours.
	abi.RegisterOp(vendorEcho{})

	// Claim the VerdictsVendor kind with a fold-rank + fail-closed fallback.
	abi.RegisterVerdictKind(vendorVerdict, "EXTDRIVER_DENY", vendorVerdictFoldRank, abi.FallbackDeny)

	// Register the driver itself at a low rank so it runs early.
	abi.RegisterAdjudicator(10, denyToolAdj)
}

func main() {
	failed := false
	fail := func(format string, a ...any) {
		failed = true
		fmt.Fprintf(os.Stderr, "FAIL: "+format+"\n", a...)
	}

	fmt.Println("fak out-of-tree driver — ABI importable via pkg/abi")
	fmt.Printf("  ABI version: v%d.%d\n", abi.ABIMajor, abi.ABIMinor)

	// 1) The claimed OpsVendor opcode resolves in the kernel's op table.
	fmt.Printf("\n[1] claimed OpCode %d (OpsVendor range [%d,%d))\n",
		vendorOp, abi.OpsVendor.Lo, abi.OpsVendor.Hi)
	if uint32(vendorOp) < abi.OpsVendor.Lo || uint32(vendorOp) >= abi.OpsVendor.Hi {
		fail("opcode %d is outside the OpsVendor range", vendorOp)
	}
	// RegisterOp ran in init without panicking; that is the claim. (The kernel's
	// LookupOp read-side accessor is intentionally not re-exported through
	// pkg/abi — it is host-internal — so a successful, clash-free registration is
	// the witness available to a driver.)
	fmt.Printf("    registered without a clash -> opcode is claimed\n")

	// 2) The claimed VerdictsVendor kind resolves via FoldRank.
	fmt.Printf("\n[2] claimed VerdictKind %d (VerdictsVendor range [%d,%d))\n",
		vendorVerdict, abi.VerdictsVendor.Lo, abi.VerdictsVendor.Hi)
	if uint32(vendorVerdict) < abi.VerdictsVendor.Lo || uint32(vendorVerdict) >= abi.VerdictsVendor.Hi {
		fail("verdict kind %d is outside the VerdictsVendor range", vendorVerdict)
	}
	if got := abi.FoldRank(vendorVerdict); got != vendorVerdictFoldRank {
		fail("FoldRank(%d) = %d, want %d (registration did not land)", vendorVerdict, got, vendorVerdictFoldRank)
	} else {
		fmt.Printf("    FoldRank(%d) = %d -> kind is registered in the frozen lattice\n", vendorVerdict, got)
	}

	// 3) The driver round-trip: deny the named tool, Defer on another.
	ctx := context.Background()
	denied := &abi.ToolCall{Op: vendorOp, Tool: deniedTool}
	v := denyToolAdj.Adjudicate(ctx, denied)
	fmt.Printf("\n[3] Adjudicate(tool=%q) -> Kind=%d Reason=%s By=%s\n",
		deniedTool, v.Kind, abi.ReasonName(v.Reason), v.By)
	if v.Kind != vendorVerdict {
		fail("expected the vendor verdict kind %d, got %d", vendorVerdict, v.Kind)
	}
	if abi.ReasonName(v.Reason) != "POLICY_BLOCK" {
		fail("expected reason POLICY_BLOCK, got %s", abi.ReasonName(v.Reason))
	}

	allowed := &abi.ToolCall{Tool: "read_file"}
	if v2 := denyToolAdj.Adjudicate(ctx, allowed); v2.Kind != abi.VerdictDefer {
		fail("expected VerdictDefer for an untargeted tool, got %d", v2.Kind)
	} else {
		fmt.Printf("    Adjudicate(tool=%q) -> Kind=%d (VerdictDefer) — fold identity holds\n", "read_file", v2.Kind)
	}

	if failed {
		fmt.Fprintln(os.Stderr, "\nextdriver: ROUND-TRIP FAILED")
		os.Exit(1)
	}
	fmt.Println("\nextdriver: OK — both vendor numbers claimed and the driver round-trip passed")
}
