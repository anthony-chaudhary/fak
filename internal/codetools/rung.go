package codetools

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// rung.go — the adjudicator that admits a codetools call and BINDS it to its engine.
//
// This rung does two jobs the kernel cannot do for us and an engine cannot do for
// itself:
//
//  1. ROUTING. abi.ToolCall.Engine is what kernel.routeFor dispatches on, and a loop that
//     does not know about this package leaves it empty — which routes the call to the
//     kernel's DEFAULT engine (the airline demo's "localtools" on the owned loop), where
//     a Read is an unknown tool. Adjudicate receives the call by POINTER and runs before
//     dispatch, so pinning c.Engine here is what makes the six engines reachable at all,
//     without any edit to the loop that proposed the call.
//
//  2. PRE-DISPATCH ENFORCEMENT. Confinement, schema, and policy are decided BEFORE the
//     engine exists in the story. A refusal here is a kernel VerdictDeny — it is counted
//     in k.Counters().Denies, it rides the decision journal with By=codetools, and the
//     engine is never entered. That is the difference between a policy and a convention:
//     a denied Read never reaches code that could open a file.
//
// ORDER IS PART OF THE CONTRACT. decode -> validate -> canonicalize -> confine ->
// protected -> policy -> cache-scope -> pin. Canonicalization strictly precedes policy so
// policy is asked about a real file rather than about the caller's spelling of one.
//
// EVERYTHING ELSE DEFERS. A tool this package does not own returns VerdictDefer, which
// leaves the rest of the chain to decide — so installing this rung cannot change the
// verdict of a single call belonging to another toolset.

// Caps advertises no optional capabilities.
func (t *Toolset) Caps() []abi.Capability { return nil }

// Adjudicate decides one proposed call. See the file comment for the ordering contract.
func (t *Toolset) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungName}
	}
	engine, mine := engineFor(c.Tool)
	if !mine {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungName}
	}
	if err := ctx.Err(); err != nil {
		return t.deny(refuse(CodeCanceled, err.Error()), c.Tool)
	}
	if r := t.admit(ctx, c); r != nil {
		return t.deny(r, c.Tool)
	}
	// Pin the route only after every check has passed. Binding the engine on a call we
	// are about to deny would leave a denied ToolCall carrying a live engine id — a
	// forensically misleading record, and a live hazard for any later code that reads
	// Engine to decide whether a call was dispatchable.
	c.Engine = engine
	return abi.Verdict{Kind: abi.VerdictAllow, By: RungName}
}

// admit runs the ordered checks and returns the first refusal, or nil to allow.
func (t *Toolset) admit(ctx context.Context, c *abi.ToolCall) *Refusal {
	body := bytesOf(ctx, c.Args)
	// Canonical path FIRST: every path-bearing tool resolves its operand to one
	// (Abs, Rel) pair before cache-scope or policy matching, so both reason about a real
	// workspace target rather than a caller-controlled spelling.
	if _, r := t.targetOf(c.Tool, body); r != nil {
		return r
	}
	if r := t.cacheScope(c); r != nil {
		return r
	}
	// Policy LAST among the semantic checks, on the canonical operand. A tool the
	// policy does not name is denied fail-closed.
	if !t.policy.Allow[c.Tool] {
		return refuse(CodeDefaultDeny, "no policy admits tool "+c.Tool)
	}
	return nil
}

// cacheScope refuses a call whose vDSO hints contradict its tool's real write shape.
//
// The vDSO fast path runs BEFORE adjudication (kernel.Submit consults FastPaths first),
// so by the time this rung sees a call the cache has already had its chance to answer it.
// That ordering is exactly why the hints have to be adjudicated at all: a Write stamped
// readOnlyHint=true is a call the cache is entitled to serve from a previous result —
// i.e. a mutation that silently does not happen, reported as success. Refusing the
// mislabeled call is what makes that unrepresentable. A call carrying NO hints is
// accepted: it is simply cache-ineligible, which is safe for both shapes.
func (t *Toolset) cacheScope(c *abi.ToolCall) *Refusal {
	if readOnlyTool(c.Tool) || c.Meta == nil {
		return nil
	}
	if c.Meta["readOnlyHint"] == "true" || c.Meta["idempotentHint"] == "true" {
		return refuse(CodeCacheScope, c.Tool+" is write-shaped and must not assert readOnlyHint/idempotentHint")
	}
	return nil
}

// targetOf decodes a call's arguments, validates them against the tool's own schema, and
// returns the canonical workspace-relative path the call targets ("" for a Grep/Glob left
// at the workspace root).
//
// Decoding and validating HERE, in the rung, is what lets a malformed or escaping call be
// refused as an adjudication rather than surface as an engine error: the two are not
// interchangeable, because only the first is counted, journaled, and kept away from the
// engine entirely.
func (t *Toolset) targetOf(tool string, body []byte) (string, *Refusal) {
	switch tool {
	case ToolRead:
		var a ReadArgs
		return decodeTarget(t, body, &a, func() string { return a.FilePath })
	case ToolWrite:
		var a WriteArgs
		return decodeTarget(t, body, &a, func() string { return a.FilePath })
	case ToolEdit:
		var a EditArgs
		return decodeTarget(t, body, &a, func() string { return a.FilePath })
	case ToolBash:
		var a BashArgs
		return decodeTarget(t, body, &a, func() string { return a.Cwd })
	case ToolGrep:
		var a GrepArgs
		return decodeTarget(t, body, &a, func() string { return a.Path })
	case ToolGlob:
		var a GlobArgs
		return decodeTarget(t, body, &a, func() string { return a.Path })
	case ToolApplyPatch:
		var a PatchArgs
		return decodeTarget(t, body, &a, func() string { return "" })
	}
	return "", refuse(CodeMalformed, "unknown tool "+tool)
}

type validToolArgs interface {
	Validate() *Refusal
}

// decodeTarget keeps every path-bearing tool on the same decode -> validate -> resolve
// path. Empty optional operands (Bash cwd and Grep/Glob path) deliberately address the
// workspace root without calling resolve; required operands are rejected by Validate.
func decodeTarget[A validToolArgs](t *Toolset, body []byte, args A, path func() string) (string, *Refusal) {
	if r := decodeArgs(body, args); r != nil {
		return "", r
	}
	if r := args.Validate(); r != nil {
		return "", r
	}
	target := path()
	if target == "" {
		return "", nil
	}
	res, r := t.resolve(target)
	return res.Rel, r
}

// deny renders a Refusal as a kernel Verdict. The local code rides Meta so an operator
// reading the journal sees WHICH invariant refused, while Reason stays in the closed abi
// vocabulary the chain's fold and disposition logic understand.
func (t *Toolset) deny(r *Refusal, tool string) abi.Verdict {
	reason := r.Reason
	if reason == abi.ReasonNone {
		// A code that describes an engine-level failure has no business becoming an
		// adjudication verdict; if one ever reaches here, fail CLOSED rather than
		// allowing a call whose refusal we could not classify.
		reason = abi.ReasonPolicyBlock
	}
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: reason,
		By:     RungName,
		Meta:   map[string]string{"code": r.Code, "tool": tool, "detail": r.Detail},
	}
}
