// Package provenance is the single, kernel-authored answer to one question:
// "where did this byte come from, and may the kernel trust it?" — and it is the
// ONE place that answer is decided.
//
// THE INVARIANT IT ENFORCES: authorship of trust belongs to the KERNEL, never to
// the model. The model emits the ToolCall — its Op, its Tool, its Args, and the
// OPEN Meta map are all under the model's control — so ANY trust signal carried
// *inside the call* is a self-report an injected/derailed agent can forge. The
// lexical gates (canon, normgate) and the information-flow gate (ifc) both used to
// honor a `Meta["provenance"]="trusted_local"` tag; that tag rides the model's own
// request envelope, so a poisoned `read_webpage` could mint itself "trusted",
// walk past normgate's quarantine AND skip ifc's session taint, and then
// exfiltrate freely. That is the model authoring its own trust — exactly the hole
// this package closes.
//
// Trust here is a function of EXACTLY TWO kernel-controlled facts, never the call:
//
//  1. The kernel-stamped RESULT state. A result the detector chain SEALED stays
//     Quarantined; a result the producing tool/kernel stamped Trusted is Trusted.
//     The Result envelope is written by dispatch + the ResultAdmitter chain AFTER
//     the model's call returns, so it is kernel-authored, not a model claim.
//
//  2. The tool's host-registered SOURCE CLASS. A "Read" is TrustedLocal because the
//     HOST registered the local-filesystem read channel as such at boot (in its
//     own address space, via RegisterSource) — not because the model named the
//     call "Read". The model picks WHICH tool to invoke, but dispatch binds that
//     name to a real channel, and the model cannot add a tool to the trusted set:
//     RegisterSource is a Go API the kernel calls, unreachable from a tool call.
//
// What it deliberately does NOT consult: ToolCall.Meta. The forgeable self-tag is
// recognized only so AttemptedSelfTrust can SURFACE the forgery attempt for a
// steward/auditor — it never changes a verdict. Default for an unregistered tool
// is Untrusted (fail-closed): a source the host never blessed cannot be trusted.
//
// This package is a pure classification LIBRARY (the canon/grammar shape), not a
// registered driver: ifc's StampGate/SinkGate and normgate consume it so the whole
// kernel agrees on provenance from one definition instead of three drifting copies.
package provenance

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Source is the host-authored origin class of a tool's CHANNEL. It is CLOSED and
// additive (new classes append). The zero value is the fail-closed default.
type Source uint8

const (
	// Untrusted is the fail-closed default: an external / egress / unknown channel,
	// or any tool the host never registered. Bytes from here are never trusted.
	Untrusted Source = iota
	// TrustedLocal is a read of the agent's OWN local environment (its filesystem,
	// its workspace) — a non-adversarial source the host blessed at boot.
	TrustedLocal
)

// selfTrustMetaKey / selfTrustMetaValue is the historical model-authored trust
// self-declaration. It is recognized ONLY to be ignored for trust and surfaced as
// a forgery attempt (AttemptedSelfTrust) — never honored.
const (
	selfTrustMetaKey   = "provenance"
	selfTrustMetaValue = "trusted_local"
)

// reg is the host-authored source registry: tool name -> Source. It is populated
// by the kernel/host (init defaults below + RegisterSource), never by anything a
// tool call can reach, which is what keeps trust-authorship out of the model.
var (
	mu  sync.RWMutex
	reg = map[string]Source{}
)

func init() {
	// Built-in trusted-local read tools — the agent reading its own environment.
	// A host extends this set with RegisterSource; it never shrinks the kernel's
	// guarantee. Kept identical to the set the ifc/normgate drivers used inline so
	// routing them through this package preserves their behavior (minus the hole).
	for _, t := range []string{
		"Read", "Grep", "Glob", "LS", "NotebookRead", "read_file", "cat",
	} {
		reg[t] = TrustedLocal
	}
}

// RegisterSource declares a tool's origin class. It is the kernel-side authorship
// channel: the host calls it at boot (an init, internal/registrations, or a loaded
// policy), so the trusted set is fixed before any call is adjudicated and is
// unreachable from a model-emitted call. Registering the same tool again overrides.
func RegisterSource(tool string, s Source) {
	mu.Lock()
	reg[tool] = s
	mu.Unlock()
}

// SourceOf returns a tool's registered source class, or Untrusted (fail-closed) if
// the host never registered it.
func SourceOf(tool string) Source {
	mu.RLock()
	defer mu.RUnlock()
	return reg[tool] // zero value Untrusted for an unregistered tool
}

// Sources returns a snapshot copy of the source registry — for diagnostics and a
// `--dump`-style view of what the host has blessed. Mutating the copy is harmless.
func Sources() map[string]Source {
	mu.RLock()
	defer mu.RUnlock()
	out := make(map[string]Source, len(reg))
	for k, v := range reg {
		out[k] = v
	}
	return out
}

// Taint is the authoritative provenance taint of a result, on the frozen
// abi.TaintLabel lattice (trusted < tainted < quarantined). It derives the label
// from kernel-controlled facts ONLY — never from the model-authored ToolCall.Meta:
//
//  1. A SEALED result (Ref.Taint==Quarantined, or a detector stamped a
//     quarantine_id) stays Quarantined — the most restrictive, kernel-authored.
//  2. A result the producing tool/kernel stamped Trusted is Trusted (the Result
//     envelope is written after the call, by dispatch + the admitter chain).
//  3. Otherwise the tool's HOST-REGISTERED source class decides: TrustedLocal =>
//     Trusted, anything else => Tainted (fail-closed). Meta is never read.
func Taint(c *abi.ToolCall, r *abi.Result) abi.TaintLabel {
	if r != nil {
		if r.Payload.Taint == abi.TaintQuarantined || sealed(r) {
			return abi.TaintQuarantined
		}
		if r.Payload.Taint == abi.TaintTrusted {
			return abi.TaintTrusted
		}
	}
	if c != nil && SourceOf(c.Tool) == TrustedLocal {
		return abi.TaintTrusted
	}
	return abi.TaintTainted
}

// Trusted reports whether (c, r) has trusted-local provenance — the boolean form
// of Taint for the normgate policy ("trusted-local injection-shaped content pages
// out retrievable instead of sealed"). A sealed or tainted result is never
// trusted, so a forged Meta tag can never flip this.
func Trusted(c *abi.ToolCall, r *abi.Result) bool {
	return Taint(c, r) == abi.TaintTrusted
}

// AttemptedSelfTrust reports whether the call carries the model-authored trust
// self-declaration (the historical Meta["provenance"]="trusted_local" tag). It is
// a FORENSIC signal only: Taint and Trusted ignore Meta entirely, so a forged tag
// never changes a verdict. A steward/auditor can use this to flag a model that is
// trying to launder its own provenance — a strong injection/derailment tell.
func AttemptedSelfTrust(c *abi.ToolCall) bool {
	return c != nil && c.Meta != nil && c.Meta[selfTrustMetaKey] == selfTrustMetaValue
}

// sealed reports whether a detector chain sealed this result (the ctxmmu/normgate
// convention: a quarantine_id stamped into Meta). Kept inline so this library
// depends only on abi, matching canon/grammar.
func sealed(r *abi.Result) bool {
	return r.Meta != nil && r.Meta["quarantine_id"] != ""
}

func (s Source) String() string {
	switch s {
	case TrustedLocal:
		return "trusted_local"
	}
	return "untrusted"
}
