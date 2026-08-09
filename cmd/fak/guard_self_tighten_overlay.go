// guard_self_tighten_overlay.go ARMS the #5181 self-amendment admission gate
// (epic #5170, Track F). guard_self_tighten.go shipped the DECISION half —
// admitSelfTightenOverlay — with nothing calling it; this file supplies the seam
// the design spike specified (docs/notes/RESEARCH-MONOTONE-TIGHTEN-SELF-AMENDMENT-2026-07-17.md,
// "Mechanism 1"): an optional on-disk overlay the WRAPPED AGENT may write, unioned
// into the capability floor at launch, and admitted only through that gate.
//
// Two independent barriers, exactly as the spike specifies:
//
//  1. SCHEMA. guardSelfTightenOverlay carries only ratchet-direction fields — deny,
//     block_hosts, self_modify_globs. There is no allow / allow_prefix / posture
//     field, and the decoder runs DisallowUnknownFields, so a widening cannot even
//     be SPELLED: a forged {"allow":[...]} fails to decode and the overlay is
//     refused wholesale rather than partially applied.
//  2. DELTA. Whatever the union produces is still classified by
//     admitSelfTightenOverlay before it is installed, so even if the union builder
//     were later changed to emit a widening, the apply path refuses it. That is
//     what guardAdmitSelfTightenProposal exists to make testable on its own.
//
// WHY THIS FILE IS NOT UNDER .fak/guard/. The spike names
// `.fak/guard/self-tighten.json` and requires that the agent be able to WRITE it
// (unlike allow.json / deny.json, which are self-modify-protected). Those two
// requirements contradict each other against the shipped floor: `.fak/guard/` is
// itself a self_modify_glob (cmd/fak/guard-default-policy.json), so every path
// under it is unwritable by the wrapped agent. Punching a hole in that glob would
// be a WIDENING of the shipped floor — the one thing this epic exists to prevent —
// so the overlay lives at `.fak/agent/self-tighten.json` instead, a path no shipped
// glob covers. The floor is unchanged byte for byte and the agent can still write
// its own tightenings. See TestGuardSelfTightenOverlayPathIsAgentWritable.
//
// LAUNCH BOUNDARY ONLY. The union happens in loadGuardCapabilityFloor
// (cmd/fak/guard_startup.go). Mid-session live application is explicitly deferred by
// the spike's own Open Questions ("v1 is launch-boundary only"), so the reload paths
// are untouched and a running gateway's behaviour is byte-identical to before.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const (
	guardSelfTightenOverlayVersion = "fak-guard-self-tighten/v1"
	guardSelfTightenOverlayEnv     = "FAK_GUARD_SELF_TIGHTEN_OVERLAY"
)

// guardSelfTightenOverlay is the on-disk schema of the agent's own tighten overlay.
// It is a strict RATCHET-only subset of the policy manifest: every field can only
// ever narrow what the floor admits. The absence of allow / allow_prefix / posture
// is load-bearing, not an oversight — it is barrier 1 of the two above.
type guardSelfTightenOverlay struct {
	Version string `json:"version"`
	// Deny names tools the agent wants provably refused for itself. Added deny is
	// the RATCHET direction (internal/policy/amendment_delta.go).
	Deny []string `json:"deny,omitempty"`
	// BlockHosts extends the egress LIST layer's refused hosts. Added block host
	// tightens.
	BlockHosts []string `json:"block_hosts,omitempty"`
	// SelfModifyGlobs extends the paths a write-shaped call may not target. Added
	// glob tightens; note the union below never REMOVES one, which would widen.
	SelfModifyGlobs []string `json:"self_modify_globs,omitempty"`
}

// empty reports the "nothing proposed" overlay — the universal case today, and the
// one that must stay byte-identical to the pre-#5181 floor.
func (ov guardSelfTightenOverlay) empty() bool {
	return len(ov.Deny) == 0 && len(ov.BlockHosts) == 0 && len(ov.SelfModifyGlobs) == 0
}

// guardSelfTightenOverlayPath is the overlay location: an env override when set
// (tests and non-repo launches), otherwise repo-local `.fak/agent/self-tighten.json`.
// Deliberately NOT under `.fak/guard/` — see the file header.
func guardSelfTightenOverlayPath() string {
	if p := strings.TrimSpace(os.Getenv(guardSelfTightenOverlayEnv)); p != "" {
		return p
	}
	return filepath.Join(findRepoRoot("."), ".fak", "agent", "self-tighten.json")
}

// loadGuardSelfTightenOverlay reads and validates the overlay. A MISSING file is the
// common no-op and yields an empty overlay. A present-but-malformed one — including
// one carrying a field this schema does not declare, which is how a forged widening
// arrives — returns an error, and the caller refuses the overlay WHOLESALE rather
// than applying part of it.
func loadGuardSelfTightenOverlay(path string) (guardSelfTightenOverlay, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return guardSelfTightenOverlay{Version: guardSelfTightenOverlayVersion}, nil
		}
		return guardSelfTightenOverlay{}, fmt.Errorf("guard self-tighten overlay %s: %w", path, err)
	}
	var ov guardSelfTightenOverlay
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&ov); err != nil {
		return guardSelfTightenOverlay{}, fmt.Errorf("guard self-tighten overlay %s: invalid: %w", path, err)
	}
	if v := strings.TrimSpace(ov.Version); v != "" && v != guardSelfTightenOverlayVersion {
		return guardSelfTightenOverlay{}, fmt.Errorf("guard self-tighten overlay %s: unsupported version %q (want %s)", path, ov.Version, guardSelfTightenOverlayVersion)
	}
	ov.Version = guardSelfTightenOverlayVersion
	ov.Deny = guardAllowNormalize(ov.Deny)
	ov.BlockHosts = guardAllowNormalize(ov.BlockHosts)
	ov.SelfModifyGlobs = guardAllowNormalize(ov.SelfModifyGlobs)
	return ov, nil
}

// saveGuardSelfTightenOverlay writes the overlay through the same atomic-replace
// writer the sibling overlays use. Present so a wrapped agent (and the tests) have a
// normalized writer rather than hand-rolling the JSON.
func saveGuardSelfTightenOverlay(path string, ov guardSelfTightenOverlay) error {
	ov.Version = guardSelfTightenOverlayVersion
	ov.Deny = guardAllowNormalize(ov.Deny)
	ov.BlockHosts = guardAllowNormalize(ov.BlockHosts)
	ov.SelfModifyGlobs = guardAllowNormalize(ov.SelfModifyGlobs)
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("guard self-tighten overlay: mkdir %s: %w", dir, err)
		}
	}
	b, err := json.MarshalIndent(ov, "", "  ")
	if err != nil {
		return err
	}
	return writeGuardAllowOverlayAtomic(path, append(b, '\n'))
}

// guardSelfTightenProposal builds `next = current UNION overlay` and reports how many
// distinct elements it would add. Union-only and copy-on-write: the returned policy
// shares no map or slice backing with `current`, so the caller can diff the two
// without the proposal having already mutated the floor it is being compared against.
// It NEVER removes an element — removal is the widening direction for every field
// here — but the gate below re-derives that from the delta rather than trusting it.
func guardSelfTightenProposal(current adjudicator.Policy, ov guardSelfTightenOverlay) (adjudicator.Policy, int) {
	next := current
	added := 0

	if len(ov.Deny) > 0 {
		deny := make(map[string]abi.ReasonCode, len(current.Deny)+len(ov.Deny))
		for tool, reason := range current.Deny {
			deny[tool] = reason
		}
		for _, tool := range ov.Deny {
			if _, exists := deny[tool]; exists {
				continue // already refused; relabelling a reason is not a delta
			}
			deny[tool] = abi.ReasonPolicyBlock
			added++
		}
		next.Deny = deny
	}
	if len(ov.BlockHosts) > 0 {
		hosts, n := guardSelfTightenUnion(current.EgressBlockHosts, ov.BlockHosts)
		next.EgressBlockHosts = hosts
		added += n
	}
	if len(ov.SelfModifyGlobs) > 0 {
		globs, n := guardSelfTightenUnion(current.SelfModifyGlobs, ov.SelfModifyGlobs)
		next.SelfModifyGlobs = globs
		added += n
	}
	return next, added
}

// guardSelfTightenUnion appends the not-yet-present members of `extra` onto a fresh
// copy of `base`, returning the copy and the number added. The copy matters: appending
// straight onto `base` could write through spare capacity into the live floor's
// backing array, so the "current" side of the diff would silently already contain the
// proposal.
func guardSelfTightenUnion(base, extra []string) ([]string, int) {
	out := make([]string, len(base), len(base)+len(extra))
	copy(out, base)
	present := make(map[string]struct{}, len(base))
	for _, v := range base {
		present[v] = struct{}{}
	}
	added := 0
	for _, v := range extra {
		if _, ok := present[v]; ok {
			continue
		}
		present[v] = struct{}{}
		out = append(out, v)
		added++
	}
	return out, added
}

// guardAdmitSelfTightenProposal IS the armed gate: it routes (installed, proposed)
// through admitSelfTightenOverlay and installs the proposal on rt ONLY on an admit
// verdict. A refusal leaves rt untouched — wholesale, never partially applied.
//
// It takes an already-built proposal rather than the overlay so the DELTA barrier is
// exercisable independently of the schema barrier: a test can hand it a widening
// proposal that the schema could never have produced and prove the apply path still
// refuses it. That is the difference between a gate that runs and a gate that is
// merely reachable.
func guardAdmitSelfTightenProposal(rt *policy.Runtime, proposed adjudicator.Policy) (admit bool, class, reason string) {
	if rt == nil {
		return false, policy.AmendmentFrozenViolation, "refused: no runtime to amend"
	}
	admit, class, reason = admitSelfTightenOverlay(rt.Adjudicator, proposed)
	if !admit {
		return admit, class, reason
	}
	rt.Adjudicator = proposed
	return admit, class, reason
}

// guardApplySelfTightenOverlay is the launch-boundary entry point
// loadGuardCapabilityFloor calls. It composes the union with the gate and reports the
// amendment class alongside the verdict so the caller can put the class in the floor's
// provenance — the spike's "journaled with its class" requirement, rendered on the
// surface the sibling overlays already use.
//
// An empty overlay short-circuits to a no-op admit, so the overwhelmingly common
// "no self-tighten overlay on disk" launch does not even build a proposal and the
// floor is byte-identical to the pre-#5181 one.
func guardApplySelfTightenOverlay(rt *policy.Runtime, ov guardSelfTightenOverlay) (admit bool, class, reason string, added int) {
	if rt == nil || ov.empty() {
		return true, policy.AmendmentNone, "no-op: no self-tighten overlay to apply", 0
	}
	proposed, added := guardSelfTightenProposal(rt.Adjudicator, ov)
	admit, class, reason = guardAdmitSelfTightenProposal(rt, proposed)
	if !admit {
		return admit, class, reason, 0
	}
	return admit, class, reason, added
}

// guardSelfTightenFloorNote renders the one-line provenance the launch banner carries
// for this overlay, or "" when there is nothing to say (the no-op launch). Split out
// so the floor assembly stays a straight line and the wording is unit-testable.
func guardSelfTightenFloorNote(path string, admit bool, class, reason string, added int) string {
	switch {
	case !admit:
		return fmt.Sprintf("; agent self-tighten overlay REFUSED (%s: %s; %s)", class, reason, path)
	case added > 0:
		return fmt.Sprintf(" + agent self-tighten overlay (%s: %d element(s); %s)", class, added, path)
	default:
		return ""
	}
}
