package gitgate

// ATTRIBUTION-AWARE SWEEP GUARD (#3879, epic #3871). A path-scoped mutating git op —
// `git commit -- <path>`, `git checkout <path>`, `git reset <path>` — silently
// overwrites whatever uncommitted delta sits under that pathspec. On a shared
// working tree that delta may belong to a PEER session, not the actor. gitgate's
// pre-existing trunk laws refuse a whole-tree `git stash` and warn on shared-tree
// mutation, but that refusal is BLUNT: it cannot tell "this dirty path is mine"
// from "this belongs to a crashed peer." This rung makes the gate PRECISE by
// consulting the pure attribution fold (internal/wipattr, C2 #3874): every dirty
// hunk the op would sweep is already classified OWNED (a session's WIP checkpoint
// records the identical edit) / SHARED / ORPHAN (no checkpoint records it).
//
// The escalation the issue asks for falls straight out of that classification,
// because the AttrState IS the checkpoint axis:
//
//   - OWNED-by-self  -> SAFE     -> ALLOW (your own path; unaffected).
//   - OWNED-by-peer  -> checkpointed peer delta -> ADVISE (defer): the delta is
//     recoverable from refs/fak/wip/<owner>, so warn + name the peer and let the
//     operator own the blast radius (the advisory-first override path).
//   - SHARED         -> checkpointed but ambiguous -> ADVISE (defer): name the
//     claimants; a human reconciles.
//   - ORPHAN         -> un-checkpointed peer WIP with NO refs/fak/wip/* safety net
//     -> REFUSE (deny): irrecoverable if swept, so this is the one case that
//     escalates past advisory to a provable POLICY_BLOCK.
//
// It never shells out and reads no repo state: the caller (cmd/fak) parses the
// op's `git diff` into wipattr.Hunks, runs wipattr.Attribute against the live
// refs/fak/wip/* checkpoints, and hands the resulting []wipattr.Attribution here
// as the SweepGuardPlan.Targets. Like the collective-commit barrier, this is a
// synthetic tool (ToolSweepGuard) whose args are a JSON plan, not a shell string.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// ToolSweepGuard is the synthetic tool name for the attribution-aware sweep guard.
// It never shells out; its args are a SweepGuardPlan JSON object.
const ToolSweepGuard = "gitgate.sweep_guard"

// ReasonPeerWIPCollision reports that a path-scoped git op would sweep WIP owned
// by a peer session (#11232).
const ReasonPeerWIPCollision = "PEER_WIP_COLLISION"

// SweepGuardPlan is the attribution-decidable shape screened before a path-scoped
// mutating git op. Op is the human label of the op being screened (e.g.
// "git commit -- internal/foo", "git checkout", "git reset internal/bar"); Self is
// the acting session id; Live lists the currently-live session ids (liveness
// sharpens the warning but never downgrades a hazard to safe); Targets is the
// attribution of every dirty hunk the op's pathspec would sweep, built by the
// caller with wipattr.Attribute over the op's `git diff`.
type SweepGuardPlan struct {
	Op            string                `json:"op"`
	Self          string                `json:"self"`
	Live          []string              `json:"live"`
	Targets       []wipattr.Attribution `json:"targets"`
	QuarantineRef string                `json:"quarantine_ref,omitempty"`
	AutoArchive   bool                  `json:"auto_archive,omitempty"`
}

// SweepDisposition is the escalation level of a sweep-guard finding.
type SweepDisposition int

const (
	// SweepClear: no peer WIP sits under the pathspec — safe to proceed.
	SweepClear SweepDisposition = iota
	// SweepAdvise: a CHECKPOINTED peer (OWNED-by-peer or SHARED) delta is in the
	// sweep — recoverable from refs/fak/wip/<owner>, so warn and let the operator
	// decide (advisory-first, non-blocking).
	SweepAdvise
	// SweepRefuse: an ORPHAN (un-checkpointed) delta is in the sweep — no safety
	// net, irrecoverable if swept, so refuse.
	SweepRefuse
)

// SweepGuardFinding is the structured result of CheckSweepGuard.
type SweepGuardFinding struct {
	Disposition SweepDisposition
	Reason      string                 // structured refusal reason (e.g. ReasonPeerWIPCollision)
	Claim       string                 // human-readable readout naming the peer session(s) + checkpoint ref
	Hazards     []wipattr.SweepVerdict // the non-safe verdicts (peer/shared/orphan), for callers and tests
}

// CheckSweepGuard folds a SweepGuardPlan into a disposition without reading repo
// state. It reuses wipattr.SweepGuard (the pure SAFE/HAZARD fold) and then applies
// gitgate's own escalation policy: an ORPHAN hazard REFUSES (irrecoverable), every
// other hazard ADVISES (checkpointed, recoverable). All-safe is CLEAR.
func CheckSweepGuard(plan SweepGuardPlan) SweepGuardFinding {
	live := make(map[string]bool, len(plan.Live))
	for _, id := range plan.Live {
		if id = strings.TrimSpace(id); id != "" {
			live[id] = true
		}
	}
	verdicts := wipattr.SweepGuard(plan.Targets, strings.TrimSpace(plan.Self), live)
	hazards := wipattr.SweepHazards(verdicts)
	if len(hazards) == 0 {
		return SweepGuardFinding{Disposition: SweepClear}
	}

	op := strings.TrimSpace(plan.Op)
	if op == "" {
		op = "the path-scoped git op"
	}
	// An ORPHAN in the sweep is the one irrecoverable case — escalate past advisory.
	// When backed by a quarantine ref archive (#11238), orphan files have been captured
	// into refs/fak/quarantine/* and are recoverable, permitting the clean/purge.
	for _, h := range hazards {
		if h.State == wipattr.AttrOrphan {
			if plan.QuarantineRef != "" || plan.AutoArchive {
				continue // Backed by quarantine archive; recoverable.
			}
			return SweepGuardFinding{
				Disposition: SweepRefuse,
				Claim:       sweepRefuseClaim(op, hazards),
				Hazards:     hazards,
			}
		}
	}
	hasNonOrphanHazards := false
	for _, h := range hazards {
		if h.State != wipattr.AttrOrphan {
			hasNonOrphanHazards = true
			break
		}
	}
	if !hasNonOrphanHazards && (plan.QuarantineRef != "" || plan.AutoArchive) {
		claim := fmt.Sprintf("%s allowed: orphan files backed by quarantine archive", op)
		if plan.QuarantineRef != "" {
			claim = fmt.Sprintf("%s allowed: orphan files backed by quarantine archive (%s)", op, plan.QuarantineRef)
		}
		return SweepGuardFinding{
			Disposition: SweepClear,
			Claim:       claim,
			Hazards:     hazards,
		}
	}
	// Elevate OWNED-by-peer from SweepAdvise to SweepRefuse with ReasonPeerWIPCollision (#11232).
	for _, h := range hazards {
		if h.State == wipattr.AttrOwned {
			return SweepGuardFinding{
				Disposition: SweepRefuse,
				Reason:      ReasonPeerWIPCollision,
				Claim:       sweepPeerWIPCollisionClaim(op, hazards),
				Hazards:     hazards,
			}
		}
	}
	return SweepGuardFinding{
		Disposition: SweepAdvise,
		Claim:       sweepAdviseClaim(op, hazards),
		Hazards:     hazards,
	}
}

// sweepPeerWIPCollisionClaim reports the peer WIP that causes a refusal under strict file-level attribution (#11232).
func sweepPeerWIPCollisionClaim(op string, hazards []wipattr.SweepVerdict) string {
	var peers []string
	for _, h := range hazards {
		if h.State == wipattr.AttrOwned {
			peers = append(peers, fmt.Sprintf("%s belongs to session %s (checkpointed at %s)",
				h.File, h.Owner, wipref.SessionRef(h.Owner)))
		}
	}
	return fmt.Sprintf("%s refused: %s: it would sweep a peer's WIP: %s — reconcile rather than sweep, or narrow the pathspec to your own files before retrying.",
		op, ReasonPeerWIPCollision, strings.Join(peers, "; "))
}

// sweepAdviseClaim names each checkpointed peer/shared delta and its refs/fak/wip/*
// safety net so the operator can reconcile instead of sweeping.
func sweepAdviseClaim(op string, hazards []wipattr.SweepVerdict) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s would sweep a peer's CHECKPOINTED WIP: ", op)
	for i, h := range hazards {
		if i > 0 {
			b.WriteString("; ")
		}
		switch h.State {
		case wipattr.AttrShared:
			fmt.Fprintf(&b, "%s is shared across sessions %s (checkpointed at %s)",
				h.File, strings.Join(h.Owners, ","), sweepRefList(h.Owners))
		default: // AttrOwned by a peer
			fmt.Fprintf(&b, "%s belongs to session %s (checkpointed at %s)",
				h.File, h.Owner, wipref.SessionRef(h.Owner))
		}
	}
	b.WriteString(" — advisory: the delta is recoverable from its checkpoint; reconcile rather than sweep, or override to proceed.")
	return b.String()
}

// sweepRefuseClaim reports the ORPHAN drivers — un-checkpointed peer WIP with no
// safety net — that make the op irrecoverable and so refused.
func sweepRefuseClaim(op string, hazards []wipattr.SweepVerdict) string {
	var orphans []string
	for _, h := range hazards {
		if h.State == wipattr.AttrOrphan {
			orphans = append(orphans, h.File)
		}
	}
	return fmt.Sprintf("%s refused: it would sweep UN-CHECKPOINTED WIP with no refs/fak/wip/* safety net (irrecoverable): %s"+
		" — checkpoint the delta (fak wip) or narrow the pathspec to your own files before retrying.",
		op, strings.Join(orphans, ", "))
}

// sweepRefList renders the refs/fak/wip/<owner> checkpoint refs for a SHARED set.
func sweepRefList(owners []string) string {
	refs := make([]string, len(owners))
	for i, o := range owners {
		refs[i] = wipref.SessionRef(o)
	}
	return strings.Join(refs, ",")
}

// adjudicateSweepGuard is the ToolSweepGuard entrypoint: it unmarshals the plan,
// folds it, and maps the disposition to a verdict. Refuse is a provable
// POLICY_BLOCK deny (recorded to the witness sink); Advise is a non-blocking Defer
// carrying the warning; Clear allows.
func (g *GitGate) adjudicateSweepGuard(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	var plan SweepGuardPlan
	b := refBytes(ctx, c.Args)
	if len(b) == 0 {
		return sweepDeny(abi.ReasonMalformed, "sweep-guard missing JSON args")
	}
	if err := json.Unmarshal(b, &plan); err != nil {
		return sweepDeny(abi.ReasonMalformed, "sweep-guard malformed JSON args: "+err.Error())
	}
	finding := CheckSweepGuard(plan)
	switch finding.Disposition {
	case SweepRefuse:
		v := sweepDeny(abi.ReasonPolicyBlock, finding.Claim)
		reasonClass := abi.ReasonName(v.Reason)
		if finding.Reason != "" {
			reasonClass = finding.Reason
		}
		g.recordRefusal(ctx, ToolSweepGuard, reasonClass, []string{ToolSweepGuard}, sweepFiles(finding.Hazards))
		return v
	case SweepAdvise:
		// Advisory-first: gitgate does not block; it surfaces a warning naming the
		// peer + its checkpoint and defers the blast-radius call to the operator.
		return abi.Verdict{
			Kind:    abi.VerdictDefer,
			By:      ToolSweepGuard,
			Payload: abi.WitnessPayload{Claim: finding.Claim},
			Meta:    map[string]string{"warn": finding.Claim},
		}
	default: // SweepClear
		return abi.Verdict{Kind: abi.VerdictAllow, By: ToolSweepGuard}
	}
}

func sweepDeny(reason abi.ReasonCode, claim string) abi.Verdict {
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  reason,
		By:      ToolSweepGuard,
		Payload: abi.WitnessPayload{Claim: claim},
		Meta:    map[string]string{"fix": claim},
	}
}

func sweepFiles(hazards []wipattr.SweepVerdict) []string {
	out := make([]string, 0, len(hazards))
	for _, h := range hazards {
		out = append(out, h.File)
	}
	return out
}
