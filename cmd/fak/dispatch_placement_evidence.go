package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// THE CALLER THAT MAKES THE PLACEMENT EVIDENCE REAL (#5416 tracks E + F).
//
// internal/dispatchtick and internal/modelroute already hold every pure piece of this: the
// work-class parser, the rung attributor, the witness→outcome producer, and the durable
// journal writer. Until this file existed none of them had a caller, so a live fleet's
// journal was empty and the epic's headline question — "what share of our tokens did we
// self-host, and which models earned which rung?" — had no data behind it at all.
//
// The wiring is deliberately three small moves, each at the only place its fact exists:
//
//	spawn    the tick knows the issue's labels and the resolved model -> write .workclass
//	         and .zone beside the log, because both are point-in-time facts (labels get
//	         re-tagged, rosters get re-bound) that a later sweep cannot reconstruct.
//	witness  the sweep scrapes them back through the allowlist parsers and hands them to
//	         the producer as its Class / Zone hooks.
//	journal  the produced outcomes are appended to a jsonl the grader reads.
//
// The whole seam is OFF unless `fak dispatch tick --placement-evidence` says otherwise,
// matching how every other dispatch-layer knob here lands: an unconfigured fleet writes no
// extra sidecars, adds no payload keys, and creates no journal — byte-identical to before.
//
// Two refusals are load-bearing and both point the same way, away from over-claiming:
//
//   - The roster is read ONLY through Roster.BoundZone, never Resolve. Resolve is a
//     dispatch primitive and falls back to the default account for an unbound id; used for
//     attribution it would report the default's rung for every model nobody bound, which on
//     a fleet-default roster means counting vendor spend as self-hosted.
//   - A slot whose class or rung cannot be named produces a REASON key, not a value. The
//     reason is the work item ("triage this backlog", "bind this model"); a default would
//     be a fabricated measurement of the thing the epic exists to measure.

// dispatchTurnJournalName is the append-only per-turn evidence journal, written inside the
// runs directory so it lives and dies with the dispatch state it describes.
const dispatchTurnJournalName = "turn-outcomes.jsonl"

// payload keys carrying the resolved facts from the prepare step to the spawn step. The
// tick's payload is the established channel between the two (see placement_gate,
// startup_bundle), and it doubles as the operator-visible record of what was decided.
const (
	dispatchWorkClassKey        = "work_class"
	dispatchWorkClassUnknownKey = "work_class_unknown"
	dispatchZoneKey             = "placement_zone"
	dispatchZoneUnknownKey      = "placement_zone_unknown"
)

// The two DECLARATIONS behind this seam, and why they are declarations rather than
// environment reads.
//
// Both are behavioral settings — one switches an evidence-recording seam on, the other names
// a file to attribute against — and neither is a credential, so both belong on the config
// surface and not in the environment. That is the rule internal/envconfiglint ratchets
// (CONFIG_NOT_ENV): an os.LookupEnv of a non-secret name is refused, because a setting read
// out of the ambient environment is invisible to `--help`, unrecorded in the tick payload,
// and impossible to give a per-invocation value. `fak dispatch tick` owns them now, as
// --placement-evidence and --accounts-roster, and evaluateDispatchTick publishes the parsed
// dispatchTickOptions here.
//
// Package seams rather than parameters, for the reason dispatchTickView is one: the readers
// hang off three different call chains (prepare → record, spawn → sidecars, witness sweep →
// journal, plus the rung ladder next door), so a parameter would grow the signature of every
// helper and stub in between to carry a value only the leaves use. The zero values are the
// pre-seam posture exactly — off, and no roster override.
var (
	dispatchPlacementEvidence bool
	dispatchAccountsRoster    string
)

// dispatchPlacementEvidenceEnabled reports whether the opt-in placement-evidence seam
// (`fak dispatch tick --placement-evidence`) is switched on. Default (undeclared) is OFF.
func dispatchPlacementEvidenceEnabled() bool { return dispatchPlacementEvidence }

// dispatchAccountsRosterPath locates the model-account roster used for ZONE ATTRIBUTION
// only — this path never dispatches anything. A declared --accounts-roster wins; otherwise a
// conventional tools/model-accounts.json is used when it exists. An empty result means no
// roster, which attributes nothing rather than defaulting a rung.
func dispatchAccountsRosterPath(root string) string {
	if p := strings.TrimSpace(dispatchAccountsRoster); p != "" {
		return p
	}
	p := filepath.Join(root, "tools", "model-accounts.json")
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		return p
	}
	return ""
}

// dispatchZoneResolver loads the roster and returns its BOUND-ONLY zone lookup, or nil when
// there is no readable roster. Nil is the honest answer: AttributeZone turns it into the
// "no-roster" reason, which is a configuration work item, not a rung.
//
// LoadRoster validates, so a malformed roster attributes nothing instead of half-attributing
// from a partially parsed file.
func dispatchZoneResolver(root string) dispatchtick.ZoneResolver {
	path := dispatchAccountsRosterPath(root)
	if path == "" {
		return nil
	}
	r, err := modelroute.LoadRoster(path)
	if err != nil {
		return nil
	}
	return r.BoundZone
}

// recordDispatchPlacementEvidence resolves both point-in-time facts at prepare time and
// writes them into the payload — the value when it is known, the reason when it is not.
//
// It is called only when the seam is enabled, so an unconfigured tick's payload keeps
// exactly the keys it had before.
func recordDispatchPlacementEvidence(root string, labels []string, model string, payload map[string]any) {
	if payload == nil {
		return
	}
	if class, why := dispatchtick.WorkClassForIssue(labels); class != "" {
		payload[dispatchWorkClassKey] = string(class)
	} else {
		payload[dispatchWorkClassUnknownKey] = string(why)
	}
	zone, why := dispatchtick.AttributeZone(dispatchZoneResolver(root), model)
	if zone != "" {
		payload[dispatchZoneKey] = string(zone)
	} else {
		payload[dispatchZoneUnknownKey] = string(why)
	}
}

// writeDispatchPlacementSidecars persists whatever recordDispatchPlacementEvidence resolved
// beside the worker log. A missing key writes no file — a slot nothing could classify or
// attribute leaves no sidecar, so the sweep reads its absence as the same "not recorded" the
// payload already said out loud. Fail-open, like every other sidecar write here.
func writeDispatchPlacementSidecars(logPath string, payload map[string]any) {
	write := func(suffix, value string) {
		p := dispatchtick.SidecarPath(logPath, suffix)
		if p == "" || strings.TrimSpace(value) == "" {
			return
		}
		_ = os.WriteFile(p, []byte(value), 0o644)
	}
	write(dispatchtick.WorkClassSidecarSuffix, dispatchMapString(payload, dispatchWorkClassKey))
	write(dispatchtick.ZoneSidecarSuffix, dispatchMapString(payload, dispatchZoneKey))
}

// dispatchScrapedWorkClasses reads each finished slot's .workclass sidecar, keyed by the
// same basename WitnessRecord.Log carries. Values that fail the allowlist are dropped, which
// classes that slot as ungraded rather than letting an edited file pick its own rung.
func dispatchScrapedWorkClasses(runsDir string, records []dispatchtick.WitnessRecord) map[string]modelroute.WorkClass {
	out := map[string]modelroute.WorkClass{}
	for _, r := range records {
		p := dispatchtick.SidecarPath(filepath.Join(runsDir, r.Log), dispatchtick.WorkClassSidecarSuffix)
		if p == "" {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		if c, ok := dispatchtick.WorkClassFromSidecar(string(b)); ok {
			out[r.Log] = c
		}
	}
	return out
}

// appendDispatchTurnOutcomes folds this sweep's witness records into capability evidence and
// appends it to the durable journal. It returns the accounting for the tick payload, so an
// operator sees how much of the sweep became evidence AND how much could not — the two
// numbers that decide whether a later grade describes the fleet or a corner of it.
//
// Double counting is structurally prevented upstream: witnessExitedWorkers skips any slot
// that already has a .witness sidecar, so a finished worker is audited — and so journalled —
// exactly once no matter how many ticks run. modelroute's own read-side dedupe on
// TurnOutcome.ID is the second line, not the first.
//
// Fail-open: a journal that cannot be opened or written reports the error in the payload and
// never fails the tick. Dispatch continuing matters more than evidence completeness, and the
// gap is surfaced rather than swallowed.
func appendDispatchTurnOutcomes(runsDir string, records []dispatchtick.WitnessRecord) map[string]any {
	if len(records) == 0 {
		return nil
	}
	classes := dispatchScrapedWorkClasses(runsDir, records)
	outcomes, stats := dispatchtick.TurnOutcomesFromWitness(records, dispatchtick.WitnessEvidenceOptions{
		Class: dispatchtick.ClassResolverFromSidecars(classes),
		At:    dispatchLogFinishedAt(runsDir),
		Zone: func(r dispatchtick.WitnessRecord) modelroute.PlacementZone {
			z, _ := dispatchtick.ZoneFromSidecar(r.Zone)
			return z
		},
	})
	share := dispatchtick.FoldZoneShare(records)
	out := map[string]any{
		"produced":     stats.Produced,
		"unattributed": stats.Unattributed,
		"unclassified": stats.Unclassified,
		"undated":      stats.Undated,
		"unidentified": stats.Unidentified,
		"zone_share":   share.Headline(),
	}
	if len(outcomes) == 0 {
		return out
	}
	path := filepath.Join(runsDir, dispatchTurnJournalName)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		out["journal_error"] = err.Error()
		return out
	}
	defer f.Close()
	written := 0
	for _, o := range outcomes {
		if err := modelroute.AppendTurnOutcome(f, o); err != nil {
			out["journal_error"] = err.Error()
			break
		}
		written++
	}
	out["journal"] = path
	out["appended"] = written
	if written != len(outcomes) {
		out["journal_short"] = fmt.Sprintf("%d of %d outcome(s) appended", written, len(outcomes))
	}
	return out
}

// dispatchLogFinishedAt stamps each outcome with its worker log's last-write time — the
// closest durable proxy for when the slot finished. An unstattable log leaves the zero time,
// which the producer counts as undated: such a row still grades, it just cannot satisfy a
// freshness window later.
func dispatchLogFinishedAt(runsDir string) func(dispatchtick.WitnessRecord) time.Time {
	return func(r dispatchtick.WitnessRecord) time.Time {
		if strings.TrimSpace(r.Log) == "" {
			return time.Time{}
		}
		st, err := os.Stat(filepath.Join(runsDir, r.Log))
		if err != nil {
			return time.Time{}
		}
		return st.ModTime()
	}
}
