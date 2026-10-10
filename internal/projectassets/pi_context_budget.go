package projectassets

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// pi_context_budget.go — the context envelope fak writes into Pi's configuration. The
// derivation itself lives in harnesskit.DeriveContextEnvelope (one canonical rule, applied
// once to the RAW served window); this file maps it onto Pi's keys:
//
//   - models.json contextWindow = envelope.ContextWindow: the served window, capped by the
//     quality cap only on large (256k, 1M) windows. A 128k slot is used in full.
//   - models.json maxTokens = envelope.OutputTokens (also Pi's summary maxTokens bound).
//   - settings.json compaction.reserveTokens = envelope.ReserveTokens and
//     keepRecentTokens = envelope.KeepRecentTokens, so Pi compacts at CompactTrigger
//     (contextWindow - reserveTokens) and a compaction reclaims real room.
//
// The cap-vs-target split of docs/long-context-defaults.md is carried by the compaction
// trigger, not by halving the window. The retired 50% halving was applied to every window
// and could stack (131072 -> 65536), which made Halo Qwen sessions compact at 38% of the
// slot and storm.

// PiBudgetProvenance labels the evidence class behind the written budget.
const PiBudgetProvenance = harnesskit.ContextEnvelopeProvenance

// DefaultPiServedWindow is the served window assumed when the caller does not know the
// served model's real window (no backend probe, no explicit flag): the Halo Qwen slot.
const DefaultPiServedWindow = 131072

// PiContextBudget is the Pi-shaped view of one harnesskit.ContextEnvelope.
type PiContextBudget struct {
	// ServedWindow is the hard cap the backend advertises (input + output).
	ServedWindow int
	// ResidentTarget is written as Pi's contextWindow (envelope.ContextWindow).
	ResidentTarget int
	// MaxOutputTokens is written as the model's maxTokens.
	MaxOutputTokens int
	// ReserveTokens is written as compaction.reserveTokens.
	ReserveTokens int
	// KeepRecentTokens is written as compaction.keepRecentTokens.
	KeepRecentTokens int
	// Viable is false when the window cannot host the harness without a compaction
	// storm; Reason then carries the closed harnesskit reason token.
	Viable     bool
	Reason     string
	Provenance string
	Envelope   harnesskit.ContextEnvelope
}

// HaloPiColdAdmission is the cold-prompt admission a Halo serving Pi applies, so the
// trigger stays below the context a Halo admits cold. The gateway prices every chat
// request cold (internal/gateway/deadline_admission.go gives no prompt-cache credit),
// so a 0.97 KV-reuse rate does not raise this bound. MODELED priors, 2026-10-09:
//   - 600s: Pi passes no timeoutMs, so the OpenAI SDK default X-Stainless-Timeout applies;
//   - 0.9: deadlineadmit.DefaultHeadroom;
//   - 162 tok/s cold prefill and 21 tok/s decode: Halo Qwen 27B, measured by Fak Ops;
//   - 1024 completion tokens and 4096 tokens of one turn's growth past the trigger.
//
// MaxTrigger() = 74752.
var HaloPiColdAdmission = harnesskit.ColdAdmission{
	ClientDeadline:      600 * time.Second,
	Headroom:            0.9,
	PrefillTokensPerSec: 162,
	DecodeTokens:        1024,
	DecodeTokensPerSec:  21,
	TurnGrowthTokens:    4096,
}

// PiServedByHalo reports whether a router catalog owner names a Halo appliance
// (owned_by "halo-<node>").
func PiServedByHalo(ownedBy string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(ownedBy)), "halo")
}

// PiSafeContextBudget derives the Pi context budget from a RAW served window. A
// non-positive window falls back to DefaultPiServedWindow. The input must never be a
// value this function (or any other budget) already produced.
func PiSafeContextBudget(servedWindow int) PiContextBudget {
	return PiContextBudgetFor(servedWindow, 0)
}

// PiContextBudgetFor is PiSafeContextBudget with the compaction trigger capped at
// maxCompactTrigger when positive (HaloPiColdAdmission.MaxTrigger() for a Halo).
func PiContextBudgetFor(servedWindow, maxCompactTrigger int) PiContextBudget {
	if servedWindow <= 0 {
		servedWindow = DefaultPiServedWindow
	}
	env, err := harnesskit.DeriveContextEnvelope(harnesskit.ContextEnvelopeInput{
		ServedWindow:      servedWindow,
		Source:            harnesskit.WindowServed,
		MaxCompactTrigger: maxCompactTrigger,
	})
	if err != nil {
		// Unreachable: the source is served and the window positive.
		env, _ = harnesskit.DeriveContextEnvelope(harnesskit.ContextEnvelopeInput{
			ServedWindow: DefaultPiServedWindow,
			Source:       harnesskit.WindowServed,
		})
	}
	return PiContextBudget{
		ServedWindow:     env.ServedWindow,
		ResidentTarget:   env.ContextWindow,
		MaxOutputTokens:  env.OutputTokens,
		ReserveTokens:    env.ReserveTokens,
		KeepRecentTokens: env.KeepRecentTokens,
		Viable:           env.Viable,
		Reason:           env.Reason,
		Provenance:       env.Provenance,
		Envelope:         env,
	}
}

// repairPiModelBudget rewrites an existing fak model entry's contextWindow/maxTokens to the
// derived budget, returning true when it changed anything. contextWindow must equal the
// derived window in both directions (a halved 65536 entry is raised, a raw 1M entry is
// capped). maxTokens equals the derived output budget in both directions too: the
// compaction reserve is built from that budget, and fak records no maxTokens pin, so a
// stale smaller value (a hand-set 4096) only disagrees with the reserve.
func repairPiModelBudget(mObj map[string]interface{}, budget PiContextBudget) bool {
	changed := false
	if cw, ok := numericField(mObj["contextWindow"]); !ok || cw != budget.ResidentTarget {
		mObj["contextWindow"] = budget.ResidentTarget
		changed = true
	}
	if mt, ok := numericField(mObj["maxTokens"]); !ok || mt != budget.MaxOutputTokens {
		mObj["maxTokens"] = budget.MaxOutputTokens
		changed = true
	}
	return changed
}

// numericField reads a JSON number that may have decoded as float64 (the default for
// encoding/json into interface{}) or int, returning the value and whether it was a number.
func numericField(v interface{}) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}
