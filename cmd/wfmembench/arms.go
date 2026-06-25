package main

import (
	"context"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/contextq"
)

// This file is the three-arm workflow-memory comparator that issue #434 asks for.
// It scores three memory-substrate POLICIES over the same finished session, on the
// metric set the issue names: resident bytes/tokens, view/page fault rate, source
// coverage, stale-view reuse, poison leakage, task success, and fallback-to-raw.
//
// Two of the arms are MODELED baselines — closed-form reductions over the image's
// own page table, so they are deterministic and need no model in the loop:
//
//	full-transcript        keep every page raw and resident; filter nothing.
//	naive-global-summary   compress every page into one lossy blob; no provenance.
//
// The third arm is MEASURED: it drives the real provenance-bound derived-view
// substrate (contextq.RunWorkflowMemoryBench) and reads the metrics back from the
// per-turn verdicts and the page table. Its two fail-closed witnesses — StaleReuse
// and PoisonLeakage — are zero because the substrate enforces it, not because the
// workload never tried (the poison probe fires; the policy boundary is crossed).
//
// The `Kind` field labels each arm so a reader never mistakes a modeled baseline
// for a measured number. The comparison's conclusion is invariant to the one stated
// model parameter (the naive summary's per-page extractive budget).

// summaryHeadBytes is the per-page extractive budget the naive-global-summary arm
// keeps. It is a stated model parameter: the leak/provenance/fallback verdicts the
// arm reports do not depend on its exact value, only that the summary is lossy.
const summaryHeadBytes = 48

// tokensOf is the repo's rough byte->token estimate (~4 bytes/token), ceiling-rounded.
func tokensOf(b int64) int64 { return (b + 3) / 4 }

// ArmReport is one memory-substrate policy's behavior on the fixture, on the metric
// set issue #434 names. Lower is safer for StaleReuse and PoisonLeak (0 is ideal).
type ArmReport struct {
	Name string `json:"name"`
	Kind string `json:"kind"` // "modeled" (closed-form baseline) | "measured" (real substrate)

	ResidentBytes  int64   `json:"resident_bytes"`  // bytes the arm keeps available to the model
	ResidentTokens int64   `json:"resident_tokens"` // ~bytes/4 estimate
	ViewFaultRate  float64 `json:"view_fault_rate"` // fraction of materializations that page in a raw page
	SourceCoverage float64 `json:"source_coverage"` // fraction of benign sources individually attributable
	StaleReuse     int     `json:"stale_reuse"`     // stale/tombstoned source served without rejection
	PoisonLeak     int     `json:"poison_leak"`     // sealed/quarantined content reaching the model
	TaskSuccess    bool    `json:"task_success"`    // could the arm answer the goal
	FallbackToRaw  float64 `json:"fallback_to_raw"` // fraction of materializations that fell back to a raw page-in
	Note           string  `json:"note"`
}

// Fixture is the structural decomposition of the benchmark session.
type Fixture struct {
	Pages         int      `json:"pages"`
	Benign        int      `json:"benign"`
	Sealed        int      `json:"sealed"`
	Tombstoned    int      `json:"tombstoned"`
	RawBytes      int64    `json:"raw_bytes"`
	ResidentBytes int64    `json:"resident_bytes"`
	Hazards       []string `json:"hazards"`
}

// StaleReplay is acceptance #4: a source changes and the old view is rejected.
type StaleReplay struct {
	Description     string `json:"description"`
	Recomputes      int    `json:"recomputes"`        // a PolicyA view served under PolicyB -> RECOMPUTE
	StaleReuse      int    `json:"stale_reuse"`       // must be 0: a stale view served as a HIT
	OldViewRejected bool   `json:"old_view_rejected"` // Recomputes>0 AND StaleReuse==0
}

// PoisonReplay is acceptance #5: a sealed source cannot leak through a derived view.
type PoisonReplay struct {
	Description     string `json:"description"`
	SealedRefused   int    `json:"sealed_refused"`   // the poison probe was refused
	PoisonLeakage   int    `json:"poison_leakage"`   // must be 0
	SealedContained bool   `json:"sealed_contained"` // SealedRefused>0 AND PoisonLeakage==0
}

// Comparison is the full #434 deliverable: the fixture, the three arms, and the two
// replay witnesses, plus the command that reproduces it.
type Comparison struct {
	Issue   string       `json:"issue"`
	Command string       `json:"command"`
	Fixture Fixture      `json:"fixture"`
	Arms    []ArmReport  `json:"arms"`
	Stale   StaleReplay  `json:"stale_replay"`
	Poison  PoisonReplay `json:"poison_replay"`
}

// fullTranscriptArm models the do-nothing baseline: keep the entire flat transcript
// resident and untrusted. It is cheap to be "correct" (the answer is in the bytes)
// but carries the most bytes AND leaks every sealed/tombstoned page — nothing is
// filtered. Computed in closed form from the page table.
func fullTranscriptArm(im *cdb.Image) ArmReport {
	info := im.Info()
	var poison, stale int
	for _, f := range im.Backtrace() {
		if f.Sealed {
			poison++ // sealed/poisoned content sits raw in the kept transcript
		}
		if f.Tombstoned {
			stale++ // a page the agent asked to suppress is replayed anyway
		}
	}
	return ArmReport{
		Name:           "full-transcript",
		Kind:           "modeled",
		ResidentBytes:  info.RawBytes,
		ResidentTokens: tokensOf(info.RawBytes),
		ViewFaultRate:  1.0, // no view cache: every page is a raw page-in every turn
		SourceCoverage: 1.0, // every source is present verbatim and attributable
		StaleReuse:     stale,
		PoisonLeak:     poison,
		TaskSuccess:    true, // the answer is in the raw bytes
		FallbackToRaw:  1.0,  // it is all raw
		Note:           "baseline: carries the whole flat transcript; sealed and tombstoned pages leak because nothing is filtered.",
	}
}

// naiveSummaryArm models the chatbot-style baseline: compress every page into one
// global blob with no trust gate and no provenance. It is the cheapest in bytes but
// folds sealed content into the summary, cannot reject a stale source, destroys
// per-source attribution, and — having discarded the raw pages — cannot fall back to
// a source. Computed in closed form from the page table.
func naiveSummaryArm(im *cdb.Image) ArmReport {
	var resident int64
	var poison, stale int
	for _, f := range im.Backtrace() {
		head := f.Len
		if head > summaryHeadBytes {
			head = summaryHeadBytes
		}
		resident += head
		if f.Sealed {
			poison++ // a naive summarizer folds raw sealed bytes into the global blob
		}
		if f.Tombstoned {
			stale++ // summarized before suppression, never re-checked
		}
	}
	return ArmReport{
		Name:           "naive-global-summary",
		Kind:           "modeled",
		ResidentBytes:  resident,
		ResidentTokens: tokensOf(resident),
		ViewFaultRate:  0.0, // one-shot summary: never faults, but never refreshes
		SourceCoverage: 0.0, // a single blob destroys per-source provenance: nothing is individually attributable
		StaleReuse:     stale,
		PoisonLeak:     poison,
		TaskSuccess:    true, // the headline fact usually survives truncation
		FallbackToRaw:  0.0,  // the raw pages are discarded: it CANNOT fall back to a source
		Note:           "baseline: one lossy blob; cheapest in bytes but loses provenance, leaks sealed content, and cannot fall back to raw.",
	}
}

// virtualViewsArm is the MEASURED arm: it drives the real provenance-bound derived-
// view substrate and reads every metric back from the per-turn results. It returns
// the underlying report too, so the comparator can surface the stale/poison replay
// witnesses without re-running the workload.
func virtualViewsArm(ctx context.Context, im *cdb.Image, req contextq.BenchRequest) (ArmReport, contextq.WorkflowMemoryReport) {
	rep := contextq.RunWorkflowMemoryBench(ctx, im, req)
	return ArmReport{
		Name:           "provenance-bound-virtual-views",
		Kind:           "measured",
		ResidentBytes:  rep.ResidentBytes,
		ResidentTokens: tokensOf(rep.ResidentBytes),
		ViewFaultRate:  rep.FaultRate,
		SourceCoverage: rep.SourceCoverage,
		StaleReuse:     rep.StaleReuse,
		PoisonLeak:     rep.PoisonLeakage,
		TaskSuccess:    rep.TaskSuccess,
		FallbackToRaw:  rep.FaultRate, // falls back to a gated raw page-in when no fresh view exists
		Note:           "fak: demand-paged provenance-bound views; sealed pages refused, stale views recomputed across the policy boundary — fail-closed.",
	}, rep
}

// Compare runs all three arms over the attached image and assembles the #434
// deliverable. The image must already carry any context-control mutations (e.g. the
// tombstone on the superseded preference page) the caller wants exercised.
func Compare(ctx context.Context, im *cdb.Image, req contextq.BenchRequest, command string) Comparison {
	info := im.Info()
	views, rep := virtualViewsArm(ctx, im, req)
	return Comparison{
		Issue:   "#434",
		Command: command,
		Fixture: Fixture{
			Pages:         info.Pages,
			Benign:        info.Benign,
			Sealed:        info.Sealed,
			Tombstoned:    info.Tombstoned,
			RawBytes:      info.RawBytes,
			ResidentBytes: info.ResidentBytes,
			Hazards:       fixtureHazards(),
		},
		Arms: []ArmReport{fullTranscriptArm(im), naiveSummaryArm(im), views},
		Stale: StaleReplay{
			Description:     "a goal view built under policy epoch A is served again under epoch B; the stale view is rejected and RECOMPUTEd, never served as a HIT.",
			Recomputes:      rep.Recomputes,
			StaleReuse:      rep.StaleReuse,
			OldViewRejected: rep.Recomputes > 0 && rep.StaleReuse == 0,
		},
		Poison: PoisonReplay{
			Description:     "a query targeting a sealed source's descriptor is REFUSED; no sealed or tombstoned page enters any derived view.",
			SealedRefused:   rep.SealedRefused,
			PoisonLeakage:   rep.PoisonLeakage,
			SealedContained: rep.SealedRefused > 0 && rep.PoisonLeakage == 0,
		},
	}
}

// fixtureHazards is the catalog of workflow-memory hazards the embedded session
// encodes — the six issue #434 requires, plus the second sealed page that hardens
// the poison-leak probe and the verified/unverified effect-claim pair.
func fixtureHazards() []string {
	return []string{
		"clean tool result (account record, order status)",
		"stale mutable source (order world_epoch A, superseded preference)",
		"poisoned/sealed result (prompt injection)",
		"poisoned/sealed result (secret exfil)",
		"tombstoned page (agent-suppressed stale preference)",
		"multi-agent handoff (audit-bot sub-agent)",
		"verified effect claim (refund settled=true, receipt_sha)",
		"unverified effect claim (note: no confirmation checked)",
	}
}

// tombstoneStalePreference files a context-control tombstone on the superseded
// preference page, exercising the "tombstoned page" hazard: a page the agent
// explicitly suppressed must be absent from every derived view (and is — only the
// full-transcript baseline replays it). It is a no-op if the page is absent.
func tombstoneStalePreference(im *cdb.Image) {
	for _, f := range im.Backtrace() {
		if f.Sealed || f.Tombstoned {
			continue
		}
		if strings.Contains(f.Descriptor, "superseded") {
			// Digest is omitted on purpose: Frame.Digest is truncated for display, so
			// passing it would trip the manifest's full-digest equality check.
			_, _ = im.RequestContextChange(recallTombstone(f.Step))
			return
		}
	}
}
