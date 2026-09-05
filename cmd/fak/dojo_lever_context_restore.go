package main

import (
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// dojo_lever_context_restore.go — the context-restore/restore_recall lever (#4486),
// registered through the additive RegisterLever seam (#5108) so the cell lands in
// its own file with no edit to cmd/fak/dojo.go. The pure fold + the one anchored
// claim live in internal/dojo (claim_context_restore.go); this file is only the
// registration plus the thin ledger adapter the tier-1 core must not do itself.
//
// The lever scores fak_context_restore's hit rate on dropped context spans against
// fak's OWN durable context-span ledger — the gateway-usage ledger
// (.fak/nightrun/gateway-usage.jsonl), which records each compaction's dropped
// turns (CompactionDroppedTurns). That ledger records DROPS but carries no restore
// counter today, so the cell scores UNMEASURED honestly (never a fabricated 0.0)
// until a restore field lands on the gateway usage row — the concrete extension
// seam this KPI calibrates against. `fak dojo run` folds the cell like any other.
var _ = RegisterLever(dojoLeverInfo{
	Name:    "context-restore",
	Summary: "fak_context_restore hit rate on dropped context spans — restored spans over dropped spans, folded from fak's own durable context-span ledger (the gateway-usage ledger's compaction_dropped_turns). The KPI for fak's context-continuity concept (#4486); the ledger records drops but relays no restore counter yet, so the cell scores UNMEASURED honestly until a restore field lands, never a fabricated 0.0",
	Metrics: []dojoMetricInfo{
		{Name: "restore_recall", Theory: "about half of dropped context spans are later paged back in by fak_context_restore (claim 0.5 — a seeded estimate the RSI loop recalibrates toward the measured fraction; UNMEASURED until the ledger records restores)"},
	},
}, func(env dojoLeverEnv) dojo.Lever { return contextRestoreLever{root: env.Root} })

// contextRestoreLever folds fak's durable context-span drop/restore ledger into the
// restore-recall cell. root is the workspace root the gateway-usage ledger lives
// under.
type contextRestoreLever struct{ root string }

func (contextRestoreLever) Name() string { return "context-restore" }

func (l contextRestoreLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return dojo.ContextRestoreEpisodes(loadContextSpanLedger(l.root)), nil
}

// loadContextSpanLedger reduces fak's durable context-span ledger — the
// gateway-usage ledger, which records each compaction's dropped turns — to the
// drop/restore counts the pure fold needs. Only exit rows are summed (the same
// choice compaction_segments makes: a periodic or carryforward snapshot would
// double-count a session's drops). Fail-open: a missing/unreadable ledger yields
// a zero-drop ledger and the fold reports the honest "no dropped spans" UNMEASURED.
func loadContextSpanLedger(root string) dojo.ContextSpanLedger {
	rows := gatewayusageledger.ReadLedgerFile(filepath.Join(root, gatewayusageledger.DefaultLedgerRel))
	var dropped uint64
	var restored uint64
	var hasRestoreRecorded bool
	for _, r := range rows {
		if r.Kind != "exit" {
			continue // fold only exit rows so periodic/carryforward snapshots don't double-count
		}
		dropped += r.Counters.CompactionDroppedTurns
		restored += r.Counters.CompactionRestoredTurns
		hasRestoreRecorded = true
	}
	return dojo.ContextSpanLedger{
		DroppedSpans:    int(dropped),
		RestoredSpans:   int(restored),
		RestoreRecorded: hasRestoreRecorded,
	}
}
