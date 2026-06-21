package turnbench

import (
	"fmt"
	"io"
)

// PrintReport renders the turn-tax report as an operator-readable summary. It
// leads with the two axes kept apart — the happy-path turn-tax (what the 1-shot
// deletes) and the deterministic safety floor (what it prevents) — then the
// ablation table, the ON/OFF path-swap proof, and the cost sensitivity.
func PrintReport(w io.Writer, r *Report) {
	fmt.Fprintf(w, "== fak turntax: %s  (%d calls, hash %s) ==\n", r.Provenance.SliceID, r.Calls, r.Provenance.WorkloadHash)
	fmt.Fprintf(w, "consistency guard (counters==classification, not an independent oracle): %s\n\n", r.ConsistencyCheck)

	fmt.Fprintln(w, "-- class breakdown (live kernel verdicts) --")
	fmt.Fprintf(w, "  grammar repair (TRANSFORM)  : %d   -> saved %d baseline reparse turns\n", r.Class.Grammar, r.Class.Grammar)
	fmt.Fprintf(w, "  vdso tier-1 pure            : %d\n", r.Class.VDSOPure)
	fmt.Fprintf(w, "  vdso tier-2 dedup (cache)   : %d\n", r.Class.VDSODedup)
	fmt.Fprintf(w, "  vdso tier-3 static          : %d\n", r.Class.VDSOStatic)
	fmt.Fprintf(w, "  vdso total                  : %d   -> saved %d baseline round-trip turns\n", r.Class.vdsoTotal(), r.Class.vdsoTotal())
	fmt.Fprintf(w, "  quarantine (poison held)    : %d   [safety floor]\n", r.Class.Quarantine)
	fmt.Fprintf(w, "  deny (capability floor)     : %d   [safety floor]\n", r.Class.Deny)
	fmt.Fprintf(w, "  pass (allow+engine, control): %d   -> 0 saved (both arms pay it)\n\n", r.Class.Pass)

	fmt.Fprintln(w, "-- NET turn-tax (happy path, deterministic) --")
	fmt.Fprintf(w, "  turns saved                 : %d  (forced %d = grammar+dedup; elision %d = pure+static)\n",
		r.Net.TurnsSaved, r.TurnKinds.Forced, r.TurnKinds.Elision)
	fmt.Fprintf(w, "  tokens saved                : %d  (@ %d+%d tok/turn)\n",
		r.Net.TokensSaved, r.Cost.PromptTokensPerTurn, r.Cost.CompletionTokensPerTurn)
	fmt.Fprintf(w, "  dollars saved               : $%.5f\n", r.Net.DollarsSaved)
	fmt.Fprintf(w, "  latency saved               : %.2f s  (@ %.0f ms/turn; 1-shot serve p50 = %d ns)\n\n",
		r.Net.LatencySavedMs/1000, r.Cost.ModelTurnLatencyMs, r.LocalServeNs)

	fmt.Fprintln(w, "-- vDSO ablation (REAL ON/OFF path swap) --")
	fmt.Fprintf(w, "  turns saved  vdso ON  : %d\n", r.Net.TurnsSaved)
	fmt.Fprintf(w, "  turns saved  vdso OFF : %d\n", r.VDSOOffNet.TurnsSaved)
	fmt.Fprintf(w, "  vdso lever contribution: %d turns  (== VDSOHits %d)\n\n",
		r.Net.TurnsSaved-r.VDSOOffNet.TurnsSaved, r.Counters.VDSOHits)

	fmt.Fprintln(w, "-- safety floor (deterministic moat, NOT a turn count) --")
	fmt.Fprintf(w, "  injections admitted   baseline=%d  fak=%d\n",
		r.Safety.InjectionsAdmittedBaseline, r.Safety.InjectionsAdmittedFak)
	fmt.Fprintf(w, "  destructive executed  baseline=%d  fak=%d\n\n",
		r.Safety.DestructiveExecutedBaseline, r.Safety.DestructiveExecutedFak)

	fmt.Fprintln(w, "-- ablation levers --")
	for _, l := range r.Levers {
		fmt.Fprintf(w, "  %-14s %-9s turns=%-3d  %s\n", l.Name, "["+l.Axis+"]", l.TurnsSaved, l.Mechanism)
	}
	fmt.Fprintln(w)

	fmt.Fprintln(w, "-- cost sensitivity (turns fixed by the kernel; per-turn price varies) --")
	for _, s := range r.Sensitivity {
		fmt.Fprintf(w, "  %-30s tokens=%-7d  $%.5f  %.2fs\n",
			s.Scenario, s.TokensSaved, s.DollarsSaved, s.LatencySavedSec)
	}
}
