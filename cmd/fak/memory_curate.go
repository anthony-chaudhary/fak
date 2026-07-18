package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memq"
)

// curateEnvelope is the `fak memory curate --json` machine surface (#3908 DoD 4
// via #5110): the typed CurateReport verdict, the regret witness, and — when
// anything was selected for eviction — the (proposed or applied) tombstone
// effect. The text mode renders memq.CurateReportText over the same data.
type curateEnvelope struct {
	Store  string            `json:"store"`
	Report memq.CurateReport `json:"report"`
	Regret memq.RegretReport `json:"regret"`
	Effect *memq.Effect      `json:"effect,omitempty"`
}

// runMemoryCurate is `fak memory curate` (#5110): the thin CLI shell over
// budget-curated forgetting (#3908). It loads a memq backend through the same
// seam as `fak memory recall` (--store names a markdown memory store; --dir a
// recall core image; neither, the in-memory demo corpus), runs memq.BudgetCurate
// under the --budget byte cap, and prints memq.CurateReportText — the byte
// budget, the evicted set, and the running regret rate. Ranking rides each
// cell's persisted witnessed value (Attrs[memq.ValueAttr] — realized recall
// value, never size); an un-witnessed cell fails closed to 0 and is evicted
// first. The tombstone eviction is applied via memq.ApplyCurate ONLY under
// --apply; without it (or against a backend with no Tombstoner, like the
// read-only notes store) the effect stays a proposal — fail-closed.
func runMemoryCurate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("memory curate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	budget := fs.Int64("budget", 0, "hard byte cap the surviving store must fit (required, positive)")
	dir := fs.String("dir", "", "recall core image directory (default: the in-memory demo corpus)")
	store := fs.String("store", "", "markdown memory store dir (the `fak memory recall` seam; overrides --dir)")
	needed := fs.String("needed", "", "comma-separated cell IDs a LATER recall needed — the regret witness (#3908 DoD 3)")
	apply := fs.Bool("apply", false, "APPLY the tombstone evictions (default: propose only — fail-closed)")
	asJSON := fs.Bool("json", false, "emit the CurateReport + RegretReport envelope as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *budget <= 0 {
		fmt.Fprintln(stderr, "fak memory curate: --budget N (a positive byte cap) is required")
		return 2
	}

	var backend memq.Backend
	var blabel string
	if *store != "" {
		backend, blabel = notesMemoryBackend(*store)
	} else {
		backend, blabel = memoryBackend(*dir)
	}
	c := ctx()
	cells, err := backend.Cells(c)
	if err != nil {
		fmt.Fprintf(stderr, "fak memory curate: %v\n", err)
		return 1
	}

	// nil value map: BudgetCurate falls back to the persisted Attrs[memq.ValueAttr]
	// per cell — the witnessed-value signal, failing closed to 0 when un-witnessed.
	rep := memq.BudgetCurate(cells, *budget, nil)
	reg := memq.CurateRegret(rep, splitCommaIDs(*needed))

	caps := memq.Caps{}
	if *apply {
		caps = memq.AllowAll()
	}
	var effect *memq.Effect
	if len(rep.Evicted) > 0 {
		e := memq.ApplyCurate(c, backend, rep, caps)
		effect = &e
	}

	if *asJSON {
		fmt.Fprintln(stdout, string(jsonIndent(curateEnvelope{Store: blabel, Report: rep, Regret: reg, Effect: effect})))
		return 0
	}
	fmt.Fprintf(stdout, "== fak memory curate (%s) ==\n", blabel)
	fmt.Fprint(stdout, memq.CurateReportText(rep, &reg))
	if effect != nil {
		state := "PROPOSED"
		if effect.Applied {
			state = "APPLIED"
		}
		note := ""
		if effect.Note != "" {
			note = "  — " + effect.Note
		}
		fmt.Fprintf(stdout, "effect %s [%s] %d cell(s)%s\n", effect.Kind, state, len(effect.Cells), note)
	}
	return 0
}

// splitCommaIDs parses a comma-separated ID list, dropping empties — the
// --needed regret-witness input.
func splitCommaIDs(raw string) []string {
	var out []string
	for _, tok := range strings.Split(raw, ",") {
		if tok = strings.TrimSpace(tok); tok != "" {
			out = append(out, tok)
		}
	}
	return out
}
