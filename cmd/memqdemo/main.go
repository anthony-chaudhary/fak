// Command memqdemo is a no-model, deterministic walkthrough of the memq memory-
// operation algebra: the substrate that lets an agent author its OWN memory strategy
// (render / clean / compact / dream / a novel one) instead of the kernel hardcoding a
// single compaction path. "Build SQL, not a specific query."
//
// It runs entirely offline over the in-memory demo corpus and prints the story:
//
//  1. the corpus — the two axes (durable preferences vs ephemeral observations);
//  2. the drivers — five strategies, each a composable query (not a function);
//  3. EXPLAIN — step through a plan before running it;
//  4. render — what an authored retrieve actually pulls into context;
//  5. the fail-closed effect default — a clean proposes, does not apply, without caps;
//  6. compact — fold unreferenced cells into a derived disposition artifact;
//  7. a NOVEL agent-authored query the kernel never anticipated;
//  8. the safety floor — render EVERYTHING and watch the sealed poison get refused.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/memq"
)

func main() {
	if err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "memqdemo:", err)
		os.Exit(1)
	}
}

func run(out io.Writer) error {
	ctx := context.Background()
	w := func(format string, a ...any) { fmt.Fprintf(out, format, a...) }

	report := map[string]any{
		"demo": "memq: an agent authoring its own memory strategy (build SQL, not a specific query)",
	}
	w("%s\n\n", report["demo"])

	// 1. The corpus.
	store := memq.NewDemoStore()
	cells, _ := store.Cells(ctx)
	w("== 1. the memory (the in-memory demo corpus) ==\n")
	w("two axes from CONTEXT-IS-NOT-MEMORY.md: durable standing facts vs ephemeral observations.\n")
	for _, c := range cells {
		tag := "    "
		if c.Sealed {
			tag = "SEAL"
		}
		w("  %s %-9s %-8s %s\n", tag, c.Durability, c.Role, c.Descriptor)
	}
	report["cells"] = len(cells)

	// 2. The drivers.
	w("\n== 2. the strategies (each a composable query, not a hardcoded function) ==\n")
	names := []string{}
	for _, d := range memq.Drivers() {
		w("  %-9s %s\n", d.Name, d.Doc)
		names = append(names, d.Name)
	}
	report["drivers"] = names

	// 3. EXPLAIN — step through the compact plan before running it.
	w("\n== 3. EXPLAIN compact (the plan, before anything runs) ==\n")
	compact, _ := memq.Get("compact")
	plan := memq.Explain(compact.Build(memq.Params{}))
	w("%s", plan.Text())

	// 4. render — an authored retrieve.
	w("\n== 4. render (what an authored retrieve pulls into context for an intent) ==\n")
	render, _ := memq.Get("render")
	rr, err := memq.Run(ctx, store, render.Build(memq.Params{Intent: "refund fee", Budget: 600}), memq.Caps{})
	if err != nil {
		return err
	}
	for _, it := range rr.Rendered {
		w("  + %s\n", it.Descriptor)
	}
	w("  (%d item(s), ~%d tokens; sealed/irrelevant cells excluded)\n", rr.Stats.Rendered, rr.Stats.EstimatedTokens)

	// 5. fail-closed effect default — clean proposes, does not apply, without caps.
	w("\n== 5. the fail-closed effect default (clean WITHOUT caps proposes, mutates nothing) ==\n")
	clean, _ := memq.Get("clean")
	dry, err := memq.Run(ctx, memq.NewDemoStore(), clean.Build(memq.Params{}), memq.Caps{})
	if err != nil {
		return err
	}
	for _, e := range dry.Effects {
		w("  %s: applied=%v  (%s)\n", e.Kind, e.Applied, e.Note)
	}
	report["clean_applied_without_caps"] = dry.Stats.EffectsApplied

	// 6. compact --apply — the derived disposition artifact.
	w("\n== 6. compact WITH caps (fold unreferenced cells into a derived disposition) ==\n")
	cstore := memq.NewDemoStore()
	cr, err := memq.Run(ctx, cstore, compact.Build(memq.Params{}), memq.AllowAll())
	if err != nil {
		return err
	}
	for _, e := range cr.Effects {
		if e.Derived != nil {
			w("  consolidated %d source(s) -> %s (%s, %d bytes)\n", len(e.Cells), e.Derived.ID, e.Derived.Durability, e.Derived.Bytes)
			w("    %s\n", e.Derived.Descriptor)
		} else {
			w("  %s: applied=%v\n", e.Kind, e.Applied)
		}
	}

	// 7. a NOVEL agent-authored query the kernel never anticipated.
	w("\n== 7. a NOVEL query the agent authored (no built-in driver expresses it) ==\n")
	w("   \"render only my durable standing preferences, newest first\"\n")
	novel := memq.Query{
		Intent: "standing preferences",
		Ops: []memq.Op{
			{Kind: memq.OpScan},
			{Kind: memq.OpFilter, Pred: &memq.Pred{Op: memq.PredEq, Field: "durability", Value: memq.DurabilityDurable}},
			{Kind: memq.OpRank, By: memq.RankStep, Desc: true},
			{Kind: memq.OpRender},
		},
	}
	nr, err := memq.Run(ctx, store, novel, memq.Caps{})
	if err != nil {
		return err
	}
	for _, it := range nr.Rendered {
		w("  + %s\n", it.Descriptor)
	}
	report["novel_rendered"] = nr.Stats.Rendered

	// 8. the safety floor — render EVERYTHING; the sealed poison is refused.
	w("\n== 8. the safety floor (render EVERYTHING — the sealed poison is refused) ==\n")
	everything := memq.Query{Ops: []memq.Op{{Kind: memq.OpScan}, {Kind: memq.OpRender}}}
	er, err := memq.Run(ctx, store, everything, memq.Caps{})
	if err != nil {
		return err
	}
	w("  rendered %d benign cell(s); refused %d:\n", er.Stats.Rendered, er.Stats.Refused)
	for _, rf := range er.Refused {
		w("    REFUSE %-9s %s\n", rf.ID, rf.Reason)
	}
	poisonLeaked := false
	for _, it := range er.Rendered {
		if it.Role == "read_webpage" {
			poisonLeaked = true
		}
	}
	report["poison_leaked"] = poisonLeaked
	report["everything_refused"] = er.Stats.Refused

	w("\nwitness: an agent composed retrieve/clean/compact AND a novel query over one\n")
	w("algebra; effects defaulted to proposed-not-applied; the sealed span was refused\n")
	w("from every render — the substrate is the origin, the strategies are the queries.\n")

	b, _ := json.MarshalIndent(report, "", "  ")
	_ = os.WriteFile("memqdemo-report.json", b, 0o644)
	w("\nreport written: memqdemo-report.json\n")
	if poisonLeaked {
		return fmt.Errorf("INVARIANT VIOLATED: a sealed cell was rendered into context")
	}
	return nil
}
