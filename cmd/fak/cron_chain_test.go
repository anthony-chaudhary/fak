package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// witnessMarker is a distinctive payload job A emits so job B's assembled prompt
// can be proven to carry A's exact witnessed bytes. Substring-matched, so shell
// echo quirks (a trailing CRLF on Windows vs LF on POSIX) do not matter.
const witnessMarker = "WITNESS-A-PAYLOAD-42"

// TestCronPromptChainConsumesWitnessedOutput is the two-job chain contract: job A
// runs a --script whose stdout is recorded OBSERVED, then job B --context-from A
// provably consumes A's witnessed bytes AND the A→B handoff is a ledger edge.
func TestCronPromptChainConsumesWitnessedOutput(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "chain.jsonl")
	const jobA, jobB = "collect-a", "summarize-b"
	const slotA, slotB = "2026-07-10T00:00:00Z", "2026-07-10T01:00:00Z"

	// Job A: a pre-run script emits the marker. `echo` exists in both `sh` (POSIX)
	// and `cmd` (Windows), so this one script is portable across both hosts.
	var outA, errA bytes.Buffer
	codeA := runCron(&outA, &errA, []string{
		"prompt", "--job", jobA, "--ledger", ledger, "--slot", slotA,
		"--script", "echo " + witnessMarker,
	})
	if codeA != 0 {
		t.Fatalf("job A: exit %d, want 0; stderr=%q", codeA, errA.String())
	}

	// A's output must be recorded OBSERVED and carry the witnessed bytes.
	outs, err := cronReadOutputs(ledger)
	if err != nil {
		t.Fatalf("read outputs: %v", err)
	}
	var recA *cronOutputRecord
	for i := range outs {
		if outs[i].Job == jobA {
			recA = &outs[i]
		}
	}
	if recA == nil {
		t.Fatalf("no witnessed output row for job %q; ledger outputs=%+v", jobA, outs)
	}
	if recA.Provenance != cronProvObserved {
		t.Fatalf("job A output provenance = %q, want %q (OBSERVED, not authored)", recA.Provenance, cronProvObserved)
	}
	if !strings.Contains(recA.Text, witnessMarker) {
		t.Fatalf("job A witnessed text %q does not contain marker %q", recA.Text, witnessMarker)
	}

	// Job B: chain from A. Its assembled prompt must carry A's witnessed bytes,
	// the OBSERVED provenance label, and B's own authored base prompt.
	var outB, errB bytes.Buffer
	codeB := runCron(&outB, &errB, []string{
		"prompt", "--job", jobB, "--ledger", ledger, "--slot", slotB,
		"--context-from", jobA, "--base", "summarize the input",
	})
	if codeB != 0 {
		t.Fatalf("job B: exit %d, want 0; stderr=%q", codeB, errB.String())
	}
	got := outB.String()
	if !strings.Contains(got, witnessMarker) {
		t.Fatalf("job B prompt does not consume A's witnessed output %q:\n%s", witnessMarker, got)
	}
	if !strings.Contains(got, "OBSERVED") {
		t.Fatalf("job B prompt is missing the OBSERVED provenance label:\n%s", got)
	}
	if !strings.Contains(got, "summarize the input") {
		t.Fatalf("job B prompt is missing its authored base:\n%s", got)
	}

	// The A→B handoff must be a recorded ledger edge, not a silent pass-through.
	edges, err := cronReadEdges(ledger)
	if err != nil {
		t.Fatalf("read edges: %v", err)
	}
	found := false
	for _, e := range edges {
		if e.FromJob == jobA && e.ToJob == jobB && e.FromSlot == slotA && e.ToSlot == slotB {
			found = true
		}
	}
	if !found {
		t.Fatalf("no A→B ledger edge {from:%s->to:%s}; edges=%+v", jobA, jobB, edges)
	}
}

// TestCronPromptRefusesUnwitnessedContext fails closed: chaining from a job that
// never recorded a witnessed output is exit 2, not an empty injection — B may
// consume only what A provably produced.
func TestCronPromptRefusesUnwitnessedContext(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "chain.jsonl")
	var out, errb bytes.Buffer
	code := runCron(&out, &errb, []string{
		"prompt", "--job", "b", "--ledger", ledger, "--context-from", "never-ran",
	})
	if code != 2 {
		t.Fatalf("unwitnessed context: exit %d, want 2; stderr=%q", code, errb.String())
	}
	// No edge may be recorded when the chain is refused.
	edges, err := cronReadEdges(ledger)
	if err != nil {
		t.Fatalf("read edges: %v", err)
	}
	if len(edges) != 0 {
		t.Fatalf("refused chain still recorded %d edge(s): %+v", len(edges), edges)
	}
}

// TestCronChainReadsEdgesBack is the readback contract: after a two-job chain
// records an A→B edge, `fak cron chain` surfaces that same edge as operator-readable
// evidence — in the text table, as JSON, and under a --job filter. This closes the
// loop the audit goal needs: prompt WRITES the edge, chain makes it queryable, so a
// chained pipeline is traversable after the fact rather than write-only.
func TestCronChainReadsEdgesBack(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "chain.jsonl")
	const jobA, jobB = "collect-a", "summarize-b"
	const slotA, slotB = "2026-07-10T00:00:00Z", "2026-07-10T01:00:00Z"

	// Build the A→B edge exactly the way the chain contract does.
	var sink bytes.Buffer
	if code := runCron(&sink, &sink, []string{
		"prompt", "--job", jobA, "--ledger", ledger, "--slot", slotA,
		"--script", "echo " + witnessMarker,
	}); code != 0 {
		t.Fatalf("job A: exit %d; out=%q", code, sink.String())
	}
	sink.Reset()
	if code := runCron(&sink, &sink, []string{
		"prompt", "--job", jobB, "--ledger", ledger, "--slot", slotB,
		"--context-from", jobA, "--base", "summarize the input",
	}); code != 0 {
		t.Fatalf("job B: exit %d; out=%q", code, sink.String())
	}

	// Text readout: the A→B handoff must be visible with both endpoints and the arrow.
	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, []string{"chain", "--ledger", ledger}); code != 0 {
		t.Fatalf("cron chain: exit %d, want 0; stderr=%q", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{jobA, slotA, jobB, slotB, "->"} {
		if !strings.Contains(got, want) {
			t.Fatalf("chain readout missing %q:\n%s", want, got)
		}
	}

	// JSON readout: must decode into the same edge, not prose.
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, []string{"chain", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("cron chain --json: exit %d, want 0; stderr=%q", code, errb.String())
	}
	var edges []cronEdgeRecord
	if err := json.Unmarshal(out.Bytes(), &edges); err != nil {
		t.Fatalf("chain --json is not valid JSON: %v\n%s", err, out.String())
	}
	if len(edges) != 1 || edges[0].FromJob != jobA || edges[0].ToJob != jobB {
		t.Fatalf("chain --json edges = %+v, want one %s->%s edge", edges, jobA, jobB)
	}

	// --job filter keeps the edge that touches the named job and drops the rest.
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, []string{"chain", "--ledger", ledger, "--job", jobB}); code != 0 {
		t.Fatalf("cron chain --job: exit %d, want 0; stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), jobB) {
		t.Fatalf("chain --job %s dropped the touching edge:\n%s", jobB, out.String())
	}
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, []string{"chain", "--ledger", ledger, "--job", "unrelated"}); code != 0 {
		t.Fatalf("cron chain --job unrelated: exit %d, want 0", code)
	}
	if !strings.Contains(out.String(), "no cron chain edges recorded") {
		t.Fatalf("chain --job unrelated should report an empty chain:\n%s", out.String())
	}
}

// TestCronChainEmptyAndUsage: an empty ledger is a valid empty chain (exit 0), and
// a missing --ledger is a usage error (exit 2).
func TestCronChainEmptyAndUsage(t *testing.T) {
	empty := filepath.Join(t.TempDir(), "none.jsonl")
	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, []string{"chain", "--ledger", empty}); code != 0 {
		t.Fatalf("empty ledger: exit %d, want 0; stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "no cron chain edges recorded") {
		t.Fatalf("empty ledger readout = %q, want empty-chain notice", out.String())
	}
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, []string{"chain"}); code != 2 {
		t.Fatalf("chain without --ledger: exit %d, want 2", code)
	}
}

// TestCronPromptRequiresJobAndLedger rejects missing required flags with exit 2.
func TestCronPromptRequiresJobAndLedger(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCron(&out, &errb, []string{"prompt", "--ledger", "x.jsonl"}); code != 2 {
		t.Fatalf("prompt without --job: exit %d, want 2", code)
	}
	out.Reset()
	errb.Reset()
	if code := runCron(&out, &errb, []string{"prompt", "--job", "j"}); code != 2 {
		t.Fatalf("prompt without --ledger: exit %d, want 2", code)
	}
}
