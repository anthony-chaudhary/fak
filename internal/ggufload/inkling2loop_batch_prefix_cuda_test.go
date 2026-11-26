//go:build cuda && cgo

package ggufload

import (
	"fmt"
	"os"
	"slices"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// cohortRun is one cohort's greedy tokens, results and wall time.
type cohortRun struct {
	tokens  [][]int
	results []Inkling2LoopBatchResult
}

func runCohort(t *testing.T, bt *Inkling2LoopCohorts, prompts [][]int, n int) cohortRun {
	t.Helper()
	got := make([][]int, len(prompts))
	reqs := make([]Inkling2LoopBatchRequest, len(prompts))
	for i, p := range prompts {
		i := i
		reqs[i] = Inkling2LoopBatchRequest{Prompt: p, MaxTokens: n, Emit: func(tok int) bool { got[i] = append(got[i], tok); return true }}
	}
	res, err := bt.Generate(reqs)
	if err != nil {
		t.Fatalf("cohort of %d: %v", len(prompts), err)
	}
	return cohortRun{tokens: got, results: res}
}

func catTokens(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// Restoring a cohort row from the cross-request store gives the greedy tokens the
// same cohorts give with the store off, on the 1946-token prompt whose local rings
// wrap: a next turn restores its conversation's whole history, a second
// conversation on the same long prefix is checkpointed at its branch point, and a
// third, a one-row cohort, restores that branch point into rows of another stride
// than it was saved from.
func TestInkling2CUDABatchPrefixStoreParity(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	t.Setenv("FAK_INKLING2_COHORT_SHARED_PREFIX", "0")
	var long, capitals, fib []int
	for _, c := range inkling2OracleCases {
		switch c.name {
		case "passphrase-recall-1946":
			long = c.prompt
		case "capitals":
			capitals = c.prompt
		case "fibonacci":
			fib = c.prompt
		}
	}
	if long == nil || capitals == nil || fib == nil {
		t.Fatal("oracle cases missing")
	}
	const n = 16
	// three long rows of f32 KV do not fit a 16 GB card beside the model, so the
	// cohorts are two rows wide, then one; the store-off arm cannot run a lone request
	// as a cohort, so its last cohort pairs the row with a throwaway one
	cohorts := func(bt *Inkling2LoopCohorts, last ...[]int) []cohortRun {
		a := runCohort(t, bt, [][]int{long, capitals}, n)
		turn2 := catTokens(long, a.tokens[0])
		b := runCohort(t, bt, [][]int{turn2, catTokens(long, capitals)}, n)
		c := runCohort(t, bt, append([][]int{catTokens(long, fib)}, last...), n)
		return []cohortRun{a, b, c}
	}
	off := NewInkling2LoopCohorts(m, be)
	defer off.Close()
	ref := cohorts(off, fib)
	off.Close()
	on := NewInkling2LoopCohorts(m, be)
	defer on.Close()
	on.store = &inkling2PrefixStore{capBytes: 4 << 30, minTokens: inkling2PrefixMinDefault}
	got := cohorts(on)
	t.Logf("store %+v", on.PrefixStats())
	on.Close()
	prompts := [][][]int{{long, capitals}, {catTokens(long, got[0].tokens[0]), catTokens(long, capitals)}, {catTokens(long, fib)}}
	wantCached := [][]int{{0, 0}, {len(long) + n - 1, 0}, {len(long)}}
	var rows []prefixRow
	for ci := range got {
		for i := range got[ci].tokens {
			r, g := ref[ci].results[i], got[ci].results[i]
			t.Logf("cohort %d row %d: cached %d, ttft off %s on %s", ci, i, g.Restored, r.Stats.Prefill, g.Stats.Prefill)
			row := prefixRow{name: fmt.Sprintf("cohort %d row %d", ci, i), prompt: prompts[ci][i], on: got[ci].tokens[i]}
			if ci != 1 || slices.Equal(got[0].tokens[0], ref[0].tokens[0]) {
				row.off = ref[ci].tokens[i]
			}
			rows = append(rows, row)
			if g.Restored != wantCached[ci][i] {
				t.Errorf("cohort %d row %d: %d cached prompt tokens, want %d", ci, i, g.Restored, wantCached[ci][i])
			}
		}
	}
	adjudicatePrefixRows(t, m, be, rows, n)
}

// prefixRow is one store-on row: its prompt and greedy tokens, and the store-off
// arm's tokens for the same prompt when that arm fed it the same prompt.
type prefixRow struct {
	name    string
	prompt  []int
	off, on []int
}

// adjudicatePrefixRows checks every store-on row against the single-sequence
// session, as the cohort-prefill parity test checks cohort rows: a row that leaves
// the scalar tokens other than at a near tie fails, and so do more near ties than
// that test allows per eight rows. The store-off cohort is logged beside it but is no
// oracle: its own chunked prefill can leave scalar at a near tie.
func adjudicatePrefixRows(t *testing.T, m *Inkling2LoopModel, be compute.Backend, rows []prefixRow, n int) {
	t.Helper()
	ties := 0
	for _, r := range rows {
		want, runnerUp, margin, err := inkling2ResidentGreedyMargins(m, be, r.prompt, n)
		if err != nil {
			t.Fatal(err)
		}
		onAt, onTie, onWhy := inkling2CohortDivergence(want, runnerUp, margin, r.on)
		off := "not fed this prompt"
		if r.off != nil {
			offAt, _, offWhy := inkling2CohortDivergence(want, runnerUp, margin, r.off)
			off = fmt.Sprintf("at %d %s", offAt, offWhy)
		}
		switch {
		case onAt < 0:
			t.Logf("PREFIX_VERDICT %s PASS (%d prompt tokens); off leaves scalar %s", r.name, len(r.prompt), off)
		case onTie:
			ties++
			t.Logf("PREFIX_VERDICT %s NEAR_TIE (%d prompt tokens) %s; off leaves scalar %s", r.name, len(r.prompt), onWhy, off)
		default:
			t.Errorf("PREFIX_VERDICT %s FAIL (%d prompt tokens): %s; off leaves scalar %s", r.name, len(r.prompt), onWhy, off)
		}
	}
	if allowed := inkling2OracleMaxNearTies * ((len(rows) + 7) / 8); ties > allowed {
		t.Errorf("%d of %d store-on rows are near ties, more than %d: the store drifts from scalar, not ties", ties, len(rows), allowed)
	}
}

// FAK_INKLING2_PREFIX_TRACE=1 runs a shared-system-prompt multi-turn trace: C
// conversations (FAK_INKLING2_PREFIX_TRACE_C, default 4) open on the 1946-token
// prompt as their system prompt, each with its own first user message, and run
// FAK_INKLING2_PREFIX_TRACE_TURNS turns (default 3), each turn appending the turn's
// own reply and a next user message; every turn is one cohort of the C rows. It runs
// the trace with the store off and on and logs, per arm, the prompt tokens prefilled,
// the first-token latency p50/p95 over every row of every turn, and whether every
// conversation's greedy tokens are the same in both arms.
func TestInkling2CUDABatchPrefixTraceBench(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_PREFIX_TRACE")
	defer m.Close()
	C, turns := prefixEnvInt("FAK_INKLING2_PREFIX_TRACE_C", 4), prefixEnvInt("FAK_INKLING2_PREFIX_TRACE_TURNS", 3)
	n := prefixEnvInt("FAK_INKLING2_PREFIX_TRACE_TOKENS", 16)
	var sys []int
	var users [][]int
	for _, c := range inkling2OracleCases {
		if c.name == "passphrase-recall-1946" {
			sys = c.prompt
		} else if len(c.prompt) < 256 {
			users = append(users, c.prompt)
		}
	}
	if sys == nil || len(users) == 0 {
		t.Fatal("oracle cases missing")
	}
	user := func(conv, turn int) []int {
		u := users[(conv+turn)%len(users)]
		// a per-conversation tail keeps every conversation's prompt its own
		return catTokens(u, []int{1000 + conv, 2000 + turn})
	}
	type arm struct {
		prefilled, cached int
		ttft              []time.Duration
		gens, prompts     [][]int
	}
	trace := func(bt *Inkling2LoopCohorts) arm {
		var a arm
		hist := make([][]int, C)
		for c := range hist {
			hist[c] = catTokens(sys, user(c, 0))
		}
		for turn := 0; turn < turns; turn++ {
			r := runCohort(t, bt, hist, n)
			for c := range hist {
				a.prefilled += len(hist[c]) - r.results[c].Restored
				a.cached += r.results[c].Restored
				a.ttft = append(a.ttft, r.results[c].Stats.Prefill)
				a.gens = append(a.gens, r.tokens[c])
				a.prompts = append(a.prompts, hist[c])
				hist[c] = catTokens(hist[c], r.tokens[c], user(c, turn+1))
			}
		}
		return a
	}
	pct := func(d []time.Duration, q float64) time.Duration {
		s := slices.Clone(d)
		sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
		return s[min(len(s)-1, int(q*float64(len(s))))]
	}
	off := NewInkling2LoopCohorts(m, be)
	defer off.Close()
	trace(off) // warm the binding and the allocator
	ao := trace(off)
	off.Close()
	on := NewInkling2LoopCohorts(m, be)
	defer on.Close()
	mib := prefixEnvInt("FAK_INKLING2_BATCH_PREFIX_CACHE_MIB", 4096)
	on.store = &inkling2PrefixStore{capBytes: int64(mib) << 20, minTokens: inkling2PrefixMinDefault}
	an := trace(on)
	stats := on.PrefixStats()
	on.Close()
	parity := true
	rows := make([]prefixRow, len(an.gens))
	for i := range an.gens {
		parity = parity && slices.Equal(ao.gens[i], an.gens[i])
		rows[i] = prefixRow{name: fmt.Sprintf("turn %d conversation %d", i/C, i%C), prompt: an.prompts[i], on: an.gens[i]}
		// each arm feeds its own reply to the next turn
		if slices.Equal(ao.prompts[i], an.prompts[i]) {
			rows[i].off = ao.gens[i]
		}
	}
	for _, x := range []struct {
		name string
		a    arm
	}{{"off", ao}, {"on", an}} {
		t.Logf("inkling2 prefix trace: arm=%s C=%d turns=%d sys=%d prefilled=%d cached=%d ttft_p50=%s ttft_p95=%s",
			x.name, C, turns, len(sys), x.a.prefilled, x.a.cached, pct(x.a.ttft, 0.5), pct(x.a.ttft, 0.95))
	}
	t.Logf("inkling2 prefix trace: greedy parity=%v store=%+v", parity, stats)
	fmt.Fprintf(os.Stderr, "PREFIX_TRACE parity=%v\n", parity)
	adjudicatePrefixRows(t, m, be, rows, n)
}

func prefixEnvInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}

// FAK_INKLING2_PREFIX_DIAG=1 runs one prompt with the store off beside companions of
// different lengths, which moves its chunk boundaries, and logs whether its greedy
// tokens change: a reference for how exact store-on parity can be.
func TestInkling2CUDABatchPrefixChunkBoundaryDiag(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_PREFIX_DIAG")
	defer m.Close()
	t.Setenv("FAK_INKLING2_COHORT_SHARED_PREFIX", "0")
	var long, capitals, fib []int
	for _, c := range inkling2OracleCases {
		switch c.name {
		case "passphrase-recall-1946":
			long = c.prompt
		case "capitals":
			capitals = c.prompt
		case "fibonacci":
			fib = c.prompt
		}
	}
	target := catTokens(long, capitals)
	bt := NewInkling2LoopCohorts(m, be)
	defer bt.Close()
	var ref []int
	for _, cmp := range []struct {
		name  string
		first bool
		other []int
	}{
		{"first,capitals", true, capitals},
		{"first,fib", true, fib},
		{"second,capitals", false, capitals},
		{"second,long+fib", false, catTokens(long, fib)},
		{"second,long", false, long},
	} {
		prompts := [][]int{target, cmp.other}
		row := 0
		if !cmp.first {
			prompts, row = [][]int{cmp.other, target}, 1
		}
		r := runCohort(t, bt, prompts, 16)
		if ref == nil {
			ref = r.tokens[row]
		}
		t.Logf("chunk diag %s: same=%v tokens %v", cmp.name, slices.Equal(ref, r.tokens[row]), r.tokens[row])
	}
}
