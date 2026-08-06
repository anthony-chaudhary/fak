package workflow

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// twoStep is the fixture shape the resume tests share: a pure step feeding an effectful
// one, so both evidence rungs are exercised on one graph.
func twoStep(t *testing.T) *Graph {
	t.Helper()
	g, err := Compile(Spec{Name: "ship", Tasks: []TaskSpec{
		{ID: "build", Op: "emit", Payload: "artifact"},
		{ID: "land", Op: "emit", Payload: "commit", Needs: []string{"build"}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	return g
}

// journalFor writes the rows a fully-completed run of g would have left, so a resume test
// starts from a journal that narrates every step as done.
func journalFor(g *Graph, epoch string, outputs map[string]string, kinds map[string]StepKind, claims map[string]string) []Entry {
	hashes := map[string]string{}
	var rows []Entry
	for _, n := range g.Nodes {
		out := outputs[n.ID]
		h := HashOutput(out)
		rows = append(rows, Entry{
			Schema: JournalSchema, Run: g.Name, Step: n.ID, Kind: kinds[n.ID],
			InputsHash: StepInputsHash(n, hashes), EpochHash: epoch,
			OutputHash: h, Output: out, Claim: claims[n.ID], TSMS: 1000,
		})
		hashes[n.ID] = h
	}
	return rows
}

func mustFold(t *testing.T, rows []Entry, nowMS int64) State {
	t.Helper()
	st, err := Fold(rows, nowMS)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	return st
}

// A fold is pure: the same rows and the same injected clock produce byte-identical state,
// and a re-execution's later row supersedes the earlier one regardless of arrival order.
func TestFoldDeterministicAndLastRowWins(t *testing.T) {
	rows := []Entry{
		{Schema: JournalSchema, Run: "r", Step: "b", Kind: StepPure, OutputHash: "h2", TSMS: 2},
		{Schema: JournalSchema, Run: "r", Step: "a", Kind: StepPure, OutputHash: "h0", TSMS: 1},
		{Schema: JournalSchema, Run: "r", Step: "a", Kind: StepPure, OutputHash: "h1", TSMS: 3},
	}
	first, err := json.Marshal(mustFold(t, rows, 42))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	second, err := json.Marshal(mustFold(t, rows, 42))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(first) != string(second) {
		t.Fatalf("fold is not deterministic:\n%s\n%s", first, second)
	}
	st := mustFold(t, rows, 42)
	if got := strings.Join(st.Order, ","); got != "a,b" {
		t.Fatalf("step order = %q, want sorted a,b", got)
	}
	if st.Steps["a"].OutputHash != "h1" {
		t.Fatalf("last row must win, got %q", st.Steps["a"].OutputHash)
	}
	if st.FoldedMS != 42 || st.Rows != 3 {
		t.Fatalf("fold state = %+v, want the injected clock and row count", st)
	}
}

// The fold fails closed at the boundary: a foreign schema, an empty step, or an unknown
// kind refuses the journal rather than reading as a cache line.
func TestFoldRefusesMalformedRows(t *testing.T) {
	for name, row := range map[string]Entry{
		"schema": {Schema: "other/2", Step: "a", Kind: StepPure},
		"step":   {Schema: JournalSchema, Step: "  ", Kind: StepPure},
		"kind":   {Schema: JournalSchema, Step: "a", Kind: "narrated"},
	} {
		if _, err := Fold([]Entry{row}, 1); err == nil {
			t.Fatalf("%s: fold accepted a malformed row", name)
		}
	}
}

func TestReadJournalRejectsUnknownField(t *testing.T) {
	rows, err := ReadJournal(strings.NewReader(`{"schema":"` + JournalSchema + `","step":"a","kind":"pure","ts_ms":1}` + "\n"))
	if err != nil || len(rows) != 1 || rows[0].Step != "a" {
		t.Fatalf("read a valid row: rows=%v err=%v", rows, err)
	}
	if _, err := ReadJournal(strings.NewReader(`{"step":"a","kind":"pure","surprise":1}`)); err == nil {
		t.Fatal("read accepted an unknown field")
	}
}

// A pure step re-derives its own evidence; an effectful step is skipped only when the
// injected oracle corroborates its claim, and the report names which source answered.
func TestResumeCorroboratedEffectfulStepIsSkipped(t *testing.T) {
	g := twoStep(t)
	epoch := GraphEpoch(g, "e1")
	rows := journalFor(g, epoch,
		map[string]string{"build": "artifact-out", "land": "9f3c1a2"},
		map[string]StepKind{"build": StepPure, "land": StepEffectful},
		map[string]string{"land": "ancestor:9f3c1a2"})
	st := mustFold(t, rows, 7)

	asked := ""
	r := Resume(context.Background(), g, st, epoch, func(_ context.Context, step, claim string) (string, bool) {
		asked = step + "/" + claim
		return "dos_verify:" + claim, true
	})
	if r.Skips != 2 || r.Reruns != 0 {
		t.Fatalf("skips=%d reruns=%d, want 2/0 (%+v)", r.Skips, r.Reruns, r.Steps)
	}
	if asked != "land/ancestor:9f3c1a2" {
		t.Fatalf("the effectful step was not corroborated, asked=%q", asked)
	}
	if src := r.Steps[0].Source; !strings.HasPrefix(src, "journal-hash:") {
		t.Fatalf("pure step source = %q, want a journaled output hash", src)
	}
	if src := r.Steps[1].Source; src != "dos_verify:ancestor:9f3c1a2" {
		t.Fatalf("effectful step source = %q, want the ladder that answered", src)
	}
}

// The core refusal: the journal still narrates the step as done, but the claim no longer
// corroborates (its commit was reverted), so the step re-executes — and so does everything
// downstream of it, whose cached inputs can no longer be trusted.
func TestResumeReExecutesUncorroboratedClaim(t *testing.T) {
	g, err := Compile(Spec{Name: "ship", Tasks: []TaskSpec{
		{ID: "land", Op: "emit", Payload: "commit"},
		{ID: "announce", Op: "emit", Payload: "note", Needs: []string{"land"}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	epoch := GraphEpoch(g, "e1")
	rows := journalFor(g, epoch,
		map[string]string{"land": "9f3c1a2", "announce": "posted"},
		map[string]StepKind{"land": StepEffectful, "announce": StepPure},
		map[string]string{"land": "ancestor:9f3c1a2"})
	st := mustFold(t, rows, 7)

	r := Resume(context.Background(), g, st, epoch, func(context.Context, string, string) (string, bool) {
		return "", false // the reverted commit no longer corroborates
	})
	if r.Skips != 0 || r.Reruns != 2 {
		t.Fatalf("skips=%d reruns=%d, want 0/2 (%+v)", r.Skips, r.Reruns, r.Steps)
	}
	if r.Steps[0].Reason != ReasonClaimUnverified {
		t.Fatalf("land reason = %q, want %q", r.Steps[0].Reason, ReasonClaimUnverified)
	}
	if r.Steps[1].Reason != ReasonUpstreamRerun {
		t.Fatalf("announce reason = %q, want %q", r.Steps[1].Reason, ReasonUpstreamRerun)
	}
}

// A nil oracle is not permission to trust the journal.
func TestResumeFailsClosedWithoutOracle(t *testing.T) {
	g := twoStep(t)
	epoch := GraphEpoch(g, "e1")
	st := mustFold(t, journalFor(g, epoch,
		map[string]string{"build": "a", "land": "b"},
		map[string]StepKind{"build": StepPure, "land": StepEffectful},
		map[string]string{"land": "ancestor:9f3c1a2"}), 7)
	r := Resume(context.Background(), g, st, epoch, nil)
	if r.Steps[1].Disposition != DispExecute || r.Steps[1].Reason != ReasonClaimUnverified {
		t.Fatalf("effectful step without an oracle = %+v, want a re-execution", r.Steps[1])
	}
}

// Each cache-invalidating drift cites its own closed reason.
func TestResumeReasonVocabulary(t *testing.T) {
	g := twoStep(t)
	epoch := GraphEpoch(g, "e1")
	kinds := map[string]StepKind{"build": StepPure, "land": StepPure}
	outs := map[string]string{"build": "a", "land": "b"}

	always := func(context.Context, string, string) (string, bool) { return "reg", true }

	t.Run("unjournaled", func(t *testing.T) {
		r := Resume(context.Background(), g, mustFold(t, nil, 1), epoch, always)
		if r.Steps[0].Reason != ReasonUnjournaled {
			t.Fatalf("reason = %q", r.Steps[0].Reason)
		}
	})
	t.Run("epoch drift", func(t *testing.T) {
		st := mustFold(t, journalFor(g, GraphEpoch(g, "e0"), outs, kinds, nil), 1)
		r := Resume(context.Background(), g, st, epoch, always)
		if r.Steps[0].Reason != ReasonEpochDrift {
			t.Fatalf("reason = %q", r.Steps[0].Reason)
		}
	})
	t.Run("inputs drift", func(t *testing.T) {
		rows := journalFor(g, epoch, outs, kinds, nil)
		rows[0].InputsHash = "stale"
		r := Resume(context.Background(), g, mustFold(t, rows, 1), epoch, always)
		if r.Steps[0].Reason != ReasonInputsDrift {
			t.Fatalf("reason = %q", r.Steps[0].Reason)
		}
	})
	t.Run("output mismatch", func(t *testing.T) {
		rows := journalFor(g, epoch, outs, kinds, nil)
		rows[0].Output = "tampered"
		r := Resume(context.Background(), g, mustFold(t, rows, 1), epoch, always)
		if r.Steps[0].Reason != ReasonOutputMismatch {
			t.Fatalf("reason = %q", r.Steps[0].Reason)
		}
	})
	t.Run("claim missing", func(t *testing.T) {
		k := map[string]StepKind{"build": StepEffectful, "land": StepPure}
		st := mustFold(t, journalFor(g, epoch, outs, k, nil), 1)
		r := Resume(context.Background(), g, st, epoch, always)
		if r.Steps[0].Reason != ReasonClaimMissing {
			t.Fatalf("reason = %q", r.Steps[0].Reason)
		}
	})
}

// The runner half: a skipped step's journaled output is replayed to its dependents without
// the base runner ever seeing it, and an unwitnessed step really does reach the base.
func TestResumeRunnerReplaysSkipsAndExecutesReruns(t *testing.T) {
	g := twoStep(t)
	epoch := GraphEpoch(g, "e1")
	rows := journalFor(g, epoch,
		map[string]string{"build": "artifact-out", "land": "9f3c1a2"},
		map[string]StepKind{"build": StepPure, "land": StepEffectful},
		map[string]string{"land": "ancestor:9f3c1a2"})
	r := Resume(context.Background(), g, mustFold(t, rows, 7), epoch,
		func(context.Context, string, string) (string, bool) { return "", false })

	var saw []string
	base := RunnerFunc(func(_ context.Context, in RunInput) (string, error) {
		saw = append(saw, in.Node.ID)
		return "fresh:" + in.Deps["build"], nil
	})
	rr := NewResumeRunner(r, base)
	res := Execute(context.Background(), g, rr, Options{Concurrency: 1})
	if res.Failed {
		t.Fatalf("run failed: %+v", res.Nodes)
	}
	if strings.Join(saw, ",") != "land" {
		t.Fatalf("base runner saw %v, want only the unwitnessed step", saw)
	}
	if strings.Join(rr.Skipped(), ",") != "build" || strings.Join(rr.Executed(), ",") != "land" {
		t.Fatalf("measured skipped=%v executed=%v", rr.Skipped(), rr.Executed())
	}
	if res.Nodes["build"].Output != "artifact-out" {
		t.Fatalf("skipped step must replay its journaled output, got %q", res.Nodes["build"].Output)
	}
	if res.Nodes["land"].Output != "fresh:artifact-out" {
		t.Fatalf("re-executed step must see the replayed upstream output, got %q", res.Nodes["land"].Output)
	}
}

// A step that must re-execute with no runner bound refuses rather than reporting success.
func TestResumeRunnerWithoutBaseRefuses(t *testing.T) {
	g := twoStep(t)
	rr := NewResumeRunner(Resume(context.Background(), g, mustFold(t, nil, 1), "e", nil), nil)
	if _, err := rr.Run(context.Background(), RunInput{Node: g.Nodes[0]}); err == nil {
		t.Fatal("an unbound runner reported success")
	}
}

func TestAppendEntryRoundTrips(t *testing.T) {
	var b strings.Builder
	in := Entry{Run: "r", Step: "a", Kind: StepPure, OutputHash: "h", Output: "o", TSMS: 5}
	if err := AppendEntry(&b, in); err != nil {
		t.Fatalf("append: %v", err)
	}
	rows, err := ReadJournal(strings.NewReader(b.String()))
	if err != nil || len(rows) != 1 {
		t.Fatalf("round trip: rows=%v err=%v", rows, err)
	}
	if rows[0].Schema != JournalSchema || rows[0].Step != "a" || rows[0].Output != "o" {
		t.Fatalf("round trip lost fields: %+v", rows[0])
	}
}
