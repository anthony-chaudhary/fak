package memq

import (
	"context"
	"sort"
	"testing"
)

// dedupRecallQueryForTest is the default recall-assembly shape with the read-side
// exact-digest collapse (OpDedup) wired in — the #2506 default-on dedup step that
// runs AFTER admission (filter) and BEFORE page-in (render). The opt-in near-dup
// advisory is set separately on the Query (see the advisory test).
func dedupRecallQueryForTest(intent string) Query {
	return Query{
		Intent: intent,
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
				{Op: PredEq, Field: "sealed", Value: "false"},
				{Op: PredEq, Field: "tombstoned", Value: "false"},
			}}},
			{Kind: OpDedup},
			{Kind: OpRender},
		},
	}
}

// #2506: two byte-identical notes (same content Digest) collapse to ONE cell at
// recall assembly; the folded sibling's provenance rides on a conflation Effect so
// the collapse is auditable, and it never pages into context twice. The exact-digest
// collapse is default-on (dedup is a standard recall-assembly step), read-only, and
// changes behavior only for byte-identical cells.
func TestRecallCollapsesDigestIdenticalCells(t *testing.T) {
	const dupBody = "The build server is named acme and runs in the east region."
	dir := fixtureNotesStore(t,
		"# Memory index\n\n- [Dup one](dup1.md) — build server\n- [Dup two](dup2.md) — build server\n- [Solo](solo.md) — a distinct fact\n",
		map[string]string{
			"dup1.md": "---\nname: dup-one\ndescription: build server location\nmetadata:\n  type: reference\n---\n\n" + dupBody + "\n",
			"dup2.md": "---\nname: dup-two\ndescription: build server location\nmetadata:\n  type: reference\n---\n\n" + dupBody + "\n",
			"solo.md": "---\nname: solo\ndescription: a distinct fact\nmetadata:\n  type: user\n---\n\nThe user prefers bullet points over long prose passages.\n",
		},
	)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)

	res, err := Run(context.Background(), b, dedupRecallQueryForTest("build server acme east"), Caps{})
	if err != nil {
		t.Fatal(err)
	}

	// The identical pair collapsed to one cell; the distinct note is untouched.
	rendered := map[string]bool{}
	for _, it := range res.Rendered {
		rendered[it.ID] = true
	}
	if len(rendered) != 2 || !rendered["dup1.md"] || !rendered["solo.md"] || rendered["dup2.md"] {
		t.Fatalf("identical pair must collapse to dup1 only + solo; rendered=%+v", rendered)
	}

	// The conflation Effect carries the folded sibling's ID (auditable provenance).
	var conflation *Effect
	for i := range res.Effects {
		if res.Effects[i].Kind == OpDedup {
			conflation = &res.Effects[i]
			break
		}
	}
	if conflation == nil {
		t.Fatalf("a dedup conflation Effect must be recorded; effects=%+v", res.Effects)
	}
	if len(conflation.Cells) != 1 || conflation.Cells[0] != "dup2.md" {
		t.Fatalf("conflation must name the folded sibling dup2.md, got %v", conflation.Cells)
	}
	if res.Stats.DedupCollapsed != 1 {
		t.Fatalf("DedupCollapsed stat = %d, want 1", res.Stats.DedupCollapsed)
	}
	if len(res.Working) != 2 {
		t.Fatalf("working set = %d cells, want 2 (collapsed dup + solo)", len(res.Working))
	}
}

// #2506 default-on behavior: the built-in recall driver inserts dedup before the
// top-K limit, so byte-identical twins do not consume multiple recall slots.
func TestRecallDriverDedupsBeforeLimit(t *testing.T) {
	const dupBody = "The build server named acme runs in the east region."
	dir := fixtureNotesStore(t,
		"# Memory index\n\n- [Dup one](dup1.md) — build server\n- [Dup two](dup2.md) — build server\n- [Solo](solo.md) — fallback build server\n",
		map[string]string{
			"dup1.md": "---\nname: dup-one\ndescription: build server acme east\nmetadata:\n  type: reference\n---\n\n" + dupBody + "\n",
			"dup2.md": "---\nname: dup-two\ndescription: build server acme east\nmetadata:\n  type: reference\n---\n\n" + dupBody + "\n",
			"solo.md": "---\nname: solo\ndescription: build server acme\nmetadata:\n  type: reference\n---\n\nThe build server has a documented fallback in the west region.\n",
		},
	)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)

	q := Get0(t, "recall").Build(Params{Intent: "build server acme east", K: 2})
	dedupAt, limitAt := -1, -1
	for i, op := range q.Ops {
		switch op.Kind {
		case OpDedup:
			dedupAt = i
		case OpLimit:
			limitAt = i
		}
	}
	if dedupAt < 0 || limitAt < 0 || dedupAt > limitAt {
		t.Fatalf("recall driver must dedup before limit; dedupAt=%d limitAt=%d ops=%+v", dedupAt, limitAt, q.Ops)
	}
	res, err := Run(context.Background(), b, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}

	rendered := map[string]bool{}
	for _, it := range res.Rendered {
		rendered[it.ID] = true
	}
	if len(rendered) != 2 || !rendered["dup1.md"] || !rendered["solo.md"] || rendered["dup2.md"] {
		t.Fatalf("recall driver must render dedup representative + next distinct cell; rendered=%+v", rendered)
	}
	if res.Stats.DedupCollapsed != 1 {
		t.Fatalf("DedupCollapsed stat = %d, want 1", res.Stats.DedupCollapsed)
	}
}

// #2506 advisory: with the opt-in near-dup flag on, a PARAPHRASED near-twin pair
// (distinct digests, high body similarity) is reported on Result.Advisory for the
// compactor to consume. The advisory never collapses (a fuzzy signal never silently
// decides); distinct notes are not flagged.
func TestNearDupAdvisoryFlagsParaphrasedPair(t *testing.T) {
	dir := fixtureNotesStore(t,
		"# Memory index\n\n- [Original](original.md) — build server\n- [Paraphrase](para.md) — reworded\n- [Unrelated](unrelated.md) — a different fact\n",
		map[string]string{
			"original.md":  "---\nname: original\ndescription: build server\nmetadata:\n  type: reference\n---\n\nThe build server is named acme and it runs in the east region.\n",
			"para.md":      "---\nname: paraphrase\ndescription: reworded build server\nmetadata:\n  type: reference\n---\n\nThe build server named acme runs in the east region.\n",
			"unrelated.md": "---\nname: unrelated\ndescription: a different fact\nmetadata:\n  type: user\n---\n\nThe user wants concise answers with the outcome stated first.\n",
		},
	)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)

	q := dedupRecallQueryForTest("build server acme east")
	q.NearDupThreshold = 0.70 // opt-in advisory
	res, err := Run(context.Background(), b, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}

	// The paraphrased pair is flagged; the distinct note is not.
	if len(res.Advisory) != 1 {
		t.Fatalf("advisory = %+v, want exactly the paraphrased pair", res.Advisory)
	}
	p := res.Advisory[0]
	if p.A != "original.md" || p.B != "para.md" {
		t.Fatalf("advisory pair = (%s, %s), want (original.md, para.md)", p.A, p.B)
	}
	if p.Score < 0.70 {
		t.Fatalf("paraphrase cosine %.3f below the 0.70 threshold", p.Score)
	}
	if res.Stats.NearDupAdvisory != 1 {
		t.Fatalf("NearDupAdvisory stat = %d, want 1", res.Stats.NearDupAdvisory)
	}
	// The advisory must NOT have collapsed the near-twins — exact digest differs, and
	// the fuzzy signal never silently decides.
	if res.Stats.DedupCollapsed != 0 {
		t.Fatalf("near-dup advisory must not collapse; DedupCollapsed=%d", res.Stats.DedupCollapsed)
	}
}

// #2506: distinct notes (different digests, low body similarity) are untouched — no
// collapse, no advisory — so recall behavior changes only for byte-identical cells.
func TestDistinctCellsUntouched(t *testing.T) {
	dir := fixtureNotesStore(t,
		"# Memory index\n\n- [Alpha](alpha.md) — one\n- [Beta](beta.md) — two\n- [Gamma](gamma.md) — three\n",
		map[string]string{
			"alpha.md": "---\nname: alpha\nmetadata:\n  type: user\n---\n\nThe user prefers terse answers with the outcome first.\n",
			"beta.md":  "---\nname: beta\nmetadata:\n  type: feedback\n---\n\nAlways run the gate helper before claiming the task is done.\n",
			"gamma.md": "---\nname: gamma\nmetadata:\n  type: project\n---\n\nShip the release notes under the docs folder once green.\n",
		},
	)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)

	q := dedupRecallQueryForTest("distinct facts gate release")
	q.NearDupThreshold = 0.70
	res, err := Run(context.Background(), b, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}

	if res.Stats.DedupCollapsed != 0 {
		t.Fatalf("distinct notes must not collapse; DedupCollapsed=%d", res.Stats.DedupCollapsed)
	}
	if len(res.Working) != 3 || len(res.Rendered) != 3 {
		t.Fatalf("all three distinct notes must survive; working=%d rendered=%d", len(res.Working), len(res.Rendered))
	}
	if len(res.Advisory) != 0 {
		// sort ids for a stable failure message
		got := make([]string, 0, len(res.Advisory))
		for _, p := range res.Advisory {
			got = append(got, p.A+"~"+p.B)
		}
		sort.Strings(got)
		t.Fatalf("distinct notes must yield no advisory pairs; got %+v", got)
	}
}
