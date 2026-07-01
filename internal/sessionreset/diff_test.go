package sessionreset

import (
	"strings"
	"testing"
)

// TestDiffResetClassifiesEveryPartAndSpan proves the four-bucket accounting: every
// seed Part and every OmittedSpan lands in exactly one bucket, with the warm-prefix
// digest surfaced as its own must-requery row.
func TestDiffResetClassifiesEveryPartAndSpan(t *testing.T) {
	in := Input{Trace: "trace-parent", Messages: sampleTranscript(), FreshBudgetTok: 75}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)

	d := DiffReset(in, seed, tx)

	if d.RowCount() != len(seed.Parts)+len(tx.OmittedSpans)+1 {
		t.Fatalf("RowCount = %d, want %d (parts=%d omitted=%d +1 warm prefix)",
			d.RowCount(), len(seed.Parts)+len(tx.OmittedSpans)+1, len(seed.Parts), len(tx.OmittedSpans))
	}
	if len(d.Survived)+len(d.Summarized) != len(seed.Parts) {
		t.Fatalf("survived+summarized = %d, want %d (every Part classified once)",
			len(d.Survived)+len(d.Summarized), len(seed.Parts))
	}
	// Every OmittedSpan classified once, plus the synthetic warm-prefix row.
	if len(d.MustRequery)+len(d.Expired) != len(tx.OmittedSpans)+1 {
		t.Fatalf("must_requery+expired = %d, want %d (every omitted span classified once, +1 warm prefix)",
			len(d.MustRequery)+len(d.Expired), len(tx.OmittedSpans)+1)
	}
}

// TestDiffResetShowsConcreteBeforeAfterDelta proves the diff surfaces an actual,
// content-level delta across a real reset — not just that a struct got populated:
// the durable preference SURVIVES (summarized bucket, folded by durability_facts),
// the ephemeral timestamp line EXPIRES, and the system preamble needs a follow-up
// query to page back in.
func TestDiffResetShowsConcreteBeforeAfterDelta(t *testing.T) {
	in := Input{Trace: "trace-parent", Messages: sampleTranscript(), FreshBudgetTok: 75}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)
	d := DiffReset(in, seed, tx)

	// BEFORE: the drained transcript had 7 spans (sampleTranscript).
	if d.BeforeSpans != len(sampleTranscript()) {
		t.Fatalf("BeforeSpans = %d, want %d", d.BeforeSpans, len(sampleTranscript()))
	}
	// AFTER: the fresh session opens with a non-empty carried-over recap.
	if d.AfterChars == 0 || d.AfterChars != len(seed.Recap) {
		t.Fatalf("AfterChars = %d, want len(seed.Recap)=%d", d.AfterChars, len(seed.Recap))
	}

	// The durable preference ("I prefer concise answers") was FOLDED by
	// durability_facts into a summarized part, not carried verbatim.
	foundSummarized := false
	for _, p := range d.Summarized {
		if p.Name == "durability_facts" {
			foundSummarized = true
			if !strings.Contains(p.Text, "I prefer concise answers") {
				t.Fatalf("summarized durability_facts part missing the durable preference: %q", p.Text)
			}
		}
	}
	if !foundSummarized {
		t.Fatalf("expected durability_facts in Summarized, got: %+v", d.Summarized)
	}

	// verbatim_tail and task_distill carry forward near-verbatim -> survived.
	survivedNames := map[string]bool{}
	for _, p := range d.Survived {
		survivedNames[p.Name] = true
	}
	if !survivedNames["verbatim_tail"] {
		t.Fatalf("expected verbatim_tail in Survived, got: %+v", d.Survived)
	}

	// The ephemeral "it's 3pm..." line never appears ANYWHERE in the after-state
	// (neither survived nor summarized) — proving it actually expired, not just that
	// some span was labeled "expired".
	for _, p := range append(append([]DiffPart{}, d.Survived...), d.Summarized...) {
		if strings.Contains(p.Text, "3pm") {
			t.Fatalf("ephemeral line leaked into a carried-over part: %q", p.Text)
		}
	}
	// And a concrete EXPIRED span exists — the ephemeral line's digest, with a
	// recorded reason, present in the Expired bucket (not must_requery).
	if len(d.Expired) == 0 {
		t.Fatal("expected at least one expired span (the ephemeral line)")
	}
	for _, s := range d.Expired {
		if s.Digest == "" {
			t.Fatalf("expired span missing digest: %+v", s)
		}
		if requeryOmitReasons[s.Reason] {
			t.Fatalf("expired span carries a must-requery reason: %+v", s)
		}
	}

	// The system preamble is NOT lost — it needs an explicit follow-up (warm-prefix
	// replay) to page back in, so it must appear in MustRequery with the warm prefix
	// digest, not silently vanish.
	if len(d.MustRequery) == 0 {
		t.Fatal("expected at least one must-requery row (the warm prefix)")
	}
	foundWarmPrefix := false
	for _, s := range d.MustRequery {
		if s.Reason == "warm_prefix_replay" {
			foundWarmPrefix = true
			if s.Digest == "" || s.Digest != tx.WarmPrefixDigest {
				t.Fatalf("warm prefix must-requery row digest = %q, want tx.WarmPrefixDigest = %q", s.Digest, tx.WarmPrefixDigest)
			}
		}
	}
	if !foundWarmPrefix {
		t.Fatalf("expected a warm_prefix_replay row in MustRequery, got: %+v", d.MustRequery)
	}
}

// TestDiffResetIsDeterministic proves the same Input/Seed/ResetTransaction always
// renders the same ResetDiff — a reset diff is reproducible and auditable, matching
// the determinism the rest of sessionreset guarantees.
func TestDiffResetIsDeterministic(t *testing.T) {
	in := Input{Trace: "trace-parent", Messages: sampleTranscript(), FreshBudgetTok: 75}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)

	a := DiffReset(in, seed, tx)
	b := DiffReset(in, seed, tx)
	if a.Explain() != b.Explain() {
		t.Fatal("DiffReset.Explain() is not deterministic across identical inputs")
	}
	if a.Markdown() != b.Markdown() {
		t.Fatal("DiffReset.Markdown() is not deterministic across identical inputs")
	}
	if a.SeedDigest != DigestSeed(seed) {
		t.Fatalf("diff seed digest = %q, want %q", a.SeedDigest, DigestSeed(seed))
	}
}

// TestResetDiffExplainNamesAllFourBuckets proves the operator-readable report names
// every bucket the issue's Done condition asks for, by label.
func TestResetDiffExplainNamesAllFourBuckets(t *testing.T) {
	in := Input{Trace: "trace-parent", Messages: sampleTranscript(), FreshBudgetTok: 75}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)
	out := DiffReset(in, seed, tx).Explain()

	for _, want := range []string{"SURVIVED", "SUMMARIZED", "MUST-REQUERY", "EXPIRED", "seed_digest="} {
		if !strings.Contains(out, want) {
			t.Fatalf("Explain() missing %q:\n%s", want, out)
		}
	}
}

// TestResetDiffMarkdownRendersAllFourSections proves the shareable Markdown report
// has one section per bucket, mirroring ctxplan.Preview.Markdown's convention.
func TestResetDiffMarkdownRendersAllFourSections(t *testing.T) {
	in := Input{Trace: "trace-parent", Messages: sampleTranscript(), FreshBudgetTok: 75}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)
	md := DiffReset(in, seed, tx).Markdown()

	for _, want := range []string{
		"# Session reset diff",
		"## Survived (carried forward near-verbatim)",
		"## Summarized (folded/distilled into the seed)",
		"## Must re-query (cold, recoverable via an explicit follow-up)",
		"## Expired (dropped, no recovery handle beyond the digest)",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("Markdown() missing %q:\n%s", want, md)
		}
	}
}

// TestDiffResetOnEmptyTranscriptIsTotalNotError proves a no-op reset (nothing to
// carry, nothing omitted) still renders a valid, total diff — mirroring
// ctxplan.Preview and memview.Timeline's "empty is a valid state" posture.
func TestDiffResetOnEmptyTranscriptIsTotalNotError(t *testing.T) {
	in := Input{Trace: "trace-parent", FreshBudgetTok: 75}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)
	d := DiffReset(in, seed, tx)

	if d.BeforeSpans != 0 {
		t.Fatalf("BeforeSpans = %d, want 0 for an empty transcript", d.BeforeSpans)
	}
	if d.RowCount() != 0 {
		t.Fatalf("RowCount = %d, want 0 for an empty transcript with no seed/omissions", d.RowCount())
	}
	// Explain/Markdown must not panic and must still show all four bucket headers.
	if !strings.Contains(d.Explain(), "SURVIVED") {
		t.Fatal("Explain() should still render bucket headers on an empty diff")
	}
}
