package docfreshrsi

import (
	"strings"
	"testing"
)

// docA is defect-free: an orientation signpost, a fresh version pin, a `## Read next`
// link, and a citation into docB's "Deep dive" section (b.md#deep-dive).
const docA = `# A Guide

> Orientation: who this is for.

See the [deep dive](b.md#deep-dive) for the details.

Pinned to v0.7.0.

## Read next

- [b.md](b.md)
`

// docB's ONLY mechanical defect is a stale version pin (v0.5.0) that lives INSIDE the
// "Deep dive" section docA cites — so the two ways to retire that defect (rewrite the
// pin in place vs delete the whole section) differ only in whether docA's link survives.
const docB = `# B Guide

> Orientation: who this is for.

## Deep dive

Released in v0.5.0 — the gory details.

## Read next

- [a.md](a.md)
`

func pairCorpus() Corpus { return Corpus{"a.md": docA, "b.md": docB} }
func target() Target     { return Target{Version: "v0.7.0"} }

// deleteSection returns a Patch that removes a `## <heading>` section (heading line
// through the line before the next heading, or EOF). This is the "fewer defects by
// deleting the content" move the truth fence must reject.
func deleteSection(doc, heading string) Patch {
	return func(c Corpus) Corpus {
		lines := strings.Split(c[doc], "\n")
		var out []string
		skip := false
		for _, ln := range lines {
			if strings.HasPrefix(ln, "## ") {
				skip = strings.TrimSpace(strings.TrimPrefix(ln, "##")) == heading
			}
			if !skip {
				out = append(out, ln)
			}
		}
		c[doc] = strings.Join(out, "\n")
		return c
	}
}

// TestAntiGoodhartDeleteRevertsRewriteKeeps is the paired witness the issue asks for:
// two candidates retire the SAME defect for the SAME debt drop (1 -> 0), but only the
// one that keeps docA's cited link resolving is KEPT. The debt-equal pair proves the
// gate is not a bare debt-count — the truth syscall (all links resolve) is what
// separates them.
func TestAntiGoodhartDeleteRevertsRewriteKeeps(t *testing.T) {
	tgt := target()

	// The honest fix: rewrite the stale pin in place. Debt 1->0, link survives -> KEEP.
	rewrite := Candidate{Label: "rewrite", Class: VersionPin, Doc: "b.md",
		Apply: func(c Corpus) Corpus { c["b.md"] = rewriteStalePins(c["b.md"], tgt.Version); return c }}
	vR, keptR := EvaluateCandidate(pairCorpus(), rewrite, tgt)
	if !(vR.DebtBefore == 1 && vR.DebtAfter == 0) {
		t.Fatalf("rewrite debt = %d->%d, want 1->0", vR.DebtBefore, vR.DebtAfter)
	}
	if !vR.Improved || !vR.Clean || !vR.LinksResolve || !vR.Kept || vR.Decision != "KEEP" {
		t.Fatalf("rewrite should KEEP: %+v", vR)
	}
	if vR.Score.Name != "doc_freshness_debt" || vR.Score.Grade != "clear" {
		t.Fatalf("rewrite score = %+v, want doc_freshness_debt/clear", vR.Score)
	}
	if got := scoreComponentValue(vR.Score, "debt_delta"); got != 1 {
		t.Fatalf("rewrite debt_delta score = %.4g, want 1", got)
	}
	if Debt(keptR, tgt) != 0 {
		t.Fatalf("kept corpus debt = %d, want 0", Debt(keptR, tgt))
	}

	// The goodhart move: delete the cited section. SAME debt drop (1->0), but docA's
	// b.md#deep-dive link now dangles -> truth-dirty -> REVERT.
	del := Candidate{Label: "delete-deep-dive", Class: VersionPin, Doc: "b.md",
		Apply: deleteSection("b.md", "Deep dive")}
	vD, keptD := EvaluateCandidate(pairCorpus(), del, tgt)
	if !(vD.DebtBefore == 1 && vD.DebtAfter == 0) {
		t.Fatalf("delete debt = %d->%d, want 1->0 (the debt drop the fence must NOT reward)", vD.DebtBefore, vD.DebtAfter)
	}
	if !vD.Improved {
		t.Fatal("delete must show a debt improvement — otherwise the test does not isolate the truth fence")
	}
	if vD.LinksResolve {
		t.Fatalf("delete must break docA's cited link, got LinksResolve=true (dangling=%v)", vD.Dangling)
	}
	if vD.Kept || vD.Decision != "REVERT" {
		t.Fatalf("delete must REVERT despite the debt drop: %+v", vD)
	}
	if vD.Score.Grade != "truth-dirty" {
		t.Fatalf("delete score grade = %q, want truth-dirty (score=%+v)", vD.Score.Grade, vD.Score)
	}
	if got := scoreComponentValue(vD.Score, "debt_delta"); got != 1 {
		t.Fatalf("delete debt_delta score = %.4g, want 1", got)
	}
	if got := scoreComponentValue(vD.Score, "dangling"); got != 1 {
		t.Fatalf("delete dangling score = %.4g, want 1", got)
	}
	if len(vD.Dangling) == 0 || !strings.Contains(vD.Dangling[0], "b.md#deep-dive") {
		t.Fatalf("REVERT diagnostic must name the dangling link, got %v", vD.Dangling)
	}
	// On REVERT the loop carries the UNCHANGED corpus forward (debt still 1).
	if Debt(keptD, tgt) != 1 {
		t.Fatalf("reverted corpus debt = %d, want the unchanged 1", Debt(keptD, tgt))
	}
}

// TestKeepNeedsAllThreeSignals proves each of the three keep conditions is load-bearing:
// drop the gain, the clean-status, or the link-resolution and the candidate REVERTs;
// only all-three KEEPs. This is shipgate's ClassFull contract, exercised through the loop.
func TestKeepNeedsAllThreeSignals(t *testing.T) {
	tgt := target()

	good := Candidate{Label: "good", Class: VersionPin, Doc: "b.md",
		Apply: func(c Corpus) Corpus { c["b.md"] = rewriteStalePins(c["b.md"], tgt.Version); return c }}
	if v, _ := EvaluateCandidate(pairCorpus(), good, tgt); !v.Kept {
		t.Fatalf("gain+clean+links should KEEP: %+v", v)
	}

	// No gain: a no-op candidate (debt unchanged) reverts even though clean+links hold.
	noop := Candidate{Label: "noop", Class: VersionPin, Doc: "b.md", Apply: func(c Corpus) Corpus { return c }}
	if v, _ := EvaluateCandidate(pairCorpus(), noop, tgt); v.Kept || v.Improved {
		t.Fatalf("no debt gain must REVERT: %+v", v)
	}

	// Not clean: the fix lowers debt and keeps links resolving, but ALSO edits a doc
	// other than its declared target (a stray, dirty-worktree change) -> REVERT.
	stray := Candidate{Label: "stray", Class: VersionPin, Doc: "b.md",
		Apply: func(c Corpus) Corpus {
			c["b.md"] = rewriteStalePins(c["b.md"], tgt.Version)
			c["a.md"] = c["a.md"] + "\nstray edit outside the declared target.\n"
			return c
		}}
	if v, _ := EvaluateCandidate(pairCorpus(), stray, tgt); v.Kept || v.Clean {
		t.Fatalf("a stray off-target edit must REVERT (clean=false): %+v", v)
	}

	// Not link-clean is the delete case, covered by TestAntiGoodhartDeleteRevertsRewriteKeeps.
}

// TestRefreshKeepsMechanicalFixesNeverMutatesInput proves the end-to-end loop: it
// retires the proposed mechanical defects in an isolated copy, returns a debt-0 corpus
// plus a kept-verdict log, and leaves the caller's corpus (the `main` analog) untouched.
func TestRefreshKeepsMechanicalFixesNeverMutatesInput(t *testing.T) {
	tgt := Target{Version: "v0.7.0"}
	base := Corpus{
		// One doc missing every signpost (orientation, read-next) AND stale-pinned.
		"raw.md": "# Raw\n\nReleased in v0.4.0.\n",
		// A second doc so the appended read-next link has a real target to resolve to.
		"hub.md": "# Hub\n\n> Orientation: the hub.\n\nPinned v0.7.0.\n\n## Read next\n\n- [raw.md](raw.md)\n",
	}
	before := Debt(base, tgt)
	if before == 0 {
		t.Fatal("test corpus must start with debt to retire")
	}

	kept, log := Refresh(base, tgt)

	if Debt(kept, tgt) != 0 {
		t.Fatalf("Refresh left debt = %d, want 0 (every mechanical defect retired)", Debt(kept, tgt))
	}
	if ok, dangling := LinksResolve(kept); !ok {
		t.Fatalf("kept corpus must be link-clean, dangling=%v", dangling)
	}
	keptCount := 0
	for _, v := range log {
		if v.Kept {
			keptCount++
		}
	}
	if keptCount == 0 || keptCount != before {
		t.Fatalf("kept %d verdicts, want %d (one per retired defect)", keptCount, before)
	}
	// The loop never mutates `main`: the input corpus is byte-identical afterward.
	if Debt(base, tgt) != before {
		t.Fatalf("Refresh mutated the input corpus: debt %d -> %d", before, Debt(base, tgt))
	}
	if !strings.Contains(base["raw.md"], "v0.4.0") {
		t.Fatal("Refresh must NOT rewrite the caller's corpus — landing is a separate step")
	}
}

// TestTruthCleanCatchesUnknownCommand proves the truth syscall also fences cited
// commands when an oracle is supplied: a doc citing a command outside the known set is
// truth-dirty (so a refresh that introduces an unrunnable command cannot be kept).
func TestTruthCleanCatchesUnknownCommand(t *testing.T) {
	c := Corpus{"q.md": "# Q\n\n```sh\n./fak serve --policy p.json\n```\n"}

	if ok, _ := truthClean(c, Target{KnownCommands: map[string]bool{"./fak": true}}); !ok {
		t.Fatal("a cited command in the known set must be truth-clean")
	}
	ok, bad := truthClean(c, Target{KnownCommands: map[string]bool{"make": true}})
	if ok || len(bad) == 0 || !strings.Contains(bad[0], "cmd:./fak") {
		t.Fatalf("a cited command outside the oracle must be truth-dirty, got ok=%v bad=%v", ok, bad)
	}
	// No oracle -> command checking is skipped (links remain the truth signal).
	if ok, _ := truthClean(c, Target{}); !ok {
		t.Fatal("an empty command oracle must skip command checking")
	}
}

func scoreComponentValue(score Scorecard, name string) float64 {
	for _, c := range score.Components {
		if c.Name == name {
			return c.Value
		}
	}
	return -1
}

// TestLinkAndSlugResolution locks the resolver's contract the fence depends on.
func TestLinkAndSlugResolution(t *testing.T) {
	if got := slug("Deep Dive (v0.5.0)"); got != "deep-dive-v050" {
		t.Fatalf("slug = %q, want deep-dive-v050", got)
	}
	c := Corpus{
		"x.md": "# X\n\n## Section One\n\n[self](#section-one) [cross](y.md#target) [ext](https://e.com) [dead](y.md#gone)\n",
		"y.md": "# Y\n\n## Target\n\nbody\n",
	}
	ok, dangling := LinksResolve(c)
	if ok {
		t.Fatalf("expected the #gone anchor to dangle, got clean (dangling=%v)", dangling)
	}
	if len(dangling) != 1 || !strings.Contains(dangling[0], "y.md#gone") {
		t.Fatalf("only y.md#gone should dangle, got %v", dangling)
	}
}
