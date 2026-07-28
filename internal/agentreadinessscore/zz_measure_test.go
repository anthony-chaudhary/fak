package agentreadinessscore

// End-to-end cover for Build's IMPURE shell — the disk + git gather wrapped around the
// pure KPI functions. The pure funcs are unit-tested directly in agentreadinessscore_test.go
// and TestLivePayloadIsWellFormed pins the SHAPE of a live payload; neither proves that
// Build actually READS the tree, i.e. that putting a real affordance on disk moves the
// number. These tests run Build against a hermetic fixture root and change ONE file at a
// time, so a break in the gather wiring (a dropped present() lookup, a KPI unhooked from
// the payload, a hard defect silently demoted to a soft signal) fails here.

import (
	"path/filepath"
	"strings"
	"testing"
)

// fixtureRoot is a throwaway tree Build accepts as a repo root. A `.git` MARKER is all the
// repo guard stats for, and `git ls-files` then fails inside it, so every KPI resolves the
// tree through the fileExists fallback — the path a fresh, nothing-staged clone takes.
func fixtureRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, ".git"), "gitdir: ./no-such-dir\n")
	return root
}

// kpiByName returns the named KPI from a payload, failing the test if the gather dropped it.
func kpiByName(t *testing.T, p Payload, name string) KPI {
	t.Helper()
	for _, k := range p.KPIs {
		if k.Kpi == name {
			return k
		}
	}
	t.Fatalf("payload carries no %q KPI (has %d)", name, len(p.KPIs))
	return KPI{}
}

// TestBuildRefusesARootThatIsNotARepo pins the guard at the top of Build: a directory with
// no .git is a tooling error, not a zero-score repo — scoring an arbitrary directory would
// report a catastrophic grade for what is really "you ran it from the wrong place".
func TestBuildRefusesARootThatIsNotARepo(t *testing.T) {
	p := Build(t.TempDir())

	if p.OK {
		t.Errorf("a non-repo root must not be OK: %+v", p)
	}
	if p.Verdict != "AUDIT_ERROR" || p.Finding != "tooling_error" {
		t.Errorf("verdict/finding = %q/%q, want AUDIT_ERROR/tooling_error", p.Verdict, p.Finding)
	}
	if !strings.Contains(p.Reason, "not a git repo") {
		t.Errorf("reason %q does not name the cause", p.Reason)
	}
	// An error payload must not look like a measurement: no KPIs, no corpus numbers a
	// caller could mistake for a real score.
	if len(p.KPIs) != 0 {
		t.Errorf("error payload carries %d KPIs, want none", len(p.KPIs))
	}
	if len(p.Corpus) != 0 {
		t.Errorf("error payload carries a corpus %v, want empty", p.Corpus)
	}
	if !filepath.IsAbs(p.Workspace) {
		t.Errorf("workspace %q is not absolute — Build must report the resolved root", p.Workspace)
	}
}

// TestBuildReadsTheDocMapFromDisk is the single-variable experiment: llms.txt and
// llms-full.txt feed exactly one KPI, and one is HARD while the other is SOFT. Writing each
// file in turn must move friction_debt and soft_signals INDEPENDENTLY — the clearest proof
// that gather reads the tree and that buildPayload keeps the two ledgers apart.
func TestBuildReadsTheDocMapFromDisk(t *testing.T) {
	root := fixtureRoot(t)

	bare := Build(root)
	if bare.OK {
		t.Fatalf("an empty tree cannot be agent-ready: %+v", bare.Corpus)
	}
	if k := kpiByName(t, bare, "llms_map"); len(k.Defects) != 1 || len(k.Soft) != 1 {
		t.Fatalf("bare tree llms_map = %d hard / %d soft, want 1/1: %+v", len(k.Defects), len(k.Soft), k)
	}
	bareDebt := corpusInt(bare.Corpus, "friction_debt")
	bareSoft := corpusInt(bare.Corpus, "soft_signals")
	bareScore := corpusFloat(bare.Corpus, "score")

	// (1) the HARD half — llms.txt on disk clears the defect and nothing else.
	mustWrite(t, filepath.Join(root, llmsFile), "# fak doc map\n- [AGENTS.md](AGENTS.md)\n")
	withMap := Build(root)
	if k := kpiByName(t, withMap, "llms_map"); len(k.Defects) != 0 {
		t.Errorf("%s on disk did not clear llms_map: %v", llmsFile, k.Defects)
	}
	if got, want := corpusInt(withMap.Corpus, "friction_debt"), bareDebt-1; got != want {
		t.Errorf("friction_debt = %d, want %d (exactly the one defect %s retires)", got, want, llmsFile)
	}
	if got := corpusInt(withMap.Corpus, "soft_signals"); got != bareSoft {
		t.Errorf("soft_signals = %d, want %d unchanged — a hard fix must not move the soft ledger", got, bareSoft)
	}
	if got := corpusFloat(withMap.Corpus, "score"); got <= bareScore {
		t.Errorf("score = %v, want above the bare %v after retiring a defect", got, bareScore)
	}

	// (2) the SOFT half — llms-full.txt clears an advisory only: soft drops, debt holds.
	mustWrite(t, filepath.Join(root, llmsFullFile), "# fak doc map (inlined)\n")
	withFull := Build(root)
	if k := kpiByName(t, withFull, "llms_map"); len(k.Defects) != 0 || len(k.Soft) != 0 {
		t.Errorf("llms_map = %d hard / %d soft with both files present, want 0/0: %+v",
			len(k.Defects), len(k.Soft), k)
	}
	if got, want := corpusInt(withFull.Corpus, "soft_signals"), bareSoft-1; got != want {
		t.Errorf("soft_signals = %d, want %d after %s", got, want, llmsFullFile)
	}
	if got, want := corpusInt(withFull.Corpus, "friction_debt"), bareDebt-1; got != want {
		t.Errorf("friction_debt = %d, want %d held — an advisory must never gate", got, want)
	}
	if got := corpusFloat(withFull.Corpus, "score"); got <= corpusFloat(withMap.Corpus, "score") {
		t.Errorf("score = %v, want above %v — clearing an advisory still scores", got,
			corpusFloat(withMap.Corpus, "score"))
	}
}

// TestBuildScoresTheAgentsEntrypointFromItsText walks AGENTS.md from absent, to present but
// contentless, to complete. It covers the branch the doc-map test cannot: a KPI whose score
// comes from SCANNING a file's text rather than from its mere presence — so a gather that
// finds the file but forgets to read it is caught.
func TestBuildScoresTheAgentsEntrypointFromItsText(t *testing.T) {
	root := fixtureRoot(t)

	absent := kpiByName(t, Build(root), "agents_entrypoint")
	if len(absent.Defects) != 1 || !strings.Contains(absent.Defects[0], agentsFile) {
		t.Fatalf("absent %s = %v, want one defect naming the file", agentsFile, absent.Defects)
	}
	if absent.Score != 0 {
		t.Errorf("absent %s scored %d, want 0", agentsFile, absent.Score)
	}

	// Present but useless: it says what the project is and nothing an agent can run, so the
	// three command elements (build / test / first verb) must each be a defect.
	mustWrite(t, filepath.Join(root, agentsFile), "# AGENTS.md\n\nWhat this is: **fak** is a kernel.\n")
	stub := kpiByName(t, Build(root), "agents_entrypoint")
	if len(stub.Defects) != 3 {
		t.Errorf("contentless %s = %d defect(s), want 3 (build/test/run): %v",
			agentsFile, len(stub.Defects), stub.Defects)
	}
	for _, want := range []string{"build command", "test command", "runnable first verb"} {
		if !anyContains(stub.Defects, want) {
			t.Errorf("no defect mentions %q: %v", want, stub.Defects)
		}
	}
	if stub.Score <= absent.Score || stub.Score >= 100 {
		t.Errorf("contentless %s scored %d, want between %d and 100", agentsFile, stub.Score, absent.Score)
	}

	// Complete: identity + build + test + a runnable first verb.
	mustWrite(t, filepath.Join(root, agentsFile), "# AGENTS.md\n\n"+
		"What this is: **fak** is a kernel.\n\n"+
		"    go build ./...\n    make test\n    go run ./cmd/fak preflight\n")
	full := kpiByName(t, Build(root), "agents_entrypoint")
	if len(full.Defects) != 0 {
		t.Errorf("complete %s still has defects: %v", agentsFile, full.Defects)
	}
	if full.Score != 100 {
		t.Errorf("complete %s scored %d, want 100", agentsFile, full.Score)
	}
}
