package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/corelockaudit"
	"github.com/anthony-chaudhary/fak/internal/corelocks"
)

// auditPaths is a small helper: load the shipped taxonomy and audit a path set, failing the test
// if the shipped fixture is malformed (the corelocks package's own tests guard that, but a
// fail-open auditCoreLockPaths would silently return nil here, so assert it loads).
func auditPaths(t *testing.T, paths ...string) corelockaudit.Report {
	t.Helper()
	tax, err := corelocks.LoadFixture()
	if err != nil {
		t.Fatalf("corelocks.LoadFixture: %v", err)
	}
	return corelockaudit.Audit(tax, paths)
}

// TestCoreLockWarnings_onlyWarnVerdict proves the fold to advisory rows keeps the warn-verdict
// findings and DROPS the advisory-ok ones (open-leaf, shadow-learn) so ordinary leaf edits stay
// quiet. A soft-contract path warns with its reason, paths, witnesses, and mode=warning.
func TestCoreLockWarnings_onlyWarnVerdict(t *testing.T) {
	rep := auditPaths(t,
		"internal/canon/schema.go", // soft-contract -> warn
		"internal/rsiloop/loop.go", // shadow-learn  -> ok (dropped)
		"cmd/fak/info.go",          // open-leaf     -> ok (dropped)
	)
	ws := coreLockWarnings(rep)
	if len(ws) != 1 {
		t.Fatalf("want exactly 1 warn row (soft-contract), got %d: %+v", len(ws), ws)
	}
	w := ws[0]
	if w.Class != "soft-contract" {
		t.Errorf("class = %q, want soft-contract", w.Class)
	}
	if w.LockID != "soft-contract" {
		t.Errorf("lock_id = %q, want soft-contract", w.LockID)
	}
	if w.Reason != "CORE_CONTRACT_WITNESS_MISSING" {
		t.Errorf("reason = %q, want CORE_CONTRACT_WITNESS_MISSING", w.Reason)
	}
	if w.Mode != coreLockModeWarning {
		t.Errorf("mode = %q, want %q", w.Mode, coreLockModeWarning)
	}
	if len(w.Witness) == 0 {
		t.Fatalf("warn row must carry a witness command, got none")
	}
	if want := "go test ./internal/corelockaudit/"; w.Witness[0] != want {
		t.Errorf("first witness = %q, want %q", w.Witness[0], want)
	}
	if len(w.Paths) != 1 || w.Paths[0] != "internal/canon/schema.go" {
		t.Errorf("paths = %v, want [internal/canon/schema.go]", w.Paths)
	}
}

// TestAuditCoreLockPaths_emptyAndLeafQuiet confirms the advisory entry point stays quiet for the
// two cases that must NOT warn: an empty path set and a set of only ordinary leaf edits.
func TestAuditCoreLockPaths_emptyAndLeafQuiet(t *testing.T) {
	if ws := auditCoreLockPaths(nil); len(ws) != 0 {
		t.Errorf("empty path set: want no warnings, got %+v", ws)
	}
	if ws := auditCoreLockPaths([]string{"cmd/fak/info.go", "README.md"}); len(ws) != 0 {
		t.Errorf("ordinary leaf edits: want no warnings, got %+v", ws)
	}
}

// TestRenderCoreLockWarnings_humanRender exercises the human render: it must name the class, the
// reason, the offending path, and the literal witness command, and announce that it is advisory.
func TestRenderCoreLockWarnings_humanRender(t *testing.T) {
	ws := auditCoreLockPaths([]string{"internal/canon/schema.go"})
	var b bytes.Buffer
	n := renderCoreLockWarnings(&b, ws)
	if n != 1 {
		t.Fatalf("renderCoreLockWarnings returned %d, want 1", n)
	}
	out := b.String()
	for _, want := range []string{
		"core-lock advisory:",
		"advisory — NOT blocking",
		"soft-contract",
		"CORE_CONTRACT_WITNESS_MISSING",
		"internal/canon/schema.go",
		"witness to clear:",
		"go test ./internal/corelockaudit/",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("human render missing %q; got:\n%s", want, out)
		}
	}
}

// TestRenderCoreLockWarnings_quietWhenNone confirms the renderer writes NOTHING when there are no
// warnings — an ordinary leaf edit produces no noise.
func TestRenderCoreLockWarnings_quietWhenNone(t *testing.T) {
	var b bytes.Buffer
	if n := renderCoreLockWarnings(&b, nil); n != 0 || b.Len() != 0 {
		t.Errorf("want silent render for no warnings; n=%d out=%q", n, b.String())
	}
}

// TestEmitHygieneJSON_render exercises the JSON render of the warnings: the advisory rows live
// under core_lock_warnings with the lock id / class / reason / witness / mode fields a metrics
// consumer needs, separate from the blocking findings.
func TestEmitHygieneJSON_render(t *testing.T) {
	ws := auditCoreLockPaths([]string{"internal/canon/schema.go"})
	var out, errb bytes.Buffer
	emitHygieneJSON(&out, &errb, nil, ws)

	var got struct {
		Findings         []map[string]any  `json:"findings"`
		Count            int               `json:"count"`
		CoreLockWarnings []coreLockWarning `json:"core_lock_warnings"`
		CoreLockWarnMode string            `json:"core_lock_warn_mode"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("emitHygieneJSON did not emit valid JSON: %v\n%s", err, out.String())
	}
	if got.CoreLockWarnMode != coreLockModeWarning {
		t.Errorf("core_lock_warn_mode = %q, want %q", got.CoreLockWarnMode, coreLockModeWarning)
	}
	if len(got.CoreLockWarnings) != 1 {
		t.Fatalf("want 1 core_lock_warning, got %d", len(got.CoreLockWarnings))
	}
	w := got.CoreLockWarnings[0]
	if w.LockID != "soft-contract" || w.Reason != "CORE_CONTRACT_WITNESS_MISSING" {
		t.Errorf("warning row = %+v, want soft-contract / CORE_CONTRACT_WITNESS_MISSING", w)
	}
	if w.Mode != coreLockModeWarning {
		t.Errorf("row mode = %q, want %q", w.Mode, coreLockModeWarning)
	}
	if len(w.Witness) == 0 {
		t.Errorf("JSON warning row must carry a witness, got %+v", w)
	}
}

// TestRunCommitPreview_softContractWarnsButExitsZero is the CLI-level witness: a preview of a
// commit touching a declared soft-contract path emits the advisory warning AND its witness
// command, yet exits SUCCESS (0) — warning mode never blocks. The message is clean so the lint
// verdict itself is OK; the only output beyond the OK line is the advisory warning.
func TestRunCommitPreview_softContractWarnsButExitsZero(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(canon): tighten the schema (fak canon)",
		"--path", "internal/canon/schema.go",
	})
	if code != 0 {
		t.Fatalf("warning mode must exit 0, got %d (out=%q err=%q)", code, out.String(), errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "commit-preview OK") {
		t.Errorf("want the clean lint OK line, got:\n%s", s)
	}
	if !strings.Contains(s, "core-lock advisory:") {
		t.Errorf("want the advisory warning header, got:\n%s", s)
	}
	if !strings.Contains(s, "soft-contract") {
		t.Errorf("want the soft-contract class named, got:\n%s", s)
	}
	if !strings.Contains(s, "witness to clear:") || !strings.Contains(s, "go test ./internal/corelockaudit/") {
		t.Errorf("want the witness command named, got:\n%s", s)
	}
}

// TestRunCommitPreview_ordinaryLeafNotWarned is the negative fixture the ticket requires: an
// ordinary leaf edit (no declared glob claims it) produces NO core-lock warning — the advisory
// surface stays quiet — while the preview still exits 0.
func TestRunCommitPreview_ordinaryLeafNotWarned(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(gateway): add the reclaim path (fak gateway)",
		"--path", "internal/gateway/server.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (out=%q)", code, out.String())
	}
	if strings.Contains(out.String(), "core-lock advisory:") {
		t.Errorf("ordinary leaf edit must NOT be warned; got:\n%s", out.String())
	}
}

// TestRunCommitPreview_jsonCarriesCoreLockWarnings confirms the preview --json shape exposes the
// advisory rows for later metrics, with mode=warning, while still exiting 0.
func TestRunCommitPreview_jsonCarriesCoreLockWarnings(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--json", "--dir", tmp,
		"-m", "feat(canon): tighten the schema (fak canon)",
		"--path", "internal/canon/schema.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (err=%q)", code, errb.String())
	}
	var got struct {
		OK               bool              `json:"ok"`
		CoreLockWarnings []coreLockWarning `json:"core_lock_warnings"`
		CoreLockWarnMode string            `json:"core_lock_warn_mode"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("preview --json invalid: %v\n%s", err, out.String())
	}
	if got.CoreLockWarnMode != coreLockModeWarning {
		t.Errorf("core_lock_warn_mode = %q, want %q", got.CoreLockWarnMode, coreLockModeWarning)
	}
	if len(got.CoreLockWarnings) != 1 || got.CoreLockWarnings[0].Class != "soft-contract" {
		t.Fatalf("want one soft-contract warning in JSON, got %+v", got.CoreLockWarnings)
	}
	if len(got.CoreLockWarnings[0].Witness) == 0 {
		t.Errorf("JSON warning must carry a witness command, got %+v", got.CoreLockWarnings[0])
	}
}
