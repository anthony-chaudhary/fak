package dispatchaudit

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// witnessJSON renders a .witness sidecar in the EXACT shape the tick sweep
// writes (dispatchtick.WitnessRecord.Map), so the fixture tracks the real
// on-disk bytes rather than a hand-rolled approximation of them.
func witnessJSON(t *testing.T, rec dispatchtick.WitnessRecord) string {
	t.Helper()
	b, err := json.Marshal(rec.Map())
	if err != nil {
		t.Fatalf("marshal witness record: %v", err)
	}
	return string(b)
}

// TestScanDirWitnessSidecarShipsOverErrorLines pins the #4476 regression: a
// worker that wrote a CLAIM_WITNESSED .witness sidecar (a real diff-witnessed
// ship) but NO .commit sidecar — and whose log ALSO carries provider-error
// lines (an inflated ErrorLines, e.g. a route-health probe's own findings) —
// must classify SHIPPED. Workers write .witness, not .commit, so before the fix
// CommitSHA stayed empty and the session fell through the outcome switch to the
// `ErrorLines > 0` catch-all → a false ERRORED. Sourcing CommitSHA from the
// .witness sidecar promotes it at the TOP of the switch, ahead of ErrorLines.
func TestScanDirWitnessSidecarShipsOverErrorLines(t *testing.T) {
	dir := t.TempDir()
	const base = "resolve-3429-20260713-035929"
	const sha = "31c6538bd44b4536f66b9cfa724a791b9e84d841"

	// A clean session that shipped, whose log ALSO carries a provider-error token
	// (the probe's deliverable). Without the fix this reads ERRORED.
	writeFile(t, dir, base+".log",
		"# fak-spawn 20260713-035929 issue=3429 lane=cmd backend=claude argv0=claude\n"+
			"probing route health...\n"+
			"timestamp=2026-07-13T03:59:31.000Z level=ERROR provider error 503 observed by probe\n")
	// The worker writes .witness (NOT the .commit sidecar the classifier used to
	// read exclusively).
	writeFile(t, dir, base+dispatchtick.WitnessSidecarSuffix, witnessJSON(t, dispatchtick.WitnessRecord{
		Issue: 3429, Log: base + ".log", SHA: sha,
		Claim: dispatchtick.ClaimWitnessed, Verdict: "OK", Witness: dispatchtick.WitnessOK,
	}))

	workers, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	if len(workers) != 1 {
		t.Fatalf("ScanDir = %d workers, want 1", len(workers))
	}
	w := workers[0]
	if w.CommitSHA != sha {
		t.Fatalf("CommitSHA sourced from .witness = %q, want the ship SHA %q", w.CommitSHA, sha)
	}
	if w.ErrorLines == 0 {
		t.Fatalf("fixture must carry >0 ErrorLines to exercise the ERRORED misclassification the fix prevents")
	}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeShipped {
		t.Fatalf("witnessed ship classified %s (%q), want SHIPPED — the #4476 regression", c.Outcome, c.Reason)
	}
}

// TestReadWitnessSidecarSelectivity pins that ONLY an authoritative witnessed
// commit sources a ship SHA: a no-commit / unwitnessed / subject-only / non-OK
// / malformed record yields nothing, so the .witness read can never
// over-promote a non-shipping worker to SHIPPED.
func TestReadWitnessSidecarSelectivity(t *testing.T) {
	const sha = "31c6538bd44b4536f66b9cfa724a791b9e84d841"
	cases := []struct {
		name    string
		body    string
		wantSHA string
		wantOK  bool
	}{
		{"witnessed diff-witness", `{"claim":"CLAIM_WITNESSED","issue":1,"sha":"` + sha + `","verdict":"OK","witness":"diff-witnessed"}`, sha, true},
		{"no-commit", `{"claim":"CLAIM_NO_COMMIT","issue":1,"sha":null,"verdict":null,"witness":null,"reason":"usage_cap"}`, "", false},
		{"unwitnessed subject-only", `{"claim":"CLAIM_UNWITNESSED","issue":1,"sha":"` + sha + `","verdict":"OK","witness":"subject-only"}`, "", false},
		{"witnessed but verdict not OK", `{"claim":"CLAIM_WITNESSED","issue":1,"sha":"` + sha + `","verdict":"ABSTAIN","witness":"diff-witnessed"}`, "", false},
		{"claim ok but non-hex sha", `{"claim":"CLAIM_WITNESSED","issue":1,"sha":"not-a-sha","verdict":"OK","witness":"diff-witnessed"}`, "", false},
		{"malformed json", `{not json`, "", false},
		{"empty", ``, "", false},
	}
	dir := t.TempDir()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(dir, "w.witness")
			if err := os.WriteFile(p, []byte(tc.body), 0o644); err != nil {
				t.Fatalf("write fixture: %v", err)
			}
			gotSHA, gotOK := readWitnessSidecar(p)
			if gotSHA != tc.wantSHA || gotOK != tc.wantOK {
				t.Fatalf("readWitnessSidecar = (%q, %v), want (%q, %v)", gotSHA, gotOK, tc.wantSHA, tc.wantOK)
			}
		})
	}
	// A missing file (the common case: no .witness at all) is a clean miss.
	if got, ok := readWitnessSidecar(filepath.Join(dir, "absent.witness")); ok || got != "" {
		t.Fatalf("absent .witness = (%q, %v), want (\"\", false)", got, ok)
	}
}

// TestWitnessSidecarPrecedenceOverCommit pins the sidecar precedence: when BOTH
// a witnessed .witness and a legacy .commit sidecar exist, the .witness SHA
// wins; and a worker with ONLY a .commit sidecar (no .witness) still ships via
// the fallback — the fix ADDS the authoritative source, it does not drop the
// legacy one.
func TestWitnessSidecarPrecedenceOverCommit(t *testing.T) {
	dir := t.TempDir()
	const witnessSHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const commitSHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

	// Worker A: both sidecars present — .witness must win.
	baseA := "resolve-11-20260713-000000"
	writeFile(t, dir, baseA+".log", "# fak-spawn 20260713-000000 issue=11 lane=cmd backend=claude\nwork\n")
	writeFile(t, dir, baseA+dispatchtick.WitnessSidecarSuffix, witnessJSON(t, dispatchtick.WitnessRecord{
		Issue: 11, Log: baseA + ".log", SHA: witnessSHA,
		Claim: dispatchtick.ClaimWitnessed, Verdict: "OK", Witness: dispatchtick.WitnessOK,
	}))
	writeFile(t, dir, baseA+".commit", commitSHA)

	// Worker B: ONLY the legacy .commit sidecar — the fallback still ships it.
	baseB := "resolve-22-20260713-000000"
	writeFile(t, dir, baseB+".log", "# fak-spawn 20260713-000000 issue=22 lane=cmd backend=claude\nwork\n")
	writeFile(t, dir, baseB+".commit", commitSHA)

	workers, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	got := map[string]string{}
	for _, w := range workers {
		got[w.Issue] = w.CommitSHA
	}
	if got["11"] != witnessSHA {
		t.Fatalf("issue 11 CommitSHA = %q, want the .witness SHA %q (precedence over .commit)", got["11"], witnessSHA)
	}
	if got["22"] != commitSHA {
		t.Fatalf("issue 22 CommitSHA = %q, want the .commit fallback SHA %q", got["22"], commitSHA)
	}
}
