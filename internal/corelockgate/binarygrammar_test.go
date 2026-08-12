package corelockgate

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestGrammarMarkersAreSlicesOfTheRemedy is the drift tripwire the whole probe rests
// on. Both markers claim to be VERBATIM slices of CoreLockRemedyCommit; if a reword
// breaks either relation the probe silently starts answering the wrong question —
// a broken changed-marker calls every binary in the fleet STALE, and a broken anchor
// makes every stale binary UNKNOWN, i.e. silently unreportable. Neither failure is
// visible from any other test, because both markers would still match a binary built
// from the same reworded source.
func TestGrammarMarkersAreSlicesOfTheRemedy(t *testing.T) {
	for _, marker := range []string{GrammarChangedMarker, GrammarAnchorMarker} {
		if !strings.Contains(CoreLockRemedyCommit, marker) {
			t.Fatalf("marker %q is no longer a substring of CoreLockRemedyCommit:\n%s", marker, CoreLockRemedyCommit)
		}
	}
	// The anchor must be the half that predates the verb, so it cannot be a substring
	// of the changed-marker's sentence — otherwise "old fak" and "not a fak" collapse.
	if strings.Contains(GrammarAnchorMarker, GrammarChangedMarker) {
		t.Fatal("the anchor must not contain the changed-verb marker, or STALE can never be distinguished from UNKNOWN")
	}
}

// TestScanBinaryGrammarClassifies drives the three verdicts through synthetic
// executables: binary noise with the remedy embedded at a realistic offset. The STALE
// case is the pre-verb remedy exactly as it shipped from d0f14083d9 through
// 1574ce07f7 — the bytes a real stale fak.exe carries.
func TestScanBinaryGrammarClassifies(t *testing.T) {
	const preVerbRemedy = "Use a privileged maintenance path, or rerun fak commit with --core-lock-maintenance-witness <claim> after independent read-back confirms the edit."
	for _, tc := range []struct {
		name    string
		payload string
		want    BinaryGrammar
	}{
		{"current", CoreLockRemedyCommit, GrammarCurrent},
		{"stale", preVerbRemedy, GrammarStale},
		{"not a fak binary", "MZ\x90\x00 some other program entirely", GrammarUnknown},
		{"empty", "", GrammarUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScanBinaryGrammar(bytes.NewReader(synthBinary(tc.payload)))
			if err != nil {
				t.Fatalf("scan: %v", err)
			}
			if got != tc.want {
				t.Fatalf("verdict %s, want %s", got, tc.want)
			}
		})
	}
}

// TestScanBinaryGrammarFindsAMarkerSplitAcrossReads is the property that makes the
// streaming probe honest: a 70MB executable is never read in one call, and the answer
// must not depend on where the OS happened to split it. Driven through a reader that
// yields ONE BYTE at a time — the most adversarial split there is — so every marker
// straddles every possible boundary.
func TestScanBinaryGrammarFindsAMarkerSplitAcrossReads(t *testing.T) {
	got, err := ScanBinaryGrammar(&byteAtATimeReader{data: synthBinary(CoreLockRemedyCommit)})
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if got != GrammarCurrent {
		t.Fatalf("a marker split across reads must still be found, got %s", got)
	}
}

// TestScanBinaryGrammarEstablishesNothingOnAReadError pins the fail-quiet direction: a
// probe that could not finish must not report STALE. Half a file is not evidence of an
// old grammar — the verb could have been in the half that never arrived.
func TestScanBinaryGrammarEstablishesNothingOnAReadError(t *testing.T) {
	r := io.MultiReader(bytes.NewReader(synthBinary("Use a privileged maintenance path, or rerun fak commit with --core-lock-maintenance-witness <claim>")), errReader{})
	got, err := ScanBinaryGrammar(r)
	if err == nil {
		t.Fatal("a read error must be reported")
	}
	if got != GrammarUnknown {
		t.Fatalf("an unfinished scan establishes nothing, got %s", got)
	}
}

// TestStaleBinaryNoteNamesStalenessNotABadClaim is the point of the whole issue. The
// expensive part of #6005 was never the refusal — it was that the refusal read as a
// mis-spelled witness, so the maintainer re-spelled the claim instead of rebuilding the
// tool. The note must lead with the tool, name the path, and carry the cure the stale
// binary's own remedy text cannot print.
func TestStaleBinaryNoteNamesStalenessNotABadClaim(t *testing.T) {
	const p = "tools/.bin/fak.exe"
	note := StaleBinaryNote(p, GrammarStale)
	for _, want := range []string{"STALE", p, ChangedWitnessKind + ":", "ABSTAIN", "go build -o " + p + " ./cmd/fak"} {
		if !strings.Contains(note, want) {
			t.Fatalf("stale note missing %q:\n%s", want, note)
		}
	}
	// CURRENT and UNKNOWN say nothing at all: the probe never invents a skew, so a
	// caller needs no second condition to stay quiet.
	for _, g := range []BinaryGrammar{GrammarCurrent, GrammarUnknown} {
		if got := StaleBinaryNote(p, g); got != "" {
			t.Fatalf("verdict %s must produce no note, got %q", g, got)
		}
	}
}

// TestWorkerBinaryResolvesTheChangedVerb is the regression test #6005 names: pin that a
// `changed:<path>` claim naming a path in the changed set resolves CONFIRMED through
// whatever binary the repo tells workers to use.
//
// It binds BOTH halves of that sentence, because either alone is satisfiable while the
// bug is live:
//
//   - the SOURCE half — the gate this checkout compiles clears the claim (if this ever
//     regressed, a fresh binary would be just as blind as a stale one); and
//   - the ARTIFACT half — the executable the worker-launch precedence actually resolves
//     (FAK_BIN -> <root>/tools/.bin/fak[.exe] -> `fak` on PATH, mirroring
//     cmd/dispatchworker/guard.go:resolveFakBin and tools/dispatch_worker.resolve_fak_bin)
//     carries the verb at all.
//
// It runs from SOURCE, which is the one vantage point that cannot itself be stale.
// A host with no fak built anywhere SKIPS, and an executable the probe cannot classify
// (UNKNOWN: stripped, packed, or not a fak) is not accused — the probe never invents a
// skew, so this test can only go red on a POSITIVE reading of an old grammar.
func TestWorkerBinaryResolvesTheChangedVerb(t *testing.T) {
	// The source half. No resolver is registered in this test binary, which is the
	// point: `changed:` is decided from the gate's own changed pathset, so the verb
	// works in any build (changedwitness.go).
	withFactory(t, nil)
	if detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Changed: []string{lockedPath},
		Witness: ChangedWitnessKind + ":" + lockedPath,
	}); fired {
		t.Fatalf("the gate this checkout compiles must CONFIRM a changed: claim naming a changed path:\n%s", detail)
	}

	// The artifact half.
	bin := workerFakBinary(t)
	if bin == "" {
		t.Skip("no fak binary resolves on this host (FAK_BIN, tools/.bin, PATH all empty) — nothing to pin")
	}
	got, err := BinaryGrammarAt(bin)
	if err != nil {
		t.Skipf("could not read %s: %v", bin, err)
	}
	if got == GrammarStale {
		t.Fatalf("the binary workers are fronted with is behind the gate's grammar.\n%s", StaleBinaryNote(bin, got))
	}
	if got == GrammarUnknown {
		t.Skipf("%s carries neither grammar marker, so its core-lock verbs cannot be established (stripped/packed build, or not a fak)", bin)
	}
}

// workerFakBinary resolves the executable the repo tells workers to use, by the same
// precedence both dispatch launchers document. "" means this host has none.
func workerFakBinary(t *testing.T) string {
	t.Helper()
	if explicit := strings.TrimSpace(os.Getenv("FAK_BIN")); explicit != "" {
		if info, err := os.Stat(explicit); err == nil && !info.IsDir() {
			return explicit
		}
	}
	exe := "fak"
	if runtime.GOOS == "windows" {
		exe = "fak.exe"
	}
	if root := repoRoot(t); root != "" {
		intree := filepath.Join(root, "tools", ".bin", exe)
		if info, err := os.Stat(intree); err == nil && !info.IsDir() {
			return intree
		}
	}
	if p, err := exec.LookPath("fak"); err == nil {
		return p
	}
	return ""
}

// repoRoot walks up from the test's working directory to the module root (the directory
// holding go.mod). "" when there is none, which keeps the test a SKIP rather than a
// failure in a vendored or extracted tree.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if info, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !info.IsDir() {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// synthBinary wraps a payload in enough surrounding noise that the marker is found at a
// realistic offset rather than at byte 0 — including a run longer than one scan chunk,
// so the boundary logic is exercised by the ordinary cases too.
func synthBinary(payload string) []byte {
	var b bytes.Buffer
	noise := bytes.Repeat([]byte{0x00, 0x4d, 0x5a, 0xff}, (grammarScanChunk/4)+7)
	b.Write(noise)
	b.WriteString(payload)
	b.Write(noise[:1024])
	return b.Bytes()
}

// byteAtATimeReader is the worst-case chunker: every Read returns exactly one byte, so
// every marker straddles a boundary.
type byteAtATimeReader struct {
	data []byte
	i    int
}

func (r *byteAtATimeReader) Read(p []byte) (int, error) {
	if r.i >= len(r.data) {
		return 0, io.EOF
	}
	if len(p) == 0 {
		return 0, nil
	}
	p[0] = r.data[r.i]
	r.i++
	return 1, nil
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("device fell off the bus") }
