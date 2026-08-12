package main

import (
	"bytes"
	"io"
	"reflect"
	"strings"
	"testing"
)

// Regression guard for #6012. The advisory test-quality call at the push seam once passed the
// resolved repo root POSITIONALLY — runTestQuality(r, io.Discard, stderr, nil) — to a function
// whose root arrives as a `--root` flag inside argv. Because that call site lives in cmd/fak,
// the package `fak hooks pre-push` IS, the arity break disarmed the very gate meant to catch a
// red trunk. `go vet` re-catches the arity itself; what it cannot check is the part the fix has
// to keep true — that the resolved root actually travels, and that a non-zero scanner code stays
// advisory. Those are what these two pin.

// setupTestQualitySeam swaps the test-quality seam for a stub returning code, records the argv it
// was handed, and restores the real scanner at test end.
func setupTestQualitySeam(t *testing.T, code int) *[]string {
	t.Helper()
	saved := prepushTestQuality
	t.Cleanup(func() { prepushTestQuality = saved })
	var got []string
	prepushTestQuality = func(stdout, stderr io.Writer, argv []string) int {
		got = argv
		return code
	}
	return &got
}

func TestPrePushTestQualityGetsResolvedRootAndStaysAdvisory(t *testing.T) {
	setupHappyPrepushSeams(t)
	root := t.TempDir()
	// 1 = the ratchet found NEW findings beyond the baseline floor: the case that most tempts a
	// gate into refusing a push over unrelated test debt.
	gotArgv := setupTestQualitySeam(t, 1)

	var out, errOut bytes.Buffer
	code := runHooksPrePush(&out, &errOut, []string{"--root", root})

	if want := []string{"--root", root}; !reflect.DeepEqual(*gotArgv, want) {
		t.Fatalf("test-quality argv = %q, want %q — the resolved root must travel as a --root FLAG,\n"+
			"not as a positional and not left to the process CWD", *gotArgv, want)
	}
	if code != 0 {
		t.Fatalf("advisory test-quality growth changed the push decision: exit %d, want 0", code)
	}
	if !strings.Contains(errOut.String(), "WARNING: test-quality ratchet") {
		t.Fatalf("non-zero test-quality code printed no advisory WARNING; stderr = %q", errOut.String())
	}
}

// A clean ratchet must stay silent — otherwise the WARNING is unconditional noise and stops
// meaning anything on the run where it matters.
func TestPrePushTestQualityCleanIsQuiet(t *testing.T) {
	setupHappyPrepushSeams(t)
	setupTestQualitySeam(t, 0)

	var out, errOut bytes.Buffer
	if code := runHooksPrePush(&out, &errOut, []string{"--root", t.TempDir()}); code != 0 {
		t.Fatalf("clean gate exited %d, want 0", code)
	}
	if strings.Contains(errOut.String(), "WARNING: test-quality ratchet") {
		t.Fatalf("clean test-quality ratchet still warned; stderr = %q", errOut.String())
	}
}
