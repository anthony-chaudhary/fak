package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// redirectTempDir points os.TempDir() at a per-test root so these cases never create (or
// sweep) entries in the machine's shared temp dir, which live peer processes use. The
// assertions below never depend on the redirect having taken effect — they compare against
// os.TempDir() as observed at call time — so a platform where it does not stick loses only
// the tidiness, never the witness.
func redirectTempDir(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	t.Setenv("TMPDIR", root)
	t.Setenv("TMP", root)
	t.Setenv("TEMP", root)
	if got := os.TempDir(); !sameReplayAuditPath(got, root) {
		t.Logf("note: os.TempDir() = %s did not follow the redirect to %s; the test still holds, it just writes to the shared root", got, root)
	}
	return root
}

// TestGuardReplayDefaultAuditPathIsReservedNotJustNamed is the #5524 witness, and it is
// deterministic: the generated default must be RESERVED on disk by the call that hands it
// out, not merely NAMED in a way that is unlikely to repeat.
//
// The unfixed default was
//
//	filepath.Join(os.TempDir(), fmt.Sprintf("fak-guard-replay-%d.jsonl", time.Now().UnixNano()))
//
// — a bare name in the SHARED temp root derived from the wall clock alone, with no PID and
// no randomness. time.Now()'s wall component comes off a coarse system timer (on Windows the
// granularity is milliseconds, not nanoseconds), so UnixNano() returns the same value for
// calls far apart in real time and two `fak guard replay` runs that start inside one tick
// resolve to the SAME path. Nothing then fails loudly: internal/journal.Open opens
// O_CREATE|O_WRONLY|O_APPEND (internal/journal/journal.go:198), so the second run INTERLEAVES
// its hash-chained rows into the first run's file. The damage surfaces one layer away as
// `journal: sequence gap: seq=1 want 3` (internal/journal/journal.go:945) — a replay-INTEGRITY
// complaint that blames the trace while every per-call row still prints ✓ — so it reads as
// flake. Pass-alone / fail-together is the signature.
//
// Asserting "the two names differ" would only be a probabilistic witness. Asserting the name
// is reserved is the real property and it holds every run: a name that exists only in memory
// can still be handed to a second process, whereas a directory created with O_EXCL cannot.
func TestGuardReplayDefaultAuditPathIsReservedNotJustNamed(t *testing.T) {
	redirectTempDir(t)

	first, optedOut, requested := guardReplayAuditPlan("", false)
	if optedOut || requested {
		t.Fatalf("default plan: optedOut=%v requested=%v, want false/false", optedOut, requested)
	}
	second, _, _ := guardReplayAuditPlan("", false)

	for i, p := range []string{first, second} {
		if p == "" {
			t.Fatalf("default %d: empty path", i)
		}
		dir := filepath.Dir(p)
		if sameReplayAuditPath(dir, os.TempDir()) {
			t.Fatalf("default %d = %s sits directly in the shared temp root %s: the name is only UNLIKELY to repeat, never claimed. "+
				"A concurrent replay that starts in the same clock tick gets the same path and journal.Open appends into it "+
				"instead of failing, which resurfaces as a bogus `journal: sequence gap`.", i, p, os.TempDir())
		}
		st, err := os.Stat(dir)
		if err != nil || !st.IsDir() {
			t.Fatalf("default %d: containing dir %s was not created by the call that returned it (stat err=%v): "+
				"nothing on disk claims the name, so a peer can still be handed it", i, dir, err)
		}
	}

	if sameReplayAuditPath(filepath.Dir(first), filepath.Dir(second)) {
		t.Fatalf("two successive defaults shared the reservation dir %s — the reservation is not per-call", filepath.Dir(first))
	}

	// The default must stay recognizable to an operator (and to
	// TestGuardReplayRunsCleanOnBothWires, which greps the report for it).
	if !strings.Contains(strings.ToLower(first), "fak-guard-replay-") {
		t.Fatalf("default %s no longer carries the fak-guard-replay- marker operators grep for", first)
	}
}

// TestGuardReplayDefaultAuditPathIsUniquePerCall is the volume half of the #5524 witness: a
// tight burst of defaults — the shape a fleet of replays starting together produces — must
// yield no duplicates at all. Under the clock-only default this burst runs entirely inside
// one timer tick and collapses to a handful of distinct names (often exactly one).
func TestGuardReplayDefaultAuditPathIsUniquePerCall(t *testing.T) {
	redirectTempDir(t)

	const n = 64
	seen := map[string]string{}
	for i := 0; i < n; i++ {
		p, _, _ := guardReplayAuditPlan("", false)
		key := strings.ToLower(filepath.Clean(p))
		if prev, dup := seen[key]; dup {
			t.Fatalf("default audit path collided on call %d: %s was already handed out as %s. "+
				"Two replays that resolve to one journal file interleave (journal.Open is O_APPEND) "+
				"rather than fail, and the corruption is reported as a sequence gap in the trace.", i, p, prev)
		}
		seen[key] = p
	}
	if len(seen) != n {
		t.Fatalf("got %d distinct defaults across %d calls, want %d", len(seen), n, n)
	}
}

// TestGuardReplayAuditPlanExplicitAndOffUnchanged pins the two branches #5524 must NOT
// touch: an operator-supplied --audit PATH is still returned verbatim (only trimmed) and
// still marked requested, and both opt-out spellings still return the empty path. The fix is
// only about the generated default, so these must be byte-identical to before.
func TestGuardReplayAuditPlanExplicitAndOffUnchanged(t *testing.T) {
	root := redirectTempDir(t)
	explicit := filepath.Join(root, "operator-chosen", "audit.jsonl")

	p, optedOut, requested := guardReplayAuditPlan(explicit, false)
	if p != explicit || optedOut || !requested {
		t.Fatalf("explicit --audit %q -> (%q, optedOut=%v, requested=%v), want (%q, false, true)",
			explicit, p, optedOut, requested, explicit)
	}
	// The plan must not pre-create anything for an explicit path: journal.Enable owns that
	// (it MkdirAlls the parent), and silently materializing an operator's directory here
	// would change what a bad --audit reports.
	if _, err := os.Stat(filepath.Dir(explicit)); !os.IsNotExist(err) {
		t.Fatalf("explicit --audit parent %s was created by the plan (stat err=%v), want untouched", filepath.Dir(explicit), err)
	}

	if p, _, _ := guardReplayAuditPlan("   "+explicit+"\t", false); p != explicit {
		t.Fatalf("padded explicit --audit -> %q, want the trimmed original %q", p, explicit)
	}

	for _, off := range []string{"off", "OFF", "Off", "  off  "} {
		p, optedOut, requested := guardReplayAuditPlan(off, false)
		if p != "" || !optedOut || requested {
			t.Errorf("--audit %q -> (%q, optedOut=%v, requested=%v), want (\"\", true, false)", off, p, optedOut, requested)
		}
	}

	for _, in := range []string{"", explicit, "off"} {
		p, optedOut, requested := guardReplayAuditPlan(in, true)
		if p != "" || !optedOut || requested {
			t.Errorf("--no-audit with --audit %q -> (%q, optedOut=%v, requested=%v), want (\"\", true, false)", in, p, optedOut, requested)
		}
	}

	// A no-audit / off plan must not leave a reservation behind either.
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read temp root: %v", err)
	}
	for _, e := range entries {
		if strings.HasPrefix(strings.ToLower(e.Name()), "fak-guard-replay-") {
			t.Fatalf("opt-out/explicit plans left a generated reservation %s in the temp root", e.Name())
		}
	}
}
