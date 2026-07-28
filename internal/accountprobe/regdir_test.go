package accountprobe

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// regFixture is a hermetic stand-in for a real host's registry layout. Every rung the
// production resolver consults is pinned inside t.TempDir() — including the per-user
// Fleet root and the Windows temp root — so no test ever stats, let alone reads, the
// operator's own fleet registry.
type regFixture struct {
	userReg     string // <tmp>/localappdata/Fleet/registry — where the prober writes its ledger
	cloneRegAbs string // <tmp>/clone/tools/_registry — the cwd-relative clone-root registry
	cloneRegRel string // the path shape RegDir actually returns for the clone rung
}

func newRegFixture(t *testing.T) regFixture {
	t.Helper()
	root := t.TempDir() // taken BEFORE TMP/TEMP are repointed, so it lands in the real temp root
	userHome := filepath.Join(root, "localappdata")
	tempHome := filepath.Join(root, "temproot")
	cloneRoot := filepath.Join(root, "clone")
	for _, d := range []string{userHome, tempHome, cloneRoot} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("FLEET_REG_DIR", "")   // the unset case this issue is about
	t.Setenv("FLEET_STATE_DIR", "") // no installed-service declaration either
	t.Setenv("LOCALAPPDATA", userHome)
	t.Setenv("TMP", tempHome)
	t.Setenv("TEMP", tempHome)
	t.Chdir(cloneRoot) // the fak binary runs from the clone root; that is the whole bug
	return regFixture{
		userReg:     filepath.Join(userHome, "Fleet", "registry"),
		cloneRegAbs: filepath.Join(cloneRoot, "tools", "_registry"),
		cloneRegRel: filepath.Join("tools", "_registry"),
	}
}

// writeRegistry lays down a registry dir. withLedger=false reproduces the exact #5390
// shape: sessions.json present, probe_ledger.jsonl absent, so no block is derivable and
// the dir reports a fleet with zero blocked seats.
func writeRegistry(t *testing.T, dir string, withLedger bool) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	sessions := `{"generated_utc":"2026-07-28T12:54:04Z","app_version":"0.41.0","sessions":[]}`
	if err := os.WriteFile(filepath.Join(dir, sessionsFile), []byte(sessions), 0o644); err != nil {
		t.Fatal(err)
	}
	if !withLedger {
		return
	}
	line := `{"ts":"2026-07-28T12:54:00Z","account":"acct-a","status":"AUTH","block_reason":"entitlement"}` + "\n"
	if err := os.WriteFile(filepath.Join(dir, ledgerFile), []byte(line), 0o644); err != nil {
		t.Fatal(err)
	}
}

func touchRegistry(t *testing.T, dir string, when time.Time) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if err := os.Chtimes(filepath.Join(dir, e.Name()), when, when); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Chtimes(dir, when, when); err != nil {
		t.Fatal(err)
	}
}

func newestMTime(t *testing.T, dir string) time.Time {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var newest time.Time
	for _, e := range entries {
		fi, err := e.Info()
		if err != nil {
			t.Fatal(err)
		}
		if fi.ModTime().After(newest) {
			newest = fi.ModTime()
		}
	}
	return newest
}

// TestRegDirPrefersLedgerBearingRegistry is the issue's headline: with FLEET_REG_DIR
// unset, the cwd-relative clone-root registry must NOT win over the per-user registry
// that actually carries the probe ledger. It drives the real RegDir() and, past it, the
// real LastProbeByAccount("") — the exact chain a production
// fleetaccounts.FreshProbeFromLedger(account, "", …) call takes.
func TestRegDirPrefersLedgerBearingRegistry(t *testing.T) {
	fx := newRegFixture(t)
	writeRegistry(t, fx.userReg, true)      // sessions + ledger: blocks derivable
	writeRegistry(t, fx.cloneRegAbs, false) // sessions only: block-blind

	got := RegDir()
	if got != fx.userReg {
		t.Fatalf("RegDir() = %q, want the ledger-bearing per-user registry %q (the block-blind clone root is %q)",
			got, fx.userReg, fx.cloneRegRel)
	}
	c := ResolveRegDir()
	if c.Rung != RungUser {
		t.Fatalf("ResolveRegDir().Rung = %q, want %q", c.Rung, RungUser)
	}
	if c.Health != RegHealthBlocksKnown || !c.BlocksDerivable() {
		t.Fatalf("chosen health = %q / BlocksDerivable = %v, want %q / true",
			c.Health, c.BlocksDerivable(), RegHealthBlocksKnown)
	}

	// Non-vacuity past the resolver: the default-path ledger read must now SEE the block
	// the per-user registry carries. Under the old cwd-relative default this map is empty
	// and every seat reads as unblocked.
	entry, ok := LastProbeByAccount("")["acct-a"]
	if !ok {
		t.Fatalf(`LastProbeByAccount("") has no acct-a entry — the default path is still reading a ledger-less registry`)
	}
	if entry.Status != "AUTH" {
		t.Fatalf("acct-a probe status = %q, want AUTH (the entitlement block the ledger carries)", entry.Status)
	}
}

// TestRegDirIgnoresMtime pins the "authority, not mtime" rule. Two writers run on their
// own schedules, so each is the freshest for part of every minute; a recency rule would
// make the chosen registry flap between ticks. Making the WRONG registry the freshest
// thing on disk by 72 hours must change nothing.
func TestRegDirIgnoresMtime(t *testing.T) {
	fx := newRegFixture(t)
	writeRegistry(t, fx.userReg, true)
	writeRegistry(t, fx.cloneRegAbs, false)

	first := RegDir()
	if first != fx.userReg {
		t.Fatalf("RegDir() = %q, want %q", first, fx.userReg)
	}

	now := time.Now()
	touchRegistry(t, fx.userReg, now.Add(-72*time.Hour)) // authoritative registry: stale
	touchRegistry(t, fx.cloneRegAbs, now)                // block-blind registry: freshest
	t.Logf("authoritative (user) registry newest mtime: %s", newestMTime(t, fx.userReg).UTC().Format(time.RFC3339))
	t.Logf("block-blind (clone) registry newest mtime: %s", newestMTime(t, fx.cloneRegAbs).UTC().Format(time.RFC3339))
	if !newestMTime(t, fx.cloneRegAbs).After(newestMTime(t, fx.userReg)) {
		t.Fatal("fixture failed to make the block-blind registry the freshest; the mtime claim would be vacuous")
	}

	for i := 0; i < 5; i++ {
		if got := RegDir(); got != first {
			t.Fatalf("call %d: RegDir() = %q, want the stable %q — resolution must not follow mtime", i, got, first)
		}
	}
	t.Logf("RegDir() stable across 5 calls with the wrong dir freshest: %s", first)
}

// TestRegDirForkIsObservable covers the issue's real complaint: a second registry
// appearing silently. The fork signal must fire when two registries carry state and stay
// quiet when only one does.
func TestRegDirForkIsObservable(t *testing.T) {
	t.Run("two registries fork", func(t *testing.T) {
		fx := newRegFixture(t)
		writeRegistry(t, fx.userReg, true)
		writeRegistry(t, fx.cloneRegAbs, false)

		c := ResolveRegDir()
		if !c.Forked {
			t.Fatalf("Forked = false with two registries carrying state; sites = %+v", c.Sites)
		}
		note := c.ForkNote()
		if note == "" {
			t.Fatal("ForkNote() = \"\" on a forked host")
		}
		for _, want := range []string{fx.userReg, fx.cloneRegRel, string(RegHealthBlocksUnknown), RungClone} {
			if !strings.Contains(note, want) {
				t.Fatalf("ForkNote() = %q, missing %q", note, want)
			}
		}
		t.Logf("fork note: %s", note)
	})

	t.Run("one registry does not fork", func(t *testing.T) {
		fx := newRegFixture(t)
		writeRegistry(t, fx.userReg, true) // the clone root is left absent

		c := ResolveRegDir()
		if c.Forked {
			t.Fatalf("Forked = true with a single registry; sites = %+v", c.Sites)
		}
		if note := c.ForkNote(); note != "" {
			t.Fatalf("ForkNote() = %q on an unforked host, want \"\"", note)
		}
	})

	// The note has to reach a human, not just a struct field: RegDir() itself emits it
	// once per process on stderr, so a real `fak resume resolve` run reports the fork with
	// no extra wiring anywhere else.
	t.Run("RegDir emits the note on stderr", func(t *testing.T) {
		fx := newRegFixture(t)
		writeRegistry(t, fx.userReg, true)
		writeRegistry(t, fx.cloneRegAbs, false)

		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		saved := os.Stderr
		os.Stderr = w
		dir := RegDir()
		os.Stderr = saved
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		out, err := io.ReadAll(r)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "FORKED") || !strings.Contains(string(out), fx.cloneRegRel) {
			t.Fatalf("RegDir() stderr = %q, want a FORKED note naming the second registry %q", string(out), fx.cloneRegRel)
		}
		if dir != fx.userReg {
			t.Fatalf("RegDir() = %q, want %q", dir, fx.userReg)
		}
		t.Logf("stderr: %s", strings.TrimSpace(string(out)))
	})
}

// TestRegistryWithoutLedgerIsUnknownNotZeroBlocks is the "absence is not neutral" rule.
// A registry holding sessions.json with no ledger beside it cannot derive a block, so it
// must grade unknown — a third state, distinct from both "healthy, zero blocked" and
// "blocked". Grading it blocked would strand seats; grading it healthy is the bug.
func TestRegistryWithoutLedgerIsUnknownNotZeroBlocks(t *testing.T) {
	fx := newRegFixture(t)
	writeRegistry(t, fx.cloneRegAbs, false) // the only registry on this host is block-blind

	c := ResolveRegDir()
	if c.Dir != fx.cloneRegRel {
		t.Fatalf("ResolveRegDir().Dir = %q, want the only registry present %q", c.Dir, fx.cloneRegRel)
	}
	if c.Health != RegHealthBlocksUnknown {
		t.Fatalf("health = %q, want %q for sessions.json with no ledger beside it", c.Health, RegHealthBlocksUnknown)
	}
	if c.BlocksDerivable() {
		t.Fatal("BlocksDerivable() = true for a ledger-less registry: it would report zero blocked seats as a finding")
	}
	if c.Forked {
		t.Fatal("Forked = true with a single registry")
	}

	// The contrast that makes "unknown" mean something: the SAME dir, once a ledger lands
	// beside its sessions.json, grades derivable — so zero-blocked from it is a finding.
	writeRegistry(t, fx.cloneRegAbs, true)
	c2 := ResolveRegDir()
	if c2.Health != RegHealthBlocksKnown || !c2.BlocksDerivable() {
		t.Fatalf("health after the ledger landed = %q / BlocksDerivable = %v, want %q / true",
			c2.Health, c2.BlocksDerivable(), RegHealthBlocksKnown)
	}
	if c2.Dir != c.Dir {
		t.Fatalf("chosen dir moved from %q to %q when only the ledger appeared", c.Dir, c2.Dir)
	}
}

// TestRegDirHonorsExplicitOverrides keeps the two DECLARED rungs outranking discovery: an
// operator (or the installed service) naming the dir is the highest authority there is,
// and every existing fleet wiring depends on FLEET_REG_DIR still winning outright.
func TestRegDirHonorsExplicitOverrides(t *testing.T) {
	fx := newRegFixture(t)
	writeRegistry(t, fx.userReg, true)
	writeRegistry(t, fx.cloneRegAbs, false)

	named := filepath.Join(t.TempDir(), "named-registry")
	t.Setenv("FLEET_REG_DIR", named)
	if got := RegDir(); got != named {
		t.Fatalf("RegDir() = %q, want the explicit FLEET_REG_DIR %q even with no state under it", got, named)
	}
	if got := ProbeLedgerPath(""); got != filepath.Join(named, ledgerFile) {
		t.Fatalf("ProbeLedgerPath(\"\") = %q, want it under the explicit override", got)
	}

	stateRoot := t.TempDir()
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FLEET_STATE_DIR", stateRoot)
	if got, want := RegDir(), filepath.Join(stateRoot, "registry"); got != want {
		t.Fatalf("RegDir() = %q, want the service-declared %q", got, want)
	}
}

// TestRegDirFallsBackToCloneRootWhenNothingExists pins the conservative floor: on a fresh
// checkout or in CI, where no registry exists anywhere, the answer is the pre-existing
// cwd-relative default, unchanged.
func TestRegDirFallsBackToCloneRootWhenNothingExists(t *testing.T) {
	fx := newRegFixture(t)
	c := ResolveRegDir()
	if c.Dir != fx.cloneRegRel {
		t.Fatalf("ResolveRegDir().Dir = %q with an empty host, want the legacy default %q", c.Dir, fx.cloneRegRel)
	}
	if c.Rung != RungClone || c.Health != RegHealthEmpty {
		t.Fatalf("rung/health = %q/%q, want %q/%q", c.Rung, c.Health, RungClone, RegHealthEmpty)
	}
	if c.Forked || c.ForkNote() != "" {
		t.Fatalf("Forked = %v / note = %q on an empty host", c.Forked, c.ForkNote())
	}
	if got := RegDir(); got != fx.cloneRegRel {
		t.Fatalf("RegDir() = %q, want %q", got, fx.cloneRegRel)
	}
}

// TestRegSitesDeduplicateSamePath keeps the ordinary production wiring — FLEET_REG_DIR
// pointing AT the per-user Fleet registry — from reading as a fork of itself.
func TestRegSitesDeduplicateSamePath(t *testing.T) {
	fx := newRegFixture(t)
	writeRegistry(t, fx.userReg, true)
	t.Setenv("FLEET_REG_DIR", fx.userReg)

	c := ResolveRegDir()
	if c.Forked {
		t.Fatalf("Forked = true when FLEET_REG_DIR names the same dir as the user rung; sites = %+v", c.Sites)
	}
	seen := 0
	for _, s := range c.Sites {
		if s.Dir == fx.userReg {
			seen++
		}
	}
	if seen != 1 {
		t.Fatalf("per-user registry appears %d times in the survey, want 1 (deduplicated)", seen)
	}
	if c.Rung != RungEnv {
		t.Fatalf("Rung = %q, want %q (the highest-authority rung naming the dir)", c.Rung, RungEnv)
	}
}
