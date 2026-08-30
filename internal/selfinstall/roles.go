package selfinstall

// roles.go — the CANONICAL deployed `fak` binary per role, and the audit that says whether
// the copies actually on disk agree (#6508).
//
// A host does not run "the fak binary"; it runs several, and every consumer resolves its own
// by a different rule. The audit that opened #6508 found four live at once:
//
//	repo root  C:\work\fak\fak.exe            e5fc01af20cd +uncommitted   <- adjudicates the spawn gate
//	worker     <repo>\tools\.bin\fak.exe      0c96937b61ac                <- fronts every dispatched worker
//	PATH       C:\Users\USER\bin\fak.exe      0c96937b61ac                <- the installed fleet binary
//	Go bin     C:\Users\USER\go\bin\fak.exe   7298f8f2abbb                <- what scheduled tasks executed
//
// Three different builds, one of them an unreviewed working-tree compile, and NOTHING on the
// host could name that set: `self-update --target X` converged X (plus two ad-hoc siblings) and
// reported success, so a half-converged host was indistinguishable from a converged one.
//
// This file is the missing declaration. Roles() names ONE canonical path per role — derived
// from the repo root, the home dir, and the invoking (scheduled) binary, and NEVER from an
// ambient PATH lookup, so the table cannot shift under a changed PATH. Census() says what each
// of those files actually is, AuditCopies() says whether they agree, and Convergeable() says
// which ones an unattended self-update may swap. The rest — the roles it may NOT swap — must
// still be reported, because "converge or explicitly audit EVERY configured hot copy" is the
// contract; silently skipping one is the failure this exists to end.

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Role names a deployment slot: the JOB a fak binary does on this host, independent of which
// file currently fills it. Roles are the unit of convergence — "is the fleet converged?" is
// "does every role's canonical copy hold the same build?".
type Role string

const (
	// RoleGate is <repo>/fak[.exe]: the hand-built repo-root binary tools/dispatch_preflight.py
	// `_fak_command` picks to ADJUDICATE the spawn gate. Routinely a `+uncommitted` working-tree
	// compile, which is why it is audit-only (see Convergeable) rather than auto-swapped.
	RoleGate Role = "gate"
	// RoleWorker is <repo>/tools/.bin/fak[.exe]: the in-tree binary
	// tools/dispatch_worker.py `resolve_fak_bin` prefers AHEAD of PATH when it builds every
	// dispatched worker's `fak guard --` argv. While that file exists PATH is never consulted.
	RoleWorker Role = "worker"
	// RolePath is <home>/bin/fak[.exe]: the installed fleet binary the guard fleet launches and
	// the scheduled self-update converges by default.
	RolePath Role = "path"
	// RoleGoBin is <home>/go/bin/fak[.exe]: whatever `go install` last left behind. Nothing
	// rebuilds it, so it goes stale silently — and scheduled tasks registered against it keep
	// executing that stale copy (the logvault capture failures in #6508).
	RoleGoBin Role = "gobin"
	// RoleScheduled is the binary a scheduled task pinned at REGISTRATION time and re-executes
	// every tick. It is frozen at whatever was on disk that day unless something converges it.
	RoleScheduled Role = "scheduled"
)

// Host roots the role table. Every path is derived from these three inputs and nothing else —
// in particular never from `exec.LookPath("fak")`, so "which binary is role X?" has one answer
// that does not depend on the environment a tick happened to inherit.
type Host struct {
	RepoRoot  string // the checkout (RoleGate, RoleWorker)
	Home      string // the user home dir (RolePath, RoleGoBin)
	Scheduled string // the invoking/scheduled binary, usually os.Executable() (RoleScheduled)
}

// HotCopy is one role's canonical path plus what the file at that path actually is. A copy that
// is not Present carries no build; a copy that is Present but not Attested is a binary that
// could not say which commit it is — never coerced to "clean", because reporting an
// unreviewable build as reviewed is the exact lie this audit exists to prevent.
type HotCopy struct {
	Role     Role
	Path     string
	Present  bool   // a file exists at Path
	Build    string // the VCS revision the binary self-reports
	Dirty    bool   // the binary self-reports a `+uncommitted` working-tree build
	Attested bool   // the binary produced a usable build id at all
	Err      string // why the build could not be read, when it could not
}

// StampProbe reads a deployed binary's embedded VCS provenance, normally by running
// `<path> version` and parsing the `build: <rev>[ +uncommitted]` line. ok=false means the
// binary would not run or printed no stamp; the caller treats that as "cannot prove current",
// never as "current".
type StampProbe func(path string) (build string, dirty bool, ok bool)

// Roles returns the canonical hot copy for every role, in a stable order, with Present resolved
// by stat. A role whose root is empty (no home dir, no scheduled binary) is omitted rather than
// invented, so the table never claims an install location that does not exist.
func Roles(h Host) []HotCopy {
	exe := "fak" + binExt()
	plan := []struct {
		role Role
		path string
	}{
		{RoleGate, under(h.RepoRoot, exe)},
		{RoleWorker, under(h.RepoRoot, "tools", ".bin", exe)},
		{RolePath, under(h.Home, "bin", exe)},
		{RoleGoBin, under(h.Home, "go", "bin", exe)},
		{RoleScheduled, strings.TrimSpace(h.Scheduled)},
	}
	out := make([]HotCopy, 0, len(plan))
	for _, p := range plan {
		if p.path == "" {
			continue
		}
		abs, err := filepath.Abs(filepath.Clean(p.path))
		if err != nil {
			abs = filepath.Clean(p.path)
		}
		c := HotCopy{Role: p.role, Path: abs}
		if st, serr := os.Stat(abs); serr == nil && !st.IsDir() {
			c.Present = true
		}
		out = append(out, c)
	}
	return out
}

// Census is Roles plus the build each present copy self-reports. A nil probe skips the stamp
// read entirely (the enumeration-only case: which paths exist, no process spawns).
//
// Each DISTINCT path is probed once even when two roles name it: the scheduled task usually
// executes the PATH copy, and `<bin> version` is a process spawn we should not pay twice.
func Census(h Host, probe StampProbe) []HotCopy {
	out := Roles(h)
	if probe == nil {
		return out
	}
	seen := map[string]HotCopy{}
	for i, c := range out {
		if !c.Present {
			continue
		}
		key := strings.ToLower(c.Path)
		if done, ok := seen[key]; ok {
			out[i].Build, out[i].Dirty, out[i].Attested, out[i].Err = done.Build, done.Dirty, done.Attested, done.Err
			continue
		}
		build, dirty, ok := probe(c.Path)
		build = strings.TrimSpace(build)
		out[i].Build, out[i].Dirty = build, dirty
		out[i].Attested = ok && build != ""
		if !out[i].Attested {
			out[i].Err = "binary reports no usable VCS stamp"
		}
		seen[key] = out[i]
	}
	return out
}

// Audit is the verdict over a census: which roles are off the reference build, which are
// running unreviewed code, and which cannot attest at all.
type Audit struct {
	Copies     []HotCopy
	Want       string   // the build every copy should hold (the reference)
	Builds     []string // distinct attested builds in play, sorted
	Converged  bool     // every present copy is attested, clean, and on Want
	Divergent  []Role   // present + attested, but on a different build than Want
	Dirty      []Role   // present, self-reporting `+uncommitted`
	Unattested []Role   // present, but could not say which commit it is
	Missing    []Role   // no file at the canonical path
}

// Drift is one repair scope's unsafe portion of an Audit. A role may appear in both Dirty and
// Divergent: dirty says its provenance is unreviewable, while divergent says the stamped base
// revision is not Want. Missing roles are intentionally absent because AuditCopies reports but
// does not create install locations.
type Drift struct {
	Divergent  []Role
	Dirty      []Role
	Unattested []Role
}

// Present reports whether this scope contains any installed copy that is not clean, attested,
// and on the reference build.
func (d Drift) Present() bool {
	return len(d.Divergent) != 0 || len(d.Dirty) != 0 || len(d.Unattested) != 0
}

// AuditPartition separates drift self-update can repair from drift it may only report. It does
// not weaken Audit.Converged: that remains the strict all-role safety verdict used by admission.
// The partition exists because successful automatic convergence and safe gate admission answer
// different questions.
type AuditPartition struct {
	Convergeable Drift
	AuditOnly    Drift
}

// Partition classifies every unsafe role by the same Convergeable policy used to select swap
// targets. This keeps updater posture aligned with what the installer can actually change.
func (a Audit) Partition() AuditPartition {
	var p AuditPartition
	partition := func(roles []Role, writable, auditOnly *[]Role) {
		for _, role := range roles {
			if Convergeable(role) {
				*writable = append(*writable, role)
			} else {
				*auditOnly = append(*auditOnly, role)
			}
		}
	}
	partition(a.Divergent, &p.Convergeable.Divergent, &p.AuditOnly.Divergent)
	partition(a.Dirty, &p.Convergeable.Dirty, &p.AuditOnly.Dirty)
	partition(a.Unattested, &p.Convergeable.Unattested, &p.AuditOnly.Unattested)
	return p
}

// AuditCopies grades a census against a reference build. want is normally origin/main's
// revision; when it is empty the reference is inferred as the most common attested build (ties
// broken lexicographically so the verdict is deterministic), which lets a caller with no git
// context still ask the weaker question "do these agree with each other?".
//
// A MISSING copy is not divergence — we converge binaries, never create install locations — so
// it is reported in its own bucket and does not by itself make a host unconverged.
func AuditCopies(copies []HotCopy, want string) Audit {
	a := Audit{Copies: copies, Want: strings.TrimSpace(want)}
	counts := map[string]int{}
	for _, c := range copies {
		if !c.Present {
			a.Missing = append(a.Missing, c.Role)
			continue
		}
		if !c.Attested {
			a.Unattested = append(a.Unattested, c.Role)
			continue
		}
		if c.Dirty {
			a.Dirty = append(a.Dirty, c.Role)
		}
		counts[c.Build]++
	}
	for b := range counts {
		a.Builds = append(a.Builds, b)
	}
	sort.Strings(a.Builds)
	if a.Want == "" {
		best, bestN := "", 0
		for _, b := range a.Builds { // a.Builds is sorted, so ties resolve lexicographically
			if counts[b] > bestN {
				best, bestN = b, counts[b]
			}
		}
		a.Want = best
	}
	for _, c := range copies {
		if c.Present && c.Attested && !sameRev(c.Build, a.Want) {
			a.Divergent = append(a.Divergent, c.Role)
		}
	}
	a.Converged = len(a.Divergent) == 0 && len(a.Dirty) == 0 && len(a.Unattested) == 0
	return a
}

// Lines renders the audit as one operator-facing line per hot copy plus a verdict line. This is
// the "explicitly audits every configured hot copy" half of the contract: a role self-update may
// not swap still has to appear here, named, with the build it is stuck on.
func (a Audit) Lines() []string {
	want := a.Want
	if want == "" {
		want = "(unknown)"
	}
	out := make([]string, 0, len(a.Copies)+1)
	for _, c := range a.Copies {
		switch {
		case !c.Present:
			out = append(out, "hot-copy role="+string(c.Role)+" path="+c.Path+" MISSING (no binary installed for this role)")
		case !c.Attested:
			out = append(out, "hot-copy role="+string(c.Role)+" path="+c.Path+" UNATTESTED ("+orElse(c.Err, "no VCS stamp")+")")
		default:
			state := "converged"
			if !sameRev(c.Build, a.Want) {
				state = "DIVERGENT (want " + shortRev(want) + ")"
			}
			if c.Dirty {
				state = "DIRTY — unreviewed working-tree build; " + state
			}
			if !Convergeable(c.Role) {
				state += " [audit-only role: self-update never swaps it]"
			}
			out = append(out, "hot-copy role="+string(c.Role)+" path="+c.Path+" build="+shortRev(c.Build)+" "+state)
		}
	}
	verdict := "hot-copy audit: CONVERGED on " + shortRev(want)
	if !a.Converged {
		verdict = "hot-copy audit: NOT CONVERGED — want " + shortRev(want) +
			"; divergent=" + roleList(a.Divergent) +
			" dirty=" + roleList(a.Dirty) +
			" unattested=" + roleList(a.Unattested)
	}
	return append(out, verdict)
}

// Convergeable reports whether an UNATTENDED self-update may swap this role's copy.
//
// RoleGate is deliberately excluded. The repo-root binary is a developer hand-build in a shared,
// permanently-dirty checkout and may be held open by a live session; overwriting it unattended
// would destroy work-in-progress a maintainer is mid-test on. It is instead audited every tick,
// and the spawn gate refuses to let a dirty/divergent gate binary adjudicate at all — which is
// the durable fix, since converging it would only paper over the next hand-build.
func Convergeable(r Role) bool {
	switch r {
	case RoleWorker, RolePath, RoleGoBin, RoleScheduled:
		return true
	default:
		return false
	}
}

// ConvergeTargets lists the DISTINCT existing paths an unattended self-update may swap, in role
// order, excluding any path equal to skip (normally the primary --target, already installed).
// Deduped case-insensitively so a Windows host never swaps the same file twice in one tick.
func ConvergeTargets(copies []HotCopy, skip string) []string {
	seen := map[string]bool{}
	if s := strings.TrimSpace(skip); s != "" {
		if abs, err := filepath.Abs(filepath.Clean(s)); err == nil {
			seen[strings.ToLower(abs)] = true
		}
	}
	out := []string{}
	for _, c := range copies {
		if !c.Present || !Convergeable(c.Role) {
			continue
		}
		key := strings.ToLower(c.Path)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, c.Path)
	}
	return out
}

// NeedsConverge reports whether the copy at path still needs a gated swap to reach want.
// It demands PROOF to skip: a path the census does not know, one whose binary could not attest
// its build, one running an `+uncommitted` compile, and one on a different revision all converge.
// Only a present, attested, clean copy already on want is left alone — because "we could not
// tell" is precisely the state that let a stale fleet binary survive every tick.
func NeedsConverge(copies []HotCopy, path, want string) bool {
	abs := strings.TrimSpace(path)
	if a, err := filepath.Abs(filepath.Clean(abs)); err == nil {
		abs = a
	}
	for _, c := range copies {
		if !strings.EqualFold(c.Path, abs) {
			continue
		}
		if !c.Present || !c.Attested || c.Dirty {
			return true
		}
		return !sameRev(c.Build, want)
	}
	return true // not in the census at all — never assume a binary we did not look at is current
}

// Pin is the binary provenance a scheduled-task REGISTRATION reviewed and froze: the absolute
// path the task will execute every tick, and the build that path held when a human approved it.
//
// The pin is on the PATH, not on the build id: convergence is supposed to advance the build at
// that path, so pinning the id would make every post-update tick look skewed. What must not
// drift is WHICH file the scheduler runs, and whether that file is still an attestable,
// reviewed build rather than someone's working-tree compile.
type Pin struct {
	Path  string // the absolute binary path reviewed at registration
	Build string // the build it held then (advisory: it advances as self-update converges)
}

// PinSkew reports whether the binary actually executing has drifted from what the scheduled task
// pinned, and why — the "detects skew BEFORE execution" half of the registration contract. An
// empty Pin.Path means nothing was pinned, which is itself skew: an unpinned scheduled task is
// how a stale Go-bin copy ended up certifying evidence in #6508.
func PinSkew(pin Pin, actual HotCopy) (bool, string) {
	pinned := strings.TrimSpace(pin.Path)
	if pinned == "" {
		return true, "scheduled task pinned no binary provenance: it will execute whatever `fak` its registration happened to resolve, so a stale or unreviewed copy can adjudicate unnoticed"
	}
	if abs, err := filepath.Abs(filepath.Clean(pinned)); err == nil {
		pinned = abs
	}
	got := strings.TrimSpace(actual.Path)
	if abs, err := filepath.Abs(filepath.Clean(got)); err == nil {
		got = abs
	}
	if !strings.EqualFold(pinned, got) {
		return true, "executing binary " + got + " is not the pinned one " + pinned +
			" — the reviewed provenance does not describe the binary about to run"
	}
	if !actual.Present {
		return true, "pinned binary " + pinned + " is no longer on disk"
	}
	if !actual.Attested {
		return true, "pinned binary " + pinned + " reports no VCS stamp, so it cannot attest which commit it is" +
			pinnedAt(pin.Build)
	}
	if actual.Dirty {
		return true, "pinned binary " + pinned + " now holds an +uncommitted working-tree build (" +
			shortRev(actual.Build) + "), which no commit reviews" + pinnedAt(pin.Build)
	}
	return false, ""
}

func pinnedAt(build string) string {
	if b := strings.TrimSpace(build); b != "" {
		return " (registration pinned " + shortRev(b) + ")"
	}
	return ""
}

// sameRev compares two VCS revisions tolerantly: `fak version` may print a short rev while git
// hands back the full 40-hex one, so a prefix match of at least 7 hex chars counts as the same
// commit. Anything shorter is treated as a mismatch rather than guessed at.
func sameRev(a, b string) bool {
	a, b = strings.ToLower(strings.TrimSpace(a)), strings.ToLower(strings.TrimSpace(b))
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	short, long := a, b
	if len(short) > len(long) {
		short, long = long, short
	}
	return len(short) >= 7 && strings.HasPrefix(long, short)
}

func shortRev(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func roleList(rs []Role) string {
	if len(rs) == 0 {
		return "-"
	}
	parts := make([]string, 0, len(rs))
	for _, r := range rs {
		parts = append(parts, string(r))
	}
	return strings.Join(parts, ",")
}

func orElse(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}

// under joins a path under base, returning "" when base is empty so an unresolvable root yields
// no role row instead of a bogus relative path.
func under(base string, parts ...string) string {
	if strings.TrimSpace(base) == "" {
		return ""
	}
	return filepath.Join(append([]string{base}, parts...)...)
}

func binExt() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}
