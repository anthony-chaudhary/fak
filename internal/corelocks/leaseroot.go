package corelocks

// leaseroot.go — the POSITIONALLY-RESOLVED lease journal, made answerable.
//
// Issue #5933 (epic #5949), generalized from tb/tools/lanescan's dosroot.go
// (their #743/#682, sharpening #640).
//
// ⛔ THE DEFECT CLASS. fak's trust kernel keeps its lease journal under a `.dos`
// directory that is found by an UPWARD POSITIONAL WALK from whatever working
// directory the caller happened to be in. A caller that never sets an explicit
// root therefore reads whichever `.dos` is nearest — and there are `.dos` roots
// nested INSIDE this repository, so the walk can land on one without the caller
// ever leaving the tree. Measured on this checkout on 2026-08-08 by filesystem
// walk (`ShadowRoots` below), three of them:
//
//	docs/.dos
//	tools/.dos
//	tools/concept_disambiguation_scorecard.data/.dos
//
// In the upstream measurement the same topology produced three byte-identical
// "no live leases — every lane is free" banners at the same instant that 22
// leases were genuinely held. The mitigation of record there was "run every
// lease verb from the repo root", and two of the three offending directories
// SATISFIED it: the caller was inside the repository the entire time.
//
// ⛔ AND IT FAILS OPEN, into the referee. internal/laneadmit.Decide ends with
//
//	if len(conflicts) == 0 { return Verdict{Admit: true, Tree: tree} }
//
// so an under-reported census does not merely withhold information — it
// affirmatively tells the arbitrator that every lane is admissible. A close-out
// sweep handed "0 leases held" releases a lane under a live worker; a reap
// handed nothing finds nothing stale. That is fak's lived experience already
// (stale-lease reaps in the experiments lane), and it is why the count alone is
// never a sufficient answer.
//
// ⛔ NO GIT ENUMERATION CAN FIND THE SHADOW ROOTS. An UNANCHORED ignore pattern
// (`.dos/`, or `**/.dos/`) matches at every depth, so every nested root returns
// 0 from `git ls-files`, is invisible to `git status`, and is excluded from
// `ls-files --others --exclude-standard` BY CONSTRUCTION. Only a filesystem walk
// finds them — which is why ShadowRoots is a walk and not a git subprocess, and
// why UnanchoredStateIgnores exists to pin the ignore pattern anchored.
//
// ⭐ THE TWO RULES this file mechanizes, for any positionally-resolved store:
//
//  1. An empty census must be DISTINGUISHABLE from "I read the wrong store".
//     The resolved root is a first-class field of every census and of every
//     refusal, so "every lane is free" and "I read the wrong journal" can never
//     print the same bytes.
//  2. FAIL CLOSED. A census resolved from a non-canonical root is an error, not
//     `0 leases` — because every consumer of a lease census is a safety check,
//     and a safety check handed nothing admits everything. ReadCensus enforces
//     this by CONSTRUCTION: it never calls the counter at all unless the root
//     is authoritative, so a blind count cannot be produced by a caller who
//     forgets to check.
//
// ⭐ WHAT IS AND IS NOT THE HAZARD, because the two invite opposite fixes. A
// shadow root sitting somewhere in the tree harms nobody standing at the git
// toplevel: the walk from there stops at the toplevel's own `.dos` and the
// census is correct however many shadows exist elsewhere. The hazard is
// POSITIONAL — it is being inside one. So the refusal is positional too: a check
// that refused merely because a shadow root exists somewhere would deny every
// session in the fleet the next time an agent ran a tool from a subdirectory,
// and the remedy (deleting a peer's untracked state directory) belongs to
// whoever owns it. CheckRoot refuses the caller standing in the wrong place, and
// only that caller. ShadowRoots is the separate, non-refusing census for a
// report.
//
// This file stays stdlib-only and imports nothing internal, like the rest of
// corelocks.

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// StateDir is the directory name the trust kernel's workspace state — the lease
// journal among it — is anchored on. It is the token the positional walk looks
// for and the token whose ignore pattern must stay anchored.
const StateDir = ".dos"

// RootVerdict answers "would a lease census read from Start describe THIS
// repository?".
//
// ⛔ It carries a named Cause rather than returning a bare error, because an
// unmet precondition here is a fact about the world the caller must be able to
// PRINT. Returning only `error` invites `if err != nil { /* best effort */ }`
// and a blind census carried on regardless — precisely the fail-open shape under
// repair. Authoritative false ALWAYS comes with a Cause, and the causes always
// distinguish the reasons from one another: a shared message would repeal rule 1
// one layer up.
type RootVerdict struct {
	// Start is the absolute directory the upward walk began in.
	Start string `json:"start"`
	// Resolved is the state root that walk landed on, or "" when the walk
	// reached the filesystem root without finding one. This is THE field the
	// ticket exists for: it is populated on every verdict, refusal included.
	Resolved string `json:"resolved"`
	// GitTop is the git toplevel the caller believes it is working in — the one
	// canonical root.
	GitTop string `json:"git_top"`
	// Authoritative reports that Resolved is GitTop: the ONLY state in which a
	// lease census read here describes this repository.
	Authoritative bool `json:"authoritative"`
	// Cause names why not, in a sentence fit to print at a refusal. Empty if and
	// only if Authoritative.
	Cause string `json:"cause,omitempty"`
	// Shadowed separates the "resolved root is INSIDE the repository" case from
	// a root outside it and from no root at all. The remedy differs: a shadow
	// root is a repository defect somebody owns, while standing outside the tree
	// is the caller's own mistake.
	Shadowed bool `json:"shadowed"`
}

// ResolveRoot walks up from start and returns the first directory containing a
// StateDir directory — the same positional rule the kernel applies, which is
// what makes this able to PREDICT the journal a subprocess would read.
//
// ⛔ It requires StateDir to be a DIRECTORY. A regular file of that name is
// walked past by the kernel and must be walked past here too, or this reports a
// root the kernel does not use and the prediction is worse than none.
func ResolveRoot(start string) (string, bool) { return resolveRootWithin(start, "") }

// resolveRootWithin is ResolveRoot with an explicit CEILING: the walk gives up
// after examining ceiling, as though ceiling's parent were the filesystem root.
// A ceiling of "" is the production rule — walk until the real filesystem root,
// because that is what the kernel's own subprocess does.
//
// ⛔ The ceiling exists so the "no root ANYWHERE above me" verdict is reachable
// from a test. A fixture in a temp directory cannot control its own ancestors,
// and on the host this was measured on the OS temp directory ITSELF carries a
// `.dos` — which is not an accident of the harness but another instance of the
// finding: somebody once ran a lease verb from /tmp and left a rogue journal
// there. Without a ceiling the no-root row silently becomes the outside-the-repo
// row, and the one verdict most easily confused with a healthy quiet repository
// is the one that never gets exercised.
func resolveRootWithin(start, ceiling string) (string, bool) {
	dir, err := filepath.Abs(start)
	if err != nil {
		return "", false
	}
	stop := ""
	if ceiling != "" {
		if abs, err := filepath.Abs(ceiling); err == nil {
			stop = abs
		}
	}
	for {
		if fi, err := os.Stat(filepath.Join(dir, StateDir)); err == nil && fi.IsDir() {
			return dir, true
		}
		if stop != "" && sameDir(dir, stop) {
			return "", false
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// CheckRoot reports whether a lease census read from start describes the
// repository at gitTop.
//
// gitTop may be "" — a caller that could not resolve a git toplevel gets a
// refusal NAMING that, rather than a comparison against the empty string that
// would accidentally succeed for a start directory with no root either.
func CheckRoot(start, gitTop string) RootVerdict { return checkRootWithin(start, gitTop, "") }

// checkRootWithin is CheckRoot with the walk ceiling of resolveRootWithin, which
// is what lets a constructed fixture stand in for the whole filesystem. Every
// verdict below is computed identically either way; the ceiling only bounds how
// far the walk may escape the fixture.
func checkRootWithin(start, gitTop, ceiling string) RootVerdict {
	v := RootVerdict{Start: start, GitTop: gitTop}
	if abs, err := filepath.Abs(start); err == nil {
		v.Start = abs
	}
	if gitTop != "" {
		if abs, err := filepath.Abs(gitTop); err == nil {
			v.GitTop = abs
		}
	}
	resolved, found := resolveRootWithin(v.Start, ceiling)
	v.Resolved = resolved

	switch {
	case v.GitTop == "":
		v.Cause = fmt.Sprintf("no git toplevel was resolved for %s, so there is nothing to check the "+
			"%s root against; a census read here describes an unknown workspace", v.Start, StateDir)
	case !found:
		// ⛔ A REFUSAL, not "clean, zero leases". With no journal there is
		// nothing to read, so every lane reads free — the same bytes as a
		// genuinely quiet repository and the opposite claim.
		v.Cause = fmt.Sprintf("no %s directory at or above %s, so there is no lease journal to read: "+
			"a census here reports every lane free because it found NOTHING, not because "+
			"nothing is held", StateDir, v.Start)
	case sameDir(resolved, v.GitTop):
		v.Authoritative = true
	case underDir(resolved, v.GitTop):
		v.Shadowed = true
		rel, err := filepath.Rel(v.GitTop, resolved)
		if err != nil {
			rel = resolved
		}
		v.Cause = fmt.Sprintf("a SHADOW %s root inside the repository shadows the real one: the walk from "+
			"%s stopped at %s/%s instead of the git toplevel %s. Every lane will read free, and an "+
			"empty lease census ADMITS every lane (internal/laneadmit.Decide) (#5933)",
			StateDir, v.Start, filepath.ToSlash(rel), StateDir, v.GitTop)
	default:
		v.Cause = fmt.Sprintf("the %s root resolved from %s is %s, which is OUTSIDE the repository at %s, "+
			"so the census describes a DIFFERENT workspace (#5933)", StateDir, v.Start, resolved, v.GitTop)
	}
	return v
}

// sameDir compares two cleaned absolute paths for directory identity. It is
// case-insensitive on Windows, where C:\Work\fak and C:\work\fak are the same
// directory and a byte comparison would call the repository its own shadow.
func sameDir(a, b string) bool {
	if a == b {
		return true
	}
	if filepath.Separator == '\\' {
		return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
	}
	return false
}

// underDir reports whether path is a STRICT descendant of base. It compares
// cleaned paths component-wise rather than with strings.HasPrefix, which would
// call /repo-backup a directory inside /repo — and so report a wholly separate
// checkout's root as a shadow of this one, in a refusal that reads
// authoritative.
func underDir(path, base string) bool {
	rel, err := filepath.Rel(base, path)
	if err != nil {
		return false
	}
	if rel == "." || rel == ".." {
		return false
	}
	if filepath.Separator == '\\' {
		rel = strings.ToLower(rel)
	}
	return !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// ShadowRoots enumerates the state roots nested INSIDE gitTop, excluding
// gitTop's own. The result is repo-relative, slash-separated, in walk (lexical)
// order.
//
// ⛔ It is a filesystem walk ON PURPOSE. Every git enumeration is blind to these
// by construction while the ignore pattern is unanchored — see this file's
// header — so asking git would produce exactly the confident empty answer that
// is the defect under repair.
//
// `.git` is skipped: a linked worktree's administrative directory can contain
// anything and is not part of the tree anybody edits. A directory that cannot be
// read is REPORTED rather than skipped — a partial census that reads as complete
// is the failure this file exists to refuse.
func ShadowRoots(gitTop string) ([]string, error) {
	top, err := filepath.Abs(gitTop)
	if err != nil {
		return nil, fmt.Errorf("corelocks: %s: %w", gitTop, err)
	}
	var out []string
	err = filepath.WalkDir(top, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return fmt.Errorf("corelocks: walking %s: %w", p, walkErr)
		}
		if !d.IsDir() {
			return nil
		}
		if d.Name() == ".git" {
			return fs.SkipDir
		}
		if d.Name() != StateDir {
			return nil
		}
		// The toplevel's own root is the CORRECT one, and nothing beneath a
		// state directory is another project root.
		if parent := filepath.Dir(p); !sameDir(parent, top) {
			rel, relErr := filepath.Rel(top, p)
			if relErr != nil {
				return fmt.Errorf("corelocks: %s: %w", p, relErr)
			}
			out = append(out, filepath.ToSlash(rel))
		}
		return fs.SkipDir
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ErrBlindCensus is the sentinel every refusal to report a lease count wraps. A
// caller that cares only "may I trust this number?" tests errors.Is against it;
// a caller that must PRINT the reason reads the Census the error carries.
var ErrBlindCensus = errors.New("corelocks: lease census refused: the journal root read is not this repository's")

// ErrCensusUnreadable is the sentinel for "the canonical root was found but its
// journal could not be counted". It is deliberately NOT the same sentinel as
// ErrBlindCensus and deliberately NOT zero: an unreadable journal is unknown,
// and unknown fails closed.
var ErrCensusUnreadable = errors.New("corelocks: lease census refused: the canonical journal could not be read")

// Census is a lease/lane count TOGETHER WITH the journal root it was read from.
//
// ⛔ Root is not decoration. The whole finding is that a count alone cannot
// distinguish "every lane is free" from "I read the wrong journal", so this type
// makes it impossible to obtain one without the other: every constructor
// populates Root (or, when nothing resolved, says so), and Line() names it in
// every branch.
type Census struct {
	// Root is the state root the census was (or would have been) read from —
	// "" only when the upward walk found none at all.
	Root string `json:"root"`
	// Start is the directory the walk began in: the caller's position, which is
	// the actual variable in the defect.
	Start string `json:"start"`
	// GitTop is the one canonical root the census is checked against.
	GitTop string `json:"git_top"`
	// Held is the number of live leases. It is meaningful ONLY when Counted;
	// it is -1 otherwise so a caller that ignores the error cannot read a zero.
	Held int `json:"held"`
	// Counted reports that Held was actually measured from an authoritative
	// root. False means "no number was produced", never "zero".
	Counted bool `json:"counted"`
	// Authoritative mirrors RootVerdict.Authoritative.
	Authoritative bool `json:"authoritative"`
	// Shadowed mirrors RootVerdict.Shadowed.
	Shadowed bool `json:"shadowed"`
	// Cause is the refusal sentence; empty if and only if Counted.
	Cause string `json:"cause,omitempty"`
}

// Line renders the census as one printable line.
//
// ⭐ THE INVARIANT under test: the four outcomes — a proven non-zero, a proven
// zero, a blind read from a shadow root, and a blind read with no root at all —
// print PAIRWISE DIFFERENT bytes, and every one of them names the root (or names
// the start directory and the absence of a root). "no live leases — every lane
// is free" printed identically from three different directories is the whole
// bug; a shared refusal string would reintroduce it one layer up.
func (c Census) Line() string {
	if c.Counted {
		if c.Held == 0 {
			return fmt.Sprintf("0 live lease(s) — journal root %s (authoritative): every lane is genuinely free", c.Root)
		}
		return fmt.Sprintf("%d live lease(s) — journal root %s (authoritative)", c.Held, c.Root)
	}
	if c.Root == "" {
		return fmt.Sprintf("UNPROVEN lease census — no %s root at or above %s, so NOTHING was read: %s",
			StateDir, c.Start, c.Cause)
	}
	kind := "non-canonical"
	if c.Shadowed {
		kind = "SHADOW"
	}
	return fmt.Sprintf("UNPROVEN lease census — %s journal root %s (walked up from %s, canonical root is %s): %s",
		kind, c.Root, c.Start, c.GitTop, c.Cause)
}

// BlindCensusError is the error a refused census carries, so a caller that only
// logs `err` still prints the resolved root.
type BlindCensusError struct {
	// Census is the refused census: Root, Start, GitTop and Cause are all set.
	Census Census
	// sentinel is the wrapped classification (ErrBlindCensus or
	// ErrCensusUnreadable).
	sentinel error
}

func (e *BlindCensusError) Error() string { return e.Census.Line() }
func (e *BlindCensusError) Unwrap() error { return e.sentinel }

// NewCensus pairs a root verdict with a count, applying the fail-closed rule: a
// count from a non-authoritative root is an ERROR, never a number. The returned
// Census is populated either way — that is the point — so a caller may print it
// on the error path.
func NewCensus(v RootVerdict, held int) (Census, error) {
	c := Census{
		Root:          v.Resolved,
		Start:         v.Start,
		GitTop:        v.GitTop,
		Held:          -1,
		Authoritative: v.Authoritative,
		Shadowed:      v.Shadowed,
		Cause:         v.Cause,
	}
	if !v.Authoritative {
		return c, &BlindCensusError{Census: c, sentinel: ErrBlindCensus}
	}
	c.Held = held
	c.Counted = true
	return c, nil
}

// ReadCensus is the whole mechanism in one call: resolve the journal root from
// start, refuse unless it is gitTop, and only THEN ask count for the number of
// live leases at that resolved root.
//
// ⛔ The ordering is the safety property, and it is structural rather than
// advisory: count is NEVER invoked from a non-authoritative root, so a blind
// count cannot be produced even by a caller who ignores the returned error. A
// counter that itself fails is reported as unreadable (ErrCensusUnreadable), not
// as zero — the one substitution that turns this whole file back into the bug.
//
// count receives the RESOLVED ABSOLUTE ROOT and must read only from there: the
// third rule of this file is that a caller never re-derives the root by standing
// somewhere and hoping.
func ReadCensus(start, gitTop string, count func(root string) (int, error)) (Census, error) {
	v := CheckRoot(start, gitTop)
	c, err := NewCensus(v, 0)
	if err != nil {
		return c, err
	}
	if count == nil {
		c.Held, c.Counted = -1, false
		c.Cause = fmt.Sprintf("no lease counter was supplied for %s, so no census was taken", c.Root)
		return c, &BlindCensusError{Census: c, sentinel: ErrCensusUnreadable}
	}
	n, cerr := count(v.Resolved)
	if cerr != nil {
		c.Held, c.Counted = -1, false
		c.Cause = fmt.Sprintf("the journal at %s could not be counted (%v), which is UNKNOWN, not zero", c.Root, cerr)
		return c, &BlindCensusError{Census: c, sentinel: ErrCensusUnreadable}
	}
	c.Held, c.Counted = n, true
	c.Cause = ""
	return c, nil
}

// UnanchoredStateIgnores returns the .gitignore patterns that hide the state
// directory AT EVERY DEPTH — the patterns that make nested roots invisible to
// every git enumeration.
//
// An unanchored `.dos/` matches at any depth; so does `**/.dos/`. Only a
// LEADING-SLASH pattern (`/.dos/`) is confined to the repository root, which is
// what lets `git status` see a nested root at all. Negations (`!`) are examined
// too: a negated unanchored pattern is still a depth-blind rule about this
// directory and is reported, so an operator sees the whole picture.
//
// Returned patterns are the raw (comment- and whitespace-stripped) lines, in
// file order, so a caller can name the exact line to anchor.
func UnanchoredStateIgnores(gitignore []byte, dirName string) []string {
	var out []string
	for _, raw := range strings.Split(strings.ReplaceAll(string(gitignore), "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		pat := strings.TrimPrefix(line, "!")
		if strings.HasPrefix(pat, "/") {
			continue // anchored to the repository root: exactly what we want
		}
		norm := strings.TrimSuffix(pat, "/")
		norm = strings.TrimPrefix(norm, "**/")
		if norm != dirName {
			continue
		}
		out = append(out, line)
	}
	return out
}

// ReaderDisposition classifies how a destructive reader of a lease census
// behaves when the census comes back EMPTY.
type ReaderDisposition string

const (
	// DispositionFailsClosed — an empty or unproven census cannot authorize the
	// destructive act. This is the required disposition for anything that
	// releases, reaps or admits on the strength of "nothing is held".
	DispositionFailsClosed ReaderDisposition = "fails-closed"
	// DispositionFailOpenIsCorrect — an empty census makes the reader do LESS,
	// not more, so failing open is the safe direction and is documented as such.
	DispositionFailOpenIsCorrect ReaderDisposition = "fail-open-is-correct"
	// DispositionFailsOpen — an empty census makes the reader do MORE. This is
	// the hazard; a row carrying it must name its mitigation.
	DispositionFailsOpen ReaderDisposition = "fails-open"
)

// DestructiveReader is one audited consumer of a lease/lane census that can act
// destructively (reap, release, sweep, admit) on the strength of an empty one.
type DestructiveReader struct {
	// Path is the repo-relative file the reader lives in.
	Path string
	// Symbol is the function whose empty-census behaviour was audited. It must
	// appear literally in Path (the audit test greps for it), so a rename cannot
	// leave a stale row behind claiming a function that no longer exists.
	Symbol string
	// Disposition is the audited empty-census behaviour.
	Disposition ReaderDisposition
	// Note is the argument: why the disposition is what it is, and — for a
	// fail-open row — what keeps it from firing.
	Note string
}

// DestructiveReaders is the audited list the ticket's last acceptance item asks
// for: every reader in this repository that can act destructively on an empty
// lease census, and what each does when handed one.
//
// ⭐ It is DATA rather than prose so it is checkable: the package's test asserts
// each Path exists and each Symbol is present in it, so a rename or a deletion
// reds the audit instead of silently rotting it. It is deliberately not
// exhaustive of everything that reads a lease — only of the readers whose EMPTY
// answer changes what they DO.
func DestructiveReaders() []DestructiveReader {
	return []DestructiveReader{
		{
			Path:        "internal/laneadmit/laneadmit.go",
			Symbol:      "func Decide(",
			Disposition: DispositionFailsOpen,
			Note: "THE hazard, and the reason this file exists: Decide ends `if len(conflicts) == 0 " +
				"{ return Verdict{Admit: true} }`, so a census that under-reports does not merely " +
				"withhold — it tells the arbitrator every lane is admissible. Decide is PURE and " +
				"takes the lease set as an argument, so it cannot check the root itself; the " +
				"mitigation is that whoever produces that lease set must obtain it through " +
				"ReadCensus (or an explicit root) and refuse before calling Decide.",
		},
		{
			Path:        "cmd/fak/dispatch_tick_lease_beat.go",
			Symbol:      "func dispatchLaneBeatLiveLeasesDos(",
			Disposition: DispositionFailsClosed,
			Note: "Reads the structurally-live lane leases from the kernel to decide which to " +
				"heartbeat; an empty read means no beat, and an un-beaten live lease EXPIRES and is " +
				"released under its worker. It fails closed on both axes: it sets an explicit " +
				"cmd.Dir = root (never the positional walk) and returns ok=false — not an empty " +
				"slice — on any error, so the caller can distinguish 'none' from 'unknown'.",
		},
		{
			Path:        "cmd/fak/loop_drive.go",
			Symbol:      "func runDOSLoopGateWitness(",
			Disposition: DispositionFailsClosed,
			Note: "The loop-gate witness adapter. It shells to the kernel and its verdict gates " +
				"whether a loop may proceed, so a read against the wrong workspace grants a " +
				"witness this repository never produced. It sets cmd.Dir = repoRoot() (the " +
				"go.mod module root), pinning the workspace instead of inheriting the loop's " +
				"cwd; it was the one caller in the tree still relying on the positional walk, " +
				"and it stayed that way undetected because the audit's own scanner was vacuous " +
				"(see dosExecSite in leaseroot_test.go).",
		},
		{
			Path:        "internal/leaseref/reap.go",
			Symbol:      "func (s *Store) Reap(",
			Disposition: DispositionFailOpenIsCorrect,
			Note: "An empty expired set deletes NOTHING, so an under-read makes the reaper do less, " +
				"not more. It is also not positionally resolved: the ref namespace comes from the " +
				"git dir, not an upward walk for a state directory.",
		},
		{
			Path:        "internal/leaseref/lockfile.go",
			Symbol:      "func walkLockFiles(",
			Disposition: DispositionFailOpenIsCorrect,
			Note: "Same direction: an absent or empty locks directory removes nothing, and the " +
				"directory is derived from `rev-parse --git-common-dir` rather than a positional " +
				"walk. A future-dated mtime is KEPT — unknown age already fails closed there.",
		},
		{
			Path:        "internal/leaseref/release.go",
			Symbol:      "func (s *Store) ReleaseFenced(",
			Disposition: DispositionFailsClosed,
			Note: "A release requires a matching (holder, generation) compare-and-swap against the " +
				"live ref, so an empty census cannot authorize one: there is no code path where " +
				"'I saw no leases' becomes 'therefore delete this one'.",
		},
		{
			Path:        "internal/gpulease/lease.go",
			Symbol:      "func Acquire(",
			Disposition: DispositionFailsClosed,
			Note: "Not census-based at all: an OS advisory flock whose holder set is owned by the " +
				"kernel and released on fd close. There is no journal to under-report, which is " +
				"why this family member needs no root check.",
		},
	}
}
