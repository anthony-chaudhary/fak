package envconfiglint

import (
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// This file is the liveness half of the ratchet — the thing doc.go's own post-mortem asked
// for and did not build: "A ratchet needs its own liveness witness, not just its rule."
//
// The rule half (ScanTree + TestNoNewNonSecretEnvReads) answers "is the tree clean?". It
// cannot answer "is the gate still gating?", because both questions produce the same single
// bit: a red. When a red is NEW it means an author just added a behavioral env read and the
// gate caught it at the door, which is the ratchet working. When a red is OLD it means the
// gate has been failing for so long that reads now land THROUGH it — the failure #2863 set
// out to prevent, reproduced inside the fix, twice (10 offenders the first time, 16 the
// second, the oldest of them 435 trunk advances stale).
//
// So the liveness witness measures the red's AGE. A read that survives more than one trunk
// advance was never refused at the door; it landed against a gate that was already broken.
// That is a distinct defect from "someone added a read", it names a process failure rather
// than an author, and — unlike the rule — it is a question the rule alone cannot ask.

// ReasonRatchetUnwatched is the closed-vocabulary refusal code for a ratchet that has been
// RED across more than one trunk advance: not "a new read arrived" but "the gate stopped
// gating". Same structured-refusal idiom as ReasonConfigNotEnv.
const ReasonRatchetUnwatched = "RATCHET_UNWATCHED"

// UnwatchedTolerance is how many trunk advances an outstanding offense may survive before
// the ratchet counts as unwatched. ONE: an author who lands a read reds their own CI run and
// the next commit either fixes it or admits it. Surviving a second trunk advance means
// somebody looked at the red and shipped anyway — or, far more likely, nobody looked.
const UnwatchedTolerance = 1

// AdvancesSince reports how many trunk commits have landed since the env read an offense
// names was introduced. Zero means HEAD itself introduced it. A NEGATIVE result means the
// age is unknown (a rename, a squash, a shallow clone), which is never treated as unwatched:
// erring toward "watched" yields fewer refusals, the same safe direction IsSecretName takes.
type AdvancesSince func(Offense) int

// UnwatchedRead is one outstanding offense together with the age of its red.
type UnwatchedRead struct {
	Offense
	Advances int // trunk commits landed since the read was introduced
}

// LivenessVerdict is the ratchet's verdict on ITSELF: how many reads are outstanding, and
// which of them are old enough to prove the gate was not enforcing when they landed.
type LivenessVerdict struct {
	Offenses  int             // outstanding non-secret reads at HEAD
	Unwatched []UnwatchedRead // those older than UnwatchedTolerance, oldest first
}

// Watched reports whether the ratchet is still doing its job. A CLEAN tree is watched, and
// so is a tree whose only offense arrived on the current trunk tip — that red is the gate
// working, not the gate failing.
func (v LivenessVerdict) Watched() bool { return len(v.Unwatched) == 0 }

// Advances is the age of the oldest unwatched read: the number of trunk commits that landed
// against a gate already known to be red. Zero when the ratchet is watched.
func (v LivenessVerdict) Advances() int {
	if len(v.Unwatched) == 0 {
		return 0
	}
	return v.Unwatched[0].Advances
}

// String renders the refusal, naming the age (the part the rule cannot say) before the
// offenders, because the age is the finding: the count is recoverable from ScanTree.
func (v LivenessVerdict) String() string {
	if v.Watched() {
		return "CONFIG_NOT_ENV ratchet is watched: " +
			strconv.Itoa(v.Offenses) + " outstanding read(s), none older than " +
			strconv.Itoa(UnwatchedTolerance) + " trunk advance(s)"
	}
	oldest := v.Unwatched[0]
	var b strings.Builder
	b.WriteString("the CONFIG_NOT_ENV ratchet stopped gating (" + ReasonRatchetUnwatched + "): ")
	b.WriteString(strconv.Itoa(len(v.Unwatched)) + " of " + strconv.Itoa(v.Offenses))
	b.WriteString(" outstanding non-secret env read(s) survived more than ")
	b.WriteString(strconv.Itoa(UnwatchedTolerance) + " trunk advance(s); the oldest, ")
	b.WriteString(oldest.Offense.String())
	b.WriteString(", landed " + strconv.Itoa(oldest.Advances) + " trunk advance(s) ago.\n")
	b.WriteString("A red ratchet refuses nothing, so every commit since then shipped against a gate " +
		"that was already failing. Dispose of the reads (relocate, or record them in " +
		"admittedPostFreeze with a reason), and treat the AGE as the defect: a gate that can stay " +
		"red across the trunk is not a gate.")
	for _, u := range v.Unwatched {
		b.WriteString("\n  " + strconv.Itoa(u.Advances) + " advance(s): " + u.Offense.String())
	}
	return b.String()
}

// ClassifyLiveness is the pure liveness core (verify the verifier — no git, no tree): given
// the outstanding offenses and a way to age each one, decide whether the ratchet is still
// gating. Kept separate from the git wiring so the judgment is testable on synthetic ages,
// which is the only way to exercise the RED path once the trunk is green.
func ClassifyLiveness(offenses []Offense, since AdvancesSince) LivenessVerdict {
	v := LivenessVerdict{Offenses: len(offenses)}
	for _, o := range offenses {
		age := -1
		if since != nil {
			age = since(o)
		}
		if age > UnwatchedTolerance {
			v.Unwatched = append(v.Unwatched, UnwatchedRead{Offense: o, Advances: age})
		}
	}
	// Oldest first: the age that matters is the worst one, and String() leads with it.
	for i := 1; i < len(v.Unwatched); i++ {
		for j := i; j > 0 && v.Unwatched[j].Advances > v.Unwatched[j-1].Advances; j-- {
			v.Unwatched[j], v.Unwatched[j-1] = v.Unwatched[j-1], v.Unwatched[j]
		}
	}
	return v
}

// TreeLiveness is the live liveness gate: scan the committed tree, then age every offense it
// finds against the trunk. It costs nothing while the ratchet is green — a clean scan means
// no ages to look up — and only shells to git once per outstanding offense when it is not.
func TreeLiveness(repoRoot string) (LivenessVerdict, error) {
	offenses, err := ScanTree(repoRoot)
	if err != nil {
		return LivenessVerdict{}, err
	}
	return ClassifyLiveness(offenses, GitAdvancesSince(repoRoot)), nil
}

// GitAdvancesSince ages an offense against the real trunk: find the commit that introduced
// the read's quoted name into its file (`git log -S`, the pickaxe — the same committed-HEAD
// evidence ScanTree uses, never the working tree), then count the commits between it and
// HEAD. Results are memoized per name+file because a scan can report the same read twice.
func GitAdvancesSince(repoRoot string) AdvancesSince {
	cache := map[string]int{}
	return func(o Offense) int {
		key := o.Name + "\x00" + o.File
		if n, ok := cache[key]; ok {
			return n
		}
		n := gitAdvancesSince(repoRoot, o.Name, o.File)
		cache[key] = n
		return n
	}
}

// gitAdvancesSince returns the trunk advances since name was introduced in file, or -1 when
// the introduction cannot be located. It retries across all Go source when the per-file
// pickaxe finds nothing, so a read that moved files still ages from its true first landing.
func gitAdvancesSince(repoRoot, name, file string) int {
	sha := gitIntroducedAt(repoRoot, name, file)
	if sha == "" && file != "" {
		sha = gitIntroducedAt(repoRoot, name, "")
	}
	if sha == "" {
		return -1
	}
	out, err := gitOutput(repoRoot, "rev-list", "--count", sha+"..HEAD")
	if err != nil {
		return -1
	}
	n, err := strconv.Atoi(strings.TrimSpace(out))
	if err != nil {
		return -1
	}
	return n
}

// gitIntroducedAt returns the OLDEST commit whose diff changed the number of occurrences of
// the quoted env name in pathspec — the commit that first spelled the read at a call site.
func gitIntroducedAt(repoRoot, name, pathspec string) string {
	if pathspec == "" {
		pathspec = "*.go"
	}
	out, err := gitOutput(repoRoot, "log", "--format=%H", "--reverse", `-S"`+name+`"`, "HEAD", "--", pathspec)
	if err != nil {
		return ""
	}
	first, _, _ := strings.Cut(strings.TrimSpace(out), "\n")
	return strings.TrimSpace(first)
}

// gitOutput runs one git command in repoRoot under the same background-window configuration
// committedEnvReadMatches uses, so the liveness lookups cannot pop a console on Windows.
func gitOutput(repoRoot string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
