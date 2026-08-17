package appversion

import (
	"path/filepath"
	"strings"
	"time"
)

// pathscan answers the question the sibling-only binary doctor cannot: "when I
// type `fak`, WHICH of the fak binaries scattered across my PATH actually runs,
// and is it the current one?" DefaultBinaryDoctorCandidates only inspects the
// running exe's OWN directory, so it is blind to the common real failure — a
// stale `fak.exe` early on PATH (e.g. ~/bin) shadowing a freshly built one later
// (e.g. ~/go/bin or a repo checkout). Worse, the shadowing binary is often
// UNSTAMPED (placed by `go install …@latest` or a plain copy), so `fak version`
// line 1 still prints a current-looking app version — read from the working
// directory's VERSION file, not the binary — and hides that the running code is
// old. This walks PATH in resolution order and reads each binary's OWN embedded
// stamp so the age of the binary, not the tree it happens to run against, is what
// gets judged.

// PathStamp is the build identity read out of one fak binary on the PATH — the
// subset of `fak version --json` that identifies WHICH binary a file is: the
// embedded VCS commit (empty when the binary carries no stamp), its commit time,
// the dirty bit, whether any stamp is present at all, and the app-version string
// the binary self-reports. A caller fills it by executing the candidate;
// ScanPathForFak stays pure over injected probes so it is testable without exec.
type PathStamp struct {
	Commit     string // vcs.revision (may be short); "" when unstamped
	CommitTime string // RFC3339 vcs.time; "" when unstamped
	Dirty      bool
	Stamped    bool
	AppVersion string // line-1 version the binary printed (may be CWD-derived)
}

// PathBinary is one fak executable resolvable on the PATH, ranked by PATH order.
// Rank 0 (Winner) is the file the shell runs when you type `fak`; every other
// entry is shadowed by it. The stamp fields are the binary's OWN provenance, so
// two entries can be compared for age even when both self-report the same
// app-version (which they will, since app-version comes from the working dir).
type PathBinary struct {
	Path       string `json:"path"`
	Rank       int    `json:"rank"`   // position in PATH resolution order; 0 wins
	Winner     bool   `json:"winner"` // rank 0 — the one `fak` actually runs
	Size       int64  `json:"size,omitempty"`
	ModTime    string `json:"mod_time,omitempty"` // file mtime (RFC3339, UTC)
	Commit     string `json:"commit,omitempty"`
	CommitTime string `json:"commit_time,omitempty"`
	Dirty      bool   `json:"dirty,omitempty"`
	Stamped    bool   `json:"stamped"`
	AppVersion string `json:"app_version,omitempty"`
	StampError string `json:"stamp_error,omitempty"` // set when the binary could not self-report
}

// ScanPathForFak walks dirs in PATH order, looks for each name in each dir, and
// returns the resolvable fak binaries in resolution order with Rank/Winner
// assigned (rank 0 = what `fak` runs). probe reports whether a candidate path
// exists and fills its stat + stamp fields; a (dir,name) pair probe rejects is
// skipped. A path already emitted (same cleaned key) is not repeated — PATH
// routinely lists a directory more than once, and only the first occurrence can
// win. All control flow is pure; every side effect lives in probe, so tests
// inject a map instead of touching the filesystem.
func ScanPathForFak(dirs []string, names []string, probe func(path string) (PathBinary, bool)) []PathBinary {
	var out []PathBinary
	seen := map[string]bool{}
	for _, d := range dirs {
		d = strings.TrimSpace(d)
		if d == "" {
			continue
		}
		for _, name := range names {
			p := filepath.Join(d, name)
			key := cleanPathKey(p)
			if seen[key] {
				continue
			}
			entry, ok := probe(p)
			if !ok {
				continue
			}
			seen[key] = true
			entry.Path = p
			entry.Rank = len(out)
			entry.Winner = len(out) == 0
			out = append(out, entry)
		}
	}
	return out
}

// PathShadowRecommendation judges the PATH-ranked fak binaries. Only the winner
// (rank 0) is what `fak` runs, so the finding is about IT — the two WARN cases
// are exactly the reported "my fak is old / a different folder loads a different
// version" failure:
//
//   - the winner carries no VCS stamp (Stamped=false): it cannot attest which
//     commit it was built from, so "is my fak current?" is structurally
//     unanswerable, and a normal rebuild that lands in a different PATH dir will
//     never replace it. This is the load-bearing case.
//   - a lower-priority binary on PATH is provably NEWER than the winner (newer
//     commit time when both are stamped, else newer file mtime): typing `fak`
//     runs the OLD one while a fresher build sits shadowed behind it.
//
// A single fak on PATH, or a stamped winner that is the newest, is OK.
func PathShadowRecommendation(bins []PathBinary) BinaryRecommendation {
	const check = "binary-path-shadow"
	if len(bins) == 0 {
		return BinaryRecommendation{
			Check:    check,
			Severity: SeverityOK,
			Finding:  "no fak found on PATH to compare",
		}
	}
	winner := bins[0]
	newer, hasNewer := newestShadowedNewerThan(winner, bins[1:])

	switch {
	case !winner.Stamped:
		rec := BinaryRecommendation{
			Check:    check,
			Severity: SeverityWarn,
			Finding: "the fak that wins on PATH (" + winner.Path + ") carries no VCS stamp — it cannot attest which commit it was built from, " +
				"so its staleness is UNVERIFIABLE and a rebuild landing elsewhere on PATH will never replace it",
			Recommend: "replace the PATH winner with a stamped build: `fak self-update --force --target " + winner.Path + "` " +
				"builds+gates+installs current origin/main over it. A plain `go install …@latest` or file copy strips the stamp — " +
				"prefer `go build ./cmd/fak` inside the repo (which stamps the commit) then swap it into the PATH dir.",
		}
		if hasNewer {
			rec.Finding += "; a newer fak is already present later on PATH at " + newer.Path
		}
		return rec
	case hasNewer:
		return BinaryRecommendation{
			Check:    check,
			Severity: SeverityWarn,
			Finding: "the fak that wins on PATH (" + winner.Path + ", " + describeAge(winner) + ") is older than a shadowed sibling at " +
				newer.Path + " (" + describeAge(newer) + ") — typing `fak` runs the stale one",
			Recommend: "put the newer binary earlier on PATH, or replace the winner: " +
				"`fak self-update --force --target " + winner.Path + "`.",
		}
	default:
		return BinaryRecommendation{
			Check:    check,
			Severity: SeverityOK,
			Finding:  "the fak that wins on PATH (" + winner.Path + ", " + describeAge(winner) + ") is stamped and not shadowed by a newer sibling",
		}
	}
}

// newestShadowedNewerThan returns the newest entry among shadowed that is
// provably newer than winner, and whether any such entry exists. "Newer" is
// decided by commit time when both binaries are stamped (the intrinsic, trusted
// signal), falling back to file mtime otherwise. An unstamped winner has no
// commit time, so any stamped sibling is compared by mtime — enough to say "a
// fresher file is being shadowed" without over-claiming a commit ordering we
// cannot prove.
func newestShadowedNewerThan(winner PathBinary, shadowed []PathBinary) (PathBinary, bool) {
	var best PathBinary
	found := false
	for _, s := range shadowed {
		if !binaryNewer(s, winner) {
			continue
		}
		if !found || binaryNewer(s, best) {
			best = s
			found = true
		}
	}
	return best, found
}

// binaryNewer reports whether a is provably newer than b. It prefers commit time
// (both must be stamped with a parseable vcs.time), and falls back to file mtime.
// When neither ordering is decidable it returns false — the doctor never invents
// an age it cannot witness.
func binaryNewer(a, b PathBinary) bool {
	if a.Stamped && b.Stamped && !a.Dirty && !b.Dirty && a.Commit != "" && a.Commit == b.Commit {
		return false
	}
	if a.Stamped && b.Stamped {
		at, aok := parseStampTime(a.CommitTime)
		bt, bok := parseStampTime(b.CommitTime)
		if aok && bok {
			return at.After(bt)
		}
	}
	at, aok := parseStampTime(a.ModTime)
	bt, bok := parseStampTime(b.ModTime)
	if aok && bok {
		return at.After(bt)
	}
	return false
}

// describeAge renders the human age tag for a PATH binary: its short commit and
// commit date when stamped, or an explicit "unstamped" note plus the file mtime
// when not — so the line never reads as if an unstamped binary has a known
// commit.
func describeAge(b PathBinary) string {
	if b.Stamped {
		short := b.Commit
		if len(short) > 12 {
			short = short[:12]
		}
		if b.CommitTime != "" {
			s := short + " @ " + b.CommitTime
			if b.Dirty {
				s += " +uncommitted"
			}
			return s
		}
		return short
	}
	if b.ModTime != "" {
		return "unstamped, file mtime " + b.ModTime
	}
	return "unstamped"
}

func parseStampTime(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}
