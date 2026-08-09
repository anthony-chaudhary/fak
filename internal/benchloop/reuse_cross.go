package benchloop

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/benchruns"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// CrossCommit supplies the ingredients for the finer #5088 freshness predicate:
// hooks to decode a prior catalog run's benchmark_artifact envelope and to list
// the repo paths changed between that run's commit and the current HEAD, plus an
// optional identity for the run that would be launched. With both hooks set, a
// prior run at a DIFFERENT commit is still reusable when
// benchcli.DetectInvalidation judges the commit window untouched (no bench
// code/harness/model/config change). The zero value disables the cross-commit
// path entirely, leaving #4600's conservative commit-exact rule in force.
type CrossCommit struct {
	// Artifact decodes catalog run r's benchmark_artifact. ok=false means the
	// artifact is unreadable; that run then stays under the commit-exact rule.
	Artifact func(r benchruns.Run) (art benchcli.BenchmarkArtifact, ok bool)
	// ChangedPaths lists the paths changed between the prior run's commit and the
	// current HEAD (git diff --name-only prev..head). ok=false means the window
	// could not be resolved (unknown commit, no git) — fail closed, no reuse.
	ChangedPaths func(prevCommit, headCommit string) (paths []string, ok bool)
	// Next is the identity of the run that would be launched, feeding the
	// model/config drift rungs of DetectInvalidation. Zero-value model+config
	// carries the prior artifact's identity forward: only prior runs that already
	// clear the key's machine/model/precision axes are candidates here (#5087), so
	// the reused artifact IS the config it would re-run.
	Next benchcli.BenchmarkArtifact
}

// enabled reports whether the cross-commit path may run at all.
func (cc CrossCommit) enabled() bool { return cc.Artifact != nil && cc.ChangedPaths != nil }

// LineageReuseAcross is LineageReuse plus the #5088 cross-commit gate. The exact
// #4600 predicate runs first and still wins when it finds a same-commit covering
// run. Only when it does not — and both hooks are live — is the freshest prior
// run on the other lineage axes (machine/model/precision) re-judged through
// benchcli.DetectInvalidation over the actual changed paths of prevCommit..HEAD:
// reuse iff not IsInvalid. Every failure to resolve evidence (no artifact, no
// diff window) falls back to the conservative no-reuse verdict.
func LineageReuseAcross(runs []benchruns.Run, key LineageKey, cc CrossCommit) ReuseVerdict {
	exact := LineageReuse(runs, key)
	if exact.Reuse || !cc.enabled() || key.Commit == "" {
		return exact
	}
	var best benchruns.Run
	for _, r := range runs {
		if !coversAxes(r, key) {
			continue
		}
		prev := runCommit(r)
		if len(prev) < minCommitLen || commitMatch(prev, key.Commit) {
			continue
		}
		if best == nil || runString(r, "timestamp") > runString(best, "timestamp") {
			best = r
		}
	}
	if best == nil {
		return exact
	}
	prevCommit := runCommit(best)
	prevArt, ok := cc.Artifact(best)
	if !ok {
		return exact
	}
	changed, ok := cc.ChangedPaths(prevCommit, key.Commit)
	if !ok {
		return exact
	}
	next := cc.Next
	if next.Model == (benchcli.ModelSnapshot{}) && next.Config.Hash == "" {
		next.Model = prevArt.Model
		next.Config = prevArt.Config
	}
	runID := runString(best, "run_id")
	if inv := benchcli.DetectInvalidation(prevArt, next, changed); inv.IsInvalid {
		why := "invalidated"
		if inv.Reason != nil {
			why = *inv.Reason
		}
		return ReuseVerdict{
			Commit: key.Commit,
			Reason: "prior run " + runID + " at " + shortCommit(prevCommit) + " is invalidated by " + shortCommit(key.Commit) + ": " + why + "; must run",
		}
	}
	return ReuseVerdict{
		Reuse:  true,
		RunID:  runID,
		Path:   runString(best, "path"),
		Commit: key.Commit,
		Reason: "prior run " + runID + " at " + shortCommit(prevCommit) + " is still valid at " + shortCommit(key.Commit) + " (window touched no bench code/harness/model/config); skip the re-run",
	}
}

// coversAxes is coversLineage minus the commit axis: machine/model/precision must
// each match unless the key leaves them a wildcard (empty).
func coversAxes(r benchruns.Run, key LineageKey) bool {
	if key.Machine != "" && !strings.EqualFold(runString(r, "machine_id"), key.Machine) {
		return false
	}
	if key.Model != "" && !strings.EqualFold(runString(r, "model"), key.Model) {
		return false
	}
	if key.Precision != "" && !strings.EqualFold(runString(r, "precision"), key.Precision) {
		return false
	}
	return true
}

// DefaultCrossCommit wires the real evidence hooks for a repo checkout at root:
// the artifact hook reads the run's recorded path (a report file or a run
// directory) and decodes the first benchmark_artifact found; the changed-paths
// hook shells out to git diff --name-only prev..head. Both fail soft to !ok so
// LineageReuseAcross degrades to the commit-exact rule.
func DefaultCrossCommit(root string) CrossCommit {
	return CrossCommit{
		Artifact: func(r benchruns.Run) (benchcli.BenchmarkArtifact, bool) {
			p := runString(r, "path")
			if strings.TrimSpace(p) == "" {
				return benchcli.BenchmarkArtifact{}, false
			}
			abs := filepath.FromSlash(p)
			if !filepath.IsAbs(abs) {
				abs = filepath.Join(root, abs)
			}
			fi, err := os.Stat(abs)
			if err != nil {
				return benchcli.BenchmarkArtifact{}, false
			}
			if !fi.IsDir() {
				raw, err := os.ReadFile(abs)
				if err != nil {
					return benchcli.BenchmarkArtifact{}, false
				}
				return benchcli.DecodeArtifact(raw)
			}
			var art benchcli.BenchmarkArtifact
			found := false
			_ = filepath.WalkDir(abs, func(path string, d os.DirEntry, err error) error {
				if err != nil || found || d.IsDir() || !strings.HasSuffix(strings.ToLower(d.Name()), ".json") {
					return nil
				}
				raw, err := os.ReadFile(path)
				if err != nil {
					return nil
				}
				if a, ok := benchcli.DecodeArtifact(raw); ok {
					art, found = a, true
				}
				return nil
			})
			return art, found
		},
		ChangedPaths: func(prevCommit, headCommit string) ([]string, bool) {
			cmd := exec.Command("git", "-C", root, "diff", "--name-only", prevCommit+".."+headCommit)
			windowgate.ConfigureBackgroundCommand(cmd)
			out, err := cmd.Output()
			if err != nil {
				return nil, false
			}
			var paths []string
			for _, ln := range strings.Split(string(out), "\n") {
				if ln = strings.TrimSpace(ln); ln != "" {
					paths = append(paths, ln)
				}
			}
			return paths, true
		},
	}
}
