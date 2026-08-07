package benchloop

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchruns"
)

// NoReuseEnv, set to a truthy value, forces the next benchmark run to execute even
// when a prior catalog run already covers the same lineage — the force-rerun escape
// hatch #4600 requires (its explicit non-goal fence). It is the in-lane, operator-
// and test-settable switch; a cmd-lane `--no-reuse` flag that sets it is a follow-up.
const NoReuseEnv = "FAK_BENCH_NO_REUSE"

// minCommitLen is the shortest hex run treated as a git commit when one is parsed out
// of a run_id/path. Git short SHAs are >=12 (the fleet harness embeds 16); an 8-digit
// date token (…-20260618) is thus never mistaken for a commit.
const minCommitLen = 12

// LineageKey is a benchmark run's reuse identity: the git commit that built the code
// under test, the machine it ran on, and its model/config. Two runs are
// interchangeable — one may be skipped in favor of the other — only when these agree.
// Model and Precision are wildcards when empty, so a caller that knows only
// (commit × machine) still matches any config recorded for that commit on that box.
type LineageKey struct {
	Commit    string
	Machine   string
	Model     string
	Precision string
}

// ReuseVerdict is the launch-gate verdict at benchloop.chooseAction: whether the
// next run can be skipped because a prior catalog run already covers this lineage.
type ReuseVerdict struct {
	Reuse  bool   `json:"reuse"`
	RunID  string `json:"run_id,omitempty"`
	Path   string `json:"path,omitempty"`
	Commit string `json:"commit,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// LineageReuse decides whether any prior run in the catalog makes the next run at
// key redundant. It is the reuse freshness predicate #4600 asks for: a prior run is
// reusable only when it was built at the SAME commit (a code/harness/config change
// moves HEAD, so a commit-exact match is the conservative realization of
// benchcli.DetectInvalidation's code/model/config-drift rule — a drifted prior run
// simply fails to match and is re-run) AND ran on the same machine AND, when the
// caller pins them, the same model/precision. The most recent covering run wins, so
// the reused artifact is the freshest one for that lineage. An empty key.Commit is a
// hard no-reuse: without a current commit there is no lineage to match.
func LineageReuse(runs []benchruns.Run, key LineageKey) ReuseVerdict {
	if key.Commit == "" {
		return ReuseVerdict{Reason: "no current commit lineage; must run"}
	}
	var best benchruns.Run
	for _, r := range runs {
		if !coversLineage(r, key) {
			continue
		}
		if best == nil || runString(r, "timestamp") > runString(best, "timestamp") {
			best = r
		}
	}
	if best == nil {
		return ReuseVerdict{Commit: key.Commit, Reason: "no prior run covers commit " + shortCommit(key.Commit) + " on " + blank(key.Machine)}
	}
	return ReuseVerdict{
		Reuse:  true,
		RunID:  runString(best, "run_id"),
		Path:   runString(best, "path"),
		Commit: key.Commit,
		Reason: "prior run " + runString(best, "run_id") + " already covers commit " + shortCommit(key.Commit) + " on " + blank(key.Machine) + "; skip the re-run",
	}
}

// taskConfig resolves the benchmark config the SELECTED nightrun task would run under
// — the model and precision its Run command names — so the launch gate can key reuse
// on that config instead of leaving both axes wildcard (#5087). A nightrun Task
// carries no config fields; its Run command is the only place it declares one, and
// these are the flags whose values land in the catalog run's model/precision columns:
//
//	-model / --model NAME       the served model id (fak serve, fak macbench, swebench)
//	-name  / --name  NAME       the bench report's model name (modelbench et al.)
//	-dir   / --dir   PATH       a local export dir; its basename is the report name
//	                            modelbench derives when -name is absent
//	-precision / --precision P  the recorded precision label
//	-quant     / --quant     P  the same axis spelled as a quantization — value form
//	                            only, so a valueless boolean -quant pins nothing
//
// Both spellings of a value are read (`-f v` and `-f=v`), and an explicit model name
// beats the -dir fallback wherever the two disagree. An axis the Run does not name
// stays "" — the pre-#5087 wildcard — so a task that pins nothing is neither over- nor
// under-matched. An operator fill-me-in placeholder (<glm-5.2.gguf>) or a shell
// variable ($FAK_GGUF) names no concrete config and is ignored.
func taskConfig(run string) (model, precision string) {
	var dirModel string
	fields := strings.Fields(run)
	for i, f := range fields {
		if !strings.HasPrefix(f, "-") {
			continue
		}
		name, val, eq := strings.Cut(strings.TrimLeft(f, "-"), "=")
		if !eq && i+1 < len(fields) && !strings.HasPrefix(fields[i+1], "-") {
			val = fields[i+1]
		}
		if val = configValue(val); val == "" {
			continue
		}
		switch strings.ToLower(name) {
		case "model", "name":
			if model == "" {
				model = val
			}
		case "dir":
			if dirModel == "" {
				dirModel = pathBase(val)
			}
		case "precision", "quant":
			if precision == "" {
				precision = val
			}
		}
	}
	if model == "" {
		model = dirModel
	}
	return model, precision
}

// configValue normalizes one parsed flag value and rejects the non-concrete ones: an
// empty value, an operator placeholder, and a shell variable all name no config.
func configValue(v string) string {
	v = strings.TrimSpace(strings.Trim(v, `"'`))
	if v == "" || strings.ContainsAny(v, "<>$") {
		return ""
	}
	return v
}

// pathBase is the last element of a path-valued flag, tolerating either separator: a
// nightrun Run is written for the box that will execute it, not for the box reading it.
func pathBase(v string) string {
	v = strings.TrimRight(strings.ReplaceAll(v, `\`, "/"), "/")
	if i := strings.LastIndex(v, "/"); i >= 0 {
		return v[i+1:]
	}
	return v
}

// coversLineage reports whether catalog run r is a valid reuse for key: commit must
// match, and each of machine/model/precision must match unless the key leaves it a
// wildcard (empty).
func coversLineage(r benchruns.Run, key LineageKey) bool {
	if !commitMatch(runCommit(r), key.Commit) {
		return false
	}
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

// runCommit derives the git commit a catalog run was built at. An explicit lineage
// field on the entry wins; otherwise it falls back to the commit the fleet harness
// embeds in the run_id/path (…-bench-fleet-<sha>-…) as the longest hex token.
func runCommit(r benchruns.Run) string {
	for _, k := range []string{"git_commit", "fak_commit", "commit"} {
		if v := runString(r, k); v != "" && v != "unknown" {
			return v
		}
	}
	return longestHexToken(runString(r, "run_id") + "-" + runString(r, "path"))
}

// commitMatch reports whether two commit strings name the same commit, tolerating the
// different truncations in play (the current HEAD is a full 40-hex SHA; the catalog
// embeds 16; a legacy field may carry 12) via a case-insensitive prefix match. Both
// must be at least minCommitLen so a short/empty token can never spuriously match.
func commitMatch(a, b string) bool {
	a, b = strings.ToLower(strings.TrimSpace(a)), strings.ToLower(strings.TrimSpace(b))
	if len(a) < minCommitLen || len(b) < minCommitLen {
		return false
	}
	if len(a) <= len(b) {
		return strings.HasPrefix(b, a)
	}
	return strings.HasPrefix(a, b)
}

// longestHexToken returns the longest run of hex digits of length >= minCommitLen in
// s (splitting on any non-hex byte), or "" when none qualifies. A date token like
// 20260618 (8 chars) never qualifies, so it is not mistaken for a commit.
func longestHexToken(s string) string {
	best := ""
	cur := strings.Builder{}
	flush := func() {
		if cur.Len() >= minCommitLen && cur.Len() > len(best) {
			best = cur.String()
		}
		cur.Reset()
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') {
			cur.WriteByte(c)
			continue
		}
		flush()
	}
	flush()
	return strings.ToLower(best)
}

// shortCommit renders at most the first 12 chars of a commit for a human reason line.
func shortCommit(c string) string {
	if len(c) > 12 {
		return c[:12]
	}
	return c
}

// reuseEnabled reports whether the reuse gate is active: it is on by default and
// disabled only when the operator sets NoReuseEnv to a truthy value.
func reuseEnabled(noReuseEnv string) bool {
	v := strings.ToLower(strings.TrimSpace(noReuseEnv))
	switch v {
	case "", "0", "false", "no", "off":
		return true
	default:
		return false
	}
}
