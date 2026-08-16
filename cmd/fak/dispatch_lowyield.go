package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// dispatchLowYieldExcludes returns the #2062 low-yield soft-exclude lane set the
// Python fold (low_yield_soft_excludes) flags: lanes whose recent FINISHED sessions
// burned turns yet closed nothing on their tree. It shells the read-only emitter
// (tools/issue_resolve_dispatch.py --low-yield-excludes) so the SAME trust-gated fold
// that demotes a poison-pill lane inside the Python busiest-pick also steers this live
// Go wave/tick launcher, instead of the two paths disagreeing and the Go fleet happily
// re-storming a lane the fold already flagged (#4285).
//
// Fail-open throughout: a missing interpreter, a non-zero exit, or unparseable output
// all yield nil, so this advisory probe can never starve the picker. The caller merges
// the result into its exclude map only when NO explicit lane is pinned, so an operator
// --lane <flagged> still overrides the soft demote (parity with the Python picker's
// soft_exclude.discard(lane)).
func dispatchLowYieldExcludes(root string) map[string]bool {
	// Synthetic test repositories intentionally omit the legacy Python helper. Treat
	// that as no advisory exclusions instead of spawning interpreters that cannot
	// possibly succeed (and can race the picker's short-lived fixture state).
	if _, err := os.Stat(filepath.Join(root, "tools", "issue_resolve_dispatch.py")); err != nil {
		return nil
	}
	out, err := runPythonTool(root, []string{"tools/issue_resolve_dispatch.py", "--low-yield-excludes"})
	if err != nil {
		return nil
	}
	var doc struct {
		Exclude []string `json:"exclude"`
	}
	if err := json.Unmarshal(out, &doc); err != nil {
		return nil
	}
	if len(doc.Exclude) == 0 {
		return nil
	}
	m := make(map[string]bool, len(doc.Exclude))
	for _, lane := range doc.Exclude {
		if lane != "" {
			m[lane] = true
		}
	}
	return m
}
