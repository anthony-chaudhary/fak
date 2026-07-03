// env.go — the shared env-map → exec.Cmd.Env slice helper. Both dispatch
// launchers (cmd/dispatchworker and the fak dispatch tick) build a child
// process environment from a map and need the same deterministic ordering so
// launch payloads and hermetic launch-witness tests compare stably across
// runs; this is the one copy (#1419).
package procguard

import "sort"

// EnvSlice flattens env into sorted "KEY=VALUE" entries suitable for
// exec.Cmd.Env. A nil or empty map yields an empty (non-nil) slice, and the
// sort makes the order deterministic regardless of map iteration.
func EnvSlice(env map[string]string) []string {
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	return out
}
