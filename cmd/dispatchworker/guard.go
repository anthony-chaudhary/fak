// guard.go — front each dispatch worker with the kernel (`fak guard`), a Go port
// of the dogfood-guard family in tools/dispatch_worker.py.
//
// A dispatch worker IS the highest-volume dev work on a fleet node, and the LIVE
// production path is this compiled binary (dos.toml `worker_launch_template =
// 'tools/.bin/dispatchworker --lane {lane}'`, preferred over the Python sibling).
// Before this file the Go path talked STRAIGHT to the provider API — the kernel
// adjudicated NONE of the concurrent dispatch fleet, even though the Python sibling
// had guarded-by-default since #... . That made "the dispatch fleet dogfoods fak
// guard" true only on a path nothing runs in production.
//
// Fronting the worker with `fak guard` puts the SAME kernel `fak serve` runs in
// front of every tool call the worker proposes (deny by structure, repair malformed
// args, quarantine poisoned results) and records every verdict in a durable,
// hash-chained DECISION JOURNAL under the gitignored .dispatch-runs/guard-audit/ —
// so the fleet eats the product on the real workflow, WITH a witness. Default ON;
// opt out per node with FLEET_DOGFOOD_GUARD=0. resolveFakBin fails OPEN to an
// unwrapped worker on a host that has not built `fak`, so the default never breaks
// dispatch.
//
// The pure functions mirror dispatch_worker.py 1:1 so the ported guard_test table is
// a parity witness; only the OS touches (process env, crypto/rand token) differ.
package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"
)

// guardOffValues: a FLEET_DOGFOOD_GUARD whose normalized value is one of these turns
// dogfood guard OFF (mirrors dispatch_worker.GUARD_OFF_VALUES). Empty string is an
// explicit off so `FLEET_DOGFOOD_GUARD=` reads as a disable, not an unset.
var guardOffValues = map[string]struct{}{
	"0": {}, "off": {}, "false": {}, "no": {}, "": {}, "disable": {}, "disabled": {},
}

// guardTimeoutFloorS raises a guarded worker's gateway planner/write timeouts so a
// long frontier turn (extended thinking) is not TRUNCATED at fak serve's default
// 60s/90s floors. Mirrors dispatch_worker.GUARD_TIMEOUT_FLOOR_S.
const guardTimeoutFloorS = 600

// guardEnabled reports whether to front a worker with `fak guard`. Dogfood-by-default
// (ON); a node opts out with FLEET_DOGFOOD_GUARD in {0,off,false,no,"",disable,disabled}.
// Mirrors dispatch_worker.guard_enabled: an ABSENT key is ON, a present-but-off value
// is OFF.
func guardEnabled(env map[string]string) bool {
	raw, ok := env["FLEET_DOGFOOD_GUARD"]
	if !ok {
		return true
	}
	_, off := guardOffValues[strings.ToLower(strings.TrimSpace(raw))]
	return !off
}

// resolveFakBin locates a `fak` binary to front the worker with, or "" (fail OPEN).
// Precedence: $FAK_BIN (if it exists) -> the in-tree tools/.bin/fak[.exe] the dogfood
// launcher builds -> `fak` on the supplied env's PATH. "" means the caller launches
// the worker UNWRAPPED rather than breaking dispatch on a host that has not built fak.
// Mirrors dispatch_worker.resolve_fak_bin.
func resolveFakBin(workspace string, env map[string]string) string {
	if explicit := strings.TrimSpace(env["FAK_BIN"]); explicit != "" && fileExists(explicit) {
		return explicit
	}
	exe := "fak"
	if runtime.GOOS == "windows" {
		exe = "fak.exe"
	}
	intree := filepath.Join(workspace, "tools", ".bin", exe)
	if fileExists(intree) {
		return intree
	}
	// Honor the supplied env's PATH (so the env param fully governs resolution); an
	// ABSENT PATH key falls back to the process PATH via exec.LookPath.
	pathVal, hasPath := env["PATH"]
	return whichOnExactPath("fak", pathVal, hasPath)
}

// whichOnExactPath resolves name using exactly pathVal (when hasPath), honoring
// PATHEXT for command shims on Windows. When the PATH key is absent (hasPath=false)
// it falls back to exec.LookPath over the process PATH. Mirrors
// dispatch_worker._which_on_exact_path; returns "" for "not found" (Python None).
func whichOnExactPath(name, pathVal string, hasPath bool) string {
	if !hasPath {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
		return ""
	}
	suffixes := []string{""}
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		pathext := os.Getenv("PATHEXT")
		if pathext == "" {
			pathext = ".COM;.EXE;.BAT;.CMD"
		}
		for _, ext := range strings.Split(pathext, string(os.PathListSeparator)) {
			if ext == "" {
				continue
			}
			suffixes = append(suffixes, strings.ToLower(ext), strings.ToUpper(ext))
		}
	}
	for _, dir := range strings.Split(pathVal, string(os.PathListSeparator)) {
		if dir == "" {
			continue
		}
		for _, suf := range suffixes {
			cand := filepath.Join(dir, name+suf)
			if fileExists(cand) {
				return cand
			}
		}
	}
	return ""
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// guardProvider is the upstream wire `fak guard` proxies for a worker backend.
// claude -> the Anthropic API (passthrough/subscription); every other backend is
// OpenAI-wire. Mirrors dispatch_worker.guard_provider.
func guardProvider(backend string) string {
	if backend == "claude" {
		return "anthropic"
	}
	return "openai"
}

// guardAuditPath is a PER-SESSION durable decision journal under the gitignored
// .dispatch-runs/guard-audit/. The filename is keyed on the lane+backend (for
// globbing) PLUS a per-process discriminator (pid + a random token), because
// `fak guard`'s hash-chained journal has NO inter-process lock: two concurrent
// workers sharing ONE file would braid two independent sha256 chains into a forked,
// unverifiable journal. A unique-per-session file lets each `fak guard` own its own
// valid chain; the rollup globs the lane prefix to aggregate them. Mirrors
// dispatch_worker.guard_audit_path.
func guardAuditPath(workspace, lane, backend string) string {
	safe := sanitizeAuditName(lane + "-" + backend)
	token := fmt.Sprintf("%d-%s", os.Getpid(), randToken())
	return filepath.Join(workspace, ".dispatch-runs", "guard-audit", safe+"-"+token+".jsonl")
}

// sanitizeAuditName keeps the lane/backend prefix globbable: alnum and -_. survive,
// everything else (path separators, spaces) becomes _. Mirrors the Python
// comprehension in guard_audit_path.
func sanitizeAuditName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		} else {
			b.WriteRune('_')
		}
	}
	return b.String()
}

// randToken returns 8 hex chars from crypto/rand — the per-session discriminator that
// keeps two workers on the SAME lane from colliding on one journal file. Date.now /
// rand are fine here (off the resume-cacheable workflow path); a rand failure falls
// back to a pid-derived token (still unique enough paired with the pid prefix).
func randToken() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%08x", os.Getpid())
	}
	return hex.EncodeToString(b[:])
}

// guardWrap fronts a raw worker argv with `fak guard -- <worker>` so the kernel
// adjudicates every tool call. Pure given fakBin. Returns the command UNCHANGED when:
//
//   - fakBin is "" (no binary resolved -> fail open), or
//   - the backend fronts a LOCAL upstream we have not been told the base URL of.
//     claude proxies the public Anthropic API (passthrough/subscription) with no
//     base-URL override; opencode (and friends) front a local server (e.g. a GLM
//     endpoint), so guard would MISROUTE them to the provider's public API unless
//     FLEET_DOGFOOD_GUARD_BASEURL names that upstream. We refuse to misroute.
//
// Mirrors dispatch_worker.guard_wrap.
func guardWrap(command []string, fakBin, lane, backend, workspace string, env map[string]string) []string {
	if len(command) == 0 || fakBin == "" {
		return command
	}
	provider := guardProvider(backend)
	var extra []string
	if backend != "claude" {
		base := strings.TrimSpace(env["FLEET_DOGFOOD_GUARD_BASEURL"])
		if base == "" {
			return command // don't misroute a local-upstream worker
		}
		extra = []string{"--base-url", base}
	}
	audit := guardAuditPath(workspace, lane, backend)
	out := []string{fakBin, "guard", "--provider", provider}
	out = append(out, extra...)
	out = append(out, "--audit", audit, "--")
	out = append(out, command...)
	return out
}

// guardEnvAugment ensures a guarded worker's gateway won't truncate a long frontier
// turn: it sets FAK_PLANNER_TIMEOUT_S / FAK_HTTP_WRITE_TIMEOUT_S to a generous floor
// when unset (an explicit operator value is left as-is). Mutates and returns env.
// Mirrors dispatch_worker.guard_env_augment.
func guardEnvAugment(env map[string]string) map[string]string {
	for _, key := range []string{"FAK_PLANNER_TIMEOUT_S", "FAK_HTTP_WRITE_TIMEOUT_S"} {
		if env[key] == "" {
			env[key] = strconv.Itoa(guardTimeoutFloorS)
		}
	}
	return env
}

// guardedLaunchCommand resolves the argv to actually launch: command fronted by
// `fak guard` when dogfood mode is on and a fak binary resolves, else command
// unchanged. Returns (launchCommand, guarded) so callers can both run it and report
// what ran. env defaults to the process environment when nil. Mirrors
// dispatch_worker.guarded_launch_command.
func guardedLaunchCommand(command []string, lane, backend, workspace string, env map[string]string) ([]string, bool) {
	if env == nil {
		env = processEnvMap()
	}
	fakBin := ""
	if guardEnabled(env) {
		fakBin = resolveFakBin(workspace, env)
	}
	if fakBin == "" {
		return command, false
	}
	wrapped := guardWrap(command, fakBin, lane, backend, workspace, env)
	return wrapped, !sliceEqual(wrapped, command)
}

// processEnvMap snapshots the process environment as a map (the default env for the
// guard helpers, mirroring Python's os.environ default).
func processEnvMap() map[string]string {
	m := map[string]string{}
	for _, kv := range os.Environ() {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			m[kv[:i]] = kv[i+1:]
		}
	}
	return m
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
