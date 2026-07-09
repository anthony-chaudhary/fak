package main

import (
	"sort"
	"strconv"
	"strings"
)

// launcher_saver_parity.go — the on-by-default token-savers must survive EVERY launch surface,
// not just the two front doors the token-defaults scorecard reads.
//
// The scorecard (cmd/fak/token_defaults.go) derives each saver's on/off state from the guard.go
// and serve.go source, so it locks the interactive `fak guard -- claude` / `fak serve` path. But
// the fleet's own automated sessions do NOT reach guard through those front doors: the headless
// dispatch worker (dispatchtick.GuardedLaunchCommand → guardedDispatchCommand), the account
// switcher / ultracode launch (buildLaunchArgv, incl. --settings '{"ultracode":true}'), and the
// codex launcher (buildCodexLaunchArgv) each assemble their OWN `fak guard … --` argv. They front
// the same guard binary, so today they inherit guard's full default-on saver stack for free — but
// nothing pins that. A future edit that spliced `--ctx-view-budget 0` or `--vdso=false` into any
// of those builders would silently strip a saver from every headless/ultracode session while the
// front-door scorecard stayed green.
//
// guardArgvDisabledSavers is that missing invariant: given a guard-fronting argv it returns which
// on-by-default token-savers the argv would turn OFF. The launcher-parity regression test asserts
// it is empty for every launch surface, so "the savers are on by default" holds for our own
// automated fleet — not just for a human at the keyboard.

// guardSaverBudgetFlags maps each guard flag that governs an on-by-default budget saver onto the
// token-defaults lever key it controls. Each is armed by a positive budget and disabled by a
// value <= 0 (gateway.go: "CompactHistoryBudget, when > 0, wires…"; "if cfg.CtxViewBudget > 0";
// agent.ctxplan_seam: budget <= 0 falls back to the default — but an EXPLICIT 0 override on the
// guard argv is the disable form a launcher edit would introduce). Keys match the scorecard's
// lever keys (cmd/fak/token_defaults.go) so a reader can cross-reference the two locks.
var guardSaverBudgetFlags = map[string]string{
	"--compact-history-budget": "compacthistory",
	"--elide-result-bytes":     "elideresult",
	"--ctx-view-budget":        "ctxview",
}

// guardVDSOFlag is the boolean vDSO dedup fast-path flag (serve/guard default it true). Its
// disable form on a guard argv is `--vdso=<falsey>` (or the bare negated `--no-vdso`); a bare
// `--vdso` or `--vdso=true` keeps it on. `--vdso false` (space-separated) is NOT a disable: guard
// parses vDSO as a bare bool, so the following token is a positional, not the flag's value — the
// detector mirrors that parsing rather than guessing.
const guardVDSOFlag = "--vdso"

// debug-stats is deliberately NOT in either table. It is the observable per-turn cache/token
// layer, legitimately silenced by --quiet on a headless worker (no human is watching stderr), and
// silencing it costs zero tokens — it is observability, not a saver. Only levers whose omission
// actually forfeits token savings belong here.

// guardArgvDisabledSavers scans the guard-flag segment of a guard-fronting argv — the tokens
// before the first standalone "--", which is exactly what guard itself parses as its own flags —
// and returns the sorted set of on-by-default saver keys the argv would DISABLE. An argv with no
// "--" (an unguarded launch, or a bare command) has no guard-flag segment and returns nil. Pure:
// no I/O, so every launcher's real output is locked in a unit test without spawning anything.
func guardArgvDisabledSavers(argv []string) []string {
	seg := argv
	for i, a := range argv {
		if a == "--" {
			seg = argv[:i]
			break
		}
	}
	disabled := map[string]bool{}
	for i := 0; i < len(seg); i++ {
		name, val, hasEq := strings.Cut(seg[i], "=")
		if name == "--no-vdso" {
			disabled["vdso"] = true
			continue
		}
		if name == guardVDSOFlag {
			// Bare `--vdso` (or `--vdso=true`) keeps the saver on; only a false-y `--vdso=<v>` disables.
			if hasEq && valueIsFalsey(val) {
				disabled["vdso"] = true
			}
			continue
		}
		key, ok := guardSaverBudgetFlags[name]
		if !ok {
			continue
		}
		if !hasEq {
			// `--flag value`: the budget is the next token. A trailing flag with no value can't disable.
			if i+1 >= len(seg) {
				continue
			}
			val = seg[i+1]
			i++
		}
		if n, err := strconv.Atoi(strings.TrimSpace(val)); err == nil && n <= 0 {
			disabled[key] = true
		}
	}
	if len(disabled) == 0 {
		return nil
	}
	out := make([]string, 0, len(disabled))
	for k := range disabled {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// valueIsFalsey reports whether a --vdso=<v> value turns the flag off, matching the token set
// Go's flag package (strconv.ParseBool) treats as false plus the operator-friendly "no"/"off".
func valueIsFalsey(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "false", "0", "no", "off", "f":
		return true
	default:
		return false
	}
}
