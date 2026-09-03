// guard_help.go — the curated `fak guard -h` overview.
//
// Taste contract (mirrors help.go's, for the front-door verb): a bare
// `fak guard -h` used to dump all 60+ flags alphabetically, each carrying a
// paragraph of internal implementation narrative (issue numbers, subsystem
// names like "the anchor-starved trap" or "resetScore SHADOW health") — the
// same wall a contributor debugging cache-TTL internals would see, handed to
// an operator who just ran the README's `fak guard -- claude` and wants to
// make one small adjustment. The shape now:
//
//	fak guard -h        -> the curated common-flag overview (below)
//	fak guard -h -all   -> the full flag reference, GROUPED into labeled sections
//
// Membership is a taste call, deliberately small — every flag not listed
// here is still real, still enforced, and still documented in full under
// -all; nothing is deleted or hidden, only the default DISPLAY is tiered.
// guard_help_test.go pins every curated name to a live flag and holds the
// overview under a screen, the same ratchet help_test.go runs for the
// top-level dispatcher.
package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
)

// guardCommonFlag is one curated line of the `fak guard -h` overview.
type guardCommonFlag struct {
	name  string
	blurb string
}

// guardCommonFlags is the curated flag list `fak guard -h` shows by default —
// the ones an operator reaches for in a normal session (the README/
// GETTING-STARTED examples all use one of these). Everything else — session
// budgets, Codex/MCP wiring, cache-TTL tuning, dojo mode, landlock, the
// restart/reset plumbing — is real and documented, just one `-all` away.
var guardCommonFlags = []guardCommonFlag{
	{"policy", "capability-floor manifest to enforce (default: the built-in floor; see --dump-policy)"},
	{"allow-tool", "grant one exact tool for this launch; hard danger and self-modification checks still apply (repeatable)"},
	{"provider", "upstream wire: anthropic|openai|gemini|xai (default: auto-detected from the agent name)"},
	{"api-key-env", "opt IN to API billing using this env var (default: your subscription / passthrough)"},
	{"probe", "one-shot smoke mode: prove the guarded wire without requiring a task handoff"},
	{"gguf", "run a small model in-kernel, no API key or network (e.g. --gguf qwen2.5:7b)"},
	{"local", "auto-detect an already-running local model server (Ollama / LM Studio / llama.cpp)"},
	{"log", "write per-request + per-verdict structured logs to a file (or '-' for stderr)"},
	{"audit", "change where the decision journal is written ('off' disables it)"},
	{"dump-policy", "print the built-in capability floor (an editable manifest) and exit"},
}

// guardFlagGroup is one labeled section of the `fak guard -h -all` reference.
// The front door carries 65+ flags and grows; a flat alphabetical dump makes an
// operator hunt. Grouping the whole surface into stable, purpose-named sections
// keeps it navigable AS the feature surface increases — the operator scans to
// the section they want (auth, token savers, budgets, …) instead of a wall. The
// membership below is the ONLY hand-maintained part; each flag's rendered
// default + usage is pulled from the LIVE FlagSet, so it can never drift from
// the binary's real behavior.
type guardFlagGroup struct {
	title string
	flags []string
}

// guardFlagGroups partitions EVERY registered guard flag into exactly one
// labeled section. The invariant (every live flag appears in exactly one group,
// and every listed flag is live) is locked by guard_help_test.go: a flag added
// without a home lands in the visible guardUngroupedTitle section, which the
// test forbids — so the surface stays categorized as it grows. Order here is the
// display order: the wire + floor first, the rarely-touched plumbing last.
var guardFlagGroups = []guardFlagGroup{
	{"Upstream wire & auth", []string{
		"addr", "provider", "base-url", "model", "api-key-env",
		"anthropic-oauth", "oauth-token-env", "env", "require-key-env", "rotate",
	}},
	{"Policy, floor & audit", []string{
		"policy", "allow-tool", "dump-policy", "audit", "log", "landlock-hooks", "toolcall-control",
	}},
	{"Token economy (cache & context savers)", []string{
		"compact-history-budget", "compact-anchor-head", "assume-session-turns",
		"compact-solvency-floor",
		"elide-result-bytes", "elide-stale-reads", "ctx-view-budget", "managed-cache", "compress",
		"vcache-anchor", "defer-cold-tools",
	}},
	{"Session lifecycle hooks (Claude Code)", []string{
		"precompact-hook", "deny-all-continue", "toolproc-hooks", "task-handoff", "task-handoff-file",
		"task-handoff-repo", "task-handoff-live", "operator-directed", "host-recovery",
	}},
	{"Budgets, resets & session governance", []string{
		"context-budget-tokens", "max-duration", "budget-envelope",
		"reset-on-budget", "restart-on-budget", "restart-limit", "restart-seed-dir", "restart-seed-handback",
		"session-id", "session-pressure-gate",
	}},
	{"Lease admission & ownership", []string{
		"lease",
	}},
	{"Local in-kernel model", []string{
		"gguf", "local", "alongside", "backend", "tokenizer", "remote-serve",
		"native-admission-token-budget",
		"native-qwen-q4k-prefill-chunk-tokens", "native-qwen35-metal-gdn-sequence",
		"native-q4k-gateup-slab", "native-prefix-profile", "vulkan-q4k-profile", "vulkan-stage-q4k",
	}},
	{"Child-harness wiring (Claude / Codex)", []string{
		"codex-config", "codex-home", "codex-loop-gate", "codex-loop-gate-limit",
		"codex-loop-gate-since-hours", "mcp-register", "pi-extension", "expose-profile", "output-profile", "work-profile",
	}},
	{"Fleet control bus", []string{
		"fleet-bus",
	}},
	{"Observability & UI", []string{
		"banner", "quiet", "split", "split-where", "split-interval", "split-dry-run",
		"debug-stats", "resource-stats", "child-max-memory-mb", "child-resource-poll", "child-resource-journal", "dojo",
	}},
	{"Diagnostics & replay", []string{
		"probe", "replay-trace", "replay-wire",
	}},
}

// guardUngroupedTitle labels the section that catches any registered flag NOT
// assigned to a group above. On a categorized tree it is never rendered; when it
// IS, guard_help_test.go fails, forcing the new flag into a real group. The help
// screen still SHOWS the flag (never hides a knob) — it just flags the omission.
const guardUngroupedTitle = "Other (uncategorized — add to guardFlagGroups in guard_help.go)"

// guardIsZeroDefault reports whether a flag's default is the zero value for its
// type (the case Go's own flag.PrintDefaults omits "(default …)" for). Every
// guard flag type — string/bool/int/int64/uint/float64/duration — renders its
// zero as one of these strings, so this matches stdlib's reflective isZeroValue
// for the whole live set without reflection.
func guardIsZeroDefault(def string) bool {
	switch def {
	case "", "false", "0", "0s":
		return true
	}
	return false
}

// guardFlagIsString reports whether a flag's value is a string (so its default
// is shown quoted, `(default "x")`, exactly as flag.PrintDefaults does). It uses
// only the exported flag.Getter, so it needs no unexported stdlib types.
func guardFlagIsString(f *flag.Flag) bool {
	g, ok := f.Value.(flag.Getter)
	if !ok {
		return false
	}
	_, isStr := g.Get().(string)
	return isStr
}

// writeGuardFlagRef renders one flag exactly as flag.PrintDefaults would — the
// single-dash name, the value-type placeholder, the indented usage, and the
// DEFAULT shown inline (quoted for strings) so the operator sees what they get
// with no flag. Reproducing the stdlib format byte-for-byte means grouping only
// adds headers + reorders; no per-flag rendering regresses.
func writeGuardFlagRef(w io.Writer, f *flag.Flag) {
	typeName, usage := flag.UnquoteUsage(f)
	head := "  -" + f.Name
	if typeName != "" {
		head += " " + typeName
	}
	fmt.Fprintln(w, head)
	line := "    \t" + strings.ReplaceAll(usage, "\n", "\n    \t")
	if !guardIsZeroDefault(f.DefValue) {
		if guardFlagIsString(f) {
			line += fmt.Sprintf(" (default %q)", f.DefValue)
		} else {
			line += fmt.Sprintf(" (default %v)", f.DefValue)
		}
	}
	fmt.Fprintln(w, line)
}

// printGuardAllGrouped prints the full flag reference grouped into the labeled
// sections above, rendered from the LIVE FlagSet so every default + usage is the
// binary's real value. Any registered flag not assigned to a group falls into a
// visible guardUngroupedTitle section — surfaced, never hidden — which the test
// forbids, so a newly added flag must be categorized. That ratchet is what keeps
// the front door navigable as the flag surface grows.
func printGuardAllGrouped(w io.Writer, fs *flag.FlagSet) {
	grouped := map[string]bool{}
	for _, g := range guardFlagGroups {
		var flags []*flag.Flag
		for _, name := range g.flags {
			if f := fs.Lookup(name); f != nil {
				flags = append(flags, f)
				grouped[name] = true
			}
		}
		if len(flags) == 0 {
			continue
		}
		fmt.Fprintf(w, "\n%s:\n", g.title)
		for _, f := range flags {
			writeGuardFlagRef(w, f)
		}
	}
	var other []*flag.Flag
	fs.VisitAll(func(f *flag.Flag) {
		if !grouped[f.Name] {
			other = append(other, f)
		}
	})
	if len(other) > 0 {
		fmt.Fprintf(w, "\n%s:\n", guardUngroupedTitle)
		for _, f := range other {
			writeGuardFlagRef(w, f)
		}
	}
}

// printGuardLaunchPostures renders the launch switches that are PEELED off argv
// in cmdGuard instead of being registered on the FlagSet, so fs.VisitAll can
// never see them and printGuardAllGrouped can never render them. Without this
// section they would be undiscoverable from `fak guard -h -all` — the same
// problem the footer already solves for `-all` itself.
//
// It goes in the `-h -all` reference and NOT the curated overview on purpose:
// the overview is held to a 20-line budget (TestGuardHelpOverviewStaysCompact),
// which exists so this help does not grow back into a wall, and a posture an
// operator opts into deliberately belongs in the full reference.
func printGuardLaunchPostures(w io.Writer) {
	fmt.Fprintln(w, "\nLaunch postures (peeled from argv before the flag parse, so they are not in the count below):")
	fmt.Fprintln(w, "  --core-lock-all")
	fmt.Fprintln(w, "        session-wide RATCHET (#5423): for the life of this launch NO channel may WIDEN the")
	fmt.Fprintln(w, "        capability floor — not an operator allow/deny overlay edit picked up by the watcher,")
	fmt.Fprintln(w, "        not POST /v1/fak/policy/reload, not a --policy file swap, and not")
	fmt.Fprintln(w, "        FAK_POLICY_RELOAD_ALLOW_WIDEN=1, which this posture outranks. Tighten-only and")
	fmt.Fprintln(w, "        no-op amendments still apply normally, so the floor can only ever get stricter.")
	fmt.Fprintln(w, "        Pass it before the `--`; an occurrence after it belongs to the wrapped agent.")
}

// guardFlagCount counts every flag registered on fs, for the "N flags in
// this build" footer — so the footer can never drift from the real set.
func guardFlagCount(fs *flag.FlagSet) int {
	n := 0
	fs.VisitAll(func(*flag.Flag) { n++ })
	return n
}

// printGuardUsage prints `fak guard`'s usage: the fixed synopsis/examples,
// then either the curated common-flag overview or, with all=true (`fak
// guard -h -all`), the full flag reference grouped into labeled sections.
func printGuardUsage(w io.Writer, fs *flag.FlagSet, commandName string, all bool) {
	fmt.Fprintf(w, "usage: fak %s [flags] [--] <agent command...>\n", commandName)
	if commandName == "guard" {
		fmt.Fprintln(w, "  deprecated: use fak manage (or fak m); guard remains a compatibility alias")
	}
	fmt.Fprintf(w, "  e.g. fak %s claude\n", commandName)
	fmt.Fprintf(w, "       fak %s --provider openai -- codex\n", commandName)
	fmt.Fprintf(w, "       fak %s --policy my-floor.json -- claude\n", commandName)
	fmt.Fprintf(w, "       fak %s allow <tool> | disable [--reason TEXT] | policy explain|diff   # operator subcommands (-h on each)\n", commandName)
	if all {
		printGuardAllGrouped(w, fs)
		printGuardLaunchPostures(w)
		fmt.Fprintf(w, "\n%d flags in this build, grouped above. docs/fak/api-reference.md has the deep dive.\n", guardFlagCount(fs))
		return
	}
	fmt.Fprintln(w, "\ncommon flags:")
	for _, f := range guardCommonFlags {
		fmt.Fprintf(w, "  --%-14s %s\n", f.name, f.blurb)
	}
	fmt.Fprintf(w, "\n%d flags in this build. 'fak %s -h -all' lists every one grouped; docs/fak/api-reference.md has the deep dive.\n", guardFlagCount(fs), commandName)
}

// guardArgvHasAll reports whether argv requests the FULL flag reference —
// `-all` / `--all` anywhere in argv — scanned before fs.Parse so the right
// Usage closure is wired up before the flag package's own -h/-help
// short-circuit can fire mid-parse.
func guardArgvHasAll(argv []string) bool {
	for _, a := range argv {
		if a == "-all" || a == "--all" {
			return true
		}
	}
	return false
}
