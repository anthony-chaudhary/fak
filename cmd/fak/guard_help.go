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
//	fak guard -h -all   -> the full flag reference (fs.PrintDefaults, unchanged)
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
	{"provider", "upstream wire: anthropic|openai|gemini|xai (default: auto-detected from the agent name)"},
	{"api-key-env", "opt IN to API billing using this env var (default: your subscription / passthrough)"},
	{"probe", "one-shot smoke mode: prove the guarded wire without requiring a task handoff"},
	{"gguf", "run a small model in-kernel, no API key or network (e.g. --gguf qwen2.5:7b)"},
	{"local", "auto-detect an already-running local model server (Ollama / LM Studio / llama.cpp)"},
	{"log", "write per-request + per-verdict structured logs to a file (or '-' for stderr)"},
	{"audit", "change where the decision journal is written ('off' disables it)"},
	{"quiet", "suppress the startup banner and the exit audit summary"},
	{"dump-policy", "print the built-in capability floor (an editable manifest) and exit"},
	{"split", "auto|on|off: a live fak-info pane beside the agent (auto-on inside a multiplexer)"},
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
// guard -h -all`), the full alphabetical flag reference.
func printGuardUsage(w io.Writer, fs *flag.FlagSet, all bool) {
	fmt.Fprintln(w, "usage: fak guard [flags] -- <agent command...>")
	fmt.Fprintln(w, "  e.g. fak guard -- claude")
	fmt.Fprintln(w, "       fak guard --provider openai -- codex")
	fmt.Fprintln(w, "       fak guard --policy my-floor.json -- claude")
	if all {
		fmt.Fprintln(w)
		fs.PrintDefaults()
		return
	}
	fmt.Fprintln(w, "\ncommon flags:")
	for _, f := range guardCommonFlags {
		fmt.Fprintf(w, "  --%-14s %s\n", f.name, f.blurb)
	}
	fmt.Fprintf(w, "\n%d flags in this build. 'fak guard -h -all' lists every one; docs/fak/api-reference.md has the deep dive.\n", guardFlagCount(fs))
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
