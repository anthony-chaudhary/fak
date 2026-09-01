package main

import (
	"fmt"
	"io"
)

// guard_capabilities.go answers the guard-preamble half of #1500 (C2 of the
// #1494 self-knowledge epic): the wrapped agent should learn, from the guard's
// own startup banner, that a memory-forward "what can I do?" surface exists —
// without a live selfquery.Load() call at guard startup (that would add
// per-session catalog-build cost/fragility to every guard launch just to print
// a banner line). This mirrors printGuardMCPNote/printGuardCodexNote: a plain,
// static Printf, gated only on --quiet.

// printGuardCapabilitiesNote tells the wrapped agent, once per session, that
// `fak-dev capabilities` / fak_capabilities exists: the narrower, memory-forward
// twin of fak_feature_query (memq drivers, fak-dev index * verbs, and the kernel
// shared-path verbs), each card carrying the exact call to make.
func printGuardCapabilitiesNote(w io.Writer, mcpInstall guardMCPInstall) {
	if mcpInstall.Applied {
		fmt.Fprintf(w, "fak guard: self-describe — `fak-dev capabilities [<intent>]` or the fak_capabilities MCP tool lists the memory-forward toolbelt (memq drivers, fak-dev index * verbs, fak_changes/dos_arbitrate), each card carrying the exact call to make; before starting more shared-tree work, query WIP with `fak_capabilities` intent `WIP` or run `fak wip queue --json`\n")
		return
	}
	fmt.Fprintf(w, "fak guard: self-describe — `fak-dev capabilities [<intent>]` lists the memory-forward toolbelt (memq drivers, fak-dev index * verbs, fak_changes/dos_arbitrate), each card carrying the exact call to make; before starting more shared-tree work, run `fak-dev capabilities WIP` or `fak wip queue --json`\n")
}
