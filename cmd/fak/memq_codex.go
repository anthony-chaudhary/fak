package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memq"
)

// memq_codex.go — the thin cmd surface for the read-only Codex memories backend in
// internal/memq. The backend itself (scan, label-external, gated page-in) lives in
// internal/memq/codexbackend.go; this file resolves the Codex home for the CLI and
// builds the backend, so `fak memory run --backend codex` can select the external
// recall corpus without expanding the gateway's trust boundary.
//
// WIRING NOTE (residual): the `--backend codex` selection itself belongs in
// cmd/fak/memory.go's memoryBackend (the backend dispatch seam), which is OUTSIDE
// this worker's declared lane (internal/memq/** + cmd/fak/memq_codex.go). The exact
// one-line change there is, inside memoryBackend, before the --dir branch:
//
//	if backend == "codex" { return codexMemoryBackend(codexHome, includeChronicle) }
//
// plus the `--backend`, `--codex-home`, and `--include-chronicle` flags on the
// `memory run` flag set. Shipped here: the backend, its driver, its tests, and this
// constructor — all callable the moment that one line lands.

// resolveCodexHome resolves the Codex home for the CLI: explicit flag > CODEX_HOME >
// the platform default (~/.codex). The platform default is only used when optIn is
// true — i.e. the operator explicitly selected the codex backend — so this never
// silently reaches into a real ~/.codex on an unrelated command. Returns the home
// and how it resolved ("flag" | "env" | "default" | "unresolved").
func resolveCodexHome(flagHome string, optIn bool) (home, source string) {
	if h := strings.TrimSpace(flagHome); h != "" {
		return expandHome(h), "flag"
	}
	if h := strings.TrimSpace(os.Getenv("CODEX_HOME")); h != "" {
		return expandHome(h), "env"
	}
	if optIn {
		if h, err := os.UserHomeDir(); err == nil && h != "" {
			return filepath.Join(h, ".codex"), "default"
		}
	}
	return "", "unresolved"
}

// expandHome expands a leading ~ in a user-supplied path (the CLI convention the
// rest of `fak memory` uses via pathutil.ExpandTilde; replicated here so this file
// stays self-contained within the declared lane).
func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") || strings.HasPrefix(p, `~\`) {
		if h, err := os.UserHomeDir(); err == nil && h != "" {
			return filepath.Join(h, strings.TrimLeft(p[1:], `/\`))
		}
	}
	return p
}

// codexMemoryBackend builds the read-only Codex memq backend for the CLI and a
// human label. It resolves the home (flag > CODEX_HOME > ~/.codex when opted in),
// constructs the backend (a missing/partial home yields an empty corpus, never an
// error), and labels it so the operator sees which external home was read. The
// backend treats every Codex file as external/untrusted and gates every page-in.
func codexMemoryBackend(flagHome string, includeChronicle bool) (memq.Backend, string) {
	home, source := resolveCodexHome(flagHome, true)
	b, err := memq.NewCodexBackend(home, includeChronicle)
	if err != nil {
		// NewCodexBackend does not error on a missing home; an error here is a real
		// failure, so surface it and fall back to an empty backend (fail-closed).
		fmt.Fprintf(os.Stderr, "fak memory: codex backend over %q: %v; using an empty external corpus\n", home, err)
		b, _ = memq.NewCodexBackend("", includeChronicle)
		home, source = "", "unresolved"
	}
	label := "codex memories (external/untrusted)"
	if home != "" {
		label = fmt.Sprintf("codex memories %s [%s]", home, source)
	} else {
		label = "codex memories (no home resolved; empty external corpus)"
	}
	if includeChronicle {
		label += " +chronicle"
	}
	return b, label
}
