package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// guard_pi.go — the first-class `fak guard -- pi` wiring for Pi (earendil-works). It is the
// Anthropic-wire twin of installGuardCodexConfig (guard_codex.go): the piece that actually
// routes the wrapped Pi child through the in-process kernel gateway, on the wire Pi natively
// speaks (Anthropic Messages).
//
// Why Pi needs its own install path. The other Anthropic-wire agent, Claude Code, reads
// ANTHROPIC_BASE_URL, so guardInjectedEnv repoints it with an env var alone. Pi does NOT: its
// Anthropic client takes the base URL from provider config, not the environment
// (packages/ai/src/api/anthropic-messages.ts — `baseURL: model.baseUrl`). An injected
// ANTHROPIC_BASE_URL is therefore ignored and Pi would talk straight to api.anthropic.com,
// bypassing the gateway. Pi's supported repoint is instead an extension: `pi -e <ext.ts>`
// loads a TypeScript module, and `pi.registerProvider("anthropic", { baseUrl })` — with no
// `models` — preserves every existing Claude model and swaps only the endpoint. So to put the
// kernel in front of Pi we write a session-scoped extension that repoints the anthropic
// provider at the gateway origin and prepend `-e <path>` to the child command. Pi then POSTs
// `/v1/messages` at the gateway, every proposed tool call crosses the same capability floor as
// the Claude path, and the gateway proxies upstream on the Anthropic wire.
//
// Credential posture. Pi is on the Anthropic wire, so — unlike Codex — no eager auth-env
// injection is needed: the child keeps its own ANTHROPIC_OAUTH_TOKEN / ANTHROPIC_API_KEY (or
// ~/.pi/agent/auth.json) and sends it to the gateway origin exactly as Claude Code does under
// ANTHROPIC_BASE_URL; guard holds the real upstream credential and proxies with it. The
// extension only rewrites the endpoint, never the auth.
//
// Pi contract this wiring depends on (verified against earendil-works/pi, main, 2026-07):
//   - Endpoint comes from provider config, NOT env: packages/ai/src/api/anthropic-messages.ts
//     builds `new Anthropic({ baseURL: model.baseUrl })`; no ANTHROPIC_BASE_URL / process.env
//     base-url read anywhere in the anthropic client. Hence the -e extension, not an env var.
//   - baseUrl is a BARE ORIGIN, no /v1: the @anthropic-ai/sdk appends `/v1/messages` itself;
//     packages/ai/src/providers/anthropic.ts defaults `baseUrl: "https://api.anthropic.com"`.
//     A /v1 suffix double-appends to `/v1/v1/messages` → 404 (earendil-works/pi issue #1777).
//   - registerProvider("anthropic", { baseUrl }) with NO `models` is a partial override that
//     PRESERVES the existing Claude model list and swaps only the endpoint (merge, not replace):
//     docs/custom-provider.md "When only baseUrl and/or headers are provided (no models), all
//     existing models for that provider are preserved with the new endpoint" (PR #3651).
//   - `-e <ext.ts>` is a global CLI flag loaded before the trust decision (docs/extensions.md),
//     so it may precede the prompt: `pi -e ./ext.ts <prompt>`.
// The guard_pi_test.go assertions witness the emitted extension string; the bullets above pin
// the external runtime contract those assertions assume, so a future reader can re-verify it.

// guardPiExtensionFileName is the session-scoped extension file `-e` points at. jiti loads it
// as TypeScript with no compile step (docs/extensions.md), so a plain `.ts` is enough.
const guardPiExtensionFileName = "fak-pi-provider.ts"

// guardIsPi reports whether the wrapped agent takes the Pi `extension` repoint — the
// `-e <ext.ts>` registerProvider override installGuardPiExtension prepends. Like guardIsCodex,
// it delegates to the profile registry (a harness gets the extension repoint iff its
// HarnessProfile declares RepointExtension, which today is exactly the pi profile), so the
// SELECTION is data (profile.Repoint), not a hardcoded base-name check.
func guardIsPi(command string) bool {
	return guardProfileHasRepoint(command, harnessprofile.RepointExtension)
}

// guardPiInstall records what the Pi extension injection did, for the banner and tests.
type guardPiInstall struct {
	Applied       bool
	ExtensionPath string
	BaseURL       string
	Reason        string
}

// installGuardPiExtension rewrites a Pi child command to route through the gateway by writing a
// session-scoped extension that registers the anthropic provider at the gateway origin and
// prepending `-e <path>` to the child. A non-Pi agent, or enabled=false, is returned unchanged
// with no install performed, so the path is inert for every other wrapped agent. An empty
// command is a no-op.
func installGuardPiExtension(command []string, enabled bool, gwURL string) ([]string, guardPiInstall, error) {
	if !enabled {
		return command, guardPiInstall{Reason: "disabled"}, nil
	}
	if len(command) == 0 || !guardIsPi(command[0]) {
		return command, guardPiInstall{Reason: "non-pi-child"}, nil
	}
	dir, err := guardSessionTempDir("pi")
	if err != nil {
		return command, guardPiInstall{}, err
	}
	return installGuardPiExtensionAt(command, gwURL, dir)
}

// installGuardPiExtensionAt is installGuardPiExtension with the session directory injected, so
// tests can assert on the written extension without touching the OS temp dir. It performs the
// same Pi-only gate as installGuardPiExtension.
func installGuardPiExtensionAt(command []string, gwURL, dir string) ([]string, guardPiInstall, error) {
	if len(command) == 0 || !guardIsPi(command[0]) {
		return command, guardPiInstall{Reason: "non-pi-child"}, nil
	}
	if strings.TrimSpace(dir) == "" {
		return command, guardPiInstall{}, fmt.Errorf("empty Pi extension directory")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return command, guardPiInstall{}, err
	}
	base := guardPiBaseURL(gwURL)
	extPath := filepath.Join(dir, guardPiExtensionFileName)
	if err := writeGuardPiExtension(extPath, base); err != nil {
		return command, guardPiInstall{}, err
	}
	install := guardPiInstall{Applied: true, ExtensionPath: extPath, BaseURL: base}
	return appendPiExtensionArg(command, extPath), install, nil
}

// guardPiBaseURL is the base URL the extension hands Pi's anthropic provider: the bare gateway
// origin, with any trailing slash trimmed. It carries NO /v1 suffix — Pi's Anthropic client
// appends the Messages path (`/v1/messages`) itself, exactly as ANTHROPIC_BASE_URL does for
// Claude Code (see cmdGuard's `gwURL := "http://" + ln.Addr().String()`, a bare origin). So an
// injected gwURL that already carries a trailing slash never doubles up.
func guardPiBaseURL(gwURL string) string {
	return strings.TrimRight(strings.TrimSpace(gwURL), "/")
}

// writeGuardPiExtension writes the session-scoped Pi extension that repoints the anthropic
// provider at baseURL. Registering "anthropic" with only a baseUrl (no `models`) preserves
// every Claude model Pi already knows and swaps only the endpoint. The URL is JSON-encoded into
// the module so it is a safe, correctly-escaped TypeScript string literal.
//
// The file lives alone in a fresh per-session MkdirTemp dir and is written in full BEFORE the
// child launches and reads it once at startup, so there is no concurrent reader to tear a
// partial write — a plain 0600 write is sufficient (no temp+rename dance needed here).
func writeGuardPiExtension(path, baseURL string) error {
	data := []byte(guardPiExtensionSource(baseURL))
	return os.WriteFile(path, data, 0o600)
}

// guardPiExtensionSource renders the extension module. The default-exported factory calls
// pi.registerProvider("anthropic", { baseUrl }) — the override Pi flushes after the factory
// returns (docs/extensions.md) — so the anthropic provider's endpoint becomes the gateway while
// its model list is untouched.
func guardPiExtensionSource(baseURL string) string {
	url, _ := json.Marshal(baseURL) // string -> safe TS literal; never errors for a string
	return "// fak guard: session-scoped Pi repoint. Registers the anthropic provider at the\n" +
		"// in-process kernel gateway so every Pi request crosses the guard capability floor.\n" +
		"// Generated per session by installGuardPiExtension (cmd/fak/guard_pi.go); safe to delete.\n" +
		"export default function (pi) {\n" +
		"  pi.registerProvider(\"anthropic\", { baseUrl: " + string(url) + " });\n" +
		"}\n"
}

// appendPiExtensionArg inserts `-e <path>` immediately after the pi executable — before any
// prompt or user args, since Pi's `-e`/`--extension` is a global flag (docs/extensions.md:
// `pi -e ./path.ts`). Mirrors appendClaudeMCPConfigArg (guard_mcp.go).
func appendPiExtensionArg(command []string, extPath string) []string {
	if len(command) == 0 {
		return command
	}
	out := make([]string, 0, len(command)+2)
	out = append(out, command[0], "-e", extPath)
	return append(out, command[1:]...)
}

// printGuardPiNote explains the Pi repoint on the banner: the gateway the anthropic provider
// was repointed at, and the extension that carries it. Mirrors printGuardCodexNote /
// printGuardMCPNote (guard_codex.go / guard_mcp.go).
func printGuardPiNote(w io.Writer, in guardPiInstall) {
	if !in.Applied {
		return
	}
	fmt.Fprintf(w, "fak guard: Pi wired via -e extension (registerProvider anthropic baseUrl=%s) — every tool call crosses the kernel floor (ext %s)\n", in.BaseURL, in.ExtensionPath)
}
