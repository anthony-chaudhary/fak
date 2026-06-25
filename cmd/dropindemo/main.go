// Command dropindemo is the splashy, on-box demo of fak's DISTRIBUTION story — the
// "drop-in" entry point. It answers one question in a glance: how little does it take
// to put the kernel in front of the coding agent you already run? The answer is a
// single command, one binary, zero code change:
//
//	fak guard -- claude                  # Claude Code, on your subscription, kernel-gated
//	fak guard -- codex                   # OpenAI Codex (wire autodetected)
//	fak guard -- opencode                # OpenCode
//	fak guard -- aider                   # Aider
//
// The gallery is a GALLERY OF THE REAL WIRING, not a brochure. Every card — the
// provider wire guard autodetects, the upstream base URL, the env var(s) injected into
// the CHILD process only — is computed live through internal/dropin, the same
// resolution `fak guard` runs at startup (the two are pinned identical by a shared test
// truth table; see internal/dropin). So what the demo shows is what guard does, by
// construction.
//
// Serve it (browser), or self-check it (headless — CI / cross-platform dog-food):
//
//	go run ./cmd/dropindemo -addr 127.0.0.1:8154
//	# open http://127.0.0.1:8154 → the entry-point gallery
//
//	go run ./cmd/dropindemo -print
//	# the 30-second point with ZERO setup: render the drop-in gallery as a colored
//	# table in the terminal (no browser, no port). Honors NO_COLOR.
//
//	go run ./cmd/dropindemo -agent codex
//	# the single-agent "dry run": print exactly what `fak guard -- codex` would wire.
//
//	go run ./cmd/dropindemo -selfcheck
//	# browserless: resolve every entry point through internal/dropin, assert the
//	# documented drop-in invariants, exit non-zero on any drift.
//
// It needs no model weights, no key, no network — only the pure wire resolution — so it
// reproduces identically on any box. The SAFETY-floor demo (what the gate blocks once
// it is in front) is its sibling cmd/guarddemo; this one is purely the ADOPTION axis.
package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/dropin"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-dropindemo-v1"

// exampleGateway is the illustrative loopback URL shown in the gallery's injected-env
// values. The real `fak guard` binds an OS-picked free port; this fixed value keeps the
// copy-paste env lines concrete and identical across runs (override with -gateway).
const exampleGateway = "http://127.0.0.1:8080"

// card is one entry-point tile: the agent's display metadata plus the wiring
// `fak guard -- <command>` resolves for it. Every wiring field is computed through
// dropin.PlanFor, never a hand-copied table, so a card cannot drift from guard.
type card struct {
	Command      string      `json:"command"`
	Display      string      `json:"display"`
	Note         string      `json:"note"`
	Home         string      `json:"home"`
	Invocation   string      `json:"invocation"`   // the exact line you type
	Provider     string      `json:"provider"`     // resolved upstream wire
	Wire         string      `json:"wire"`         // human label for that wire
	Autodetected bool        `json:"autodetected"` // wire inferred from the name (no --provider)
	BaseURL      string      `json:"base_url"`     // upstream public API base
	EnvVars      [][2]string `json:"env_vars"`     // env var(s) injected into the CHILD only
}

// wireLabel is the human one-liner for a provider wire, naming the HTTP route the
// client actually hits so the card explains why the /v1 suffix differs by wire.
func wireLabel(provider string) string {
	switch provider {
	case "anthropic":
		return "Anthropic Messages · POST /v1/messages"
	case "openai":
		return "OpenAI Chat Completions · POST /v1/chat/completions"
	default:
		return provider
	}
}

// gallery builds the resolved entry-point cards from dropin.KnownAgents(), each pointed
// at gwURL. The wiring comes entirely from dropin.PlanFor, so the gallery is a reading
// of what `fak guard -- <agent>` resolves — add an agent to dropin and it appears here.
func gallery(gwURL string) []card {
	agents := dropin.KnownAgents()
	out := make([]card, 0, len(agents))
	for _, a := range agents {
		p := dropin.PlanFor(a.Command, "", "", gwURL)
		out = append(out, card{
			Command:      a.Command,
			Display:      a.Display,
			Note:         a.Note,
			Home:         a.Home,
			Invocation:   "fak guard -- " + a.Command,
			Provider:     p.Provider,
			Wire:         wireLabel(p.Provider),
			Autodetected: p.Autodetected,
			BaseURL:      p.BaseURL,
			EnvVars:      p.EnvVars,
		})
	}
	return out
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	b, err := pageFS.ReadFile("page.html")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(b)
}

// handleGallery returns the resolved entry-point cards plus the hardware probe. The
// browser renders one tile per card; each tile is the live resolution, not a fixture.
func handleGallery(w http.ResponseWriter, r *http.Request) {
	gw := strings.TrimSpace(r.URL.Query().Get("gateway"))
	if gw == "" {
		gw = exampleGateway
	}
	cards := gallery(gw)
	writeJSON(w, map[string]any{
		"cards":           cards,
		"agent_count":     len(cards),
		"example_gateway": gw,
		"matrix":          "https://github.com/anthony-chaudhary/fak/blob/main/docs/integrations/compatibility-matrix.md",
		"hardware":        demoui.Probe(),
	})
}

func main() {
	const defaultAddr = "127.0.0.1:8154"
	addr := flag.String("addr", defaultAddr, "listen address")
	basePath := demoui.BasePathFlag(flag.CommandLine, "/dropin")
	print := flag.Bool("print", false, "render the drop-in entry-point gallery as a colored table in the TERMINAL (no browser, no port) and exit. The 30-second point with zero setup. Honors NO_COLOR.")
	selfcheck := flag.Bool("selfcheck", false, "run HEADLESS: resolve every entry point through internal/dropin (the same path the browser drives), assert the documented drop-in invariants, print a witness table, and exit non-zero on any mismatch. No browser, no network — the CI / cross-platform dog-food of this demo's data path.")
	agent := flag.String("agent", "", "print exactly what `fak guard -- <agent>` would wire for this one agent (the single-agent dry run), then exit. Any command name works, not only the autodetected ones.")
	provider := flag.String("provider", "", "with -agent, the explicit --provider to resolve as (mirrors `fak guard --provider P -- <agent>`); empty = autodetect from the name")
	gateway := flag.String("gateway", exampleGateway, "illustrative gateway URL shown in the injected-env values")
	flag.Parse()

	switch {
	case *print:
		os.Exit(runPrint(*gateway))
	case *agent != "":
		os.Exit(runAgent(*agent, *provider, *gateway))
	case *selfcheck:
		os.Exit(runSelfcheck(*gateway))
	}

	app := http.NewServeMux()
	app.HandleFunc("/", handleIndex)
	app.HandleFunc("/api/gallery", handleGallery)
	mux := http.NewServeMux()
	base := demoui.MountWithBasePath(mux, *basePath, app)

	bind := demoui.ListenAddr(*addr, defaultAddr)
	fmt.Fprintf(os.Stderr, "dropindemo %s on %s\n", version, demoui.LocalURL(bind, base))
	fmt.Fprintf(os.Stderr, "drop fak in front of the agent you already run: %d autodetected entry points\n", len(dropin.KnownAgents()))
	if base != "" {
		fmt.Fprintf(os.Stderr, "base path: %s (set by -base-path or %s)\n", base, demoui.DemoBasePathEnv)
	}
	if err := http.ListenAndServe(bind, mux); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}
