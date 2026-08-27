package main

import (
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/dropin"
)

// ---------------------------------------------------------------------------
// The terminal twins of the browser gallery. Same dropin resolution, rendered
// to stdout so the drop-in story lands in ~30s with zero setup (no browser, no
// port). -print is the full gallery; -agent is the single-agent dry run;
// -selfcheck is the browserless invariant check the CI dog-foods.
// ---------------------------------------------------------------------------

// renderPlan writes one resolved drop-in plan as an indented block: the command you
// type, the wire it resolves to, the upstream base URL, and the env var(s) injected
// into the child. Shared by -print (per agent) and -agent (one agent).
func renderPlan(p demoui.Palette, display, invocation string, plan dropin.Plan) {
	wire := wireLabel(plan.Provider)
	detect := ""
	if plan.Autodetected {
		detect = p.Paint(p.Dim, "  (wire autodetected from the name)")
	} else if plan.Recognized {
		detect = p.Paint(p.Dim, "  (recognized)")
	} else {
		detect = p.Paint(p.Yellow, "  (not autodetected — wire from --provider/anthropic fallback)")
	}
	dot := p.Paint(p.Green, "●")
	fmt.Printf("  %s %s\n", dot, p.Paint(p.Bold, display))
	fmt.Printf("      %s %s\n", p.Paint(p.Dim, "run     "), p.Paint(p.Cyan, invocation))
	fmt.Printf("      %s %s%s\n", p.Paint(p.Dim, "wire    "), wire, detect)
	base := plan.BaseURL
	if base == "" {
		base = p.Paint(p.Yellow, "(no public default — pass --base-url)")
	}
	fmt.Printf("      %s %s\n", p.Paint(p.Dim, "upstream"), base)
	for i, kv := range plan.EnvVars {
		label := "injects "
		if i > 0 {
			label = "        "
		}
		val := p.Paint(p.Green, kv[0]+"="+kv[1])
		suffix := ""
		if i == 0 {
			suffix = p.Paint(p.Dim, "   (child process only — your shell is untouched)")
		}
		fmt.Printf("      %s %s%s\n", p.Paint(p.Dim, label), val, suffix)
	}
}

// runPrint renders the whole drop-in gallery to stdout. Returns a process exit code.
func runPrint(gwURL string) int {
	p := demoui.TerminalPalette(os.Stdout)
	agents := dropin.KnownAgents()

	fmt.Printf("\n  %s\n", p.Paint(p.Bold, "fak · drop it in front of the agent you already run"))
	fmt.Printf("  %s\n\n", p.Paint(p.Dim, fmt.Sprintf(
		"one command · one static binary · zero code change — %d autodetected entry points", len(agents))))

	for _, a := range agents {
		plan := dropin.PlanFor(a.Command, "", "", gwURL)
		renderPlan(p, a.Display, "fak guard -- "+a.Command, plan)
		fmt.Printf("      %s\n\n", p.Paint(p.Dim, a.Note))
	}

	fmt.Printf("  %s\n", strings.Repeat("─", 72))
	fmt.Printf("  %s\n", p.Paint(p.Bold, "the long tail — any tool that lets you set a base URL"))
	fmt.Printf("  %s\n", p.Paint(p.Dim, "  not autodetected? name the wire explicitly, same one command:"))
	fmt.Printf("      %s\n", p.Paint(p.Cyan, "fak guard --provider openai -- <your-cli>"))
	fmt.Printf("  %s\n", p.Paint(p.Dim, "  an IDE / GUI agent (Cursor, Cline, Continue, Zed)? point its base URL at:"))
	fmt.Printf("      %s\n", p.Paint(p.Cyan, "fak serve --addr 127.0.0.1:8080   →   set the base URL to http://127.0.0.1:8080"))
	fmt.Printf("  %s\n\n", p.Paint(p.Dim, "  44 surveyed harnesses, frameworks, backends & protocols: docs/integrations/compatibility-matrix.md"))
	return 0
}

// runAgent prints exactly what `fak guard [--provider P] -- <agentCmd>` would wire —
// the single-agent dry run. Any command name works, not only the autodetected ones, so
// an operator can preview the wiring for their own CLI before launching it.
func runAgent(agentCmd, providerFlag, gwURL string) int {
	p := demoui.TerminalPalette(os.Stdout)
	plan := dropin.PlanFor(agentCmd, providerFlag, "", gwURL)
	invocation := "fak guard "
	if strings.TrimSpace(providerFlag) != "" {
		invocation += "--provider " + strings.TrimSpace(providerFlag) + " "
	}
	invocation += "-- " + agentCmd

	fmt.Printf("\n  %s\n\n", p.Paint(p.Bold, "fak guard — drop-in plan (dry run, nothing launched)"))
	renderPlan(p, agentCmd, invocation, plan)
	fmt.Println()
	return 0
}

// runSelfcheck resolves every entry point through internal/dropin and asserts the
// documented drop-in invariants — the exact PlanFor path the browser and -print drive.
// Returns 0 when every invariant holds, 1 on any mismatch. No browser, no network.
func runSelfcheck(gwURL string) int {
	fmt.Printf("== dropindemo -selfcheck: resolve each entry point through internal/dropin (browserless) ==\n")
	fmt.Printf("example gateway: %s\n\n", gwURL)

	agents := dropin.KnownAgents()
	failed := 0
	fail := func(name, msg string) {
		failed++
		fmt.Printf("  %-14s FAIL   %s\n", name, msg)
	}

	for _, a := range agents {
		plan := dropin.PlanFor(a.Command, "", "", gwURL)
		provider, recognized := dropin.DetectProvider(a.Command)
		var miss []string
		if !recognized {
			miss = append(miss, "DetectProvider does not recognize it")
		}
		if !plan.Autodetected {
			miss = append(miss, "not autodetected")
		}
		if plan.Provider != provider {
			miss = append(miss, fmt.Sprintf("provider=%q want %q", plan.Provider, provider))
		}
		// Env shape: anthropic = one var, bare host; openai = OpenAI aliases
		// with /v1 plus the bare Anthropic URL for the local Messages surface.
		switch plan.Provider {
		case "anthropic":
			if len(plan.EnvVars) != 1 || plan.EnvVars[0] != [2]string{"ANTHROPIC_BASE_URL", gwURL} {
				miss = append(miss, fmt.Sprintf("env=%v want one ANTHROPIC_BASE_URL=%s", plan.EnvVars, gwURL))
			}
		case "openai":
			wantV := strings.TrimRight(gwURL, "/") + "/v1"
			want := [][2]string{
				{"OPENAI_BASE_URL", wantV},
				{"OPENAI_API_BASE", wantV},
				{"ANTHROPIC_BASE_URL", gwURL},
			}
			if !reflect.DeepEqual(plan.EnvVars, want) {
				miss = append(miss, fmt.Sprintf("env=%v want %v", plan.EnvVars, want))
			}
		}
		if len(miss) > 0 {
			fail(a.Command, strings.Join(miss, "; "))
			continue
		}
		fmt.Printf("  %-14s PASS   fak guard -- %-9s → %s · injects %s\n",
			a.Command, a.Command, plan.Provider, plan.EnvVars[0][0])
	}

	// The negative invariants: an unknown agent with no flag falls back to anthropic
	// passthrough (NOT flagged autodetected), and an explicit --provider on an unknown
	// agent still resolves that wire (the universal `--provider openai -- <tool>`).
	if fb := dropin.PlanFor("some-unknown-cli", "", "", gwURL); fb.Provider != "anthropic" || fb.Autodetected || fb.Recognized {
		fail("fallback", fmt.Sprintf("unknown/no-provider = %q/auto=%v/recognized=%v, want anthropic/false/false", fb.Provider, fb.Autodetected, fb.Recognized))
	} else {
		fmt.Printf("  %-14s PASS   unknown agent, no --provider → anthropic passthrough (not autodetected)\n", "fallback")
	}
	if ex := dropin.PlanFor("some-unknown-cli", "openai", "", gwURL); ex.Provider != "openai" || ex.Autodetected || ex.Recognized {
		fail("explicit", fmt.Sprintf("unknown/--provider openai = %q/auto=%v/recognized=%v, want openai/false/false", ex.Provider, ex.Autodetected, ex.Recognized))
	} else {
		fmt.Printf("  %-14s PASS   unknown agent + --provider openai → openai wire (universal recipe)\n", "explicit")
	}

	fmt.Println()
	if failed > 0 {
		fmt.Printf("SELFCHECK FAILED — %d invariant(s) mismatched\n", failed)
		return 1
	}
	fmt.Printf("OK — %d autodetected entry point(s) + 2 universal-recipe invariants reproduced (browserless)\n", len(agents))
	return 0
}
