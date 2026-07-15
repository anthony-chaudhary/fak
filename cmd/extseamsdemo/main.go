// Command extseamsdemo makes fak's extension choices and trust boundaries
// inspectable without loading third-party code.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

const schema = "fak-extension-seams/1"

type seam struct {
	Name       string `json:"name"`
	Purpose    string `json:"purpose"`
	Attachment string `json:"attachment"`
	Trust      string `json:"trust"`
	Failure    string `json:"failure"`
	HotPath    bool   `json:"hot_path"`
	Source     string `json:"source"`
	UseWhen    string `json:"use_when"`
}

type report struct {
	Schema    string `json:"schema"`
	Principle string `json:"principle"`
	Seams     []seam `json:"seams"`
}

func catalog() []seam {
	return []seam{
		{Name: "policy-bundle", Purpose: "declarative capability restrictions", Attachment: "manifest", Trust: "data", Failure: "fail-closed", Source: "internal/policy", UseWhen: "the extension can be expressed as restrictions, not executable code"},
		{Name: "agent-hook", Purpose: "user or agent-authored linters and automation", Attachment: "out-of-process", Trust: "untrusted", Failure: "declared per hook; security gates fail-closed", Source: "internal/hooks", UseWhen: "code is user-authored, agent-authored, replaceable, or needs process isolation"},
		{Name: "middleware", Purpose: "observe or adjudicate model and tool calls", Attachment: "in-process", Trust: "trusted-compiled", Failure: "observers fail-open; adjudicators fail-closed", HotPath: true, Source: "internal/metrics/middleware_contract.go", UseWhen: "the code must surround every call and ships in the trusted binary"},
		{Name: "kernel-abi", Purpose: "adjudicators, engines, storage, emitters, and witnesses", Attachment: "in-process", Trust: "trusted-compiled", Failure: "duplicate registration panics; gates fail-closed", HotPath: true, Source: "internal/abi/registry.go", UseWhen: "a low-level mechanism must participate in the frozen kernel contract"},
		{Name: "quality-oracle", Purpose: "task-specific quality checks and linters", Attachment: "in-process", Trust: "trusted-compiled", Failure: "unknown or duplicate oracle fails loudly", Source: "internal/quality/oracle.go", UseWhen: "a stable high-volume checker is reviewed and compiled with fak"},
		{Name: "trajectory-scorer", Purpose: "custom trajectory and dispatch scoring", Attachment: "in-process", Trust: "trusted-compiled", Failure: "invalid or duplicate scorer is refused", Source: "internal/trajctl/scorer.go", UseWhen: "a deterministic scorer feeds routing or trajectory control"},
		{Name: "console-pane", Purpose: "operator TUI panes and controls", Attachment: "in-process", Trust: "trusted-compiled", Failure: "duplicate pane panics", Source: "internal/tuiplugin/registry.go", UseWhen: "a reviewed pane must render inside the console"},
		{Name: "capability-resolver", Purpose: "skills, MCP tools, and A2A capabilities", Attachment: "lazy protocol", Trust: "adjudicated", Failure: "fault or capability negotiation is explicit", Source: "internal/capindex", UseWhen: "a capability body should page in only after discovery"},
		{Name: "compute-backend", Purpose: "accelerator and execution backends", Attachment: "in-process", Trust: "trusted-compiled", Failure: "registration validates identity", Source: "internal/compute/compute.go", UseWhen: "a reviewed backend needs zero-copy or device-local integration"},
		{Name: "improvement-proposal", Purpose: "agent-authored linters, skills, patches, and policy candidates", Attachment: "artifact", Trust: "untrusted", Failure: "fail-to-reject: no witness means no keep", Source: "internal/rsiloop", UseWhen: "an agent proposes an improvement; an independent witness decides keep or revert"},
	}
}

func buildReport() report {
	ss := append([]seam(nil), catalog()...)
	sort.Slice(ss, func(i, j int) bool { return ss[i].Name < ss[j].Name })
	return report{Schema: schema, Principle: "choose the least-privileged seam that works; executable in-process extensions are trusted binary code, while agent-authored improvements remain proposals until independently witnessed", Seams: ss}
}

func validate(r report) error {
	if r.Schema != schema || len(r.Seams) < 8 {
		return fmt.Errorf("incomplete catalog")
	}
	seen := map[string]bool{}
	for _, s := range r.Seams {
		if s.Name == "" || s.Purpose == "" || s.Attachment == "" || s.Trust == "" || s.Failure == "" || s.Source == "" || s.UseWhen == "" {
			return fmt.Errorf("seam %q has an empty contract field", s.Name)
		}
		if seen[s.Name] {
			return fmt.Errorf("duplicate seam %q", s.Name)
		}
		seen[s.Name] = true
		if s.Attachment == "in-process" && s.Trust != "trusted-compiled" {
			return fmt.Errorf("in-process seam %q is not marked trusted-compiled", s.Name)
		}
	}
	for _, required := range []string{"agent-hook", "improvement-proposal", "kernel-abi", "policy-bundle"} {
		if !seen[required] {
			return fmt.Errorf("missing required seam %q", required)
		}
	}
	return nil
}

func run(out, errw io.Writer, args []string) int {
	fs := flag.NewFlagSet("extseamsdemo", flag.ContinueOnError)
	fs.SetOutput(errw)
	jsonOut := fs.Bool("json", false, "emit the versioned machine-readable catalog")
	selfcheck := fs.Bool("selfcheck", false, "validate the catalog and print a proof summary")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(errw, "extseamsdemo: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	r := buildReport()
	if err := validate(r); err != nil {
		fmt.Fprintf(errw, "extseamsdemo: %v\n", err)
		return 1
	}
	if *selfcheck {
		fmt.Fprintf(out, "PASS %s: %d seams; custom linters isolated; improvements witness-gated; in-process code trusted-compiled\n", r.Schema, len(r.Seams))
		return 0
	}
	if *jsonOut {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			fmt.Fprintf(errw, "extseamsdemo: encode: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(out, "fak extension seams")
	fmt.Fprintln(out, r.Principle)
	fmt.Fprintln(out)
	fmt.Fprintf(out, "%-22s %-14s %-17s %s\n", "SEAM", "ATTACHMENT", "TRUST", "USE WHEN")
	for _, s := range r.Seams {
		fmt.Fprintf(out, "%-22s %-14s %-17s %s\n", s.Name, s.Attachment, s.Trust, s.UseWhen)
	}
	return 0
}

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }
