package harnessgallery

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Schema identifies the schema format for harness pack manifests.
const Schema = "fak.harness-pack/v1alpha1"

// Blueprint specifies a bounded starter pack definition, including problem statement,
// capability contracts, owned artifacts, and delivery spines.
type Blueprint struct {
	ID                   string   `json:"id"`
	Name                 string   `json:"name"`
	For                  string   `json:"for"`
	Problem              string   `json:"problem"`
	Today                string   `json:"today"`
	BetterBecause        string   `json:"better_because"`
	Witness              string   `json:"witness"`
	Seam                 string   `json:"seam"`
	RequiredCapabilities []string `json:"required_capabilities"`
	ExcludedCapabilities []string `json:"excluded_capabilities"`
	OwnedArtifacts       []string `json:"owned_artifacts"`
	TenMinuteSpine       []string `json:"ten_minute_spine"`
	WeekendExtension     []string `json:"weekend_extension"`
}

// Pack represents the serialized manifest written to harness.pack.json when initializing a pack.
type Pack struct {
	Schema               string   `json:"schema"`
	BlueprintID          string   `json:"blueprint_id"`
	Name                 string   `json:"name"`
	Seam                 string   `json:"seam"`
	SystemPrompt         string   `json:"system_prompt"`
	Task                 string   `json:"task"`
	RequiredCapabilities []string `json:"required_capabilities"`
	ExcludedCapabilities []string `json:"excluded_capabilities"`
	Witness              string   `json:"witness"`
	Upgrade              string   `json:"upgrade"`
}

// InitResult records the target directory and files created or preserved during pack initialization.
type InitResult struct {
	Directory string   `json:"directory"`
	Created   []string `json:"created"`
	Preserved []string `json:"preserved"`
}

// Builtins returns an isolated, sorted clone of all built-in starter blueprints.
func Builtins() []Blueprint {
	out := append([]Blueprint(nil), builtins...)
	for i := range out {
		out[i].RequiredCapabilities = append([]string(nil), out[i].RequiredCapabilities...)
		out[i].ExcludedCapabilities = append([]string(nil), out[i].ExcludedCapabilities...)
		out[i].OwnedArtifacts = append([]string(nil), out[i].OwnedArtifacts...)
		out[i].TenMinuteSpine = append([]string(nil), out[i].TenMinuteSpine...)
		out[i].WeekendExtension = append([]string(nil), out[i].WeekendExtension...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Find searches the built-in catalog for a blueprint matching the given ID.
func Find(id string) (Blueprint, bool) {
	for _, b := range builtins {
		if b.ID == id {
			return b, true
		}
	}
	return Blueprint{}, false
}

// Validate verifies structural completeness, non-overlap of capability rules,
// and artifact path safety across a slice of blueprints.
func Validate(blueprints []Blueprint) error {
	seen := map[string]bool{}
	for _, b := range blueprints {
		if b.ID == "" || b.Name == "" || b.For == "" || b.Problem == "" || b.Today == "" || b.BetterBecause == "" || b.Witness == "" || b.Seam == "" {
			return fmt.Errorf("blueprint %q is incomplete", b.ID)
		}
		if seen[b.ID] {
			return fmt.Errorf("duplicate blueprint id %q", b.ID)
		}
		seen[b.ID] = true
		if len(b.OwnedArtifacts) == 0 || len(b.TenMinuteSpine) == 0 || len(b.WeekendExtension) == 0 {
			return fmt.Errorf("blueprint %q has no bounded delivery path", b.ID)
		}
		excluded := map[string]bool{}
		for _, c := range b.ExcludedCapabilities {
			excluded[c] = true
		}
		for _, c := range b.RequiredCapabilities {
			if excluded[c] {
				return fmt.Errorf("blueprint %q both requires and excludes %q", b.ID, c)
			}
		}
		for _, path := range b.OwnedArtifacts {
			clean := filepath.Clean(filepath.FromSlash(path))
			if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
				return fmt.Errorf("blueprint %q has unsafe owned artifact %q", b.ID, path)
			}
		}
	}
	return nil
}

// Init scaffolds a blueprint into target dir, emitting harness.pack.json and README.md
// while idempotently preserving existing user-authored files.
func Init(id, dir string) (InitResult, error) {
	b, ok := Find(id)
	if !ok {
		return InitResult{}, fmt.Errorf("unknown blueprint %q", id)
	}
	if dir == "" {
		return InitResult{}, errors.New("--dir is required")
	}
	if err := Validate([]Blueprint{b}); err != nil {
		return InitResult{}, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return InitResult{}, err
	}
	pack := Pack{Schema: Schema, BlueprintID: b.ID, Name: b.Name, Seam: b.Seam, SystemPrompt: systemPrompt(b), Task: b.Problem, RequiredCapabilities: b.RequiredCapabilities, ExcludedCapabilities: b.ExcludedCapabilities, Witness: b.Witness, Upgrade: "fak harness gallery init --id " + b.ID + " --dir ."}
	body, err := json.MarshalIndent(pack, "", "  ")
	if err != nil {
		return InitResult{}, err
	}
	body = append(body, '\n')
	readme := renderREADME(b)
	result := InitResult{Directory: dir}
	for _, f := range []struct {
		name string
		body []byte
	}{{"harness.pack.json", body}, {"README.md", []byte(readme)}} {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); err == nil {
			result.Preserved = append(result.Preserved, f.name)
			continue
		} else if !os.IsNotExist(err) {
			return result, err
		}
		if err := os.WriteFile(path, f.body, 0o644); err != nil {
			return result, err
		}
		result.Created = append(result.Created, f.name)
	}
	return result, nil
}

func systemPrompt(b Blueprint) string {
	return "You are the " + b.Name + ". Use only declared capabilities; preserve exclusions. " + b.BetterBecause
}
func renderREADME(b Blueprint) string {
	return fmt.Sprintf(`# %s

This pack is a decision scaffold: it records the outcome, capability boundary, public
extension seam, and proof you should build toward. It is not a finished runnable harness.

## Why this pack exists

- **For:** %s
- **Problem:** %s
- **Today:** %s
- **Better because:** %s
- **Build from:** %s
- **Proof to capture:** %s

## Required capabilities

%s

## Explicitly excluded

%s

## Ten-minute spine

%s

## Weekend extension

%s

## What to do next

1. Read `+"`harness.pack.json`"+` and replace the sample task and system prompt with your real job.
2. Map each required capability to a concrete adapter; do not add an excluded capability by accident.
3. Build the ten-minute spine through **%s** before starting the weekend extensions.
4. Capture this proof: **%s**
5. Run `+"`fak harness gallery selfcheck`"+` to validate the built-in catalog.

Owned starter files are user-controlled. Running `+"`fak harness gallery init --id %s --dir .`"+`
again preserves both this README and the manifest.
`, b.Name, b.For, b.Problem, b.Today, b.BetterBecause, b.Seam, b.Witness, bullets(b.RequiredCapabilities), bullets(b.ExcludedCapabilities), numbered(b.TenMinuteSpine), numbered(b.WeekendExtension), b.Seam, b.Witness, b.ID)
}
func bullets(xs []string) string {
	if len(xs) == 0 {
		return "- none"
	}
	out := make([]string, len(xs))
	for i, x := range xs {
		out[i] = "- " + x
	}
	return strings.Join(out, "\n")
}
func numbered(xs []string) string {
	out := make([]string, len(xs))
	for i, x := range xs {
		out[i] = fmt.Sprintf("%d. %s", i+1, x)
	}
	return strings.Join(out, "\n")
}

var builtins = []Blueprint{
	{ID: "readonly-support", Name: "Readonly Support Desk", For: "a support team answering from an approved knowledge base", Problem: "draft grounded answers without refunds, account mutation, or arbitrary tools", Today: "use a general coding agent and rely on instructions not to mutate customer state", BetterBecause: "the pack requires retrieval and citations while structurally excluding write capabilities", Witness: "offline selfcheck emits a cited answer and denies refund_payment before any model call", Seam: "generated product config plus policy manifest", RequiredCapabilities: []string{"knowledge retrieval", "citation projection", "policy preflight"}, ExcludedCapabilities: []string{"payments write", "account mutation", "shell execution"}, OwnedArtifacts: []string{"harness.pack.json", "policy.json", "skills/support/SKILL.md"}, TenMinuteSpine: []string{"initialize the generated product", "set the support system prompt and task", "attach readonly policy", "run offline selfcheck and one denied write preflight"}, WeekendExtension: []string{"add a real knowledge adapter", "brand the browser UI", "archive grounded-answer and denial receipts"}},
	{ID: "coding-workspace", Name: "Local Coding Workspace", For: "a developer replacing a hosted coding-agent shell with a local fak-native loop", Problem: "edit and verify one repository while every command remains scoped and reviewable", Today: "hand a general agent broad shell access or keep using an external harness", BetterBecause: "workspace, command, diff, approval, and session boundaries are explicit public adapters", Witness: "browser session edits a fixture, renders the diff, runs its focused test, and survives restart", Seam: "public harnesskit UI plus workspace/tool adapters", RequiredCapabilities: []string{"workspace-scoped files", "allowlisted command execution", "diff artifacts", "approval events", "durable sessions"}, ExcludedCapabilities: []string{"filesystem outside workspace", "silent destructive commands", "ambient credentials"}, OwnedArtifacts: []string{"harness.pack.json", "ui/theme.json", "tools/workspace.json", "skills/coding/SKILL.md"}, TenMinuteSpine: []string{"initialize product and coding pack", "bind one workspace", "enable read, patch, diff, and focused-test tools", "run the fixture coding selfcheck"}, WeekendExtension: []string{"connect live native inference", "add model and effort selection", "capture restart-safe browser coding witness"}},
	{ID: "cited-research", Name: "Cited Research Notebook", For: "an analyst producing a review from frozen primary sources", Problem: "separate sourced observations from inference and never invent unavailable social evidence", Today: "collect links in chat and manually audit whether conclusions remain reproducible", BetterBecause: "source capture, citation requirements, and not-yet labels are part of the pack contract", Witness: "replay a frozen source bundle and reject an uncited claim or unavailable-X assertion", Seam: "skill pack plus trace and artifact sinks", RequiredCapabilities: []string{"source capture", "citation validation", "artifact export", "replay"}, ExcludedCapabilities: []string{"unsourced market claims", "mutable-source-only evidence", "fabricated social posts"}, OwnedArtifacts: []string{"harness.pack.json", "skills/research/SKILL.md", "sources/manifest.json"}, TenMinuteSpine: []string{"initialize research pack", "declare the question and source cutoff", "load one frozen source bundle", "run citation selfcheck"}, WeekendExtension: []string{"add a browser/search adapter", "compare one tuned alternative", "publish a reproducible evidence report"}},
	{ID: "incident-operations", Name: "Incident Operations Copilot", For: "an on-call operator diagnosing production without granting immediate remediation power", Problem: "summarize signals and propose bounded actions while preserving human approval and rollback", Today: "paste logs into a general assistant or expose broad infrastructure credentials", BetterBecause: "observation, proposal, approval, execution, and rollback are separate typed stages", Witness: "replay an incident fixture, produce a cited diagnosis, request approval, and deny an unapproved mutation", Seam: "policy, approval, telemetry, and secret-provider adapters", RequiredCapabilities: []string{"readonly telemetry", "runbook retrieval", "approval identity", "audit receipts"}, ExcludedCapabilities: []string{"unapproved mutation", "raw secret disclosure", "cross-tenant access"}, OwnedArtifacts: []string{"harness.pack.json", "policy.json", "skills/incident/SKILL.md", "runbooks/manifest.json"}, TenMinuteSpine: []string{"initialize incident pack", "bind readonly fixture telemetry", "load one runbook", "run diagnosis and denied-mutation selfcheck"}, WeekendExtension: []string{"connect staged remediation adapter", "add rollback receipt", "exercise an operator-approved sandbox incident"}},
}
