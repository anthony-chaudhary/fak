package armbench

// Ponytail Promptfoo reproduction records the pinned upstream benchmark inputs and
// independently attempted config/provider cells. It deliberately does not interpret
// comparator results as a fak value claim.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const PonytailPromptfooRevision = "2ed6c52c9d7e5e56942508591085fd45dea277d3"
const PonytailPromptfooVersion = "0.122.0"

type PromptfooInput struct {
	Path   string `json:"path"`
	Kind   string `json:"kind"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}
type PromptfooCell struct {
	Config     string   `json:"config"`
	Provider   string   `json:"provider"`
	Arms       []string `json:"arms"`
	Status     string   `json:"status"`
	ExitCode   int      `json:"exit_code"`
	ResultPath string   `json:"result_path,omitempty"`
	StdoutPath string   `json:"stdout_path"`
	StderrPath string   `json:"stderr_path"`
	Attempts   int      `json:"attempts"`
	Detail     string   `json:"detail,omitempty"`
	Command    []string `json:"command,omitempty"`
}
type PromptfooReproduction struct {
	Schema           string           `json:"schema"`
	Upstream         string           `json:"upstream"`
	Revision         string           `json:"revision"`
	PromptfooVersion string           `json:"promptfoo_version"`
	CapturedAt       string           `json:"captured_at"`
	Inputs           []PromptfooInput `json:"inputs"`
	Cells            []PromptfooCell  `json:"cells"`
	ValueClaim       string           `json:"fak_value_claim"`
	Complete         bool             `json:"all_declared_cells_attempted"`
}

var promptfooInputs = map[string]string{
	"benchmarks/promptfooconfig.yaml": "config", "benchmarks/promptfooconfig.gpt.yaml": "config", "benchmarks/promptfooconfig.gpt-newest.yaml": "config", "benchmarks/promptfooconfig.gemini.yaml": "config",
	"benchmarks/prompts.json": "prompt", "benchmarks/arms/baseline.js": "arm", "benchmarks/arms/caveman.js": "arm", "benchmarks/arms/ponytail.js": "arm", "benchmarks/arms/caveman-SKILL.md": "skill", "skills/ponytail/SKILL.md": "skill",
	"benchmarks/loc.js": "assertion", "benchmarks/correctness.js": "assertion",
}

type promptfooDecl struct {
	Config   string
	Provider string
	Arms     []string
}

var promptfooDecls = []promptfooDecl{
	{"promptfooconfig.yaml", "anthropic:messages:claude-haiku-4-5-20251001", []string{"baseline", "caveman", "ponytail"}},
	{"promptfooconfig.yaml", "anthropic:messages:claude-sonnet-4-6", []string{"baseline", "caveman", "ponytail"}},
	{"promptfooconfig.yaml", "anthropic:messages:claude-opus-4-8", []string{"baseline", "caveman", "ponytail"}},
	{"promptfooconfig.gpt.yaml", "openai:gpt-4.1-mini", []string{"baseline", "ponytail"}},
	{"promptfooconfig.gpt.yaml", "openai:gpt-5.4-mini", []string{"baseline", "ponytail"}},
	{"promptfooconfig.gpt-newest.yaml", "openai:gpt-5.5", []string{"baseline", "ponytail"}},
	{"promptfooconfig.gpt-newest.yaml", "openai:gpt-4.1-mini", []string{"baseline", "ponytail"}},
	{"promptfooconfig.gpt-newest.yaml", "openai:gpt-5.4-mini", []string{"baseline", "ponytail"}},
	{"promptfooconfig.gemini.yaml", "google:gemini-3.5-flash", []string{"baseline", "ponytail"}},
	{"promptfooconfig.gemini.yaml", "google:gemini-3.1-pro-preview", []string{"baseline", "ponytail"}},
}

func RunPonytailPromptfoo(source, outDir string, execute bool) (PromptfooReproduction, error) {
	if err := verifyPonytailRevision(source); err != nil {
		return PromptfooReproduction{}, err
	}
	return runPonytailPromptfoo(source, outDir, execute)
}

func verifyPonytailRevision(source string) error {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = source
	configureDispatchHelperCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("verify pinned Ponytail revision: %w", err)
	}
	got := strings.TrimSpace(string(b))
	if got != PonytailPromptfooRevision {
		return fmt.Errorf("Ponytail checkout revision %s; want pinned %s", got, PonytailPromptfooRevision)
	}
	return nil
}

func runPonytailPromptfoo(source, outDir string, execute bool) (PromptfooReproduction, error) {
	r := PromptfooReproduction{Schema: "fak.armbench.ponytail-promptfoo.v1", Upstream: "DietrichGebert/ponytail", Revision: PonytailPromptfooRevision, PromptfooVersion: PonytailPromptfooVersion, CapturedAt: time.Now().UTC().Format(time.RFC3339), ValueClaim: "none"}
	for path, kind := range promptfooInputs {
		b, err := os.ReadFile(filepath.Join(source, filepath.FromSlash(path)))
		if err != nil {
			return r, fmt.Errorf("required pinned input %s: %w", path, err)
		}
		h := sha256.Sum256(b)
		r.Inputs = append(r.Inputs, PromptfooInput{path, kind, hex.EncodeToString(h[:]), int64(len(b))})
	}
	sort.Slice(r.Inputs, func(i, j int) bool { return r.Inputs[i].Path < r.Inputs[j].Path })
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return r, err
	}
	for _, d := range promptfooDecls {
		c := PromptfooCell{Config: d.Config, Provider: d.Provider, Arms: d.Arms, Status: "not-attempted"}
		if execute {
			c = runPromptfooCell(source, outDir, d)
		}
		r.Cells = append(r.Cells, c)
	}
	r.Complete = true
	for _, c := range r.Cells {
		if c.Attempts != 1 {
			r.Complete = false
		}
	}
	return r, nil
}

var unsafeName = regexp.MustCompile(`[^a-zA-Z0-9._-]+`)

func runPromptfooCell(source, out string, d promptfooDecl) PromptfooCell {
	base := strings.TrimSuffix(d.Config, ".yaml") + "--" + unsafeName.ReplaceAllString(d.Provider, "-")
	c := PromptfooCell{Config: d.Config, Provider: d.Provider, Arms: d.Arms, Attempts: 1, StdoutPath: base + ".stdout.txt", StderrPath: base + ".stderr.txt", ResultPath: base + ".json"}
	args := []string{"promptfoo@" + PonytailPromptfooVersion, "eval", "-c", filepath.Join("benchmarks", d.Config), "--filter-providers", d.Provider, "--no-cache", "--max-concurrency", "1", "--output", filepath.Join(out, c.ResultPath)}
	c.Command = append([]string{"npx"}, args...)
	cmd := exec.Command("npx", args...)
	cmd.Dir = source
	configureDispatchHelperCommand(cmd)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if writeErr := os.WriteFile(filepath.Join(out, c.StdoutPath), []byte(stdout.String()), 0644); writeErr != nil {
		c.Status, c.Detail, c.ExitCode = "capture-failed", writeErr.Error(), -1
		return c
	}
	if writeErr := os.WriteFile(filepath.Join(out, c.StderrPath), []byte(stderr.String()), 0644); writeErr != nil {
		c.Status, c.Detail, c.ExitCode = "capture-failed", writeErr.Error(), -1
		return c
	}
	if err == nil {
		expectedRows := 5 * len(d.Arms)
		if got, checkErr := promptfooResultRows(filepath.Join(out, c.ResultPath)); checkErr != nil || got != expectedRows {
			c.Status, c.ExitCode = "incomplete-result", -1
			if checkErr != nil {
				c.Detail = checkErr.Error()
			} else {
				c.Detail = fmt.Sprintf("result rows %d; want %d", got, expectedRows)
			}
			return c
		}
		c.Status = "completed"
		return c
	}
	if strings.Contains(stdout.String(), "Missing ANTHROPIC_API_KEY") || strings.Contains(stdout.String(), "Missing GOOGLE_API_KEY") {
		c.Status = "unavailable"
	} else {
		c.Status = "failed"
	}
	c.Detail = err.Error()
	if ee, ok := err.(*exec.ExitError); ok {
		c.ExitCode = ee.ExitCode()
	} else {
		c.ExitCode = -1
	}
	return c
}

func promptfooResultRows(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read Promptfoo result: %w", err)
	}
	var doc struct {
		Results struct {
			Results []json.RawMessage `json:"results"`
		} `json:"results"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return 0, fmt.Errorf("parse Promptfoo result: %w", err)
	}
	return len(doc.Results.Results), nil
}

func MarshalPromptfooReproduction(r PromptfooReproduction) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
