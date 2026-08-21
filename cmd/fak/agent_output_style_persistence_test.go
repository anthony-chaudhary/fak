package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestAgentOutputStylePreferencePrecedenceAndRefusals(t *testing.T) {
	path := saveAgentOutputStylePreference(t, "caveman:high")

	persisted, err := resolveAgentOutputStylePreference(agentDefaultOutputStyle, false, path)
	if err != nil {
		t.Fatal(err)
	}
	if persisted.Style.Style != "caveman:native:high" || persisted.Source != agentOutputStyleSourcePersisted {
		t.Fatalf("persisted preference = %+v", persisted)
	}

	cli, err := resolveAgentOutputStylePreference("caveman:low", true, path)
	if err != nil {
		t.Fatal(err)
	}
	if cli.Style.Style != "caveman:native:low" || cli.Source != agentOutputStyleSourceCLI {
		t.Fatalf("CLI preference = %+v", cli)
	}

	shipped, err := resolveAgentOutputStylePreference(agentDefaultOutputStyle, false, t.TempDir()+"/missing.json")
	if err != nil {
		t.Fatal(err)
	}
	if shipped.Style.Style != "caveman:native:medium" || shipped.Source != agentOutputStyleSourceDefault {
		t.Fatalf("shipped preference = %+v", shipped)
	}

	offPath := saveAgentOutputStylePreference(t, "full")
	off, err := resolveAgentOutputStylePreference(agentDefaultOutputStyle, false, offPath)
	if err != nil {
		t.Fatal(err)
	}
	if off.Style.Style != "full" || off.Style.Applied || off.Source != agentOutputStyleSourcePersisted {
		t.Fatalf("off preference = %+v", off)
	}

	invalidPath := writeAgentOutputStyleConfig(t, `{"pane_defaults":{"adapt":{"output-style":"caveman:original:*"}}}`)
	if _, err := resolveAgentOutputStylePreference(agentDefaultOutputStyle, false, invalidPath); err == nil || !strings.Contains(err.Error(), "persisted") {
		t.Fatalf("invalid persisted preference error = %v", err)
	}
}

func TestPersistedAgentOutputStyleReachesOwnedBlockAndRunReport(t *testing.T) {
	path := saveAgentOutputStylePreference(t, "caveman:high")
	pref, err := resolveAgentOutputStylePreference(agentDefaultOutputStyle, false, path)
	if err != nil {
		t.Fatal(err)
	}
	restore, err := applyAgentOutputStyle(pref.Style)
	if err != nil {
		t.Fatal(err)
	}
	defer restore()

	planner := &systemCapturePlanner{inner: agent.NewMockPlanner("profile-witness")}
	result, _, err := agent.Run(
		context.Background(), planner, agent.DefaultTask, 12,
		agent.WithResponseProfileSource(pref.Source),
	)
	if err != nil {
		t.Fatal(err)
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(planner.system), &blocks); err != nil {
		t.Fatalf("decode owned system block: %v", err)
	}
	foundWitness := false
	for _, block := range blocks {
		if syspromptmmu.WitnessFor([]byte(block.Text)) == pref.Style.Witness {
			foundWitness = true
			break
		}
	}
	if !foundWitness {
		t.Fatalf("owned system block omitted persisted profile witness %q", pref.Style.Witness)
	}
	if result.OutputStyle != pref.Style.Style || result.OutputStyleSource != agentOutputStyleSourcePersisted || result.OutputStyleWitness != pref.Style.Witness {
		t.Fatalf("run report profile = style=%q source=%q witness=%q", result.OutputStyle, result.OutputStyleSource, result.OutputStyleWitness)
	}
}

func TestConsoleSettingsReportsCanonicalResponseProfileAndSource(t *testing.T) {
	path := saveAgentOutputStylePreference(t, "caveman:medium")
	report := buildTUIConfigReport(path, fixedTUITime(t))
	for _, setting := range report.Settings {
		if setting.Pane == "adapt" && setting.Control == "output-style" {
			if setting.Effective != "caveman:medium" || setting.Canonical != "caveman:native:medium" || setting.Source != "saved" {
				t.Fatalf("response profile setting = %+v", setting)
			}
			return
		}
	}
	t.Fatalf("adapt.output-style absent from settings: %+v", report.Settings)
}

func saveAgentOutputStylePreference(t *testing.T, value string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "console.json")
	var stdout, stderr bytes.Buffer
	if code := runTUIConfig(&stdout, &stderr, []string{"--path", path, "--set-default", "adapt.output-style=" + value, "--json"}); code != 0 {
		t.Fatalf("save response profile code=%d stderr=%s", code, stderr.String())
	}
	return path
}

func writeAgentOutputStyleConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "console.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

type systemCapturePlanner struct {
	inner  agent.Planner
	system string
}

func (p *systemCapturePlanner) Model() string { return p.inner.Model() }

func (p *systemCapturePlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if p.system == "" && len(messages) > 0 {
		p.system = messages[0].Content
	}
	return p.inner.Complete(ctx, messages, tools, opts...)
}

func fixedTUITime(t *testing.T) time.Time {
	t.Helper()
	got, err := parseTUITime("2026-08-20T12:00:00Z")
	if err != nil {
		t.Fatal(err)
	}
	return got
}
