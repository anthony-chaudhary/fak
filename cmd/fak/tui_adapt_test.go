package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

func TestTUIAdaptCapturedRenderShowsDefaultOnAndAblationControls(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runTUIAdapt(&out, &errOut, []string{"--width", "100"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	got := out.String()
	for _, want := range []string{
		"Agent Adaptations · default posture and ablation controls",
		"response  ON · default",
		"caveman:native:medium",
		"fak agent --output-style full",
		"work      ON · default",
		"ponytail:native:medium",
		"fak agent --work-profile standard",
		"measure effects separately: fak console ablate",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("captured render missing %q:\n%s", want, got)
		}
	}
}

func TestTUIAdaptCapturedRenderShowsBothAxesAblated(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runTUIAdapt(&out, &errOut, []string{"--output-style", "full", "--work-profile", "standard", "--width", "100"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if got := strings.Count(out.String(), "OFF · ablated"); got != 2 {
		t.Fatalf("ablated rows=%d want 2:\n%s", got, out.String())
	}
}

func TestTUIAdaptJSONCarriesCanonicalDefaults(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runTUIAdapt(&out, &errOut, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got tuiAdaptReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Axes) != 2 || got.Axes[0].Canonical != "caveman:native:medium" || got.Axes[1].Canonical != "ponytail:native:medium" {
		t.Fatalf("report=%+v", got)
	}
}

func TestTUIAdaptPaneIsRegisteredWithAblationControls(t *testing.T) {
	pane, ok := tuiplugin.Lookup("adapt")
	if !ok {
		t.Fatal("adapt pane not registered")
	}
	defaults := map[string]string{}
	for _, control := range pane.Controls {
		defaults[control.ID] = control.Default
	}
	if len(pane.Controls) != 3 || defaults["output-style"] != agentDefaultOutputStyle || defaults["work-profile"] != agentDefaultWorkProfile {
		t.Fatalf("controls=%+v", pane.Controls)
	}
}
