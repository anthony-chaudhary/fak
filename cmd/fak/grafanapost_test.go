package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

// writeFileForTest writes content to a path under the test's cwd (set by t.Chdir).
func writeFileForTest(t *testing.T, name, content string) error {
	t.Helper()
	return os.WriteFile(name, []byte(content), 0o644)
}

// grafanaCleanEnv blanks the grafana + scoreboard Slack keys and moves to a clean cwd
// so a dry-run renders without touching the dev box's real config.
func grafanaCleanEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"FAK_GRAFANA_TOKEN", "FAK_GRAFANA_CHANNEL",
		"FAK_SCOREBOARD_TOKEN", "FAK_SCOREBOARD_CHANNEL", "FAK_SCOREBOARD_SOURCE",
	} {
		t.Setenv(k, "")
	}
	t.Chdir(t.TempDir())
}

func TestGrafanaPostSnapshotDryRun(t *testing.T) {
	grafanaCleanEnv(t)
	var out, errb bytes.Buffer
	code := runGrafanaPost(&out, &errb, []string{
		"--snapshot", "--title", "p99 spike",
		"--url", "https://grafana.example/d/snap/xyz",
		"--dashboard", "FAK Gateway Observability", "--range", "last 6h",
		"--dry-run",
	})
	if code != 0 {
		t.Fatalf("snapshot dry-run exit = %d, stderr=%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{"grafana snapshot — p99 spike", "FAK Gateway Observability", "https://grafana.example/d/snap/xyz"} {
		if !strings.Contains(s, want) {
			t.Fatalf("snapshot dry-run missing %q:\n%s", want, s)
		}
	}
}

func TestGrafanaPostSnapshotRequiresURLWhenLive(t *testing.T) {
	grafanaCleanEnv(t)
	var out, errb bytes.Buffer
	// No --url and not --dry-run => exit 2 before any network.
	if code := runGrafanaPost(&out, &errb, []string{"--snapshot", "--title", "x"}); code != 2 {
		t.Fatalf("live snapshot without --url should exit 2, got %d (%s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "--url") {
		t.Fatalf("error should name the missing --url: %s", errb.String())
	}
}

func TestGrafanaPostRollupDryRun(t *testing.T) {
	grafanaCleanEnv(t)
	// Write a minimal registry into the temp cwd.
	if err := writeFileForTest(t, "links.json", `{"schema":"fak-grafana-links/1","base_url":"http://localhost:3000","links":[{"title":"Gateway Obs","uid":"fak-gateway-observability","category":"debug","lifetime":"stack-local"}]}`); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runGrafanaPost(&out, &errb, []string{"--rollup", "all", "--registry", "links.json", "--dry-run"})
	if code != 0 {
		t.Fatalf("rollup dry-run exit = %d, stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "1 link(s)") || !strings.Contains(s, "http://localhost:3000/d/fak-gateway-observability") {
		t.Fatalf("rollup dry-run unexpected:\n%s", s)
	}
}

func TestGrafanaPostLinkDryRunWithBaseOverride(t *testing.T) {
	grafanaCleanEnv(t)
	if err := writeFileForTest(t, "links.json", `{"schema":"fak-grafana-links/1","links":[{"title":"Gateway Obs","uid":"fak-gateway-observability","category":"debug"}]}`); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runGrafanaPost(&out, &errb, []string{
		"--link", "fak-gateway-observability", "--registry", "links.json",
		"--base-url", "https://grafana.tailnet.example", "--dry-run",
	})
	if code != 0 {
		t.Fatalf("link dry-run exit = %d, stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "https://grafana.tailnet.example/d/fak-gateway-observability") {
		t.Fatalf("--base-url override should reshape the resolved URL:\n%s", out.String())
	}
}

func TestGrafanaPostLinkUnknownUID(t *testing.T) {
	grafanaCleanEnv(t)
	if err := writeFileForTest(t, "links.json", `{"schema":"fak-grafana-links/1","links":[]}`); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runGrafanaPost(&out, &errb, []string{"--link", "nope", "--registry", "links.json", "--dry-run"}); code != 2 {
		t.Fatalf("unknown uid should exit 2, got %d", code)
	}
}

func TestGrafanaPostRequiresExactlyOneMode(t *testing.T) {
	grafanaCleanEnv(t)
	var out, errb bytes.Buffer
	// Zero modes.
	if code := runGrafanaPost(&out, &errb, []string{"--dry-run"}); code != 2 {
		t.Fatalf("no mode should exit 2, got %d", code)
	}
	out.Reset()
	errb.Reset()
	// Two modes.
	if code := runGrafanaPost(&out, &errb, []string{"--snapshot", "--rollup", "all", "--dry-run"}); code != 2 {
		t.Fatalf("two modes should exit 2, got %d", code)
	}
	if !strings.Contains(errb.String(), "exactly one") {
		t.Fatalf("error should ask for exactly one mode: %s", errb.String())
	}
}

func TestGrafanaSurfaceRegisteredInSlackCheck(t *testing.T) {
	clearSlackEnv(t)
	reports := buildSurfaceReports()
	g := reportByName(reports, "grafana")
	if g == nil {
		t.Fatalf("grafana surface not registered in fak slack check")
	}
	// Public channel default => channel resolves even with nothing set (like blockers/dojo).
	if g.Channel == "" || g.ChannelSource != "built-in default" {
		t.Fatalf("grafana should use its built-in channel default: %+v", g)
	}
}
