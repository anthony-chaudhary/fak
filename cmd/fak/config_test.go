package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/configguide"
	"github.com/anthony-chaudhary/fak/internal/configsurface"
	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

func TestConfigGuideDefaultRequiresNoFile(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runConfigGuide(&out, &errb, []string{"guide"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "No config file required") || !strings.Contains(out.String(), "Run: fak serve") {
		t.Fatalf("output does not make defaults effortless:\n%s", out.String())
	}
}

func TestConfigGuideJSONIsDeterministicAndExplained(t *testing.T) {
	argv := []string{"guide", "--posture", "team-gateway", "--key-env", "OUR_TOKEN", "--bind", "10.0.0.8:9443", "--json"}
	var first, second, errb bytes.Buffer
	if code := runConfigGuide(&first, &errb, argv); code != 0 {
		t.Fatalf("first code=%d stderr=%s", code, errb.String())
	}
	errb.Reset()
	if code := runConfigGuide(&second, &errb, argv); code != 0 {
		t.Fatalf("second code=%d stderr=%s", code, errb.String())
	}
	if first.String() != second.String() {
		t.Fatalf("JSON changed between identical runs:\n%s\n---\n%s", first.String(), second.String())
	}
	var result configguide.Result
	if err := json.Unmarshal(first.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Changes) != 2 || !result.NeedsConfig {
		t.Fatalf("result = %+v", result)
	}
	if _, err := deploymanifest.Parse([]byte(result.Manifest)); err != nil {
		t.Fatalf("CLI manifest does not round-trip: %v", err)
	}
}

func TestConfigGuideWriteRefusesClobber(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fak.toml")
	var out, errb bytes.Buffer
	argv := []string{"guide", "--posture", "long-session", "--budget", "200000", "--write", path}
	if code := runConfigGuide(&out, &errb, argv); code != 0 {
		t.Fatalf("write code=%d stderr=%s", code, errb.String())
	}
	body, err := os.ReadFile(path)
	if err != nil || !strings.Contains(string(body), "default_tokens = 200000") {
		t.Fatalf("written body=%q err=%v", body, err)
	}
	out.Reset()
	errb.Reset()
	if code := runConfigGuide(&out, &errb, argv); code != 1 || !strings.Contains(errb.String(), "already exists") {
		t.Fatalf("clobber code/output = %d %q", code, errb.String())
	}
}

func TestConfigAuditReportsBoundedDiscoverableSurface(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runConfigGuide(&out, &errb, []string{"audit", "--json", "--check"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var report struct {
		Keys                int     `json:"keys"`
		Postures            int     `json:"postures"`
		DefaultCoverage     float64 `json:"default_coverage"`
		DescriptionCoverage float64 `json:"description_coverage"`
		Discoverable        bool    `json:"discoverable"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Keys != configsurface.MaxKeys || report.Postures != 4 || report.DefaultCoverage != 1 || report.DescriptionCoverage != 1 || !report.Discoverable {
		t.Fatalf("audit = %+v", report)
	}
}
