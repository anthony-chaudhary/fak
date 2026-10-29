package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperfcoverage"
)

// coverageAcceptingChecker counts every extracted query and accepts any
// non-empty normalized expression, keeping the cmd-level witness hermetic
// (no promtool binary required on the test host).
type coverageAcceptingChecker struct{ calls int }

func (c *coverageAcceptingChecker) Check(_ context.Context, expr string) error {
	c.calls++
	if strings.TrimSpace(expr) == "" {
		return errors.New("empty expression")
	}
	return nil
}

// installCoverageSeam points the cmd seam at the real nativeperfcoverage.Validate
// with an accepting checker and returns that checker so callers can assert the
// command actually drove query extraction. The promtool lookup seam is pointed
// at a resolvable dummy so the command reaches validation without a real
// binary on PATH.
func installCoverageSeam(t *testing.T) *coverageAcceptingChecker {
	t.Helper()
	checker := &coverageAcceptingChecker{}
	original := grafanaCoverageValidate
	originalLook := coveragePromtoolLookPath
	t.Cleanup(func() {
		grafanaCoverageValidate = original
		coveragePromtoolLookPath = originalLook
	})
	grafanaCoverageValidate = func(ctx context.Context, cfg nativeperfcoverage.Config) (nativeperfcoverage.Matrix, error) {
		cfg.Checker = checker
		return nativeperfcoverage.Validate(ctx, cfg)
	}
	coveragePromtoolLookPath = func(name string) (string, error) { return name, nil }
	return checker
}

// copyCommittedNativePerfTree mirrors the committed tools/grafana contract set
// (prometheus config + all four dashboard/contract/fixture triples) into a
// throwaway root so the command validates data it cannot mutate.
func copyCommittedNativePerfTree(t *testing.T) string {
	t.Helper()
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	sources := []string{"tools/grafana/prometheus.yml"}
	for _, spec := range nativeperfcoverage.Specs() {
		sources = append(sources, spec.Dashboard, spec.Contract, spec.Fixture)
	}
	for _, source := range sources {
		raw, err := os.ReadFile(filepath.Join(repoRoot, source))
		if err != nil {
			t.Fatal(err)
		}
		destination := filepath.Join(root, source)
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(destination, raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// fak-test:runtime fast est=2s
func TestGrafanaCoverageTextMatrix(t *testing.T) {
	installCoverageSeam(t)
	root := copyCommittedNativePerfTree(t)
	var out, errb bytes.Buffer
	code := runGrafanaCoverage(&out, &errb, []string{"--root", root})
	if code != 0 {
		t.Fatalf("coverage exit = %d, stderr=%s", code, errb.String())
	}
	text := out.String()
	for _, spec := range nativeperfcoverage.Specs() {
		if !strings.Contains(text, "DASHBOARD\t"+spec.UID) {
			t.Fatalf("text matrix missing dashboard %s:\n%s", spec.UID, text)
		}
	}
	if !strings.Contains(text, "PANEL\t") || !strings.Contains(text, "QUERY\t") {
		t.Fatalf("text matrix missing panel/query rows:\n%s", text)
	}
}

// fak-test:runtime fast est=2s
func TestGrafanaCoverageJSONRoundTrip(t *testing.T) {
	checker := installCoverageSeam(t)
	root := copyCommittedNativePerfTree(t)
	var out, errb bytes.Buffer
	code := runGrafanaCoverage(&out, &errb, []string{"--root", root, "--json"})
	if code != 0 {
		t.Fatalf("coverage --json exit = %d, stderr=%s", code, errb.String())
	}
	var matrix nativeperfcoverage.Matrix
	if err := json.Unmarshal(out.Bytes(), &matrix); err != nil {
		t.Fatalf("decode matrix: %v\n%s", err, out.String())
	}
	specs := nativeperfcoverage.Specs()
	if len(matrix.Dashboards) != len(specs) {
		t.Fatalf("dashboards=%d, want %d", len(matrix.Dashboards), len(specs))
	}
	for i, spec := range specs {
		if matrix.Dashboards[i].UID != spec.UID {
			t.Fatalf("dashboard[%d].UID=%q, want %q", i, matrix.Dashboards[i].UID, spec.UID)
		}
	}
	if checker.calls == 0 {
		t.Fatal("checker saw no queries; the command did not drive the real Validate")
	}
}

// fak-test:runtime fast est=2s
func TestGrafanaCoverageFailClosedOnMissingTree(t *testing.T) {
	installCoverageSeam(t)
	var out, errb bytes.Buffer
	code := runGrafanaCoverage(&out, &errb, []string{"--root", t.TempDir()})
	if code != 1 {
		t.Fatalf("missing tree exit = %d, want 1 (stderr=%s)", code, errb.String())
	}
	first := nativeperfcoverage.Specs()[0].UID
	if !strings.Contains(errb.String(), first+":") {
		t.Fatalf("fail-closed error must name the first spec UID %q: %s", first, errb.String())
	}
}

// fak-test:runtime fast est=2s
func TestGrafanaCoverageFlagParseError(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runGrafanaCoverage(&out, &errb, []string{"--bogus"}); code != 2 {
		t.Fatalf("unknown flag exit = %d, want 2", code)
	}
}

// fak-test:runtime fast est=2s
func TestGrafanaCoveragePromtoolUnavailable(t *testing.T) {
	originalLook := coveragePromtoolLookPath
	t.Cleanup(func() { coveragePromtoolLookPath = originalLook })
	coveragePromtoolLookPath = func(name string) (string, error) {
		return "", errors.New("promtool not on PATH")
	}
	var out, errb bytes.Buffer
	code := runGrafanaCoverage(&out, &errb, []string{})
	if code != 2 {
		t.Fatalf("unavailable promtool exit = %d, want 2 (stderr=%s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "PROMTOOL_UNAVAILABLE") {
		t.Fatalf("stderr must name PROMTOOL_UNAVAILABLE: %s", errb.String())
	}
}
