package logvault

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func captureMetricsVault(t *testing.T, vaultDir, srcRoot string) *Vault {
	t.Helper()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "src", Root: srcRoot}}}
	if _, err := v.Capture(); err != nil {
		t.Fatalf("capture: %v", err)
	}
	return v
}

func writeMetricSrc(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func TestMetricsText_FreshCapture(t *testing.T) {
	src := t.TempDir()
	vaultDir := filepath.Join(t.TempDir(), "vault")
	writeMetricSrc(t, src, "a.log", "hello")
	writeMetricSrc(t, src, "sub/b.log", "world!!")
	v := captureMetricsVault(t, vaultDir, src)

	txt, err := v.MetricsText(0, 5_000_000_000)
	if err != nil {
		t.Fatalf("MetricsText: %v", err)
	}
	for _, name := range []string{
		"fak_logvault_last_capture_age_seconds",
		"fak_logvault_vault_bytes",
		"fak_logvault_verify_mismatches",
	} {
		if !strings.Contains(txt, "# HELP "+name+" ") || !strings.Contains(txt, "# TYPE "+name+" gauge") {
			t.Errorf("missing HELP/TYPE for %s in:\n%s", name, txt)
		}
	}
	// Conflation law: every family declares its values WITNESSED.
	for _, line := range strings.Split(txt, "\n") {
		if strings.HasPrefix(line, "# HELP ") && !strings.Contains(line, "WITNESSED") {
			t.Errorf("help line lacks WITNESSED disclosure: %q", line)
		}
	}
	if !strings.Contains(txt, "fak_logvault_vault_bytes 12\n") { // len("hello")+len("world!!")
		t.Errorf("want vault_bytes 12, got:\n%s", txt)
	}
	if !strings.Contains(txt, "fak_logvault_verify_mismatches 0\n") {
		t.Errorf("fresh capture should verify 0 mismatches, got:\n%s", txt)
	}
	if strings.Contains(txt, "fak_logvault_last_capture_age_seconds -1") {
		t.Errorf("a captured vault must not read the never-captured age sentinel:\n%s", txt)
	}
}

func TestMetricsText_EmptyVaultSentinel(t *testing.T) {
	v := &Vault{Dir: filepath.Join(t.TempDir(), "never")}
	txt, err := v.MetricsText(0, 1_000_000_000)
	if err != nil {
		t.Fatalf("MetricsText over empty vault must not error: %v", err)
	}
	if !strings.Contains(txt, "fak_logvault_last_capture_age_seconds -1\n") {
		t.Errorf("empty vault age gauge should be -1, got:\n%s", txt)
	}
	if !strings.Contains(txt, "fak_logvault_vault_bytes 0\n") || !strings.Contains(txt, "fak_logvault_verify_mismatches 0\n") {
		t.Errorf("empty vault should read zero bytes + zero mismatches, got:\n%s", txt)
	}
}

func TestMetricsText_ForcedMismatchShowsNonZero(t *testing.T) {
	src := t.TempDir()
	vaultDir := filepath.Join(t.TempDir(), "vault")
	writeMetricSrc(t, src, "a.log", "original-content")
	v := captureMetricsVault(t, vaultDir, src)

	if txt, _ := v.MetricsText(0, 2_000_000_000); !strings.Contains(txt, "fak_logvault_verify_mismatches 0\n") {
		t.Fatalf("pre-tamper mismatch gauge should be 0, got:\n%s", txt)
	}

	// Force a mirror mismatch — the silent backup corruption the issue says must
	// surface within one capture cycle. The next scrape (MetricsText) must show it.
	mirror := filepath.Join(vaultDir, "by-source", "src", "a.log")
	if err := os.WriteFile(mirror, []byte("CORRUPTED"), 0o644); err != nil {
		t.Fatalf("tamper mirror: %v", err)
	}
	txt, err := v.MetricsText(0, 2_000_000_000)
	if err != nil {
		t.Fatalf("MetricsText post-tamper: %v", err)
	}
	if strings.Contains(txt, "fak_logvault_verify_mismatches 0\n") {
		t.Errorf("forced mismatch must show a non-zero verify gauge, got:\n%s", txt)
	}
}

func TestRenderLogvaultGauges_ChainBrokenFloorsToOne(t *testing.T) {
	txt := RenderLogvaultGauges(10, 5, 100, 0, true)
	if !strings.Contains(txt, "fak_logvault_verify_mismatches 1\n") {
		t.Errorf("a broken chain must floor the mismatch gauge to >=1, got:\n%s", txt)
	}
}
