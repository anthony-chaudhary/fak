package validate

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

const (
	hwTestTip     = "796f3207aa11bb22cc33dd44ee55ff6677889900"
	hwTestArchive = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
)

var hwTestNow = time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC)

// hwTestReceipt is a passing native SSD restart receipt as a stamped WSL
// controller emits it, including forward-schema witness fields HEAD's struct
// may not know yet.
func hwTestReceipt(mut func(map[string]any)) []byte {
	r := map[string]any{
		"schema":        amdgpu.StrixSSDRestartSchema,
		"timestamp":     hwTestNow.Add(-time.Hour).Format(time.RFC3339),
		"verdict":       "PASS",
		"hardware":      true,
		"evidence_rung": "HW_WITNESSED",
		"target":        map[string]any{"host": "strix1", "reachable": true, "target_isa": "gfx1151"},
		"provenance": map[string]any{
			"git_tip":               hwTestTip,
			"source_archive_sha256": hwTestArchive,
			"generated_by":          "fak-dev amd-strix-validate",
			"command":               "fak-dev amd-strix-validate --native-ssd-restart --mine internal/amdgpu/strix_ssd_restart.go",
			"cleanup_observed":      true,
		},
		"witness":  map[string]any{"schema": "fak-vulkan-warm-disk-restart/2", "expected_warm_h2d_count": 3},
		"digest":   "sha256:2222222222222222222222222222222222222222222222222222222222222222",
		"verified": true,
	}
	if mut != nil {
		mut(r)
	}
	raw, _ := json.Marshal(r)
	return raw
}

func hwTestCode(t *testing.T, err error) string {
	t.Helper()
	if err == nil {
		return ""
	}
	refusal, ok := err.(*hwReceiptRefusal)
	if !ok {
		t.Fatalf("refusal is %T (%v), want *hwReceiptRefusal", err, err)
	}
	return refusal.code
}

var hwTestMine = []string{"internal/amdgpu/strix_ssd_restart.go", "internal/devcmd/amd_strix_ssd_restart_test.go"}

func TestCheckHWReceiptAdmitsExactFreshBinding(t *testing.T) {
	adm, err := checkHWReceipt(hwTestReceipt(nil), hwTestTip, hwTestArchive, hwTestMine, hwTestNow, hwReceiptDefaultMaxAge)
	if err != nil {
		t.Fatalf("matching receipt refused: %v", err)
	}
	if !adm.Admitted || adm.Verifier != "candidate" || adm.GitTip != hwTestTip || adm.SourceArchiveSHA256 != hwTestArchive || adm.Host != "strix1" {
		t.Fatalf("admission record = %+v", adm)
	}
}

func TestCheckHWReceiptRefusals(t *testing.T) {
	otherTip := strings.Repeat("a", 40)
	otherArchive := "sha256:" + strings.Repeat("3", 64)
	prov := func(key string, val any) func(map[string]any) {
		return func(r map[string]any) { r["provenance"].(map[string]any)[key] = val }
	}
	cases := []struct {
		name    string
		raw     []byte
		tip     string
		archive string
		want    string
	}{
		{"stale tip", hwTestReceipt(nil), otherTip, hwTestArchive, hwReceiptStaleTip},
		{"receipt from older tip", hwTestReceipt(prov("git_tip", otherTip)), hwTestTip, hwTestArchive, hwReceiptStaleTip},
		{"mismatched archive digest", hwTestReceipt(nil), hwTestTip, otherArchive, hwReceiptArchiveMismatch},
		{"missing archive digest", hwTestReceipt(prov("source_archive_sha256", "")), hwTestTip, hwTestArchive, hwReceiptArchiveMismatch},
		{"expired", hwTestReceipt(func(r map[string]any) { r["timestamp"] = hwTestNow.Add(-25 * time.Hour).Format(time.RFC3339) }), hwTestTip, hwTestArchive, hwReceiptExpired},
		{"future", hwTestReceipt(func(r map[string]any) { r["timestamp"] = hwTestNow.Add(time.Hour).Format(time.RFC3339) }), hwTestTip, hwTestArchive, hwReceiptExpired},
		{"fail verdict", hwTestReceipt(func(r map[string]any) { r["verdict"] = "FAIL" }), hwTestTip, hwTestArchive, hwReceiptNotPass},
		{"not hardware", hwTestReceipt(func(r map[string]any) { r["hardware"] = false }), hwTestTip, hwTestArchive, hwReceiptNotPass},
		{"recorded failures", hwTestReceipt(func(r map[string]any) { r["failures"] = []any{map[string]any{"code": "X"}} }), hwTestTip, hwTestArchive, hwReceiptNotPass},
		{"cleanup unobserved", hwTestReceipt(prov("cleanup_observed", false)), hwTestTip, hwTestArchive, hwReceiptNotPass},
		{"foreign producer", hwTestReceipt(prov("generated_by", "hand-written")), hwTestTip, hwTestArchive, hwReceiptProducerRefused},
		{"unknown schema", hwTestReceipt(func(r map[string]any) { r["schema"] = "fak.strix.validation/v1" }), hwTestTip, hwTestArchive, hwReceiptSchema},
		{"garbage", []byte("{not json"), hwTestTip, hwTestArchive, hwReceiptUnreadable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			adm, err := checkHWReceipt(tc.raw, tc.tip, tc.archive, hwTestMine, hwTestNow, hwReceiptDefaultMaxAge)
			if got := hwTestCode(t, err); got != tc.want {
				t.Fatalf("code = %q (%v), want %q", got, err, tc.want)
			}
			if adm.Admitted {
				t.Fatalf("refused receipt marked admitted: %+v", adm)
			}
		})
	}
}

// A candidate that does not change the receipt verifier is held to HEAD's
// strict decode and Validate: a hand-assembled body cannot pass on binding alone.
func TestCheckHWReceiptHeadVerifierIsStrict(t *testing.T) {
	_, err := checkHWReceipt(hwTestReceipt(nil), hwTestTip, hwTestArchive, []string{"internal/compute/vulkan_shim.cpp"}, hwTestNow, hwReceiptDefaultMaxAge)
	if got := hwTestCode(t, err); got != hwReceiptInvalid {
		t.Fatalf("code = %q (%v), want %q", got, err, hwReceiptInvalid)
	}
}

// A Windows checkout archives .gitattributes text files with CRLF unless the
// gate pins core.eol; the pin must make it reproduce the Linux controller's LF
// archive, and must leave the environment as it found it.
func TestPinCanonicalArchiveEOLMatchesLinuxArchive(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	t.Setenv("GIT_CONFIG_COUNT", "")
	os.Unsetenv("GIT_CONFIG_COUNT")
	root := t.TempDir()
	git := func(args ...string) string {
		cmd := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@example.invalid"}, args...)...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("config", "core.autocrlf", "false")
	for name, body := range map[string]string{".gitattributes": "*.md text\n", "notes.md": "a\nb\n"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("add", ".")
	git("commit", "-q", "-m", "base")
	tip := git("rev-parse", "HEAD")
	git("config", "core.eol", "lf")
	lf, err := amdgpu.BuildStrixCandidateArchive(context.Background(), root, tip, nil)
	if err != nil {
		t.Fatal(err)
	}
	git("config", "core.eol", "crlf") // native Windows behaviour
	crlf, err := amdgpu.BuildStrixCandidateArchive(context.Background(), root, tip, nil)
	if err != nil {
		t.Fatal(err)
	}
	if crlf.SourceArchiveSHA256 == lf.SourceArchiveSHA256 {
		t.Fatal("fixture does not exercise eol conversion")
	}
	restore := pinCanonicalArchiveEOL()
	pinned, err := amdgpu.BuildStrixCandidateArchive(context.Background(), root, tip, nil)
	restore()
	if err != nil {
		t.Fatal(err)
	}
	if pinned.SourceArchiveSHA256 != lf.SourceArchiveSHA256 {
		t.Fatalf("pinned archive %s != Linux LF archive %s", pinned.SourceArchiveSHA256, lf.SourceArchiveSHA256)
	}
	if _, ok := os.LookupEnv("GIT_CONFIG_COUNT"); ok {
		t.Fatal("pin leaked GIT_CONFIG_COUNT into the environment")
	}
}

func TestHWReceiptPathFlagThenEnv(t *testing.T) {
	t.Setenv(hwReceiptEnv, " env.json ")
	if got := hwReceiptPath(""); got != "env.json" {
		t.Fatalf("env fallback = %q", got)
	}
	if got := hwReceiptPath("flag.json"); got != "flag.json" {
		t.Fatalf("flag precedence = %q", got)
	}
	t.Setenv(hwReceiptMaxAgeEnv, "2h")
	if got := hwReceiptMaxAge(); got != 2*time.Hour {
		t.Fatalf("max age = %s", got)
	}
}

// TestAdmitHWReceiptRecomputesProspectiveArchive drives the real archive
// builder over a real repository: the receipt only satisfies the gate while the
// owned working-tree bytes are exactly the ones the controller validated.
func TestAdmitHWReceiptRecomputesProspectiveArchive(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	git := func(args ...string) string {
		cmd := exec.Command("git", append([]string{"-c", "core.autocrlf=false", "-c", "user.name=t", "-c", "user.email=t@example.invalid"}, args...)...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	owned := "internal/amdgpu/strix_ssd_restart.go"
	full := filepath.Join(root, filepath.FromSlash(owned))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte("package amdgpu\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("init", "-q")
	git("add", ".")
	git("commit", "-q", "-m", "base")
	tip := git("rev-parse", "HEAD")
	if err := os.WriteFile(full, []byte("package amdgpu\n\n// candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mine := []string{owned}
	candidate, err := amdgpu.BuildStrixCandidateArchive(context.Background(), root, tip, mine)
	if err != nil {
		t.Fatal(err)
	}

	orig := validateNow
	validateNow = func() time.Time { return hwTestNow }
	t.Cleanup(func() { validateNow = orig })
	receiptPath := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(receiptPath, hwTestReceipt(func(r map[string]any) {
		p := r["provenance"].(map[string]any)
		p["git_tip"], p["source_archive_sha256"] = tip, candidate.SourceArchiveSHA256
	}), 0o644); err != nil {
		t.Fatal(err)
	}

	admit := func() *validateResult {
		res := &validateResult{OK: true, Tip: tip}
		recorder := &validateRecorder{ctx: context.Background(), stderr: io.Discard, res: res}
		if ok := admitHWReceipt(context.Background(), res, recorder, root, receiptPath, mine); ok != res.OK {
			t.Fatalf("admit returned %t but res.OK=%t", ok, res.OK)
		}
		return res
	}
	if res := admit(); !res.OK || res.StrixReceipt == nil || !res.StrixReceipt.Admitted || !strings.HasPrefix(res.StrixReceipt.FileSHA256, "sha256:") {
		t.Fatalf("exact prospective tree refused: %+v failures=%+v", res.StrixReceipt, res.Failures)
	}

	// One byte of drift after the controller ran must refuse admission.
	if err := os.WriteFile(full, []byte("package amdgpu\n\n// candidate!\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res := admit()
	if res.OK || res.StrixReceipt.Code != hwReceiptArchiveMismatch || len(res.Failures) != 1 || res.Failures[0].Step != "strix-receipt-admission" {
		t.Fatalf("drifted tree admitted or misreported: %+v failures=%+v", res.StrixReceipt, res.Failures)
	}
	if err := os.WriteFile(full, []byte("package amdgpu\n\n// candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A peer landing only hardware-irrelevant paths carries the receipt forward.
	peerCommit := func(path, body string) string {
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		git("add", "--", path)
		git("commit", "-q", "-m", "peer", "--", path)
		return git("rev-parse", "HEAD")
	}
	tip = peerCommit("docs/notes.md", "peer docs\n")
	if res := admit(); !res.OK || res.StrixReceipt.AdvancedTo != tip {
		t.Fatalf("irrelevant trunk motion refused: %+v failures=%+v", res.StrixReceipt, res.Failures)
	}

	// A peer landing a hardware-relevant path makes the receipt stale.
	tip = peerCommit("internal/compute/kernel.go", "package compute\n")
	if res := admit(); res.OK || res.StrixReceipt.Code != hwReceiptStaleTip {
		t.Fatalf("hardware-relevant trunk motion admitted: %+v failures=%+v", res.StrixReceipt, res.Failures)
	}
}
