package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/readmenext"
)

const testSampleReadme = `# fak — the fast local runtime for coding agents

**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**

## Latest hardware results — 2026-09-01

| Platform | Latest witnessed result | Status | Details |
|---|---|---|---|
| Mac | Qwen3.8-27B on M3 Pro: 7.61 tok/s. | Verified. | [Mac](docs/mac.md) |
| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |

## Why run coding agents on fak

- **Workflow batching and cache reuse:** Multi-agent coding loops reuse prompt context across turns.

## Default priorities & operating modes

1. **fak all in one**
2. **fak serving only**
3. **fak harness only**
4. **other things**

<!-- readme-verified: 2026-09-01 vs VERSION 0.50.0 -->
`

func setupHermeticReadmeRepo(t *testing.T) (string, string) {
	t.Helper()
	repoRoot := t.TempDir()

	// Write README.md
	if err := os.WriteFile(filepath.Join(repoRoot, readmenext.DefaultReadmePath), []byte(testSampleReadme), 0644); err != nil {
		t.Fatalf("failed to write test README: %v", err)
	}

	// Write BENCHMARK-AUTHORITY.md
	authContent := `# BENCHMARK AUTHORITY
## quick-reference-primary-numbers
| Claim | Number | Commit | Artifact |
| NVIDIA Hopper H100 | 111.9 tok/s | abc1234 | docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md |
`
	if err := os.WriteFile(filepath.Join(repoRoot, readmenext.DefaultBenchmarkAuthorityPath), []byte(authContent), 0644); err != nil {
		t.Fatalf("failed to write test benchmark authority: %v", err)
	}

	// Write witness receipt
	receiptDir := filepath.Join(repoRoot, "docs", "_witnesses", "issue-10944-nvidia-gcp-overnight")
	if err := os.MkdirAll(receiptDir, 0755); err != nil {
		t.Fatalf("failed to create receipt dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(receiptDir, "README.md"), []byte("# Witness Receipt"), 0644); err != nil {
		t.Fatalf("failed to write receipt file: %v", err)
	}

	// Write hardware-latest.json
	hwDir := filepath.Join(repoRoot, "docs", "benchmarks")
	if err := os.MkdirAll(hwDir, 0755); err != nil {
		t.Fatalf("failed to create hw dir: %v", err)
	}
	manifest := readmenext.HardwareLatestManifest{
		Schema: "fak-hardware-latest/1",
		AsOf:   "2026-09-01",
		Platforms: map[string]*readmenext.HardwarePlatformEntry{
			"NVIDIA": {
				Observed: "2026-09-01",
				Detail:   "docs/nvidia.md",
				Row:      "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |",
			},
		},
	}
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, readmenext.DefaultHardwareJSONPath), manifestBytes, 0644); err != nil {
		t.Fatalf("failed to write hardware-latest.json: %v", err)
	}

	stagingDir := filepath.Join(repoRoot, "docs", "readme-next")
	if err := os.MkdirAll(stagingDir, 0755); err != nil {
		t.Fatalf("failed to create staging dir: %v", err)
	}

	return repoRoot, stagingDir
}

func TestReadme_HelpAndUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// No args -> prints usage, returns 0
	if code := runReadme(&stdout, &stderr, []string{}); code != 0 {
		t.Fatalf("expected code 0 on empty args, got %d", code)
	}
	if !strings.Contains(stdout.String(), "Usage: fak readme") {
		t.Errorf("expected usage output, got: %s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadme(&stdout, &stderr, []string{"help"}); code != 0 {
		t.Fatalf("expected code 0 on help, got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadme(&stdout, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("expected code 0 on --help, got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadme(&stdout, &stderr, []string{"unknown-cmd"}); code != 2 {
		t.Fatalf("expected code 2 on unknown command, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown readme command") {
		t.Errorf("expected unknown command error, got: %s", stderr.String())
	}
}

func TestReadme_InitFragment(t *testing.T) {
	repoRoot, stagingDir := setupHermeticReadmeRepo(t)

	// Validation errors
	var stdout, stderr bytes.Buffer
	if code := runReadmeInitFragment(&stdout, &stderr, []string{"--issue", "0", "--topic", "test"}); code != 1 {
		t.Errorf("expected code 1 on issue <= 0, got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmeInitFragment(&stdout, &stderr, []string{"--issue", "100", "--topic", ""}); code != 1 {
		t.Errorf("expected code 1 on empty topic, got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmeInitFragment(&stdout, &stderr, []string{"--issue", "100", "--topic", "foo", "--section", "bad-section"}); code != 1 {
		t.Errorf("expected code 1 on invalid section, got %d", code)
	}

	// Valid initialization into staging directory
	stdout.Reset()
	stderr.Reset()
	if code := runReadmeInitFragment(&stdout, &stderr, []string{
		"--issue", "11881",
		"--topic", "pipeline-staging",
		"--section", "why_fak",
		"--dir", stagingDir,
	}); code != 0 {
		t.Fatalf("expected code 0 on init-fragment, got %d (stderr: %s)", code, stderr.String())
	}

	expectedFile := filepath.Join(stagingDir, "issue-11881-pipeline-staging.json")
	if _, err := os.Stat(expectedFile); err != nil {
		t.Fatalf("expected fragment file %s to exist: %v", expectedFile, err)
	}

	frag, err := readmenext.LoadCandidateFragmentFile(expectedFile)
	if err != nil {
		t.Fatalf("failed to load created fragment: %v", err)
	}
	if frag.Issue != 11881 || frag.Topic != "pipeline-staging" || frag.TargetSection != "why_fak" {
		t.Errorf("fragment fields mismatch: %+v", frag)
	}
	if err := readmenext.ValidateFragment(frag, repoRoot); err != nil {
		t.Errorf("created fragment should validate cleanly: %v", err)
	}

	// Test init-fragment with custom --out path
	customOut := filepath.Join(repoRoot, "custom", "custom-frag.json")
	stdout.Reset()
	stderr.Reset()
	if code := runReadmeInitFragment(&stdout, &stderr, []string{
		"--issue", "11882",
		"--topic", "custom-topic",
		"--out", customOut,
	}); code != 0 {
		t.Fatalf("expected code 0 on init-fragment custom out, got %d (stderr: %s)", code, stderr.String())
	}
	if _, err := os.Stat(customOut); err != nil {
		t.Fatalf("expected custom output file to exist: %v", err)
	}

	// Test init-fragment with --out - (stdout)
	stdout.Reset()
	stderr.Reset()
	if code := runReadmeInitFragment(&stdout, &stderr, []string{
		"--issue", "11883",
		"--topic", "stdout-topic",
		"--out", "-",
	}); code != 0 {
		t.Fatalf("expected code 0 on init-fragment stdout, got %d", code)
	}
	if !strings.Contains(stdout.String(), `"topic": "stdout-topic"`) {
		t.Errorf("expected stdout to contain fragment JSON: %s", stdout.String())
	}

	// Test init-fragment for hardware_table section
	hwOut := filepath.Join(stagingDir, "issue-10944-hw.json")
	stdout.Reset()
	stderr.Reset()
	if code := runReadmeInitFragment(&stdout, &stderr, []string{
		"--issue", "10944",
		"--topic", "nvidia-hopper",
		"--section", "hardware_table",
		"--out", hwOut,
	}); code != 0 {
		t.Fatalf("expected code 0 on hardware_table init, got %d", code)
	}
	hwFrag, err := readmenext.LoadCandidateFragmentFile(hwOut)
	if err != nil {
		t.Fatalf("failed to load hw fragment: %v", err)
	}
	if hwFrag.RetireTarget.Action != readmenext.RetireActionReplaceRow {
		t.Errorf("expected hardware fragment to have replace_row action, got %s", hwFrag.RetireTarget.Action)
	}
}

func TestReadme_LintStaged_ValidAndInvalid(t *testing.T) {
	repoRoot, stagingDir := setupHermeticReadmeRepo(t)

	// Case 1: Empty staging directory
	var stdout, stderr bytes.Buffer
	if code := runReadmeLintStaged(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot}); code != 0 {
		t.Fatalf("expected code 0 on empty staging dir, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "0 fragments found") {
		t.Errorf("expected '0 fragments found' message, got: %s", stdout.String())
	}

	// With --json on empty directory
	stdout.Reset()
	stderr.Reset()
	if code := runReadmeLintStaged(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--json"}); code != 0 {
		t.Fatalf("expected code 0 on empty staging dir json, got %d", code)
	}
	var emptyReport struct {
		Valid bool `json:"valid"`
		Total int  `json:"total"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &emptyReport); err != nil {
		t.Fatalf("failed to unmarshal empty json report: %v", err)
	}
	if !emptyReport.Valid || emptyReport.Total != 0 {
		t.Errorf("expected valid=true, total=0, got %+v", emptyReport)
	}

	// Case 2: Valid fragments
	validWhyFak := `{
		"schema": "fak-readme-candidate/1",
		"issue": 11881,
		"topic": "cache-pipelining",
		"target_section": "why_fak",
		"candidate_content": "- **Cache pipelining:** Staged fragments verify benchmark authority before publication.",
		"retire_target": { "action": "none" },
		"laws_checklist": { "sota_comparison": true, "feynman_gloss": true, "wide_audience": true },
		"date": "2026-09-06"
	}`
	if err := os.WriteFile(filepath.Join(stagingDir, "issue-11881-whyfak.json"), []byte(validWhyFak), 0644); err != nil {
		t.Fatalf("failed to write valid fragment: %v", err)
	}

	validHW := `{
		"schema": "fak-readme-candidate/1",
		"issue": 10944,
		"topic": "nvidia-h100-update",
		"target_section": "hardware_table",
		"candidate_content": "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Verified | [NVIDIA](docs/nv.md) |",
		"retire_target": {
			"action": "replace_row",
			"target_text": "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
		},
		"witness": {
			"authority_entry": "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
			"receipt_path": "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
			"hardware_json_row": "NVIDIA"
		},
		"laws_checklist": { "sota_comparison": true, "feynman_gloss": true, "wide_audience": true },
		"date": "2026-09-06"
	}`
	if err := os.WriteFile(filepath.Join(stagingDir, "issue-10944-hw.json"), []byte(validHW), 0644); err != nil {
		t.Fatalf("failed to write valid hw fragment: %v", err)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmeLintStaged(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot}); code != 0 {
		t.Fatalf("expected code 0 on valid fragments, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "2/2 fragments valid") {
		t.Errorf("expected 2/2 valid summary, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "[PASS] issue-11881-whyfak.json") {
		t.Errorf("missing pass marker: %s", stdout.String())
	}

	// JSON format check
	stdout.Reset()
	stderr.Reset()
	if code := runReadmeLintStaged(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--json"}); code != 0 {
		t.Fatalf("expected code 0 on valid json report, got %d", code)
	}
	var validReport struct {
		Valid     bool `json:"valid"`
		Total     int  `json:"total"`
		Passed    int  `json:"passed"`
		Failed    int  `json:"failed"`
		Fragments []struct {
			File  string `json:"file"`
			Valid bool   `json:"valid"`
		} `json:"fragments"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &validReport); err != nil {
		t.Fatalf("failed to unmarshal valid json report: %v", err)
	}
	if !validReport.Valid || validReport.Total != 2 || validReport.Passed != 2 || validReport.Failed != 0 {
		t.Errorf("unexpected report contents: %+v", validReport)
	}

	// Case 3: Add an invalid fragment
	invalidFrag := `{
		"schema": "fak-readme-candidate/1",
		"issue": 9999,
		"topic": "invalid-fragment",
		"target_section": "hardware_table",
		"candidate_content": "| AMD | New row |",
		"retire_target": {
			"action": "none"
		}
	}`
	if err := os.WriteFile(filepath.Join(stagingDir, "issue-9999-invalid.json"), []byte(invalidFrag), 0644); err != nil {
		t.Fatalf("failed to write invalid fragment: %v", err)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmeLintStaged(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot}); code != 1 {
		t.Fatalf("expected code 1 on invalid fragment, got %d", code)
	}
	if !strings.Contains(stdout.String(), "[FAIL] issue-9999-invalid.json") {
		t.Errorf("expected fail marker in output: %s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmeLintStaged(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--json"}); code != 1 {
		t.Fatalf("expected code 1 on invalid fragment json, got %d", code)
	}
	if err := json.Unmarshal(stdout.Bytes(), &validReport); err != nil {
		t.Fatalf("failed to unmarshal invalid json report: %v", err)
	}
	if validReport.Valid || validReport.Failed != 1 || validReport.Passed != 2 {
		t.Errorf("unexpected invalid report contents: %+v", validReport)
	}
}

func TestReadme_PreviewNext(t *testing.T) {
	repoRoot, stagingDir := setupHermeticReadmeRepo(t)

	// Valid fragment
	oldRow := "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
	newRow := "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Witnessed | [NVIDIA](docs/nv.md) |"
	hwFrag := `{
		"schema": "fak-readme-candidate/1",
		"issue": 10944,
		"topic": "nvidia-update",
		"target_section": "hardware_table",
		"candidate_content": "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Witnessed | [NVIDIA](docs/nv.md) |",
		"retire_target": {
			"action": "replace_row",
			"target_text": "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
		},
		"witness": {
			"authority_entry": "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
			"receipt_path": "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
			"hardware_json_row": "NVIDIA"
		},
		"laws_checklist": { "sota_comparison": true, "feynman_gloss": true, "wide_audience": true },
		"date": "2026-09-06"
	}`
	if err := os.WriteFile(filepath.Join(stagingDir, "issue-10944-hw.json"), []byte(hwFrag), 0644); err != nil {
		t.Fatalf("failed to write hw fragment: %v", err)
	}

	// 1. Preview summary to stdout
	var stdout, stderr bytes.Buffer
	if code := runReadmePreviewNext(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot}); code != 0 {
		t.Fatalf("expected code 0 on preview-next, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "README-NEXT Preview: 1 fragments applied") {
		t.Errorf("expected 1 fragment applied summary, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Replaced row in hardware_table") {
		t.Errorf("expected replaced row in changes: %s", stdout.String())
	}

	// Original README should NOT be modified
	origContent, err := os.ReadFile(filepath.Join(repoRoot, readmenext.DefaultReadmePath))
	if err != nil {
		t.Fatalf("failed to read original README: %v", err)
	}
	if !strings.Contains(string(origContent), oldRow) || strings.Contains(string(origContent), newRow) {
		t.Errorf("preview should not touch README on disk")
	}

	// 2. Preview written to --out path
	previewOutFile := filepath.Join(repoRoot, "README-NEXT.md")
	stdout.Reset()
	stderr.Reset()
	if code := runReadmePreviewNext(&stdout, &stderr, []string{
		"--dir", stagingDir,
		"--repo", repoRoot,
		"--out", previewOutFile,
	}); code != 0 {
		t.Fatalf("expected code 0 on preview-next with --out, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Preview written to") {
		t.Errorf("expected preview written notice: %s", stdout.String())
	}

	previewBytes, err := os.ReadFile(previewOutFile)
	if err != nil {
		t.Fatalf("failed to read written preview: %v", err)
	}
	if !strings.Contains(string(previewBytes), newRow) {
		t.Errorf("preview file does not contain new row")
	}
	if strings.Contains(string(previewBytes), oldRow) {
		t.Errorf("preview file still contains old row")
	}

	// 3. Preview written to stdout with --out -
	stdout.Reset()
	stderr.Reset()
	if code := runReadmePreviewNext(&stdout, &stderr, []string{
		"--dir", stagingDir,
		"--repo", repoRoot,
		"--out", "-",
	}); code != 0 {
		t.Fatalf("expected code 0 on preview-next --out -, got %d", code)
	}
	if !strings.Contains(stdout.String(), newRow) {
		t.Errorf("preview stdout does not contain new row")
	}
}

func TestReadme_Publish_DryRunAndApply(t *testing.T) {
	repoRoot, stagingDir := setupHermeticReadmeRepo(t)

	oldRow := "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
	newRow := "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Witnessed | [NVIDIA](docs/nv.md) |"
	hwFrag := `{
		"schema": "fak-readme-candidate/1",
		"issue": 10944,
		"topic": "nvidia-publish",
		"target_section": "hardware_table",
		"candidate_content": "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Witnessed | [NVIDIA](docs/nv.md) |",
		"retire_target": {
			"action": "append_to_legacy",
			"target_text": "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
		},
		"witness": {
			"authority_entry": "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
			"receipt_path": "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
			"hardware_json_row": "NVIDIA"
		},
		"laws_checklist": { "sota_comparison": true, "feynman_gloss": true, "wide_audience": true },
		"date": "2026-09-06"
	}`
	if err := os.WriteFile(filepath.Join(stagingDir, "issue-10944-hw.json"), []byte(hwFrag), 0644); err != nil {
		t.Fatalf("failed to write hw fragment: %v", err)
	}

	// 1. Dry run (default when neither flag passed)
	var stdout, stderr bytes.Buffer
	if code := runReadmePublish(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot}); code != 0 {
		t.Fatalf("expected code 0 on default publish dry-run, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "[DRY-RUN]") {
		t.Errorf("expected [DRY-RUN] in stdout: %s", stdout.String())
	}

	// Verify no changes to README.md
	diskReadme, err := os.ReadFile(filepath.Join(repoRoot, readmenext.DefaultReadmePath))
	if err != nil {
		t.Fatalf("failed to read README: %v", err)
	}
	if strings.Contains(string(diskReadme), newRow) {
		t.Errorf("README was modified during dry run")
	}

	// Verify legacy doc was not created
	legacyDoc := filepath.Join(repoRoot, readmenext.DefaultLegacyArchiveDoc)
	if _, err := os.Stat(legacyDoc); !os.IsNotExist(err) {
		t.Errorf("legacy archive doc should not exist after dry run")
	}

	// 2. Dry run with --json
	stdout.Reset()
	stderr.Reset()
	if code := runReadmePublish(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--dry-run", "--json"}); code != 0 {
		t.Fatalf("expected code 0 on publish dry-run json, got %d", code)
	}
	var dryResult readmenext.PublishResult
	if err := json.Unmarshal(stdout.Bytes(), &dryResult); err != nil {
		t.Fatalf("failed to unmarshal dry-run json result: %v", err)
	}
	if !dryResult.DryRun || !dryResult.Success {
		t.Errorf("unexpected dry result: %+v", dryResult)
	}

	// 3. Live publish with --apply
	stdout.Reset()
	stderr.Reset()
	if code := runReadmePublish(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--apply"}); code != 0 {
		t.Fatalf("expected code 0 on publish apply, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "[SUCCESS]") {
		t.Errorf("expected [SUCCESS] in stdout: %s", stdout.String())
	}

	// Check updated README.md
	diskReadme, err = os.ReadFile(filepath.Join(repoRoot, readmenext.DefaultReadmePath))
	if err != nil {
		t.Fatalf("failed to read README after apply: %v", err)
	}
	if !strings.Contains(string(diskReadme), newRow) {
		t.Errorf("README does not contain new row after publish")
	}
	if strings.Contains(string(diskReadme), oldRow) {
		t.Errorf("README still contains old row after publish")
	}

	// Check legacy archive doc
	legacyBytes, err := os.ReadFile(legacyDoc)
	if err != nil {
		t.Fatalf("expected legacy archive doc to be created: %v", err)
	}
	if !strings.Contains(string(legacyBytes), oldRow) {
		t.Errorf("legacy doc does not contain retired old row")
	}
	if !strings.Contains(string(legacyBytes), "issue-10944-hw.json") && !strings.Contains(string(legacyBytes), "Issue #10944: nvidia-publish") {
		t.Errorf("legacy doc missing metadata header")
	}

	// Check hardware-latest.json updated
	hwBytes, err := os.ReadFile(filepath.Join(repoRoot, readmenext.DefaultHardwareJSONPath))
	if err != nil {
		t.Fatalf("failed to read hardware manifest: %v", err)
	}
	var manifest readmenext.HardwareLatestManifest
	if err := json.Unmarshal(hwBytes, &manifest); err != nil {
		t.Fatalf("failed to parse updated manifest: %v", err)
	}
	if manifest.Platforms["NVIDIA"].Row != newRow {
		t.Errorf("hardware manifest row was not updated")
	}

	// 4. Broken fragment publishes fail cleanly
	badFrag := `{
		"schema": "fak-readme-candidate/1",
		"issue": 5555,
		"topic": "bad-target",
		"target_section": "hardware_table",
		"candidate_content": "| NVIDIA | Broken row |",
		"retire_target": {
			"action": "replace_row",
			"target_text": "| NVIDIA | Non-existent row in README |"
		},
		"witness": {
			"authority_entry": "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
			"receipt_path": "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
			"hardware_json_row": "NVIDIA"
		},
		"laws_checklist": { "sota_comparison": true, "feynman_gloss": true, "wide_audience": true },
		"date": "2026-09-06"
	}`
	badFragPath := filepath.Join(stagingDir, "issue-5555-bad.json")
	if err := os.WriteFile(badFragPath, []byte(badFrag), 0644); err != nil {
		t.Fatalf("failed to write bad fragment: %v", err)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmePublish(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--apply"}); code != 1 {
		t.Fatalf("expected code 1 on failed publish, got %d", code)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runReadmePublish(&stdout, &stderr, []string{"--dir", stagingDir, "--repo", repoRoot, "--apply", "--json"}); code != 1 {
		t.Fatalf("expected code 1 on failed publish json, got %d", code)
	}
	var errResult map[string]interface{}
	if err := json.Unmarshal(stdout.Bytes(), &errResult); err != nil {
		t.Fatalf("failed to unmarshal error json: %v", err)
	}
	if errResult["success"] != false {
		t.Errorf("expected success: false in json error, got: %+v", errResult)
	}
}

func TestReadme_VerbRouting(t *testing.T) {
	oldExit := readmeExit
	defer func() { readmeExit = oldExit }()
	var exitCode int
	readmeExit = func(code int) {
		exitCode = code
	}

	// Verify that dispatchExtendedVerbB handles both "readme" and "readmenext"
	if !dispatchExtendedVerbB("readme", []string{"help"}) {
		t.Errorf("expected dispatchExtendedVerbB to return true for 'readme'")
	}
	if exitCode != 0 {
		t.Errorf("expected exitCode 0, got %d", exitCode)
	}

	if !dispatchExtendedVerbB("readmenext", []string{"help"}) {
		t.Errorf("expected dispatchExtendedVerbB to return true for 'readmenext'")
	}
	if exitCode != 0 {
		t.Errorf("expected exitCode 0, got %d", exitCode)
	}
}
