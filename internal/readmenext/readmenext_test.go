package readmenext

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const sampleReadme = `# fak — the fast local runtime for coding agents

**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**

## Latest hardware results — 2026-09-01

| Platform | Latest witnessed result | Status | Details |
|---|---|---|---|
| Mac | Qwen3.8-27B on M3 Pro: 7.61 tok/s. | Verified. | [Mac](docs/mac.md) |
| AMD | Qwen3.6-27B on RX 7600: 1.15 tok/s. | Narrow microbench. | [AMD](docs/amd.md) |
| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |

## Why run coding agents on fak

- **Workflow batching and cache reuse:** Multi-agent coding loops reuse prompt context across turns.
- **Local execution on your hardware:** Run models directly with native inference.

## Default priorities & operating modes

1. **fak all in one**
2. **fak serving only**
3. **fak harness only**
4. **other things**

<!-- readme-verified: 2026-09-01 vs VERSION 0.50.0 -->
`

func setupHermeticRepo(t testing.TB) string {
	t.Helper()
	repoRoot := t.TempDir()

	// Write README.md
	if err := os.WriteFile(filepath.Join(repoRoot, DefaultReadmePath), []byte(sampleReadme), 0644); err != nil {
		t.Fatalf("failed to write sample README: %v", err)
	}

	// Write BENCHMARK-AUTHORITY.md
	authContent := `# BENCHMARK AUTHORITY
## quick-reference-primary-numbers
| Claim | Number | Commit | Artifact |
| NVIDIA Hopper H100 | 111.9 tok/s | abc1234 | docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md |
`
	if err := os.WriteFile(filepath.Join(repoRoot, DefaultBenchmarkAuthorityPath), []byte(authContent), 0644); err != nil {
		t.Fatalf("failed to write sample benchmark authority: %v", err)
	}

	// Write receipt file
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
	manifest := HardwareLatestManifest{
		Schema: "fak-hardware-latest/1",
		AsOf:   "2026-09-01",
		Platforms: map[string]*HardwarePlatformEntry{
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
	if err := os.WriteFile(filepath.Join(repoRoot, DefaultHardwareJSONPath), manifestBytes, 0644); err != nil {
		t.Fatalf("failed to write hardware-latest.json: %v", err)
	}

	return repoRoot
}

func TestParseCandidateFragment(t *testing.T) {
	rawJSON := `{
		"schema": "fak-readme-candidate/1",
		"issue": 10944,
		"topic": "nvidia-h100-q8",
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
		"laws_checklist": {
			"sota_comparison": true,
			"feynman_gloss": true,
			"wide_audience": true
		},
		"proposed_by": "tester",
		"date": "2026-09-06"
	}`

	frag, err := ParseCandidateFragment([]byte(rawJSON))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if frag.Schema != SchemaCandidate {
		t.Errorf("expected schema %q, got %q", SchemaCandidate, frag.Schema)
	}
	if frag.Issue != 10944 {
		t.Errorf("expected issue 10944, got %d", frag.Issue)
	}
	if frag.Topic != "nvidia-h100-q8" {
		t.Errorf("expected topic nvidia-h100-q8, got %s", frag.Topic)
	}
	if frag.RetireTarget.LegacyArchiveDoc != DefaultLegacyArchiveDoc {
		t.Errorf("expected default legacy doc %q, got %q", DefaultLegacyArchiveDoc, frag.RetireTarget.LegacyArchiveDoc)
	}
}

func TestValidateFragment_BasicConstraints(t *testing.T) {
	repoRoot := setupHermeticRepo(t)

	// Nil fragment
	if err := ValidateFragment(nil, repoRoot); err == nil {
		t.Error("expected error for nil fragment, got nil")
	}

	// Invalid schema
	f := &CandidateFragment{
		Schema:           "wrong-schema",
		Issue:            123,
		Topic:            "topic",
		TargetSection:    TargetWhyFak,
		CandidateContent: "content",
	}
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "invalid schema") {
		t.Errorf("expected schema error, got %v", err)
	}

	// Invalid issue
	f.Schema = SchemaCandidate
	f.Issue = 0
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "invalid issue") {
		t.Errorf("expected issue error, got %v", err)
	}

	// Empty topic
	f.Issue = 100
	f.Topic = ""
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "topic cannot be empty") {
		t.Errorf("expected empty topic error, got %v", err)
	}

	// Empty candidate content
	f.Topic = "valid-topic"
	f.CandidateContent = "   "
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "candidate_content cannot be empty") {
		t.Errorf("expected empty candidate_content error, got %v", err)
	}

	// Unsupported section
	f.CandidateContent = "valid content"
	f.TargetSection = "unsupported_section"
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "unsupported target_section") {
		t.Errorf("expected unsupported section error, got %v", err)
	}

	// Invalid date
	f.TargetSection = TargetWhyFak
	f.Date = "2026/09/06"
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "invalid date format") {
		t.Errorf("expected invalid date format error, got %v", err)
	}
}

func TestValidateFragment_BoundedSections(t *testing.T) {
	repoRoot := setupHermeticRepo(t)

	// Bounded section (hardware_table) requires action replace_row or append_to_legacy with non-empty target_text
	f := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            10944,
		Topic:            "nvidia-update",
		TargetSection:    TargetHardwareTable,
		CandidateContent: "| NVIDIA | new row |",
		RetireTarget: RetireTarget{
			Action: RetireActionNone,
		},
	}
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "bounded section") {
		t.Errorf("expected bounded section error on Action none, got %v", err)
	}

	// Action replace_row but empty target_text
	f.RetireTarget.Action = RetireActionReplaceRow
	f.RetireTarget.TargetText = ""
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "requires non-empty target_text") {
		t.Errorf("expected target_text required error, got %v", err)
	}

	// Hero headline is also bounded
	f.TargetSection = TargetHeroHeadline
	f.RetireTarget.Action = RetireActionNone
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "bounded section") {
		t.Errorf("expected bounded section error on hero_headline with action none, got %v", err)
	}
}

func TestValidateFragment_WitnessValidation(t *testing.T) {
	repoRoot := setupHermeticRepo(t)

	f := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            10944,
		Topic:            "nvidia-update",
		TargetSection:    TargetWhyFak,
		CandidateContent: "Simple content item without performance comparison",
		Witness: Witness{
			ReceiptPath: "docs/_witnesses/non-existent-dir/README.md",
		},
	}

	// Missing receipt path
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "witness receipt_path") {
		t.Errorf("expected missing receipt error, got %v", err)
	}

	// Correct receipt path, but non-existent authority key
	f.Witness.ReceiptPath = "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md"
	f.Witness.AuthorityEntry = "BENCHMARK-AUTHORITY.md#missing-key-404"
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "authority_entry") {
		t.Errorf("expected authority entry missing error, got %v", err)
	}

	// Valid authority key, but missing hardware-latest manifest
	f.Witness.AuthorityEntry = "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers"
	f.Witness.HardwareJSONRow = "NVIDIA"

	// Temporarily remove hardware manifest to test failure
	hwPath := filepath.Join(repoRoot, DefaultHardwareJSONPath)
	backup := hwPath + ".bak"
	if err := os.Rename(hwPath, backup); err != nil {
		t.Fatalf("failed to backup hw file: %v", err)
	}
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "hardware latest manifest not found") {
		t.Errorf("expected missing hardware manifest error, got %v", err)
	}
	if err := os.Rename(backup, hwPath); err != nil {
		t.Fatalf("failed to restore hw file: %v", err)
	}

	// Now everything should pass
	if err := ValidateFragment(f, repoRoot); err != nil {
		t.Errorf("expected clean validation, got %v", err)
	}
}

func TestValidateFragment_SOTALawAndNaiveBaselines(t *testing.T) {
	repoRoot := setupHermeticRepo(t)

	// Bold headline leading with naive without tuned/sota
	f := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            123,
		Topic:            "naive-violation",
		TargetSection:    TargetWhyFak,
		CandidateContent: "We achieve **60x vs naive** in standard benchmarks.",
		LawsChecklist: LawsChecklist{
			SOTAComparison: true,
		},
	}
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "violates SOTA-vs-us law") {
		t.Errorf("expected SOTA-vs-us law error on bold naive headline, got %v", err)
	}

	// Content-level naive lead without tuned/sota
	f.CandidateContent = "fak is 10x faster vs a naive loop."
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "violates SOTA-vs-us law") {
		t.Errorf("expected SOTA-vs-us law error on vs naive lead, got %v", err)
	}

	// Honest ledger framing: both naive AND tuned named
	f.CandidateContent = "Session accounting delivers **60.3× vs naive · 4.1× vs tuned** baselines."
	if err := ValidateFragment(f, repoRoot); err != nil {
		t.Errorf("expected honest ledger framing to pass, got %v", err)
	}

	// Comparative performance claim without SOTAComparison checklist checked
	f.CandidateContent = "Hopper H100 achieves 111.9 tok/s (+17.4% vs f32) with 4.84x TTFT speedup."
	f.LawsChecklist.SOTAComparison = false
	if err := ValidateFragment(f, repoRoot); err == nil || !strings.Contains(err.Error(), "LawsChecklist.SOTAComparison is false") {
		t.Errorf("expected SOTAComparison checklist requirement error, got %v", err)
	}

	// Comparative claim with SOTAComparison checklist checked
	f.LawsChecklist.SOTAComparison = true
	if err := ValidateFragment(f, repoRoot); err != nil {
		t.Errorf("expected valid comparative claim to pass with checklist, got %v", err)
	}
}

func TestPreviewNext_Synthesis(t *testing.T) {
	oldNvidiaRow := "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
	newNvidiaRow := "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s (+17.4% vs f32) | Witnessed | [NVIDIA](docs/new_nv.md) |"

	fragments := []*CandidateFragment{
		{
			Schema:           SchemaCandidate,
			Issue:            10944,
			Topic:            "nvidia-update",
			TargetSection:    TargetHardwareTable,
			CandidateContent: newNvidiaRow,
			RetireTarget: RetireTarget{
				Action:     RetireActionReplaceRow,
				TargetText: oldNvidiaRow,
			},
			Date: "2026-09-06",
		},
		{
			Schema:           SchemaCandidate,
			Issue:            11200,
			Topic:            "why-fak-addition",
			TargetSection:    TargetWhyFak,
			CandidateContent: "- **Zero-copy storage overflow:** Stream paged KV cache directly from NVMe.",
			RetireTarget: RetireTarget{
				Action: RetireActionNone,
			},
		},
	}

	preview, changes, err := PreviewNext(sampleReadme, fragments)
	if err != nil {
		t.Fatalf("unexpected PreviewNext error: %v", err)
	}

	if !strings.Contains(preview, newNvidiaRow) {
		t.Errorf("preview does not contain new NVIDIA row")
	}
	if strings.Contains(preview, oldNvidiaRow) {
		t.Errorf("preview still contains old NVIDIA row")
	}
	if !strings.Contains(preview, "## Latest hardware results — 2026-09-06") {
		t.Errorf("preview did not update hardware results header date")
	}
	if !strings.Contains(preview, "- **Zero-copy storage overflow:** Stream paged KV cache directly from NVMe.") {
		t.Errorf("preview did not append why_fak bullet item")
	}
	if len(changes) == 0 {
		t.Errorf("expected changes list to be populated")
	}

	// Verify error on missing target_text
	missingFragment := []*CandidateFragment{
		{
			Schema:           SchemaCandidate,
			Issue:            9999,
			Topic:            "missing",
			TargetSection:    TargetHardwareTable,
			CandidateContent: "| Mac | New |",
			RetireTarget: RetireTarget{
				Action:     RetireActionReplaceRow,
				TargetText: "| Mac | This string does not exist in README |",
			},
		},
	}
	if _, _, err := PreviewNext(sampleReadme, missingFragment); err == nil {
		t.Errorf("expected error for missing target_text, got nil")
	}
}

func TestPublish_DryRun(t *testing.T) {
	repoRoot := setupHermeticRepo(t)
	oldNvidiaRow := "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
	newNvidiaRow := "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Verified | [NVIDIA](docs/nv.md) |"

	f := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            10944,
		Topic:            "nvidia-update",
		TargetSection:    TargetHardwareTable,
		CandidateContent: newNvidiaRow,
		RetireTarget: RetireTarget{
			Action:     RetireActionAppendToLegacy,
			TargetText: oldNvidiaRow,
		},
		Witness: Witness{
			ReceiptPath:     "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
			AuthorityEntry:  "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
			HardwareJSONRow: "NVIDIA",
		},
		LawsChecklist: LawsChecklist{
			SOTAComparison: true,
			FeynmanGloss:   true,
			WideAudience:   true,
		},
		Date: "2026-09-06",
	}

	res, err := Publish(repoRoot, []*CandidateFragment{f}, true)
	if err != nil {
		t.Fatalf("unexpected Publish dry-run error: %v", err)
	}

	if !res.DryRun {
		t.Error("expected DryRun to be true")
	}
	if !res.Success {
		t.Error("expected Success to be true")
	}

	// Verify README was NOT modified on disk
	contentOnDisk, err := os.ReadFile(filepath.Join(repoRoot, DefaultReadmePath))
	if err != nil {
		t.Fatalf("failed to read README: %v", err)
	}
	if strings.Contains(string(contentOnDisk), newNvidiaRow) {
		t.Error("dry-run should not have modified README on disk")
	}

	// Verify docs/README-legacy.md was NOT created
	legacyPath := filepath.Join(repoRoot, DefaultLegacyArchiveDoc)
	if _, err := os.Stat(legacyPath); !os.IsNotExist(err) {
		t.Errorf("legacy archive should not exist after dry run")
	}
}

func TestPublish_Live(t *testing.T) {
	repoRoot := setupHermeticRepo(t)
	oldNvidiaRow := "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
	newNvidiaRow := "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Verified | [NVIDIA](docs/nv.md) |"

	f := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            10944,
		Topic:            "nvidia-update",
		TargetSection:    TargetHardwareTable,
		CandidateContent: newNvidiaRow,
		RetireTarget: RetireTarget{
			Action:     RetireActionAppendToLegacy,
			TargetText: oldNvidiaRow,
		},
		Witness: Witness{
			ReceiptPath:     "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
			AuthorityEntry:  "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
			HardwareJSONRow: "NVIDIA",
		},
		LawsChecklist: LawsChecklist{
			SOTAComparison: true,
			FeynmanGloss:   true,
			WideAudience:   true,
		},
		Date: "2026-09-06",
	}

	res, err := Publish(repoRoot, []*CandidateFragment{f}, false)
	if err != nil {
		t.Fatalf("unexpected Publish live error: %v", err)
	}

	if res.DryRun {
		t.Error("expected DryRun to be false")
	}
	if !res.Success {
		t.Error("expected Success to be true")
	}
	if !res.HardwareJSONUpdated {
		t.Error("expected HardwareJSONUpdated to be true")
	}

	// 1. Verify README.md on disk
	readmeData, err := os.ReadFile(filepath.Join(repoRoot, DefaultReadmePath))
	if err != nil {
		t.Fatalf("failed to read README on disk: %v", err)
	}
	readmeStr := string(readmeData)
	if !strings.Contains(readmeStr, newNvidiaRow) {
		t.Errorf("README does not contain new NVIDIA row on disk")
	}
	if strings.Contains(readmeStr, oldNvidiaRow) {
		t.Errorf("README still contains old NVIDIA row on disk")
	}
	if !strings.Contains(readmeStr, "## Latest hardware results — 2026-09-06") {
		t.Errorf("README does not contain updated hardware header date")
	}

	// 2. Verify docs/README-legacy.md on disk
	legacyData, err := os.ReadFile(filepath.Join(repoRoot, DefaultLegacyArchiveDoc))
	if err != nil {
		t.Fatalf("failed to read legacy doc on disk: %v", err)
	}
	legacyStr := string(legacyData)
	if !strings.Contains(legacyStr, oldNvidiaRow) {
		t.Errorf("legacy doc does not contain retired text: %s", legacyStr)
	}
	if !strings.Contains(legacyStr, "Issue #10944: nvidia-update") {
		t.Errorf("legacy doc does not contain issue/topic header: %s", legacyStr)
	}

	// 3. Verify hardware-latest.json on disk
	hwData, err := os.ReadFile(filepath.Join(repoRoot, DefaultHardwareJSONPath))
	if err != nil {
		t.Fatalf("failed to read hardware manifest on disk: %v", err)
	}
	var manifest HardwareLatestManifest
	if err := json.Unmarshal(hwData, &manifest); err != nil {
		t.Fatalf("failed to unmarshal hardware manifest: %v", err)
	}
	if manifest.AsOf != "2026-09-06" {
		t.Errorf("expected manifest as_of to be 2026-09-06, got %s", manifest.AsOf)
	}
	nvEntry, ok := manifest.Platforms["NVIDIA"]
	if !ok {
		t.Fatalf("NVIDIA platform entry missing from manifest")
	}
	if nvEntry.Observed != "2026-09-06" {
		t.Errorf("expected observed 2026-09-06, got %s", nvEntry.Observed)
	}
	if nvEntry.Row != newNvidiaRow {
		t.Errorf("expected row %q, got %q", newNvidiaRow, nvEntry.Row)
	}
	if nvEntry.Detail != "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md" {
		t.Errorf("expected detail path to match receipt, got %s", nvEntry.Detail)
	}
}

func TestSynthesizeNextDraft(t *testing.T) {
	oldNvidiaRow := "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |"
	newNvidiaRow := "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s | Verified | [NVIDIA](docs/nv.md) |"

	fragments := []*CandidateFragment{
		{
			Schema:           SchemaCandidate,
			Issue:            10944,
			Topic:            "nvidia-update",
			TargetSection:    TargetHardwareTable,
			CandidateContent: newNvidiaRow,
			RetireTarget: RetireTarget{
				Action:     RetireActionReplaceRow,
				TargetText: oldNvidiaRow,
			},
			Date: "2026-09-06",
		},
	}

	draft, changes, err := SynthesizeNextDraft(sampleReadme, fragments)
	if err != nil {
		t.Fatalf("unexpected error from SynthesizeNextDraft: %v", err)
	}
	preview, previewChanges, previewErr := PreviewNext(sampleReadme, fragments)
	if previewErr != nil {
		t.Fatalf("unexpected error from PreviewNext: %v", previewErr)
	}

	if draft != preview {
		t.Errorf("expected SynthesizeNextDraft output to match PreviewNext exactly")
	}
	if len(changes) != len(previewChanges) {
		t.Errorf("expected changes count to match: got %d, want %d", len(changes), len(previewChanges))
	}
}
