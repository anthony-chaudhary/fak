package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestRunTrajectoryAuditPinnedCrossHarness(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	markdownPath := filepath.Join(temp, "audit.md")
	claudeRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "claude", "projects")
	codexRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "codex", "sessions")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--jsonl", jsonlPath, "--md", markdownPath,
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	jsonl, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	markdown, err := os.ReadFile(markdownPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"schema":"fak-trajectory-audit/1"`, `"kind":"source_denominator"`, `"source":"claude"`, `"source":"codex"`, `"rank":1`} {
		if !bytes.Contains(jsonl, []byte(want)) {
			t.Fatalf("JSONL missing %q:\n%s", want, jsonl)
		}
	}
	if !bytes.Contains(markdown, []byte("Highest-cost bottleneck: claude/`claude-session`")) {
		t.Fatalf("markdown missing deterministic top row:\n%s", markdown)
	}
	if !strings.Contains(stderr.String(), "sessions=2 exact_usage=6 refused=0") {
		t.Fatalf("stderr = %q", stderr.String())
	}

	baselineJSONL := filepath.Join(temp, "with-baseline.jsonl")
	stderr.Reset()
	rc = runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--baseline", jsonlPath, "--jsonl", baselineJSONL,
	})
	if rc != 0 {
		t.Fatalf("baseline rc=%d stderr=%s", rc, stderr.String())
	}
	withBaseline, err := os.ReadFile(baselineJSONL)
	if err != nil {
		t.Fatal(err)
	}
	if count := bytes.Count(withBaseline, []byte(`"kind":"baseline_delta"`)); count != 4 {
		t.Fatalf("baseline delta rows = %d, want 4:\n%s", count, withBaseline)
	}
}

func TestRunTrajectoryAuditRendersCodexCacheAttribution(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	markdownPath := filepath.Join(temp, "audit.md")
	claudeRoot := filepath.Join(temp, "missing-claude")
	codexRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "codex", "sessions")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--jsonl", jsonlPath, "--md", markdownPath,
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	jsonl := mustReadTrajectoryAuditFile(t, jsonlPath)
	markdown := mustReadTrajectoryAuditFile(t, markdownPath)
	for _, want := range []string{
		`"model_provider":"openai"`,
		`"last_token_usage_cached_input_samples":2`,
		`"last_token_usage_cached_input_min":40`,
		`"last_token_usage_cached_input_max":40`,
		`"physical_provider_cache_residency":"not_inferable_from_cached_input_tokens"`,
		`"fak_owned_cache_coverage":"not_observed_by_codex_token_count"`,
	} {
		if !bytes.Contains(jsonl, []byte(want)) {
			t.Fatalf("JSONL missing %q:\n%s", want, jsonl)
		}
	}
	for _, want := range []string{
		"## Codex per-request cache observations",
		"| `codex-session` | `codex` | `openai` | 2 | 40 | 40 |",
		"does not prove physical provider cache residency or process-local ownership",
		"fak-owned caches are not observed by Codex `token_count` rows",
	} {
		if !bytes.Contains(markdown, []byte(want)) {
			t.Fatalf("markdown missing %q:\n%s", want, markdown)
		}
	}
}

func TestRunTrajectoryAuditUnsupportedShapeWritesArtifactAndFails(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	claudeRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "unsupported", "claude", "projects")
	missingCodex := filepath.Join(temp, "missing-codex")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", missingCodex, "--jsonl", jsonlPath,
	})
	if rc != 1 {
		t.Fatalf("rc=%d, want 1; stderr=%s", rc, stderr.String())
	}
	artifact, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(artifact, []byte(`"kind":"refusal"`)) || !strings.Contains(stderr.String(), "TRAJECTORY_SCHEMA_REFUSED") {
		t.Fatalf("missing visible unsupported shape; stderr=%s artifact=%s", stderr.String(), artifact)
	}
}

func TestRunTrajectoryAuditIssue9418SharedCodexSessionIDsStayDistinct(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	missingClaude := filepath.Join(temp, "missing-claude")
	codexRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "issue-9418", "codex-shared-session", "sessions")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", missingClaude, "--codex-root", codexRoot, "--jsonl", jsonlPath,
	})
	if rc != 0 {
		t.Fatalf("rc=%d, want 0; stderr=%s", rc, stderr.String())
	}
	artifact, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(artifact, []byte(`"kind":"refusal"`)) {
		t.Fatalf("unexpected refusal for distinct primary rollout ids:\n%s", artifact)
	}
	for _, want := range []string{`"session_id":"root-rollout"`, `"session_id":"child-rollout"`, `"raw_fragments":2`, `"canonical_transcripts":2`, `"refused_records":0`} {
		if !bytes.Contains(artifact, []byte(want)) {
			t.Fatalf("artifact missing %q:\n%s", want, artifact)
		}
	}
	if !strings.Contains(stderr.String(), "sessions=2 exact_usage=2 refused=0") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestRunTrajectoryAuditRefusalMatrix(t *testing.T) {
	fixtureRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "issue-8493", "refusals")
	issue9418Root := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "issue-9418")
	missing := filepath.Join(t.TempDir(), "missing")
	tests := []struct {
		name       string
		claudeRoot string
		codexRoot  string
		code       string
	}{
		{"Claude duplicate mismatch", filepath.Join(fixtureRoot, "claude-duplicate", "projects"), missing, "claude_duplicate_usage_mismatch"},
		{"Claude live duplicate mismatch", filepath.Join(issue9418Root, "claude-duplicate", "projects"), missing, "claude_duplicate_usage_mismatch"},
		{"Codex canonical fragment mismatch", missing, filepath.Join(issue9418Root, "codex-ambiguous", "sessions"), "codex_fragment_usage_mismatch"},
		{"Codex cumulative decrease", missing, filepath.Join(fixtureRoot, "codex-decreasing", "sessions"), "codex_total_usage_decreased"},
		{"malformed JSON", filepath.Join(fixtureRoot, "malformed", "claude", "projects"), missing, "malformed_json"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertTrajectoryAuditRefusal(t, test.claudeRoot, test.codexRoot, test.code)
		})
	}

	t.Run("oversized line", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "fak", "oversized.jsonl")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		seed, err := os.ReadFile(filepath.Join(fixtureRoot, "oversized", "claude", "projects", "fak", "oversized.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		file, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write(bytes.TrimSpace(seed)); err != nil {
			file.Close()
			t.Fatal(err)
		}
		chunk := bytes.Repeat([]byte("x"), 1024*1024)
		for written := len(seed); written <= 32*1024*1024; written += len(chunk) {
			if _, err := file.Write(chunk); err != nil {
				file.Close()
				t.Fatal(err)
			}
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		assertTrajectoryAuditRefusal(t, root, missing, "line_too_large")
	})
}

func assertTrajectoryAuditRefusal(t *testing.T, claudeRoot, codexRoot, code string) {
	t.Helper()
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	markdownPath := filepath.Join(temp, "audit.md")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--jsonl", jsonlPath, "--md", markdownPath,
	})
	if rc != 1 {
		t.Fatalf("rc=%d, want 1; stderr=%s", rc, stderr.String())
	}
	artifact, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatalf("read refusal artifact: %v; stderr=%s", err, stderr.String())
	}
	for _, want := range []string{`"schema":"fak-trajectory-audit/1"`, `"kind":"refusal"`, `"code":"` + code + `"`} {
		if !bytes.Contains(artifact, []byte(want)) {
			t.Fatalf("artifact missing %q; stderr=%s artifact=%s", want, stderr.String(), artifact)
		}
	}
	if !strings.Contains(stderr.String(), "TRAJECTORY_SCHEMA_REFUSED") {
		t.Fatalf("missing refusal status; stderr=%s", stderr.String())
	}

	var summaryRefused, denominatorRefused, refusalRows int
	for _, line := range bytes.Split(bytes.TrimSpace(artifact), []byte{'\n'}) {
		var row struct {
			Kind           string `json:"kind"`
			RefusedRecords int    `json:"refused_records"`
		}
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatalf("decode audit row: %v\n%s", err, line)
		}
		switch row.Kind {
		case "summary":
			summaryRefused = row.RefusedRecords
		case "source_denominator":
			denominatorRefused += row.RefusedRecords
		case "refusal":
			refusalRows++
		}
	}
	if summaryRefused != refusalRows || denominatorRefused != refusalRows || refusalRows != 1 {
		t.Fatalf("refusal totals = summary:%d denominators:%d rows:%d, want 1/1/1; artifact=%s", summaryRefused, denominatorRefused, refusalRows, artifact)
	}
	if !strings.Contains(stderr.String(), "refused="+strconv.Itoa(refusalRows)) {
		t.Fatalf("stderr refusal total disagrees with artifact: %s", stderr.String())
	}
	markdown, err := os.ReadFile(markdownPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(markdown, []byte("Broad efficiency conclusions: **blocked** (refusals: "+strconv.Itoa(refusalRows)+").")) {
		t.Fatalf("markdown refusal total disagrees with artifact:\n%s", markdown)
	}
}

func TestParseTrajectoryAuditSinceDays(t *testing.T) {
	duration, err := parseTrajectoryAuditSince("7d")
	if err != nil {
		t.Fatal(err)
	}
	if duration.Hours() != 168 {
		t.Fatalf("duration = %s", duration)
	}
}

func TestRunTrajectoryAuditSnapshotCaptureDeleteRootsReplay(t *testing.T) {
	base := t.TempDir()
	claudeRoot, codexRoot := writeTrajectoryAuditSnapshotRoots(t, base)
	snapshot := filepath.Join(base, "private-snapshot")
	captureJSON := filepath.Join(base, "capture.jsonl")
	captureMD := filepath.Join(base, "capture.md")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--user-contains", "snapshot-topic",
		"--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--snapshot-out", snapshot, "--jsonl", captureJSON, "--md", captureMD,
	})
	if rc != 0 {
		t.Fatalf("capture rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "OUT_OF_TREE_WRITE operation=trajectory-audit-snapshot") || !strings.Contains(stderr.String(), "verified=true") {
		t.Fatalf("capture did not declare and verify private output: %s", stderr.String())
	}
	manifest, err := os.ReadFile(filepath.Join(snapshot, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"cli-private-prompt", claudeRoot, codexRoot, "snapshot-topic"} {
		if bytes.Contains(manifest, []byte(forbidden)) {
			t.Fatalf("manifest leaked %q: %s", forbidden, manifest)
		}
	}
	if info, err := os.Stat(snapshot); err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("snapshot mode = %v err=%v", info.Mode().Perm(), err)
	}
	if err := os.RemoveAll(filepath.Dir(claudeRoot)); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Dir(codexRoot)); err != nil {
		t.Fatal(err)
	}
	capturedJSON := mustReadTrajectoryAuditFile(t, captureJSON)
	capturedMD := mustReadTrajectoryAuditFile(t, captureMD)
	for i := 1; i <= 2; i++ {
		jsonPath := filepath.Join(base, "replay-"+strconv.Itoa(i)+".jsonl")
		mdPath := filepath.Join(base, "replay-"+strconv.Itoa(i)+".md")
		stdout.Reset()
		stderr.Reset()
		rc = runTrajectory(&stdout, &stderr, []string{"audit", "--snapshot", snapshot, "--jsonl", jsonPath, "--md", mdPath})
		if rc != 0 {
			t.Fatalf("replay %d rc=%d stderr=%s", i, rc, stderr.String())
		}
		if !bytes.Equal(capturedJSON, mustReadTrajectoryAuditFile(t, jsonPath)) || !bytes.Equal(capturedMD, mustReadTrajectoryAuditFile(t, mdPath)) {
			t.Fatalf("replay %d output differs from capture", i)
		}
		if !strings.Contains(stderr.String(), "sessions=2") || !strings.Contains(stderr.String(), "verified=true") {
			t.Fatalf("replay %d status = %s", i, stderr.String())
		}
	}
	for name, output := range map[string][]byte{"jsonl": capturedJSON, "markdown": capturedMD} {
		if bytes.Contains(output, []byte("cli-private-prompt")) {
			t.Fatalf("%s leaked transcript payload", name)
		}
		if !bytes.Contains(output, []byte("snapshot")) {
			t.Fatalf("%s does not name snapshot corpus", name)
		}
	}
}

func TestRunTrajectoryAuditSnapshotFlagAndTamperRefusals(t *testing.T) {
	base := t.TempDir()
	claudeRoot, codexRoot := writeTrajectoryAuditSnapshotRoots(t, base)
	snapshot := filepath.Join(base, "snapshot")
	var stdout, stderr bytes.Buffer
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--snapshot-out", snapshot}); rc != 0 {
		t.Fatalf("capture rc=%d stderr=%s", rc, stderr.String())
	}
	for _, args := range [][]string{
		{"audit", "--snapshot", snapshot, "--since", "0"},
		{"audit", "--snapshot", snapshot, "--user-contains", "topic"},
		{"audit", "--snapshot", snapshot, "--claude-root", claudeRoot},
		{"audit", "--snapshot", snapshot, "--snapshot-out", filepath.Join(base, "other")},
	} {
		stderr.Reset()
		if rc := runTrajectory(&stdout, &stderr, args); rc != 2 || !strings.Contains(stderr.String(), "SNAPSHOT_FLAGS_INCOMPATIBLE") {
			t.Fatalf("args=%v rc=%d stderr=%s", args, rc, stderr.String())
		}
	}
	stderr.Reset()
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--snapshot-out", snapshot}); rc != 1 || !strings.Contains(stderr.String(), "SNAPSHOT_TARGET_EXISTS") {
		t.Fatalf("existing target rc=%d stderr=%s", rc, stderr.String())
	}
	manifestPath := filepath.Join(snapshot, "manifest.json")
	if err := os.WriteFile(manifestPath, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	stderr.Reset()
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--snapshot", snapshot}); rc != 1 || !strings.Contains(stderr.String(), "SNAPSHOT_SCHEMA_INCOMPATIBLE") {
		t.Fatalf("tamper rc=%d stderr=%s", rc, stderr.String())
	}
}

func TestRunTrajectoryAuditSnapshotUsageLedgerAndFold(t *testing.T) {
	base := t.TempDir()
	claudeRoot, codexRoot := writeTrajectoryAuditSnapshotRoots(t, base)
	snapshot := filepath.Join(base, "snapshot")
	ledger := filepath.Join(base, "usage.jsonl")
	var stdout, stderr bytes.Buffer
	args := []string{"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--snapshot-out", snapshot, "--snapshot-usage-ledger", ledger}
	if rc := runTrajectory(&stdout, &stderr, args); rc != 0 {
		t.Fatalf("capture rc=%d stderr=%s", rc, stderr.String())
	}
	usageDeclaration := strings.Index(stderr.String(), "OUT_OF_TREE_WRITE operation=trajectory-audit-snapshot-usage")
	snapshotDeclaration := strings.Index(stderr.String(), "OUT_OF_TREE_WRITE operation=trajectory-audit-snapshot target=")
	if usageDeclaration < 0 || snapshotDeclaration < 0 || usageDeclaration > snapshotDeclaration {
		t.Fatalf("write declarations out of order: %s", stderr.String())
	}
	withoutLedger := filepath.Join(base, "without-ledger.md")
	withLedger := filepath.Join(base, "with-ledger.md")
	stdout.Reset()
	stderr.Reset()
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--snapshot", snapshot, "--md", withoutLedger}); rc != 0 {
		t.Fatalf("plain replay rc=%d stderr=%s", rc, stderr.String())
	}
	if strings.Contains(stderr.String(), "snapshot-usage") {
		t.Fatalf("absent option changed stderr: %s", stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--snapshot", snapshot, "--md", withLedger, "--snapshot-usage-ledger", ledger}); rc != 0 {
		t.Fatalf("ledger replay rc=%d stderr=%s", rc, stderr.String())
	}
	plain, err := os.ReadFile(withoutLedger)
	if err != nil {
		t.Fatal(err)
	}
	tracked, err := os.ReadFile(withLedger)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(plain, tracked) {
		t.Fatal("usage option changed replay output")
	}
	payload, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{base, claudeRoot, codexRoot, snapshot, "claude-snapshot", "codex-snapshot", "cli-private-prompt", strings.Repeat("a", 64)} {
		if bytes.Contains(payload, []byte(forbidden)) {
			t.Fatalf("usage ledger leaked %q: %s", forbidden, payload)
		}
	}
	if info, err := os.Stat(ledger); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("ledger mode=%v err=%v", info.Mode().Perm(), err)
	}
	stdout.Reset()
	stderr.Reset()
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--snapshot-usage-fold", ledger}); rc != 0 {
		t.Fatalf("fold rc=%d stderr=%s", rc, stderr.String())
	}
	for _, want := range []string{`"schema":"fak-trajectory-audit-snapshot-usage-fold/1"`, `"total":2`, `"capture":1`, `"replay":1`, `"success":2`} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("fold missing %q: %s", want, stdout.String())
		}
	}
}

func TestRunTrajectoryAuditSnapshotUsageRecordsRefusalAndError(t *testing.T) {
	base := t.TempDir()
	claudeRoot, codexRoot := writeTrajectoryAuditSnapshotRoots(t, base)
	snapshot := filepath.Join(base, "snapshot")
	ledger := filepath.Join(base, "usage.jsonl")
	var stdout, stderr bytes.Buffer
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--snapshot-out", snapshot}); rc != 0 {
		t.Fatalf("setup rc=%d stderr=%s", rc, stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--snapshot-out", snapshot, "--snapshot-usage-ledger", ledger}); rc != 1 {
		t.Fatalf("refusal rc=%d stderr=%s", rc, stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	missingOutput := filepath.Join(base, "missing", "audit.jsonl")
	if rc := runTrajectory(&stdout, &stderr, []string{"audit", "--snapshot", snapshot, "--jsonl", missingOutput, "--snapshot-usage-ledger", ledger}); rc != 1 {
		t.Fatalf("error rc=%d stderr=%s", rc, stderr.String())
	}
	payload, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"outcome":"refused"`, `"reason":"SNAPSHOT_TARGET_EXISTS"`, `"outcome":"error"`, `"reason":"OUTPUT_WRITE_FAILED"`} {
		if !bytes.Contains(payload, []byte(want)) {
			t.Fatalf("ledger missing %q: %s", want, payload)
		}
	}
}

func writeTrajectoryAuditSnapshotRoots(t *testing.T, base string) (string, string) {
	t.Helper()
	claudeRoot := filepath.Join(base, "claude-live", "projects")
	codexRoot := filepath.Join(base, "codex-live", "sessions")
	claudePath := filepath.Join(claudeRoot, "project", "claude.jsonl")
	codexPath := filepath.Join(codexRoot, "2026", "08", "28", "codex.jsonl")
	for _, path := range []string{claudePath, codexPath} {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	claude := "{\"type\":\"user\",\"sessionId\":\"claude-snapshot\",\"message\":{\"role\":\"user\",\"content\":\"snapshot-topic cli-private-prompt\"}}\n" +
		"{\"type\":\"assistant\",\"sessionId\":\"claude-snapshot\",\"message\":{\"id\":\"msg-1\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"cache_read_input_tokens\":3,\"cache_creation_input_tokens\":1},\"content\":\"done\"}}\n"
	codex := "{\"timestamp\":\"2026-08-28T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"codex-snapshot\",\"model_provider\":\"openai\"}}\n" +
		"{\"timestamp\":\"2026-08-28T00:00:01Z\",\"type\":\"turn_context\",\"payload\":{\"model\":\"gpt-test\",\"turn_id\":\"turn-1\"}}\n" +
		"{\"timestamp\":\"2026-08-28T00:00:02Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"snapshot-topic cli-private-prompt\"}]}}\n" +
		"{\"timestamp\":\"2026-08-28T00:00:03Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"token_count\",\"info\":{\"last_token_usage\":{\"input_tokens\":20,\"cached_input_tokens\":5,\"cache_write_input_tokens\":0,\"output_tokens\":4,\"reasoning_output_tokens\":0,\"total_tokens\":24},\"total_token_usage\":{\"input_tokens\":20,\"cached_input_tokens\":5,\"cache_write_input_tokens\":0,\"output_tokens\":4,\"reasoning_output_tokens\":0,\"total_tokens\":24}}}}\n"
	if err := os.WriteFile(claudePath, []byte(claude), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexPath, []byte(codex), 0o600); err != nil {
		t.Fatal(err)
	}
	return claudeRoot, codexRoot
}

func mustReadTrajectoryAuditFile(t *testing.T, path string) []byte {
	t.Helper()
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return payload
}

func TestRunTrajectoryRoutesAssurance(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTrajectory(&stdout, &stderr, []string{"assurance", "unexpected"})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak trajectory assurance") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestTrajectoryUsageListsAssurance(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runTrajectory(&stdout, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if !strings.Contains(stdout.String(), "assurance") {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestRunTrajectoryAuditProgressFlag(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	claudeRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "claude", "projects")
	codexRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "codex", "sessions")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--jsonl", jsonlPath, "--progress",
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "trajectory audit: [") {
		t.Fatalf("stderr missing progress: %s", stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout polluted by progress: %s", stdout.String())
	}
	jsonl, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(jsonl, []byte(`"schema":"fak-trajectory-audit/1"`)) {
		t.Fatalf("JSONL missing schema: %s", jsonl)
	}
}
