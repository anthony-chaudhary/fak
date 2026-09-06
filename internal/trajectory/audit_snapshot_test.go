package trajectory

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

const auditSnapshotSecret = "private-prompt-never-publish"

func TestAuditSnapshotLargeCorpusToolMetadataRendersDeterministically(t *testing.T) {
	base := t.TempDir()
	claudeRoot, codexRoot := writeAuditSnapshotLargeMetadataRoots(t, base)
	staged := filepath.Join(base, "staged")
	if err := os.Mkdir(staged, 0o700); err != nil {
		t.Fatal(err)
	}
	manifest := AuditSnapshotManifest{
		Schema: AuditSnapshotSchema, AuditSchema: AuditSchema,
		CapturedAtUTC: time.Date(2026, 8, 28, 0, 0, 0, 0, time.UTC).Format(time.RFC3339Nano),
	}
	for _, source := range []AuditSource{
		{Name: AuditSourceClaude, Root: claudeRoot, RootLabel: "claude/projects"},
		{Name: AuditSourceCodex, Root: codexRoot, RootLabel: "codex/sessions"},
	} {
		meta, files, err := captureAuditSnapshotSource(staged, source, AuditOptions{})
		if err != nil {
			t.Fatal(err)
		}
		manifest.Sources = append(manifest.Sources, meta)
		manifest.Files = append(manifest.Files, files...)
	}
	sortSnapshotManifest(&manifest)
	var err error
	manifest.CorpusDigest, err = auditSnapshotCorpusDigest(manifest)
	if err != nil {
		t.Fatal(err)
	}

	var firstJSON, firstMarkdown []byte
	for run := 0; run < 16; run++ {
		result, err := runAuditSnapshotCorpus(staged, manifest)
		if err != nil {
			t.Fatal(err)
		}
		if result.Summary.RawFragments != 432 || result.Summary.CanonicalTranscripts != 430 {
			t.Fatalf("corpus shape = %d raw/%d canonical, want 432/430", result.Summary.RawFragments, result.Summary.CanonicalTranscripts)
		}
		if result.Summary.UnknownExemplars.DroppedObservations == 0 {
			t.Fatal("large corpus did not cross the bounded unknown-exemplar reservoir")
		}
		metadata := auditToolResultByName(result.Summary.ToolResults, "exec_command")
		if metadata.Results != 432 || metadata.ExitKnown != 432 || metadata.ExitZero != 432 || metadata.ExitNonzero != 0 || metadata.DurationKnown != 432 || metadata.DurationMS != 4320 {
			t.Fatalf("run %d tool metadata = %+v, want lexical nested-envelope precedence", run, metadata)
		}
		jsonl, markdown := renderAuditSnapshotResult(t, result)
		if run == 0 {
			firstJSON, firstMarkdown = jsonl, markdown
			continue
		}
		if !bytes.Equal(firstJSON, jsonl) || !bytes.Equal(firstMarkdown, markdown) {
			t.Fatalf("run %d rendered audit drifted: jsonl_equal=%t markdown_equal=%t", run, bytes.Equal(firstJSON, jsonl), bytes.Equal(firstMarkdown, markdown))
		}
	}
}

func writeAuditSnapshotLargeMetadataRoots(t *testing.T, base string) (string, string) {
	t.Helper()
	claudeRoot := filepath.Join(base, "live-claude", "projects")
	codexRoot := filepath.Join(base, "live-codex", "sessions")
	if err := os.MkdirAll(claudeRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	for fragment := 0; fragment < 432; fragment++ {
		session := fragment
		if session >= 430 {
			session -= 430
		}
		shape := map[string]any{"type": fmt.Sprintf("unknown_%03d", fragment%64)}
		for field := 0; field < 32; field++ {
			shape[fmt.Sprintf("shape_field_%02d_with_bounded_content_free_name", field)] = map[string]any{"nested": field}
		}
		shapeJSON, err := json.Marshal(shape)
		if err != nil {
			t.Fatal(err)
		}
		rows := []string{
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:00Z","type":"session_meta","payload":{"id":"session-%03d","model_provider":"test-provider","cli_version":"1.2.3"}}`, session),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:01Z","type":"turn_context","payload":{"model":"qwen-test-%d","turn_id":"turn-a"}}`, fragment%3),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:02Z","type":"turn_context","payload":{"model":"qwen-test-%d","turn_id":"turn-b"}}`, (fragment+1)%3),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:03Z","type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"call-%03d","arguments":"{}"}}`, fragment),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:04Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call-%03d","output":{"z_later":{"exit_code":9,"duration_ms":90},"a_first":{"exit_code":0,"duration_ms":10}}}}`, fragment),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:05Z","type":"response_item","payload":%s}`, shapeJSON),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:06Z","type":"event_msg","payload":{"type":"unknown_event_%03d","shape":{"leaf":true}}}`, fragment%64),
			fmt.Sprintf(`{"timestamp":"2026-08-28T00:00:07Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":%d,"cached_input_tokens":1,"cache_write_input_tokens":0,"output_tokens":2,"reasoning_output_tokens":0,"total_tokens":%d}}}}`, session+10, session+12),
		}
		path := filepath.Join(codexRoot, fmt.Sprintf("batch-%03d", fragment/50), fmt.Sprintf("fragment-%03d.jsonl", fragment))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return claudeRoot, codexRoot
}

func TestAuditSnapshotCaptureDeleteRootsReplayDeterministically(t *testing.T) {
	base := t.TempDir()
	claudeRoot, codexRoot := writeAuditSnapshotSyntheticRoots(t, base)
	target := filepath.Join(base, "private-snapshot")
	manifest, captured, err := CaptureAuditSnapshot(target, auditSnapshotOptions(claudeRoot, codexRoot))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Schema != AuditSnapshotSchema || manifest.AuditSchema != AuditSchema || len(manifest.Files) != 2 || manifest.CorpusDigest == "" || manifest.CapturedOutputDigest == "" {
		t.Fatalf("manifest = %+v", manifest)
	}
	assertAuditSnapshotPrivateTree(t, target)
	manifestBytes, err := os.ReadFile(filepath.Join(target, auditSnapshotManifestName))
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{auditSnapshotSecret, claudeRoot, codexRoot} {
		if bytes.Contains(manifestBytes, []byte(forbidden)) {
			t.Fatalf("manifest leaked %q: %s", forbidden, manifestBytes)
		}
	}
	if err := os.RemoveAll(claudeRoot); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(codexRoot); err != nil {
		t.Fatal(err)
	}
	firstManifest, first, err := ReplayAuditSnapshot(target)
	if err != nil {
		t.Fatal(err)
	}
	secondManifest, second, err := ReplayAuditSnapshot(target)
	if err != nil {
		t.Fatal(err)
	}
	if firstManifest.CorpusDigest != manifest.CorpusDigest || secondManifest.CorpusDigest != manifest.CorpusDigest {
		t.Fatal("replay corpus identity changed")
	}
	if captured.Summary.Transcripts != 2 || first.Summary.Transcripts != 2 || second.Summary.Transcripts != 2 {
		t.Fatalf("sessions = capture:%d first:%d second:%d", captured.Summary.Transcripts, first.Summary.Transcripts, second.Summary.Transcripts)
	}
	if !reflect.DeepEqual(captured.Bottlenecks, first.Bottlenecks) || !reflect.DeepEqual(first.Bottlenecks, second.Bottlenecks) {
		t.Fatalf("top ordering changed: capture=%+v first=%+v second=%+v", captured.Bottlenecks, first.Bottlenecks, second.Bottlenecks)
	}
	capJSON, capMD := renderAuditSnapshotResult(t, captured)
	firstJSON, firstMD := renderAuditSnapshotResult(t, first)
	secondJSON, secondMD := renderAuditSnapshotResult(t, second)
	if !bytes.Equal(capJSON, firstJSON) || !bytes.Equal(firstJSON, secondJSON) || !bytes.Equal(capMD, firstMD) || !bytes.Equal(firstMD, secondMD) {
		t.Fatal("capture/replay artifacts are not byte-identical")
	}
	for name, output := range map[string][]byte{"jsonl": firstJSON, "markdown": firstMD} {
		if bytes.Contains(output, []byte(auditSnapshotSecret)) {
			t.Fatalf("%s leaked transcript payload: %s", name, output)
		}
		if !bytes.Contains(output, []byte(manifest.CorpusDigest)) {
			t.Fatalf("%s does not name corpus digest", name)
		}
	}
}

func TestAuditSnapshotRefusalEnvelope(t *testing.T) {
	t.Run("existing target", func(t *testing.T) {
		base := t.TempDir()
		claudeRoot, codexRoot := writeAuditSnapshotSyntheticRoots(t, base)
		target := filepath.Join(base, "exists")
		if err := os.Mkdir(target, 0o700); err != nil {
			t.Fatal(err)
		}
		_, _, err := CaptureAuditSnapshot(target, auditSnapshotOptions(claudeRoot, codexRoot))
		assertAuditSnapshotCode(t, err, "SNAPSHOT_TARGET_EXISTS")
	})

	for _, test := range []struct {
		name string
		code string
		edit func(*testing.T, string)
	}{
		{"missing file", "SNAPSHOT_FILE_MISSING", func(t *testing.T, root string) { os.Remove(snapshotFirstFile(t, root)) }},
		{"changed bytes", "SNAPSHOT_FILE_CHANGED", func(t *testing.T, root string) { os.WriteFile(snapshotFirstFile(t, root), []byte("changed\n"), 0o600) }},
		{"extra file", "SNAPSHOT_FILE_EXTRA", func(t *testing.T, root string) {
			os.WriteFile(filepath.Join(root, "claude", "projects", "extra.jsonl"), []byte("{}\n"), 0o600)
		}},
		{"path traversal", "SNAPSHOT_PATH_INVALID", func(t *testing.T, root string) {
			mutateAuditSnapshotManifest(t, root, func(m *AuditSnapshotManifest) { m.Files[0].RelativePath = "../escape.jsonl" })
		}},
		{"malformed manifest", "SNAPSHOT_MANIFEST_MALFORMED", func(t *testing.T, root string) {
			os.WriteFile(filepath.Join(root, auditSnapshotManifestName), []byte("{not-json"), 0o600)
		}},
		{"incompatible schema", "SNAPSHOT_SCHEMA_INCOMPATIBLE", func(t *testing.T, root string) {
			mutateAuditSnapshotManifest(t, root, func(m *AuditSnapshotManifest) { m.Schema = "future/99" })
		}},
		{"insecure permissions", "SNAPSHOT_PERMISSION_INSECURE", func(t *testing.T, root string) { os.Chmod(snapshotFirstFile(t, root), 0o644) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.name == "insecure permissions" && runtime.GOOS == "windows" {
				t.Skip("POSIX file permissions not supported on Windows")
			}
			root := freshAuditSnapshot(t)
			test.edit(t, root)
			_, _, err := ReplayAuditSnapshot(root)
			assertAuditSnapshotCode(t, err, test.code)
		})
	}

	t.Run("concurrent mutation", func(t *testing.T) {
		root := freshAuditSnapshot(t)
		_, _, err := replayAuditSnapshot(root, func() {
			if err := os.WriteFile(snapshotFirstFile(t, root), []byte("changed during parse\n"), 0o600); err != nil {
				t.Fatal(err)
			}
		})
		assertAuditSnapshotCode(t, err, "SNAPSHOT_FILE_CHANGED")
	})
}

func writeAuditSnapshotSyntheticRoots(t *testing.T, base string) (string, string) {
	t.Helper()
	claudeRoot := filepath.Join(base, "live-claude", "projects")
	codexRoot := filepath.Join(base, "live-codex", "sessions")
	claudePath := filepath.Join(claudeRoot, "project", "claude-session.jsonl")
	codexPath := filepath.Join(codexRoot, "2026", "08", "28", "codex-session.jsonl")
	for _, path := range []string{claudePath, codexPath} {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	claude := strings.Join([]string{
		`{"type":"user","sessionId":"claude-session","message":{"role":"user","content":"snapshot-topic ` + auditSnapshotSecret + `"}}`,
		`{"type":"assistant","sessionId":"claude-session","message":{"id":"msg-1","model":"claude-test","usage":{"input_tokens":10,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":1},"content":"done"}}`,
	}, "\n") + "\n"
	codex := strings.Join([]string{
		`{"timestamp":"2026-08-28T00:00:00Z","type":"session_meta","payload":{"id":"codex-session","model_provider":"openai"}}`,
		`{"timestamp":"2026-08-28T00:00:01Z","type":"turn_context","payload":{"model":"gpt-test","turn_id":"turn-1"}}`,
		`{"timestamp":"2026-08-28T00:00:02Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"snapshot-topic ` + auditSnapshotSecret + `"}]}}`,
		`{"timestamp":"2026-08-28T00:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":20,"cached_input_tokens":5,"cache_write_input_tokens":0,"output_tokens":4,"reasoning_output_tokens":0,"total_tokens":24},"total_token_usage":{"input_tokens":20,"cached_input_tokens":5,"cache_write_input_tokens":0,"output_tokens":4,"reasoning_output_tokens":0,"total_tokens":24}}}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(claudePath, []byte(claude), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexPath, []byte(codex), 0o600); err != nil {
		t.Fatal(err)
	}
	return claudeRoot, codexRoot
}

func auditSnapshotOptions(claudeRoot, codexRoot string) AuditOptions {
	return AuditOptions{
		Sources: []AuditSource{
			{Name: AuditSourceClaude, Root: claudeRoot, RootLabel: "claude/projects"},
			{Name: AuditSourceCodex, Root: codexRoot, RootLabel: "codex/sessions"},
		},
		Since:        time.Hour,
		Now:          time.Now().UTC(),
		UserContains: "snapshot-topic",
	}
}

func freshAuditSnapshot(t *testing.T) string {
	t.Helper()
	base := t.TempDir()
	claudeRoot, codexRoot := writeAuditSnapshotSyntheticRoots(t, base)
	target := filepath.Join(base, "snapshot")
	if _, _, err := CaptureAuditSnapshot(target, auditSnapshotOptions(claudeRoot, codexRoot)); err != nil {
		t.Fatal(err)
	}
	return target
}

func renderAuditSnapshotResult(t *testing.T, result AuditResult) ([]byte, []byte) {
	t.Helper()
	var jsonl, markdown bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	return jsonl.Bytes(), markdown.Bytes()
}

func assertAuditSnapshotPrivateTree(t *testing.T, root string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		return
	}
	if err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.Mode().Perm()&0o077 != 0 {
			t.Errorf("%s permissions = %04o", path, info.Mode().Perm())
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func snapshotFirstFile(t *testing.T, root string) string {
	t.Helper()
	payload, err := os.ReadFile(filepath.Join(root, auditSnapshotManifestName))
	if err != nil {
		t.Fatal(err)
	}
	var manifest AuditSnapshotManifest
	if err := json.Unmarshal(payload, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Files) == 0 {
		t.Fatal("snapshot has no files")
	}
	return filepath.Join(root, filepath.FromSlash(snapshotSourcePath(manifest.Files[0].Source)), filepath.FromSlash(manifest.Files[0].RelativePath))
}

func mutateAuditSnapshotManifest(t *testing.T, root string, mutate func(*AuditSnapshotManifest)) {
	t.Helper()
	path := filepath.Join(root, auditSnapshotManifestName)
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest AuditSnapshotManifest
	if err := json.Unmarshal(payload, &manifest); err != nil {
		t.Fatal(err)
	}
	mutate(&manifest)
	payload, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
}

func assertAuditSnapshotCode(t *testing.T, err error, code string) {
	t.Helper()
	var typed *AuditSnapshotError
	if !errors.As(err, &typed) || typed.Code != code {
		t.Fatalf("error = %v, want %s", err, code)
	}
}
