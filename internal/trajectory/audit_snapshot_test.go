package trajectory

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const auditSnapshotSecret = "private-prompt-never-publish"

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
