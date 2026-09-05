package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func decodeReadBody(t testing.TB, raw []byte) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode read result: %v", err)
	}
	return body
}

func TestReadEngineOutcomeMatrix(t *testing.T) {
	root := t.TempDir()
	e := readEngine{root: root}
	large := strings.Repeat("abcdefgh", 32<<10)
	cases := []struct {
		name string
		path string
		data []byte
		code string
	}{
		{name: "missing path", code: "missing_path"},
		{name: "not found", path: "absent.txt", code: "not_found"},
		{name: "directory", path: "dir", code: "is_directory"},
		{name: "unicode spaces", path: "snow man \u2603.txt", data: []byte("hello \u2603")},
		{name: "empty", path: "empty.txt", data: []byte{}},
		{name: "binary", path: "binary.bin", data: []byte{0, 0xff, 1, 0xfe}},
		{name: "large", path: "large.txt", data: []byte(large)},
	}
	if err := os.Mkdir(filepath.Join(root, "dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.data != nil {
				if err := os.WriteFile(filepath.Join(root, tc.path), tc.data, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			raw, isErr := e.read(tc.path)
			body := decodeReadBody(t, raw)
			if tc.code != "" {
				if !isErr || body["error_code"] != tc.code {
					t.Fatalf("error=%v body=%v, want code %q", isErr, body, tc.code)
				}
				joined := string(raw)
				if strings.Contains(joined, root) || strings.Contains(joined, "outside-unique") {
					t.Fatalf("error leaked path: %s", joined)
				}
				return
			}
			if isErr {
				t.Fatalf("unexpected error: %v", body)
			}
			if tc.name == "binary" {
				if enc, ok := body["encoding"].(string); !ok || enc != "base64" {
					t.Fatalf("binary encoding: got %v, want base64", body["encoding"])
				}
				if c, ok := body["content"].(string); !ok || c != "" {
					t.Fatalf("binary content: got %q, want empty string", c)
				}
				encoded, ok := body["content_base64"].(string)
				if !ok {
					t.Fatalf("binary content_base64 missing")
				}
				decoded, err := base64.StdEncoding.DecodeString(encoded)
				if err != nil {
					t.Fatalf("base64 decode: %v", err)
				}
				if string(decoded) != string(tc.data) {
					t.Fatalf("bytes differ: got %d want %d", len(decoded), len(tc.data))
				}
				return
			}
			got := []byte(body["content"].(string))
			if encoded, ok := body["content_base64"].(string); ok {
				var err error
				got, err = base64.StdEncoding.DecodeString(encoded)
				if err != nil {
					t.Fatal(err)
				}
			}
			if string(got) != string(tc.data) {
				t.Fatalf("bytes differ: got %d want %d", len(got), len(tc.data))
			}
		})
	}

	outside := filepath.Join(filepath.Dir(root), "outside-unique.txt")
	raw, isErr := e.read(outside)
	body := decodeReadBody(t, raw)
	if !isErr || body["error_code"] != "path_escape" {
		t.Fatalf("escape body=%v", body)
	}
	if strings.Contains(string(raw), outside) {
		t.Fatalf("escape error leaked rejected path: %s", raw)
	}
}

func TestReadEngineMutationShapes(t *testing.T) {
	root := t.TempDir()
	e := readEngine{root: root}
	path := filepath.Join(root, "mutable.txt")
	for _, data := range [][]byte{[]byte("aaaa"), []byte("bbbb"), []byte("x"), []byte("growth-value")} {
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		raw, isErr := e.read(path)
		if isErr {
			t.Fatalf("read %q failed: %s", data, raw)
		}
		if got := decodeReadBody(t, raw)["content"]; got != string(data) {
			t.Fatalf("got %q want %q", got, data)
		}
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if raw, isErr := e.read(path); !isErr || decodeReadBody(t, raw)["error_code"] != "not_found" {
		t.Fatalf("deleted read=%s", raw)
	}
	if err := os.WriteFile(path, []byte("recreated"), 0o644); err != nil {
		t.Fatal(err)
	}
	if raw, isErr := e.read(path); isErr || decodeReadBody(t, raw)["content"] != "recreated" {
		t.Fatalf("recreated read=%s", raw)
	}
}

func TestReadEngineConcurrentReads(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "concurrent.txt")
	want := strings.Repeat("concurrent-fixture", 1024)
	if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	e := readEngine{root: root}
	var wg sync.WaitGroup
	errs := make(chan string, 32)
	for i := 0; i < cap(errs); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			raw, isErr := e.read(path)
			if isErr {
				errs <- string(raw)
				return
			}
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				errs <- err.Error()
				return
			}
			if body["content"] != want {
				errs <- "returned bytes differ"
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestReadEngineSymlinkConfinement(t *testing.T) {
	temp := t.TempDir()
	rootDir := filepath.Join(temp, "root")
	outsideDir := filepath.Join(temp, "outside")
	if err := os.Mkdir(rootDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outsideDir, 0o755); err != nil {
		t.Fatal(err)
	}

	secretFile := filepath.Join(outsideDir, "secret.txt")
	if err := os.WriteFile(secretFile, []byte("super-secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	outsideSymlink := filepath.Join(rootDir, "link_outside.txt")
	if err := os.Symlink(secretFile, outsideSymlink); err != nil {
		t.Skipf("skipping symlink test: symlink creation not permitted: %v", err)
	}

	e := readEngine{root: rootDir}

	// 1. In-root symlink pointing outside e.root must be refused with path_escape / confinement.
	raw, isErr := e.read("link_outside.txt")
	body := decodeReadBody(t, raw)
	if !isErr {
		t.Fatalf("expected symlink escape to fail, got success: %v", body)
	}
	if body["error_code"] != "path_escape" {
		t.Fatalf("error_code: got %v, want %q", body["error_code"], "path_escape")
	}
	if body["error_source"] != "confinement" {
		t.Fatalf("error_source: got %v, want %q", body["error_source"], "confinement")
	}

	// 2. Legitimate symlink inside root pointing to a valid file inside root should succeed.
	insideFile := filepath.Join(rootDir, "inside.txt")
	if err := os.WriteFile(insideFile, []byte("inside-content"), 0o644); err != nil {
		t.Fatal(err)
	}
	insideSymlink := filepath.Join(rootDir, "link_inside.txt")
	if err := os.Symlink(insideFile, insideSymlink); err != nil {
		t.Skipf("skipping inside symlink test: symlink creation failed: %v", err)
	}

	raw, isErr = e.read("link_inside.txt")
	if isErr {
		t.Fatalf("expected inside symlink to succeed, got error: %s", raw)
	}
	body = decodeReadBody(t, raw)
	if body["content"] != "inside-content" {
		t.Fatalf("content: got %q, want %q", body["content"], "inside-content")
	}
}

func TestReadEngineLinePaginationAndNumbering(t *testing.T) {
	root := t.TempDir()
	e := readEngine{root: root}
	filePath := filepath.Join(root, "lines.txt")
	var lines []string
	for i := 1; i <= 10; i++ {
		lines = append(lines, fmt.Sprintf("line %d", i))
	}
	if err := os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Sliced read with offset=3, limit=4
	raw, isErr := e.readWithOptions("lines.txt", 3, 4, false)
	if isErr {
		t.Fatalf("unexpected error: %s", raw)
	}
	body := decodeReadBody(t, raw)
	wantContent := "line 3\nline 4\nline 5\nline 6"
	if body["content"] != wantContent {
		t.Fatalf("got content %q, want %q", body["content"], wantContent)
	}
	if offset, ok := body["offset"].(float64); !ok || int(offset) != 3 {
		t.Fatalf("got offset %v, want 3", body["offset"])
	}
	if limit, ok := body["limit"].(float64); !ok || int(limit) != 4 {
		t.Fatalf("got limit %v, want 4", body["limit"])
	}
	if body["truncated"] != true {
		t.Fatalf("got truncated %v, want true", body["truncated"])
	}
	if total, ok := body["total_lines"].(float64); !ok || int(total) != 10 {
		t.Fatalf("got total_lines %v, want 10", body["total_lines"])
	}

	// 2. Offset past end
	raw, isErr = e.readWithOptions("lines.txt", 25, 5, false)
	if isErr {
		t.Fatalf("unexpected error: %s", raw)
	}
	body = decodeReadBody(t, raw)
	if body["content"] != "" {
		t.Fatalf("got content %q, want empty string", body["content"])
	}

	// 3. Line numbering with offset=2, limit=3, line_numbers=true
	raw, isErr = e.readWithOptions("lines.txt", 2, 3, true)
	if isErr {
		t.Fatalf("unexpected error: %s", raw)
	}
	body = decodeReadBody(t, raw)
	wantNumbered := "2: line 2\n3: line 3\n4: line 4"
	if body["content"] != wantNumbered {
		t.Fatalf("got content %q, want %q", body["content"], wantNumbered)
	}

	// 4. Invocation via Complete with abi.ToolCall
	call := &abi.ToolCall{
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(`{"file_path":"lines.txt","offset":5,"limit":2,"line_numbers":true}`),
		},
	}
	res, err := e.Complete(context.Background(), call)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	var completeBody map[string]any
	if err := json.Unmarshal(res.Payload.Inline, &completeBody); err != nil {
		t.Fatalf("json unmarshal: %v", err)
	}
	wantCompleteNumbered := "5: line 5\n6: line 6"
	if completeBody["content"] != wantCompleteNumbered {
		t.Fatalf("got Complete content %q, want %q", completeBody["content"], wantCompleteNumbered)
	}
}
