package agent

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
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
