package agent

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func FuzzReadEngineRoundTrip(f *testing.F) {
	f.Add("plain.txt", []byte("hello"))
	f.Add("space name.txt", []byte{})
	f.Add("unicode.bin", []byte{0, 0xff, 1})
	f.Fuzz(func(t *testing.T, name string, data []byte) {
		if name == "" || !utf8.ValidString(name) || strings.ContainsAny(name, `/\\\x00`) {
			t.Skip()
		}
		root := t.TempDir()
		path := filepath.Join(root, name)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Skip()
		}
		raw, isErr := (readEngine{root: root}).read(name)
		if isErr {
			t.Fatalf("read failed: %s", raw)
		}
		body := decodeReadBody(t, raw)
		if body["file_path"] != name {
			t.Fatalf("path changed: %v", body["file_path"])
		}
		got := []byte(body["content"].(string))
		if encoded, ok := body["content_base64"].(string); ok {
			var err error
			got, err = base64.StdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatalf("decode binary content: %v", err)
			}
		}
		if string(got) != string(data) {
			t.Fatalf("read bytes changed: got %x want %x", got, data)
		}
		// Caller-controlled file bytes must never appear in typed metadata keys or errors.
		for _, key := range []string{"error", "error_code", "error_source"} {
			if v, ok := body[key]; ok && strings.Contains(v.(string), string(data)) && len(data) > 0 {
				t.Fatalf("file bytes leaked through %s", key)
			}
		}
	})
}
