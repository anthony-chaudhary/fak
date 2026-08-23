package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestCommitPreviewDisclosesDatedNoteIndexEffect(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runCommitPreview(&out, &errOut, "docs(notes): add witness (#8567) (fak docs)", []string{"docs/notes/WITNESS-2026-08-23.md"}, t.TempDir(), "main", true, false)
	if code != 0 {
		t.Fatalf("preview code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Paths []string `json:"required_paths"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, path := range got.Paths {
		if path == "INDEX.md" {
			found = true
		}
	}
	if !found {
		t.Fatalf("preview paths=%v, want generated INDEX.md effect", got.Paths)
	}
}
