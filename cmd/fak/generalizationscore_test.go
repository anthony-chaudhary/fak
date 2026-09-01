package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/generalizationdebt"
)

func TestGeneralizationScoreJSONIsByteIdentical(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "specific.go")
	if err := os.WriteFile(path, []byte("package p\nfunc OpenAIBackend(){}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var first, second, stderr bytes.Buffer
	args := []string{"--workspace", root, "--json"}
	if code := runGeneralizationScorecard(&first, &stderr, args); code != 0 {
		t.Fatalf("first code=%d stderr=%s", code, stderr.String())
	}
	stderr.Reset()
	if code := runGeneralizationScorecard(&second, &stderr, args); code != 0 {
		t.Fatalf("second code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatalf("JSON differs:\n%s\n%s", first.Bytes(), second.Bytes())
	}
	var report generalizationdebt.Report
	if err := json.Unmarshal(first.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != generalizationdebt.Schema || len(report.Findings) != 1 {
		t.Fatalf("report=%+v", report)
	}
}

func TestGeneralizationScoreRejectsUnexpectedArgument(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runGeneralizationScorecard(&stdout, &stderr, []string{"extra"}); code != 2 {
		t.Fatalf("code=%d", code)
	}
}
