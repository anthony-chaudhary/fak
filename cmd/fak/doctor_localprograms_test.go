package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// #13831: report installer siblings even when PATH names only the selected copy.
// fak-test:runtime fast est=20ms lane=default
func TestDoctorLocalProgramsSiblingsAreReportOnly(t *testing.T) {
	t.Parallel()
	appData := t.TempDir()
	exe := "fak"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	paths := make([]string, 3)
	for i, build := range []string{"old", "middle", "newest"} {
		paths[i] = filepath.Join(appData, "Programs", "fak", build, "bin", exe)
		if err := os.MkdirAll(filepath.Dir(paths[i]), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(paths[i], []byte(build), 0o755); err != nil {
			t.Fatal(err)
		}
		mod := time.Unix(1700000000+int64(i), 0)
		if err := os.Chtimes(paths[i], mod, mod); err != nil {
			t.Fatal(err)
		}
	}
	h := selfinstall.Host{LocalAppData: appData, PathDirs: []string{filepath.Dir(paths[0])}}
	var rep appversion.BinaryReport
	appendLocalProgramsFindings(&rep, h)
	if rep.Findings != 2 || len(rep.Recommendations) != 2 {
		t.Fatalf("doctor report = %+v, want two sibling findings", rep)
	}
	for i, rec := range rep.Recommendations {
		if rec.Check != "binary-localprograms-sibling" || rec.Severity != appversion.SeverityWarn ||
			!strings.Contains(rec.Finding, paths[i+1]) || !strings.Contains(rec.Finding, paths[0]) {
			t.Fatalf("doctor finding = %+v, want sibling %q and selected %q", rec, paths[i+1], paths[0])
		}
	}
	var human bytes.Buffer
	writeBinaryDoctorHuman(&human, rep)
	encoded, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	var decoded appversion.BinaryReport
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.Findings != 2 {
		t.Fatalf("structured findings lost: %+v, %v", decoded, err)
	}
	for i, want := range []string{"old", "middle", "newest"} {
		got, err := os.ReadFile(paths[i])
		if err != nil || string(got) != want || !strings.Contains(human.String(), paths[i]) {
			t.Fatalf("report changed or omitted %q: bytes=%q err=%v report=%s", paths[i], got, err, human.String())
		}
	}
	t.Log(human.String())
}
