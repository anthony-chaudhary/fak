package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestNativePerformanceFrontDoorMarkdownIsHonest(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runNativePerformance(&out, &errb, []string{"--frontdoor-md", "--as-of", "2026-08-26"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"2.3-2.9 decode tok/s", "3.3 vs 6.966061 tok/s (~47%)", "approximate, not accepted parity", "~0.2 tok/s with 0/5 exact", "diagnostic only"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("front-door markdown missing %q:\n%s", want, out.String())
		}
	}
}

func TestNativePerformanceFrontDoorCheckAndWriteDocs(t *testing.T) {
	root := t.TempDir()
	stale := nativeperf.FrontDoorBegin + "\nstale splice\n" + nativeperf.FrontDoorEnd
	for _, spec := range nativePerformanceFrontDoorDocs {
		path := filepath.Join(root, filepath.FromSlash(spec.path))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("before\n"+stale+"\nafter\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := nativeperf.BuildFrontDoorSnapshot(nativeperf.ActiveGraph(), time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runNativePerformanceFrontDoorDocs(&out, &errb, root, snapshot, false); code != 1 {
		t.Fatalf("stale check exit=%d out=%s err=%s", code, out.String(), errb.String())
	}
	out.Reset()
	if code := runNativePerformanceFrontDoorDocs(&out, &errb, root, snapshot, true); code != 0 {
		t.Fatalf("write exit=%d out=%s err=%s", code, out.String(), errb.String())
	}
	out.Reset()
	if code := runNativePerformanceFrontDoorDocs(&out, &errb, root, snapshot, false); code != 0 {
		t.Fatalf("fresh check exit=%d out=%s err=%s", code, out.String(), errb.String())
	}
}
