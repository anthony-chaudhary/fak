package dispatchaudit

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

// scanWorkerLogs returns the sorted Log names of a ScanDir result, for compact
// set comparisons.
func scanWorkerLogs(workers []Worker) []string {
	out := make([]string, 0, len(workers))
	for _, w := range workers {
		out = append(out, w.Log)
	}
	sort.Strings(out)
	return out
}

// TestScanDirSinceWindowsByNameStamp pins the #3466 audit window: ScanDirSince
// must skip a log whose spawn stamp (parsed from the file NAME, the
// dispatchconservation.ParseLogStampUTC grammar) falls before `since` WITHOUT
// opening it, include a log stamped exactly AT `since` (>= semantics,
// CollectUnits parity), and — with a zero window — behave identically to the
// legacy scan-everything ScanDir so existing callers are unaffected.
func TestScanDirSinceWindowsByNameStamp(t *testing.T) {
	dir := t.TempDir()

	// Old run: stamped a month before the window.
	writeFile(t, dir, "resolve-100-20260601-000000.log",
		"# fak-spawn 20260601-000000 issue=100 lane=tools backend=claude argv0=claude\n"+
			"working...\n")
	writeFile(t, dir, "resolve-100-20260601-000000.backend", "claude")

	// Boundary run: stamped exactly at the window start — must stay IN.
	writeFile(t, dir, "resolve-200-20260701-000000.log",
		"# fak-spawn 20260701-000000 issue=200 lane=docs backend=claude argv0=claude\n"+
			"working...\n")

	// Fresh run: stamped inside the window, with a sidecar that must still pair.
	writeFile(t, dir, "resolve-300-20260708-120000.log",
		"# fak-spawn 20260708-120000 issue=300 lane=cmd backend=opencode argv0=opencode\n"+
			"working...\n")
	writeFile(t, dir, "resolve-300-20260708-120000.backend", "opencode")

	since := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	windowed, err := ScanDirSince(dir, since)
	if err != nil {
		t.Fatalf("ScanDirSince: %v", err)
	}
	got := scanWorkerLogs(windowed)
	want := []string{"resolve-200-20260701-000000.log", "resolve-300-20260708-120000.log"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("windowed scan = %v, want %v (old stamp skipped, boundary stamp kept)", got, want)
	}
	// The in-window log's sidecar still pairs — the sidecar index is windowed by
	// the SAME name stamp, so it can never diverge from its log.
	for _, w := range windowed {
		if w.Issue == "300" {
			if w.SidecarMissing || w.SidecarBackend != BackendOpencode {
				t.Fatalf("in-window worker lost its sidecar pairing: %+v", w)
			}
		}
	}

	// Zero window == legacy ScanDir == scan everything, byte-identical.
	all, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	allSince, err := ScanDirSince(dir, time.Time{})
	if err != nil {
		t.Fatalf("ScanDirSince(zero): %v", err)
	}
	if len(all) != 3 || len(allSince) != 3 {
		t.Fatalf("zero-window scans = %d / %d workers, want 3 each (scan everything)", len(all), len(allSince))
	}
	gotAll, gotAllSince := scanWorkerLogs(all), scanWorkerLogs(allSince)
	for i := range gotAll {
		if gotAll[i] != gotAllSince[i] {
			t.Fatalf("zero-window ScanDirSince diverged from ScanDir: %v vs %v", gotAllSince, gotAll)
		}
	}
}

// TestScanDirSinceMtimeFallbackForStamplessName pins the fallback rung: a
// resolve log whose name carries NO parseable spawn stamp is windowed by its
// file mtime instead of being unconditionally dropped (the audit never loses
// legacy evidence to the window) — old mtime out, fresh mtime in, and the
// zero window keeps both.
func TestScanDirSinceMtimeFallbackForStamplessName(t *testing.T) {
	dir := t.TempDir()
	since := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)

	writeFile(t, dir, "resolve-77-legacy.log",
		"# fak-spawn legacy issue=77 lane=tools backend=claude argv0=claude\nworking...\n")
	oldMod := since.Add(-24 * time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "resolve-77-legacy.log"), oldMod, oldMod); err != nil {
		t.Fatalf("chtime old stampless log: %v", err)
	}

	writeFile(t, dir, "resolve-88-legacy.log",
		"# fak-spawn legacy issue=88 lane=tools backend=claude argv0=claude\nworking...\n")
	freshMod := since.Add(24 * time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "resolve-88-legacy.log"), freshMod, freshMod); err != nil {
		t.Fatalf("chtime fresh stampless log: %v", err)
	}

	windowed, err := ScanDirSince(dir, since)
	if err != nil {
		t.Fatalf("ScanDirSince: %v", err)
	}
	if got := scanWorkerLogs(windowed); len(got) != 1 || got[0] != "resolve-88-legacy.log" {
		t.Fatalf("windowed stampless scan = %v, want only resolve-88-legacy.log (mtime fallback)", got)
	}

	all, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("zero-window scan = %d workers, want 2 (window off = scan everything)", len(all))
	}
}

// TestScanDirSignaturesSinceSkipsOldLogs pins the windowed signature scan: an
// out-of-window log is skipped BEFORE its capped 2 MB text read, so a panic in
// an ancient log stops re-surfacing in every scheduled audit — while the zero
// window still aggregates across all history exactly as before (#3466).
func TestScanDirSignaturesSinceSkipsOldLogs(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "resolve-1-20260601-000001.log",
		"# fak-spawn 20260601-000001 issue=1 lane=cmd backend=claude\npanic: runtime error: index out of range [3]\n")
	writeFile(t, dir, "resolve-2-20260708-000002.log",
		"# fak-spawn 20260708-000002 issue=2 lane=cmd backend=claude\npanic: runtime error: index out of range [9]\n")

	since := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	windowed, err := ScanDirSignaturesSince(dir, DefaultSignatureThresholds(), since)
	if err != nil {
		t.Fatalf("ScanDirSignaturesSince: %v", err)
	}
	if len(windowed) != 1 {
		t.Fatalf("windowed signatures = %d findings, want 1: %+v", len(windowed), windowed)
	}
	if windowed[0].Count != 1 || len(windowed[0].Logs) != 1 || windowed[0].Logs[0] != "resolve-2-20260708-000002.log" {
		t.Fatalf("windowed panic must count only the in-window log, got count=%d logs=%v",
			windowed[0].Count, windowed[0].Logs)
	}

	all, err := ScanDirSignatures(dir, DefaultSignatureThresholds())
	if err != nil {
		t.Fatalf("ScanDirSignatures: %v", err)
	}
	if len(all) != 1 || all[0].Count != 2 || len(all[0].Logs) != 2 {
		t.Fatalf("zero-window signatures must aggregate both sessions, got %+v", all)
	}
}
