package qwen36nodereports

import (
	"archive/zip"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeZip(t *testing.T, path string, files map[string][]byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for name, body := range files {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(body); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestImportReportBundleSummarizesLatestPreflightAndLog(t *testing.T) {
	root := t.TempDir()
	archive := filepath.Join(root, "qwen36-node-reports-mac-20260619-010000.zip")
	outDir := filepath.Join(root, "out")
	preflight, _ := json.Marshal(map[string]any{
		"ok": false, "profile": "mac", "base_url": "http://example.invalid:8131/v1", "llama_server_found": false,
		"failures": []any{"llama-server was not found"},
		"checks": []any{
			map[string]any{"name": "llama_server", "ok": false},
			map[string]any{"name": "nvidia_smi", "ok": true, "required": false, "gpus": []any{map[string]any{"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "driver_version": "555.85"}}},
		},
	})
	writeZip(t, archive, map[string][]byte{
		"qwen36-reports/preflight-mac-20260619-010000.json": preflight,
		"qwen36-reports/server-mac-20260619-010000.log":     []byte("line1\nline2\n"),
	})
	dest, err := ExtractArchive(archive, outDir, false)
	if err != nil {
		t.Fatal(err)
	}
	summary := SummarizeDir(dest, 1)
	if summary["status"] != "PREFLIGHT_FAILED" || summary["preflight_count"] != 1 || summary["server_log_count"] != 1 || summary["latest_server_log_tail"] != "line2" {
		t.Fatalf("bad summary: %+v", summary)
	}
	latest := summary["latest_preflight"].(map[string]any)
	if latest["profile"] != "mac" || latest["llama_server_found"] != false {
		t.Fatalf("bad latest preflight: %+v", latest)
	}
	failed := latest["failed_checks"].([]string)
	if len(failed) != 1 || failed[0] != "llama_server" {
		t.Fatalf("failed checks = %v", failed)
	}
}

func TestImportReportBundleAcceptsWindowsUTF16Preflight(t *testing.T) {
	root := t.TempDir()
	archive := filepath.Join(root, "qwen36-node-reports-nvidia-20260619-141154.zip")
	outDir := filepath.Join(root, "out")
	body, _ := json.Marshal(map[string]any{
		"ok": true, "profile": "nvidia", "base_url": "http://example.invalid:8131/v1", "llama_server_found": true,
		"failures": []any{}, "checks": []any{map[string]any{"name": "llama_server", "ok": true}},
	})
	writeZip(t, archive, map[string][]byte{
		"qwen36-reports/preflight-nvidia-remote-20260619-141154.json": append([]byte{0xff, 0xfe}, utf16le(string(body))...),
	})
	summary := ImportReportBundle(ImportArgs{Archive: archive, OutDir: outDir, SkipTaildrop: true, Replace: true, LogTailLines: 60})
	if summary["imported"] != true || summary["status"] != "PREFLIGHT_OK" {
		t.Fatalf("bad import summary: %+v", summary)
	}
	latest := summary["latest_preflight"].(map[string]any)
	if latest["profile"] != "nvidia" || latest["llama_server_found"] != true {
		t.Fatalf("bad UTF16 preflight: %+v", latest)
	}
}

func TestArchiveCandidatesPrefersNewestBundle(t *testing.T) {
	inbox := t.TempDir()
	old := filepath.Join(inbox, "qwen36-node-reports-mac-old.zip")
	newest := filepath.Join(inbox, "qwen36-node-reports-mac-new.zip")
	writeZip(t, old, map[string][]byte{"qwen36-reports/preflight.json": []byte("{}")})
	writeZip(t, newest, map[string][]byte{"qwen36-reports/preflight.json": []byte("{}")})
	_ = os.Chtimes(old, ts(1000), ts(1000))
	_ = os.Chtimes(newest, ts(2000), ts(2000))
	candidates := ArchiveCandidates(inbox)
	if len(candidates) < 2 || candidates[0] != newest {
		t.Fatalf("candidates = %v", candidates)
	}
}

func TestExtractRejectsUnsafeZipMemberPaths(t *testing.T) {
	root := t.TempDir()
	for name, want := range map[string]string{"../escape.txt": "unsafe zip member path", "C:/Users/Public/escape.txt": "unsafe zip member path"} {
		archive := filepath.Join(root, strings.ReplaceAll(name, "/", "_")+".zip")
		writeZip(t, archive, map[string][]byte{name: []byte("bad")})
		_, err := ExtractArchive(archive, filepath.Join(root, "out"), false)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("ExtractArchive(%s) err = %v", name, err)
		}
	}
}

func TestNoBundleSummaryIsImportFailureWithoutTaildrop(t *testing.T) {
	summary := ImportReportBundle(ImportArgs{Inbox: t.TempDir(), SkipTaildrop: true, LogTailLines: 60})
	if summary["imported"] != false || !strings.Contains(summary["error"].(string), "no qwen36-node-reports") {
		t.Fatalf("bad no-bundle summary: %+v", summary)
	}
}

func utf16le(s string) []byte {
	var out []byte
	for _, r := range s {
		if r > 0xffff {
			r = '?'
		}
		out = append(out, byte(r), byte(r>>8))
	}
	return out
}

func ts(sec int64) time.Time { return time.Unix(sec, 0) }
