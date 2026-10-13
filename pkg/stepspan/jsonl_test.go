package stepspan

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func checkFiles(t *testing.T, dir string, cap int64, max int) int {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "fak-serve-*.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) > max {
		t.Fatalf("files=%d > %d", len(files), max)
	}
	count := 0
	for _, p := range files {
		info, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if info.Size() > cap {
			t.Fatalf("file %s bytes=%d", p, info.Size())
		}
		f, err := os.Open(p)
		if err != nil {
			t.Fatal(err)
		}
		scan := bufio.NewScanner(f)
		for scan.Scan() {
			var s Span
			if err := json.Unmarshal(scan.Bytes(), &s); err != nil {
				t.Fatal(err)
			}
			if err := s.Validate(); err != nil {
				t.Fatal(err)
			}
			count++
		}
		err = scan.Err()
		_ = f.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	return count
}
func TestJSONLRotationRestartAndDateRetention(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 10, 13, 0, 0, 0, 0, time.UTC)
	b, _ := json.Marshal(fixture())
	cap := int64((len(b) + 1) * 2)
	cfg := JSONLConfig{Directory: dir, Service: "fak-serve", MaxBytes: cap, MaxFiles: 3, Now: func() time.Time { return now }}
	peer := filepath.Join(dir, "fak-serve-peer-2026-10-13.jsonl")
	if err := os.WriteFile(peer, []byte("peer"), 0600); err != nil {
		t.Fatal(err)
	}
	sink, err := NewJSONLSink(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 12; i++ {
		if err := sink.WriteSpan(fixture()); err != nil {
			t.Fatal(err)
		}
		checkFilesExceptPeer(t, dir, cap, 3, peer)
	}
	if err := sink.Close(); err != nil {
		t.Fatal(err)
	}
	sink, err = NewJSONLSink(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := sink.WriteSpan(fixture()); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		now = now.Add(24 * time.Hour)
		if err := sink.WriteSpan(fixture()); err != nil {
			t.Fatal(err)
		}
		checkFilesExceptPeer(t, dir, cap, 3, peer)
	}
	if err := sink.Close(); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(peer); err != nil || string(data) != "peer" {
		t.Fatalf("peer changed %q %v", data, err)
	}
	if err := os.Remove(peer); err != nil {
		t.Fatal(err)
	}
	if n := checkFiles(t, dir, cap, 3); n < 1 {
		t.Fatal("no spans retained")
	}
	if _, err := os.Stat(filepath.Join(dir, "fak-serve-2026-10-19.jsonl")); err != nil {
		t.Fatal(err)
	}
}
func checkFilesExceptPeer(t *testing.T, dir string, cap int64, max int, peer string) {
	t.Helper()
	files, _ := os.ReadDir(dir)
	n := 0
	for _, f := range files {
		if filepath.Join(dir, f.Name()) == peer {
			continue
		}
		n++
		info, err := f.Info()
		if err != nil || info.Size() > cap {
			t.Fatalf("cap %s: %v", f.Name(), err)
		}
	}
	if n > max {
		t.Fatalf("retention %d > %d", n, max)
	}
}
func TestJSONLCapsAndErrors(t *testing.T) {
	for _, cfg := range []JSONLConfig{{Service: "../escape"}, {Service: "a/b"}, {MaxBytes: -1}, {MaxFiles: -1}} {
		cfg.Directory = t.TempDir()
		if _, err := NewJSONLSink(cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Fatal(err)
		}
	}
	dir := t.TempDir()
	sink, err := NewJSONLSink(JSONLConfig{Directory: dir, Service: "fak-serve", MaxBytes: 500, MaxFiles: 1})
	if err != nil {
		t.Fatal(err)
	}
	s := fixture()
	s.Name = strings.Repeat("x", 501)
	if err := sink.WriteSpan(s); !errors.Is(err, ErrSpanTooLarge) {
		t.Fatal(err)
	}
	s = fixture()
	for i := 0; i < 4; i++ {
		if err := sink.WriteSpan(s); err != nil {
			t.Fatal(err)
		}
	}
	if n := checkFiles(t, dir, 500, 1); n != 1 {
		t.Fatalf("records=%d", n)
	}
	if err := sink.WriteSpan(Span{}); !errors.Is(err, ErrInvalidSpan) {
		t.Fatal(err)
	}
	if err := sink.Close(); err != nil {
		t.Fatal(err)
	}
	if err := sink.WriteSpan(s); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}
func TestJSONLDefaultEmitter(t *testing.T) {
	t.Chdir(t.TempDir())
	e, err := NewDefaultEmitter("fak-serve")
	if err != nil {
		t.Fatal(err)
	}
	if !e.Emit(fixture()) {
		t.Fatal("enqueue")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(".fak", "spans", "fak-serve-*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("default files=%v %v", files, err)
	}
}
