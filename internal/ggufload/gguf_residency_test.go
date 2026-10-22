package ggufload

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestPageCacheResidencyStartupCase(t *testing.T) {
	const gib = int64(1) << 30
	cases := []struct {
		resident, file int64
		want           string
	}{
		{0, 0, ""},
		{0, 246 * gib, startupCaseCold},
		{gib, 246 * gib, startupCaseCold},
		{3 * gib, 246 * gib, startupCaseWarm},
		{246 * gib, 246 * gib, startupCaseWarm},
	}
	for _, tc := range cases {
		if got := (PageCacheResidency{ResidentBytes: tc.resident, FileBytes: tc.file}).StartupCase(); got != tc.want {
			t.Errorf("StartupCase(resident=%d, file=%d) = %q, want %q", tc.resident, tc.file, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestShardFilesExpandsSplitModels(t *testing.T) {
	got := ShardFiles("/m/V-00003-of-00003.gguf")
	want := []string{"/m/V-00001-of-00003.gguf", "/m/V-00002-of-00003.gguf", "/m/V-00003-of-00003.gguf"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ShardFiles(split) = %v, want %v", got, want)
	}
	if got := ShardFiles("/m/single.gguf"); !reflect.DeepEqual(got, []string{"/m/single.gguf"}) {
		t.Fatalf("ShardFiles(single) = %v", got)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestMeasurePageCacheResidency(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	payload := make([]byte, 3*os.Getpagesize()+17)
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	r, err := MeasurePageCacheResidency([]string{path})
	if runtime.GOOS != "linux" {
		if !errors.Is(err, errPageCacheResidencyUnsupported) {
			t.Fatalf("err = %v, want errPageCacheResidencyUnsupported", err)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if r.Files != 1 || r.FileBytes != int64(len(payload)) || r.ResidentBytes < 0 || r.ResidentBytes > r.FileBytes {
		t.Fatalf("residency = %+v, want 1 file of %d bytes and resident within it", r, len(payload))
	}
	if _, err := MeasurePageCacheResidency([]string{filepath.Join(dir, "absent.gguf")}); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing shard err = %v, want os.ErrNotExist", err)
	}
}
