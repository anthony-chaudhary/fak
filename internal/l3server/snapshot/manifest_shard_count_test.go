package snapshot

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Tiny temporary manifests and one tiny shard; estimate is unmeasured.
// fak-test:runtime fast est=10ms lane=default
func TestSnapshotManifestShardCount(t *testing.T) {
	t.Parallel()
	minInt := -int(^uint(0)>>1) - 1
	for _, count := range []int{-1, minInt} {
		t.Run(fmt.Sprintf("negative_%d", count), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "snapshot.json")
			data := []byte(fmt.Sprintf(`{"version":1,"shard_count":%d,"total_keys":7}`, count))
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			r := NewReader(dir)
			m, err := r.ReadManifest()
			if err == nil || !strings.Contains(err.Error(), "shard_count must be non-negative") || !reflect.DeepEqual(m, Manifest{}) {
				t.Fatalf("ReadManifest = %#v, %v; want zero manifest and negative-count error", m, err)
			}
			if entries, err := r.ReadAllEntries(); err == nil || entries != nil {
				t.Fatalf("ReadAllEntries = %#v, %v; want nil entries and error", entries, err)
			}
			if m, entries, err := Load(dir, 3); err == nil || entries != nil || !reflect.DeepEqual(m, Manifest{}) {
				t.Fatalf("Load = %#v, %#v, %v; want no successful snapshot", m, entries, err)
			}
			if got, err := os.ReadFile(path); err != nil || string(got) != string(data) {
				t.Fatalf("reader changed manifest: %q, %v", got, err)
			}
		})
	}
	t.Run("empty", func(t *testing.T) {
		dir := t.TempDir()
		want := Manifest{Version: 1, ShardCount: 0}
		if err := NewWriter(dir).WriteManifest(want); err != nil {
			t.Fatal(err)
		}
		m, entries, err := Load(dir, 3)
		if err != nil || !reflect.DeepEqual(m, want) || len(entries) != 0 {
			t.Fatalf("empty Load = %#v, %#v, %v", m, entries, err)
		}
	})
	t.Run("one shard and unknown field", func(t *testing.T) {
		dir := t.TempDir()
		data := []byte(`{"version":1,"shard_count":1,"future_field":"kept compatible"}`)
		if err := os.WriteFile(filepath.Join(dir, "snapshot.json"), data, 0600); err != nil {
			t.Fatal(err)
		}
		r := NewReader(dir)
		if m, err := r.ReadManifest(); err != nil || m.ShardCount != 1 {
			t.Fatalf("positive ReadManifest = %#v, %v", m, err)
		}
		if _, entries, err := Load(dir, 3); err == nil || entries != nil {
			t.Fatalf("missing shard Load = %#v, %v; want error", entries, err)
		}
		want := []KVEntry{{Key: []byte("k"), Value: []byte("v"), TTLMs: 42}}
		if err := NewWriter(dir).WriteShard(0, want); err != nil {
			t.Fatal(err)
		}
		m, entries, err := Load(dir, 3)
		if err != nil || m.ShardCount != 1 || !reflect.DeepEqual(entries, want) {
			t.Fatalf("positive Load = %#v, %#v, %v", m, entries, err)
		}
	})
}
