package snapshot

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Estimate only: temporary files and keys up to 65,537 bytes, no serving manager.
func TestWriteShardRequiresRepresentableKeys(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		key  []byte
	}{
		{"empty", nil},
		{"wraps_to_sentinel", bytes.Repeat([]byte{'x'}, 1<<16)},
		{"wraps_to_short_key", bytes.Repeat([]byte{'x'}, 1<<16+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entries := []KVEntry{
				{Key: []byte("valid"), Value: []byte("first")},
				{Key: tc.key, Value: []byte("invalid")},
			}
			t.Run("preserve_existing", func(t *testing.T) {
				dir := t.TempDir()
				path := filepath.Join(dir, "shard-0.dat")
				original := []byte("preserve this existing shard")
				if err := os.WriteFile(path, original, 0600); err != nil {
					t.Fatal(err)
				}
				err := NewWriter(dir).WriteShard(0, entries)
				if err == nil || !strings.Contains(err.Error(), "entry 1 key length") {
					t.Fatalf("unrepresentable key length %d: %v", len(tc.key), err)
				}
				got, err := os.ReadFile(path)
				if err != nil || !bytes.Equal(got, original) {
					t.Fatalf("rejected shard changed: bytes=%q err=%v", got, err)
				}
			})
			t.Run("do_not_create_directory", func(t *testing.T) {
				dir := filepath.Join(t.TempDir(), "not-created")
				if err := NewWriter(dir).WriteShard(0, entries); err == nil {
					t.Fatalf("accepted unrepresentable key length %d", len(tc.key))
				}
				if _, err := os.Stat(dir); !os.IsNotExist(err) {
					t.Fatalf("rejected shard created directory: %v", err)
				}
			})
		})
	}

	t.Run("valid_boundaries", func(t *testing.T) {
		dir := t.TempDir()
		entries := []KVEntry{
			{Key: []byte("k"), TTLMs: 0},
			{Key: bytes.Repeat([]byte{'z'}, 1<<16-1), Value: []byte("value"), TTLMs: 17},
		}
		if err := NewWriter(dir).WriteShard(0, entries); err != nil {
			t.Fatal(err)
		}
		got, err := NewReader(dir).ReadShard(0)
		if err != nil || len(got) != len(entries) {
			t.Fatalf("read valid boundaries: entries=%d err=%v", len(got), err)
		}
		for i, want := range entries {
			if !bytes.Equal(got[i].Key, want.Key) || !bytes.Equal(got[i].Value, want.Value) || got[i].TTLMs != want.TTLMs {
				t.Fatalf("entry %d changed: key length=%d value=%q ttl=%d", i, len(got[i].Key), got[i].Value, got[i].TTLMs)
			}
		}
	})
	t.Run("empty_shard", func(t *testing.T) {
		dir := t.TempDir()
		if err := NewWriter(dir).WriteShard(0, nil); err != nil {
			t.Fatal(err)
		}
		wire, err := os.ReadFile(filepath.Join(dir, "shard-0.dat"))
		if err != nil || !bytes.Equal(wire, []byte{0, 0}) {
			t.Fatalf("empty shard sentinel: bytes=%v err=%v", wire, err)
		}
		entries, err := NewReader(dir).ReadShard(0)
		if err != nil || len(entries) != 0 {
			t.Fatalf("empty shard read: entries=%d err=%v", len(entries), err)
		}
	})
}
