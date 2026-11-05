package snapshot

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
// Estimate only: tiny temporary snapshots and each prefix of one shard file.
func TestLoadRequiresShardSentinel(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	want := []KVEntry{
		{Key: []byte("a"), Value: []byte("v"), TTLMs: 25},
		{Key: []byte("b"), Value: []byte{}, TTLMs: 0},
	}
	if err := Save(dir, "test", "slab", map[int][]KVEntry{0: want}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "shard-0.dat")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	_, entries, err := Load(dir, 1)
	if err != nil || !reflect.DeepEqual(entries, want) {
		t.Fatalf("complete shard: entries=%v err=%v, want %v", entries, err, want)
	}
	firstEnd := 2 + len(want[0].Key) + 4 + len(want[0].Value) + 8
	for cut := 0; cut < len(data); cut++ {
		t.Run(fmt.Sprintf("prefix_%d", cut), func(t *testing.T) {
			if err := os.WriteFile(path, data[:cut], 0600); err != nil {
				t.Fatal(err)
			}
			_, entries, err := Load(dir, 1)
			if err == nil || entries != nil {
				t.Fatalf("truncated shard: entries=%v err=%v, want refusal without partial entries", entries, err)
			}
			if (cut == 0 || cut == firstEnd || cut == len(data)-2) && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("missing sentinel at record boundary: err=%v, want unexpected EOF", err)
			}
		})
	}
	if err := Save(dir, "test", "slab", map[int][]KVEntry{0: nil}); err != nil {
		t.Fatal(err)
	}
	_, entries, err = Load(dir, 1)
	if err != nil || len(entries) != 0 {
		t.Fatalf("complete empty shard: entries=%v err=%v, want empty success", entries, err)
	}
}
