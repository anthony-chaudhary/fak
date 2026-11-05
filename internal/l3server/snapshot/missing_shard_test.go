package snapshot

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Estimate only: three tiny temporary snapshots, without a server or shards.
func TestLoadRequiresEveryDeclaredShard(t *testing.T) {
	t.Parallel()
	for _, missing := range []int{-1, 0, 1} {
		t.Run(fmt.Sprintf("missing_%d", missing), func(t *testing.T) {
			dir := t.TempDir()
			want := []KVEntry{{Key: []byte("k"), Value: []byte("v"), TTLMs: 25}}
			if err := Save(dir, "test", "slab", map[int][]KVEntry{0: want, 1: {}}); err != nil {
				t.Fatal(err)
			}
			if missing >= 0 {
				if err := os.Remove(filepath.Join(dir, fmt.Sprintf("shard-%d.dat", missing))); err != nil {
					t.Fatal(err)
				}
			}
			// A different current layout still requires every original shard.
			manifest, entries, err := Load(dir, 1)
			if manifest.ShardCount != 2 {
				t.Fatalf("manifest shard count = %d, want 2", manifest.ShardCount)
			}
			if missing >= 0 {
				if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), fmt.Sprintf("read shard %d:", missing)) || entries != nil {
					t.Fatalf("missing shard %d: entries=%v err=%v, want contextual missing-file refusal without partial entries", missing, entries, err)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(entries, want) {
				t.Fatalf("complete snapshot with empty shard: entries=%v err=%v, want %v", entries, err, want)
			}
		})
	}
}
