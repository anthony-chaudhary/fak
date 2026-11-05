package dispatch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=5ms lane=default
// Estimate only: tiny temporary snapshots and unallocated manager shells.
func TestRestoreLoadsBeforeSubmittingFlush(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name     string
		manifest string
	}{
		{"missing_manifest", ""},
		{"invalid_manifest", "{"},
		{"negative_shard_count", `{"version":1,"shard_count":-1}`},
		{"missing_shard", `{"version":1,"shard_count":1}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.manifest != "" {
				if err := os.WriteFile(filepath.Join(dir, "snapshot.json"), []byte(tc.manifest), 0600); err != nil {
					t.Fatal(err)
				}
			}
			// This supported shell allocates no shard memory or goroutines. Its
			// nil shard slot makes any premature Submit fail immediately.
			mgr := shard.NewManagerAsync(shard.ManagerConfig{NumShards: 1})
			if mgr.NumShards() != 1 || mgr.Shard(0) != nil {
				t.Fatal("expected an unallocated one-shard shell")
			}
			d := &Dispatcher{Manager: mgr, SnapshotDir: dir}
			resp := d.HandleRestore(protocol.Message{Header: protocol.Header{OpCode: protocol.OpRestore, RequestID: 17}})
			if resp.Header.OpCode != protocol.RespError || resp.Header.RequestID != 17 || !strings.HasPrefix(string(resp.Body), "restore from "+dir+": ") {
				t.Fatalf("invalid snapshot response: header=%+v body=%q", resp.Header, resp.Body)
			}
		})
	}

	t.Run("empty_success", func(t *testing.T) {
		dir := t.TempDir()
		if err := snapshot.Save(dir, "test", "slab", nil); err != nil {
			t.Fatal(err)
		}
		// A zero-shard control exercises the existing success response without
		// claiming a successful data restore or allocating a serving manager.
		d := &Dispatcher{Manager: &shard.Manager{}, SnapshotDir: dir}
		resp := d.HandleRestore(protocol.Message{Header: protocol.Header{OpCode: protocol.OpRestore, RequestID: 18}})
		var result struct {
			Keys          int64  `json:"keys"`
			Dir           string `json:"dir"`
			ServerVersion string `json:"server_version"`
		}
		if err := json.Unmarshal(resp.Body, &result); err != nil {
			t.Fatal(err)
		}
		if resp.Header.OpCode != protocol.RespOK || resp.Header.RequestID != 18 || result.Keys != 0 || result.Dir != dir || result.ServerVersion != "test" {
			t.Fatalf("empty success response: header=%+v result=%+v", resp.Header, result)
		}
	})
}
