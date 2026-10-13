package model

import (
	"errors"
	"reflect"
	"testing"
)

// Borrowed rows must reseed the same mirror lifecycle as flat descriptors.
// This CPU-storage witness makes no physical-device performance claim.
// fak-test:runtime fast est=50ms lane=default
func TestV41BorrowedRowResidentLifecycle(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	for _, transition := range []string{"rollback", "owner-replacement", "ring-cache-replacement", "log-cache-replacement", "truncate-without-rollback"} {
		t.Run(transition, func(t *testing.T) {
			b := newV41ResidentKVTestBackend()
			s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
			selected := s.v41SharedAttentionFunc()
			if selected == nil {
				t.Fatal("shared attention was not bound")
			}
			owner := &V41AttentionState{}
			window := [][]float32{{0.25, -0.5}, {0.75, 0.125}}
			compressed := [][]float32{{-0.25, 0.5}, {0.125, 0.25}}
			invoke := func(pos int) ([]float32, error) {
				view := &v41ResidentKVView{owner: owner, pos: pos, ringRows: 2, firstKey: pos - len(window) + 1, windowRows: len(window), compressedRows: len(compressed)}
				ids := make([]int32, len(compressed))
				for i := range ids {
					ids[i] = int32(i)
				}
				return v41CombinedAttentionResident(0, pos, 1, 1, 2, []float32{0.5, -0.25}, nil, window, compressed, ids, 0.5, view,
					func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
						if request.kv != nil || request.rowKV == nil || &request.rowKV.window[0][0] != &window[0][0] || &request.rowKV.compressed[0][0] != &compressed[0][0] {
							t.Fatal("lifecycle transition did not borrow the source rows")
						}
						before := v41TestAttentionFlatKV(request)
						out, err := selected(layer, request)
						if !reflect.DeepEqual(v41TestAttentionFlatKV(request), before) {
							t.Fatal("lifecycle transition mutated borrowed rows")
						}
						return out, err
					})
			}
			if _, err := invoke(1); err != nil {
				t.Fatal(err)
			}
			ringKey, logKey := v41ResidentKVKey(0, "ring"), v41ResidentKVKey(0, "log")
			oldRing, oldLog := s.halW[ringKey], s.halW[logKey]
			if oldRing.Buf() == nil || oldLog.Buf() == nil || !b.allocations[oldRing.Buf()] || !b.allocations[oldLog.Buf()] {
				t.Fatal("seed did not register live ring and log cache entries")
			}
			pos := 2
			switch transition {
			case "rollback":
				pos = 0
				window = [][]float32{{-0.75, 0.5}}
				compressed = [][]float32{{0.875, -0.125}}
			case "owner-replacement":
				owner = &V41AttentionState{}
			case "ring-cache-replacement", "log-cache-replacement":
				key, old := ringKey, oldRing
				if transition == "log-cache-replacement" {
					key, old = logKey, oldLog
				}
				replacement := b.own(old.Shape, make([]float32, len(b.Backend.Read(old))))
				s.halW[key] = replacement
				b.Free(old)
				t.Cleanup(func() {
					if b.allocations[replacement.Buf()] {
						t.Error("replaced cache entry was not retired")
					}
				})
			case "truncate-without-rollback":
				window = [][]float32{window[1], {0.5, 0.75}}
				compressed = compressed[:1]
			}
			if transition != "rollback" && transition != "truncate-without-rollback" {
				// Changed retained values expose stale cache reuse after a reset.
				window = [][]float32{{-0.75, 0.5}, {0.5, 0.75}}
				compressed = [][]float32{{0.875, -0.125}, {-0.5, 0.625}}
			}
			got, err := invoke(pos)
			if transition == "truncate-without-rollback" {
				var closed *BackendForwardOperationError
				if !errors.Is(err, errV41ResidentKVDesync) || !errors.As(err, &closed) || closed.Stage != "resident KV" || !s.BackendSessionClosed() || len(b.calls) != 1 || len(b.allocations) != 0 {
					t.Fatalf("truncation did not refuse before dispatch and retire the session: %v", err)
				}
				if retry := recoverError(func() { _, _ = invoke(pos) }); retry != s.halFailure || len(b.calls) != 1 {
					t.Fatalf("closed session retried borrowed attention: %v", retry)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if s.halW[ringKey].Buf() == oldRing.Buf() || s.halW[logKey].Buf() == oldLog.Buf() || b.allocations[oldRing.Buf()] || b.allocations[oldLog.Buf()] {
				t.Fatal("reset retained stale mirror identity or leaked the old mirror")
			}
			var wantKV []float32
			for _, stream := range [][][]float32{window, compressed} {
				for _, row := range stream {
					wantKV = append(wantKV, row...)
				}
			}
			if len(b.calls) != 2 || !reflect.DeepEqual(b.calls[1].kv, wantKV) {
				t.Fatalf("reset dispatched stale borrowed rows: calls=%v want=%v", b.calls, wantKV)
			}
			want, err := v41SharedAttentionRoundedControl(b.calls[1])
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("reseed parity: got=%v want=%v err=%v", got, want, err)
			}
			s.Close()
			if len(b.allocations) != 0 {
				t.Fatal("closing the reseeded session leaked mirror storage")
			}
		})
	}
}
