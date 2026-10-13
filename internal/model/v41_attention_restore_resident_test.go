package model

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// An equal position does not prove equal KV contents. Restore must replace the
// host owner and reseed the target's device banks from the snapshot, while the
// same model/backend continues borrowing its immutable sink. This witness uses
// actual public capture/restore/Step entry points and CPU-storage accounting.
// It does not qualify another backend identity or make a performance claim.
// fak-test:runtime medium est=5s lane=default
func TestV41SameLengthPrefixRestoreResident(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("restore witness requires the qualified cpu-ref identity")
	}
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("compressed_source_reader_%t", compressed), func(t *testing.T) {
			var m *Model
			if compressed {
				m = v41ReaderWidthFixture(t)
			} else {
				m = v41IncrementalPlainModel(t, 1)
			}
			m.Cfg.Window = make([]int, m.Cfg.NumLayers)
			for layer := range m.Cfg.Window {
				m.Cfg.Window[layer] = 3
			}
			b := newV41ResidentKVTestBackend()
			source, target := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
			v41SharedAttentionOnly(source)
			v41SharedAttentionOnly(target)
			const prefixLen = 4
			prefix, next := v41IncrementalPrefix(m.Cfg, prefixLen)
			other := append([]int(nil), prefix...)
			for i := range other {
				other[i] = (other[i] + 1) % m.Cfg.VocabSize
			}
			source.Prefill(prefix)
			target.Prefill(other)
			wantPrefix := captureV41ForwardSnapshot(source.v41Forward)
			oldOwners := append([]*V41AttentionState(nil), target.v41Forward.layers...)
			rowsDiffer := false
			for layer, owner := range oldOwners {
				if !reflect.DeepEqual(owner.window, wantPrefix.layers[layer].window) ||
					!reflect.DeepEqual(owner.kvPublications, wantPrefix.layers[layer].kvPublications) {
					rowsDiffer = true
				}
			}
			if !rowsDiffer || len(target.v41Forward.history) != prefixLen || reflect.DeepEqual(target.v41Forward.history, prefix) {
				t.Fatal("fixture lacks different authoritative rows at the same committed position")
			}
			oldRing, oldLog, sinks := make([]compute.Tensor, m.Cfg.NumLayers), make([]compute.Tensor, m.Cfg.NumLayers), make([]compute.Tensor, m.Cfg.NumLayers)
			expectedLive := map[compute.Buffer]bool{}
			for layer := range m.Cfg.NumLayers {
				oldRing[layer] = target.halW[v41ResidentKVKey(layer, "ring")]
				oldLog[layer] = target.halW[v41ResidentKVKey(layer, "log")]
				key := "v41-attn-sink:" + layerName(layer, "attn.sink")
				sinks[layer] = target.halW[key]
				_, borrowed := target.borrowedHALW[key]
				if !b.allocations[oldRing[layer].Buf()] || !b.allocations[sinks[layer].Buf()] || !borrowed ||
					source.halW[key].Buf() != sinks[layer].Buf() || (compressed && !b.allocations[oldLog[layer].Buf()]) {
					t.Fatal("warm target lacks owned banks or the exact shared model-owned sink")
				}
				expectedLive[oldRing[layer].Buf()], expectedLive[sinks[layer].Buf()] = true, true
				if compressed {
					expectedLive[oldLog[layer].Buf()] = true
				}
			}
			snapshot, err := source.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snapshot.Close()
			beforeBytes, beforeCalls, beforeGathers := b.uploadBytes, len(b.calls), b.gathers
			if err := snapshot.Restore(target); err != nil {
				t.Fatal(err)
			}
			if snapshot.v41 != nil || snapshot.Cache != nil || target.v41Forward.callbackOwner != target ||
				!reflect.DeepEqual(captureV41ForwardSnapshot(target.v41Forward), wantPrefix) ||
				b.uploadBytes != beforeBytes || len(b.calls) != beforeCalls || b.gathers != beforeGathers {
				t.Fatal("restore lost snapshot ownership/state or performed attention device work")
			}
			for layer, owner := range target.v41Forward.layers {
				if owner == oldOwners[layer] || owner == source.v41Forward.layers[layer] ||
					target.halW[v41ResidentKVKey(layer, "ring")].Buf() != oldRing[layer].Buf() ||
					(compressed && target.halW[v41ResidentKVKey(layer, "log")].Buf() != oldLog[layer].Buf()) {
					t.Fatal("restore reused a host owner or prematurely changed the pending device bank")
				}
			}
			v41SharedAttentionOnly(target)
			type operand struct {
				layer, pos, window, groups int
				q, kv, sink, output        []float32
				idx                        []int32
				opt                        V41SparseAttentionSinkOptions
				call                       v41SharedAttentionTestCall
			}
			var authoritative []operand
			selected := source.v41Forward.sharedAttention
			source.v41Forward.sharedAttention = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
				view := request.resident
				if view == nil {
					t.Fatal("reference source lost residency")
				}
				a := operand{layer: layer, pos: view.pos, window: view.windowRows, groups: view.compressedRows,
					q: append([]float32(nil), request.q...), kv: v41TestAttentionFlatKV(request), sink: append([]float32(nil), request.sink...),
					idx: append([]int32(nil), request.idx...), opt: request.plain}
				out, err := selected(layer, request)
				if err == nil {
					a.output, a.call = append([]float32(nil), out...), b.calls[len(b.calls)-1]
					authoritative = append(authoritative, a)
				}
				return out, err
			}
			tokens := []int{next, (next + 1) % m.Cfg.VocabSize}
			var logits [][]float32
			var states []*v41ForwardSnapshot
			for _, token := range tokens {
				logits = append(logits, append([]float32(nil), source.Step(token)...))
				states = append(states, captureV41ForwardSnapshot(source.v41Forward))
			}
			if len(authoritative) != 2*m.Cfg.NumLayers {
				t.Fatal("reference did not produce every layer for both steps")
			}
			source.Close()
			if !reflect.DeepEqual(b.allocations, expectedLive) {
				t.Fatal("source close affected target banks/model sinks or retained source-owned buffers")
			}
			previousGroups := make([]int, m.Cfg.NumLayers)
			newRing, newLog := make([]compute.Tensor, m.Cfg.NumLayers), make([]compute.Tensor, m.Cfg.NumLayers)
			restoredOwners := append([]*V41AttentionState(nil), target.v41Forward.layers...)
			selected = target.v41Forward.sharedAttention
			at := 0
			target.v41Forward.sharedAttention = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
				if at >= len(authoritative) {
					t.Fatal("restored callback dispatched an extra layer")
				}
				want := authoritative[at]
				view := request.resident
				if view == nil || view.owner != restoredOwners[layer] || view.owner == oldOwners[layer] ||
					layer != want.layer || view.pos != want.pos || view.windowRows != want.window || view.compressedRows != want.groups ||
					!reflect.DeepEqual(request.idx, want.idx) || !reflect.DeepEqual(request.plain, want.opt) {
					t.Fatal("restored callback used an old owner, wrong position, or altered selection")
				}
				v41SharedAttentionSessionBits(t, "authoritative query", request.q, want.q)
				v41SharedAttentionSessionBits(t, "authoritative complete KV", v41TestAttentionFlatKV(request), want.kv)
				v41SharedAttentionSessionBits(t, "authoritative sink", request.sink, want.sink)
				first := view.pos == prefixLen
				if compressed && (want.groups != (view.pos+1)/2 || (!first && want.groups != previousGroups[layer]+1)) {
					t.Fatal("ratio-2 source and reader must retain two groups, then append one new group on the second step")
				}
				rows := 1 + want.groups - previousGroups[layer]
				if first {
					rows = want.window + want.groups
				}
				beforeBytes, beforeGathers, beforeCalls := b.uploadBytes, b.gathers, len(b.calls)
				out, err := selected(layer, request)
				if err != nil {
					return nil, err
				}
				delta := b.uploadBytes - beforeBytes
				wantBytes := 4 * (len(request.q) + rows*m.Cfg.HeadDim) // sink already belongs to this model/backend.
				if delta != wantBytes || b.gathers != beforeGathers+1 || len(b.calls) != beforeCalls+1 ||
					!reflect.DeepEqual(b.calls[len(b.calls)-1], want.call) {
					t.Fatalf("restored layer=%d pos=%d rows=%d bytes=%d want=%d; wrong operand/transfer sequence", layer, view.pos, rows, delta, wantBytes)
				}
				v41SharedAttentionSessionBits(t, "authoritative attention output", out, want.output)
				ring, log := target.halW[v41ResidentKVKey(layer, "ring")], target.halW[v41ResidentKVKey(layer, "log")]
				if first {
					if ring.Buf() == oldRing[layer].Buf() || b.allocations[oldRing[layer].Buf()] ||
						(compressed && (log.Buf() == oldLog[layer].Buf() || b.allocations[oldLog[layer].Buf()])) {
						t.Fatal("first restored step retained old bank identity or failed retirement")
					}
					newRing[layer], newLog[layer] = ring, log
				} else if ring.Buf() != newRing[layer].Buf() || (compressed && log.Buf() != newLog[layer].Buf()) {
					t.Fatal("second step rebuilt a correctly seeded bank")
				}
				key := "v41-attn-sink:" + layerName(layer, "attn.sink")
				_, borrowed := target.borrowedHALW[key]
				if target.halW[key].Buf() != sinks[layer].Buf() || !borrowed || !b.allocations[sinks[layer].Buf()] {
					t.Fatal("restore/Step replaced or retired the borrowed model sink")
				}
				previousGroups[layer] = want.groups
				at++
				t.Logf("restored layer=%d pos=%d first=%t actual KV rows=%d attention H2D bytes=%d gathers=1; sink identity retained", layer, view.pos, first, (delta-4*len(request.q))/(4*m.Cfg.HeadDim), delta)
				return out, nil
			}
			for step, token := range tokens {
				got := target.Step(token)
				v41SharedAttentionSessionBits(t, "restored full-vocabulary logits", got, logits[step])
				if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41Forward), states[step]) {
					t.Fatal("restored continuation differs from authoritative source branch")
				}
			}
			if at != len(authoritative) {
				t.Fatal("restored continuation skipped an attention call")
			}
			v41ColdAttentionClose(t, target, b.v41SharedAttentionTestBackend)
		})
	}
}
