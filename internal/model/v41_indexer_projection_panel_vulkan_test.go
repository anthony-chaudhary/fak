//go:build vulkan && (windows || linux) && cgo

// FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41IndexerProjectionPanelVulkan$' -count=1
// Synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// On the device, an f32 indexer panel is bit-identical to its one-row
// projections at the V4.1 Flash indexer shapes, and a cold prefill session
// keeps its logits and selections while making ceil(rows/chunk) calls per leaf.
//
// fak-test:justify why=integration when=changed:internal/model/v41_indexer_projection*
// fak-test:runtime slow est=20s lane=optin
func TestV41IndexerProjectionPanelVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	t.Run("f32-rows", func(t *testing.T) {
		for _, shape := range []struct{ out, in int }{{8192, 1024}, {64, 4096}, {128, 1024}, {1, 1024}} {
			w := make([]float32, shape.out*shape.in)
			for i := range w {
				w[i] = float32((i*37)%101-50) / 211
			}
			weight := be.Upload(compute.NewF32(compute.Default(), []int{shape.out, shape.in}, w), compute.F32)
			for _, rows := range []int{v41ProjPanelRowChunk, 5} {
				x := make([]float32, rows*shape.in)
				for i := range x {
					x[i] = float32((i*53)%97-48) / 89
				}
				panel := uploadHostF32Class(be, []int{rows, shape.in}, x, compute.MemoryActivation, "indexer panel witness")
				y := be.BatchedMatMul(weight, panel, rows)
				got := append([]float32(nil), be.Read(y)...)
				be.Free(y)
				be.Free(panel)
				for row := 0; row < rows; row++ {
					one := uploadHostF32Class(be, []int{shape.in}, x[row*shape.in:(row+1)*shape.in], compute.MemoryActivation, "indexer row witness")
					y := be.MatMul(weight, one)
					want := append([]float32(nil), be.Read(y)...)
					be.Free(y)
					be.Free(one)
					for o := range want {
						if math.Float32bits(got[row*shape.out+o]) != math.Float32bits(want[o]) {
							t.Fatalf("out=%d in=%d rows=%d row=%d lane=%d panel %08x one-row %08x", shape.out, shape.in, rows, row, o, math.Float32bits(got[row*shape.out+o]), math.Float32bits(want[o]))
						}
					}
				}
			}
			be.Free(weight)
		}
	})
	t.Run("session", func(t *testing.T) {
		ids := v41ParityPrompt(v41IndexerTestFixture(t).Cfg.VocabSize, v41IndexerPanelTestRows)
		run := func(disabled bool) ([]float32, [][]int32, map[string]float64) {
			prior := v41IndexerPanelDisabled
			v41IndexerPanelDisabled = disabled
			defer func() { v41IndexerPanelDisabled = prior }()
			m := v41IndexerTestFixture(t)
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			before := v41IndexerTestPhase(t, m, "prefill")
			logits := append([]float32(nil), s.Prefill(ids)...)
			delta := v41DenseTestDelta(v41IndexerTestPhase(t, m, "prefill"), before)
			rows, ratio, ok := s.v41Forward.attn.TopK()
			if !ok || ratio != 2 || len(rows) != len(ids) {
				t.Fatalf("disabled=%t published %d TopK rows at ratio %d", disabled, len(rows), ratio)
			}
			var topK [][]int32
			for _, row := range rows {
				topK = append(topK, append([]int32(nil), row...))
			}
			if s.HostFallbackObserved() || s.BackendSessionClosed() {
				t.Fatalf("disabled=%t session fell back or closed", disabled)
			}
			s.Close()
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
			return logits, topK, delta
		}
		rowLogits, rowTopK, rowDelta := run(true)
		panelLogits, panelTopK, panelDelta := run(false)
		v41IndexerPanelTestSameLogits(t, panelLogits, rowLogits)
		if !reflect.DeepEqual(panelTopK, rowTopK) {
			t.Fatal("device panel indexer selections differ from the per-row path")
		}
		n, chunks := len(ids), v41IndexerPanelTestChunks(len(ids))
		if rowDelta["indexer_projection_device_calls"] <= 0 || panelDelta["indexer_projection_device_rows"] != rowDelta["indexer_projection_device_rows"] {
			t.Fatalf("device accounting panel %v per-row %v", panelDelta, rowDelta)
		}
		if saved := rowDelta["indexer_projection_device_calls"] - panelDelta["indexer_projection_device_calls"]; saved != float64(2*(n-chunks)) {
			t.Fatalf("device panel saved %v projection calls, want %d", saved, 2*(n-chunks))
		}
		t.Logf("V41_INDEXER_PANEL_VULKAN rows=%d device_calls per-row=%v panel=%v logits_bit_identical=true selections_equal=true", n, rowDelta["indexer_projection_device_calls"], panelDelta["indexer_projection_device_calls"])
	})
}
