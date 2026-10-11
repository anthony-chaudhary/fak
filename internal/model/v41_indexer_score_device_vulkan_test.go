//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41VulkanSeamBackend(t *testing.T) compute.Backend {
	t.Helper()
	be := compute.Pick("vulkan")
	if be.Name() != "vulkan" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device not registered")
		}
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	return be
}

// fak-test:runtime integration est=5s lane=optin
// Ref #13668. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// witness. The registered Vulkan backend binds the compressor and index-key norm
// seams through its exact RMSNorm, and every selected row is bit-identical to
// the host tail it replaces (no tolerance) at the V4.1 widths 512 and 128.
func TestV41NormSeamsVulkanBitExact(t *testing.T) {
	be := v41VulkanSeamBackend(t)
	const eps = float32(1e-6)
	rng := rand.New(rand.NewSource(13668))
	row := func(width int, scale float64) []float32 {
		out := make([]float32, width)
		for i := range out {
			out[i] = float32(rng.NormFloat64() * scale)
		}
		return out
	}
	for _, width := range []int{512, 128} {
		gain := row(width, 0.3)
		for i := range gain {
			gain[i] = v41RoundBF16(gain[i] + 1)
		}
		compressor := &Session{M: v41CompressorNormTinyModel(t, gain, eps), Backend: be, halW: map[string]compute.Tensor{}}
		t.Cleanup(compressor.Close)
		index := &Session{M: v41IndexKeyNormTinyModel(t, gain, eps), Backend: be, halW: map[string]compute.Tensor{}}
		t.Cleanup(index.Close)
		compress, key := compressor.v41CompressorNormFunc(), index.v41IndexKeyNormFunc()
		if width == 512 && compress == nil {
			t.Fatal("Vulkan left the compressor norm seam unbound")
		}
		if width == 128 && key == nil {
			t.Fatal("Vulkan left the index-key norm seam unbound")
		}
		for r := 0; r < 256; r++ {
			x := row(width, math.Ldexp(1, rng.Intn(80)-40))
			if compress != nil {
				got, err := compress(0, x, gain, eps)
				want := v41CompressorNormRefTail(x, gain, eps, "")
				if err != nil || !v41CompressorNormEqualBits(got, want) {
					t.Fatalf("width %d row %d: device compressor norm differs from the host tail (err=%v)", width, r, err)
				}
			}
			if key != nil {
				got, err := key(0, x, gain, eps)
				want := v41IndexKeyNormReference(x, gain, eps)
				if err != nil || !v41CompressorNormEqualBits(got, want) {
					t.Fatalf("width %d row %d: device index-key norm differs from the host RMSNorm (err=%v)", width, r, err)
				}
			}
		}
	}
}

// v41IndexerSelectionFixture draws BF16 queries and weights like the indexer
// projection publishes, keys with duplicated rows (score ties broken by
// position), and a band of keys opposite the query so many heads rectify to 0.
func v41IndexerSelectionFixture(rng *rand.Rand, heads, dim, rows int) (q, keys, weights []float32) {
	q = make([]float32, heads*dim)
	for i := range q {
		q[i] = v41RoundBF16(float32(rng.NormFloat64()))
	}
	keys = make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		key := keys[r*dim : (r+1)*dim]
		switch {
		case r%17 == 5 && r >= 17:
			copy(key, keys[(r-17)*dim:(r-16)*dim])
		case r%29 == 3:
			for d := range key {
				key[d] = -q[d] * 0.5
			}
		default:
			for d := range key {
				key[d] = float32(rng.NormFloat64() * 0.125)
			}
		}
	}
	weights = make([]float32, heads)
	for h := range weights {
		weights[h] = v41RoundBF16(float32(rng.NormFloat64()) / float32(math.Sqrt(float64(dim*heads))))
	}
	return q, keys, weights
}

// fak-test:runtime integration est=10s lane=optin
// Ref #13668. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// witness. Through the session seams the forward binds, device score feeding
// device index select publishes exactly the index rows host score feeding host
// selection publishes, at the V4.1 indexer geometry (64 heads x 128) including
// the candidate-block layer (2048 blocks of 8). Scores are compared bitwise
// first, so a selection match cannot hide a score drift.
func TestV41IndexerSelectionVulkanParity(t *testing.T) {
	be := v41VulkanSeamBackend(t)
	s := &Session{M: &Model{}, Backend: be}
	t.Cleanup(s.Close)
	score, health := s.v41IndexerScoreFunc(), s.v41IndexerScoreHealthFunc()
	sel := s.v41DeviceGatherOpsFunc().indexSelectFunc()
	if score == nil || health == nil {
		reason := ""
		if op, ok := be.(compute.V41IndexerScoreBackend); ok {
			reason = op.V41IndexerScoreUnavailableReason()
		}
		t.Fatalf("Vulkan left the indexer score seam unbound: %s", reason)
	}
	if sel == nil {
		t.Fatal("Vulkan left the index-select seam unbound")
	}
	const heads, dim = 64, 128
	for _, tc := range []struct{ rows, topK, topKBlocks, blockSize int }{
		{1, 512, 0, 0}, {7, 512, 0, 0}, {600, 512, 0, 0}, {4096, 512, 0, 0},
		{4096, 2048, 2048, 8}, {20000, 2048, 2048, 8}, {300, 64, 16, 8},
	} {
		t.Run(fmt.Sprintf("rows%d_topk%d_blocks%d", tc.rows, tc.topK, tc.topKBlocks), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(13668 + tc.rows + tc.topK)))
			q, keys, weights := v41IndexerSelectionFixture(rng, heads, dim, tc.rows)
			hostScores, err := V41IndexerScore(q, keys, weights, heads, dim, tc.rows)
			if err != nil {
				t.Fatal(err)
			}
			deviceScores, err := score(0, q, keys, weights, heads, dim, tc.rows)
			if err != nil {
				t.Fatal(err)
			}
			for i := range hostScores {
				if math.Float32bits(deviceScores[i]) != math.Float32bits(hostScores[i]) {
					t.Fatalf("row %d: device score %#08x, host %#08x", i, math.Float32bits(deviceScores[i]), math.Float32bits(hostScores[i]))
				}
			}
			want, err := v41IndexRowsSelected(0, q, keys, weights, heads, dim, tc.rows, tc.topKBlocks, tc.blockSize, tc.topK, nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := v41IndexRowsSelected(0, q, keys, weights, heads, dim, tc.rows, tc.topKBlocks, tc.blockSize, tc.topK, score, health, sel)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want) {
				first := 0
				for first < min(len(got), len(want)) && got[first] == want[first] {
					first++
				}
				t.Fatalf("device score -> device select differs from host at position %d of %d/%d", first, len(got), len(want))
			}
			if s.BackendSessionClosed() {
				t.Fatal("selection parity closed the session")
			}
		})
	}
}
