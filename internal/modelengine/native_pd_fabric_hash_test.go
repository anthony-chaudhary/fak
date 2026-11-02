package modelengine

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=50ms lane=default
func TestNativePDFabricDigestBatchedBytes(t *testing.T) {
	cfg := model.Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 3}
	for _, tc := range []struct {
		tokens int
		digest string
	}{
		{1, "cbc863ce2a0a7b31911e30636305019639437d1d91fb749d6cd1ade41cb7f991"},
		{171, "4379c9149c2553e94fe9b395eb043185eb7e6b2686bb2a3af2ad5617d3d9d8b4"},
	} {
		row := nativePDFabricHashRows(cfg, tc.tokens)
		want, wantBytes, err := nativePDFabricScalarDigestBaseline(cfg, tc.tokens, row)
		if err != nil || want != tc.digest {
			t.Fatalf("scalar framing oracle changed for %d tokens: digest=%s err=%v", tc.tokens, want, err)
		}
		got, gotBytes, err := nativePDFabricDigest(cfg, tc.tokens, row)
		if err != nil || got != want || gotBytes != wantBytes {
			t.Fatalf("batched witness changed for %d tokens: digest=%s bytes=%d err=%v", tc.tokens, got, gotBytes, err)
		}
	}
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		cfg    model.Config
		tokens int
	}{
		{cfg, 0},
		{cfg, -1},
		{model.Config{}, 1},
		{model.Config{NumLayers: maxInt, NumKVHeads: maxInt, HeadDim: maxInt}, maxInt},
	} {
		called := false
		row := func(int, int) []float32 { called = true; return nil }
		for _, digest := range []func(model.Config, int, func(int, int) []float32) (string, uint64, error){nativePDFabricScalarDigestBaseline, nativePDFabricDigest} {
			if got, n, err := digest(tc.cfg, tc.tokens, row); err == nil || got != "" || n != 0 || called {
				t.Fatal("invalid geometry/overflow changed its early refusal")
			}
		}
	}
	short := func(int, int) []float32 { return []float32{1} }
	if got, n, err := nativePDFabricDigest(cfg, 1, short); err == nil || got != "" || n != 0 {
		t.Fatal("incomplete plane was not refused")
	}
}

// Both arms hash the same immutable logical payload with the same canonical
// framing. This isolates witness CPU cost, not prefill or whole-Admit latency.
func BenchmarkNativePDFabricDigest(b *testing.B) {
	cfg := model.Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const tokens = 128
	row := nativePDFabricHashRows(cfg, tokens)
	for _, arm := range []struct {
		name string
		hash func(model.Config, int, func(int, int) []float32) (string, uint64, error)
	}{
		{"scalar-baseline", nativePDFabricScalarDigestBaseline},
		{"batched", nativePDFabricDigest},
	} {
		b.Run(arm.name, func(b *testing.B) {
			want, count, err := nativePDFabricScalarDigestBaseline(cfg, tokens, row)
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(count))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				got, n, err := arm.hash(cfg, tokens, row)
				if err != nil || got != want || n != count {
					b.Fatal("witness benchmark changed content or geometry")
				}
			}
		})
	}
}

func nativePDFabricHashRows(cfg model.Config, tokens int) func(int, int) []float32 {
	bits := []uint32{0x80000000, 0, 0x7fc00001, 0x7fc01234, 0x7f800000, 0xff800000, 0x3f800000, 0xbf800000, 1, 0x7f7fffff}
	rows := make([][][]float32, cfg.NumLayers)
	for l := range rows {
		rows[l] = make([][]float32, 3)
		for p := range rows[l] {
			rows[l][p] = make([]float32, tokens*cfg.NumKVHeads*cfg.HeadDim)
			for i := range rows[l][p] {
				rows[l][p][i] = math.Float32frombits(bits[(i+l*3+p)%len(bits)])
			}
		}
	}
	return func(layer, plane int) []float32 { return rows[layer][plane] }
}

// Scalar baseline is the unchanged nativePDFabricDigest at source-file SHA-256
// d36a3f6901b213cb3029910b917a18c086b90dfc28501c83bb5f01bb3b4c6f89.
// Keep its 4-byte writes intact so the benchmark compares the actual old loop.
func nativePDFabricScalarDigestBaseline(cfg model.Config, tokens int, row func(layer, plane int) []float32) (string, uint64, error) {
	if tokens <= 0 || cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return "", 0, errors.New("invalid native P/D fabric geometry")
	}
	count := uint64(tokens)
	for _, factor := range []int{cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, 3, 4} {
		if uint64(factor) > ^uint64(0)/count {
			return "", 0, errors.New("native P/D fabric logical size overflows")
		}
		count *= uint64(factor)
	}
	width := uint64(tokens) * uint64(cfg.NumKVHeads) * uint64(cfg.HeadDim)
	if width > uint64(^uint(0)>>1) {
		return "", 0, errors.New("native P/D fabric row extent overflows")
	}
	h := sha256.New()
	_, _ = h.Write([]byte("fak.native-pd.logical-dense-f32/v1\x00"))
	for _, value := range []uint64{uint64(tokens), uint64(cfg.NumLayers), uint64(cfg.NumKVHeads), uint64(cfg.HeadDim), 3} {
		nativePDFabricHashUint(h, value)
	}
	for position := 0; position < tokens; position++ {
		nativePDFabricHashUint(h, uint64(position))
	}
	var bits [4]byte
	for layer := 0; layer < cfg.NumLayers; layer++ {
		for plane := 0; plane < 3; plane++ {
			values := row(layer, plane)
			if uint64(len(values)) != width {
				return "", 0, errors.New("native P/D fabric has incomplete logical KV rows")
			}
			nativePDFabricHashUint(h, uint64(layer))
			nativePDFabricHashUint(h, uint64(plane))
			nativePDFabricHashUint(h, uint64(len(values)))
			for _, value := range values {
				binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
				_, _ = h.Write(bits[:])
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil)), count, nil
}
