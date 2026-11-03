package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"testing"
)

// Fixtures fill all physical slots, including unoccupied and reserved tail
// blocks. Comparisons use raw bits, not floating-point equality.
func pagedSwapCodecFixture(cfg Config, raw bool, blockTokens, tokens, blocks int) *PagedKV {
	pool := NewPagedKVPool(cfg, blockTokens)
	if raw {
		pool = NewPagedKVPoolWithRaw(cfg, blockTokens)
	}
	seq := pool.NewSequence()
	seq.nTokens = tokens
	pattern := [...]uint32{0, 0x80000000, 0x7fc01234, 0x7fa00001, 0x7f800000, 0xff800000, 1, 0x3f800000}
	for i := 0; i < blocks; i++ {
		id := pool.alloc()
		seq.table = append(seq.table, id)
		for j := range pool.blocks[id] {
			pool.blocks[id][j] = math.Float32frombits(pattern[(i+j)%len(pattern)])
		}
	}
	return seq
}

func TestPagedKVHostCodecBytes(t *testing.T) {
	cfg := Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 3}
	for _, raw := range []bool{false, true} {
		for _, tc := range []struct {
			name           string
			cfg            Config
			tokens, blocks int
		}{
			{"empty", cfg, 0, 0},
			{"partial-and-reserved-tail", cfg, 5, 3},
			{"zero-width-reserved", Config{}, 0, 2},
		} {
			t.Run(fmt.Sprintf("%s/raw=%v", tc.name, raw), func(t *testing.T) {
				seq := pagedSwapCodecFixture(tc.cfg, raw, 4, tc.tokens, tc.blocks)
				want, err := seq.swapToHostScalarBaseline()
				if err != nil {
					t.Fatal(err)
				}
				got, err := seq.SwapToHost()
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Fatal("FAKPKV1 bytes differ from pinned scalar codec")
				}
				seq.Free()
				for _, block := range seq.pool.blocks {
					clear(block)
				}
				if !bytes.Equal(got, want) {
					t.Fatal("serialized blob aliases source pages")
				}

				foreign := pagedSwapCodecFixture(tc.cfg, raw, 4, 0, 1)
				pool := foreign.pool
				foreignBefore, err := foreign.SwapToHost()
				if err != nil {
					t.Fatal(err)
				}
				restored, err := pool.RestoreFromHost(got)
				if err != nil {
					t.Fatal(err)
				}
				if restored.Len() != tc.tokens || restored.Blocks() != tc.blocks {
					t.Fatalf("Len/Blocks=%d/%d, want %d/%d", restored.Len(), restored.Blocks(), tc.tokens, tc.blocks)
				}
				clear(got)
				back, err := restored.SwapToHost()
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(back, want) {
					t.Fatal("restored pages changed bits or alias input blob")
				}
				restored.Free()
				restored.Free()
				foreignAfter, err := foreign.SwapToHost()
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(foreignAfter, foreignBefore) || pool.PhysicalBlocks() != 1 || pool.MisuseReleases() != 0 {
					t.Fatal("restore/free altered unrelated ownership")
				}
				foreign.Free()
			})
		}
	}
}

func TestPagedKVHostCodecRefusals(t *testing.T) {
	cfg := Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 3}
	src := pagedSwapCodecFixture(cfg, true, 4, 5, 3)
	defer src.Free()
	blob, err := src.swapToHostScalarBaseline()
	if err != nil {
		t.Fatal(err)
	}
	cases := make(map[string][]byte)
	for _, n := range []int{0, 1, 7, 8, 9, 31, 32, 33, 32 + src.pool.blockFloats()*4, len(blob) - 4, len(blob) - 3, len(blob) - 1} {
		cases[fmt.Sprintf("truncated-%d", n)] = append([]byte(nil), blob[:n]...)
	}
	cases["trailing"] = append(append([]byte(nil), blob...), 1)
	cases["magic"] = append([]byte(nil), blob...)
	cases["magic"][0] ^= 1
	for name, field := range map[string]int{"geometry": 1, "tokens-without-blocks": 5, "too-many-tokens": 4} {
		b := append([]byte(nil), blob...)
		value := uint32(0)
		if name == "geometry" {
			value = 99
		}
		if name == "too-many-tokens" {
			value = 13
		}
		binary.LittleEndian.PutUint32(b[len(pagedKVSwapMagic)+field*4:], value)
		cases[name] = b
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			var wantError string
			for arm := 0; arm < 2; arm++ {
				foreign := pagedSwapCodecFixture(cfg, true, 4, 1, 1)
				pool := foreign.pool
				before, _ := foreign.SwapToHost()
				var restored *PagedKV
				var err error
				if arm == 0 {
					restored, err = pool.restoreFromHostScalarBaseline(input)
				} else {
					restored, err = pool.RestoreFromHost(input)
				}
				if err == nil || restored != nil {
					t.Fatal("malformed blob was accepted")
				}
				if arm == 0 {
					wantError = err.Error()
				} else if err.Error() != wantError {
					t.Fatalf("error changed: got %q want %q", err, wantError)
				}
				after, afterErr := foreign.SwapToHost()
				if afterErr != nil || !bytes.Equal(before, after) || pool.PhysicalBlocks() != 1 || pool.MisuseReleases() != 0 {
					t.Fatal("failed restore leaked pages or changed unrelated sequence")
				}
				foreign.Free()
			}
		})
	}
	for _, id := range []int{-1, len(src.pool.blocks)} {
		bad := &PagedKV{pool: src.pool, table: []int{id}}
		_, oldErr := bad.swapToHostScalarBaseline()
		_, newErr := bad.SwapToHost()
		if oldErr == nil || newErr == nil || oldErr.Error() != newErr.Error() {
			t.Fatalf("invalid block %d error changed", id)
		}
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := pagedKVSwapGrowSize(maxInt-3, 1); err == nil {
		t.Fatal("output size overflow accepted")
	}
	if n, err := pagedKVSwapGrowSize(32, 0); err != nil || n != 32 {
		t.Fatal("zero payload size changed")
	}
	if n, err := pagedKVSwapGrowSize(maxInt-4, 1); err != nil || n != maxInt {
		t.Fatal("addressable size boundary changed")
	}
}

var pagedSwapCodecBlobSink []byte
var pagedSwapCodecCacheSink *KVCache

// Serialization excludes fixture construction. Restore includes the ordinary
// dense path's fresh pool, owned page restore, contiguous materialization and
// page release. The arms use identical bytes; this is not whole-Engine latency.
func BenchmarkPagedKVHostCodec(b *testing.B) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	src := pagedSwapCodecFixture(cfg, true, 16, 128, 8)
	defer src.Free()
	blob, err := src.swapToHostScalarBaseline()
	if err != nil {
		b.Fatal(err)
	}
	for _, arm := range []struct {
		name string
		bulk bool
	}{{"scalar-baseline", false}, {"bulk", true}} {
		b.Run("serialize/"+arm.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(blob)))
			for i := 0; i < b.N; i++ {
				var out []byte
				var err error
				if arm.bulk {
					out, err = src.SwapToHost()
				} else {
					out, err = src.swapToHostScalarBaseline()
				}
				if err != nil {
					b.Fatal(err)
				}
				pagedSwapCodecBlobSink = out
			}
			b.StopTimer()
			if !bytes.Equal(pagedSwapCodecBlobSink, blob) {
				b.Fatal("serialized bytes changed")
			}
		})
		b.Run("restore-fresh-materialize/"+arm.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(blob)))
			for i := 0; i < b.N; i++ {
				pool := NewPagedKVPoolWithRaw(cfg, 16)
				var seq *PagedKV
				var err error
				if arm.bulk {
					seq, err = pool.RestoreFromHost(blob)
				} else {
					seq, err = pool.restoreFromHostScalarBaseline(blob)
				}
				if err != nil {
					b.Fatal(err)
				}
				pagedSwapCodecCacheSink = seq.ToKVCache(cfg)
				seq.Free()
			}
			b.StopTimer()
			if pagedSwapCodecCacheSink.Len() != 128 {
				b.Fatal("restored token count changed")
			}
		})
	}
}

// Baseline below is the pinned old implementation, unchanged except the two
// method names. It supplies byte/error equivalence and matched benchmark arms.
func (s *PagedKV) swapToHostScalarBaseline() ([]byte, error) {
	if s == nil || s.pool == nil {
		return nil, errors.New("model: cannot swap nil PagedKV")
	}
	p := s.pool
	var b bytes.Buffer
	b.WriteString(pagedKVSwapMagic)
	for _, n := range []int{p.blockTokens, p.stride, p.nLayers, p.planes, s.nTokens, len(s.table)} {
		if n < 0 {
			return nil, fmt.Errorf("model: cannot swap PagedKV with negative geometry %d", n)
		}
		if err := binary.Write(&b, binary.LittleEndian, uint32(n)); err != nil {
			return nil, err
		}
	}
	for _, id := range s.table {
		if id < 0 || id >= len(p.blocks) || p.ref[id] == 0 {
			return nil, fmt.Errorf("model: cannot swap PagedKV with invalid block id %d", id)
		}
		for _, f := range p.blocks[id] {
			if err := binary.Write(&b, binary.LittleEndian, math.Float32bits(f)); err != nil {
				return nil, err
			}
		}
	}
	return b.Bytes(), nil
}

// RestoreFromHost restores a SwapToHost blob into fresh owned pages in this pool. The pool
// must have the same page geometry and plane count as the source; that is the structural
// gate that keeps swap tied to the paged/block allocator rather than the contiguous cache.
func (p *PagedKVPool) restoreFromHostScalarBaseline(data []byte) (*PagedKV, error) {
	if p == nil {
		return nil, errors.New("model: cannot restore PagedKV into nil pool")
	}
	r := bytes.NewReader(data)
	magic := make([]byte, len(pagedKVSwapMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("model: restore paged KV header: %w", err)
	}
	if string(magic) != pagedKVSwapMagic {
		return nil, errors.New("model: invalid paged KV swap blob")
	}
	readU32 := func(name string) (int, error) {
		var v uint32
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return 0, fmt.Errorf("model: restore paged KV %s: %w", name, err)
		}
		return int(v), nil
	}
	blockTokens, err := readU32("blockTokens")
	if err != nil {
		return nil, err
	}
	stride, err := readU32("stride")
	if err != nil {
		return nil, err
	}
	nLayers, err := readU32("layers")
	if err != nil {
		return nil, err
	}
	planes, err := readU32("planes")
	if err != nil {
		return nil, err
	}
	nTokens, err := readU32("tokens")
	if err != nil {
		return nil, err
	}
	nBlocks, err := readU32("blocks")
	if err != nil {
		return nil, err
	}
	if blockTokens != p.blockTokens || stride != p.stride || nLayers != p.nLayers || planes != p.planes {
		return nil, fmt.Errorf("model: paged KV swap geometry mismatch: blob block=%d stride=%d layers=%d planes=%d, pool block=%d stride=%d layers=%d planes=%d",
			blockTokens, stride, nLayers, planes, p.blockTokens, p.stride, p.nLayers, p.planes)
	}
	if nBlocks == 0 && nTokens != 0 {
		return nil, errors.New("model: paged KV swap blob has tokens but no blocks")
	}
	if nBlocks > 0 && nTokens > nBlocks*p.blockTokens {
		return nil, errors.New("model: paged KV swap blob token count exceeds block table")
	}

	// Keep only owned block IDs in the table so failure cannot release another sequence's pages.
	seq := &PagedKV{pool: p, table: make([]int, 0, nBlocks), nTokens: nTokens}
	fail := func(err error) (*PagedKV, error) {
		seq.Free()
		return nil, err
	}
	for i := 0; i < nBlocks; i++ {
		id := p.alloc()
		seq.table = append(seq.table, id)
		blk := p.blocks[id]
		if len(blk) != p.blockFloats() {
			return fail(errors.New("model: paged KV pool block geometry changed under restore"))
		}
		for j := range blk {
			var bits uint32
			if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
				return fail(fmt.Errorf("model: restore paged KV block %d: %w", i, err))
			}
			blk[j] = math.Float32frombits(bits)
		}
	}
	if r.Len() != 0 {
		return fail(errors.New("model: paged KV swap blob has trailing bytes"))
	}
	return seq, nil
}
