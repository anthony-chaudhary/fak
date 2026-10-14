package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type ownerRangeReadBand struct {
	start, end, lo, hi, per int64
}

// This sits behind the parsed directory: header prefetch is outside the payload
// witness. Even one full-slab read is refused, not merely counted after loading.
type ownerRangeReader struct {
	io.ReaderAt
	bands           []ownerRangeReadBand
	mu              sync.Mutex
	readBytes       int64
	reads, rejected int
}

func (r *ownerRangeReader) ReadAt(p []byte, off int64) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	end := off + int64(len(p))
	for _, b := range r.bands {
		if off >= b.end || end <= b.start {
			continue
		}
		if off < b.lo || end > b.hi || int64(len(p)) != b.per || (off-b.start)%b.per != 0 {
			r.rejected++
			return 0, fmt.Errorf("nonowned or nonsingle-expert read [%d,%d), owner [%d,%d)", off, end, b.lo, b.hi)
		}
		n, err := r.ReaderAt.ReadAt(p, off)
		r.readBytes += int64(n)
		r.reads++
		return n, err
	}
	return r.ReaderAt.ReadAt(p, off)
}

func ownerRangeBand(info TensorInfo, band ExpertShard) ownerRangeReadBand {
	full := tensorOnDiskBytes(info)
	per := full / int64(info.Dims[2])
	return ownerRangeReadBand{info.FileOffset, info.FileOffset + full,
		info.FileOffset + int64(band.Lo)*per, info.FileOffset + int64(band.Hi)*per, per}
}

// The mixed production-loader witness is host byte ownership only. It does not
// qualify device residency, the EP serve fence, or throughput on physical GPUs.
// fak-test:runtime fast est=1s lane=default
func TestDeepSeek41OwnedExpertRangeLoader(t *testing.T) {
	path := writeDeepSeek41ExpertHALFile(t)
	oracle, err := LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = oracle.CloseWeights() })
	const experts = model.V41RouterExperts
	if oracle.KQuantCount() != 3*experts {
		t.Fatalf("full-load oracle has %d raw expert projections, want %d", oracle.KQuantCount(), 3*experts)
	}
	for rank := range 3 {
		t.Run(fmt.Sprintf("rank%d", rank), func(t *testing.T) {
			band, err := ExpertShardForRank(experts, 3, rank)
			if err != nil {
				t.Fatal(err)
			}
			ws, err := OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			defer ws.Close()
			r := &ownerRangeReader{ReaderAt: ws.r}
			var wantBytes int64
			for _, info := range ws.File.Tensors {
				if _, _, ok := glmMoeDsaBatchedExpert(info.Name); ok {
					b := ownerRangeBand(info, band)
					r.bands = append(r.bands, b)
					wantBytes += b.hi - b.lo
				}
			}
			if len(r.bands) != 3 {
				t.Fatal("fixture must carry gate/up Q2_K and down Q3_K")
			}
			ws.r = r
			m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, WithExpertShard(band.Lo, band.Hi))
			if err != nil {
				t.Fatal(err)
			}
			defer m.CloseWeights()
			if r.rejected != 0 || r.readBytes != wantBytes || r.reads != 3*(band.Hi-band.Lo) {
				t.Fatalf("expert reads=%d bytes=%d rejected=%d, want reads=%d bytes=%d rejected=0", r.reads, r.readBytes, r.rejected, 3*(band.Hi-band.Lo), wantBytes)
			}
			report := m.ResidentReport()
			if report.KQuantTensors != 3*(band.Hi-band.Lo) || report.KQuantBytes != wantBytes {
				t.Fatalf("resident raw count=%d bytes=%d, want %d/%d", report.KQuantTensors, report.KQuantBytes, 3*(band.Hi-band.Lo), wantBytes)
			}
			fullReport := oracle.ResidentReport()
			if report.Q8Tensors != fullReport.Q8Tensors || report.Q8Bytes != fullReport.Q8Bytes || report.F32Tensors != fullReport.F32Tensors || report.F32Bytes != fullReport.F32Bytes || report.Q4KTensors != fullReport.Q4KTensors {
				t.Fatal("shard changed dense inventory or materialized nonowned experts in another store")
			}
			for expert := range experts {
				for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
					name := batchedExpertCanonicalName("deepseek41", 0, expert, proj)
					raw, present := m.KQuantRaw(name)
					owned := expert >= band.Lo && expert < band.Hi
					if present != owned {
						t.Fatalf("%s present=%v, owned=%v", name, present, owned)
					}
					if owned {
						want, ok := oracle.KQuantRaw(name)
						if !ok || !bytes.Equal(raw, want) || len(raw) != cap(raw) {
							t.Fatalf("%s differs from full-load bytes or retains a larger slab", name)
						}
					}
				}
			}
			t.Logf("rank=%d band=[%d,%d) expert_read_bytes=%d resident_raw_bytes=%d nonowned_reads=0", rank, band.Lo, band.Hi, r.readBytes, report.KQuantBytes)
		})
	}
}

func ownerRangeFixture(t *testing.T, proj string, typ TensorType) (*WeightSource, TensorInfo, model.Config, []byte) {
	t.Helper()
	cfg := model.Config{ModelType: "deepseek41", NumLayers: 1, NumExperts: 7, HiddenSize: 256, MoEIntermediateSize: 512}
	out, in := cfg.MoEIntermediateSize, cfg.HiddenSize
	if proj == "down" {
		out, in = in, out
	}
	info := TensorInfo{Name: "blk.0.ffn_" + proj + "_exps.weight", Dims: []uint64{uint64(in), uint64(out), 7}, Type: typ, FileOffset: 64}
	per := int(tensorOnDiskBytes(info)) / cfg.NumExperts
	raw := make([]byte, 64+cfg.NumExperts*per)
	for expert := range cfg.NumExperts {
		for i := range per {
			raw[64+expert*per+i] = byte((expert*47 + i*13) % 251)
		}
	}
	ws, err := NewWeightSource(&File{Tensors: []TensorInfo{info}}, bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatal(err)
	}
	return ws, info, cfg, raw
}

// fak-test:runtime fast est=10ms lane=default
func TestDeepSeek41OwnedExpertRangeUneven(t *testing.T) {
	for _, tt := range []struct {
		proj string
		typ  TensorType
	}{{"gate", TensorQ2_K}, {"up", TensorQ2_K}, {"down", TensorQ3_K}} {
		ws, info, cfg, raw := ownerRangeFixture(t, tt.proj, tt.typ)
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			t.Fatal(err)
		}
		bw, bb, _ := residentExpertBlockGeometry(info.Type)
		oracle, aligned, err := splitGLMMoeDsaExpertsRawQuant(cfg.ModelType, 0, tt.proj+"_proj", shape, raw[64:], bw, bb)
		if err != nil || !aligned {
			t.Fatalf("full-split oracle: %v, aligned=%v", err, aligned)
		}
		for rank := range 3 {
			band, err := ExpertShardForRank(cfg.NumExperts, 3, rank)
			if err != nil {
				t.Fatal(err)
			}
			r := &ownerRangeReader{ReaderAt: bytes.NewReader(raw), bands: []ownerRangeReadBand{ownerRangeBand(info, band)}}
			ws.r = r
			opts, err := resolveQ4KLoadOptions(cfg, []Q4KLoadOption{WithExpertShard(band.Lo, band.Hi)})
			if err != nil {
				t.Fatal(err)
			}
			tw := ws.computeQ4KTensorWork(info, cfg, false, nil, opts, 1)
			owned := band.Hi - band.Lo
			if tw.err != nil || len(tw.pending) != owned || cap(tw.pending) != owned {
				t.Fatalf("%s rank %d: pending=%d/%d, err=%v", tt.proj, rank, len(tw.pending), cap(tw.pending), tw.err)
			}
			if !tw.acctExpert || !tw.acctResident || tw.acctType != info.Type.String() || tw.acctTensors != owned || tw.acctBytes != r.readBytes || tw.tickBytes != r.readBytes || r.reads != owned || r.rejected != 0 {
				t.Fatalf("%s rank %d: wrong owner-only accounting", tt.proj, rank)
			}
			for i, p := range tw.pending {
				want := oracle[band.Lo+i]
				if p.name != want.Name || !p.resident || p.residentType != info.Type || len(p.f32) != 0 || len(p.shape) != 2 || p.shape[0] != want.Shape[0] || p.shape[1] != want.Shape[1] || !bytes.Equal(p.raw, want.Raw) || len(p.raw) != cap(p.raw) {
					t.Fatalf("%s rank %d expert %d: wrong name, dtype, shape, raw segment, or capacity", tt.proj, rank, band.Lo+i)
				}
			}
		}
	}
}

// The authenticated secondary reader must supply its own offsets and bounds.
// Neither the primary reader nor replacement pathnames may supply owner bytes.
// fak-test:runtime fast est=10ms lane=default
func TestDeepSeek41OwnedExpertRangeSplitReader(t *testing.T) {
	_, sourceInfo, cfg, payload := ownerRangeFixture(t, "gate", TensorQ2_K)
	dir := t.TempDir()
	var shards []ShardIdentity
	for shard := range 2 {
		name := "primary.weight"
		raw := bytes.Repeat([]byte{0xee}, len(payload)-64)
		if shard == 1 {
			name, raw = sourceInfo.Name, payload[64:]
		}
		data := writeSplitShard(t, uint32(shard+1), 2, 2, shard == 0, []splitTensor{{name: name, dims: sourceInfo.Dims, typ: sourceInfo.Type, data: raw}})
		path := filepath.Join(dir, fmt.Sprintf("owned-%05d-of-00002.gguf", shard+1))
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		shards = append(shards, ShardIdentity{Path: path, Size: int64(len(data)), RawFileSHA256: fmt.Sprintf("%x", sum)})
	}
	manifest, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	setHash := sha256.Sum256(manifest)
	ws, err := OpenWeightsAuthenticated(t.Context(), shards, fmt.Sprintf("%x", setHash))
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if runtime.GOOS != "windows" {
		for _, shard := range shards {
			if err := os.Rename(shard.Path, shard.Path+".retained"); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(shard.Path, []byte("replacement must never be reopened"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	info, ok := ws.Tensor(sourceInfo.Name)
	if !ok {
		t.Fatal("secondary expert missing")
	}
	band := ExpertShard{Lo: 3, Hi: 5}
	idx := ws.byName[info.Name]
	r := &ownerRangeReader{ReaderAt: ws.readerFor[idx], bands: []ownerRangeReadBand{ownerRangeBand(info, band)}}
	ws.readerFor[idx] = r
	// The owner read must use both the secondary reader and its size.
	ws.r, ws.size = nil, 0
	tw := ws.computeOwnedExpertRange(info, cfg, band)
	if tw.err != nil {
		t.Fatal(tw.err)
	}
	per := len(payload[64:]) / cfg.NumExperts
	for i, p := range tw.pending {
		start := 64 + (band.Lo+i)*per
		if !bytes.Equal(p.raw, payload[start:start+per]) {
			t.Fatal("read primary or wrong secondary offset")
		}
	}
	if r.readBytes != int64(2*per) || r.reads != 2 || r.rejected != 0 {
		t.Fatal("secondary owner read accounting")
	}
	// A primary-file size cannot rescue a truncated secondary payload, even
	// when the requested owned band would fit before the missing nonowned tail.
	ws.size = math.MaxInt64
	ws.sizeFor[idx] = info.FileOffset + tensorOnDiskBytes(info) - 1
	tw = ws.computeOwnedExpertRange(info, cfg, ExpertShard{Lo: 0, Hi: 1})
	if tw.err == nil || r.reads != 2 {
		t.Fatal("secondary full-payload bounds were ignored")
	}
}

// Formats outside this optimization keep their existing sharded resident path.
// fak-test:runtime fast est=10ms lane=default
func TestDeepSeek41OwnedExpertRangePreservesOtherFormats(t *testing.T) {
	for _, typ := range []TensorType{TensorQ4_K, TensorQ6_K} {
		ws, info, cfg, raw := ownerRangeFixture(t, "gate", typ)
		tw := ws.computeQ4KTensorWork(info, cfg, false, nil, q4kLoadOptions{expertShardSet: true, expertShard: ExpertShard{Lo: 2, Hi: 4}}, 1)
		if tw.err != nil || len(tw.pending) != 2 {
			t.Fatalf("%s existing path: %v", typ, tw.err)
		}
		per := len(raw[64:]) / cfg.NumExperts
		for i, p := range tw.pending {
			start := 64 + (i+2)*per
			if !p.resident || p.residentType != typ || !bytes.Equal(p.raw, raw[start:start+per]) {
				t.Fatalf("%s changed resident bytes", typ)
			}
		}
	}
}

type ownerRangeFailReader struct {
	calls int
	short bool
}

func (r *ownerRangeFailReader) ReadAt(p []byte, _ int64) (int, error) {
	r.calls++
	if r.calls == 1 {
		clear(p)
		return len(p), nil
	}
	if r.short {
		return len(p) - 1, nil
	}
	return 0, io.ErrUnexpectedEOF
}

// fak-test:runtime fast est=10ms lane=default
func TestDeepSeek41OwnedExpertRangeRefuses(t *testing.T) {
	for _, kind := range []string{"rank", "zero", "dimension overflow", "experts", "hidden", "projection", "layer", "band negative", "band empty", "band end", "row alignment", "element overflow", "full payload overflow", "offset negative", "offset overflow", "nonowned tail truncated", "missing reader", "unsupported dtype"} {
		t.Run(kind, func(t *testing.T) {
			ws, info, cfg, _ := ownerRangeFixture(t, "gate", TensorQ2_K)
			band := ExpertShard{Lo: 0, Hi: 1}
			switch kind {
			case "rank":
				info.Dims = info.Dims[:2]
			case "zero":
				info.Dims[0] = 0
			case "dimension overflow":
				info.Dims[0] = math.MaxUint64
			case "experts":
				cfg.NumExperts++
			case "hidden":
				cfg.HiddenSize++
			case "projection":
				info.Name = "blk.0.ffn_unknown_exps.weight"
			case "layer":
				cfg.NumLayers = 0
			case "band negative":
				band.Lo = -1
			case "band empty":
				band.Hi = 0
			case "band end":
				band.Hi = cfg.NumExperts + 1
			case "row alignment":
				info.Dims[0], cfg.HiddenSize = 128, 128
			case "element overflow":
				info.Dims[1], cfg.MoEIntermediateSize = uint64(math.MaxInt), math.MaxInt
			case "full payload overflow":
				info.Dims[2], cfg.NumExperts = uint64(math.MaxInt), math.MaxInt
			case "offset negative":
				info.FileOffset = -1
			case "offset overflow":
				info.FileOffset = math.MaxInt64 - 1
			case "nonowned tail truncated":
				ws.size--
			case "missing reader":
				ws.r = nil
			case "unsupported dtype":
				info.Type = TensorF32
			}
			read := &ownerRangeFailReader{}
			if ws.r != nil {
				ws.r = read
			}
			tw := ws.computeOwnedExpertRange(info, cfg, band)
			if tw.err == nil || len(tw.pending) != 0 || read.calls != 0 {
				t.Fatalf("err=%v pending=%d reads=%d; malformed slab must refuse before allocation/read", tw.err, len(tw.pending), read.calls)
			}
			// Selected Q2/Q3 projection failures must not fall back to TensorBytes.
			if _, _, selected := glmMoeDsaBatchedExpert(info.Name); selected && info.Type == TensorQ2_K {
				opts := q4kLoadOptions{expertShardSet: true, expertShard: band}
				tw = ws.computeQ4KTensorWork(info, cfg, false, nil, opts, 1)
				if tw.err == nil || len(tw.pending) != 0 || read.calls != 0 {
					t.Fatal("selected malformed range fell back")
				}
			}
		})
	}
	for _, short := range []bool{false, true} {
		ws, info, cfg, _ := ownerRangeFixture(t, "gate", TensorQ2_K)
		r := &ownerRangeFailReader{short: short}
		ws.r = r
		tw := ws.computeQ4KTensorWork(info, cfg, false, nil, q4kLoadOptions{expertShardSet: true, expertShard: ExpertShard{Lo: 2, Hi: 5}}, 1)
		if !errors.Is(tw.err, io.ErrUnexpectedEOF) || len(tw.pending) != 0 || r.calls != 2 || tw.acctBytes != 0 {
			t.Fatalf("short=%v: err=%v calls=%d pending=%d; partial owner load must fail without fallback", short, tw.err, r.calls, len(tw.pending))
		}
	}
	_, err := resolveQ4KLoadOptions(model.Config{NumExperts: 7}, []Q4KLoadOption{WithStreamedExperts(0), WithExpertShard(0, 3)})
	if err == nil || !strings.Contains(err.Error(), "cannot both") {
		t.Fatal("streamed+shard refusal changed")
	}
}
