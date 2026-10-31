package model

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Reuse the existing owner probe; encoded extents borrow a different ReaderAt
// so the witness can distinguish the original source hold from payload reads.
type v41EngramIQ4NLStagingProbe struct {
	*v41EngramOwnerProbe
	raw      *v41EngramOwnerProbe
	encoding string
	sizeSkew int64
	calls    [][2]int
}

func (p *v41EngramIQ4NLStagingProbe) V41EngramEncodedRows(start, count int) (io.ReaderAt, int64, int64, string, error) {
	p.calls = append(p.calls, [2]int{start, count})
	return p.raw, int64(start) * v41EngramIQ4NLEncodedRowBytes,
		int64(count)*v41EngramIQ4NLEncodedRowBytes + p.sizeSkew, p.encoding, nil
}

// One authenticated, offset view proves exact encoded staging and consumption;
// metadata refusal is observed before payload reads, not inferred from an error.
// fak-test:runtime fast est=100ms lane=default
func TestV41EngramIQ4NLAuthenticatedStaging(t *testing.T) {
	const base, dim = 5, 97
	cfg := Config{VocabSize: 4, DeepSeekV41: &DeepSeekV41Config{
		EngramLayerIDs: []int{1}, EngramHeadDim: dim,
	}}
	layout := v41EngramTestLayout(cfg)
	rows := int(layout.Rows[0])
	raw := make([]byte, (base+rows)*v41EngramIQ4NLEncodedRowBytes)
	f32 := make([]byte, (base+rows)*V41EngramF32RowBytes)
	for r := 0; r < base+rows; r++ {
		var values [qkK]float32
		for b := 0; b < qkK/iq4nlBlockWeights; b++ {
			block := raw[r*v41EngramIQ4NLEncodedRowBytes+b*iq4nlBlockBytes:][:iq4nlBlockBytes]
			binary.LittleEndian.PutUint16(block, 0x3c00) // exact f16 1
			for j := 2; j < len(block); j++ {
				block[j] = byte((r+b+j)&15) | byte((r+3*b+j+7)&15)<<4
			}
			iq4nlDequantBlock(values[b*iq4nlBlockWeights:], block)
		}
		for j, value := range values {
			binary.LittleEndian.PutUint32(f32[r*V41EngramF32RowBytes+4*j:], math.Float32bits(value))
		}
	}
	owner := &v41EngramOwnerProbe{ReaderAt: bytes.NewReader(f32), closing: make(chan struct{}), accept: true}
	payload := &v41EngramOwnerProbe{ReaderAt: bytes.NewReader(raw), closing: make(chan struct{}), accept: true}
	probe := &v41EngramIQ4NLStagingProbe{v41EngramOwnerProbe: owner, raw: payload, encoding: V41EngramEncodingIQ4NL}
	digest, err := v41EngramTreeDigest(bytes.NewReader(raw), base*v41EngramIQ4NLEncodedRowBytes,
		int64(rows*v41EngramIQ4NLEncodedRowBytes), V41EngramEncodingIQ4NL, 1)
	if err != nil {
		t.Fatal(err)
	}
	binding := V41EngramArtifactBinding{Shard: "iq4nl-offset-view", Offset: base * V41EngramF32RowBytes,
		Size: int64(rows * V41EngramF32RowBytes), Rows: rows, RowBytes: V41EngramF32RowBytes, Digest: digest}
	shard := V41EngramShard{ID: binding.Shard, ExpectedID: binding.Shard, Offset: binding.Offset, Size: binding.Size, Rows: rows}
	src, err := NewV41VerifiedEngramRowSource(probe, shard, binding)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := V41EngramBinding(src); !ok || got != binding || src.RowBytes() != V41EngramF32RowBytes {
		t.Fatalf("served binding units changed: %+v, %t", got, ok)
	}
	form, ok := v41EngramEncodedForm(src)
	if !ok || form.RowBytes() != 144 || form.encoding != V41EngramEncodingIQ4NL || form.base != base {
		t.Fatalf("missing exact bound IQ4_NL form: %+v, %t", form, ok)
	}
	release, closing, held := form.V41EngramHoldSourceReads()
	if !held || release == nil || closing != owner.closing || owner.holds != 1 || payload.holds != 0 {
		t.Fatal("encoded view did not retain its original source owner")
	}
	release()
	if owner.releases != 1 {
		t.Fatal("encoded source hold was not released")
	}
	m := &Model{Cfg: cfg}
	t.Cleanup(m.v41DetachEngramStage)
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, 3*144); err != nil {
		t.Fatal(err)
	}
	stage := m.v41EngramStageFor()
	if stage == nil || stage.rowBytes != 144 || stage.caches[0].src.RowBytes() != 144 {
		t.Fatal("verified IQ4_NL was expanded or mislabeled at cache fill")
	}
	probe.calls = nil
	payload.reads.Store(0)
	ids := []uint32{2, 0, 2}
	gathered, panel, err := stage.caches[0].gatherKnownRowsOwned(ids, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(panel) != len(ids)*144 || len(gathered) != len(ids) || owner.reads.Load() != 0 {
		t.Fatal("gather expanded F32 rows or returned the wrong encoded extent")
	}
	for i, id := range ids {
		want := raw[(base+int(id))*144:][:144]
		if !bytes.Equal(gathered[i], want) || &gathered[i][0] != &panel[i*144] || cap(gathered[i]) != 144 {
			t.Fatalf("local row %d did not preserve owned raw global row %d", id, base+int(id))
		}
	}
	if len(probe.calls) != 2 || probe.calls[0] != [2]int{base, 1} || probe.calls[1] != [2]int{base + 2, 1} {
		t.Fatalf("unbounded or incorrect encoded source requests: %v", probe.calls)
	}
	g := &v41EngramGatheredRows{layer: 1, stage: stage, rows: gathered, dim: dim, encoded: panel}
	if encoding, ok := g.encoding(); !ok || encoding != compute.V41EngramRowIQ4NL {
		t.Fatal("authenticated IQ4_NL was not presented as its exact compute dialect")
	}
	stored, ok := g.stored()
	if !ok || &stored[0] != &panel[0] {
		t.Fatal("resident gather lost its owned contiguous upload panel")
	}
	values, err := g.decodeHost()
	if err != nil {
		t.Fatal(err)
	}
	for i, id := range ids {
		for j := 0; j < dim; j++ {
			want := binary.LittleEndian.Uint32(f32[(base+int(id))*V41EngramF32RowBytes+4*j:])
			if math.Float32bits(values[i*dim+j]) != want {
				t.Fatalf("host consumption changed split-half row %d element %d", id, j)
			}
		}
	}
	// Equal byte width never authenticates Q4_K as IQ4_NL.
	q4Digest := strings.Replace(digest, V41EngramEncodingIQ4NL, V41EngramEncodingQ4K, 1)
	if v41EngramBoundEncoded(probe, q4Digest) != nil {
		t.Fatal("same-width Q4_K digest admitted the IQ4_NL cache")
	}
	for _, mutation := range []string{"encoding", "extent"} {
		probe.encoding, probe.sizeSkew = V41EngramEncodingIQ4NL, 0
		if mutation == "encoding" {
			probe.encoding = V41EngramEncodingQ4K
		} else {
			probe.sizeSkew = 1
		}
		before := payload.reads.Load()
		if n, err := form.ReadRows(2, 1, make([]byte, 144)); err == nil || n != 0 || payload.reads.Load() != before {
			t.Fatalf("changed %s reached payload read: n=%d err=%v", mutation, n, err)
		}
	}
	probe.encoding, probe.sizeSkew = V41EngramEncodingIQ4NL, 0
	if err := g.deviceFault([]float32{math.Float32frombits(130), 0, math.Float32frombits(130)}); err == nil || !strings.Contains(err.Error(), "iq4_nl row value 64") {
		t.Fatalf("first finite fault lost exact dialect/index: %v", err)
	}
	if owner.reads.Load() != 0 {
		t.Fatal("encoded consumption replayed through the served-F32 reader")
	}
}
