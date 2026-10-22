package model

import (
	"bytes"
	"io"
	"sync/atomic"
	"testing"
)

// This probe checks delegation only. Real retained-read/Close ordering belongs
// to the loader's separate TestV41EngramServingReaderOwnership witness.
type v41EngramOwnerProbe struct {
	io.ReaderAt
	closing  chan struct{}
	accept   bool
	holds    int
	releases int
	reads    atomic.Int64
}

func (p *v41EngramOwnerProbe) ReadAt(dst []byte, off int64) (int, error) {
	p.reads.Add(1)
	return p.ReaderAt.ReadAt(dst, off)
}

func (p *v41EngramOwnerProbe) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	p.holds++
	if !p.accept {
		return nil, nil, false
	}
	return func() { p.releases++ }, p.closing, true
}

type v41EngramEncodedOwnerProbe struct {
	*v41EngramOwnerProbe
	encoded []byte
}

func (p *v41EngramEncodedOwnerProbe) V41EngramEncodedRows(start, count int) (io.ReaderAt, int64, int64, string, error) {
	return bytes.NewReader(p.encoded), int64(start) * v41EngramQ2KEncodedRowBytes,
		int64(count) * v41EngramQ2KEncodedRowBytes, V41EngramEncodingQ2K, nil
}

// Local structural gating keeps this same test compilable before the bridge.
type v41EngramTestSourceOwner interface {
	V41EngramHoldSourceReads() (func(), <-chan struct{}, bool)
}

// TestV41EngramModelSourceOwnership observes the actual source retained by the
// production stage/cache wiring, including its Q2_K encoded-form substitution.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramModelSourceOwnership(t *testing.T) {
	cfg := Config{VocabSize: 4, DeepSeekV41: &DeepSeekV41Config{
		EngramLayerIDs: []int{1}, EngramHeadDim: 4,
	}}
	layout := v41EngramTestLayout(cfg)
	rows := int(layout.Rows[0])
	packed := v41EngramTestPackedRows(rows)
	f32 := v41EngramF32RowsForPacked(packed, rows, 256)
	encoded := make([]byte, rows*v41EngramQ2KEncodedRowBytes)
	for r := 0; r < rows; r++ {
		row := encoded[r*v41EngramQ2KEncodedRowBytes : (r+1)*v41EngramQ2KEncodedRowBytes]
		row[0], row[16], row[81] = 1, byte(r+1), 0x3c
	}

	for _, constructor := range []string{"verified", "acquired"} {
		for _, dialect := range []string{"packed", "f32", "q2-k"} {
			t.Run(constructor+"/"+dialect, func(t *testing.T) {
				data, rowBytes, wantRows := packed, V41EngramPackedRowBytes, packed
				if dialect != "packed" {
					data, rowBytes, wantRows = f32, V41EngramF32RowBytes, f32
				}
				if dialect == "q2-k" {
					wantRows = encoded
				}
				binding := V41EngramArtifactBinding{Shard: "owned-model-rows", Size: int64(len(data)),
					Rows: rows, RowBytes: rowBytes}
				shard := V41EngramShard{ID: binding.Shard, ExpectedID: binding.Shard,
					Size: binding.Size, Rows: rows}
				if constructor == "verified" {
					binding.Digest = v41EngramTestDigest(data)
					if dialect == "q2-k" {
						var err error
						binding.Digest, err = v41EngramTreeDigest(bytes.NewReader(encoded), 0, int64(len(encoded)), V41EngramEncodingQ2K, 1)
						if err != nil {
							t.Fatal(err)
						}
					}
				}
				build := func(reader io.ReaderAt) *V41EngramRowCache {
					t.Helper()
					var src V41EngramRowSource
					var err error
					if constructor == "verified" {
						src, err = NewV41VerifiedEngramRowSource(reader, shard, binding)
					} else {
						src, err = NewV41AcquiredEngramRowSource(reader, shard, binding)
					}
					if err != nil {
						t.Fatal(err)
					}
					if got, ok := V41EngramBinding(src); !ok || got.Digest == "" || got.RowBytes != rowBytes {
						t.Fatalf("binding lost: %+v, %t", got, ok)
					}
					m := &Model{Cfg: cfg}
					t.Cleanup(m.v41DetachEngramStage)
					if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(rowBytes)); err != nil {
						t.Fatal(err)
					}
					stage := m.v41EngramStageFor()
					if stage == nil || len(stage.caches) != 1 || stage.caches[0].src.RowBytes() != len(wantRows)/rows {
						t.Fatal("production stage did not retain the expected row dialect")
					}
					return stage.caches[0]
				}
				var plain io.ReaderAt = bytes.NewReader(data)
				if dialect == "q2-k" {
					plain = v41EngramEncodedFake{ReaderAt: plain, encoded: encoded, encoding: V41EngramEncodingQ2K}
				}
				unowned := build(plain)
				if owner, present := unowned.src.(v41EngramTestSourceOwner); present {
					release, closing, ok := owner.V41EngramHoldSourceReads()
					if ok || release != nil || closing != nil {
						t.Fatal("unowned stage source admitted a hold")
					}
				}
				probe := &v41EngramOwnerProbe{ReaderAt: bytes.NewReader(data), closing: make(chan struct{}), accept: true}
				var reader io.ReaderAt = probe
				if dialect == "q2-k" {
					reader = &v41EngramEncodedOwnerProbe{v41EngramOwnerProbe: probe, encoded: encoded}
				}
				cache := build(reader)
				if probe.holds != 0 || probe.releases != 0 {
					t.Fatal("construction or wiring acquired an implicit source hold")
				}
				owner, present := cache.src.(v41EngramTestSourceOwner)
				if !present {
					t.Fatal("wired Engram source is missing retained-source capability")
				}
				reads := probe.reads.Load()
				release, closing, ok := owner.V41EngramHoldSourceReads()
				if !ok || release == nil || closing != probe.closing || probe.holds != 1 ||
					probe.releases != 0 || probe.reads.Load() != reads {
					t.Fatal("hold did not preserve the exact backing owner contract")
				}
				got, err := cache.Row(1)
				width := len(wantRows) / rows
				if err != nil || !bytes.Equal(got, wantRows[width:2*width]) {
					t.Fatalf("retained row changed: %v", err)
				}
				release()
				if probe.releases != 1 {
					t.Fatal("release did not reach the exact backing owner")
				}
				close(probe.closing)
				probe.accept = false
				release, closing, ok = owner.V41EngramHoldSourceReads()
				if ok || release != nil || closing != nil || probe.holds != 2 || probe.releases != 1 {
					t.Fatal("backing-owner refusal was not preserved")
				}
				plainRow, err := unowned.Row(1)
				if err != nil || !bytes.Equal(plainRow, got) {
					t.Fatalf("ordinary unowned read changed: %v", err)
				}
			})
		}
	}
	t.Run("nil-wrappers", func(t *testing.T) {
		for _, src := range []any{(*v41PackedEngramRowSource)(nil), (*v41VerifiedEngramRowSource)(nil), (*v41EngramQ2KRowSource)(nil)} {
			if owner, present := src.(v41EngramTestSourceOwner); present {
				release, closing, ok := owner.V41EngramHoldSourceReads()
				if ok || release != nil || closing != nil {
					t.Fatal("nil wrapper admitted a source hold")
				}
			}
		}
	})
}
