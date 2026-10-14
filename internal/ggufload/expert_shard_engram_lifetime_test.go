package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"
)

// The original retained readers are closed by the model, not by expert range
// materialization. After attachment, every Engram read must remain row-bounded.
type ownerEngramReader struct {
	io.ReaderAt
	io.Closer
	mu                           sync.Mutex
	tableLo, tableHi             int64
	bound                        bool
	closed, tableReads, rowReads int
}

func (r *ownerEngramReader) ReadAt(p []byte, off int64) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed != 0 {
		return 0, os.ErrClosed
	}
	if r.tableHi > r.tableLo && off < r.tableHi && off+int64(len(p)) > r.tableLo {
		if r.bound {
			if off < r.tableLo || off+int64(len(p)) > r.tableHi || len(p) != blockQ2KBytes || (off-r.tableLo)%blockQ2KBytes != 0 {
				return 0, fmt.Errorf("Engram row read escaped bounded source: %d bytes", len(p))
			}
			r.rowReads++
		} else {
			// The existing attachment hashes the encoded table once. A second
			// full-table read would expose eager materialization or re-binding.
			r.tableReads++
			if r.tableReads != 1 || off != r.tableLo || off+int64(len(p)) != r.tableHi {
				return 0, fmt.Errorf("unexpected Engram acquisition read")
			}
		}
	}
	return r.ReaderAt.ReadAt(p, off)
}

func (r *ownerEngramReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed++
	return r.Closer.Close()
}

// This is a reduced attachment/lifetime fixture, not a runnable EP decoder.
func ownerEngramShards(t *testing.T) []ShardIdentity {
	t.Helper()
	f, err := Open(writeV41UnboundEngramFixture(t, true))
	if err != nil {
		t.Fatal(err)
	}
	meta := f.Metadata
	meta["deepseek41.embedding_length"] = Value{Type: TypeUint32, Value: uint32(256)}
	meta["deepseek41.block_count"] = Value{Type: TypeUint32, Value: uint32(1)}
	meta["deepseek41.feed_forward_length"] = Value{Type: TypeUint32, Value: uint32(512)}
	meta["deepseek41.expert_count"] = Value{Type: TypeUint32, Value: uint32(7)}
	meta["deepseek41.expert_used_count"] = Value{Type: TypeUint32, Value: uint32(1)}
	meta["deepseek41.expert_feed_forward_length"] = Value{Type: TypeUint32, Value: uint32(512)}
	meta["deepseek41.engram.layer_ids"] = vcruzI32Array([]int32{0})
	meta["split.tensors.count"] = Value{Type: TypeUint32, Value: uint32(5)}
	slabs := [][]splitTensor{{{name: "output.weight", dims: []uint64{256, 4}, typ: TensorF32, data: make([]byte, 256*4*4)}}, nil}
	for _, spec := range []struct {
		proj  string
		typ   TensorType
		shard int
	}{{"gate", TensorQ2_K, 0}, {"up", TensorQ2_K, 0}, {"down", TensorQ3_K, 1}} {
		_, info, _, raw := ownerRangeFixture(t, spec.proj, spec.typ)
		slabs[spec.shard] = append(slabs[spec.shard], splitTensor{name: info.Name, dims: info.Dims, typ: info.Type, data: raw[64:]})
	}
	var table []byte
	for row := range 72 {
		block, _ := v41Q2KBlockRow(row)
		table = append(table, block...)
	}
	slabs[1] = append(slabs[1], splitTensor{name: "blk.0.engram_embd.weight", dims: []uint64{256, 72}, typ: TensorQ2_K, data: table})
	dir := t.TempDir()
	var manifest []ShardIdentity
	for shard, tensors := range slabs {
		md := meta
		if shard == 1 {
			md = map[string]Value{"general.alignment": {Type: TypeUint32, Value: uint32(32)}, "split.no": {Type: TypeUint32, Value: uint32(1)}, "split.count": {Type: TypeUint32, Value: uint32(2)}, "split.tensors.count": {Type: TypeUint32, Value: uint32(5)}}
		}
		keys := make([]string, 0, len(md))
		for key := range md {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		var b bytes.Buffer
		writeMinimalHeader(&b, uint64(len(tensors)), uint64(len(md)))
		for _, key := range keys {
			writeMetaValueForTest(t, &b, key, md[key])
		}
		offset := 0
		for _, tensor := range tensors {
			writeTensorInfoForTest(&b, tensor.name, tensor.dims, tensor.typ, uint64(offset))
			offset = (offset + len(tensor.data) + 31) / 32 * 32
		}
		padToAlignment(&b, 32)
		for _, tensor := range tensors {
			b.Write(tensor.data)
			padToAlignment(&b, 32)
		}
		path := filepath.Join(dir, fmt.Sprintf("owner-engram-%05d-of-00002.gguf", shard+1))
		if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b.Bytes())
		manifest = append(manifest, ShardIdentity{Path: path, Size: int64(b.Len()), RawFileSHA256: fmt.Sprintf("%x", sum)})
	}
	return manifest
}

// Authentication scans whole files and attachment hashes the encoded table;
// neither is claimed to be owner-local I/O. After those existing checks, the
// actual attached stage must retain bounded row reads through model teardown.
// fak-test:runtime fast est=100ms lane=default
func TestDeepSeek41OwnedExpertRangeEngramLifetime(t *testing.T) {
	shards := ownerEngramShards(t)
	manifest, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(manifest)
	ws, err := openWeightsAuthenticated(t.Context(), shards, fmt.Sprintf("%x", digest), false, openShardReader)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	band := ExpertShard{Lo: 2, Hi: 4}
	readers := make(map[io.ReaderAt]*ownerEngramReader)
	for i, closer := range ws.closers {
		r := closer.(io.ReaderAt)
		tracked := &ownerEngramReader{ReaderAt: r, Closer: closer}
		readers[r], ws.closers[i] = tracked, tracked
	}
	var tableReader *ownerEngramReader
	var expertReaders []*ownerRangeReader
	for i, info := range ws.File.Tensors {
		tracked := readers[ws.readerFor[i]]
		if tracked == nil {
			t.Fatal("tensor was not routed to retained authenticated reader")
		}
		if info.Name == "blk.0.engram_embd.weight" {
			tracked.tableLo, tracked.tableHi = info.FileOffset, info.FileOffset+tensorOnDiskBytes(info)
			tableReader = tracked
		}
		ws.readerFor[i] = tracked
		if _, _, ok := glmMoeDsaBatchedExpert(info.Name); ok {
			r := &ownerRangeReader{ReaderAt: tracked, bands: []ownerRangeReadBand{ownerRangeBand(info, band)}}
			ws.readerFor[i] = r
			expertReaders = append(expertReaders, r)
		}
	}
	if runtime.GOOS != "windows" {
		for _, shard := range shards {
			if err := os.Rename(shard.Path, shard.Path+".retained"); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(shard.Path, []byte("replacement must not be opened"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, WithExpertShard(band.Lo, band.Hi), WithStreamedDenseQ4K(true))
	if err != nil {
		t.Fatal(err)
	}
	m.SetWeightCloser(ws)
	defer m.CloseWeights()
	if !m.V41EngramAttached() || m.KQuantCount() != 6 || tableReader == nil || tableReader.tableReads != 1 {
		t.Fatal("owner model omitted Engram or changed existing encoded acquisition")
	}
	for _, r := range expertReaders {
		if r.rejected != 0 || r.reads != 2 {
			t.Fatal("nonowned expert read during Engram attachment")
		}
	}
	for _, r := range readers {
		if r.closed != 0 {
			t.Fatal("source closed before model handoff")
		}
	}
	tableReader.bound = true
	// The existing 4*1024-byte budget holds 48 encoded Q2_K rows. This
	// prefix requests all72 fixture rows, so a second warm must still miss.
	prefix := []int{0, 1, 2, 0, 2, 1, 0, 1, 3, 2}
	encodedCapacity := int64(v41EngramServeBudgetRows * v41EngramQ2KRowBytes / blockQ2KBytes)
	for pass := range 2 {
		delta, err := m.WarmV41EngramPrefix(prefix, nil)
		if err != nil || delta.Requested <= encodedCapacity || delta.Misses == 0 || delta.BytesRead != delta.Misses*blockQ2KBytes {
			t.Fatalf("attached bounded warm %d: %+v %v", pass, delta, err)
		}
	}
	if tableReader.rowReads == 0 {
		t.Fatal("actual attached stage issued no bounded row reads")
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	for _, r := range readers {
		if r.closed != 1 {
			t.Fatalf("source closes=%d, want1 at model teardown", r.closed)
		}
	}
	if _, err := m.WarmV41EngramPrefix(prefix, nil); err == nil {
		t.Fatal("closed source was silently reused for a nonresident prefix")
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	for _, r := range readers {
		if r.closed != 1 {
			t.Fatal("model closed source more than once")
		}
	}
}
