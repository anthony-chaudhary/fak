package ggufload

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type servingReaderFactoryForTest interface {
	openV41EngramServingReader(int) (io.ReaderAt, int, int, error)
}

type servingReaderLeaseForTest interface {
	V41EngramHoldSourceReads() (func(), <-chan struct{}, bool)
}

// The barrier wraps the actual owning shard ReaderAt. It is also an explicit
// member of that WeightSource's closer list, not a model-lifetime substitute.
type servingReaderOwnerBarrier struct {
	reader  io.ReaderAt
	off     int64
	size    int
	entered chan struct{}
	resume  chan struct{}
	events  *retainedReadEvents
	mu      sync.Mutex
	closed  bool
}

func (r *servingReaderOwnerBarrier) ReadAt(dst []byte, off int64) (int, error) {
	r.events.add("read-enter")
	close(r.entered)
	<-r.resume
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return 0, fmt.Errorf("owning shard closed during retained read")
	}
	if off != r.off || len(dst) != r.size {
		return 0, fmt.Errorf("read extent [%d,%d), want [%d,%d)", off, off+int64(len(dst)), r.off, r.off+int64(r.size))
	}
	n, err := r.reader.ReadAt(dst, off)
	r.events.add("read-complete")
	return n, err
}

func (r *servingReaderOwnerBarrier) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return fmt.Errorf("owning shard closed twice")
	}
	r.closed = true
	r.events.add("reader-close")
	return nil
}

// fak-test:runtime fast est=200ms lane=default
func TestV41EngramServingReaderOwnership(t *testing.T) {
	t.Run("unowned", func(t *testing.T) {
		plain := newV41EngramRowReaderAt(&V41EngramGGUFSource{r: bytes.NewReader(v41TestRow(1)), tableRows: 1}, 1, v41EngramRowBytes)
		if holder, exists := plain.(servingReaderLeaseForTest); exists {
			release, closing, ok := holder.V41EngramHoldSourceReads()
			if release != nil {
				release()
			}
			if ok || release != nil || closing != nil {
				t.Fatal("unowned adapter advertised retained ownership")
			}
		}
		if _, exists := plain.(model.V41EngramEncodedRows); exists {
			t.Fatal("packed adapter acquired an encoded-row method")
		}
		dst := make([]byte, v41EngramRowBytes)
		if n, err := plain.ReadAt(dst, 0); err != nil || n != len(dst) || !bytes.Equal(dst, v41TestRow(1)) {
			t.Fatalf("unowned ordinary read: bytes=%d err=%v", n, err)
		}
	})
	t.Run("owned", func(t *testing.T) {
		if _, ok := any(&WeightSource{}).(servingReaderFactoryForTest); !ok {
			t.Fatal("WeightSource is missing the owned Engram serving-reader factory")
		}
		for _, dialect := range []string{"packed", "q2-k-encoded", "q4-k-decoded"} {
			t.Run(dialect, func(t *testing.T) {
				var ws *WeightSource
				var want []byte
				stride := v41EngramRowBytes
				switch dialect {
				case "packed":
					dir := t.TempDir()
					path := filepath.Join(dir, "v41-00001-of-00002.gguf")
					if err := os.WriteFile(path, v41BuildShard1(t, 2), 0600); err != nil {
						t.Fatal(err)
					}
					rows := [][]byte{v41TestRow(1), v41TestRow(2), v41TestRow(3)}
					if err := os.WriteFile(filepath.Join(dir, "v41-00002-of-00002.gguf"), v41BuildShard2(t, "blk.1.engram_embd.weight", rows), 0600); err != nil {
						t.Fatal(err)
					}
					var err error
					ws, err = OpenWeights(path)
					if err != nil {
						t.Fatal(err)
					}
					want = rows[1]
				case "q2-k-encoded":
					ws = v41Q2KOpenFixture(t, 3, true)
					want, _ = v41Q2KBlockRow(1)
					stride = blockQ2KBytes
				case "q4-k-decoded":
					path, _ := writeV41Q4KEngramFixture(t)
					var err error
					ws, err = OpenWeights(path)
					if err != nil {
						t.Fatal(err)
					}
					_, decoded := v41Q4KBlockRow(1)
					want = f32BytesQ2KForTest(decoded)
					stride = blockQ4KBytes
				}
				t.Cleanup(func() { ws.Close() })
				info, ok := ws.Tensor("blk.1.engram_embd.weight")
				if !ok {
					t.Fatal("fixture has no Engram table")
				}
				original, _, err := ws.tensorReader(info)
				if err != nil {
					t.Fatal(err)
				}
				var events retainedReadEvents
				barrier := &servingReaderOwnerBarrier{reader: original, off: info.FileOffset + int64(stride), size: stride, entered: make(chan struct{}), resume: make(chan struct{}), events: &events}
				idx := ws.byName[info.Name]
				if idx >= len(ws.readerFor) || ws.readerFor[idx] == nil {
					t.Fatal("fixture lost its noninitial-shard route")
				}
				ws.readerFor[idx] = barrier
				ws.closers = append([]io.Closer{barrier}, ws.closers...)
				reader, rows, rowBytes, err := any(ws).(servingReaderFactoryForTest).openV41EngramServingReader(1)
				if err != nil || rows < 2 {
					t.Fatalf("serving factory: rows=%d err=%v", rows, err)
				}
				holder, ok := reader.(servingReaderLeaseForTest)
				if !ok {
					t.Fatal("serving reader lost its exact source owner")
				}
				enc, hasEncoded := reader.(model.V41EngramEncodedRows)
				if hasEncoded != (dialect != "packed") {
					t.Fatal("serving adapter changed encoded-method presence")
				}
				readFrom, readOff := reader, int64(rowBytes)
				if hasEncoded {
					r, off, size, encoding, err := enc.V41EngramEncodedRows(1, 1)
					wantEncoding := model.V41EngramEncodingQ2K
					if dialect == "q4-k-decoded" {
						wantEncoding = model.V41EngramEncodingQ4K
					}
					if err != nil || r != barrier || off != barrier.off || size != int64(stride) || encoding != wantEncoding {
						t.Fatalf("encoded source/extent changed: offset=%d size=%d encoding=%q err=%v", off, size, encoding, err)
					}
					if dialect == "q2-k-encoded" {
						readFrom, readOff = r, off
					}
				}
				release, closing, ok := holder.V41EngramHoldSourceReads()
				if !ok || release == nil || closing == nil {
					if release != nil {
						release()
					}
					t.Fatal("live serving owner refused retention")
				}
				defer release()
				dst := make([]byte, len(want))
				readDone, readJoined := make(chan error, 1), make(chan struct{})
				go func() {
					defer close(readJoined)
					n, err := readFrom.ReadAt(dst, readOff)
					if err == nil && n != len(dst) {
						err = io.ErrUnexpectedEOF
					}
					readDone <- err
				}()
				select {
				case <-barrier.entered:
				case err := <-readDone:
					<-readJoined
					t.Fatalf("read bypassed the owning-shard barrier: %v", err)
				}
				closeDone, closeJoined := make(chan error, 1), make(chan struct{})
				go func() {
					defer close(closeJoined)
					err := ws.Close()
					events.add("close-return")
					closeDone <- err
				}()
				var closeErr error
				closedEarly := false
				select {
				case <-closing:
				case closeErr = <-closeDone:
					closedEarly = true
					t.Error("source Close returned before the retained read was joined")
				}
				lateRelease, lateClosing, lateOK := holder.V41EngramHoldSourceReads()
				if lateRelease != nil {
					lateRelease()
				}
				if lateOK || lateRelease != nil || lateClosing != nil {
					t.Error("closing serving owner admitted a late borrower")
				}
				close(barrier.resume)
				<-readJoined
				readErr := <-readDone
				events.add("read-joined")
				events.add("release")
				release()
				release()
				if !closedEarly {
					closeErr = <-closeDone
				}
				<-closeJoined
				if readErr != nil || closeErr != nil || !bytes.Equal(dst, want) {
					t.Errorf("retained read=%v Close=%v bytes-equal=%v", readErr, closeErr, bytes.Equal(dst, want))
				}
				if err := ws.Close(); err != nil {
					t.Errorf("completed repeat Close: %v", err)
				}
				events.check(t, "read-enter", "read-complete", "read-joined", "release", "reader-close", "close-return")
			})
		}
	})
}
