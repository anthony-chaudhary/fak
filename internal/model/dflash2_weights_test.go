package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type dflash2TestCloser struct {
	calls atomic.Int32
	err   error
}

func (c *dflash2TestCloser) Close() error {
	c.calls.Add(1)
	return c.err
}

type dflash2TestReaderAt func([]byte, int64) (int, error)

func (f dflash2TestReaderAt) ReadAt(p []byte, off int64) (int, error) { return f(p, off) }

func dflash2TestFile(t *testing.T, header string, payload []byte) ([]byte, dflash2WeightContract) {
	t.Helper()
	file := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(file, uint64(len(header)))
	copy(file[8:], header)
	copy(file[8+len(header):], payload)
	digest := sha256.Sum256(file)
	return file, dflash2WeightContract{
		sha256: hex.EncodeToString(digest[:]), fileBytes: int64(len(file)),
		shapes: map[string][]int{"a": {2, 2}, "b": {2}},
	}
}

func dflash2TestFixture(t *testing.T) ([]byte, dflash2WeightContract) {
	t.Helper()
	header := `{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}}`
	payload := make([]byte, 12)
	for i, bits := range []uint16{0x3f80, 0xc020, 0x0000, 0x8000, 0x7f80, 0x7fc1} {
		binary.LittleEndian.PutUint16(payload[2*i:], bits)
	}
	return dflash2TestFile(t, header, payload)
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2WeightsStockContract(t *testing.T) {
	t.Parallel()
	c := qwen38DFlash2WeightContract()
	if len(c.shapes) != 81 { //boundarylint:ignore CHANGE_DETECTOR_TEST pinned five-layer artifact requires 6 global plus 5*15 layer tensors
		t.Fatalf("unexpected stock tensor count=%d", len(c.shapes))
	}
	if c.fileBytes != 3848817896 {
		t.Fatalf("unexpected stock file bytes=%d", c.fileBytes)
	}
	digest, err := hex.DecodeString(c.sha256)
	if err != nil || len(digest) != sha256.Size {
		t.Fatalf("stock digest is not a SHA-256 encoding: %q (%v)", c.sha256, err)
	}
	if c.sha256 != "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c" {
		t.Fatalf("stock digest differs from the pinned artifact: %s", c.sha256)
	}
	var total int64
	for _, shape := range c.shapes {
		n := int64(2)
		for _, dim := range shape {
			n *= int64(dim)
		}
		total += n
	}
	if total != 3848808960 {
		t.Fatalf("payload bytes=%d", total)
	}
	c.shapes["layers.0.self_attn.q_proj.weight"][0] = 1
	if c.shapes["layers.1.self_attn.q_proj.weight"][0] != 4096 || qwen38DFlash2WeightContract().shapes["layers.0.self_attn.q_proj.weight"][0] != 4096 {
		t.Fatal("stock shape storage aliases another layer or contract")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2WeightsNativeSpansAndClose(t *testing.T) {
	t.Parallel()
	file, contract := dflash2TestFixture(t)
	closer := &dflash2TestCloser{}
	w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), closer, contract)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Error(err)
		}
		if closer.calls.Load() != 1 {
			t.Errorf("close count=%d", closer.calls.Load())
		}
	})
	for _, tc := range []struct {
		name string
		bits []uint32
	}{
		{"a", []uint32{0x3f800000, 0xc0200000, 0, 0x80000000}},
		{"b", []uint32{0x7f800000, 0x7fc10000}},
	} {
		dst := make([]float32, len(tc.bits))
		if err := w.ReadBF16(context.Background(), tc.name, 0, dst); err != nil {
			t.Fatal(err)
		}
		for i, want := range tc.bits {
			if got := math.Float32bits(dst[i]); got != want {
				t.Fatalf("%s[%d]=%08x want %08x", tc.name, i, got, want)
			}
		}
	}
	dst := []float32{77}
	if err := w.ReadBF16(context.Background(), "a", 1, dst); err != nil || dst[0] != -2.5 {
		t.Fatalf("offset span=%v err=%v", dst, err)
	}
	for _, tc := range []struct {
		name  string
		first int64
		count int
	}{{"missing", 0, 1}, {"a", -1, 1}, {"a", 4, 1}, {"a", math.MaxInt64, 1}, {"a", 0, dflash2MaxReadElements + 1}} {
		if err := w.ReadBF16(context.Background(), tc.name, tc.first, make([]float32, tc.count)); err == nil {
			t.Fatalf("accepted invalid span %+v", tc)
		}
	}
	if err := w.ReadBF16(context.Background(), "a", 4, nil); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := w.ReadBF16(context.Background(), "a", 0, dst); err == nil {
		t.Fatal("read after close")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestDFlash2WeightsRejectMetadata(t *testing.T) {
	t.Parallel()
	validFile, _ := dflash2TestFixture(t)
	hlen := int(binary.LittleEndian.Uint64(validFile))
	base := string(validFile[8 : 8+hlen])
	cases := map[string]string{
		"wrong dtype": strings.Replace(base, `"BF16"`, `"F16"`, 1),
		"wrong shape": strings.Replace(base, `"shape":[2,2]`, `"shape":[4]`, 1),
		"missing key": strings.Replace(base, `"a":`, `"other":`, 1),
		"extra key":   strings.Replace(base, `"b":`, `"extra":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]},"b":`, 1),
		"overlap":     strings.Replace(base, `[8,12]`, `[6,10]`, 1),
		"gap":         strings.Replace(base, `[8,12]`, `[10,14]`, 1),
		"odd offset":  strings.Replace(base, `[8,12]`, `[7,11]`, 1),
		"negative":    strings.Replace(base, `[0,8]`, `[-1,7]`, 1),
		"short range": strings.Replace(base, `[0,8]`, `[0,6]`, 1),
		"duplicates":  strings.Replace(base, `"dtype":"BF16"`, `"dtype":"F16","dtype":"BF16"`, 1),
		"unknown":     strings.Replace(base, `"shape":[2,2]`, `"unknown":1,"shape":[2,2]`, 1),
		"metadata":    strings.Replace(base, `{"format":"pt"}`, `{"format":1}`, 1),
	}
	for name, header := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			file, c := dflash2TestFile(t, header, make([]byte, 12))
			closer := &dflash2TestCloser{}
			w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), closer, c)
			if w != nil {
				_ = w.Close()
			}
			if err == nil || closer.calls.Load() != 1 {
				t.Fatalf("invalid header admitted or ownership leaked: err=%v closes=%d", err, closer.calls.Load())
			}
		})
	}
	for _, suffix := range [][]byte{{0}, {0, 0}} {
		file, c := dflash2TestFile(t, base, make([]byte, 12+len(suffix)))
		if w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), nil, c); err == nil {
			_ = w.Close()
			t.Fatal("unclaimed trailing bytes accepted")
		}
	}
	file, c := dflash2TestFixture(t)
	c.shapes["a"] = []int{int(^uint(0) >> 1), 2}
	header, err := json.Marshal(map[string]stEntry{
		"a": {Dtype: "BF16", Shape: c.shapes["a"], DataOffsets: []int{0, 8}},
		"b": {Dtype: "BF16", Shape: []int{2}, DataOffsets: []int{8, 12}},
	})
	if err != nil {
		t.Fatal(err)
	}
	file, shaped := dflash2TestFile(t, string(header), make([]byte, 12))
	shaped.shapes = c.shapes
	if w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), nil, shaped); err == nil {
		_ = w.Close()
		t.Fatal("overflowing shape accepted")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2WeightsRejectSizeHashAndHeaderBudget(t *testing.T) {
	t.Parallel()
	file, c := dflash2TestFixture(t)
	// An inconsistent source must not authenticate a different header than the
	// one parsed. Old parse-then-full-reread hashing would accept this source.
	headerLength := int(binary.LittleEndian.Uint64(file))
	changed := append([]byte(nil), file...)
	copy(changed[8:8+headerLength], strings.Replace(string(file[8:8+headerLength]), `"pt"`, `"np"`, 1))
	changing := dflash2TestReaderAt(func(p []byte, off int64) (int, error) {
		if off == 8 {
			return bytes.NewReader(changed).ReadAt(p, off)
		}
		return bytes.NewReader(file).ReadAt(p, off)
	})
	if w, err := newDFlash2Weights(context.Background(), changing, int64(len(file)), nil, c); err == nil || !strings.Contains(err.Error(), "SHA256 mismatch") {
		if w != nil {
			_ = w.Close()
		}
		t.Fatalf("parsed header was not authenticated: %v", err)
	}
	file[len(file)-1] ^= 1
	if w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), nil, c); err == nil {
		_ = w.Close()
		t.Fatal("wrong payload digest accepted")
	}
	var calls int
	reader := dflash2TestReaderAt(func(p []byte, off int64) (int, error) {
		calls++
		if len(p) != binary.Size(uint64(0)) || off != 0 {
			t.Fatalf("read beyond header-length budget: len=%d off=%d", len(p), off)
		}
		binary.LittleEndian.PutUint64(p, dflash2MaxHeaderBytes+1)
		return len(p), nil
	})
	c.fileBytes = dflash2MaxHeaderBytes + 16
	if w, err := newDFlash2Weights(context.Background(), reader, c.fileBytes, nil, c); err == nil || calls != 1 {
		if w != nil {
			_ = w.Close()
		}
		t.Fatalf("oversized header err=%v reads=%d", err, calls)
	}
	calls = 0
	if _, err := newDFlash2Weights(context.Background(), reader, c.fileBytes-1, nil, c); err == nil || calls != 0 {
		t.Fatalf("wrong size read source: err=%v calls=%d", err, calls)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2WeightsCancellationAndReadFailure(t *testing.T) {
	t.Parallel()
	file, c := dflash2TestFixture(t)
	for _, duringHash := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		closer := &dflash2TestCloser{}
		if !duringHash {
			cancel()
		}
		reader := dflash2TestReaderAt(func(p []byte, off int64) (int, error) {
			n, err := bytes.NewReader(file).ReadAt(p, off)
			if off >= 8+int64(binary.LittleEndian.Uint64(file)) {
				cancel()
			}
			return n, err
		})
		w, err := newDFlash2Weights(ctx, reader, int64(len(file)), closer, c)
		cancel()
		if w != nil {
			_ = w.Close()
		}
		if !errors.Is(err, context.Canceled) || closer.calls.Load() != 1 {
			t.Fatalf("duringHash=%v err=%v closes=%d", duringHash, err, closer.calls.Load())
		}
	}
	var fail bool
	var cancelRead context.CancelFunc
	reader := dflash2TestReaderAt(func(p []byte, off int64) (int, error) {
		if fail {
			copy(p, file[off:off+1])
			return 1, io.ErrUnexpectedEOF
		}
		n, err := bytes.NewReader(file).ReadAt(p, off)
		if cancelRead != nil {
			cancelRead()
		}
		return n, err
	})
	w, err := newDFlash2Weights(context.Background(), reader, int64(len(file)), nil, c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	dst := []float32{77, 78}
	fail = true
	if err := w.ReadBF16(context.Background(), "a", 0, dst); err == nil || dst[0] != 77 || dst[1] != 78 {
		t.Fatalf("failed read changed destination: %v %v", dst, err)
	}
	fail = false
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cancelRead = cancel
	if err := w.ReadBF16(ctx, "a", 0, dst); !errors.Is(err, context.Canceled) || dst[0] != 77 || dst[1] != 78 {
		t.Fatalf("cancelled read changed destination: %v %v", dst, err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestDFlash2WeightsConcurrentReaders(t *testing.T) {
	t.Parallel()
	file, c := dflash2TestFixture(t)
	closer := &dflash2TestCloser{}
	w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), closer, c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil || closer.calls.Load() != 1 {
			t.Errorf("close err=%v count=%d", err, closer.calls.Load())
		}
	})
	for i := 0; i < 8; i++ {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			t.Parallel()
			dst := []float32{0, 0}
			if err := w.ReadBF16(context.Background(), "a", 0, dst); err != nil || dst[0] != 1 || dst[1] != -2.5 {
				t.Fatalf("span=%v err=%v", dst, err)
			}
		})
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestDFlash2WeightsCloseWaitsForRead(t *testing.T) {
	t.Parallel()
	file, c := dflash2TestFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	var block atomic.Bool
	reader := dflash2TestReaderAt(func(p []byte, off int64) (int, error) {
		if block.Load() {
			close(entered)
			<-release
		}
		return bytes.NewReader(file).ReadAt(p, off)
	})
	closeError := errors.New("fixture close failure")
	closer := &dflash2TestCloser{err: closeError}
	w, err := newDFlash2Weights(context.Background(), reader, int64(len(file)), closer, c)
	if err != nil {
		t.Fatal(err)
	}
	// Release precedes cleanup Close even on an assertion failure.
	t.Cleanup(func() { _ = w.Close() })
	t.Cleanup(unblock)
	block.Store(true)
	readDone, closeDone := make(chan error, 1), make(chan error, 1)
	go func() { readDone <- w.ReadBF16(context.Background(), "a", 0, make([]float32, 2)) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("reader did not enter")
	}
	go func() { closeDone <- w.Close() }()
	select {
	case err := <-closeDone:
		t.Fatalf("Close passed in-flight read: %v", err)
	case <-time.After(300 * time.Millisecond):
	}
	unblock()
	select {
	case err := <-readDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("reader did not finish")
	}
	select {
	case err := <-closeDone:
		if err != closeError {
			t.Fatalf("close error identity: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not finish")
	}
	if err := w.Close(); err != closeError || closer.calls.Load() != 1 {
		t.Fatalf("repeated Close err=%v count=%d", err, closer.calls.Load())
	}
}
