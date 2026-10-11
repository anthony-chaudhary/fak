package model

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41Q2KEmbeddingTestRaw builds a Q2_K table that exercises the zero-sign and
// rounding corners: negative d, zero scale/min nibbles, an f16 subnormal min
// and both signs of a zero min.
func v41Q2KEmbeddingTestRaw(vocab, hidden int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	blocks := vocab * hidden / qkK
	raw := make([]byte, blocks*q2kBlockBytes)
	halves := []uint16{0x3c00, 0xbc00, 0x2e66, 0xb555, 0x0001, 0x8000, 0x0000, 0x3555, 0x1400}
	for b := 0; b < blocks; b++ {
		blk := raw[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		rng.Read(blk[:80])
		if b%3 == 0 {
			for i := 0; i < 16; i++ {
				blk[i] &= 0xf0 // zero scale nibbles: dl is a signed zero
			}
		}
		binary.LittleEndian.PutUint16(blk[80:], halves[rng.Intn(len(halves))])
		binary.LittleEndian.PutUint16(blk[82:], halves[rng.Intn(len(halves))])
	}
	return raw
}

// v41Q2KEmbeddingShaderOracle mirrors the shader's per-element index mapping and
// operation order, independent of the host super-block loop it must equal.
func v41Q2KEmbeddingShaderOracle(raw []byte, hidden int, ids []int) []float32 {
	out := make([]float32, len(ids)*hidden)
	rowBytes := hidden / qkK * q2kBlockBytes
	for row, id := range ids {
		for col := 0; col < hidden; col++ {
			base := id*rowBytes + col/qkK*q2kBlockBytes
			e := col % qkK
			half, r := e>>7, e&127
			j, h, l := r>>5, (r>>4)&1, r&15
			sc := raw[base+half*8+j*2+h]
			code := (raw[base+16+half*32+h*16+l] >> (2 * j)) & 3
			d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+80:])))
			m := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+82:])))
			dl, ml := d*float32(sc&15), m*float32(sc>>4)
			var v float32
			if code == 0 {
				v = 0 - ml
			} else {
				scaled := dl * float32(code)
				v = scaled - ml
			}
			out[row*hidden+col] = v
		}
	}
	return out
}

type v41Q2KEmbeddingTestBackend struct {
	*v41DenseTestBackend
	raw      []byte
	tables   int
	calls    [][]int
	outputs  map[compute.Buffer]bool
	fault    string
	failCall int
	cause    error
}

func (b *v41Q2KEmbeddingTestBackend) V41Q2KEmbeddingGatherAvailable() bool { return true }

func (b *v41Q2KEmbeddingTestBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt == compute.Q2_K {
		b.tables++
		if dt != x.Dtype {
			panic("Q2_K table upload changed dtype")
		}
		return x // the cpu-ref recorder keeps the packed host bytes resident
	}
	if dt == compute.F32 && len(x.Shape) == 2 && x.Shape[0] > 64 {
		panic("Q2_K embedding gather materialized an F32 table")
	}
	return b.v41DenseTestBackend.Upload(x, dt)
}

func (b *v41Q2KEmbeddingTestBackend) V41Q2KEmbeddingGather(table compute.Tensor, ids []int, hidden int) (compute.Tensor, error) {
	if table.Dtype != compute.Q2_K || len(table.Shape) != 2 || table.Shape[1] != hidden {
		return compute.Tensor{}, errors.New("invalid Q2_K gather contract")
	}
	b.calls = append(b.calls, append([]int(nil), ids...))
	if len(b.calls) == b.failCall && b.fault == "dispatch" {
		return compute.Tensor{}, b.cause
	}
	out := compute.NewF32(b, []int{len(ids), hidden}, v41Q2KEmbeddingShaderOracle(b.raw, hidden, ids))
	b.outputs[out.Buf()] = true
	return out, nil
}

func (b *v41Q2KEmbeddingTestBackend) Free(x compute.Tensor) {
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

// Refs #13668. A Q2_K token table gathers on the device without an F32 table:
// the panel equals the host Q2_K dequantization bit for bit (signed zeros
// included), the packed table is staged once, a selected failure fails closed,
// and other packed formats keep a typed device-only refusal.
// fak-test:runtime fast est=500ms lane=default
func TestV41Q2KEmbeddingGatherDeviceDispatch(t *testing.T) {
	const vocab, hidden = 11, 512
	raw := v41Q2KEmbeddingTestRaw(vocab, hidden, 13668)
	newModel := func(t *testing.T, embed *Q2KEmbedding) *Model {
		t.Helper()
		return &Model{Cfg: Config{HiddenSize: hidden, VocabSize: vocab}, Q2KEmbedding: embed}
	}
	embed, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{3, 0, 10, 3, 7}

	t.Run("matches-host-dequant", func(t *testing.T) {
		m := newModel(t, embed)
		b := &v41Q2KEmbeddingTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), raw: raw, outputs: map[compute.Buffer]bool{}}
		s := &Session{M: m, Backend: b}
		t.Cleanup(s.Close)
		ops := s.v41DeviceGatherOpsFunc()
		if ops.embedGather == nil {
			t.Fatal("Q2_K table did not bind the device gather")
		}
		want, err := m.v41EmbeddingPanel(ids)
		if err != nil {
			t.Fatal(err)
		}
		zeros := 0
		for _, v := range want {
			if v == 0 {
				zeros++
			}
		}
		if zeros == 0 {
			t.Fatal("fixture lacks the signed-zero corners")
		}
		for call := 0; call < 2; call++ {
			got, err := m.v41EmbeddingPanelOn(ids, ops)
			if err != nil {
				t.Fatal(err)
			}
			v41TailRoPECheckBits(t, "Q2_K device panel", got, want)
		}
		if b.tables != 1 || len(b.calls) != 2 || len(b.outputs) != 0 {
			t.Fatalf("staging/ownership: tables=%d calls=%d live=%d", b.tables, len(b.calls), len(b.outputs))
		}
	})

	t.Run("selected-failure", func(t *testing.T) {
		m := newModel(t, embed)
		b := &v41Q2KEmbeddingTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), raw: raw, outputs: map[compute.Buffer]bool{},
			fault: "dispatch", failCall: 1, cause: &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}}
		s := &Session{M: m, Backend: b}
		t.Cleanup(s.Close)
		got, err := m.v41EmbeddingPanelOn(ids, s.v41DeviceGatherOpsFunc())
		var closed *BackendForwardOperationError
		if got != nil || !errors.As(err, &closed) || closed.Path != "v41-embedding-gather" || !errors.Is(err, b.cause) || !s.BackendSessionClosed() {
			t.Fatalf("selected Q2_K gather failure lost identity: %v", err)
		}
	})

	t.Run("other-packed-formats-refuse-under-device-only", func(t *testing.T) {
		q4, err := NewQ4KEmbedding(makeTestQ4KRaw(vocab, hidden), vocab, hidden)
		if err != nil {
			t.Fatal(err)
		}
		m := newModel(t, q4)
		b := &v41Q2KEmbeddingTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), raw: raw, outputs: map[compute.Buffer]bool{}}
		s := &Session{M: m, Backend: b}
		t.Cleanup(s.Close)
		ops := s.v41DeviceGatherOpsFunc()
		if ops.embedGather != nil {
			t.Fatal("a Q4_K table selected the Q2_K device gather")
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		got, err := m.v41EmbeddingPanelOn(ids, ops)
		var closed *BackendForwardOperationError
		if got != nil || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Path != "device-only" || len(b.calls) != 0 {
			t.Fatalf("Q4_K table under device-only did not refuse: %v", err)
		}
	})
}
