// Default suites: go test ./internal/model and go test ./... (also under -race).
// Focused suite: go test ./internal/model -run '^TestRetainMappedQ4KResidency' -count=1.
package model

import (
	"bytes"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type mappedCPUAcceptanceReader struct {
	data     []byte
	reads    int
	fail     error
	shortNil bool
}

func (r *mappedCPUAcceptanceReader) ReadAt(dst []byte, off int64) (int, error) {
	r.reads++
	if off < 0 || off > int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(dst, r.data[off:])
	if r.fail != nil {
		return n / 2, r.fail
	}
	if r.shortNil {
		return n / 2, nil
	}
	if n != len(dst) {
		return n, io.EOF
	}
	return n, nil
}

func mappedCPUAcceptanceTensor(raw []byte, reader io.ReaderAt, span []byte, mappedOffset int) *q4kTensor {
	return &q4kTensor{out: len(raw) / q4kBlockBytes, in: qkK, nblk: 1, lazy: &LazyQ4KRange{
		Reader: reader, Offset: 64, Bytes: len(raw), MappedSpan: span, MappedOffset: mappedOffset,
	}}
}

func assertMappedCPUAcceptanceLogits(t *testing.T, got *q4kTensor, raw []byte) {
	t.Helper()
	if !bytes.Equal(got.raw, raw) {
		t.Fatal("retained CPU payload differs from checkpoint Q4_K bytes")
	}
	x := make([]float32, qkK)
	for i := range x {
		x[i] = float32(i%19-9) / 16
	}
	want := q4kMatRows(quantizeQ4KFromRaw(raw, got.out, got.in), x)
	logits := q4kMatRows(got, x)
	var nonzero bool
	for i := range want {
		if math.IsNaN(float64(logits[i])) || math.IsInf(float64(logits[i]), 0) || math.Float32bits(logits[i]) != math.Float32bits(want[i]) {
			t.Fatalf("CPU Q4_K logit %d=%v, want bit-identical owned-byte reference %v", i, logits[i], want[i])
		}
		nonzero = nonzero || logits[i] != 0
	}
	if !nonzero {
		t.Fatal("nonzero Q4_K fixture produced only zero logits")
	}
}

func assertMappedCPUAcceptanceReport(t *testing.T, m *Model, raw []byte) {
	t.Helper()
	reference := &Model{q4kw: map[string]*q4kTensor{
		"lm_head.weight": quantizeQ4KFromRaw(raw, len(raw)/q4kBlockBytes, qkK),
	}}
	got, want := m.ResidentReport(), reference.ResidentReport()
	if *got != *want || got.Q4KBytes != int64(len(raw)) || got.TotalResidentBytes != int64(len(raw)) || got.DecodeBytesPerToken != int64(len(raw)) {
		t.Fatalf("retained CPU accounting=%+v, want owned-resident accounting=%+v despite preserved lazy metadata", got, want)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRetainMappedQ4KResidencyExactViewAndCPULogitParity(t *testing.T) {
	raw := buildRawQ4K(t, 4, qkK, 13569)
	span := makePageAlignedResidentBytes(os.Getpagesize())
	copy(span[32:], raw)
	reader := &mappedCPUAcceptanceReader{data: append(make([]byte, 64), raw...), fail: errors.New("mapped path must not read")}
	qt := mappedCPUAcceptanceTensor(raw, reader, span, 32)
	lazy := qt.lazy
	m := &Model{q4kw: map[string]*q4kTensor{"lm_head.weight": qt}, q4khead: qt}
	m.SetWeightCloser(closeFunc(func() error { return nil }))
	t.Cleanup(func() { _ = m.CloseWeights() })
	if report := m.ResidentReport(); report.Q4KBytes != 0 || report.TotalResidentBytes != 0 || report.DecodeBytesPerToken != 0 {
		t.Fatalf("lazy range without retained CPU bytes was charged as resident: %+v", report)
	}

	if err := m.RetainMappedQ4KResidency(); err != nil {
		t.Fatal(err)
	}
	if len(qt.raw) != len(raw) || &qt.raw[0] != &span[32] || reader.reads != 0 {
		t.Fatalf("mapped retention copied or read: raw bytes=%d reads=%d", len(qt.raw), reader.reads)
	}
	if qt.lazy != lazy || qt.lazy.MappedOffset != 32 || &qt.lazy.MappedSpan[0] != &span[0] || qt.lazy.Offset != 64 {
		t.Fatal("mapped CPU retention discarded the native-buffer span/offset metadata")
	}
	assertMappedCPUAcceptanceLogits(t, qt, raw)
	assertMappedCPUAcceptanceReport(t, m, raw)
	if err := m.RetainMappedQ4KResidency(); err != nil {
		t.Fatalf("repeat retention: %v", err)
	}
	if reader.reads != 0 || &qt.raw[0] != &span[32] {
		t.Fatal("repeat retention changed the exact mapped view")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRetainMappedQ4KResidencyRejectedRangesUseOwnedCPUBytes(t *testing.T) {
	raw := buildRawQ4K(t, 4, qkK, 13570)
	page := os.Getpagesize()
	for _, tc := range []struct {
		name   string
		span   func([]byte) []byte
		offset int
	}{
		{name: "unmapped", span: func([]byte) []byte { return nil }},
		{name: "negative offset", span: func(b []byte) []byte { return b }, offset: -32},
		{name: "unaligned offset", span: func(b []byte) []byte { return b }, offset: 1},
		{name: "unaligned base", span: func(b []byte) []byte { return b[1:] }},
		{name: "non-page length", span: func(b []byte) []byte { return b[:page-1] }},
		{name: "range past end", span: func(b []byte) []byte { return b }, offset: page - 32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			span := makePageAlignedResidentBytes(page)
			reader := &mappedCPUAcceptanceReader{data: append(make([]byte, 64), raw...)}
			qt := mappedCPUAcceptanceTensor(raw, reader, tc.span(span), tc.offset)
			m := &Model{q4kw: map[string]*q4kTensor{"lm_head.weight": qt}}
			m.SetWeightCloser(closeFunc(func() error { return nil }))
			t.Cleanup(func() { _ = m.CloseWeights() })
			if err := m.RetainMappedQ4KResidency(); err != nil {
				t.Fatal(err)
			}
			if reader.reads != 1 || len(qt.raw) != len(raw) || &qt.raw[0] == &reader.data[64] {
				t.Fatalf("fallback must own one complete checkpoint read: reads=%d raw bytes=%d", reader.reads, len(qt.raw))
			}
			assertMappedCPUAcceptanceLogits(t, qt, raw)
			assertMappedCPUAcceptanceReport(t, m, raw)
			reader.data[64] ^= 0xff
			if !bytes.Equal(qt.raw, raw) {
				t.Fatal("owned fallback bytes still alias the reader's storage")
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRetainMappedQ4KResidencyReadFailuresAreAtomic(t *testing.T) {
	raw := buildRawQ4K(t, 4, qkK, 13571)
	dropped := errors.New("checkpoint read interrupted")
	for _, tc := range []struct {
		name     string
		fail     error
		shortNil bool
	}{
		{name: "partial read error", fail: dropped},
		{name: "short read without error", shortNil: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			goodReader := &mappedCPUAcceptanceReader{data: append(make([]byte, 64), raw...)}
			badReader := &mappedCPUAcceptanceReader{data: append(make([]byte, 64), raw...), fail: tc.fail, shortNil: tc.shortNil}
			good := mappedCPUAcceptanceTensor(raw, goodReader, nil, 0)
			bad := mappedCPUAcceptanceTensor(raw, badReader, nil, 0)
			goodLazy, badLazy := good.lazy, bad.lazy
			m := &Model{q4kw: map[string]*q4kTensor{"a.weight": good, "z.weight": bad}}
			m.SetWeightCloser(closeFunc(func() error { return nil }))
			t.Cleanup(func() { _ = m.CloseWeights() })
			err := m.RetainMappedQ4KResidency()
			if err == nil || (tc.fail != nil && !errors.Is(err, dropped)) {
				t.Fatalf("retention error=%v, want the failed checkpoint read", err)
			}
			if len(good.raw) != 0 || len(bad.raw) != 0 || good.lazy != goodLazy || bad.lazy != badLazy {
				t.Fatal("failed startup published partial CPU residency or changed lazy descriptors")
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRetainMappedQ4KResidencyRefusesNonStartupStates(t *testing.T) {
	for _, state := range []string{"active session", "closing", "closed", "bounded dense", "streamed experts"} {
		t.Run(state, func(t *testing.T) {
			raw := buildRawQ4K(t, 1, qkK, 13572)
			reader := &mappedCPUAcceptanceReader{data: append(make([]byte, 64), raw...)}
			qt := mappedCPUAcceptanceTensor(raw, reader, nil, 0)
			m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 8, NumLayers: 0})
			m.q4kw = map[string]*q4kTensor{"lm_head.weight": qt}
			m.SetWeightCloser(closeFunc(func() error { return nil }))
			t.Cleanup(func() { _ = m.CloseWeights() })
			switch state {
			case "active session", "closing":
				s := m.NewSession()
				t.Cleanup(s.Close)
				if state == "closing" {
					if err := m.CloseWeights(); err == nil {
						t.Fatal("close with an active session did not defer teardown")
					}
				}
			case "closed":
				if err := m.CloseWeights(); err != nil {
					t.Fatal(err)
				}
			case "bounded dense":
				m.denseResidentBoundBytes = 1 << 20
			case "streamed experts":
				m.SetExpertCheckpoint(NewExpertCheckpointTier(0))
			}
			if err := m.RetainMappedQ4KResidency(); err == nil {
				t.Fatalf("retention admitted %s", state)
			}
			if reader.reads != 0 || len(qt.raw) != 0 {
				t.Fatalf("refused %s still materialized checkpoint bytes", state)
			}
		})
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestRetainMappedQ4KResidencyCheckpointLifetimeAndCloseOrder(t *testing.T) {
	requireMappedQ4KSpanDarwin(t)
	raw := buildRawQ4K(t, 4, qkK, 13573)
	payload := make([]byte, os.Getpagesize())
	copy(payload[32:], raw)
	path := filepath.Join(t.TempDir(), "mapped-cpu.gguf")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	owner, err := openMappedQ4KSpan(path, 0, len(payload))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = owner.Close() })
	reader := &mappedCPUAcceptanceReader{fail: errors.New("mapping must remain live")}
	qt := mappedCPUAcceptanceTensor(raw, reader, owner.mapped, 32)
	m := &Model{q4kw: map[string]*q4kTensor{"lm_head.weight": qt}, q4khead: qt}
	var events []string
	originalRelease := releaseModelQ4KHandles
	releaseModelQ4KHandles = func(got *Model) {
		if got != m || len(qt.raw) == 0 || qt.lazy == nil {
			t.Error("native buffers must release while their CPU backing remains live")
		}
		events = append(events, "release-native")
	}
	t.Cleanup(func() { releaseModelQ4KHandles = originalRelease })
	m.SetWeightCloser(closeFunc(func() error {
		if qt.raw != nil || qt.lazy != nil {
			t.Error("checkpoint unmap saw retained CPU payload references")
		}
		events = append(events, "unmap")
		return owner.Close()
	}))
	t.Cleanup(func() { _ = m.CloseWeights() })
	if err := m.RetainMappedQ4KResidency(); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	assertMappedCPUAcceptanceLogits(t, qt, raw)
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(events, ","); got != "release-native,unmap" || owner.bytes() != nil {
		t.Fatalf("teardown=%q, want native release then one unmap with no retained owner view", got)
	}
}
