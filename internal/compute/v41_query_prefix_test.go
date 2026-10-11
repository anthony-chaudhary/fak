package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
)

func queryPrefixMetadataFixture() V41QueryPrefixInputs {
	return V41QueryPrefixInputs{
		Collapse: make([]float32, 193), AttentionGainHost: make([]float32, 193), QueryGainHost: make([]float32, 65),
		AttentionGain: Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{193}},
		WQA:           Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{65, 193}},
		QueryGain:     Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{65}},
		Epsilon:       1e-6,
	}
}

func queryPrefixStatusRecord(count int, site, index uint32) []byte {
	raw := make([]byte, 16)
	binary.LittleEndian.PutUint32(raw[0:4], 0x42463143)
	binary.LittleEndian.PutUint32(raw[4:8], site)
	binary.LittleEndian.PutUint32(raw[8:12], index)
	binary.LittleEndian.PutUint32(raw[12:16], uint32(count))
	return raw
}

// Metadata validation does not establish live ownership or native admission.
// fak-test:runtime fast est=20ms lane=default
func TestV41QueryPrefixPayload(t *testing.T) {
	in := queryPrefixMetadataFixture()
	for i, bits := range []uint32{0, 0x80000000, 0x00000001, 0x80400000, 0xbf800000} {
		in.AttentionGainHost[i], in.QueryGainHost[i] = math.Float32frombits(bits), math.Float32frombits(bits)
	}
	// Input validity belongs to the device status boundary, not a host scan.
	in.Collapse[0] = math.Float32frombits(0x7fffffff)
	if h, r, err := validateV41QueryPrefix(in); err != nil || h != 193 || r != 65 {
		t.Fatalf("valid metadata or finite gains rejected: h=%d r=%d err=%v", h, r, err)
	}
	if math.Float32bits(in.Collapse[0]) != 0x7fffffff {
		t.Fatal("validator changed the input before device validation")
	}
	for _, tc := range []struct {
		name, stage string
		change      func(*V41QueryPrefixInputs)
	}{
		{"matrix-dtype", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Dtype = F16 }},
		{"matrix-layout", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Layout = ColMajor }},
		{"matrix-quant", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Quant = &QuantSpec{} }},
		{"matrix-rank", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Shape = []int{65, 193, 1} }},
		{"zero-height", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Shape = []int{65, 0} }},
		{"negative-rank", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Shape = []int{-1, 193} }},
		{"product-overflow", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Shape = []int{46341, 46341} }},
		{"host-byte-overflow", "payload", func(x *V41QueryPrefixInputs) { x.WQA.Shape = []int{1, math.MaxInt} }},
		{"collapse-length", "payload", func(x *V41QueryPrefixInputs) { x.Collapse = x.Collapse[:192] }},
		{"attention-host-length", "payload", func(x *V41QueryPrefixInputs) { x.AttentionGainHost = x.AttentionGainHost[:192] }},
		{"query-host-length", "payload", func(x *V41QueryPrefixInputs) { x.QueryGainHost = x.QueryGainHost[:64] }},
		{"attention-shape", "payload", func(x *V41QueryPrefixInputs) { x.AttentionGain.Shape = []int{1, 193} }},
		{"attention-dtype", "payload", func(x *V41QueryPrefixInputs) { x.AttentionGain.Dtype = F16 }},
		{"attention-layout", "payload", func(x *V41QueryPrefixInputs) { x.AttentionGain.Layout = ColMajor }},
		{"attention-quant", "payload", func(x *V41QueryPrefixInputs) { x.AttentionGain.Quant = &QuantSpec{} }},
		{"query-shape", "payload", func(x *V41QueryPrefixInputs) { x.QueryGain.Shape = []int{64} }},
		{"query-dtype", "payload", func(x *V41QueryPrefixInputs) { x.QueryGain.Dtype = F16 }},
		{"query-layout", "payload", func(x *V41QueryPrefixInputs) { x.QueryGain.Layout = ColMajor }},
		{"query-quant", "payload", func(x *V41QueryPrefixInputs) { x.QueryGain.Quant = &QuantSpec{} }},
		{"epsilon-zero", "payload", func(x *V41QueryPrefixInputs) { x.Epsilon = 0 }},
		{"epsilon-negative", "payload", func(x *V41QueryPrefixInputs) { x.Epsilon = -1 }},
		{"epsilon-nan", "payload", func(x *V41QueryPrefixInputs) { x.Epsilon = float32(math.NaN()) }},
		{"epsilon-inf", "payload", func(x *V41QueryPrefixInputs) { x.Epsilon = float32(math.Inf(1)) }},
		{"attention-nan", "attention gain", func(x *V41QueryPrefixInputs) { x.AttentionGainHost[192] = float32(math.NaN()) }},
		{"attention-positive-inf", "attention gain", func(x *V41QueryPrefixInputs) { x.AttentionGainHost[0] = float32(math.Inf(1)) }},
		{"attention-negative-inf", "attention gain", func(x *V41QueryPrefixInputs) { x.AttentionGainHost[0] = float32(math.Inf(-1)) }},
		{"query-nan", "query gain", func(x *V41QueryPrefixInputs) { x.QueryGainHost[64] = float32(math.NaN()) }},
		{"query-positive-inf", "query gain", func(x *V41QueryPrefixInputs) { x.QueryGainHost[0] = float32(math.Inf(1)) }},
		{"query-negative-inf", "query gain", func(x *V41QueryPrefixInputs) { x.QueryGainHost[0] = float32(math.Inf(-1)) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := queryPrefixMetadataFixture()
			tc.change(&x)
			h, r, err := validateV41QueryPrefix(x)
			var stage *V41QueryPrefixStageError
			if !errors.As(err, &stage) || stage.Stage != tc.stage || h != 0 || r != 0 {
				t.Fatalf("invalid payload lost attribution: h=%d r=%d err=%v", h, r, err)
			}
			// No giant host row is allocated. Require the extent refusal itself
			// so a later host-length mismatch cannot mask its removal.
			if (tc.name == "product-overflow" || tc.name == "host-byte-overflow") && !strings.Contains(stage.Cause.Error(), "extent exceeds") {
				t.Fatalf("overflow was only caught by a later guard: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41QueryPrefixStatusPolicy(t *testing.T) {
	for site := uint32(3); site <= 6; site++ {
		count := 193
		if site >= 5 {
			count = 65
		}
		t.Run(fmt.Sprintf("site-%d", site), func(t *testing.T) {
			if err := decodeV41QueryPrefixStatus(queryPrefixStatusRecord(count, site, math.MaxUint32), count, site); err != nil {
				t.Fatal(err)
			}
			for _, index := range []uint32{0, uint32(count - 1)} {
				var fault *V41BF16CheckedArithmeticError
				err := decodeV41QueryPrefixStatus(queryPrefixStatusRecord(count, site, index), count, site)
				if !errors.As(err, &fault) || fault.Site != site || fault.Index != int(index) {
					t.Fatalf("lost arithmetic identity: %v", err)
				}
			}
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41QueryPrefixStatus(queryPrefixStatusRecord(count, site, uint32(count)), count, site); !errors.As(err, &protocol) {
				t.Fatalf("out-of-range index admitted: %v", err)
			}
		})
	}
	for _, site := range []uint32{0, 1, 2, 7, math.MaxUint32} {
		var protocol *V41BF16CheckedProtocolError
		if err := decodeV41QueryPrefixStatus(queryPrefixStatusRecord(1, site, math.MaxUint32), 1, site); !errors.As(err, &protocol) {
			t.Fatalf("matching forbidden prefix site %d admitted: %v", site, err)
		}
	}
	// A valid echoed record isolates wrapper policy from envelope validation.
	for _, site := range []uint32{1, 2} {
		if err := decodeV41BF16CheckedStatus(queryPrefixStatusRecord(1, site, math.MaxUint32), 1, site); err != nil {
			t.Fatalf("original final-norm site changed: %v", err)
		}
		if bytes, err := validateV41BF16Checked(Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1}}, site); err != nil || bytes != 4 {
			t.Fatalf("original final-norm payload policy changed: bytes=%d err=%v", bytes, err)
		}
	}
	var protocol *V41BF16CheckedProtocolError
	if err := decodeV41BF16CheckedStatus(queryPrefixStatusRecord(1, 3, math.MaxUint32), 1, 3); !errors.As(err, &protocol) {
		t.Fatalf("prefix widened original final-norm policy: %v", err)
	}
	if _, err := validateV41BF16Checked(Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1}}, 3); err == nil {
		t.Fatal("prefix widened original final-norm payload policy")
	}
	maxCount := min(math.MaxInt32, math.MaxInt/4)
	for _, site := range []uint32{3, 4, 5, 6} {
		if err := decodeV41QueryPrefixStatus(queryPrefixStatusRecord(maxCount, site, math.MaxUint32), maxCount, site); err != nil {
			t.Fatalf("allocation-free exact bound rejected: %v", err)
		}
		for _, n := range []int{-1, 0, maxCount + 1, math.MaxInt} {
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41QueryPrefixStatus(queryPrefixStatusRecord(n, site, math.MaxUint32), n, site); !errors.As(err, &protocol) {
				t.Fatalf("invalid echoed count %d at site%d admitted: %v", n, site, err)
			}
		}
	}
}

// These are decoder witnesses, not a substitute for four fresh native fills.
// fak-test:runtime fast est=20ms lane=default
func TestV41QueryPrefixStatusPrecedence(t *testing.T) {
	clean := func() [4][]byte {
		return [4][]byte{queryPrefixStatusRecord(193, 3, math.MaxUint32), queryPrefixStatusRecord(193, 4, math.MaxUint32), queryPrefixStatusRecord(65, 5, math.MaxUint32), queryPrefixStatusRecord(65, 6, math.MaxUint32)}
	}
	if err := decodeV41QueryPrefixStatuses(clean(), 193, 65); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]int{{0, 1}, {2, 3}} {
		raw := clean()
		count := 193
		if pair[0] == 2 {
			count = 65
		}
		binary.LittleEndian.PutUint32(raw[pair[0]][8:12], uint32(count-1))
		binary.LittleEndian.PutUint32(raw[pair[1]][8:12], 0)
		var fault *V41BF16CheckedArithmeticError
		err := decodeV41QueryPrefixStatuses(raw, 193, 65)
		if !errors.As(err, &fault) || fault.Site != uint32(pair[0]+3) || fault.Index != count-1 {
			t.Fatalf("lowest index incorrectly overrode logical site: %v", err)
		}
	}
	raw := clean()
	binary.LittleEndian.PutUint32(raw[0][8:12], 0)
	raw[1] = raw[1][:15]
	binary.LittleEndian.PutUint32(raw[2][0:4], 0)
	firstMalformed := decodeV41QueryPrefixStatus(raw[1], 193, 4)
	var protocol *V41BF16CheckedProtocolError
	if err := decodeV41QueryPrefixStatuses(raw, 193, 65); !errors.As(err, &protocol) || err.Error() != firstMalformed.Error() {
		t.Fatalf("protocol priority or fixed malformed order changed: %v", err)
	}
	for _, tc := range []struct {
		name   string
		change func([]byte) []byte
	}{
		{"magic", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[0:4], 0); return b }},
		{"site", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[4:8], 3); return b }},
		{"count", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[12:16], 193); return b }},
		{"uninitialized", func(b []byte) []byte {
			for i := range b {
				b[i] = 255
			}
			return b
		}},
		{"truncated", func(b []byte) []byte { return b[:15] }},
		{"oversized", func(b []byte) []byte { return append(b, 0) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := clean()
			binary.LittleEndian.PutUint32(raw[0][8:12], 192)
			raw[3] = tc.change(raw[3])
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41QueryPrefixStatuses(raw, 193, 65); !errors.As(err, &protocol) {
				t.Fatalf("later protocol failure masked by early arithmetic: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41QueryPrefixResultContract(t *testing.T) {
	bits := []uint32{0, 0x80000000, 0x00010000, 0x80400000, 0x3f800000, 0xbf820000, 0x7f7f0000, 0xff7f0000}
	row := func() []float32 {
		x := make([]float32, len(bits))
		for i, b := range bits {
			x[i] = math.Float32frombits(b)
		}
		return x
	}
	out := V41QueryPrefixResult{Collapsed: row(), QLat: row()}
	if err := validateV41QueryPrefixResult(out, len(bits), len(bits)); err != nil {
		t.Fatal(err)
	}
	for i, b := range bits {
		if math.Float32bits(out.Collapsed[i]) != b || math.Float32bits(out.QLat[i]) != b {
			t.Fatal("result validator mutated owned row bits")
		}
	}
	for _, which := range []string{"collapsed", "query"} {
		for _, bad := range []uint32{0x7f800000, 0xff800000, 0x7fc00000, 0x7fffffff, 0x3f800001, 0x80000001} {
			x := V41QueryPrefixResult{Collapsed: row(), QLat: row()}
			target := x.Collapsed
			if which == "query" {
				target = x.QLat
			}
			target[len(target)-1] = math.Float32frombits(bad)
			var stage *V41QueryPrefixStageError
			if err := validateV41QueryPrefixResult(x, len(bits), len(bits)); !errors.As(err, &stage) || stage.Stage != "output" {
				t.Fatalf("malformed %s word %08x admitted: %v", which, bad, err)
			}
			if math.Float32bits(target[len(target)-1]) != bad {
				t.Fatal("failure repaired the result in place")
			}
		}
	}
	for _, x := range []V41QueryPrefixResult{{Collapsed: row()}, {QLat: row()}, {Collapsed: row()[:1], QLat: row()}} {
		var stage *V41QueryPrefixStageError
		if err := validateV41QueryPrefixResult(x, len(bits), len(bits)); !errors.As(err, &stage) || stage.Stage != "output" {
			t.Fatalf("partial/wrong-shape result admitted: %v", err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41QueryPrefixErrorCause(t *testing.T) {
	primary := errors.New("primary transport")
	err := &V41QueryPrefixStageError{Stage: "status-5", Cause: primary}
	if !errors.Is(err, primary) || err.Unwrap() != primary {
		t.Fatal("stage wrapper replaced the exact primary")
	}
	busy := &V41QueryPrefixBusyError{State: "batch"}
	var got *V41QueryPrefixBusyError
	if !errors.As(fmt.Errorf("prefix: %w", busy), &got) || got != busy {
		t.Fatal("busy identity was lost through an outer wrapper")
	}
}
