package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41AttentionBF16SitePolicy(t *testing.T) {
	record := func(count int, site, index uint32) []byte {
		raw := make([]byte, 16)
		binary.LittleEndian.PutUint32(raw[0:4], 0x42463143)
		binary.LittleEndian.PutUint32(raw[4:8], site)
		binary.LittleEndian.PutUint32(raw[8:12], index)
		binary.LittleEndian.PutUint32(raw[12:16], uint32(count))
		return raw
	}
	for _, site := range []uint32{0, 1, 2, 3, 4, 5, 6, math.MaxUint32} {
		input := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{193}}
		_, metadataErr := validateV41AttentionBF16(input, site)
		raw := record(193, site, math.MaxUint32)
		err := decodeV41AttentionBF16Status(raw, 193, site)
		attention := site == 3 || site == 4
		if (metadataErr == nil) != attention || (err == nil) != attention {
			t.Fatalf("site %d metadata=%v status=%v", site, metadataErr, err)
		}
		if (decodeV41BF16CheckedStatus(raw, 193, site) == nil) != (site == 1 || site == 2) {
			t.Fatalf("ordinary attention changed final-norm site policy at %d", site)
		}
		if (decodeV41QueryPrefixStatus(raw, 193, site) == nil) != (site >= 3 && site <= 6) {
			t.Fatalf("ordinary attention changed prefix site policy at %d", site)
		}
	}
	maxCount := min(math.MaxInt32, math.MaxInt/4)
	for _, site := range []uint32{3, 4} {
		for _, n := range []int{1, 63, 64, 65, 193, 5120, maxCount} {
			input := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{n}}
			if bytes, err := validateV41AttentionBF16(input, site); err != nil || bytes != n*4 {
				t.Fatalf("valid metadata count=%d site=%d bytes=%d err=%v", n, site, bytes, err)
			}
			if err := decodeV41AttentionBF16Status(record(n, site, math.MaxUint32), n, site); err != nil {
				t.Fatalf("valid envelope count=%d site=%d: %v", n, site, err)
			}
			var arithmetic *V41BF16CheckedArithmeticError
			if err := decodeV41AttentionBF16Status(record(n, site, uint32(n-1)), n, site); !errors.As(err, &arithmetic) || arithmetic.Site != site || arithmetic.Index != n-1 {
				t.Fatalf("last-element attribution count=%d site=%d: %v", n, site, err)
			}
		}
		for _, n := range []int{0, -1, maxCount + 1} {
			input := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{n}}
			if _, err := validateV41AttentionBF16(input, site); err == nil {
				t.Fatalf("invalid count %d admitted", n)
			}
			// Matching magic/site/count prevents a different envelope check from
			// masking a missing issued-count guard.
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41AttentionBF16Status(record(n, site, math.MaxUint32), n, site); !errors.As(err, &protocol) {
				t.Fatalf("invalid issued count %d admitted: %v", n, err)
			}
		}
		for _, tc := range []struct {
			name   string
			mutate func([]byte) []byte
		}{
			{"magic", func(b []byte) []byte { b[0] ^= 1; return b }},
			{"other-site", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[4:8], 7-site); return b }},
			{"count", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[12:16], 194); return b }},
			{"index-equals-count", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[8:12], 193); return b }},
			{"short", func(b []byte) []byte { return b[:15] }},
			{"long", func(b []byte) []byte { return append(b, 0) }},
			{"initial-all-ones", func(b []byte) []byte {
				for i := range b {
					b[i] = 255
				}
				return b
			}},
		} {
			var protocol *V41BF16CheckedProtocolError
			if err := decodeV41AttentionBF16Status(tc.mutate(record(193, site, 0)), 193, site); !errors.As(err, &protocol) {
				t.Fatalf("%s site=%d did not take protocol precedence: %v", tc.name, site, err)
			}
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41AttentionBF16Metadata(t *testing.T) {
	for _, input := range []Tensor{
		{},
		{Dtype: F16, Layout: RowMajor, Shape: []int{8}},
		{Dtype: F32, Layout: ColMajor, Shape: []int{8}},
		{Dtype: F32, Layout: RowMajor, Shape: []int{1, 8}},
		{Dtype: F32, Layout: RowMajor, Shape: []int{8}, Quant: &QuantSpec{}},
	} {
		if _, err := validateV41AttentionBF16(input, 3); err == nil {
			t.Fatalf("invalid metadata admitted: %+v", input)
		}
	}
}
