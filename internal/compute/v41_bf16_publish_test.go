package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41BF16PublishSitePolicy(t *testing.T) {
	record := func(count int, site, index uint32) []byte {
		raw := make([]byte, 16)
		binary.LittleEndian.PutUint32(raw[0:4], 0x42463143)
		binary.LittleEndian.PutUint32(raw[4:8], site)
		binary.LittleEndian.PutUint32(raw[8:12], index)
		binary.LittleEndian.PutUint32(raw[12:16], uint32(count))
		return raw
	}
	ordinary := map[uint32]bool{1: true, 2: true}
	for site := uint32(5); site <= 27; site++ {
		ordinary[site] = true
	}
	for _, site := range []uint32{0, 1, 2, 3, 4, 5, 6, 7, 16, 27, 28, 64, math.MaxUint32} {
		input := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{193}}
		_, metadataErr := validateV41BF16Publish(input, site)
		raw := record(193, site, math.MaxUint32)
		err := decodeV41BF16PublishStatus(raw, 193, site)
		if V41BF16PublishSite(site) != ordinary[site] || (metadataErr == nil) != ordinary[site] || (err == nil) != ordinary[site] {
			t.Fatalf("site %d policy=%t metadata=%v status=%v", site, V41BF16PublishSite(site), metadataErr, err)
		}
		// The ordinary entry must not widen the attention, final-norm or prefix policies.
		if (decodeV41AttentionBF16Status(raw, 193, site) == nil) != (site == 3 || site == 4) {
			t.Fatalf("ordinary publication changed attention site policy at %d", site)
		}
		if (decodeV41BF16CheckedStatus(raw, 193, site) == nil) != (site == 1 || site == 2) {
			t.Fatalf("ordinary publication changed final-norm site policy at %d", site)
		}
		if (decodeV41QueryPrefixStatus(raw, 193, site) == nil) != (site >= 3 && site <= 6) {
			t.Fatalf("ordinary publication changed prefix site policy at %d", site)
		}
	}
	for site := range ordinary {
		if name := v41BF16CheckedSiteName(site); strings.HasPrefix(name, "unknown") {
			t.Fatalf("ordinary site %d has no attribution name", site)
		}
		var arithmetic *V41BF16CheckedArithmeticError
		if err := decodeV41BF16PublishStatus(record(193, site, 192), 193, site); !errors.As(err, &arithmetic) || arithmetic.Site != site || arithmetic.Index != 192 {
			t.Fatalf("last-element attribution site=%d: %v", site, err)
		}
		var protocol *V41BF16CheckedProtocolError
		other := record(193, site, math.MaxUint32)
		binary.LittleEndian.PutUint32(other[4:8], site^1)
		if err := decodeV41BF16PublishStatus(other, 193, site); !errors.As(err, &protocol) {
			t.Fatalf("site %d accepted another site's record: %v", site, err)
		}
	}
	maxCount := min(math.MaxInt32, math.MaxInt/4)
	for _, n := range []int{0, -1, maxCount + 1} {
		if _, err := validateV41BF16Publish(Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{n}}, 7); err == nil {
			t.Fatalf("invalid count %d admitted", n)
		}
	}
	for _, input := range []Tensor{
		{},
		{Dtype: F16, Layout: RowMajor, Shape: []int{8}},
		{Dtype: F32, Layout: ColMajor, Shape: []int{8}},
		{Dtype: F32, Layout: RowMajor, Shape: []int{1, 8}},
		{Dtype: F32, Layout: RowMajor, Shape: []int{8}, Quant: &QuantSpec{}},
	} {
		if _, err := validateV41BF16Publish(input, 7); err == nil {
			t.Fatalf("invalid metadata admitted: %+v", input)
		}
	}
}
