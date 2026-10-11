package compute

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"testing"
)

// Explicit literals came from the independently reviewed integer vector packet.
// This checks oracle/literal consistency only. It does not invoke the production
// shader or establish device numerical behavior.
// fak-test:runtime fast est=20ms lane=default
func TestV41BF16CheckedBitContract(t *testing.T) {
	for _, tc := range []struct {
		input, rounded uint32
		valid          bool
	}{
		{0x00000000, 0x00000000, true},
		{0x80000000, 0x80000000, true},
		{0x00000001, 0x00000000, true},
		{0x80000001, 0x80000000, true},
		{0x00007fff, 0x00000000, true},
		{0x80007fff, 0x80000000, true},
		{0x00008000, 0x00000000, true},
		{0x80008000, 0x80000000, true},
		{0x00008001, 0x00010000, true},
		{0x80008001, 0x80010000, true},
		{0x00017fff, 0x00010000, true},
		{0x80017fff, 0x80010000, true},
		{0x00018000, 0x00020000, true},
		{0x80018000, 0x80020000, true},
		{0x00018001, 0x00020000, true},
		{0x80018001, 0x80020000, true},
		{0x007f7fff, 0x007f0000, true},
		{0x807f7fff, 0x807f0000, true},
		{0x007f8000, 0x00800000, true},
		{0x807f8000, 0x80800000, true},
		{0x007fffff, 0x00800000, true},
		{0x807fffff, 0x80800000, true},
		{0x00800000, 0x00800000, true},
		{0x80800000, 0x80800000, true},
		{0x3f807fff, 0x3f800000, true},
		{0x3f808000, 0x3f800000, true},
		{0x3f808001, 0x3f810000, true},
		{0xbf807fff, 0xbf800000, true},
		{0xbf808000, 0xbf800000, true},
		{0xbf808001, 0xbf810000, true},
		{0x3f817fff, 0x3f810000, true},
		{0x3f818000, 0x3f820000, true},
		{0x3f818001, 0x3f820000, true},
		{0xbf817fff, 0xbf810000, true},
		{0xbf818000, 0xbf820000, true},
		{0xbf818001, 0xbf820000, true},
		{0x7f7f0000, 0x7f7f0000, true},
		{0x7f7f7fff, 0x7f7f0000, true},
		{0x7f7f8000, 0x7f800000, false},
		{0x7f7f8001, 0x7f800000, false},
		{0x7f7fffff, 0x7f800000, false},
		{0xff7f0000, 0xff7f0000, true},
		{0xff7f7fff, 0xff7f0000, true},
		{0xff7f8000, 0xff800000, false},
		{0xff7f8001, 0xff800000, false},
		{0xff7fffff, 0xff800000, false},
		{0x7f800000, 0x7f800000, false},
		{0xff800000, 0xff800000, false},
		{0x7f800001, 0x7f800000, false},
		{0xff800001, 0xff800000, false},
		{0x7fbfffff, 0x7fc00000, false},
		{0xffbfffff, 0xffc00000, false},
		{0x7fc00000, 0x7fc00000, false},
		{0xffc00000, 0xffc00000, false},
		{0x7fff7fff, 0x7fff0000, false},
		{0xffff7fff, 0xffff0000, false},
		{0x7fff8000, 0x80000000, false},
		{0xffff8000, 0x00000000, false},
		{0x7fffffff, 0x80000000, false},
		{0xffffffff, 0x00000000, false},
	} {
		upper, lower := uint16(tc.input>>16), uint16(tc.input)
		if lower > 0x8000 || (lower == 0x8000 && upper&1 != 0) {
			upper++
		}
		rounded := uint32(upper) << 16
		if rounded != tc.rounded {
			t.Fatalf("literal %08x rounded %08x want %08x", tc.input, rounded, tc.rounded)
		}
		valid := tc.input&0x7f800000 != 0x7f800000 && rounded&0x7f800000 != 0x7f800000
		if valid != tc.valid {
			t.Fatalf("checked classification changed for %08x", tc.input)
		}
	}
	// Carry bounds include the largest native count without allocating its row.
	for _, n := range []uint64{1, 63, 64, 65, 127, 128, 129, math.MaxInt32} {
		groups := (n + 63) / 64
		stride := groups * 64
		if stride > 1<<31 || n-1+stride > math.MaxUint32-1 {
			t.Fatal("grid-stride bound would wrap")
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41BF16CheckedStatus(t *testing.T) {
	for _, tc := range []struct {
		name, hex, verdict string
		site               uint32
		count              int
	}{
		{"success-input-site", "4331464201000000ffffffffc1000000", "success", 1, 193},
		{"success-output-site", "4331464202000000ffffffffc1000000", "success", 2, 193},
		{"in-range-fault-0", "433146420100000000000000c1000000", "arithmetic_error", 1, 193},
		{"in-range-fault-64", "433146420100000040000000c1000000", "arithmetic_error", 1, 193},
		{"in-range-fault-output-site", "433146420200000040000000c1000000", "arithmetic_error", 2, 193},
		{"in-range-fault-192", "4331464201000000c0000000c1000000", "arithmetic_error", 1, 193},
		{"wrong-magic", "4231464201000000ffffffffc1000000", "protocol_error", 1, 193},
		{"zero-magic", "0000000001000000ffffffffc1000000", "protocol_error", 1, 193},
		{"all-ones-initialization", "ffffffffffffffffffffffffffffffff", "protocol_error", 1, 193},
		{"wrong-known-site", "4331464202000000ffffffffc1000000", "protocol_error", 1, 193},
		{"zero-site", "4331464200000000ffffffffc1000000", "protocol_error", 1, 193},
		{"unknown-site", "4331464203000000ffffffffc1000000", "protocol_error", 1, 193},
		{"byte-swapped-site", "4331464200000001ffffffffc1000000", "protocol_error", 1, 193},
		{"zero-count", "4331464201000000ffffffff00000000", "protocol_error", 1, 193},
		{"smaller-count", "4331464201000000ffffffffc0000000", "protocol_error", 1, 193},
		{"larger-count", "4331464201000000ffffffffc2000000", "protocol_error", 1, 193},
		{"max-count", "4331464201000000ffffffffffffffff", "protocol_error", 1, 193},
		{"byte-swapped-count", "4331464201000000ffffffff000000c1", "protocol_error", 1, 193},
		{"index-equals-count", "4331464201000000c1000000c1000000", "protocol_error", 1, 193},
		{"index-after-count", "4331464201000000c2000000c1000000", "protocol_error", 1, 193},
		{"index-near-sentinel", "4331464201000000feffffffc1000000", "protocol_error", 1, 193},
		{"bad-magic-with-in-range-index", "423146420100000000000000c1000000", "protocol_error", 1, 193},
		{"swapped-index-count", "4331464201000000c1000000ffffffff", "protocol_error", 1, 193},
		{"truncated-0", "", "protocol_error", 1, 193},
		{"truncated-1", "43", "protocol_error", 1, 193},
		{"truncated-4", "43314642", "protocol_error", 1, 193},
		{"truncated-8", "4331464201000000", "protocol_error", 1, 193},
		{"truncated-12", "4331464201000000ffffffff", "protocol_error", 1, 193},
		{"truncated-15", "4331464201000000ffffffffc10000", "protocol_error", 1, 193},
		{"oversized-17", "4331464201000000ffffffffc100000000", "protocol_error", 1, 193},
		{"oversized-20", "4331464201000000ffffffffc100000000000000", "protocol_error", 1, 193},
		{"big-endian-record", "4246314300000001ffffffff000000c1", "protocol_error", 1, 193},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := hex.DecodeString(tc.hex)
			if err != nil {
				t.Fatal(err)
			}
			got := decodeV41BF16CheckedStatus(raw, tc.count, tc.site)
			var arithmetic *V41BF16CheckedArithmeticError
			var protocol *V41BF16CheckedProtocolError
			switch tc.verdict {
			case "success":
				if got != nil {
					t.Fatal(got)
				}
			case "arithmetic_error":
				if !errors.As(got, &arithmetic) || arithmetic.Site != tc.site || arithmetic.Index != int(binary.LittleEndian.Uint32(raw[8:12])) {
					t.Fatalf("lost fault identity: %v", got)
				}
			case "protocol_error":
				if !errors.As(got, &protocol) {
					t.Fatalf("malformed record admitted: %v", got)
				}
			default:
				t.Fatal("unknown test verdict")
			}
		})
	}
	// Echo each issued value into an otherwise valid success record. An all-zero
	// record would fail its magic check even if the issued-geometry guards vanished.
	record := func(count int, site, index uint32) []byte {
		raw := make([]byte, 16)
		binary.LittleEndian.PutUint32(raw[0:4], 0x42463143)
		binary.LittleEndian.PutUint32(raw[4:8], site)
		binary.LittleEndian.PutUint32(raw[8:12], index)
		binary.LittleEndian.PutUint32(raw[12:16], uint32(count))
		return raw
	}
	maxCount := min(math.MaxInt32, math.MaxInt/4)
	for _, site := range []uint32{1, 2} {
		if err := decodeV41BF16CheckedStatus(record(maxCount, site, math.MaxUint32), maxCount, site); err != nil {
			t.Fatalf("exact allocation-free count bound rejected at site %d: %v", site, err)
		}
		var fault *V41BF16CheckedArithmeticError
		err := decodeV41BF16CheckedStatus(record(maxCount, site, uint32(maxCount-1)), maxCount, site)
		if !errors.As(err, &fault) || fault.Site != site || fault.Index != maxCount-1 {
			t.Fatalf("highest valid fault index lost at site %d: %v", site, err)
		}
		for _, n := range []int{-1, 0, maxCount + 1, math.MaxInt} {
			var protocol *V41BF16CheckedProtocolError
			err := decodeV41BF16CheckedStatus(record(n, site, math.MaxUint32), n, site)
			if !errors.As(err, &protocol) {
				t.Fatalf("invalid issued count %d at site %d admitted or misclassified: %v", n, site, err)
			}
		}
	}
	for _, site := range []uint32{0, 3, math.MaxUint32} {
		var protocol *V41BF16CheckedProtocolError
		err := decodeV41BF16CheckedStatus(record(1, site, math.MaxUint32), 1, site)
		if !errors.As(err, &protocol) {
			t.Fatalf("invalid echoed issued site %d admitted or misclassified: %v", site, err)
		}
	}
	// Exact stale success can decode as success. Only per-dispatch initialization
	// and memory-order instrumentation establish freshness; do not fake a test.
}
