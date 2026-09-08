package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

func TestSelectVulkanQ4KWave32RequiresExactDeviceAndCapabilities(t *testing.T) {
	admitted := vulkanQ4KWave32Device{
		vendorID:             vulkanAMDVendorID,
		deviceID:             vulkanGFX1151DeviceID,
		computeSubgroupBasic: true,
		computeSubgroupArith: true,
		defaultSubgroupSize:  vulkanWave32SubgroupSz,
	}
	tests := []struct {
		name        string
		device      vulkanQ4KWave32Device
		tokens      int
		optIn       bool
		forceScalar bool
		want        bool
	}{
		{name: "admitted effective Wave32", device: admitted, tokens: 1, optIn: true, want: true},
		{name: "admitted required Wave32", device: func() vulkanQ4KWave32Device {
			d := admitted
			d.defaultSubgroupSize = 64
			d.canRequireWave32 = true
			return d
		}(), tokens: 1, optIn: true, want: true},
		{name: "default scalar", device: admitted, tokens: 1},
		{name: "explicit scalar override", device: admitted, tokens: 1, optIn: true, forceScalar: true},
		{name: "P greater than one", device: admitted, tokens: 4, optIn: true},
		{name: "wrong vendor", device: func() vulkanQ4KWave32Device {
			d := admitted
			d.vendorID++
			return d
		}(), tokens: 1, optIn: true},
		{name: "wrong device", device: func() vulkanQ4KWave32Device {
			d := admitted
			d.deviceID++
			return d
		}(), tokens: 1, optIn: true},
		{name: "missing basic", device: func() vulkanQ4KWave32Device {
			d := admitted
			d.computeSubgroupBasic = false
			return d
		}(), tokens: 1, optIn: true},
		{name: "missing arithmetic", device: func() vulkanQ4KWave32Device {
			d := admitted
			d.computeSubgroupArith = false
			return d
		}(), tokens: 1, optIn: true},
		{name: "wrong subgroup size", device: func() vulkanQ4KWave32Device {
			d := admitted
			d.defaultSubgroupSize = 64
			return d
		}(), tokens: 1, optIn: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := selectVulkanQ4KWave32(tc.device, tc.tokens, tc.optIn, tc.forceScalar); got != tc.want {
				t.Fatalf("selectVulkanQ4KWave32() = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestVulkanQ4KWave32NativeGateMatchesSelectorContract(t *testing.T) {
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	for _, required := range []string{
		"props.vendorID == 0x1002u",
		"props.deviceID == 0x1586u",
		"VK_SUBGROUP_FEATURE_BASIC_BIT",
		"VK_SUBGROUP_FEATURE_ARITHMETIC_BIT",
		"effectiveSubgroup32 || requiredSubgroup32",
		"if (!g_have_q4k_wave32 || P != 1) return false",
		"default retains scalar",
	} {
		if !strings.Contains(source, required) {
			t.Errorf("native Wave32 gate missing %q", required)
		}
	}
}

var q4KWave32FixtureScales = [8]byte{1, 7, 15, 31, 16, 33, 47, 63}
var q4KWave32FixtureMinimums = [8]byte{2, 9, 14, 30, 17, 34, 48, 62}

func packQ4KWave32DeviceFreeBlock(dst []byte, row, block int) {
	binary.LittleEndian.PutUint16(dst[0:2], Float32ToFloat16Bits(0.125))
	binary.LittleEndian.PutUint16(dst[2:4], Float32ToFloat16Bits(0.0625))
	for j := 0; j < 4; j++ {
		dst[4+j] = q4KWave32FixtureScales[j] | (q4KWave32FixtureScales[j+4]>>4)<<6
		dst[8+j] = q4KWave32FixtureMinimums[j] | (q4KWave32FixtureMinimums[j+4]>>4)<<6
		dst[12+j] = q4KWave32FixtureScales[j+4]&15 | (q4KWave32FixtureMinimums[j+4]&15)<<4
	}
	for pair := 0; pair < 4; pair++ {
		for lane := 0; lane < 32; lane++ {
			low := byte((row*11 + block*5 + pair*14 + lane*3) & 15)
			high := byte((row*7 + block*13 + pair*9 + lane*5 + 1) & 15)
			dst[16+pair*32+lane] = low | high<<4
		}
	}
}

func q4KWave32DeviceFreeScaleMin(block []byte, group int) (byte, byte) {
	if group < 4 {
		return block[4+group] & 63, block[8+group] & 63
	}
	tail := block[8+group]
	return tail&15 | (block[group]>>6)<<4,
		tail>>4 | (block[4+group]>>6)<<4
}

func simulateQ4KWave32DeviceFreeRow(raw []byte, row, rows, columns int, x []float32) float32 {
	if row >= rows {
		return 0
	}
	blocks := columns / q4kSuper
	partials := [32]float32{}
	for lane := 0; lane < 32; lane++ {
		for block := 0; block < blocks; block++ {
			offset := (row*blocks + block) * q4kSuperBlock
			packed := raw[offset : offset+q4kSuperBlock]
			d := math.Float32frombits(f16bitsToF32(binary.LittleEndian.Uint16(packed[0:2])))
			dm := math.Float32frombits(f16bitsToF32(binary.LittleEndian.Uint16(packed[2:4])))
			for group := 0; group < 8; group++ {
				scale, minimum := q4KWave32DeviceFreeScaleMin(packed, group)
				codes := packed[16+(group/2)*32+lane]
				quant := codes & 15
				if group&1 != 0 {
					quant = codes >> 4
				}
				weight := d*float32(scale)*float32(quant) - dm*float32(minimum)
				partials[lane] += weight * x[block*q4kSuper+group*32+lane]
			}
		}
	}
	var sum float32
	for _, partial := range partials {
		sum += partial
	}
	return sum
}

func TestQ4KWave32DeviceFreeLaneMappingMetadataNibblesAndOddRows(t *testing.T) {
	for _, columns := range []int{256, 512, 768} {
		t.Run(fmt.Sprintf("in_%d", columns), func(t *testing.T) {
			const rows = 3
			raw := make([]byte, rows*(columns/q4kSuper)*q4kSuperBlock)
			for row := 0; row < rows; row++ {
				for block := 0; block < columns/q4kSuper; block++ {
					offset := (row*(columns/q4kSuper) + block) * q4kSuperBlock
					packQ4KWave32DeviceFreeBlock(raw[offset:offset+q4kSuperBlock], row, block)
				}
			}
			for group := 0; group < 8; group++ {
				scale, minimum := q4KWave32DeviceFreeScaleMin(raw[:q4kSuperBlock], group)
				if scale != q4KWave32FixtureScales[group] || minimum != q4KWave32FixtureMinimums[group] {
					t.Fatalf("group %d metadata = (%d,%d), want (%d,%d)", group, scale, minimum, q4KWave32FixtureScales[group], q4KWave32FixtureMinimums[group])
				}
			}
			var lowSeen, highSeen [16]bool
			for offset := 16; offset < len(raw); offset += q4kSuperBlock {
				for _, packed := range raw[offset : offset+128] {
					lowSeen[packed&15] = true
					highSeen[packed>>4] = true
				}
			}
			for nibble := 0; nibble < 16; nibble++ {
				if !lowSeen[nibble] || !highSeen[nibble] {
					t.Fatalf("nibble %d coverage low=%t high=%t", nibble, lowSeen[nibble], highSeen[nibble])
				}
			}
			x := make([]float32, columns)
			for i := range x {
				x[i] = float32((i%29)-14) / 17
			}
			scratch := make([]float32, q4kSuper)
			rowBytes := (columns / q4kSuper) * q4kSuperBlock
			for row := 0; row < rows; row++ {
				got := simulateQ4KWave32DeviceFreeRow(raw, row, rows, columns, x)
				want := q4kRowDot(raw[row*rowBytes:(row+1)*rowBytes], x, scratch)
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
					t.Fatalf("row %d cooperative result is non-finite: %g", row, got)
				}
				delta := math.Abs(float64(got - want))
				limit := 1e-5 * math.Max(1, math.Abs(float64(want)))
				if delta > limit {
					t.Fatalf("row %d cooperative result = %g, scalar = %g (delta %g > %g)", row, got, want, delta, limit)
				}
			}
			if got := simulateQ4KWave32DeviceFreeRow(raw, rows, rows, columns, x); got != 0 {
				t.Fatalf("odd-row tail = %g, want 0", got)
			}
		})
	}
}
