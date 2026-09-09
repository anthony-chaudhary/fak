package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func TestMetalGGUFPeakCapacity(t *testing.T) {
	const gib = int64(1 << 30)
	tests := []struct {
		name   string
		metal  bool
		steady int64
		total  int64
		known  bool
		refuse bool
	}{
		{
			name:   "Qwen3.8 27B Q4_K_M refuses 36 GiB Mac",
			metal:  true,
			steady: 1592 * gib / 100,
			total:  36 * gib,
			known:  true,
			refuse: true,
		},
		{
			name:   "larger unified memory admits same checkpoint",
			metal:  true,
			steady: 1592 * gib / 100,
			total:  64 * gib,
			known:  true,
			refuse: false,
		},
		{
			name:   "non-Metal remains unchanged",
			metal:  false,
			steady: 1592 * gib / 100,
			total:  36 * gib,
			known:  true,
			refuse: false,
		},
		{
			name:   "unknown host memory remains unchanged",
			metal:  true,
			steady: 1592 * gib / 100,
			total:  0,
			known:  false,
			refuse: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			peak, refuse := metalGGUFPeakCapacity(tt.metal, tt.steady, tt.total, tt.known)
			if refuse != tt.refuse {
				t.Fatalf("refuse = %v, want %v (peak=%d)", refuse, tt.refuse, peak)
			}
		})
	}
}

func TestStreamedQ4KMetalCapacityUsesMeasuredNativePeak(t *testing.T) {
	if streamedQ4KMeasuredPeakBytes != 22754885632 {
		t.Fatalf("measured peak = %d, want canonical no-FREE_CPU receipt's /usr/bin/time RSS", streamedQ4KMeasuredPeakBytes)
	}
	if streamedQ4KMetalCapacityBytes < streamedQ4KMeasuredPeakBytes {
		t.Fatalf("capacity bound %d is below measured peak %d", streamedQ4KMetalCapacityBytes, streamedQ4KMeasuredPeakBytes)
	}
	if streamedQ4KFreeCPUMetalCapacityBytes != 36<<30 {
		t.Fatalf("FREE_CPU capacity = %d, want exact #8964 36 GiB host envelope (not the old 18 GiB RSS)", streamedQ4KFreeCPUMetalCapacityBytes)
	}

	tests := []struct {
		name      string
		total     int64
		known     bool
		freeCPU   bool
		required  int64
		mode      string
		refuse    bool
		wantError bool
	}{
		{name: "FREE_CPU one byte below 36 GiB refuses", total: (36 << 30) - 1, known: true, freeCPU: true, required: 36 << 30, mode: streamedQ4KModeFreeCPU, refuse: true, wantError: true},
		{name: "FREE_CPU 36 GiB proceeds", total: 36 << 30, known: true, freeCPU: true, required: 36 << 30, mode: streamedQ4KModeFreeCPU},
		{name: "retained CPU 36 GiB refuses", total: 36 << 30, known: true, required: 44 << 30, mode: streamedQ4KModeRetainedCPU, refuse: true, wantError: true},
		{name: "retained CPU 44 GiB proceeds", total: 44 << 30, known: true, required: 44 << 30, mode: streamedQ4KModeRetainedCPU},
		{name: "unknown host memory preserves existing behavior", freeCPU: true, mode: streamedQ4KModeFreeCPU},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			required, refuse, mode := streamedQ4KMetalCapacity(tt.total, tt.known, tt.freeCPU)
			if required != tt.required || refuse != tt.refuse || mode != tt.mode {
				t.Fatalf("capacity = (%d, %v, %q), want (%d, %v, %q)", required, refuse, mode, tt.required, tt.refuse, tt.mode)
			}
			err := refuseStreamedQ4KMetalCapacity(tt.total, tt.known, tt.freeCPU)
			if (err != nil) != tt.wantError {
				t.Fatalf("error = %v, wantError %v", err, tt.wantError)
			}
			if err != nil {
				for _, want := range []string{"METAL_STREAM_Q4K_PEAK_TOO_BIG", "mode=" + tt.mode, fmt.Sprintf("requires %d bytes", tt.required)} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not contain %q", err, want)
					}
				}
			}
		})
	}
}

func TestNonStreamingMetalCapacityIsUnchangedByFREECPUDeclaration(t *testing.T) {
	const gib = int64(1 << 30)
	t.Setenv("FAK_Q4K_FREE_CPU", "1")
	peak, refuse := metalGGUFPeakCapacity(true, 1592*gib/100, 36*gib, true)
	if peak != int64(float64(1592*gib/100)*metalGGUFObservedPeakMultiplier) || !refuse {
		t.Fatalf("non-streaming capacity = (%d, %v), want unchanged generic Metal refusal", peak, refuse)
	}
}

func writeSynth27BGGUF(t *testing.T, path string, isUDQ2KXL bool) {
	t.Helper()
	var b bytes.Buffer
	b.WriteString("GGUF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(3))

	writeString := func(s string) {
		_ = binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	writeKV := func(k string, typ uint32, writeVal func()) {
		writeString(k)
		_ = binary.Write(&b, binary.LittleEndian, typ)
		writeVal()
	}

	nTensors := uint64(3)
	nKV := uint64(3)

	_ = binary.Write(&b, binary.LittleEndian, nTensors)
	_ = binary.Write(&b, binary.LittleEndian, nKV)

	writeKV("general.architecture", uint32(ggufload.TypeString), func() { writeString("qwen2") })
	writeKV("qwen2.block_count", uint32(ggufload.TypeUint64), func() { _ = binary.Write(&b, binary.LittleEndian, uint64(1)) })
	writeKV("general.alignment", uint32(ggufload.TypeUint32), func() { _ = binary.Write(&b, binary.LittleEndian, uint32(32)) })

	writeTensor := func(name string, dims []uint64, typ ggufload.TensorType, off uint64) {
		writeString(name)
		_ = binary.Write(&b, binary.LittleEndian, uint32(len(dims)))
		for _, d := range dims {
			_ = binary.Write(&b, binary.LittleEndian, d)
		}
		_ = binary.Write(&b, binary.LittleEndian, uint32(typ))
		_ = binary.Write(&b, binary.LittleEndian, off)
	}

	if isUDQ2KXL {
		// UD-Q2_K_XL mixture: Q8_0 embedding + Q2_K matmul + IQ2_XXS
		writeTensor("token_embd.weight", []uint64{5120, 152064}, ggufload.TensorQ8_0, 0)
		writeTensor("blk.0.ffn_gate.weight", []uint64{5120, 5600000}, ggufload.TensorQ2_K, 0)
		writeTensor("blk.0.ffn_down.weight", []uint64{256, 256}, ggufload.TensorIQ2_XXS, 0)
	} else {
		// Standard Q4_K_M 27B model
		writeTensor("token_embd.weight", []uint64{5120, 152064}, ggufload.TensorQ4_K, 0)
		writeTensor("blk.0.ffn_gate.weight", []uint64{5120, 5600000}, ggufload.TensorQ4_K, 0)
		writeTensor("output_norm.weight", []uint64{5120}, ggufload.TensorF32, 0)
	}

	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("writeSynth27BGGUF: %v", err)
	}
}

func TestMetalGGUFPeakCapacityAlignsWithLoadArm(t *testing.T) {
	const gib = int64(1 << 30)
	dir := t.TempDir()
	udPath := filepath.Join(dir, "qwen38-27b-ud-q2kxl.gguf")
	writeSynth27BGGUF(t, udPath, true)

	q4kPath := filepath.Join(dir, "qwen38-27b-q4km.gguf")
	writeSynth27BGGUF(t, q4kPath, false)

	t.Run("UD-Q2_K_XL expands to Q8 and refuses 36 GiB Mac", func(t *testing.T) {
		err := refuseOversubscribedMetalGGUFForHost(udPath, 36*gib, true)
		if err == nil {
			t.Fatal("expanding Q8 load of 27B UD-Q2_K_XL must be refused on 36 GiB host, got nil")
		}
		if !strings.Contains(err.Error(), "METAL_GGUF_PEAK_TOO_BIG") {
			t.Fatalf("error = %v, want METAL_GGUF_PEAK_TOO_BIG", err)
		}
	})

	t.Run("UD-Q2_K_XL expands to Q8 and admits on 128 GiB Mac", func(t *testing.T) {
		err := refuseOversubscribedMetalGGUFForHost(udPath, 128*gib, true)
		if err != nil {
			t.Fatalf("expanding Q8 load of 27B UD-Q2_K_XL must be admitted on 128 GiB host, got error: %v", err)
		}
	})

	t.Run("resident Q4_K evaluates at on-disk size and refuses 36 GiB Mac", func(t *testing.T) {
		err := refuseOversubscribedMetalGGUFForHost(q4kPath, 36*gib, true)
		if err == nil {
			t.Fatal("resident Q4_K 27B must be refused on 36 GiB host, got nil")
		}
		if !strings.Contains(err.Error(), "METAL_GGUF_PEAK_TOO_BIG") {
			t.Fatalf("error = %v, want METAL_GGUF_PEAK_TOO_BIG", err)
		}
	})

	t.Run("resident Q4_K evaluates at on-disk size and admits on 64 GiB Mac", func(t *testing.T) {
		err := refuseOversubscribedMetalGGUFForHost(q4kPath, 64*gib, true)
		if err != nil {
			t.Fatalf("resident Q4_K 27B must be admitted on 64 GiB host, got error: %v", err)
		}
	})

	t.Run("FAK_Q4K=0 forces Q8 rollback and refuses 36 GiB Mac", func(t *testing.T) {
		t.Setenv("FAK_Q4K", "0")
		err := refuseOversubscribedMetalGGUFForHost(q4kPath, 36*gib, true)
		if err == nil {
			t.Fatal("Q8 rollback of 27B must be refused on 36 GiB host, got nil")
		}
		if !strings.Contains(err.Error(), "METAL_GGUF_PEAK_TOO_BIG") {
			t.Fatalf("error = %v, want METAL_GGUF_PEAK_TOO_BIG", err)
		}
	})
}
