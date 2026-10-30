package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestParseCUDAQ4KDecodeArm(t *testing.T) {
	for in, want := range map[string]CUDAQ4KDecodeArm{"": CUDAQ4KDecodeMMVQ, "f32": CUDAQ4KDecodeF32, "mmvq": CUDAQ4KDecodeMMVQ} {
		if got, err := ParseCUDAQ4KDecodeArm(in); err != nil || got != want {
			t.Errorf("ParseCUDAQ4KDecodeArm(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, in := range []string{"MMVQ", "mmq", "dp4a"} {
		if _, err := ParseCUDAQ4KDecodeArm(in); !errors.Is(err, ErrCUDAQ4KDecodeArmUnknown) {
			t.Errorf("ParseCUDAQ4KDecodeArm(%q) err = %v, want ErrCUDAQ4KDecodeArmUnknown", in, err)
		}
	}
	if err := SetCUDAQ4KDecodeArm(""); !errors.Is(err, ErrCUDAQ4KDecodeArmUnknown) {
		t.Errorf("SetCUDAQ4KDecodeArm(\"\") err = %v, want ErrCUDAQ4KDecodeArmUnknown", err)
	}
}

// TestCUDAQ4KDecodeDefaultIsMMVQ pins the default resolution the raider ABBA receipt chose: an
// unset FAK_CUDA_Q4K_DECODE runs mmvq (C code 1), and f32 stays a parseable escape hatch.
// fak-test:runtime fast est=1ms
func TestCUDAQ4KDecodeDefaultIsMMVQ(t *testing.T) {
	if CUDAQ4KDecodeDefault != CUDAQ4KDecodeMMVQ {
		t.Fatalf("CUDAQ4KDecodeDefault = %q, want %q", CUDAQ4KDecodeDefault, CUDAQ4KDecodeMMVQ)
	}
	if code := cudaQ4KDecodeArmCodes[CUDAQ4KDecodeDefault]; code != 1 {
		t.Fatalf("default arm C code = %d, want 1 (g_q4k_decode_arm mmvq)", code)
	}
	if got, err := ParseCUDAQ4KDecodeArm(string(CUDAQ4KDecodeF32)); err != nil || got != CUDAQ4KDecodeF32 {
		t.Fatalf("escape hatch FAK_CUDA_Q4K_DECODE=f32 = %q, %v; want f32", got, err)
	}
}
