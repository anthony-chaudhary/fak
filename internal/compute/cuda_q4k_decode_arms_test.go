package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestParseCUDAQ4KDecodeArm(t *testing.T) {
	for in, want := range map[string]CUDAQ4KDecodeArm{"": CUDAQ4KDecodeF32, "f32": CUDAQ4KDecodeF32, "mmvq": CUDAQ4KDecodeMMVQ} {
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
