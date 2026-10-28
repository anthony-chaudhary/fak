package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// writeTinyGGUF renders a minimal two-tensor F32/BF16 GGUF for the verb smoke
// path; the payload section is zero-filled.
func writeTinyGGUF(t *testing.T, gateType ggufload.TensorType) []byte {
	t.Helper()
	var b bytes.Buffer
	b.WriteString(ggufload.Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(ggufload.Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(2)) // tensors
	_ = binary.Write(&b, binary.LittleEndian, uint64(0)) // kvs
	writeTinyName(&b, "norm.weight")
	_ = binary.Write(&b, binary.LittleEndian, uint32(1))
	_ = binary.Write(&b, binary.LittleEndian, uint64(4))
	_ = binary.Write(&b, binary.LittleEndian, uint32(ggufload.TensorF32))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0))
	writeTinyName(&b, "gate.weight")
	_ = binary.Write(&b, binary.LittleEndian, uint32(1))
	_ = binary.Write(&b, binary.LittleEndian, uint64(4))
	_ = binary.Write(&b, binary.LittleEndian, uint32(gateType))
	_ = binary.Write(&b, binary.LittleEndian, uint64(32)) // 32-byte aligned, as the loader requires
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	dataStart := b.Len()
	// Generous zero tail: covers the largest one-block payload any gate type
	// used by this test can have (Q8_0 block = 34B); OpenWeights reads no
	// weight bytes on this path.
	pad := dataStart + 32 + 128
	for b.Len() < pad {
		b.WriteByte(0)
	}
	return b.Bytes()
}

func writeTinyName(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func TestRunAllocationCheck(t *testing.T) {
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(ggufPath, writeTinyGGUF(t, ggufload.TensorBF16), 0o644); err != nil {
		t.Fatal(err)
	}
	sidecarPath := filepath.Join(dir, "tiny.rco-allocation.txt")
	sidecar := "# RCO per-tensor GGUF allocation\n#\n# Target whole-file bpw: 24.00\n# Tensor count: 2\n# Quant-type counts: F32=1, BF16=1\nnorm.weight: F32\ngate.weight: BF16\n"
	if err := os.WriteFile(sidecarPath, []byte(sidecar), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("pass", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := runAllocationCheck(&out, &errOut, []string{ggufPath, "--sidecar", sidecarPath})
		if code != 0 {
			t.Fatalf("code = %d, stderr=%s", code, errOut.String())
		}
		for _, want := range []string{"PASS allocation-check", "effective_bpws=24.0000", "F32=1", "BF16=1"} {
			if !strings.Contains(out.String(), want) {
				t.Errorf("stdout missing %q:\n%s", want, out.String())
			}
		}
	})

	t.Run("corrupt sidecar refuses typed", func(t *testing.T) {
		badPath := filepath.Join(dir, "bad.rco-allocation.txt")
		if err := os.WriteFile(badPath, []byte("# RCO per-tensor GGUF allocation\nnot a row\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errOut bytes.Buffer
		code := runAllocationCheck(&out, &errOut, []string{ggufPath, "--sidecar", badPath})
		if code != 1 || !strings.Contains(errOut.String(), "FAIL ALLOCATION_SIDECAR_MALFORMED") {
			t.Fatalf("code=%d stderr=%q", code, errOut.String())
		}
	})

	t.Run("type mismatch refuses typed", func(t *testing.T) {
		wrongPath := filepath.Join(dir, "wrong.gguf")
		if err := os.WriteFile(wrongPath, writeTinyGGUF(t, ggufload.TensorQ8_0), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errOut bytes.Buffer
		code := runAllocationCheck(&out, &errOut, []string{wrongPath, "--sidecar", sidecarPath})
		if code != 1 || !strings.Contains(errOut.String(), "FAIL ALLOCATION_TYPE_MISMATCH") {
			t.Fatalf("code=%d stderr=%q", code, errOut.String())
		}
	})

	t.Run("usage errors exit 2", func(t *testing.T) {
		var out, errOut bytes.Buffer
		if code := runAllocationCheck(&out, &errOut, nil); code != 2 {
			t.Errorf("no args: code = %d", code)
		}
		if code := runAllocationCheck(&out, &errOut, []string{ggufPath}); code != 2 {
			t.Errorf("missing --sidecar: code = %d", code)
		}
	})

	t.Run("help exits 0", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := runAllocationCheck(&out, &errOut, []string{"--help"})
		if code != 0 || !strings.Contains(out.String(), "usage:") {
			t.Fatalf("code=%d out=%q", code, out.String())
		}
	})

	t.Run("flags before positional also work", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := runAllocationCheck(&out, &errOut, []string{"--sidecar", sidecarPath, ggufPath})
		if code != 0 || !strings.Contains(out.String(), "PASS allocation-check") {
			t.Fatalf("code=%d out=%q err=%q", code, out.String(), errOut.String())
		}
	})

	t.Run("bpw scope exclusion", func(t *testing.T) {
		var out, errOut bytes.Buffer
		// Excluding both tensors leaves an empty bpw scope: the validator must
		// refuse, not silently pass.
		code := runAllocationCheck(&out, &errOut, []string{ggufPath, "--sidecar", sidecarPath,
			"--exclude", "norm.weight", "--exclude", "gate.weight"})
		if code != 1 || !strings.Contains(errOut.String(), "FAIL ALLOCATION_UNSUPPORTED_TYPE") {
			t.Fatalf("code=%d stderr=%q", code, errOut.String())
		}
	})
}
