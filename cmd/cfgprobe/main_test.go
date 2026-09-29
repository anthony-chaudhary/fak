// main_test.go witnesses the cfgprobe delegation contract end to end: the package's
// real binary is compiled once in TestMain and then driven as a SUBPROCESS, so every
// assertion observes the actual process exit code and streams — not an in-process
// mock. This retires both `missing_tests` and `unproven_runtime` for the cfgprobe
// leaf: the binary must genuinely start, parse a GGUF header via internal/ggufload,
// and project the MoE/dense FFN axes main.go promises.
package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var binPath string

// TestMain compiles cmd/cfgprobe once into a temp dir, then runs the test suite
// against that real binary. A build failure fails the suite rather than silently
// skipping the runtime proof.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "cfgprobe-bin-*")
	if err != nil {
		panic("cfgprobe test: MkdirTemp: " + err.Error())
	}
	name := "cfgprobe"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binPath = filepath.Join(dir, name)

	out, err := exec.Command("go", "build", "-o", binPath, ".").CombinedOutput()
	if err != nil {
		os.RemoveAll(dir)
		panic("cfgprobe test: go build failed: " + err.Error() + "\n" + string(out))
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// runBin drives the compiled binary with args and returns its exit code plus the
// captured stdout and stderr.
func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(binPath, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		ee, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("run %v: %v", args, err)
		}
		code = ee.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// TestNoArgsPrintsUsageOnStderr proves the no-argument path is a usage refusal on
// stderr with an empty stdout and a distinct nonzero exit code.
func TestNoArgsPrintsUsageOnStderr(t *testing.T) {
	code, stdout, stderr := runBin(t)
	if stdout != "" {
		t.Fatalf("no-args stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "usage: cfgprobe") {
		t.Fatalf("no-args stderr = %q, want the usage banner", stderr)
	}
	if code != 2 {
		t.Fatalf("no-args exit = %d, want 2", code)
	}
}

// TestMissingFileRefuses proves a nonexistent path is refused loudly on stderr with
// exit 1 rather than a zero exit and silent stdout.
func TestMissingFileRefuses(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist.gguf")
	code, stdout, stderr := runBin(t, missing)
	if stdout != "" {
		t.Fatalf("missing-file stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "open:") {
		t.Fatalf("missing-file stderr = %q, want an open: error", stderr)
	}
	if code != 1 {
		t.Fatalf("missing-file exit = %d, want 1", code)
	}
}

// TestMalformedInputRefuses proves a present-but-not-GGUF file is parsed (not
// short-circuited) and refused as an open error with exit 1.
func TestMalformedInputRefuses(t *testing.T) {
	bad := filepath.Join(t.TempDir(), "garbage.gguf")
	if err := os.WriteFile(bad, []byte("NOT A GGUF FILE AT ALL"), 0o644); err != nil {
		t.Fatal(err)
	}
	code, stdout, stderr := runBin(t, bad)
	if stdout != "" {
		t.Fatalf("malformed stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "open:") {
		t.Fatalf("malformed stderr = %q, want an open: error", stderr)
	}
	if code != 1 {
		t.Fatalf("malformed exit = %d, want 1", code)
	}
}

// TestDelegationSuccess is the real runtime proof: a valid GGUF header must flow
// through main.go -> ggufload.Open -> File.Config and print the config axes, with
// the unset MoE expert width falling back to the dense IntermediateSize.
func TestDelegationSuccess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, tinyGGUF(), 0o644); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := runBin(t, path)
	if code != 0 {
		t.Fatalf("success exit = %d, stderr = %q", code, stderr)
	}
	if stderr != "" {
		t.Fatalf("success stderr = %q, want empty", stderr)
	}
	for _, want := range []string{
		"arch=qwen2\n",
		"HiddenSize=32\n",
		"IntermediateSize(dense)=64\n",
		"MoEIntermediateSize=0\n",
		"=> expert gate_proj forward requests [out=64, in=32]\n",
	} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("success stdout missing %q; got:\n%s", want, stdout)
		}
	}
}

// tinyGGUF builds a minimal but valid GGUF v3 header for a qwen2 dense model: the
// required dims plus alignment, with zero tensors (Open reads only the header, so no
// tensor-data blob is needed). Dense feed_forward_length=64 keeps MoEIntermediateSize
// unset, exercising main.go's MoE->dense fallback.
func tinyGGUF() []byte {
	var b bytes.Buffer
	writeStr := func(s string) {
		_ = binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	kvStr := func(k, v string) {
		writeStr(k)
		_ = binary.Write(&b, binary.LittleEndian, uint32(typeString))
		writeStr(v)
	}
	kvU32 := func(k string, v uint32) {
		writeStr(k)
		_ = binary.Write(&b, binary.LittleEndian, uint32(typeUint32))
		_ = binary.Write(&b, binary.LittleEndian, v)
	}
	kvU64 := func(k string, v uint64) {
		writeStr(k)
		_ = binary.Write(&b, binary.LittleEndian, uint32(typeUint64))
		_ = binary.Write(&b, binary.LittleEndian, v)
	}
	kvF32 := func(k string, v float32) {
		writeStr(k)
		_ = binary.Write(&b, binary.LittleEndian, uint32(typeFloat32))
		_ = binary.Write(&b, binary.LittleEndian, math.Float32bits(v))
	}

	b.WriteString("GGUF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(3)) // version
	_ = binary.Write(&b, binary.LittleEndian, uint64(0)) // tensor count
	_ = binary.Write(&b, binary.LittleEndian, uint64(7)) // metadata KV count

	kvStr("general.architecture", "qwen2")
	kvU32("general.alignment", 32)
	kvU64("qwen2.embedding_length", 32)
	kvU64("qwen2.block_count", 2)
	kvU64("qwen2.attention.head_count", 4)
	kvU64("qwen2.feed_forward_length", 64)
	kvF32("qwen2.attention.layer_norm_rms_epsilon", 1e-5)

	return b.Bytes()
}

// GGUF metadata value types mirrored locally so the fixture writer does not import
// internal/ggufload's numeric tags through an unexported path in the test.
const (
	typeUint32  = 4
	typeFloat32 = 6
	typeString  = 8
	typeUint64  = 10
)
