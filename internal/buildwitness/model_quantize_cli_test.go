package buildwitness

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// fak-test:justify why=regression when=changed:cmd/fak/model.go,cmd/fak/model_quantize.go,internal/ggufload/gguf_convert.go
// fak-test:runtime integration est=5m lane=default
func TestModelQuantizeCLI(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	binaryPath := filepath.Join(dir, "fak-quantize")
	if runtime.GOOS == "windows" {
		binaryPath += ".exe"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-o", binaryPath, "./cmd/fak")
	build.Dir = root
	procguard.ConfigureProcessTreeCancel(build)
	build.WaitDelay = 10 * time.Second
	var diagnostic serveProofCLIBuffer
	build.Stdout = &diagnostic
	build.Stderr = &diagnostic
	if err := build.Run(); err != nil {
		t.Fatalf("CLI compiler error (not behavioral RED): %v\n%s", err, diagnostic.String())
	}
	if diagnostic.truncated {
		t.Fatal("CLI build diagnostics truncated")
	}
	input := filepath.Join(dir, "input.gguf")
	original := modelQuantizeCLIFixture()
	if err := os.WriteFile(input, original, 0600); err != nil {
		t.Fatal(err)
	}
	run := func(args ...string) ([]byte, error) {
		t.Helper()
		ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, binaryPath, args...)
		cmd.Dir = dir
		cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + dir, "USERPROFILE=" + dir, "XDG_CONFIG_HOME=" + dir, "XDG_CACHE_HOME=" + dir, "TMPDIR=" + dir, "TMP=" + dir, "TEMP=" + dir}
		procguard.ConfigureProcessTreeCancel(cmd)
		cmd.WaitDelay = 10 * time.Second
		var out, stderr serveProofCLIBuffer
		cmd.Stdout = &out
		cmd.Stderr = &stderr
		err := cmd.Run()
		if ctx.Err() != nil || out.truncated || stderr.truncated {
			t.Fatalf("CLI command incomplete: %v: %s", ctx.Err(), stderr.String())
		}
		if err != nil {
			t.Logf("CLI refusal: %s", stderr.String())
		}
		return out.Bytes(), err
	}
	for _, kind := range []string{"q4_k", "q5_k", "q6_k"} {
		output := filepath.Join(dir, kind+".gguf")
		b, err := run("model", "quantize", "--input", input, "--output", output, "--type", kind)
		if err != nil {
			t.Fatal(err)
		}
		var report ggufload.ConversionReport
		if err := json.Unmarshal(b, &report); err != nil {
			t.Fatal(err)
		}
		if report.ConvertedTensors != 1 || report.PreservedTensors != 1 {
			t.Fatalf("report %+v", report)
		}
		ws, err := ggufload.OpenWeights(output)
		if err != nil {
			t.Fatal(err)
		}
		decoded, info, err := ws.TensorF32("matrix")
		if err != nil || len(decoded) != 256 {
			t.Fatalf("reload: %v", err)
		}
		wantType := map[string]ggufload.TensorType{"q4_k": ggufload.TensorQ4_K, "q5_k": ggufload.TensorQ5_K, "q6_k": ggufload.TensorQ6_K}[kind]
		if info.Type != wantType {
			t.Fatalf("type=%s", info.Type)
		}
		kept, _, err := ws.TensorBytes("preserved")
		if err != nil || !bytes.Equal(kept, []byte{0, 0, 128, 63}) {
			t.Fatal("preserved bytes changed")
		}
		ws.Close()
		before, _ := os.ReadFile(output)
		if _, err := run("model", "quantize", "--input", input, "--output", output, "--type", kind); err == nil {
			t.Fatal("CLI overwrote existing output")
		}
		after, _ := os.ReadFile(output)
		if !bytes.Equal(before, after) {
			t.Fatal("CLI refusal modified output")
		}
	}
	after, _ := os.ReadFile(input)
	if !bytes.Equal(original, after) {
		t.Fatal("CLI changed input")
	}
	bad := filepath.Join(dir, "bad.gguf")
	os.WriteFile(bad, original[:len(original)-1], 0600)
	badOut := filepath.Join(dir, "refused.gguf")
	if _, err := run("model", "quantize", "--input", bad, "--output", badOut, "--type", "q4_k"); err == nil {
		t.Fatal("malformed input accepted")
	}
	if _, err := os.Lstat(badOut); !os.IsNotExist(err) {
		t.Fatal("refusal published artifact")
	}
}

func modelQuantizeCLIFixture() []byte {
	var b bytes.Buffer
	u32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	u64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { u64(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	u32(3)
	u64(2)
	u64(0)
	str("matrix")
	u32(2)
	u64(256)
	u64(1)
	u32(0)
	u64(0)
	str("preserved")
	u32(1)
	u64(1)
	u32(0)
	u64(1024)
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	for i := 0; i < 256; i++ {
		u32(math.Float32bits(float32((i*73)%251-125) / 16))
	}
	u32(math.Float32bits(1))
	return b.Bytes()
}
