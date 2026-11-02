// Command refgen creates deterministic K-quant fixtures from an unchanged pinned
// llama.cpp source directory. It does not fetch sources or compile during tests.
package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

const pin = "83078fec0db82d6b5a00d9599062c38c39145755"

var sourceBlobs = map[string]string{
	"LICENSE":                           "e7dca554bcb802f98408383a864404e3aa4eacca",
	"ggml/src/ggml-quants.c":            "1ebc50a763f16db909de37090da38cc8c0fdde94",
	"ggml/src/ggml-common.h":            "1dbbe326d0fc6a5ce8a86d237cb94901340443af",
	"ggml/src/ggml-impl.h":              "ae26e0c23b461b716ed5f91977a419e10556087b",
	"ggml/src/ggml-quants.h":            "75188f1af180e3592f3c94bc4077989fb817c359",
	"ggml/src/ggml-cpu/ggml-cpu-impl.h": "5dd9ec8e628acad32537e87eccfdeb01c1c1dc46",
	"ggml/include/ggml.h":               "224bdef927ac90bfa07ee39db3247b7bac4f14ec",
	"ggml/include/ggml-cpu.h":           "dc6453c6eaa16667f720f987659ad42d03a403a2",
	"ggml/include/ggml-backend.h":       "cc3f8cd36e3549c4f3ed3d6ad5f6b577af4b837b",
	"ggml/include/gguf.h":               "b3a1e1230a06110a0a5f04469d849ac2000b467c",
	"ggml/include/ggml-alloc.h":         "a7926a21a9a20f37a4b75cf19fbca5d8d88b7779",
}

type fixture struct {
	Name  string     `json:"name"`
	Input []uint32   `json:"input_bits"`
	Q4    string     `json:"q4_k_hex"`
	Q5    string     `json:"q5_k_hex"`
	Q6    string     `json:"q6_k_hex"`
	RMSE  [3]float64 `json:"reference_rmse"`
}

func main() {
	src := flag.String("source", "", "unchanged pinned llama.cpp source directory")
	out := flag.String("output", "", "new fixture JSON (must not exist)")
	cc := flag.String("cc", "cc", "C compiler")
	flag.Parse()
	if err := run(*src, *out, *cc); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(src, out, cc string) error {
	if src == "" || out == "" || runtime.GOARCH != "amd64" || runtime.GOOS != "linux" {
		return fmt.Errorf("source/output required; this reference qualification is Linux amd64 only")
	}
	abs, err := filepath.Abs(src)
	if err != nil {
		return err
	}
	src = abs
	hashes := map[string]string{}
	for path, want := range sourceBlobs {
		b, err := os.ReadFile(filepath.Join(src, path))
		if err != nil {
			return err
		}
		h := sha1.New()
		fmt.Fprintf(h, "blob %d%c", len(b), 0)
		h.Write(b)
		if got := hex.EncodeToString(h.Sum(nil)); got != want {
			return fmt.Errorf("source %s blob %s != %s", path, got, want)
		}
		digest := sha256.Sum256(b)
		hashes[path] = hex.EncodeToString(digest[:])
	}
	compiler, err := exec.LookPath(cc)
	if err != nil {
		return err
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		return err
	}
	compilerBytes, err := os.ReadFile(compiler)
	if err != nil {
		return err
	}
	compilerHash := sha256.Sum256(compilerBytes)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	version, err := exec.CommandContext(ctx, compiler, "--version").CombinedOutput()
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "fak-kquant-reference-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	driver := filepath.Join(dir, "reference.c")
	if err := os.WriteFile(driver, []byte(cDriver), 0600); err != nil {
		return err
	}
	flags := []string{"-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fexcess-precision=standard", "-ffunction-sections", "-fdata-sections"}
	args := append(append([]string{}, flags...), "-I"+filepath.Join(src, "ggml/src"), "-I"+filepath.Join(src, "ggml/include"), driver, "-Wl,--gc-sections", "-lm", "-o", filepath.Join(dir, "reference"))
	if b, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("C reference compile: %w\n%s", err, b)
	}
	fixtures := inputs()
	for i := range fixtures {
		f := &fixtures[i]
		var in bytes.Buffer
		binary.Write(&in, binary.LittleEndian, uint32(len(f.Input)))
		binary.Write(&in, binary.LittleEndian, f.Input)
		cmd := exec.CommandContext(ctx, filepath.Join(dir, "reference"))
		cmd.Stdin = &in
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		b, err := cmd.Output()
		if err != nil {
			return fmt.Errorf("reference %s: %w: %s", f.Name, err, stderr.String())
		}
		n := len(f.Input)
		blocks := n / 256
		if len(b) != blocks*(144+176+210)+3*n*4 {
			return fmt.Errorf("reference output extent %s: %d", f.Name, len(b))
		}
		encoded := []*string{&f.Q4, &f.Q5, &f.Q6}
		for kind, size := range []int{144, 176, 210} {
			*encoded[kind] = hex.EncodeToString(b[:blocks*size])
			b = b[blocks*size:]
			var sum float64
			for j, bits := range f.Input {
				x := math.Float32frombits(bits)
				y := math.Float32frombits(binary.LittleEndian.Uint32(b[j*4:]))
				d := float64(x) - float64(y)
				sum += d * d
			}
			f.RMSE[kind] = math.Sqrt(sum / float64(n))
			b = b[n*4:]
		}
	}
	halfInputs := []uint32{0, 0x80000000, 0x33000000, 0x33000001, 0x33800000, 0x387fe000, 0x38800000, 0x3f801000, 0x3f803000, 0xbf801000, 0xbf803000, 0x477fe000, 0x477ff000, 0xc77fe000, 0x7fc12345, 0x7f800000}
	var halfIn bytes.Buffer
	binary.Write(&halfIn, binary.LittleEndian, uint32(len(halfInputs)))
	binary.Write(&halfIn, binary.LittleEndian, halfInputs)
	halfCmd := exec.CommandContext(ctx, filepath.Join(dir, "reference"))
	halfCmd.Stdin = &halfIn
	halfBytes, err := halfCmd.Output()
	if err != nil || len(halfBytes) != 2*len(halfInputs) {
		return fmt.Errorf("half reference: %w, bytes=%d", err, len(halfBytes))
	}
	halfExpected := make([]uint16, len(halfInputs))
	for i := range halfExpected {
		halfExpected[i] = binary.LittleEndian.Uint16(halfBytes[i*2:])
	}
	document := struct {
		Pin             string            `json:"upstream_pin"`
		SourceBlobs     map[string]string `json:"source_git_blobs"`
		SourceHashes    map[string]string `json:"source_sha256"`
		CompilerVersion string            `json:"compiler_version"`
		CompilerSHA256  string            `json:"compiler_sha256"`
		Flags           []string          `json:"compiler_flags"`
		DriverSHA256    string            `json:"driver_sha256"`
		HalfInputs      []uint32          `json:"half_input_bits"`
		HalfExpected    []uint16          `json:"half_expected_bits"`
		Fixtures        []fixture         `json:"fixtures"`
	}{Pin: pin, SourceBlobs: sourceBlobs, SourceHashes: hashes, CompilerVersion: string(version), CompilerSHA256: hex.EncodeToString(compilerHash[:]), Flags: flags, Fixtures: fixtures, HalfInputs: halfInputs, HalfExpected: halfExpected}
	dh := sha256.Sum256([]byte(cDriver))
	document.DriverSHA256 = hex.EncodeToString(dh[:])
	b, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(append(b, '\n'))
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

func inputs() []fixture {
	var fs []fixture
	add := func(name string, n int, bits func(int) uint32) {
		f := fixture{Name: name, Input: make([]uint32, n)}
		for i := range f.Input {
			f.Input[i] = bits(i)
		}
		fs = append(fs, f)
	}
	add("zero", 256, func(int) uint32 { return 0 })
	add("negative-zero", 256, func(int) uint32 { return 0x80000000 })
	add("positive-constant", 256, func(int) uint32 { return 0x3fa00000 })
	add("negative-constant", 256, func(int) uint32 { return 0xbfa00000 })
	add("tiny-mixed", 256, func(i int) uint32 { return []uint32{0, 0x80000000, 0x1e3ce508, 0x9e3ce508}[i%4] })
	add("half-boundaries", 256, func(i int) uint32 {
		return []uint32{0x3f801000, 0x3f803000, 0xbf801000, 0xbf803000, 0x33800000, 0xb3800000, 0x38800000, 0xb8800000}[i%8]
	})
	add("mixed-signs", 256, func(i int) uint32 { return math.Float32bits(float32((i*73)%251-125) / 16) })
	add("consecutive-blocks", 512, func(i int) uint32 { return math.Float32bits(float32((i*97)%503-251) / 32) })
	add("skewed-subblocks", 256, func(i int) uint32 {
		v := float32((i*31)%127-63) / 32
		if i/32%2 == 0 {
			v *= 16
		}
		return math.Float32bits(v)
	})
	return fs
}

// The translation unit includes the unchanged upstream implementation. Section
// GC drops unrelated quantizers. Only ggml's abort diagnostic dependency is stubbed;
// it remains fatal, and no numerical function or macro is replaced.
const cDriver = `
#include <stdarg.h>
#include "ggml-quants.c"
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "reference assertion: %s:%d: ", file, line);
    va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args); abort();
}
int main(void) {
    uint32_t n = 0; float x[512], y[512];
    block_q4_K q4[2]; block_q5_K q5[2]; block_q6_K q6[2];
    if (fread(&n,4,1,stdin)!=1 || (n!=16 && n!=256 && n!=512) || fread(x,4,n,stdin)!=n) return 2;
    if (n==16) {
        uint16_t half[16]; for (int i=0;i<16;i++) half[i]=GGML_FP32_TO_FP16(x[i]);
        if (fwrite(half,2,16,stdout)!=16) return 3;
        return fflush(stdout)!=0;
    }
    quantize_row_q4_K_ref(x,q4,n); dequantize_row_q4_K(q4,y,n);
    if(fwrite(q4,sizeof(block_q4_K),n/256,stdout)!=n/256 || fwrite(y,4,n,stdout)!=n)return 3;
    quantize_row_q5_K_ref(x,q5,n); dequantize_row_q5_K(q5,y,n);
    if(fwrite(q5,sizeof(block_q5_K),n/256,stdout)!=n/256 || fwrite(y,4,n,stdout)!=n)return 3;
    quantize_row_q6_K_ref(x,q6,n); dequantize_row_q6_K(q6,y,n);
    if(fwrite(q6,sizeof(block_q6_K),n/256,stdout)!=n/256 || fwrite(y,4,n,stdout)!=n)return 3;
    return fflush(stdout)!=0;
}
`
