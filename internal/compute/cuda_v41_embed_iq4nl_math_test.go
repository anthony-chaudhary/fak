package compute

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// One micro-scale fixture qualifies the actual portable header's IQ4_NL math
// and pre-dispatch geometry only. No CUDA wrapper, device or model claim.
// fak-test:runtime medium est=1s lane=default
// fak-test:justify Compiles the actual IQ4_NL row decoder/geometry with a host C++17 compiler.
func TestCUDAV41IQ4NLEmbeddingMath(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("C++17 compiler required for IQ4_NL host witness")
	}
	const hidden, blockBytes = 512, 18
	raw := make([]byte, 3*(hidden/32)*blockBytes)
	// Mixed finite scales, including signed zero, a subnormal and a maximum
	// finite half. Deliberately asymmetric nibbles expose adjacent-order bugs.
	scales := [...]uint16{0x3c00, 0xb123, 0x0001, 0x8000, 0x7bff, 0x3555, 0x0000, 0xbc00}
	for block := 0; block < len(raw)/blockBytes; block++ {
		binary.LittleEndian.PutUint16(raw[block*blockBytes:], scales[block%len(scales)])
		for j := 0; j < 16; j++ {
			raw[block*blockBytes+2+j] = byte((j+block)&15) | byte((15-j+3*block)&15)<<4
		}
	}
	ids := [...]int{2, 0, 2, 1}
	half := func(h uint16) float32 {
		mant, exp := int(h&1023), int((h>>10)&31)
		value := math.Ldexp(float64(mant), -24)
		if exp != 0 {
			value = math.Ldexp(float64(1024+mant), exp-25)
		}
		if h&0x8000 != 0 {
			value = math.Copysign(value, -1)
		}
		return float32(value)
	}
	// Flat-output scalar oracle, independent of the upstream 32-lane map.
	// Matches pinned Fak ggufload.dequantIQ4NLScalar and model.iq4nlDequantBlock.
	values := [...]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
	var fixture strings.Builder
	fixture.WriteString("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n")
	for _, name := range []string{"v41_embed_gather_math.h", "v41_embed_iq4nl.h"} {
		path, err := filepath.Abs(filepath.Join("cuda", name))
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s sha256=%x", name, sha256.Sum256(data))
		fmt.Fprintf(&fixture, "#include %q\n", filepath.ToSlash(path))
	}
	// Bind the tested lookup to the existing production CUDA initializer,
	// without compiling its unrelated GEMV kernels or creating a second table.
	codebook, err := os.ReadFile(filepath.Join("cuda", "iq_gemv.cuh"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("iq_gemv.cuh sha256=%x", sha256.Sum256(codebook))
	decl := regexp.MustCompile(`(?s)__device__ const signed char fcuda_kvalues_iq4nl\[16\] = \{([^}]+)\};`).FindAllSubmatch(codebook, -1)
	if len(decl) != 1 {
		t.Fatal("expected one canonical fcuda_kvalues_iq4nl initializer")
	}
	fmt.Fprintf(&fixture, "static const signed char iq4nl_values[16]={%s};\n", decl[0][1])
	fixture.WriteString("static const uint8_t raw[]={")
	for _, value := range raw {
		fmt.Fprintf(&fixture, "%d,", value)
	}
	fixture.WriteString("};\nstatic const uint32_t want[]={")
	for _, id := range ids {
		for col := 0; col < hidden; col++ {
			block := raw[(id*(hidden/32)+col/32)*blockBytes:]
			lane := col % 32
			code := (block[2+lane%16] >> (4 * (lane / 16))) & 15
			value := half(binary.LittleEndian.Uint16(block)) * values[code]
			fmt.Fprintf(&fixture, "0x%08xu,", math.Float32bits(value))
		}
	}
	fixture.WriteString("};\n")
	fixture.WriteString(v41IQ4NLMathMain)
	t.Logf("representative fixture sha256=%x", sha256.Sum256([]byte(fixture.String())))
	dir := t.TempDir()
	source, executable := filepath.Join(dir, "iq4nl_embedding.cc"), filepath.Join(dir, "iq4nl_embedding.test")
	if err := os.WriteFile(source, []byte(fixture.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-ffp-contract=off", source, "-o", executable).CombinedOutput(); err != nil {
		t.Fatalf("compile shared IQ4_NL decoder: %v\n%s", err, output)
	}
	binaryBytes, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("host executable sha256=%x", sha256.Sum256(binaryBytes))
	run, cancelRun := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancelRun()
	output, err := exec.CommandContext(run, executable).CombinedOutput()
	if err != nil || string(output) != "PASS IQ4_NL representative gather and pre-dispatch refusal; GPU unrun\n" {
		t.Fatalf("IQ4_NL shared decoder witness: %v\n%s", err, output)
	}
	t.Log(strings.TrimSpace(string(output)))
}

const v41IQ4NLMathMain = `
int main() {
  const intptr_t ids[]={2,0,2,1};
  uint64_t rb=0,ob=0;
  if(!v41_eg_iq4nl_geometry(ids,4,512,3,&rb,&ob)||rb!=288||ob!=8192)return 1;
  struct {uint32_t left;float data[2048];uint32_t right;} out;
  out.left=0x12345678;out.right=0x87654321;
  memset(out.data,0xa5,sizeof(out.data));
  for(int row=0;row<4;++row)
    for(int block=0;block<2;++block)
      for(int tid=0;tid<32;++tid)
        v41_eg_iq4nl_thread(raw+ids[row]*rb+block*144,out.data+row*512+block*256,tid,iq4nl_values);
  for(int i=0;i<2048;++i) {
    uint32_t got;memcpy(&got,out.data+i,4);
    if(got!=want[i]) {fprintf(stderr,"row=%d col=%d got=%08x want=%08x\n",i/512,i%512,got,want[i]);return 2;}
  }
  if(out.left!=0x12345678||out.right!=0x87654321)return 3;
  // Single refusal companion: malformed width, final bad/negative ID and
  // overflow-sized table/output must leave geometry outputs untouched.
  const intptr_t invalid[]={2,3},negative[]={2,-1};
  rb=17;ob=19;
  if(v41_eg_iq4nl_geometry(ids,4,511,3,&rb,&ob)||
     v41_eg_iq4nl_geometry(invalid,2,512,3,&rb,&ob)||
     v41_eg_iq4nl_geometry(negative,2,512,3,&rb,&ob)||
     v41_eg_iq4nl_geometry(ids,4,512,INT_MAX,&rb,&ob)||
     v41_eg_iq4nl_geometry(ids,INT_MAX,512,3,&rb,&ob)||
     rb!=17||ob!=19)return 4;
  puts("PASS IQ4_NL representative gather and pre-dispatch refusal; GPU unrun");
}
`
