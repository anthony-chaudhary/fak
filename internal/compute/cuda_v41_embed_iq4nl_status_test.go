package compute

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// One host fixture checks the actual shared IQ4_NL math, Engram fault-word
// helpers and poison store. CUDA launch/reduction/lifetime remain unqualified.
// fak-test:runtime medium est=1s lane=default
// fak-test:justify Compiles the actual IQ4_NL finite/status helpers with a host C++17 compiler.
func TestCUDAV41IQ4NLRowFiniteStatus(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("C++17 compiler required for IQ4_NL status host witness")
	}
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
	// Reuse the canonical production initializer, not another device codebook.
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
	fixture.WriteString(v41IQ4NLStatusMain)
	t.Logf("finite/status fixture sha256=%x", sha256.Sum256([]byte(fixture.String())))
	dir := t.TempDir()
	source, executable := filepath.Join(dir, "iq4nl_status.cc"), filepath.Join(dir, "iq4nl_status.test")
	if err := os.WriteFile(source, []byte(fixture.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-ffp-contract=off", source, "-o", executable).CombinedOutput(); err != nil {
		t.Fatalf("compile shared IQ4_NL status helpers: %v\n%s", err, output)
	}
	binaryBytes, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("host executable sha256=%x", sha256.Sum256(binaryBytes))
	run, cancelRun := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancelRun()
	output, err := exec.CommandContext(run, executable).CombinedOutput()
	if err != nil || string(output) != "PASS IQ4_NL row finite/status/poison contract; CUDA execution unrun\n" {
		t.Fatalf("IQ4_NL finite/status witness: %v\n%s", err, output)
	}
	t.Log(strings.TrimSpace(string(output)))
}

const v41IQ4NLStatusMain = `
int main() {
  // One selected-row fixture: the clean row has a NaN scale outside the
  // requested prefix; the faulty row has +Inf at element 64 and NaN at 96.
  // Selection repeats the fault around a clean row to expose stale status.
  uint8_t raw[2*144]={};
  for(int row=0;row<2;++row)for(int block=0;block<8;++block) {
    uint8_t *p=raw+row*144+block*18;p[1]=0x3c;
    for(int j=0;j<16;++j)p[2+j]=(uint8_t)(j|((15-j)<<4));
  }
  raw[7*18]=1;raw[7*18+1]=0x7e;
  raw[144+2*18+1]=0x7c;
  raw[144+3*18]=1;raw[144+3*18+1]=0x7e;
  const intptr_t ids[]={1,0,1};
  const int rows=3,dim=97;
  uint64_t tb=0,ob=0,sb=0;
  if(!v41_eg_iq4nl_rows_geometry(ids,rows,dim,2,&tb,&ob,&sb)||tb!=288||ob!=1164||sb!=12)return 1;
  struct {uint32_t left;float data[rows*dim];uint32_t right;} out;
  out.left=0x12345678;out.right=0x87654321;
  memset(out.data,0xa5,sizeof(out.data));
  struct {uint32_t left,words[rows],right;} status={0x11223344,{0xffffffffu,0xffffffffu,0xffffffffu},0x55667788};
  for(int row=0;row<rows;++row) {
    float decoded[256];
    for(int tid=0;tid<32;++tid)v41_eg_iq4nl_thread(raw+ids[row]*144,decoded,tid,iq4nl_values);
    uint32_t fault=UINT32_MAX;
    // Reverse traversal deliberately encounters later faults first. This is
    // a host reduction of actual keys, not a CUDA atomic/barrier witness.
    for(int j=dim-1;j>=0;--j) {
      out.data[row*dim+j]=decoded[j];
      const uint32_t key=v41_eg_row_fault_key(decoded[j],(uint32_t)j);
      if(key<fault)fault=key;
    }
    status.words[row]=v41_eg_row_fault_word(fault);
  }
  // Existing V41EngramRowFault: 130 -> j=64, kind=1 (not a NaN byte).
  if(status.words[0]!=130||status.words[1]!=0||status.words[2]!=130)return 2;
  if(out.left!=0x12345678||out.right!=0x87654321||status.left!=0x11223344||status.right!=0x55667788)return 3;
  uint32_t targetBits[]={0x3f800000u,0x80000000u,0xbf800000u};
  float target[3];memcpy(target,targetBits,sizeof(target));
  if(!v41_eg_row_poison_geometry(rows,3,1))return 4;
  v41_eg_row_poison(target+1,status.words[1]);
  uint32_t got[3];memcpy(got,target,sizeof(got));
  if(memcmp(got,targetBits,sizeof(got)))return 5;
  uint32_t seen=0;for(int row=0;row<rows;++row)seen|=status.words[row];
  v41_eg_row_poison(target+1,seen);memcpy(got,target,sizeof(got));
  if(got[0]!=targetBits[0]||got[1]!=0x7fc00000u||got[2]!=targetBits[2])return 6;
  // A bad final ID refuses the entire call, before output extents are set.
  const intptr_t invalid[]={1,0,2};tb=17;ob=19;sb=23;
  if(v41_eg_iq4nl_rows_geometry(invalid,rows,dim,2,&tb,&ob,&sb)||tb!=17||ob!=19||sb!=23)return 7;
  puts("PASS IQ4_NL row finite/status/poison contract; CUDA execution unrun");
}
`
