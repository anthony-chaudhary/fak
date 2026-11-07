package agent

import (
	"context"
	"crypto/sha256"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This source-only witness checks the actual transport certificate for #14198. The
// packet producer below is a named host substitute, not CUDA or served decode.
// It proves conditional compaction against the actual sampler and its existing
// full-softmax oracle; it does not qualify a producer, lifetime, D2H or speed.
// fak-test:runtime fast est=10s lane=default
func TestSamplingCompactTopKSourceContract(t *testing.T) {
	production := compactContractDeclarations(t, "inkernel_sampling.go", []string{
		"const.clusterSamplingMinVocabulary", "type.samplingTruncationPath", "const.samplingTruncationCPU",
		"type.samplingTruncationEnvelope", "func.selectSamplingTruncationPath",
		"type.samplerWorkspace", "type.probSorter", "probSorter.Len", "probSorter.Less", "probSorter.Swap",
		"samplerWorkspace.resize", "samplerWorkspace.orderBuf", "samplerWorkspace.heapBuf", "samplerWorkspace.valBuf",
		"samplerWorkspace.applyCPUSamplingTruncation", "func.sampleLogitsMinPCoreWS",
		"samplerWorkspace.smallTopKSoftmax", "func.minPTruncate", "func.sampleProbabilityRow",
		"samplerWorkspace.nucleusTruncate", "samplerWorkspace.nucleusTruncatePrefix", "samplerWorkspace.nucleusTruncateFull",
		"const.topKSmallKThreshold", "func.topKRankWorse", "func.topKHeapLess", "func.topKSiftDown",
		"func.topKSelectHeap", "func.topKRankSort", "func.topKTruncateSmallK",
		"samplerWorkspace.topKTruncate", "samplerWorkspace.topKHeap",
	})
	oracle := compactContractDeclarations(t, "inkernel_sampling_reconstruction_test.go", []string{"func.fullSoftmaxReconstructionOracle"})
	certificate := compactContractDeclarations(t, "../compute/sampling_candidates.go", []string{
		"const.SamplingCandidateMaxK", "type.SamplingCandidatePacket", "SamplingCandidatePacket.Certify",
	})
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":           "module samplingcompactcontract\n\ngo 1.26.10\n",
		"sampler.go":       "package compactcontract\nimport (\"math\"; \"math/rand\"; \"sort\")\n" + production,
		"certificate.go":   "package compactcontract\nimport \"math\"\n" + certificate,
		"oracle.go":        "package compactcontract\nimport (\"math\"; \"math/rand\")\n" + oracle,
		"contract_test.go": compactSamplingContractFixture,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", "-v", "-trimpath", "-buildvcs=false", "-p=1", "-parallel=1", "-count=1", "-timeout=60s", ".")
	cmd.Dir = dir
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		switch key {
		case "GOWORK", "GOTOOLCHAIN", "GOFLAGS", "GOPROXY", "GOSUMDB", "CGO_ENABLED":
			continue
		}
		cmd.Env = append(cmd.Env, entry)
	}
	cmd.Env = append(cmd.Env, "GOWORK=off", "GOTOOLCHAIN=local", "GOFLAGS=", "GOPROXY=off", "GOSUMDB=off", "CGO_ENABLED=0")
	cmd.WaitDelay = 5 * time.Second
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("actual-source compact contract failed: %v (context: %v)\n%s", err, ctx.Err(), out)
	}
	t.Logf("actual-source compact contract:\n%s", out)
}

// Select declarations by their parsed identities, retaining their source bytes
// rather than rewriting arithmetic or manually copying the production bodies.
func compactContractDeclarations(t *testing.T, path string, selectors []string) string {
	t.Helper()
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("source %s sha256=%x", path, sha256.Sum256(source))
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, source, 0)
	if err != nil {
		t.Fatal(err)
	}
	wanted := make(map[string]int, len(selectors))
	for _, selector := range selectors {
		wanted[selector] = 0
	}
	var body strings.Builder
	for _, decl := range f.Decls {
		var names []string
		switch d := decl.(type) {
		case *ast.FuncDecl:
			owner := "func"
			if d.Recv != nil {
				receiver := d.Recv.List[0].Type
				if ptr, ok := receiver.(*ast.StarExpr); ok {
					receiver = ptr.X
				}
				id, ok := receiver.(*ast.Ident)
				if !ok {
					t.Fatalf("unsupported receiver in %s", path)
				}
				owner = id.Name
			}
			names = append(names, owner+"."+d.Name.Name)
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					names = append(names, "type."+s.Name.Name)
				case *ast.ValueSpec:
					for _, name := range s.Names {
						names = append(names, d.Tok.String()+"."+name.Name)
					}
				}
			}
		}
		selected := false
		for _, name := range names {
			if _, ok := wanted[name]; ok {
				wanted[name]++
				selected = true
			}
		}
		if selected {
			start, end := fset.Position(decl.Pos()).Offset, fset.Position(decl.End()).Offset
			fmt.Fprintf(&body, "%s\n\n", source[start:end])
		}
	}
	for name, count := range wanted {
		if count != 1 {
			t.Fatalf("%s: declaration %s found %d times, want exactly once", path, name, count)
		}
	}
	return body.String()
}

const compactSamplingContractFixture = `package compactcontract

import (
    "math"
    "math/rand"
    "sort"
    "testing"
)

type candidatePacket struct {
    vocab int
    ids []int
    raw []float32
    wholeRowValid bool
}

// HOST SUBSTITUTE: a future CUDA producer must separately prove this complete
// row scan, global top-(k+1) ordering and bit-preserving raw-value transport.
func hostCandidatePacket(row []float32, k int) candidatePacket {
    p := candidatePacket{vocab: len(row)}
    if k < 1 || k > 64 || k >= len(row) { return p }
    finite := false
    for _, x := range row {
        if x != x || x > math.MaxFloat32 { return p }
        finite = finite || x >= -math.MaxFloat32
    }
    if !finite { return p }
    ids := make([]int, len(row))
    for i := range ids { ids[i] = i }
    sort.Slice(ids, func(i,j int) bool {
        if row[ids[i]] != row[ids[j]] { return row[ids[i]] > row[ids[j]] }
        return ids[i] < ids[j]
    })
    p.ids = append([]int(nil), ids[:k+1]...)
    for _, id := range p.ids { p.raw = append(p.raw, row[id]) }
    p.wholeRowValid = true
    return p
}

// This adapter only constructs the packet. Certify is copied verbatim from
// actual compute source above, including raw ranking and original-ID ordering.
func compactCertificate(p candidatePacket, k int, temp float64) ([]int, []float32, bool) {
    if len(p.ids)!=len(p.raw) || len(p.ids)>SamplingCandidateMaxK+1 { return nil,nil,false }
    actual:=SamplingCandidatePacket{Vocab:p.vocab,Count:len(p.ids),Valid:p.wholeRowValid}
    copy(actual.IDs[:],p.ids)
    copy(actual.Raw[:],p.raw)
    if !actual.Certify(temp,k) { return nil,nil,false }
    return actual.IDs[:actual.Count],actual.Raw[:actual.Count],true
}

func cloneF32(row []float32) []float32 { return append([]float32(nil),row...) }
func sameF32(t *testing.T, a,b []float32) {
    t.Helper()
    if len(a)!=len(b) { t.Fatal("source row changed length") }
    for i:=range a { if math.Float32bits(a[i])!=math.Float32bits(b[i]) { t.Fatalf("source row mutated at %d",i) } }
}

func TestCompactedSamplerLaw(t *testing.T) {
    rows := [][]float32{
        {0,1,2,3}, {3,3,2,1}, {0,1,1,3},
        {float32(math.Copysign(0,-1)),0,-1,-2,1},
        {float32(math.Copysign(0,-1)),0,-1,-2},
        {math.MaxFloat32,-math.MaxFloat32,1,-1},
        {float32(math.Inf(-1)),3,1,float32(math.Inf(-1)),2},
        {0,1,2,math.MaxFloat32},
        {math.SmallestNonzeroFloat32,0,-math.SmallestNonzeroFloat32,2*math.SmallestNonzeroFloat32},
    }
    rng:=rand.New(rand.NewSource(14198))
    for _,n:=range []int{3,17,65,257,1025} {
        for trial:=0;trial<3;trial++ {
            row:=make([]float32,n)
            for i:=range row { row[i]=float32(rng.NormFloat64()*5) }
            if trial==1 { for i:=range row { row[i]=float32(i%5) } }
            rows=append(rows,row)
        }
    }
    accepted,refused,boundaries:=0,0,0
    var full,compact samplerWorkspace
    for ri,row:=range rows {
        before:=cloneF32(row)
        for _,k:=range []int{1,2,4,40,64} {
            packet:=hostCandidatePacket(row,k)
            for _,temp:=range []float64{.7,1,1e-300,1e300,1e-44} {
                ids,raw,ok:=compactCertificate(packet,k,temp)
                if !ok { refused++; continue }
                accepted++
                for _,minP:=range []float64{0,.05,1.1} {
                    for _,topP:=range []float64{0,math.SmallestNonzeroFloat64,.5,.9} {
                        want:=fullSoftmaxReconstructionOracle(row,temp,topP,k,minP,nil,.3,true,&full)
                        local:=sampleLogitsMinPCoreWS(raw,temp,topP,k,minP,nil,.3,true,&compact)
                        if local<0 || local>=len(ids) || ids[local]!=want || full.nonFinite!=compact.nonFinite || math.Float64bits(full.mass)!=math.Float64bits(compact.mass) {
                            t.Fatalf("row=%d k=%d temp=%g minP=%g topP=%g token=%d/%d mass=%g/%g",ri,k,temp,minP,topP,local,want,compact.mass,full.mass)
                        }
                        restored:=make([]float64,len(row))
                        for i,id:=range ids { restored[id]=compact.probs[i] }
                        for id:=range restored {
                            if math.Float64bits(restored[id])!=math.Float64bits(full.probs[id]) { t.Fatalf("weight mismatch row=%d k=%d id=%d",ri,k,id) }
                        }
                        coins:=[]float64{0,math.Nextafter(1,0)}
                        var cum float64
                        for _,w:=range full.probs {
                            cum+=w
                            if w>0 { boundary:=cum/full.mass; coins=append(coins,math.Nextafter(boundary,0),boundary,math.Nextafter(boundary,1)) }
                        }
                        for _,coin:=range coins {
                            a:=sampleProbabilityRow(full.probs,full.mass,coin)
                            b:=sampleProbabilityRow(compact.probs,compact.mass,coin)
                            if b<0 || b>=len(ids) || ids[b]!=a { t.Fatalf("CDF mismatch row=%d k=%d coin=%.17g",ri,k,coin) }
                            boundaries++
                        }
                    }
                }
            }
        }
        sameF32(t,before,row)
    }
    if accepted==0 || refused==0 || boundaries==0 { t.Fatal("vacuous compaction witness") }
    t.Logf("accepted certificates=%d refused=%d CDF draws=%d",accepted,refused,boundaries)
}

func TestNucleusPrefixToFullTransition(t *testing.T) {
    row:=make([]float32,1025)
    for i:=range row { row[i]=-40 }
    for i,id:=range []int{2,11,87,511,900} { row[id]=-float32(i)*.1 }
    ids,raw,ok:=compactCertificate(hostCandidatePacket(row,4),4,.7)
    if !ok { t.Fatal("expected strict-gap fixture") }
    var full,compact samplerWorkspace
    fullSoftmaxReconstructionOracle(row,.7,0,4,0,nil,.3,true,&full)
    sampleLogitsMinPCoreWS(raw,.7,0,4,0,nil,.3,true,&compact)
    var a,b samplerWorkspace
    af,bf:=append([]float64(nil),full.probs...),append([]float64(nil),compact.probs...)
    _,ap:=a.nucleusTruncatePrefix(af,.9*full.mass)
    _,bp:=b.nucleusTruncatePrefix(bf,.9*compact.mass)
    if !ap || bp { t.Fatalf("expected prefix/full branch split, got %v/%v",ap,bp) }
    want:=fullSoftmaxReconstructionOracle(row,.7,.9,4,0,nil,.3,true,&full)
    got:=sampleLogitsMinPCoreWS(raw,.7,.9,4,0,nil,.3,true,&compact)
    if ids[got]!=want || math.Float64bits(full.mass)!=math.Float64bits(compact.mass) { t.Fatal("branch transition changed result") }
    for i,id:=range ids { if math.Float64bits(full.probs[id])!=math.Float64bits(compact.probs[i]) { t.Fatal("branch transition changed weight") } }
}

func TestCertificateRefusals(t *testing.T) {
    good:=hostCandidatePacket([]float32{0,3,1,2},2)
    // Physical atomic emission may be unordered. The actual certificate sorts
    // raw rank first and returns only the original-ID-ordered retained prefix.
    shuffled:=candidatePacket{vocab:4,ids:[]int{3,1,2},raw:[]float32{2,3,1},wholeRowValid:true}
    if ids,_,ok:=compactCertificate(shuffled,2,1); !ok || ids[0]!=1 || ids[1]!=3 { t.Fatal("unordered complete packet was not normalized") }
    bad:=[]candidatePacket{
        {vocab:4,ids:good.ids,raw:good.raw},
        {vocab:4,ids:[]int{1,3},raw:[]float32{3,2},wholeRowValid:true},
        {vocab:4,ids:[]int{1,1,2},raw:good.raw,wholeRowValid:true},
        {vocab:4,ids:[]int{1,4,2},raw:good.raw,wholeRowValid:true},
        {vocab:4,ids:[]int{-1,3,2},raw:good.raw,wholeRowValid:true},
        {vocab:4,ids:good.ids,raw:[]float32{3,float32(math.NaN()),1},wholeRowValid:true},
        {vocab:4,ids:good.ids,raw:[]float32{float32(math.Inf(1)),2,1},wholeRowValid:true},
        {vocab:4,ids:good.ids,raw:[]float32{float32(math.Inf(-1)),float32(math.Inf(-1)),float32(math.Inf(-1))},wholeRowValid:true},
    }
    for i,p:=range bad { if _,_,ok:=compactCertificate(p,2,1); ok { t.Fatalf("accepted malformed packet %d",i) } }
    for _,temp:=range []float64{0,-1,math.NaN(),math.Inf(1),math.Inf(-1)} { if _,_,ok:=compactCertificate(good,2,temp); ok { t.Fatalf("accepted temperature %g",temp) } }
    for _,k:=range []int{-1,0,4,65} { if _,_,ok:=compactCertificate(good,k,1); ok { t.Fatalf("accepted k=%d",k) } }
    for _,row:=range [][]float32{{0,2,1},{0,1,2,math.MaxFloat32},{3,2,float32(math.NaN()),1},{3,2,float32(math.Inf(1)),1},{float32(math.Inf(-1)),float32(math.Inf(-1))}} {
        k,temp:=1,1e300
        if len(row)==4 && row[3]==math.MaxFloat32 { k,temp=2,1 }
        if _,_,ok:=compactCertificate(hostCandidatePacket(row,k),k,temp); ok { t.Fatal("accepted collapse or invalid row") }
    }
}

func TestSeededDrawAndFallbackState(t *testing.T) {
    rows:=[][]float32{
        {0,1,2,3,4}, {3,3,3,1,0}, {3,2,float32(math.NaN()),1,0},
        {3,2,float32(math.Inf(1)),1,0},
        {float32(math.Inf(-1)),float32(math.Inf(-1)),float32(math.Inf(-1))},
        {0,1,2,math.MaxFloat32}, {0,1,2,3,4,5,6},
    }
    a,b:=rand.New(rand.NewSource(771)),rand.New(rand.NewSource(771))
    var full,work samplerWorkspace
    accepted,refused:=0,0
    for i:=0;i<112;i++ {
        row:=rows[i%len(rows)]
        ids,raw,ok:=compactCertificate(hostCandidatePacket(row,2),2,.7)
        want:=fullSoftmaxReconstructionOracle(row,.7,.9,2,.05,b,0,false,&full)
        var got int
        if ok {
            accepted++
            got=ids[sampleLogitsMinPCoreWS(raw,.7,.9,2,.05,a,0,false,&work)]
        } else {
            refused++
            // Same original row, no second Step and no preflight random draw.
            got=sampleLogitsMinPCoreWS(row,.7,.9,2,.05,a,0,false,&work)
        }
        if got!=want || work.nonFinite!=full.nonFinite || math.Float64bits(work.mass)!=math.Float64bits(full.mass) { t.Fatalf("seeded/fallback mismatch step=%d",i) }
    }
    if accepted==0 || refused==0 || a.Int63()!=b.Int63() { t.Fatal("RNG state or coverage mismatch") }
}

func TestNegativeControls(t *testing.T) {
    // Rank order is not CDF token order. This deliberately wrong mapping must fail.
    row:=[]float32{0,1,2,3,4}
    p:=hostCandidatePacket(row,3)
    var full,wrong samplerWorkspace
    want:=fullSoftmaxReconstructionOracle(row,1,0,3,0,nil,0,true,&full)
    got:=p.ids[sampleLogitsMinPCoreWS(p.raw[:3],1,0,3,0,nil,0,true,&wrong)]
    if got==want { t.Fatal("unsorted-ID negative control was not discriminating") }
    // A huge temperature collapses canonical weights; raw top-1 is then wrong.
    row=[]float32{0,2,1}
    p=hostCandidatePacket(row,1)
    if _,_,ok:=compactCertificate(p,1,1e300); ok { t.Fatal("collapse certificate unexpectedly accepted") }
    want=fullSoftmaxReconstructionOracle(row,1e300,0,1,0,nil,0,true,&full)
    got=p.ids[sampleLogitsMinPCoreWS(p.raw[:1],1e300,0,1,0,nil,0,true,&wrong)]
    if got==want { t.Fatal("missing-certificate negative control was not discriminating") }
}
`

var samplingAllocationHost = flag.Bool("fak-sampling-allocation-host", false, "run host-only CUDA allocation-status failure injection")

// Exact native cleanup/status source with fake CUDA edges, not device evidence.
// fak-test:runtime slow est=1s lane=optin
func TestSamplingCandidateAllocationStatusContract(t *testing.T) {
	if !*samplingAllocationHost {
		t.Skip("set -fak-sampling-allocation-host")
	}
	compiler, err := exec.LookPath("g++")
	if err != nil {
		t.Fatal("required host C++ compiler unavailable")
	}
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	native := read("../compute/cuda/sampling_candidates.cuh")
	end := strings.Index(native, "// 1 is supported")
	if end < 0 {
		t.Fatal("native status/release boundary missing")
	}
	alloc := strings.TrimPrefix(read("../compute/cuda/alloc_status.h"), "#pragma once\n")
	program := `
#include <cstddef>
#include <cstdio>
using cudaError_t = int;
static constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorNotReady = 34;
static int g_cuda_cleanup_status, pending, frees;
static bool uncertain;
static cudaError_t cudaGetLastError() { int e = pending; pending = 0; return e; }
static bool v41_candidates_stream_uncertain() { return uncertain || g_cuda_cleanup_status != 0; }
static void fcuda_free(void *) { if (!v41_candidates_stream_uncertain()) ++frees; }
` + alloc + native[:end] + `
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed at line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
  int owner = 0;
  pending = 719;
  CHECK(fcuda_sampling_candidates_release(&owner) == 719);
  CHECK(g_cuda_cleanup_status == 719 && frees == 0);
  pending = 700;
  CHECK(fcuda_sampling_candidates_release(&owner) == 719);
  CHECK(g_cuda_cleanup_status == 719 && frees == 0);
  g_cuda_cleanup_status = 0; pending = 0; uncertain = true;
  CHECK(fcuda_sampling_candidates_release(&owner) == cudaErrorNotReady && frees == 0);
  uncertain = false; pending = cudaErrorMemoryAllocation;
  CHECK(fcuda_alloc_status(cudaErrorMemoryAllocation) == cudaErrorMemoryAllocation);
  CHECK(pending == 0 && fcuda_sampling_candidates_release(&owner) == 0 && frees == 1);
  CHECK(fcuda_sampling_candidates_release(nullptr) == 0 && frees == 1);
  CHECK(sampling_candidates_post_submit(cudaSuccess) == 0);
  CHECK(fcuda_sampling_candidates_workspace_bytes() != 0);
  return 0;
}
`
	dir := t.TempDir()
	source, binary := filepath.Join(dir, "probe.cpp"), filepath.Join(dir, "probe")
	if err := os.WriteFile(source, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", source, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("host contract compile: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("host contract execution: %v\n%s", err, out)
	}
	binding := read("../compute/cuda_sampling_candidates.go")
	if strings.Count(binding, "err = allocationError(") != 2 || !strings.Contains(binding, "C.fcuda_sampling_candidates_allocation_status()") {
		t.Fatal("both allocation results must pass checked native status before unwind")
	}
}
