package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAGDNVLLMLanesContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	kernel := read("cuda/gdn_decode_vllm.cuh")
	caller := read("cuda/qwen35_gdn.cuh")
	for _, want := range []string{
		"fa3b9d1377219247060d3c8b49eae199c9e7e154",
		"SPDX-License-Identifier: Apache-2.0",
		"struct IndependentLaneStates",
		"float* recurrent[kMaxTokens]",
		"bool IndependentLanes = false",
		"state = lane_states.recurrent[batch]",
		"tokens = 1",
		"k_qwen35_gdn_post_conv_vllm<Ratio, true>",
		"const dim3 grid((unsigned)nV, (unsigned)lanes)",
		"k_qwen35_gdn_post_conv_vllm",
		"status = qwen35_gdn_launch_status(5));",
		"FAK_GDN_VLLM_LANES_CASE(1)", "FAK_GDN_VLLM_LANES_CASE(2)",
		"FAK_GDN_VLLM_LANES_CASE(3)", "FAK_GDN_VLLM_LANES_CASE(4)",
		"FAK_GDN_VLLM_LANES_CASE(8)",
	} {
		if !strings.Contains(kernel, want) {
			t.Errorf("missing independent-lane contract %q", want)
		}
	}
	body := cudaFunctionBody(t, caller, "fcuda_qwen35_gdn_decode_lanes_f32")
	start := strings.Index(body, `fak_gdn_vllm::eligible(getenv("FAK_CUDA_GDN_POST_CONV")`)
	if start < 0 {
		t.Fatal("real independent-lane entry does not select the retained port")
	}
	arm := body[start:]
	end := strings.Index(arm, "} else {")
	if end < 0 {
		t.Fatal("legacy lane fallback missing")
	}
	selected := strings.Join(strings.Fields(arm[:end]), " ")
	for _, want := range []string{
		"lanes, nK, nV, kHd, vHd, g_capture_open)",
		"status = qwen35_gdn_launch_status(4); if (status != 0) return qwen35_gdn_drain_after_error(status);",
		"laneStates.recurrent[l] = states.recurrent[l]",
		"status = fak_gdn_vllm::dispatch_lanes(",
		"dCore, lanes, nK, nV, convDim, rmsEps); if (status != 0) return qwen35_gdn_drain_after_error(status);",
	} {
		if !strings.Contains(selected, want) {
			t.Errorf("selected branch missing %q", want)
		}
	}
	if strings.Index(selected, "qwen35_gdn_launch_status(4)") > strings.Index(selected, "dispatch_lanes(") {
		t.Fatal("stage-4 fault was not checked before candidate launch")
	}
	after := arm[end:]
	if !strings.Contains(after, "k_qwen35_gdn_qk_norm_panel") ||
		!strings.Contains(after, "k_qwen35_gdn_recurrent_gated_norm_lanes<true>") ||
		!strings.Contains(after, "k_qwen35_gdn_recurrent_gated_norm_lanes<false>") ||
		strings.Index(after, "qwen35_gdn_launch_status(5)") > strings.Index(after, "k_qwen35_gdn_out_proj_lanes") {
		t.Fatal("fallback recurrence or checked projection ordering changed")
	}
	if !strings.Contains(read("cuda_qwen35_decode_batch.go"), "C.fcuda_qwen35_gdn_decode_lanes_f32(") {
		t.Fatal("ordinary batch consumer lost the lane entry")
	}
	if !strings.Contains(read("../../third_party/vllm-gdn-decode/LICENSE"), "Apache License") {
		t.Fatal("retained upstream license missing")
	}
}

// Uses the same exact-source cooperative SIMT machinery and FP64 recurrence as
// the ordinary port. Only the oracle initialization changes to retain FP64 state
// between two independent steps. It does not model CUDA asynchronous copies.
// fak-test:runtime medium est=15s lane=default
// fak-test:justify Compiles and executes one bounded exact-source C++14 lane-isolation witness.
func TestCUDAGDNVLLMLanesHost(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("exact-source cooperative SIMT witness requires Linux ucontext")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("C++14 compiler required for GDN independent-lane source witness: ", err)
	}
	b, err := os.ReadFile("cuda/gdn_decode_vllm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(b)
	const boundary = "template <int Ratio>\nstatic int launch("
	end := strings.Index(source, boundary)
	if end < 0 || strings.Count(source, boundary) != 1 {
		t.Fatal("exact production kernel/eligibility boundary missing")
	}
	const mainBoundary = "int main() {"
	mainStart := strings.Index(cudaGDNVLLMHostMain, mainBoundary)
	if mainStart < 0 || strings.Count(cudaGDNVLLMHostMain, mainBoundary) != 1 {
		t.Fatal("existing independent FP64 oracle boundary changed")
	}
	oracle := cudaGDNVLLMHostMain[:mainStart]
	const initialize = "state.assign(f.state.begin()+4,f.state.end()-4);"
	if strings.Count(oracle, initialize) != 1 {
		t.Fatal("FP64 reference state initializer changed")
	}
	oracle = strings.Replace(oracle, initialize, "if (state.empty()) "+initialize, 1)
	fixture := cudaGDNVLLMHostPreamble + source[:end] + "\n}\n" + oracle + cudaGDNVLLMLanesMain
	t.Logf("production source sha256=%x; host SIMT only, CUDA/device/performance UNRUN", sha256.Sum256(b))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "gdn_lanes.cpp"), filepath.Join(dir, "gdn_lanes")
	if err := os.WriteFile(cpp, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++14", "-O1", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual GDN lane source: %v\n%s", err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("actual GDN lane source arithmetic/state isolation: %v\n%s", err, out)
	}
	if strings.TrimSpace(string(out)) != "PASS exact GDN lane source, FP64 carried state, distinct lanes and guards; CUDA/device/performance unrun" {
		t.Fatalf("incomplete lane witness: %s", out)
	}
}

const cudaGDNVLLMLanesMain = `
template<int Ratio> static void check_lanes(int lanes, int nk) {
  const int nv=nk*Ratio, width=(2*nk+nv)*128, values=nv*128;
  std::vector<Fixture> f;
  f.reserve(lanes);
  for(int lane=0;lane<lanes;++lane) {
    f.emplace_back(1,nk,Ratio);
    for(size_t i=4;i+4<f.back().state.size();++i)
      f.back().state[i]+=float(lane)*.0001f;
  }
  std::vector<std::vector<double>> want_state(lanes), want_out(lanes);
  fak_gdn_vllm::IndependentLaneStates states={};
  for(int lane=0;lane<lanes;++lane) states.recurrent[lane]=f[lane].state.data()+4;
  const auto initial_table=states;
  for(int step=0;step<2;++step) {
    std::vector<float> mixed(lanes*width), a(lanes*nv), b(lanes*nv), z(lanes*values);
    std::vector<float> output(lanes*values+8,NAN);
    for(int i=0;i<4;++i) output[i]=output[output.size()-1-i]=123456.0f;
    for(int lane=0;lane<lanes;++lane) {
      f[lane].mixed=data(width,11+lane+step*19,.003f);
      f[lane].a=data(nv,23+lane+step*17,.01f);
      f[lane].b=data(nv,41+lane+step*13,.01f);
      f[lane].z=data(values,53+lane+step*7,.008f);
      if(lane==lanes-1) f[lane].a[0]=22.0f;
      reference(f[lane],want_state[lane],want_out[lane]);
      std::copy(f[lane].mixed.begin(),f[lane].mixed.end(),mixed.begin()+lane*width);
      std::copy(f[lane].a.begin(),f[lane].a.end(),a.begin()+lane*nv);
      std::copy(f[lane].b.begin(),f[lane].b.end(),b.begin()+lane*nv);
      std::copy(f[lane].z.begin(),f[lane].z.end(),z.begin()+lane*values);
    }
    const auto mixed_before=mixed, a_before=a, b_before=b, z_before=z;
    const auto alog_before=f[0].alog, bias_before=f[0].bias, norm_before=f[0].norm;
    kernel_entry=[&] {
      fak_gdn_vllm::k_qwen35_gdn_post_conv_vllm<Ratio,true>(
          mixed.data(),a.data(),b.data(),f[0].alog.data(),f[0].bias.data(),
          nullptr,z.data(),f[0].norm.data(),output.data()+4,
          1,nk,nv,width,1e-6f,states);
    };
    for(int order=0;order<lanes;++order) {
      const int lane=step==0 ? lanes-1-order : (order+1)%lanes;
      for(int head=0;head<nv;++head) { blockIdx=dim3(head,lane); run_block(); }
    }
    for(int lane=0;lane<lanes;++lane) {
      close("independent carried state",f[lane].state.data()+4,want_state[lane]);
      close("independent output",output.data()+4+lane*values,want_out[lane]);
      assert(states.recurrent[lane]==initial_table.recurrent[lane]);
      for(int i=0;i<4;++i)
        assert(f[lane].state[i]==123456 && f[lane].state[f[lane].state.size()-1-i]==123456);
    }
    assert(mixed==mixed_before && a==a_before && b==b_before && z==z_before);
    assert(f[0].alog==alog_before && f[0].bias==bias_before && f[0].norm==norm_before);
    for(int i=0;i<4;++i) assert(output[i]==123456 && output[output.size()-1-i]==123456);
    for(int lane=lanes;lane<8;++lane) assert(states.recurrent[lane]==nullptr);
  }
}
int main() {
  using fak_gdn_vllm::eligible;
  for(int ratio:{1,2,3,4,8}) for(int lanes=1;lanes<=8;++lanes)
    assert(eligible("vllm",lanes,2,2*ratio,128,128,false));
  assert(!eligible(nullptr,1,1,1,128,128,false));
  assert(!eligible("",1,1,1,128,128,false));
  assert(!eligible("vllmx",1,1,1,128,128,false));
  assert(!eligible("vllm",1,1,1,128,128,true));
  for(int lanes:{0,9}) assert(!eligible("vllm",lanes,1,1,128,128,false));
  for(int ratio:{5,6,7}) assert(!eligible("vllm",1,1,ratio,128,128,false));
  assert(!eligible("vllm",1,0,1,128,128,false));
  assert(!eligible("vllm",1,2,3,128,128,false));
  assert(!eligible("vllm",1,1,1,64,128,false));
  assert(!eligible("vllm",1,1,1,128,64,false));
  check_lanes<1>(1,1); check_lanes<2>(2,2); check_lanes<3>(8,1);
  check_lanes<4>(3,1); check_lanes<8>(2,1);
  std::puts("PASS exact GDN lane source, FP64 carried state, distinct lanes and guards; CUDA/device/performance unrun");
}
`
