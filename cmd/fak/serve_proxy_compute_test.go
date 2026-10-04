package main

import (
	"flag"
	"testing"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type independentProxyComputeRuntime interface {
	proxyOnlyCompute(*serveFlags, func(string) string) (bool, error)
}

var independentProxyComputeEnvKeys = []string{"FAK_BACKEND", "FAK_METAL", "FAK_N_CPU_MOE", "FAK_EP_RANK", "FAK_EP_COORD_ADDR", "FAK_CUDA_GRAPH", "FAK_UP_KV_PRECISION"}

func independentProxyComputeFlags(t *testing.T, args ...string) *serveFlags {
	t.Helper()
	for _, key := range independentProxyComputeEnvKeys {
		t.Setenv(key, "")
	}
	fs, sf := newServeFlagSet()
	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	sf.explicit = map[string]bool{}
	fs.Visit(func(f *flag.Flag) { sf.explicit[f.Name] = true })
	return sf
}
func independentProxyComputeMethods(t *testing.T, rt *serveRuntime) independentProxyComputeRuntime {
	t.Helper()
	method, ok := any(rt).(independentProxyComputeRuntime)
	if !ok {
		t.Fatal("serve runtime proxy-only compute classification unavailable")
	}
	return method
}

// fak-test:runtime fast est=50ms lane=default
func TestProxyComputeRemoteOnlyAndActualStage(t *testing.T) {
	for _, args := range [][]string{
		{"--base-url", "http://127.0.0.1:18091/v1"},
		{"--base-url", " http://127.0.0.1:18091/v1 "},
		{"--replica-base-url", "http://127.0.0.1:18091/v1"},
		{"--base-url", "http://127.0.0.1:18091/v1", "--replica-base-url", "http://127.0.0.1:18092/v1"},
		{"--base-url", "http://127.0.0.1:18091/v1", "--engine", "mock"},
	} {
		t.Run(args[0]+args[len(args)-1], func(t *testing.T) {
			sf := independentProxyComputeFlags(t, args...)
			rt := &serveRuntime{}
			remote, err := independentProxyComputeMethods(t, rt).proxyOnlyCompute(sf, func(string) string { return "" })
			if err != nil || !remote {
				t.Fatalf("remote-only proxy needs local compute: %v", err)
			}
			// Call the actual production stage on the SAME runtime after classification.
			// This is a software startup witness, not inference or hardware qualification.
			rt.resolveCompute(sf)
			if rt.chatBackend != nil || rt.useMetal || rt.requireDeviceExecution || rt.inKernelModel != nil {
				t.Fatal("remote-only stage initialized local inference compute")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestProxyComputeMixedAndExplicitLocalStayStrict(t *testing.T) {
	cases := [][]string{
		{}, {"--base-url", "   "},
		{"--base-url", "http://127.0.0.1:18091/v1", "--gguf", "/does-not-exist/local-model.gguf"},
		{"--base-url", "http://127.0.0.1:18091/v1", "--tokenizer", "/does-not-exist/tokenizer.json"},
		{"--base-url", "http://127.0.0.1:18091/v1", "--engine", "inkernel"},
		{"--base-url", "http://127.0.0.1:18091/v1", "--gguf", " "},
		{"--base-url", "http://127.0.0.1:18091/v1", "--tokenizer", "\t"},
	}
	localFlags := map[string]string{
		"backend": "cpu", "metal": "false", "gpu-layers": "0", "native-gpu-layers": "0",
		"cpu-offload-experts": "false", "n-cpu-moe": "0", "expert-parallel": "1", "tensor-parallel": "1",
		"cuda-graph": "false", "native-context-tokens": "0", "native-admission-token-budget": "0",
		"native-compact-history-budget": "0", "native-qwen-q4k-prefill-chunk-tokens": "0",
		"native-qwen35-metal-gdn-sequence": "false", "native-q4k-gateup-slab": "false",
		"kv-precision": "", "native-prefix-profile": "", "vulkan-q4k-profile": "false", "vulkan-stage-q4k": "false",
	}
	for name, value := range localFlags {
		cases = append(cases, []string{"--base-url", "http://127.0.0.1:18091/v1", "--" + name + "=" + value})
	}
	for i, args := range cases {
		t.Run(string(rune('A'+i)), func(t *testing.T) {
			sf := independentProxyComputeFlags(t, args...)
			rt := &serveRuntime{}
			remote, err := independentProxyComputeMethods(t, rt).proxyOnlyCompute(sf, func(string) string { return "" })
			if err != nil || remote {
				t.Fatalf("local/mixed declaration bypassed strict compute: %v", err)
			}
		})
	}
	for _, mode := range []string{"resident-model", "resident-tokenizer", "effective-backend", "effective-metal", "effective-GPU-layers", "effective-CPU-offload", "effective-expert-parallel", "effective-tensor-parallel", "effective-context", "effective-admission", "preexisting-backend", "preexisting-metal", "preexisting-required-device", "preexisting-sharded", "effective-cuda-graph", "effective-n-cpu-moe", "effective-history", "effective-prefill", "effective-GDN", "effective-slab", "effective-KV", "effective-prefix", "effective-vulkan-profile", "effective-vulkan-stage"} {
		t.Run(mode, func(t *testing.T) {
			sf := independentProxyComputeFlags(t, "--base-url", "http://127.0.0.1:18091/v1")
			rt := &serveRuntime{}
			switch mode {
			case "resident-model":
				rt.inKernelModel = &fakmodel.Model{}
			case "resident-tokenizer":
				rt.inKernelTok = &tokenizer.Tokenizer{}
			case "effective-backend":
				*sf.backendName = "cpu"
			case "effective-metal":
				*sf.metal = true
			case "effective-GPU-layers":
				*sf.nativeGPULayers = 1
			case "effective-CPU-offload":
				*sf.cpuOffloadExperts = true
			case "effective-expert-parallel":
				*sf.expertParallel = 2
			case "effective-tensor-parallel":
				*sf.tensorParallel = 2
			case "effective-context":
				*sf.nativeContextTokens = 32768
			case "effective-admission":
				*sf.nativeAdmissionTokenBudget = 65536
			case "preexisting-backend":
				rt.chatBackend = &servePreflightBackend{name: "software-sentinel"}
			case "preexisting-metal":
				rt.useMetal = true
			case "preexisting-required-device":
				rt.requireDeviceExecution = true
			case "preexisting-sharded":
				rt.ep.sharded = true
			case "effective-cuda-graph":
				*sf.cudaGraph = true
			case "effective-n-cpu-moe":
				*sf.nCPUMoE = "1"
			case "effective-history":
				*sf.nativeCompactHistoryBudget = 1024
			case "effective-prefill":
				*sf.nativeQwenQ4KPrefillChunk = 2048
			case "effective-GDN":
				*sf.nativeQwen35MetalGDNSequence = true
			case "effective-slab":
				*sf.nativeQ4KGateUpOutputSlab = true
			case "effective-KV":
				*sf.kvPrecision = "q8_0"
			case "effective-prefix":
				*sf.nativePrefixProfile = "/tmp/software-profile.jsonl"
			case "effective-vulkan-profile":
				*sf.vulkanQ4KProfile = true
			case "effective-vulkan-stage":
				*sf.vulkanStageQ4K = true
			}
			remote, err := independentProxyComputeMethods(t, rt).proxyOnlyCompute(sf, func(string) string { return "" })
			if err != nil || remote {
				t.Fatalf("effective local compute bypassed strict path: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestProxyComputeAmbientAndMalformedRefuse(t *testing.T) {
	for _, key := range independentProxyComputeEnvKeys {
		t.Run(key, func(t *testing.T) {
			sf := independentProxyComputeFlags(t, "--base-url", "http://127.0.0.1:18091/v1")
			rt := &serveRuntime{}
			remote, err := independentProxyComputeMethods(t, rt).proxyOnlyCompute(sf, func(k string) string {
				if k == key {
					return "0"
				}
				return ""
			})
			if err != nil || remote {
				t.Fatalf("ambient compute declaration bypassed strict path: %v", err)
			}
		})
	}
	for _, mode := range []string{"nil-flags", "nil-getenv", "nil-required-pointer", "nil-base-pointer", "nil-engine-pointer", "nil-gguf-pointer", "nil-tokenizer-pointer", "empty-replica"} {
		t.Run(mode, func(t *testing.T) {
			sf := independentProxyComputeFlags(t, "--base-url", "http://127.0.0.1:18091/v1")
			env := func(string) string { return "" }
			switch mode {
			case "nil-flags":
				sf = nil
			case "nil-getenv":
				env = nil
			case "nil-required-pointer":
				sf.backendName = nil
			case "nil-base-pointer":
				sf.baseURL = nil
			case "nil-engine-pointer":
				sf.engineID = nil
			case "nil-gguf-pointer":
				sf.ggufPath = nil
			case "nil-tokenizer-pointer":
				sf.tokPath = nil
			case "empty-replica":
				sf.replicaBaseURLs = append(sf.replicaBaseURLs, " ")
			}
			if remote, err := independentProxyComputeMethods(t, &serveRuntime{}).proxyOnlyCompute(sf, env); err == nil || remote {
				t.Fatal("incomplete remote compute declaration accepted")
			}
		})
	}
}
