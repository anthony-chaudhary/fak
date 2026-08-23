package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufinterop"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/llamacppinterop"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

const (
	qwen38RuntimeNative   = "native"
	qwen38RuntimeLlamaMTP = "llama-mtp"
)

type qwen38ChildProcess interface {
	BaseURL() string
	PID() int
	Stop() error
}

type qwen38RuntimeDependencies struct {
	inspect  func(string) (quantmeta.Descriptor, string, error)
	discover func(context.Context, string) llamacppinterop.Result
	freePort func() (int, error)
	start    func(context.Context, []string, time.Duration) (qwen38ChildProcess, error)
}

func normalizeQwen38Runtime(v string) (string, error) {
	switch v = strings.ToLower(strings.TrimSpace(v)); v {
	case "", qwen38RuntimeNative:
		return qwen38RuntimeNative, nil
	case qwen38RuntimeLlamaMTP:
		return v, nil
	case "auto":
		return "", fmt.Errorf("--qwen38-runtime auto was removed because it could silently delegate to llama.cpp; omit the flag for fak-native execution or pass llama-mtp explicitly for benchmark/reference interoperability")
	default:
		return "", fmt.Errorf("--qwen38-runtime must be native or llama-mtp (got %q); external llama.cpp execution requires explicit llama-mtp selection", v)
	}
}

func qwen38RuntimeIdentity(mode string) string {
	if strings.EqualFold(strings.TrimSpace(mode), qwen38RuntimeLlamaMTP) {
		return "llama.cpp"
	}
	return "fak-native"
}

func (rt *serveRuntime) maybeStartQwen38Delegation(sf *serveFlags) error {
	mode, err := normalizeQwen38Runtime(*sf.qwen38Runtime)
	if err != nil {
		return err
	}
	if mode == qwen38RuntimeNative {
		if strings.TrimSpace(*sf.ggufPath) != "" && strings.TrimSpace(*sf.baseURL) == "" && len(sf.replicaBaseURLs.Values()) == 0 {
			rt.addStartupMessage(newServeStartupMessage("serve", "qwen38-runtime", "info",
				"execution_runtime=fak-native selection=native delegation=disabled; local GGUF execution stays inside fak"))
		}
		return nil
	}
	if err := validateQwen38LlamaMTPSelection(sf); err != nil {
		return err
	}

	deps := rt.qwen38Dependencies()
	descriptor, arch, err := deps.inspect(*sf.ggufPath)
	if err != nil {
		return fmt.Errorf("inspect Qwen3.8 GGUF: %w", err)
	}
	if arch != "qwen35" {
		return fmt.Errorf("--qwen38-runtime llama-mtp requires qwen35 GGUF metadata (got %q)", arch)
	}
	binary := strings.TrimSpace(*sf.llamaServer)
	if binary == "" {
		binary = "llama-server"
	}
	probeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	discovered := deps.discover(probeCtx, binary)
	if discovered.Outcome != llamacppinterop.OutcomeDelegate || !discovered.Capability.DraftMTP || !discovered.Capability.CUDA {
		return fmt.Errorf("Qwen3.8 llama-mtp CUDA capability unavailable: %s", discovered.Reason)
	}
	port, err := deps.freePort()
	if err != nil {
		return err
	}
	ctxTokens := *sf.contextBudgetTokens
	if ctxTokens < 4096 {
		ctxTokens = 4096
	}
	plan := llamacppinterop.PlanQwen38MTP(discovered.Capability, *sf.ggufPath, descriptor, port, ctxTokens)
	if plan.Outcome != llamacppinterop.OutcomeDelegate {
		return fmt.Errorf("Qwen3.8 llama-mtp plan: %s", plan.Reason)
	}
	proc, err := deps.start(context.Background(), plan.Argv, *sf.llamaStartupTimeout)
	if err != nil {
		return fmt.Errorf("start Qwen3.8 llama-mtp runtime: %w", err)
	}
	if proc == nil || strings.TrimSpace(proc.BaseURL()) == "" {
		if proc != nil {
			_ = proc.Stop()
		}
		return fmt.Errorf("start Qwen3.8 llama-mtp runtime: delegated process returned no loopback base URL")
	}
	rt.llamaProcess = proc
	*sf.baseURL = proc.BaseURL()
	*sf.provider = "openai"
	if strings.TrimSpace(*sf.model) == "" || *sf.model == "mock" {
		*sf.model = *sf.ggufPath
	}
	*sf.ggufPath = ""
	rt.addStartupMessage(newServeStartupMessage("llamacppinterop", "qwen38-runtime", "info",
		fmt.Sprintf("execution_runtime=llama.cpp selection=llama-mtp delegation=explicit version=%s commit=%s pid=%d; fak remains the public control plane", discovered.Capability.Version, discovered.Capability.Commit, proc.PID())))
	return nil
}

func validateQwen38LlamaMTPSelection(sf *serveFlags) error {
	if strings.TrimSpace(*sf.ggufPath) == "" {
		return fmt.Errorf("--qwen38-runtime llama-mtp requires --gguf; external execution is never inferred from an upstream or an empty model source")
	}
	if strings.TrimSpace(*sf.baseURL) != "" || len(sf.replicaBaseURLs.Values()) > 0 {
		return fmt.Errorf("--qwen38-runtime llama-mtp cannot be combined with --base-url or --replica-base-url; fak must own the loopback llama-server child")
	}
	var conflicts []string
	if strings.TrimSpace(*sf.backendName) != "" {
		conflicts = append(conflicts, "--backend")
	}
	if *sf.cudaGraph {
		conflicts = append(conflicts, "--cuda-graph")
	}
	if *sf.metal {
		conflicts = append(conflicts, "--metal")
	}
	if *sf.cpuOffloadExperts {
		conflicts = append(conflicts, "--cpu-offload-experts")
	}
	if strings.TrimSpace(*sf.nCPUMoE) != "" {
		conflicts = append(conflicts, "--n-cpu-moe")
	}
	if *sf.expertParallel != 1 {
		conflicts = append(conflicts, "--expert-parallel")
	}
	if *sf.tensorParallel != 1 {
		conflicts = append(conflicts, "--tensor-parallel")
	}
	if strings.TrimSpace(*sf.tokPath) != "" {
		conflicts = append(conflicts, "--tokenizer")
	}
	if len(conflicts) > 0 {
		return fmt.Errorf("--qwen38-runtime llama-mtp cannot be combined with %s: these configure fak-native execution and would otherwise be silently ignored; remove them or select native", strings.Join(conflicts, ", "))
	}
	return nil
}

func (rt *serveRuntime) qwen38Dependencies() qwen38RuntimeDependencies {
	deps := qwen38RuntimeDependencies{
		inspect: inspectQwen38GGUF,
		discover: func(ctx context.Context, binary string) llamacppinterop.Result {
			return llamacppinterop.Discover(ctx, llamacppinterop.ExecRunner{}, binary)
		},
		freePort: freeLoopbackPort,
		start: func(ctx context.Context, argv []string, timeout time.Duration) (qwen38ChildProcess, error) {
			return llamacppinterop.Start(ctx, argv, timeout)
		},
	}
	if rt.qwen38Deps == nil {
		return deps
	}
	if rt.qwen38Deps.inspect != nil {
		deps.inspect = rt.qwen38Deps.inspect
	}
	if rt.qwen38Deps.discover != nil {
		deps.discover = rt.qwen38Deps.discover
	}
	if rt.qwen38Deps.freePort != nil {
		deps.freePort = rt.qwen38Deps.freePort
	}
	if rt.qwen38Deps.start != nil {
		deps.start = rt.qwen38Deps.start
	}
	return deps
}

func inspectQwen38GGUF(path string) (quantmeta.Descriptor, string, error) {
	gg, err := ggufload.Open(path)
	if err != nil {
		return quantmeta.Descriptor{}, "", err
	}
	mapped := ggufinterop.Map(gg)
	arch := ""
	if raw := mapped.Descriptor.Extra["gguf_architecture"]; len(raw) > 0 {
		_ = json.Unmarshal(raw, &arch)
	}
	if arch == "" {
		if entry, ok := gg.Metadata["general.architecture"]; ok {
			arch, _ = entry.Value.(string)
		}
	}
	if arch == "qwen35" && mapped.Descriptor.Artifact == nil {
		mapped.Descriptor.Artifact = &quantmeta.ArtifactSpec{ContainerID: "gguf"}
	}
	if mapped.Descriptor.Extra == nil {
		mapped.Descriptor.Extra = map[string]json.RawMessage{}
	}
	if arch == "qwen35" {
		mapped.Descriptor.Extra["gguf_architecture"] = json.RawMessage("\"qwen35\"")
	}
	return mapped.Descriptor, arch, nil
}

func freeLoopbackPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return strconv.Atoi(strings.TrimPrefix(l.Addr().String(), "127.0.0.1:"))
}

func (rt *serveRuntime) stopQwen38Delegation() {
	if rt.llamaProcess != nil {
		_ = rt.llamaProcess.Stop()
		rt.llamaProcess = nil
	}
}
