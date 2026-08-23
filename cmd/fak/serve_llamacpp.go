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
	qwen38RuntimeAuto     = "auto"
	qwen38RuntimeNative   = "native"
	qwen38RuntimeLlamaMTP = "llama-mtp"
)

func normalizeQwen38Runtime(v string) (string, error) {
	switch v = strings.ToLower(strings.TrimSpace(v)); v {
	case "", qwen38RuntimeAuto:
		return qwen38RuntimeAuto, nil
	case qwen38RuntimeNative, qwen38RuntimeLlamaMTP:
		return v, nil
	default:
		return "", fmt.Errorf("--qwen38-runtime must be auto, native, or llama-mtp (got %q)", v)
	}
}

func (rt *serveRuntime) maybeStartQwen38Delegation(sf *serveFlags) error {
	mode, err := normalizeQwen38Runtime(*sf.qwen38Runtime)
	if err != nil {
		return err
	}
	if mode == qwen38RuntimeNative || strings.TrimSpace(*sf.ggufPath) == "" || strings.TrimSpace(*sf.baseURL) != "" || len(sf.replicaBaseURLs.Values()) > 0 {
		return nil
	}
	gg, err := ggufload.Open(*sf.ggufPath)
	if err != nil {
		if mode == qwen38RuntimeAuto {
			return nil
		}
		return fmt.Errorf("inspect Qwen3.8 GGUF: %w", err)
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
	if arch != "qwen35" {
		if mode == qwen38RuntimeAuto {
			return nil
		}
		return fmt.Errorf("--qwen38-runtime llama-mtp requires qwen35 GGUF metadata (got %q)", arch)
	}
	binary := strings.TrimSpace(*sf.llamaServer)
	if binary == "" {
		binary = "llama-server"
	}
	probeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	discovered := llamacppinterop.Discover(probeCtx, llamacppinterop.ExecRunner{}, binary)
	if discovered.Outcome != llamacppinterop.OutcomeDelegate || !discovered.Capability.DraftMTP || !discovered.Capability.CUDA {
		if mode == qwen38RuntimeAuto {
			return nil
		}
		return fmt.Errorf("Qwen3.8 llama-mtp CUDA capability unavailable: %s", discovered.Reason)
	}
	if mode == qwen38RuntimeAuto && !llamacppinterop.WitnessedQwen38MTP(discovered.Capability) {
		return nil
	}
	port, err := freeLoopbackPort()
	if err != nil {
		return err
	}
	ctxTokens := *sf.contextBudgetTokens
	if ctxTokens < 4096 {
		ctxTokens = 4096
	}
	plan := llamacppinterop.PlanQwen38MTP(discovered.Capability, *sf.ggufPath, mapped.Descriptor, port, ctxTokens)
	if plan.Outcome != llamacppinterop.OutcomeDelegate {
		return fmt.Errorf("Qwen3.8 llama-mtp plan: %s", plan.Reason)
	}
	proc, err := llamacppinterop.Start(context.Background(), plan.Argv, *sf.llamaStartupTimeout)
	if err != nil {
		return fmt.Errorf("start Qwen3.8 llama-mtp runtime: %w", err)
	}
	rt.llamaProcess = proc
	*sf.baseURL = proc.BaseURL()
	*sf.provider = "openai"
	*sf.backendName = ""
	if strings.TrimSpace(*sf.model) == "" || *sf.model == "mock" {
		*sf.model = *sf.ggufPath
	}
	*sf.ggufPath = ""
	rt.addStartupMessage(newServeStartupMessage("llamacppinterop", "qwen38-runtime", "info", fmt.Sprintf("Qwen3.8 delegated through fak to llama.cpp %s commit %s draft-mtp on loopback pid %d", discovered.Capability.Version, discovered.Capability.Commit, proc.PID())))
	return nil
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
