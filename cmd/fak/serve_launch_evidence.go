package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"time"
)

type serveLaunchEvidence struct {
	Schema              string    `json:"schema"`
	CreatedAt           time.Time `json:"created_at"`
	PID                 int       `json:"pid"`
	InvocationID        string    `json:"invocation_id"`
	ExecutableSHA256    string    `json:"executable_sha256"`
	SourceRevision      string    `json:"source_revision"`
	SourceModified      *bool     `json:"source_modified"`
	ConfigSHA256        string    `json:"startup_config_sha256,omitempty"`
	Engine              string    `json:"engine"`
	Provider            string    `json:"provider"`
	Model               string    `json:"model"`
	ModelPath           string    `json:"model_path,omitempty"`
	Backend             string    `json:"backend"`
	PrefillChunkCeiling int       `json:"native_qwen_q4k_prefill_chunk_tokens"`
	ContextTokens       int       `json:"resolved_native_context_tokens"`
}

func serveLaunchDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func collectServeLaunchEvidence(sf *serveFlags, rt *serveRuntime) (serveLaunchEvidence, error) {
	receipt := serveLaunchEvidence{Schema: "fak.serve_launch_evidence.v1", CreatedAt: time.Now().UTC(), PID: os.Getpid(), InvocationID: os.Getenv("INVOCATION_ID"), ConfigSHA256: sf.startupConfigSHA256, Engine: *sf.engineID, Provider: *sf.provider, Model: *sf.model, ModelPath: *sf.ggufPath, PrefillChunkCeiling: *sf.nativeQwenQ4KPrefillChunk, ContextTokens: rt.nativeContext.ResolvedTokens, Backend: "cpu"}
	if rt.chatBackend != nil {
		receipt.Backend = rt.chatBackend.Name()
	}
	if rt.useMetal {
		receipt.Backend = "metal"
	}
	if rt.inKernelModel == nil {
		receipt.Backend = "unresolved"
	}
	executable, err := os.Executable()
	if err != nil {
		return receipt, err
	}
	// The Linux process handle binds bytes even when the executable pathname is replaced.
	if _, err := os.Stat("/proc/self/exe"); err == nil {
		executable = "/proc/self/exe"
	}
	f, err := os.Open(executable)
	if err != nil {
		return receipt, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return receipt, err
	}
	receipt.ExecutableSHA256 = hex.EncodeToString(h.Sum(nil))
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				receipt.SourceRevision = setting.Value
			case "vcs.modified":
				if setting.Value == "true" || setting.Value == "false" {
					modified := setting.Value == "true"
					receipt.SourceModified = &modified
				}
			}
		}
	}
	return receipt, nil
}

func writeServeLaunchEvidence(sf *serveFlags, rt *serveRuntime) error {
	if sf.launchEvidencePath == nil || *sf.launchEvidencePath == "" {
		return nil
	}
	if !filepath.IsAbs(*sf.launchEvidencePath) {
		return errors.New("launch evidence requires an absolute path")
	}
	receipt, err := collectServeLaunchEvidence(sf, rt)
	if err != nil {
		return err
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	if len(data) > 16383 {
		return errors.New("launch evidence exceeds 16384 bytes")
	}
	f, err := os.OpenFile(*sf.launchEvidencePath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	if _, err = f.Write(append(data, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return nil
}
