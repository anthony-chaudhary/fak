package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
)

// remoteKVLogWriter is an optional sink for tests to capture preflight log lines.
var remoteKVLogWriter io.Writer

// resolveRemoteKVConfig resolves the RemoteKVConfig from explicit serveFlags,
// environment variables, and established project defaults (#10277).
func resolveRemoteKVConfig(sf *serveFlags) l3kv.RemoteKVConfig {
	mode := l3kv.RemoteKVModeOptional
	if sf != nil && sf.remoteKVMode != nil && *sf.remoteKVMode != "" {
		mode = l3kv.RemoteKVMode(*sf.remoteKVMode)
	} else if env := strings.TrimSpace(os.Getenv("FAK_REMOTE_KV_MODE")); env != "" {
		mode = l3kv.RemoteKVMode(env)
	}

	backend := "l3kv-blobhttp"
	if sf != nil && sf.remoteKVBackend != nil {
		backend = *sf.remoteKVBackend
	} else if env, ok := os.LookupEnv("FAK_REMOTE_KV_BACKEND"); ok {
		backend = env
	}

	var remoteURL string
	if sf != nil && sf.remoteKVURL != nil {
		remoteURL = *sf.remoteKVURL
	} else if env, ok := os.LookupEnv("FAK_REMOTE_KV_URL"); ok {
		remoteURL = env
	} else if env, ok := os.LookupEnv(l3kv.EnvRemoteURL); ok {
		remoteURL = env
	}

	var token string
	if sf != nil && sf.remoteKVToken != nil {
		token = *sf.remoteKVToken
	} else if env, ok := os.LookupEnv("FAK_REMOTE_KV_TOKEN"); ok {
		token = env
	} else if env, ok := os.LookupEnv(l3kv.EnvRemoteToken); ok {
		token = env
	}

	timeout := l3kv.DefaultRemoteKVTimeout
	if sf != nil && sf.remoteKVTimeout != nil && *sf.remoteKVTimeout > 0 {
		timeout = *sf.remoteKVTimeout
	} else if env := os.Getenv("FAK_REMOTE_KV_TIMEOUT"); env != "" {
		if d, err := time.ParseDuration(env); err == nil {
			timeout = d
		}
	}

	return l3kv.RemoteKVConfig{
		Backend:   backend,
		RemoteURL: remoteURL,
		Token:     token,
		Mode:      mode,
		Timeout:   timeout,
	}
}

// isRemoteKVConfigured reports whether remote KV has been explicitly configured
// or opted into via serveFlags or environment variables.
func isRemoteKVConfigured(sf *serveFlags) bool {
	if sf != nil {
		if sf.remoteKVURL != nil && strings.TrimSpace(*sf.remoteKVURL) != "" {
			return true
		}
		if sf.remoteKVMode != nil && *sf.remoteKVMode != string(l3kv.RemoteKVModeOptional) {
			return true
		}
	}
	if strings.TrimSpace(os.Getenv("FAK_REMOTE_KV_URL")) != "" ||
		strings.TrimSpace(os.Getenv(l3kv.EnvRemoteURL)) != "" ||
		strings.TrimSpace(os.Getenv("FAK_REMOTE_KV_MODE")) != "" {
		return true
	}
	return false
}

// checkServeRemoteKV performs reachability and configuration checks
// for remote KV store before heavyweight model initialization occurs (#10277).
// It reads environment variables and flags (--remote-kv-mode, default optional),
// invokes l3kv.ProbeRemoteKV, and logs the receipt and readiness message.
func checkServeRemoteKV(ctx context.Context, sf *serveFlags, probe l3kv.RemoteKVProbeFunc) (l3kv.RemoteKVProbeReceipt, error) {
	cfg := resolveRemoteKVConfig(sf)
	receipt, err := l3kv.ProbeRemoteKV(ctx, cfg, probe)
	logServeRemoteKVReceipt(receipt)
	return receipt, err
}

// loadServeModelWithRemoteKVCheck coordinates remote KV reachability probing
// before executing the model load callback (#10277).
// In required mode, check failures block model loading and return an error.
// In optional mode, reachability probe timeouts/failures log the fallback reason and execute model loading.
// In disabled mode, model loading proceeds with local residency.
func loadServeModelWithRemoteKVCheck(
	ctx context.Context,
	sf *serveFlags,
	probe l3kv.RemoteKVProbeFunc,
	loadModel func(),
) (l3kv.RemoteKVProbeReceipt, error) {
	receipt, err := checkServeRemoteKV(ctx, sf, probe)
	if err != nil {
		return receipt, err
	}
	if loadModel != nil {
		loadModel()
	}
	return receipt, nil
}

// logServeRemoteKVReceipt logs the scrubbed preflight receipt and readiness message to stderr/log.
func logServeRemoteKVReceipt(receipt l3kv.RemoteKVProbeReceipt) {
	raw, err := json.Marshal(receipt)
	var receiptText string
	if err == nil {
		receiptText = fmt.Sprintf("fak serve: remote kv preflight receipt: %s", string(raw))
	} else {
		receiptText = fmt.Sprintf("fak serve: remote kv preflight receipt: schema=%s backend=%s mode=%s outcome=%s endpoint=%s duration_ms=%.2f err=%s",
			receipt.Schema, receipt.Backend, receipt.Mode, receipt.Outcome, receipt.SanitizedEndpoint, receipt.ProbeDurationMS, receipt.Error)
	}

	readinessMsg := fmt.Sprintf("fak serve: %s", serveRemoteKVReadinessMessage(receipt))

	if remoteKVLogWriter != nil {
		fmt.Fprintln(remoteKVLogWriter, receiptText)
		fmt.Fprintln(remoteKVLogWriter, readinessMsg)
	} else {
		log.Println(receiptText)
		log.Println(readinessMsg)
	}
}

// serveRemoteKVReadinessMessage produces a concise, human-readable status for the remote KV outcome.
func serveRemoteKVReadinessMessage(receipt l3kv.RemoteKVProbeReceipt) string {
	switch receipt.Outcome {
	case l3kv.RemoteKVOutcomeReady:
		return fmt.Sprintf("remote KV store ready: backend=%s endpoint=%s duration=%.2fms",
			receipt.Backend, receipt.SanitizedEndpoint, receipt.ProbeDurationMS)
	case l3kv.RemoteKVOutcomeDisabled:
		return "remote KV store disabled: operating with local cache residency"
	case l3kv.RemoteKVOutcomeTimeout:
		if receipt.FallbackReason != "" {
			return fmt.Sprintf("remote KV store timed out: fallback active (%s)", receipt.FallbackReason)
		}
		return fmt.Sprintf("remote KV store timed out: %s", receipt.Error)
	case l3kv.RemoteKVOutcomeUnavailable:
		if receipt.FallbackReason != "" {
			return fmt.Sprintf("remote KV store unavailable: fallback active (%s)", receipt.FallbackReason)
		}
		return fmt.Sprintf("remote KV store unavailable: %s", receipt.Error)
	case l3kv.RemoteKVOutcomeIncompleteConfig:
		return fmt.Sprintf("remote KV store incomplete configuration: %s", receipt.Error)
	case l3kv.RemoteKVOutcomeMalformedConfig:
		return fmt.Sprintf("remote KV store malformed configuration: %s", receipt.Error)
	default:
		return fmt.Sprintf("remote KV store preflight %s: %s", receipt.Outcome, receipt.Error)
	}
}

// serveRemoteKVStartupMessage converts a RemoteKVProbeReceipt into a gateway.StartupMessage.
func serveRemoteKVStartupMessage(receipt l3kv.RemoteKVProbeReceipt) gateway.StartupMessage {
	level := "info"
	if receipt.Outcome == l3kv.RemoteKVOutcomeTimeout || receipt.Outcome == l3kv.RemoteKVOutcomeUnavailable {
		level = "warning"
	} else if receipt.Outcome == l3kv.RemoteKVOutcomeMalformedConfig || receipt.Outcome == l3kv.RemoteKVOutcomeIncompleteConfig {
		level = "error"
	}
	return newServeStartupMessage("serve", "remote-kv", level, serveRemoteKVReadinessMessage(receipt))
}
