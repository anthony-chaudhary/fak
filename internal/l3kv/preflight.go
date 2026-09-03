package l3kv

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

// RemoteKVProbeSchema identifies the schema version of the probe receipt.
const RemoteKVProbeSchema = "fak.remote-kv-preflight/v1"

// DefaultRemoteKVTimeout is the default duration allotted for remote KV connectivity probe.
const DefaultRemoteKVTimeout = 5 * time.Second

// RemoteKVMode defines the operational policy for remote KV store availability.
type RemoteKVMode string

const (
	RemoteKVModeOptional RemoteKVMode = "optional"
	RemoteKVModeRequired RemoteKVMode = "required"
	RemoteKVModeDisabled RemoteKVMode = "disabled"
)

// RemoteKVOutcome defines the typed preflight classification result.
type RemoteKVOutcome string

const (
	RemoteKVOutcomeReady            RemoteKVOutcome = "ready"
	RemoteKVOutcomeDisabled         RemoteKVOutcome = "disabled"
	RemoteKVOutcomeUnavailable      RemoteKVOutcome = "unavailable"
	RemoteKVOutcomeTimeout          RemoteKVOutcome = "timeout"
	RemoteKVOutcomeMalformedConfig  RemoteKVOutcome = "malformed_config"
	RemoteKVOutcomeIncompleteConfig RemoteKVOutcome = "incomplete_config"
)

// RemoteKVConfig holds the connection and policy parameters for the remote KV backend.
type RemoteKVConfig struct {
	Backend   string        `json:"backend"`
	RemoteURL string        `json:"remote_url"`
	Token     string        `json:"token,omitempty"`
	Mode      RemoteKVMode  `json:"mode"`
	Timeout   time.Duration `json:"timeout"`
}

// RemoteKVProbeReceipt is the audit-clean, credential-scrubbed record
// produced by ProbeRemoteKV before any model initialization occurs.
type RemoteKVProbeReceipt struct {
	Schema            string          `json:"schema"`
	Backend           string          `json:"backend"`
	Mode              RemoteKVMode    `json:"mode"`
	Outcome           RemoteKVOutcome `json:"outcome"`
	SanitizedEndpoint string          `json:"sanitized_endpoint"`
	ConfigDigest      string          `json:"config_digest"`
	ProbeDurationMS   float64         `json:"probe_duration_ms"`
	Timestamp         time.Time       `json:"timestamp"`
	FallbackReason    string          `json:"fallback_reason,omitempty"`
	Error             string          `json:"error,omitempty"`
}

// RemoteKVProbeFunc evaluates the reachability of the remote KV backend.
type RemoteKVProbeFunc func(ctx context.Context, cfg RemoteKVConfig) error

// ProbeRemoteKV performs preflight validation and reachability probing for
// a remote KV store before heavyweight model initialization begins (#10277).
//
// In disabled mode, it returns RemoteKVOutcomeDisabled without touching the network.
// In required mode, any probe failure or timeout returns an error.
// In optional mode, probe timeouts or errors return a nil error with FallbackReason set.
// Malformed or incomplete configurations fail closed immediately in all active modes.
func ProbeRemoteKV(ctx context.Context, cfg RemoteKVConfig, probe RemoteKVProbeFunc) (RemoteKVProbeReceipt, error) {
	receipt := RemoteKVProbeReceipt{
		Schema:    RemoteKVProbeSchema,
		Timestamp: time.Now().UTC(),
		Mode:      cfg.Mode,
	}

	// Validate mode presence and validity.
	if cfg.Mode == "" {
		receipt.Backend = "none"
		receipt.Outcome = RemoteKVOutcomeIncompleteConfig
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, "", cfg.Mode, cfg.Timeout)
		err := fmt.Errorf("remote kv: incomplete configuration: mode is required")
		receipt.Error = err.Error()
		return receipt, err
	}
	if cfg.Mode != RemoteKVModeOptional && cfg.Mode != RemoteKVModeRequired && cfg.Mode != RemoteKVModeDisabled {
		receipt.Backend = "none"
		receipt.Outcome = RemoteKVOutcomeMalformedConfig
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, "", cfg.Mode, cfg.Timeout)
		err := fmt.Errorf("remote kv: malformed configuration: invalid mode %q", cfg.Mode)
		receipt.Error = err.Error()
		return receipt, err
	}

	// When explicitly disabled, skip network probing and return clean receipt.
	if cfg.Mode == RemoteKVModeDisabled {
		receipt.Backend = "none"
		receipt.Outcome = RemoteKVOutcomeDisabled
		sanitized, _ := sanitizeURL(cfg.RemoteURL, cfg.Token)
		receipt.SanitizedEndpoint = sanitized
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, sanitized, cfg.Mode, cfg.Timeout)
		return receipt, nil
	}

	// Validate backend presence and supported value.
	if strings.TrimSpace(cfg.Backend) == "" {
		receipt.Backend = "none"
		receipt.Outcome = RemoteKVOutcomeIncompleteConfig
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, "", cfg.Mode, cfg.Timeout)
		err := fmt.Errorf("remote kv: incomplete configuration: backend is required")
		receipt.Error = err.Error()
		return receipt, err
	}
	if cfg.Backend != "l3kv-blobhttp" && cfg.Backend != "none" {
		receipt.Backend = cfg.Backend
		receipt.Outcome = RemoteKVOutcomeMalformedConfig
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, "", cfg.Mode, cfg.Timeout)
		err := fmt.Errorf("remote kv: malformed configuration: unsupported backend %q", cfg.Backend)
		receipt.Error = err.Error()
		return receipt, err
	}
	receipt.Backend = cfg.Backend

	// Validate remote URL presence.
	if strings.TrimSpace(cfg.RemoteURL) == "" {
		receipt.Outcome = RemoteKVOutcomeIncompleteConfig
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, "", cfg.Mode, cfg.Timeout)
		err := fmt.Errorf("remote kv: incomplete configuration: remote URL is required")
		receipt.Error = err.Error()
		return receipt, err
	}

	// Sanitize endpoint and validate URL structure.
	sanitizedEndpoint, err := sanitizeURL(cfg.RemoteURL, cfg.Token)
	if err != nil {
		receipt.Outcome = RemoteKVOutcomeMalformedConfig
		receipt.SanitizedEndpoint = ""
		receipt.ConfigDigest = computeConfigDigest(receipt.Backend, "", cfg.Mode, cfg.Timeout)
		formatErr := fmt.Errorf("remote kv: malformed configuration: %w", err)
		receipt.Error = formatErr.Error()
		return receipt, formatErr
	}
	receipt.SanitizedEndpoint = sanitizedEndpoint
	receipt.ConfigDigest = computeConfigDigest(receipt.Backend, sanitizedEndpoint, cfg.Mode, cfg.Timeout)

	// Bound probe execution by configured or default timeout.
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultRemoteKVTimeout
	}
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if probe == nil {
		probe = DefaultRemoteKVProbe
	}

	start := time.Now()
	probeErr := probe(probeCtx, cfg)
	receipt.ProbeDurationMS = float64(time.Since(start).Nanoseconds()) / 1e6

	if probeErr == nil {
		receipt.Outcome = RemoteKVOutcomeReady
		return receipt, nil
	}

	receipt.Error = probeErr.Error()
	if isTimeoutError(probeErr, probeCtx) {
		receipt.Outcome = RemoteKVOutcomeTimeout
		if cfg.Mode == RemoteKVModeRequired {
			return receipt, fmt.Errorf("remote kv required: probe timed out: %w", probeErr)
		}
		receipt.FallbackReason = fmt.Sprintf("remote kv probe timed out: %v; falling back to local residency", probeErr)
		return receipt, nil
	}

	receipt.Outcome = RemoteKVOutcomeUnavailable
	if cfg.Mode == RemoteKVModeRequired {
		return receipt, fmt.Errorf("remote kv required: probe unavailable: %w", probeErr)
	}
	receipt.FallbackReason = fmt.Sprintf("remote kv probe unavailable: %v; falling back to local residency", probeErr)
	return receipt, nil
}

// DefaultRemoteKVProbe probes a remote HTTP endpoint via HEAD request.
func DefaultRemoteKVProbe(ctx context.Context, cfg RemoteKVConfig) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, cfg.RemoteURL, nil)
	if err != nil {
		return fmt.Errorf("create probe request: %w", err)
	}
	if cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.Token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("probe request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("probe authentication failed: HTTP %d", resp.StatusCode)
	}
	if resp.StatusCode >= http.StatusInternalServerError {
		return fmt.Errorf("probe server error: HTTP %d", resp.StatusCode)
	}
	return nil
}

func sanitizeURL(rawURL, token string) (string, error) {
	trimmed := strings.TrimSpace(rawURL)
	if trimmed == "" {
		return "", nil
	}
	u, err := url.Parse(trimmed)
	if err != nil {
		return "", fmt.Errorf("parse url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("unsupported scheme %q (must be http or https)", u.Scheme)
	}
	if u.Host == "" {
		return "", fmt.Errorf("missing host in URL")
	}
	if port := u.Port(); port != "" {
		p, err := strconv.Atoi(port)
		if err != nil || p < 1 || p > 65535 {
			return "", fmt.Errorf("invalid port in URL: %s", port)
		}
	}

	// Scrub userinfo credentials.
	u.User = nil

	// Scrub sensitive query parameters.
	if u.RawQuery != "" {
		q := u.Query()
		for key := range q {
			lower := strings.ToLower(key)
			if strings.Contains(lower, "token") ||
				strings.Contains(lower, "secret") ||
				strings.Contains(lower, "key") ||
				strings.Contains(lower, "pass") ||
				strings.Contains(lower, "auth") {
				q.Del(key)
			}
		}
		u.RawQuery = q.Encode()
	}

	sanitized := u.String()
	if token != "" && strings.Contains(sanitized, token) {
		sanitized = strings.ReplaceAll(sanitized, token, "[REDACTED]")
	}
	return sanitized, nil
}

func computeConfigDigest(backend, sanitizedEndpoint string, mode RemoteKVMode, timeout time.Duration) string {
	h := sha256.New()
	fmt.Fprintf(h, "backend=%s\nendpoint=%s\nmode=%s\ntimeout=%s\n", backend, sanitizedEndpoint, mode, timeout)
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}

func isTimeoutError(err error, ctx context.Context) bool {
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, os.ErrDeadlineExceeded) {
		return true
	}
	if ctx != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	return false
}
