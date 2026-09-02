//go:build darwin

package modelperfobs

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strings"
	"time"
)

const AppleMemoryCurrentProviderSchema = "fak-apple-memory-current-provider/1"

const appleMemoryProviderProbeTimeout = 5 * time.Second

var errAppleMemoryProviderProbeTimeout = errors.New("Apple memory provider probe timed out")

const (
	AppleMemoryUnavailableUnsupported      = "unsupported"
	AppleMemoryUnavailablePermissionDenied = "permission-denied"
	AppleMemoryUnavailableCanceled         = "canceled"
	AppleMemoryUnavailableTimeout          = "timeout"
	AppleMemoryUnavailableMalformed        = "malformed"
	AppleMemoryUnavailableAmbiguous        = "ambiguous"
)

// AppleMemoryProviderEvidence records why the current platform provider was
// accepted or remained unavailable. It deliberately does not turn capacity,
// pressure, residency, I/O, power, utilization, occupancy, or a roofline into
// memory traffic.
type AppleMemoryProviderEvidence struct {
	Schema             string           `json:"schema"`
	Provider           string           `json:"provider"`
	ProviderVersion    string           `json:"provider_version"`
	Status             string           `json:"status"`
	UnavailableReason  string           `json:"unavailable_reason,omitempty"`
	Detail             string           `json:"detail,omitempty"`
	Scope              AppleMemoryScope `json:"scope"`
	IntervalNS         int64            `json:"interval_ns"`
	RateOrDelta        string           `json:"rate_or_delta"`
	ResetSemantics     string           `json:"reset_semantics"`
	AdvertisedSamplers []string         `json:"advertised_samplers,omitempty"`
	RawFields          []string         `json:"raw_fields"`
	RawUnits           []string         `json:"raw_units"`
	ObservedAt         string           `json:"observed_at,omitempty"`
}

// AppleMemoryProviderUnavailableError carries a stable unavailable class plus
// the complete provider evidence suitable for a scrubbed witness.
type AppleMemoryProviderUnavailableError struct {
	Evidence AppleMemoryProviderEvidence
	Cause    error
}

func (e *AppleMemoryProviderUnavailableError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("Apple unified-memory bandwidth unavailable (%s): %s: %v", e.Evidence.UnavailableReason, e.Evidence.Detail, e.Cause)
	}
	return fmt.Sprintf("Apple unified-memory bandwidth unavailable (%s): %s", e.Evidence.UnavailableReason, e.Evidence.Detail)
}

func (e *AppleMemoryProviderUnavailableError) Unwrap() error { return e.Cause }

type appleMemoryProviderRunner func(context.Context, string, ...string) ([]byte, error)

// CollectCurrentAppleMemoryBandwidth probes Apple's current powermetrics
// surface. A provider capture is imported only through ImportAppleMemoryCounters;
// raw plist/text is never interpreted as bandwidth. macOS 26.6.2 exposes no
// documented unified-memory byte sampler, so the current truthful result is an
// explicit unsupported error rather than inferred telemetry.
func CollectCurrentAppleMemoryBandwidth(ctx context.Context, o AppleMemoryImportOptions) (BandwidthCollection, error) {
	return collectCurrentAppleMemoryBandwidth(ctx, o, runAppleMemoryProviderCommand, time.Now, appleMemoryProviderProbeTimeout)
}

func collectCurrentAppleMemoryBandwidth(ctx context.Context, o AppleMemoryImportOptions, run appleMemoryProviderRunner, now func() time.Time, probeTimeout time.Duration) (BandwidthCollection, error) {
	evidence := newAppleMemoryProviderEvidence(o, now())
	probeCtx, cancel := context.WithTimeoutCause(ctx, probeTimeout, errAppleMemoryProviderProbeTimeout)
	defer cancel()
	help, err := run(probeCtx, "/usr/bin/powermetrics", "--help")
	if err != nil {
		if errors.Is(context.Cause(probeCtx), errAppleMemoryProviderProbeTimeout) {
			return BandwidthCollection{}, unavailableAppleMemoryProvider(evidence, AppleMemoryUnavailableTimeout,
				"Apple powermetrics sampler-contract probe exceeded its provider-owned timeout", context.DeadlineExceeded)
		}
		if parentErr := ctx.Err(); parentErr != nil {
			return BandwidthCollection{}, unavailableAppleMemoryProvider(evidence, AppleMemoryUnavailableCanceled,
				"Apple powermetrics sampler-contract probe was canceled by its caller", parentErr)
		}
		reason := AppleMemoryUnavailableUnsupported
		if isAppleProviderPermissionError(err, help) {
			reason = AppleMemoryUnavailablePermissionDenied
		}
		return BandwidthCollection{}, unavailableAppleMemoryProvider(evidence, reason, "cannot inspect the Apple powermetrics sampler contract", err)
	}

	samplers, err := parsePowermetricsSamplerNames(help)
	if err != nil {
		return BandwidthCollection{}, unavailableAppleMemoryProvider(evidence, AppleMemoryUnavailableMalformed, "powermetrics help did not contain a parseable sampler list", err)
	}
	if !hasAppleMemoryByteSampler(samplers) {
		evidence.AdvertisedSamplers = samplers
		return BandwidthCollection{}, unavailableAppleMemoryProvider(evidence, AppleMemoryUnavailableUnsupported,
			"powermetrics exposes no documented package/system unified-memory read/write byte-rate or monotonic-byte-counter sampler", nil)
	}

	// A future sampler name alone is not enough evidence: its exact fields,
	// units, interval, direction split, scope, and reset behavior must be bound
	// into the strict generic JSON envelope before fak may import it.
	evidence.AdvertisedSamplers = samplers
	return BandwidthCollection{}, unavailableAppleMemoryProvider(evidence, AppleMemoryUnavailableAmbiguous,
		"a possible memory sampler is present, but its read/write fields, byte units, interval, scope, or reset semantics are not yet proven", nil)
}

func newAppleMemoryProviderEvidence(o AppleMemoryImportOptions, observed time.Time) AppleMemoryProviderEvidence {
	scope := o.Scope
	if scope.Kind == "" {
		scope.Kind = "system"
	}
	version := o.ProviderVersion
	if version == "" {
		version = "macOS-bundled"
	}
	return AppleMemoryProviderEvidence{
		Schema: AppleMemoryCurrentProviderSchema, Provider: "apple-powermetrics", ProviderVersion: version,
		Status: "unavailable", Scope: scope, IntervalNS: int64(o.Interval), RateOrDelta: "unavailable",
		ResetSemantics: "unavailable; no byte counter was accepted", RawFields: []string{}, RawUnits: []string{},
		ObservedAt: observed.UTC().Format(time.RFC3339Nano),
	}
}

func unavailableAppleMemoryProvider(e AppleMemoryProviderEvidence, reason, detail string, cause error) error {
	e.UnavailableReason = reason
	e.Detail = detail
	return &AppleMemoryProviderUnavailableError{Evidence: e, Cause: cause}
}

func parsePowermetricsSamplerNames(help []byte) ([]string, error) {
	const heading = "The following samplers are supported by --samplers:"
	text := string(help)
	start := strings.Index(text, heading)
	if start < 0 {
		return nil, errors.New("sampler heading missing")
	}
	lines := strings.Split(text[start+len(heading):], "\n")
	var names []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if len(names) > 0 {
				break
			}
			continue
		}
		if strings.HasPrefix(trimmed, "and the following") {
			break
		}
		name, _, ok := strings.Cut(trimmed, " ")
		if !ok || name == "" {
			continue
		}
		names = append(names, name)
	}
	if len(names) == 0 {
		return nil, errors.New("sampler list empty")
	}
	sort.Strings(names)
	return names, nil
}

func hasAppleMemoryByteSampler(samplers []string) bool {
	for _, sampler := range samplers {
		s := strings.ToLower(strings.ReplaceAll(sampler, "-", "_"))
		if s == "memory_bandwidth" || s == "dram_bandwidth" || s == "memory_bytes" || s == "dram_bytes" {
			return true
		}
	}
	return false
}

func isAppleProviderPermissionError(err error, output []byte) bool {
	text := strings.ToLower(err.Error() + "\n" + string(output))
	return strings.Contains(text, "permission denied") || strings.Contains(text, "not permitted") ||
		strings.Contains(text, "must be invoked as the superuser") || strings.Contains(text, "operation not permitted")
}

func importCurrentAppleProviderJSON(data []byte, o AppleMemoryImportOptions) (BandwidthCollection, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return BandwidthCollection{}, unavailableAppleMemoryProvider(newAppleMemoryProviderEvidence(o, time.Now()),
			AppleMemoryUnavailableMalformed, "provider output is not the strict Apple generic JSON envelope", nil)
	}
	collection, err := ImportAppleMemoryCounters(io.NopCloser(bytes.NewReader(trimmed)), o)
	if err != nil {
		return BandwidthCollection{}, unavailableAppleMemoryProvider(newAppleMemoryProviderEvidence(o, time.Now()),
			AppleMemoryUnavailableMalformed, "strict Apple importer rejected provider output", err)
	}
	return collection, nil
}

func runAppleMemoryProviderCommand(ctx context.Context, name string, args ...string) ([]byte, error) {
	if name != "/usr/bin/powermetrics" {
		return nil, fmt.Errorf("unsupported Apple memory provider command %q", name)
	}
	// The program is spelled as a literal (not `name`) so the architest
	// interpreter-free gate can prove the exec target is a compiled binary; the
	// guard above has already rejected every other selector value.
	cmd := exec.CommandContext(ctx, "/usr/bin/powermetrics", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	out := stdout.Bytes()
	if len(out) == 0 {
		out = stderr.Bytes()
	}
	return append([]byte(nil), out...), err
}
