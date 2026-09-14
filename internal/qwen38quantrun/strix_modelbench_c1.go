package qwen38quantrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	StrixModelbenchC1AdapterSchema = "fak.qwen38.strix-modelbench-c1-adapter/v1"

	StrixModelbenchC1PerformanceCreditReason = "pair-and-five-measured-trials-required"

	strixModelbenchC1SubprocessTimeout = 30 * time.Minute
)

var expectedObserverEnvKeys = []string{
	"PATH", "HOME", "USERPROFILE", "TMPDIR", "TMP", "TEMP",
	"SystemRoot", "COMSPEC", "PATHEXT", "WINDIR",
}

type StrixModelbenchC1Observation struct {
	observed    bool
	approved    string
	concurrency int
	packet      string
	receipt     compute.Qwen38VulkanDecodeReceipt
}

func (o StrixModelbenchC1Observation) Valid() bool { return o.observed }

func (o StrixModelbenchC1Observation) ChallengeConcurrency() (int, bool) {
	if !o.observed {
		return 0, false
	}
	return o.concurrency, true
}

func (o StrixModelbenchC1Observation) ApprovedCellDigest() (string, bool) {
	if !o.observed {
		return "", false
	}
	return o.approved, true
}

func (o StrixModelbenchC1Observation) PromptPacketSHA256() (string, bool) {
	if !o.observed {
		return "", false
	}
	return o.packet, true
}

func (o StrixModelbenchC1Observation) Receipt() (compute.Qwen38VulkanDecodeReceipt, bool) {
	if !o.observed {
		return compute.Qwen38VulkanDecodeReceipt{}, false
	}
	raw, err := json.Marshal(o.receipt)
	if err != nil {
		return compute.Qwen38VulkanDecodeReceipt{}, false
	}
	var clone compute.Qwen38VulkanDecodeReceipt
	if err := json.Unmarshal(raw, &clone); err != nil {
		return compute.Qwen38VulkanDecodeReceipt{}, false
	}
	return clone, true
}

func (o StrixModelbenchC1Observation) PerformanceCredit() (credit bool, observed bool) {
	return false, o.observed
}

func (o StrixModelbenchC1Observation) PerformanceCreditReason() (string, bool) {
	return StrixModelbenchC1PerformanceCreditReason, o.observed
}

type StrixModelbenchC1Request struct {
	Manifest           StrixComparisonCellManifest
	ManifestDigest     string
	ExecutablePath     string
	ArtifactPath       string
	ExpectedExecutable string
	Environment        []string
}

func CaptureStrixModelbenchC1(ctx context.Context, req StrixModelbenchC1Request) (StrixModelbenchC1Observation, error) {
	return captureStrixModelbenchC1(ctx, req, defaultStrixModelbenchC1Runner)
}

type strixModelbenchC1Runner func(context.Context, string, []string, []string) ([]byte, error)

func captureStrixModelbenchC1(ctx context.Context, req StrixModelbenchC1Request, run strixModelbenchC1Runner) (StrixModelbenchC1Observation, error) {
	fail := func(format string, args ...any) (StrixModelbenchC1Observation, error) {
		return StrixModelbenchC1Observation{}, fmt.Errorf("strix modelbench c1 adapter: "+format, args...)
	}
	if err := VerifyStrixComparisonCellManifest(req.Manifest, req.ManifestDigest); err != nil {
		return fail("approved cell manifest: %w", err)
	}
	if req.Manifest.Challenge.Concurrency != 1 {
		return fail("approved cell concurrency %d is not c=1", req.Manifest.Challenge.Concurrency)
	}
	if req.Manifest.Candidate.ForwardPath != "modelbench/raw-decode/vulkan" {
		return fail("approved cell candidate forward path %q is not the modelbench raw-decode path", req.Manifest.Candidate.ForwardPath)
	}
	packet, err := ImportPromptPacket(req.Manifest.Workload.PromptPacketBytes)
	if err != nil {
		return fail("approved cell prompt packet: %w", err)
	}
	if packet.PacketDigest != req.Manifest.Workload.PromptPacketDigest {
		return fail("approved cell prompt packet digest does not match workload pin")
	}
	binaryPath, err := resolveAdapterPath(req.ExecutablePath)
	if err != nil {
		return fail("executable path: %w", err)
	}
	if !validCanonicalSHA256(req.ExpectedExecutable) {
		return fail("expected executable SHA-256 is required")
	}
	artifactPath, err := resolveAdapterPath(req.ArtifactPath)
	if err != nil {
		return fail("artifact path: %w", err)
	}
	executableDigest, err := hashRegularFile(binaryPath)
	if err != nil {
		return fail("hash executable: %w", err)
	}
	if !strings.EqualFold(executableDigest, req.ExpectedExecutable) {
		return fail("executable SHA-256 mismatch: expected %s, observed %s", strings.ToLower(req.ExpectedExecutable), executableDigest)
	}
	if !strings.EqualFold(req.Manifest.Candidate.ExecutableSHA256, executableDigest) {
		return fail("approved cell executable SHA-256 does not match the observed binary")
	}
	argv := []string{
		binaryPath,
		"-raw-decode",
		"-gguf", artifactPath,
		"-raw-artifact-sha256", strings.ToLower(req.Manifest.Workload.ArtifactSHA256),
		"-raw-prompt-ids", joinPromptTokenIDs(packet.PromptTokenIDs),
		"-raw-context", strconv.Itoa(req.Manifest.Workload.ContextTokens),
		"-raw-ignore-eos",
		"-decode-steps", strconv.Itoa(req.Manifest.Workload.AcceptedOutputTokens),
		"-decode-reps", "1",
		"-backend", req.Manifest.Candidate.Backend,
		"-name", req.Manifest.Workload.Model,
		"-require-non-reference",
	}
	runCtx, cancel := context.WithTimeout(ctx, strixModelbenchC1SubprocessTimeout)
	defer cancel()
	env := scrubAdapterEnvironment(req.Environment)
	stdout, err := run(runCtx, binaryPath, argv, env)
	if err != nil {
		return fail("modelbench subprocess: %w", err)
	}
	receipt, err := parseStrixModelbenchC1Report(stdout, req.Manifest, packet)
	if err != nil {
		return fail("%w", err)
	}
	return StrixModelbenchC1Observation{
		observed:    true,
		approved:    req.ManifestDigest,
		concurrency: req.Manifest.Challenge.Concurrency,
		packet:      packet.PacketDigest,
		receipt:     receipt,
	}, nil
}

func defaultStrixModelbenchC1Runner(ctx context.Context, binaryPath string, argv, env []string) ([]byte, error) {
	if len(argv) == 0 || argv[0] != binaryPath {
		return nil, errors.New("argv does not begin with the pinned executable")
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func scrubAdapterEnvironment(override []string) []string {
	source := override
	if source == nil {
		source = os.Environ()
	}
	allow := make(map[string]struct{}, len(expectedObserverEnvKeys))
	for _, key := range expectedObserverEnvKeys {
		allow[strings.ToUpper(key)] = struct{}{}
	}
	out := make([]string, 0, len(expectedObserverEnvKeys))
	for _, entry := range source {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		if _, keep := allow[strings.ToUpper(key)]; keep {
			out = append(out, entry)
		}
	}
	slices.Sort(out)
	return out
}

func resolveAdapterPath(path string) (string, error) {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		return "", errors.New("path is required")
	}
	if !filepath.IsAbs(trimmed) {
		return "", fmt.Errorf("path %q must be absolute", trimmed)
	}
	cleaned, err := filepath.Abs(trimmed)
	if err != nil {
		return "", err
	}
	return cleaned, nil
}

func hashRegularFile(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("%q is not a regular file", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func joinPromptTokenIDs(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(id)
	}
	return strings.Join(parts, ",")
}

func parseStrixModelbenchC1Report(stdout []byte, manifest StrixComparisonCellManifest, packet PromptTokenPacket) (compute.Qwen38VulkanDecodeReceipt, error) {
	fail := func(format string, args ...any) (compute.Qwen38VulkanDecodeReceipt, error) {
		return compute.Qwen38VulkanDecodeReceipt{}, fmt.Errorf(format, args...)
	}
	if err := rejectDuplicateJSONFields(stdout); err != nil {
		return fail("modelbench report: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(stdout))
	dec.DisallowUnknownFields()
	var report strixModelbenchC1Report
	if err := dec.Decode(&report); err != nil {
		return fail("decode modelbench report: %w", err)
	}
	if err := ensureJSONEOF(dec); err != nil {
		return fail("modelbench report: %w", err)
	}
	if report.CanonicalPhysicalReceipt.Status != "AVAILABLE" {
		return fail("modelbench canonical physical receipt status %q is not AVAILABLE", report.CanonicalPhysicalReceipt.Status)
	}
	if report.CanonicalPhysicalReceipt.Receipt == nil {
		return fail("modelbench canonical physical receipt is absent")
	}
	receipt := *report.CanonicalPhysicalReceipt.Receipt
	if receipt.Schema != compute.Qwen38VulkanDecodeReceiptV3Schema {
		return fail("modelbench receipt schema %q is not the v3 schema", receipt.Schema)
	}
	if err := receipt.Validate(); err != nil {
		return fail("modelbench receipt is not validator-clean: %w", err)
	}
	if receipt.ResourceScope != compute.Qwen38VulkanResourceScopeReportedRuns || receipt.ReportedRuns != 1 || len(receipt.Runs) != 1 {
		return fail("modelbench receipt is not exactly one reported v3 run")
	}
	if receipt.Engine.Name != "fak-native" || receipt.Engine.Backend != compute.Qwen38VulkanDecodeBackend || receipt.Engine.Runtime != compute.Qwen38VulkanDecodeRuntime {
		return fail("modelbench receipt engine tuple is not fak-native/%s/%s", compute.Qwen38VulkanDecodeBackend, compute.Qwen38VulkanDecodeRuntime)
	}
	if receipt.Engine.FallbackCount == nil || *receipt.Engine.FallbackCount != 0 {
		return fail("modelbench receipt did not observe fallback_count=0")
	}
	if receipt.Model.Quantization != manifest.Workload.Quantization {
		return fail("modelbench receipt quantization %q does not match the approved cell %q", receipt.Model.Quantization, manifest.Workload.Quantization)
	}
	if !strings.EqualFold(receipt.Model.ArtifactSHA256, manifest.Workload.ArtifactSHA256) {
		return fail("modelbench receipt artifact SHA-256 does not match the approved cell")
	}
	if !strings.EqualFold(receipt.Model.TokenizerSHA256, manifest.Workload.TokenizerSHA256) ||
		!strings.EqualFold(receipt.Model.TemplateSHA256, manifest.Workload.TemplateSHA256) {
		return fail("modelbench receipt tokenizer/template identity does not match the approved cell")
	}
	if !slices.Equal(int32PromptIDs(receipt.Packet.PromptTokenIDs), packet.PromptTokenIDs) {
		return fail("modelbench receipt prompt token IDs do not match the approved packet")
	}
	if receipt.Packet.GeneratedTokenLimit != manifest.Workload.AcceptedOutputTokens {
		return fail("modelbench receipt generated-token limit %d does not match the approved cell %d", receipt.Packet.GeneratedTokenLimit, manifest.Workload.AcceptedOutputTokens)
	}
	if receipt.Runs[0].ContextLimit != manifest.Workload.ContextTokens {
		return fail("modelbench receipt context limit %d does not match the approved cell %d", receipt.Runs[0].ContextLimit, manifest.Workload.ContextTokens)
	}
	observedIgnoreEOS := receipt.Runs[0].IgnoreEOS
	if observedIgnoreEOS == nil || !*observedIgnoreEOS {
		return fail("modelbench receipt did not observe ignore_eos=true")
	}
	if receipt.Runs[0].EOSStopped == nil || *receipt.Runs[0].EOSStopped {
		return fail("modelbench receipt did not observe eos_stopped=false")
	}
	if receipt.Device.OS == "" || receipt.Device.Arch == "" || receipt.Device.Kernel == "" || receipt.Device.Name == "" {
		return fail("modelbench receipt device identity is incomplete")
	}
	return receipt, nil
}

type strixModelbenchC1Report struct {
	Schema                   string                                  `json:"schema"`
	CanonicalPhysicalReceipt strixModelbenchC1PhysicalReceiptAttempt `json:"canonical_physical_receipt"`
}

type strixModelbenchC1PhysicalReceiptAttempt struct {
	Status         string                             `json:"status"`
	CreditEligible bool                               `json:"credit_eligible"`
	Reason         string                             `json:"reason,omitempty"`
	Receipt        *compute.Qwen38VulkanDecodeReceipt `json:"receipt,omitempty"`
}

func int32PromptIDs(ids []int32) []int {
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}
