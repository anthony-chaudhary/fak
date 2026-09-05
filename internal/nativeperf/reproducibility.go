package nativeperf

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	// ReproducibilityPacketSchema identifies the contract for public reproducible roofline attainment witnesses.
	ReproducibilityPacketSchema = "fak.reproducibility_packet/v1"

	// DefaultWitnessDir is the standard witness repository directory.
	DefaultWitnessDir = "docs/_witnesses"
)

var (
	reWindowsUserPath = regexp.MustCompile(`(?i)[a-z]:\\[uU]sers\\[^\s"',;\\]+`)
	reUnixUserPath    = regexp.MustCompile(`/(?:home|Users)/[^\s"',;/]+`)
	reSecretToken     = regexp.MustCompile(`(?i)(?:sk-[a-zA-Z0-9_-]{10,}|ghp_[a-zA-Z0-9]{10,}|fak-token-[a-zA-Z0-9_-]+|Bearer\s+[a-zA-Z0-9._-]+)`)
)

// RooflineAttainmentInfo captures roofline attainment details in the reproducibility packet.
type RooflineAttainmentInfo struct {
	AchievedBandwidthGBps        float64 `json:"achieved_bandwidth_gbps"`
	TheoreticalPeakBandwidthGBps float64 `json:"theoretical_peak_bandwidth_gbps"`
	AttainmentRatio              float64 `json:"attainment_ratio"`
	AttainmentPercentage         string  `json:"attainment_percentage"`
	FloorThresholdRatio          float64 `json:"floor_threshold_ratio"`
	GatePassed                   bool    `json:"gate_passed"`
}

// NumericalAccuracyResult captures accuracy gate metrics in the packet.
type NumericalAccuracyResult struct {
	Metric                string  `json:"metric"`
	LogitCosineSimilarity float64 `json:"logit_cosine_similarity"`
	MinimumThreshold      float64 `json:"minimum_threshold"`
	GatePassed            bool    `json:"gate_passed"`
}

// TimingInfo captures execution duration and throughput metrics.
type TimingInfo struct {
	ExecutionTimeMs float64 `json:"execution_time_ms"`
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`
	PromptTokens    int     `json:"prompt_tokens,omitempty"`
	DecodeTokens    int     `json:"decode_tokens,omitempty"`
	TotalTokens     int     `json:"total_tokens,omitempty"`
}

// PacketProvenance captures execution provenance, signing attestations, and scrubbing status.
type PacketProvenance struct {
	ExecutionMode string `json:"execution_mode"` // "PHYSICAL_WITNESSED" or "SIMULATED_PROVED"
	CapturedAt    string `json:"captured_at"`
	ProofToken    string `json:"proof_token,omitempty"`
	Signature     string `json:"signature,omitempty"`
	GitRevision   string `json:"git_revision,omitempty"`
	Scrubbed      bool   `json:"scrubbed"`
}

// ScrubbedBenchmarkData contains sanitized, publication-safe benchmark metadata.
type ScrubbedBenchmarkData struct {
	ReceiptID         string            `json:"receipt_id,omitempty"`
	TargetDevice      string            `json:"target_device"`
	ModelArchitecture string            `json:"model_architecture"`
	PeakMemoryBytes   uint64            `json:"peak_memory_bytes,omitempty"`
	Environment       map[string]string `json:"environment,omitempty"`
	Metadata          map[string]string `json:"metadata,omitempty"`
}

// ReproducibilityPacket conforms to `fak.reproducibility_packet/v1` providing a sealed,
// verifiable, scrubbed artifact for public reproducibility audits.
type ReproducibilityPacket struct {
	Schema            string                  `json:"schema"`
	PacketID          string                  `json:"packet_id"`
	TargetDevice      string                  `json:"target_device"`
	ModelArchitecture string                  `json:"model_architecture"`
	AttainmentRatio   float64                 `json:"attainment_ratio"`
	Attainment        RooflineAttainmentInfo  `json:"attainment"`
	NumericalAccuracy NumericalAccuracyResult `json:"numerical_accuracy"`
	Timing            TimingInfo              `json:"timing"`
	Provenance        PacketProvenance        `json:"provenance"`
	ScrubbedReceipt   ScrubbedBenchmarkData   `json:"scrubbed_receipt"`
	Checksum          string                  `json:"checksum"`
}

func scrubText(s string) string {
	if s == "" {
		return s
	}
	s = reWindowsUserPath.ReplaceAllString(s, "[SCRUBBED_PATH]")
	s = reUnixUserPath.ReplaceAllString(s, "[SCRUBBED_PATH]")
	s = reSecretToken.ReplaceAllString(s, "[SCRUBBED_SECRET]")
	return s
}

// ComputeChecksum calculates the deterministic SHA-256 digest of canonical packet JSON.
func (p *ReproducibilityPacket) ComputeChecksum() (string, error) {
	if p == nil {
		return "", errors.New("nativeperf: packet is nil")
	}
	clone := *p
	clone.Checksum = ""
	raw, err := json.Marshal(clone)
	if err != nil {
		return "", fmt.Errorf("nativeperf: marshal reproducibility packet: %w", err)
	}
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("sha256:%s", hex.EncodeToString(sum[:])), nil
}

// VerifyChecksum asserts that the packet's embedded Checksum matches its canonical payload digest.
func (p *ReproducibilityPacket) VerifyChecksum() error {
	if p == nil {
		return errors.New("nativeperf: packet is nil")
	}
	if p.Checksum == "" {
		return errors.New("nativeperf: packet checksum is empty")
	}
	expected, err := p.ComputeChecksum()
	if err != nil {
		return fmt.Errorf("nativeperf: compute checksum failed: %w", err)
	}
	if p.Checksum != expected && p.Checksum != strings.TrimPrefix(expected, "sha256:") {
		return fmt.Errorf("nativeperf: checksum mismatch: got %s, want %s", p.Checksum, expected)
	}
	return nil
}

// JSON encodes the packet into indented, deterministic JSON bytes.
func (p *ReproducibilityPacket) JSON() ([]byte, error) {
	if p == nil {
		return nil, errors.New("nativeperf: packet is nil")
	}
	return json.MarshalIndent(p, "", "  ")
}

// String formats the reproducibility packet as indented JSON.
func (p *ReproducibilityPacket) String() string {
	raw, err := p.JSON()
	if err != nil {
		return fmt.Sprintf("ReproducibilityPacket{PacketID: %s, Checksum: %s}", p.PacketID, p.Checksum)
	}
	return string(raw)
}

// WriteWitness serializes and writes the reproducibility packet to the specified witness directory.
// If witnessDir is empty, it writes to default docs/_witnesses/issue-11623-strix-halo-80pct/.
func (p *ReproducibilityPacket) WriteWitness(witnessDir string) (string, error) {
	if p == nil {
		return "", errors.New("nativeperf: packet is nil")
	}
	if witnessDir == "" {
		witnessDir = filepath.Join("docs", "_witnesses", "issue-11623-strix-halo-80pct")
	}
	if err := os.MkdirAll(witnessDir, 0755); err != nil {
		return "", fmt.Errorf("nativeperf: create witness directory: %w", err)
	}
	filename := fmt.Sprintf("reproducibility-packet-%s.json", p.PacketID)
	outPath := filepath.Join(witnessDir, filename)
	data, err := p.JSON()
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	if err := os.WriteFile(outPath, data, 0644); err != nil {
		return "", fmt.Errorf("nativeperf: write witness packet: %w", err)
	}
	return outPath, nil
}

// GenerateReproducibilityPacket builds a sanitized, verified reproducibility packet conforming to
// schema `fak.reproducibility_packet/v1` from a benchmark receipt with deterministic checksums.
func GenerateReproducibilityPacket(receipt *BenchmarkReceipt) (*ReproducibilityPacket, error) {
	if receipt == nil {
		return nil, ErrNilReceipt
	}

	vResult, err := ValidateRooflineAttainment(receipt)
	if err != nil {
		return nil, fmt.Errorf("nativeperf: cannot generate reproducibility packet: %w", err)
	}

	targetDev := receipt.GetDevice()
	if targetDev == "" {
		targetDev = DefaultAMDStrixHaloDevice
	}

	packetID := receipt.ReceiptID
	if packetID == "" {
		h := sha256.New()
		fmt.Fprintf(h, "%s:%s:%.4f:%.6f:%v",
			targetDev, receipt.ModelArchitecture, receipt.AchievedBandwidthGBps,
			receipt.LogitCosineSimilarity, receipt.IsSim())
		packetID = fmt.Sprintf("packet-strix-halo-80pct-%s", hex.EncodeToString(h.Sum(nil))[:12])
	}

	execMode := "PHYSICAL_WITNESSED"
	if receipt.IsSim() {
		execMode = "SIMULATED_PROVED"
	}

	capturedAt := receipt.Timestamp
	if capturedAt == "" {
		capturedAt = "2026-09-05T08:00:00Z"
	}

	attainmentRatio := vResult.AttainmentRatio
	peakBW := vResult.TheoreticalPeakBandwidthGBps
	achievedBW := vResult.AchievedBandwidthGBps

	// Clean and scrub environment keys and values
	scrubbedEnv := make(map[string]string)
	for k, v := range receipt.Environment {
		upperK := strings.ToUpper(k)
		if strings.Contains(upperK, "KEY") ||
			strings.Contains(upperK, "TOKEN") ||
			strings.Contains(upperK, "SECRET") ||
			strings.Contains(upperK, "PASS") ||
			strings.Contains(upperK, "AUTH") ||
			strings.Contains(upperK, "CREDENTIAL") ||
			strings.Contains(upperK, "COOKIE") ||
			strings.Contains(upperK, "USER") ||
			strings.Contains(upperK, "HOME") ||
			strings.Contains(upperK, "SSH") {
			continue
		}
		scrubbedEnv[k] = scrubText(v)
	}

	// Clean and scrub metadata
	scrubbedMeta := make(map[string]string)
	for k, v := range receipt.Metadata {
		upperK := strings.ToUpper(k)
		if strings.Contains(upperK, "KEY") ||
			strings.Contains(upperK, "SECRET") ||
			strings.Contains(upperK, "TOKEN") ||
			strings.Contains(upperK, "PASS") {
			continue
		}
		scrubbedMeta[k] = scrubText(fmt.Sprintf("%v", v))
	}

	packet := &ReproducibilityPacket{
		Schema:            ReproducibilityPacketSchema,
		PacketID:          packetID,
		TargetDevice:      targetDev,
		ModelArchitecture: receipt.ModelArchitecture,
		AttainmentRatio:   attainmentRatio,
		Attainment: RooflineAttainmentInfo{
			AchievedBandwidthGBps:        achievedBW,
			TheoreticalPeakBandwidthGBps: peakBW,
			AttainmentRatio:              attainmentRatio,
			AttainmentPercentage:         fmt.Sprintf("%.2f%%", attainmentRatio*100.0),
			FloorThresholdRatio:          MinimumRooflineAttainmentRatio,
			GatePassed:                   attainmentRatio >= MinimumRooflineAttainmentRatio,
		},
		NumericalAccuracy: NumericalAccuracyResult{
			Metric:                "logit_cosine_similarity",
			LogitCosineSimilarity: receipt.LogitCosineSimilarity,
			MinimumThreshold:      MinimumLogitCosineSimilarity,
			GatePassed:            receipt.LogitCosineSimilarity >= MinimumLogitCosineSimilarity,
		},
		Timing: TimingInfo{
			ExecutionTimeMs: receipt.ExecutionTimeMs,
			TokensPerSecond: receipt.TokensPerSecond,
			PromptTokens:    receipt.PromptTokens,
			DecodeTokens:    receipt.DecodeTokens,
			TotalTokens:     receipt.TotalTokens,
		},
		Provenance: PacketProvenance{
			ExecutionMode: execMode,
			CapturedAt:    capturedAt,
			ProofToken:    scrubText(receipt.ProofToken),
			Signature:     scrubText(receipt.Signature),
			GitRevision:   scrubText(receipt.GitRevision),
			Scrubbed:      true,
		},
		ScrubbedReceipt: ScrubbedBenchmarkData{
			ReceiptID:         packetID,
			TargetDevice:      targetDev,
			ModelArchitecture: receipt.ModelArchitecture,
			PeakMemoryBytes:   receipt.PeakMemoryBytes,
			Environment:       scrubbedEnv,
			Metadata:          scrubbedMeta,
		},
	}

	checksum, err := packet.ComputeChecksum()
	if err != nil {
		return nil, fmt.Errorf("nativeperf: failed to compute packet checksum: %w", err)
	}
	packet.Checksum = checksum

	return packet, nil
}
