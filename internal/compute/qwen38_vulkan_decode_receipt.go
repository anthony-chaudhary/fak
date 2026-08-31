package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

const (
	Qwen38VulkanDecodePacketSchema  = "fak/qwen38-vulkan-decode-packet/v1"
	Qwen38VulkanDecodeReceiptSchema = "fak/qwen38-vulkan-decode-receipt/v1"

	Qwen38VulkanDecodeGGUFSHA256 = "7E78DA5D7E3AE28D178121F58646953305F3E5BD3CB46F4A75584E8B6C6FE169"
	Qwen38VulkanDecodeBackend    = "vulkan"
	Qwen38VulkanDecodeRuntime    = "native"
	Qwen38VulkanDecodeFallback   = "none"
)

// Qwen38VulkanDecodePacket freezes the identity and token boundary of one
// deterministic native decode request. PromptTokenIDs are the tokenizer output,
// so replay does not depend on prompt rendering or tokenizer revisions.
type Qwen38VulkanDecodePacket struct {
	Schema              string  `json:"schema"`
	ModelGGUFSHA256     string  `json:"model_gguf_sha256"`
	PromptTokenIDs      []int32 `json:"prompt_token_ids"`
	GeneratedTokenLimit int     `json:"generated_token_limit"`
	Backend             string  `json:"backend"`
	Runtime             string  `json:"runtime"`
	Fallback            string  `json:"fallback"`
}

// NewQwen38VulkanDecodePacket returns the canonical engine and model identity.
// The caller supplies only the exact prompt tokens and positive decode boundary.
func NewQwen38VulkanDecodePacket(promptTokenIDs []int32, generatedTokenLimit int) Qwen38VulkanDecodePacket {
	return Qwen38VulkanDecodePacket{
		Schema:              Qwen38VulkanDecodePacketSchema,
		ModelGGUFSHA256:     Qwen38VulkanDecodeGGUFSHA256,
		PromptTokenIDs:      slices.Clone(promptTokenIDs),
		GeneratedTokenLimit: generatedTokenLimit,
		Backend:             Qwen38VulkanDecodeBackend,
		Runtime:             Qwen38VulkanDecodeRuntime,
		Fallback:            Qwen38VulkanDecodeFallback,
	}
}

func (p Qwen38VulkanDecodePacket) Validate() error {
	if p.Schema != Qwen38VulkanDecodePacketSchema {
		return fmt.Errorf("qwen3.8 Vulkan packet schema %q, want %q", p.Schema, Qwen38VulkanDecodePacketSchema)
	}
	if p.ModelGGUFSHA256 != Qwen38VulkanDecodeGGUFSHA256 {
		return fmt.Errorf("qwen3.8 Vulkan packet model digest %q, want %q", p.ModelGGUFSHA256, Qwen38VulkanDecodeGGUFSHA256)
	}
	if len(p.PromptTokenIDs) == 0 {
		return errors.New("qwen3.8 Vulkan packet requires prompt token IDs")
	}
	if p.GeneratedTokenLimit <= 0 {
		return fmt.Errorf("qwen3.8 Vulkan generated-token limit must be positive, got %d", p.GeneratedTokenLimit)
	}
	if p.Backend != Qwen38VulkanDecodeBackend || p.Runtime != Qwen38VulkanDecodeRuntime || p.Fallback != Qwen38VulkanDecodeFallback {
		return fmt.Errorf("qwen3.8 Vulkan packet requires backend=%s runtime=%s fallback=%s, got backend=%q runtime=%q fallback=%q", Qwen38VulkanDecodeBackend, Qwen38VulkanDecodeRuntime, Qwen38VulkanDecodeFallback, p.Backend, p.Runtime, p.Fallback)
	}
	return nil
}

func (p Qwen38VulkanDecodePacket) Digest() (string, error) {
	if err := p.Validate(); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(p)
	if err != nil {
		return "", fmt.Errorf("marshal qwen3.8 Vulkan packet: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return strings.ToUpper(hex.EncodeToString(sum[:])), nil
}

type Qwen38VulkanTransferCounters struct {
	Count uint64 `json:"count"`
	Bytes uint64 `json:"bytes"`
}

type Qwen38VulkanTensorHomeCounters struct {
	Hits          uint64 `json:"hits"`
	Admissions    uint64 `json:"admissions"`
	Bypasses      uint64 `json:"bypasses"`
	ResidentBytes uint64 `json:"resident_bytes"`
	CopiedBytes   uint64 `json:"copied_bytes"`
}

type Qwen38VulkanDecodeCounters struct {
	ComputeDispatches      uint64                         `json:"compute_dispatches"`
	Q4KMatmulDispatches    uint64                         `json:"q4_k_matmul_dispatches"`
	OtherComputeDispatches uint64                         `json:"other_compute_dispatches"`
	DispatchSubmits        uint64                         `json:"dispatch_submits"`
	H2D                    Qwen38VulkanTransferCounters   `json:"h2d"`
	D2H                    Qwen38VulkanTransferCounters   `json:"d2h"`
	D2D                    Qwen38VulkanTransferCounters   `json:"d2d"`
	Q4KStageCalls          uint64                         `json:"q4_k_stage_calls"`
	Q4KStageBytes          uint64                         `json:"q4_k_stage_bytes"`
	TensorHome             Qwen38VulkanTensorHomeCounters `json:"tensor_home"`
}

// Qwen38VulkanDecodeReceipt captures comparable work and cost for one packet.
type Qwen38VulkanDecodeReceipt struct {
	Schema                 string                     `json:"schema"`
	Packet                 Qwen38VulkanDecodePacket   `json:"packet"`
	PacketSHA256           string                     `json:"packet_sha256"`
	Backend                string                     `json:"backend"`
	Runtime                string                     `json:"runtime"`
	Fallback               string                     `json:"fallback"`
	OutputTokenIDs         []int32                    `json:"output_token_ids"`
	OutputTokenIDsSHA256   string                     `json:"output_token_ids_sha256"`
	GeneratedTokens        int                        `json:"generated_tokens"`
	ElapsedNanoseconds     uint64                     `json:"elapsed_nanoseconds"`
	PeakProcessMemoryBytes uint64                     `json:"peak_process_memory_bytes"`
	PeakDeviceMemoryBytes  uint64                     `json:"peak_device_memory_bytes"`
	Counters               Qwen38VulkanDecodeCounters `json:"counters"`
}

// Qwen38VulkanTokenIDsSHA256 hashes signed token IDs as a count followed by
// big-endian 32-bit values, avoiding architecture- and JSON-dependent hashes.
func Qwen38VulkanTokenIDsSHA256(tokenIDs []int32) string {
	buf := make([]byte, 8+4*len(tokenIDs))
	binary.BigEndian.PutUint64(buf[:8], uint64(len(tokenIDs)))
	for i, tokenID := range tokenIDs {
		binary.BigEndian.PutUint32(buf[8+4*i:], uint32(tokenID))
	}
	sum := sha256.Sum256(buf)
	return strings.ToUpper(hex.EncodeToString(sum[:]))
}

func (r Qwen38VulkanDecodeReceipt) Validate() error {
	if r.Schema != Qwen38VulkanDecodeReceiptSchema {
		return fmt.Errorf("qwen3.8 Vulkan receipt schema %q, want %q", r.Schema, Qwen38VulkanDecodeReceiptSchema)
	}
	if err := r.Packet.Validate(); err != nil {
		return fmt.Errorf("qwen3.8 Vulkan receipt packet: %w", err)
	}
	packetDigest, err := r.Packet.Digest()
	if err != nil {
		return err
	}
	if r.PacketSHA256 != packetDigest {
		return fmt.Errorf("qwen3.8 Vulkan packet digest %q, want %q", r.PacketSHA256, packetDigest)
	}
	if r.Backend != r.Packet.Backend || r.Runtime != r.Packet.Runtime || r.Fallback != r.Packet.Fallback {
		return fmt.Errorf("qwen3.8 Vulkan receipt engine identity does not match packet")
	}
	if r.GeneratedTokens != r.Packet.GeneratedTokenLimit || r.GeneratedTokens != len(r.OutputTokenIDs) {
		return fmt.Errorf("qwen3.8 Vulkan work boundary mismatch: limit=%d generated=%d output_ids=%d", r.Packet.GeneratedTokenLimit, r.GeneratedTokens, len(r.OutputTokenIDs))
	}
	if r.OutputTokenIDsSHA256 != Qwen38VulkanTokenIDsSHA256(r.OutputTokenIDs) {
		return fmt.Errorf("qwen3.8 Vulkan output token digest %q does not match token IDs", r.OutputTokenIDsSHA256)
	}
	if r.ElapsedNanoseconds == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires positive elapsed time")
	}
	if r.PeakProcessMemoryBytes == 0 || r.PeakDeviceMemoryBytes == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires positive process and device peak memory")
	}
	if r.PeakDeviceMemoryBytes > r.PeakProcessMemoryBytes {
		return fmt.Errorf("qwen3.8 Vulkan device peak %d exceeds process peak %d", r.PeakDeviceMemoryBytes, r.PeakProcessMemoryBytes)
	}
	return r.Counters.validate()
}

func (c Qwen38VulkanDecodeCounters) validate() error {
	if c.ComputeDispatches == 0 || c.DispatchSubmits == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires compute dispatches and dispatch submits")
	}
	if c.ComputeDispatches != c.Q4KMatmulDispatches+c.OtherComputeDispatches {
		return fmt.Errorf("qwen3.8 Vulkan compute dispatch total %d != q4_k %d + other %d", c.ComputeDispatches, c.Q4KMatmulDispatches, c.OtherComputeDispatches)
	}
	for name, transfer := range map[string]Qwen38VulkanTransferCounters{"h2d": c.H2D, "d2h": c.D2H, "d2d": c.D2D} {
		if (transfer.Count == 0) != (transfer.Bytes == 0) {
			return fmt.Errorf("qwen3.8 Vulkan %s count/bytes presence mismatch", name)
		}
	}
	if (c.Q4KStageCalls == 0) != (c.Q4KStageBytes == 0) {
		return errors.New("qwen3.8 Vulkan q4_k stage calls/bytes presence mismatch")
	}
	if c.TensorHome.Admissions > c.D2D.Count || c.TensorHome.CopiedBytes > c.D2D.Bytes {
		return errors.New("qwen3.8 Vulkan tensor-home admissions/copies exceed d2d work")
	}
	if c.TensorHome.ResidentBytes > c.TensorHome.CopiedBytes {
		return errors.New("qwen3.8 Vulkan tensor-home resident bytes exceed copied bytes")
	}
	if c.TensorHome.Hits+c.TensorHome.Admissions+c.TensorHome.Bypasses == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires tensor-home accounting")
	}
	return nil
}

// CompareQwen38VulkanDecodeReceipts rejects unequal request, output, or token
// work boundaries while intentionally allowing performance counters to differ.
func CompareQwen38VulkanDecodeReceipts(parent, candidate Qwen38VulkanDecodeReceipt) error {
	if err := parent.Validate(); err != nil {
		return fmt.Errorf("parent receipt: %w", err)
	}
	if err := candidate.Validate(); err != nil {
		return fmt.Errorf("candidate receipt: %w", err)
	}
	if parent.PacketSHA256 != candidate.PacketSHA256 || !equalQwen38VulkanPackets(parent.Packet, candidate.Packet) {
		return errors.New("parent/candidate qwen3.8 Vulkan packet mismatch")
	}
	if parent.GeneratedTokens != candidate.GeneratedTokens {
		return errors.New("parent/candidate qwen3.8 Vulkan generated-token boundary mismatch")
	}
	if parent.OutputTokenIDsSHA256 != candidate.OutputTokenIDsSHA256 || !slices.Equal(parent.OutputTokenIDs, candidate.OutputTokenIDs) {
		return errors.New("parent/candidate qwen3.8 Vulkan output mismatch")
	}
	return nil
}

func equalQwen38VulkanPackets(a, b Qwen38VulkanDecodePacket) bool {
	return a.Schema == b.Schema &&
		a.ModelGGUFSHA256 == b.ModelGGUFSHA256 &&
		slices.Equal(a.PromptTokenIDs, b.PromptTokenIDs) &&
		a.GeneratedTokenLimit == b.GeneratedTokenLimit &&
		a.Backend == b.Backend && a.Runtime == b.Runtime && a.Fallback == b.Fallback
}

// ExecuteQwen38VulkanDecodeContract keeps lifecycle cleanup independent of the
// live runtime. The deferred call runs exactly once on return or panic.
func ExecuteQwen38VulkanDecodeContract[T any](cleanup func(), run func() (T, error)) (T, error) {
	if cleanup == nil {
		var zero T
		return zero, errors.New("qwen3.8 Vulkan decode cleanup is required")
	}
	defer cleanup()
	if run == nil {
		var zero T
		return zero, errors.New("qwen3.8 Vulkan decode runner is required")
	}
	return run()
}
