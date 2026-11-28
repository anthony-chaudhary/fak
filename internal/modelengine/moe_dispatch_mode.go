package modelengine

// Adapted from deepseek-ai/DeepEP (MIT; Copyright (c) 2025 DeepSeek). The
// two-mode expert-parallel dispatch contract is taken from the classic
// intranode/internode Buffer at csrc/config.hpp (Config::get_nvl_buffer_size_hint,
// Config::get_rdma_buffer_size_hint, LowLatencyLayout, and
// get_low_latency_rdma_size_hint) and csrc/kernels/configs.cuh
// (NUM_MAX_NVL_PEERS, NUM_MAX_RDMA_PEERS, NUM_BUFFER_ALIGNMENT_BYTES), pinned at
// fak-owned Go.
//
// DeepEP dispatches expert-parallel tokens in one of two modes. Normal
// (high-throughput) mode is the prefill path: it moves large token batches over
// NVLink within a node and over RDMA between nodes, so it accounts for two
// buffers, an intranode NVLink buffer and an internode RDMA buffer. Low-latency
// mode is the decode path: it moves a small per-rank token batch over RDMA for
// both the intra- and internode hops, so it accounts for a single RDMA buffer
// whose size is a function of the per-rank dispatch cap, the hidden width, the
// rank count, and the expert count.
//
// This leaf ports the host-side mode-selection contract and the per-mode buffer
// accounting only. It is a default-off planning step: it never touches
// RDMA/NVLink hardware, allocates no device memory, and makes no speed claim.
// The accounting mirrors the upstream byte formulas so a later leaf that wires
// the split into the MoE dispatch path can size its buffers from the same
// numbers the H200 recipe study cites as the mode split behind the largest
// published MoE serving wins.

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
)

// MoEDispatchPhase is the serving phase a MoE dispatch step is planning for.
type MoEDispatchPhase int

// The two serving phases the two-mode split distinguishes.
const (
	// MoEDispatchPhasePrefill is the high-throughput prefill phase.
	MoEDispatchPhasePrefill MoEDispatchPhase = iota
	// MoEDispatchPhaseDecode is the low-latency decode phase.
	MoEDispatchPhaseDecode
)

// String returns the phase name.
func (p MoEDispatchPhase) String() string {
	switch p {
	case MoEDispatchPhasePrefill:
		return "prefill"
	case MoEDispatchPhaseDecode:
		return "decode"
	default:
		return "unknown"
	}
}

// MoEDispatchMode is the DeepEP dispatch mode a phase selects.
type MoEDispatchMode int

// The two dispatch modes.
const (
	// MoEDispatchModeNormal is the high-throughput normal mode (prefill).
	MoEDispatchModeNormal MoEDispatchMode = iota
	// MoEDispatchModeLowLatency is the low-latency mode (decode).
	MoEDispatchModeLowLatency
)

// String returns the mode name.
func (m MoEDispatchMode) String() string {
	switch m {
	case MoEDispatchModeNormal:
		return "normal"
	case MoEDispatchModeLowLatency:
		return "low-latency"
	default:
		return "unknown"
	}
}

// ErrMoEDispatchPhaseUnknown is returned when a phase is neither prefill nor
// decode. Callers may use errors.Is.
var ErrMoEDispatchPhaseUnknown = errors.New("modelengine: moe dispatch: unknown phase")

// Closed DeepEP topology and layout constants, mirrored from
// csrc/kernels/configs.cuh and csrc/config.hpp.
const (
	// moeDispatchMaxNVLPeers is NUM_MAX_NVL_PEERS: the widest NVLink domain.
	moeDispatchMaxNVLPeers = 8
	// moeDispatchMaxRDMAPeers is NUM_MAX_RDMA_PEERS: the widest RDMA domain.
	moeDispatchMaxRDMAPeers = 20
	// moeDispatchBufferAlignmentBytes is NUM_BUFFER_ALIGNMENT_BYTES: every
	// buffer is aligned to this many bytes.
	moeDispatchBufferAlignmentBytes = 128
	// moeDispatchMaxTopK is kNumMaxTopK: the widest top-k the buffers reserve.
	moeDispatchMaxTopK = 128
	// moeDispatchMaxScales is kNumMaxScales: the widest FP8 scale set.
	moeDispatchMaxScales = 128
	// moeDispatchSourceMetaBytes is internode::get_source_meta_bytes(): the
	// per-token source-metadata header carried by the internode buffers.
	moeDispatchSourceMetaBytes = 16
	// moeDispatchInt4Bytes is sizeof(int4).
	moeDispatchInt4Bytes = 16
	// moeDispatchBFloat16Bytes is sizeof(nv_bfloat16).
	moeDispatchBFloat16Bytes = 2
	// moeDispatchBFloat162Bytes is sizeof(nv_bfloat162).
	moeDispatchBFloat162Bytes = 4
	// moeDispatchIntBytes is sizeof(int).
	moeDispatchIntBytes = 4
	// moeDispatchFloatBytes is sizeof(float).
	moeDispatchFloatBytes = 4
	// moeDispatchTopkIdxBytes is sizeof(topk_idx_t) (int32).
	moeDispatchTopkIdxBytes = 4
)

// MoEDispatchBufferAccounting is one mode's host-side buffer byte accounting.
// Normal mode carries both an intranode NVLink buffer and an internode RDMA
// buffer; low-latency mode carries a single RDMA buffer used for both the
// intra- and internode hops, so its NVLinkBytes is 0.
type MoEDispatchBufferAccounting struct {
	// NVLinkBytes is the intranode NVLink buffer size (normal mode only).
	NVLinkBytes int64
	// RDMABytes is the internode RDMA buffer size (both modes).
	RDMABytes int64
	// TotalBytes is the mode's total buffer footprint.
	TotalBytes int64
}

// MoEDispatchPlan is the result of the two-mode dispatch planning step for one
// phase (or a homogeneous run of phases).
type MoEDispatchPlan struct {
	// Mode is the dispatch mode the phase selects.
	Mode MoEDispatchMode
	// Phase is the phase the plan was computed for.
	Phase MoEDispatchPhase
	// Accounting is the mode's per-mode buffer byte accounting.
	Accounting MoEDispatchBufferAccounting
	// NoOp is true when the planning step is a no-op: the step is off, or the
	// phases are homogeneous and a single mode already covers them.
	NoOp bool
}

// IsNoOp reports whether the plan is a no-op.
func (p MoEDispatchPlan) IsNoOp() bool { return p.NoOp }

// MoEDispatchModeForPhase selects the dispatch mode for a phase: prefill takes
// the high-throughput normal mode, decode takes the low-latency mode. It fails
// closed on any other phase.
func MoEDispatchModeForPhase(phase MoEDispatchPhase) (MoEDispatchMode, error) {
	switch phase {
	case MoEDispatchPhasePrefill:
		return MoEDispatchModeNormal, nil
	case MoEDispatchPhaseDecode:
		return MoEDispatchModeLowLatency, nil
	default:
		return MoEDispatchModeNormal, fmt.Errorf("%w: %v", ErrMoEDispatchPhaseUnknown, phase)
	}
}

// moeDispatchAlignUp rounds a up to the next multiple of align.
func moeDispatchAlignUp(a, align int64) int64 {
	return (a + align - 1) / align * align
}

// MoEDispatchNormalNVLinkBytesHint mirrors Config::get_nvl_buffer_size_hint:
// the intranode NVLink buffer size for normal mode. numRanks is the EP group
// width, numSMS the SM count the high-throughput kernels launch on, hiddenBytes
// the per-token hidden width in bytes, and numMaxNVLChunkedRecvTokens the
// per-channel receive chunk cap. It is 0 when numRanks or numSMS is not
// positive.
func MoEDispatchNormalNVLinkBytesHint(numRanks, numSMS, hiddenBytes, numMaxNVLChunkedRecvTokens int) int64 {
	if numRanks <= 0 || numSMS <= 0 {
		return 0
	}
	numRDMARanks := int(math.Max(float64(numRanks/moeDispatchMaxNVLPeers), 1))
	numNVLRanks := int(math.Min(float64(numRanks), float64(moeDispatchMaxNVLPeers)))
	numChannels := numSMS / 2
	var numBytes int64
	numBytes += int64(numChannels) * int64(numNVLRanks) * int64(2*numRDMARanks+3) * moeDispatchIntBytes
	numBytes += int64(numChannels) * int64(numNVLRanks) * int64(numMaxNVLChunkedRecvTokens) * int64(hiddenBytes)
	numBytes += int64(numChannels) * int64(numNVLRanks) * int64(numMaxNVLChunkedRecvTokens) * moeDispatchSourceMetaBytes
	numBytes += int64(numChannels) * int64(numNVLRanks) * int64(numMaxNVLChunkedRecvTokens) * int64(moeDispatchMaxTopK) * moeDispatchTopkIdxBytes
	numBytes += int64(numChannels) * int64(numNVLRanks) * int64(numMaxNVLChunkedRecvTokens) * int64(moeDispatchMaxTopK) * moeDispatchFloatBytes
	numBytes += int64(numChannels) * int64(numNVLRanks) * int64(numMaxNVLChunkedRecvTokens) * int64(moeDispatchMaxScales) * moeDispatchFloatBytes
	return moeDispatchAlignUp(numBytes, moeDispatchBufferAlignmentBytes)
}

// MoEDispatchNormalRDMABytesHint mirrors Config::get_rdma_buffer_size_hint:
// the internode RDMA buffer size for normal mode. It is 0 when the EP group is
// intranode-only (numRanks <= NUM_MAX_NVL_PEERS), because then no internode
// hop exists. numRanks must be a multiple of NUM_MAX_NVL_PEERS when it exceeds
// it; the formula is still evaluated otherwise so a caller can size a buffer
// for a wider group.
func MoEDispatchNormalRDMABytesHint(numRanks, numSMS, hiddenBytes, numMaxRDMACheckedRecvTokens int) int64 {
	if numRanks <= moeDispatchMaxNVLPeers || numSMS <= 0 {
		return 0
	}
	numRDMARanks := numRanks / moeDispatchMaxNVLPeers
	numChannels := numSMS / 2
	var numBytes int64
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(moeDispatchMaxNVLPeers*2+2) * 2 * moeDispatchIntBytes
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(numMaxRDMACheckedRecvTokens) * int64(hiddenBytes) * 2
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(numMaxRDMACheckedRecvTokens) * moeDispatchSourceMetaBytes * 2
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(numMaxRDMACheckedRecvTokens) * int64(moeDispatchMaxTopK) * moeDispatchTopkIdxBytes * 2
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(numMaxRDMACheckedRecvTokens) * int64(moeDispatchMaxTopK) * moeDispatchFloatBytes * 2
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(numMaxRDMACheckedRecvTokens) * int64(moeDispatchMaxScales) * moeDispatchFloatBytes * 2
	numBytes += int64(numChannels) * int64(numRDMARanks) * int64(numMaxRDMACheckedRecvTokens) * moeDispatchInt4Bytes * 2
	return moeDispatchAlignUp(numBytes, moeDispatchBufferAlignmentBytes)
}

// MoEDispatchLowLatencyRDMABytesHint mirrors get_low_latency_rdma_size_hint:
// the single RDMA buffer size for low-latency mode, used for both the intra-
// and internode hops. numMaxDispatchTokensPerRank is the per-rank dispatch cap,
// hidden the hidden width in elements, numRanks the EP group width, and
// numExperts the total expert count. It is 0 when any dimension is not
// positive.
func MoEDispatchLowLatencyRDMABytesHint(numMaxDispatchTokensPerRank, hidden, numRanks, numExperts int) int64 {
	if numMaxDispatchTokensPerRank <= 0 || hidden <= 0 || numRanks <= 0 || numExperts <= 0 {
		return 0
	}
	numScales := hidden / 128
	// num_bytes_per_dispatch_msg = sizeof(int4) + max(hidden * sizeof(bf16),
	// hidden + num_scales * sizeof(float)).
	numBytesPerDispatchMsg := int64(moeDispatchInt4Bytes) +
		int64(math.Max(float64(hidden*moeDispatchBFloat16Bytes), float64(hidden+numScales*moeDispatchFloatBytes)))
	// num_bytes_per_combine_msg = num_scales * sizeof(bf162) + hidden * sizeof(bf16).
	numBytesPerCombineMsg := int64(numScales)*moeDispatchBFloat162Bytes + int64(hidden)*moeDispatchBFloat16Bytes

	// Send buffer: max(dispatch_send, combine_send), doubled for the two
	// symmetric odd/even send buffers.
	dispatchSend := int64(numMaxDispatchTokensPerRank) * numBytesPerDispatchMsg
	combineSend := int64(numExperts) * int64(numMaxDispatchTokensPerRank) * numBytesPerCombineMsg
	sendBuffer := int64(math.Max(float64(dispatchSend), float64(combineSend)))
	total := sendBuffer * 2

	// Receive buffer: max(dispatch_recv, combine_recv), doubled for the two
	// symmetric odd/even receive buffers.
	dispatchRecv := int64(numExperts) * int64(numMaxDispatchTokensPerRank) * numBytesPerDispatchMsg
	combineRecv := int64(numExperts) * int64(numMaxDispatchTokensPerRank) * numBytesPerCombineMsg
	recvBuffer := int64(math.Max(float64(dispatchRecv), float64(combineRecv)))
	total += recvBuffer * 2

	// Signaling buffer: num_experts * sizeof(int), aligned, doubled for the two
	// symmetric odd/even signaling buffers.
	signaling := int64(numExperts) * moeDispatchIntBytes
	signalingAligned := moeDispatchAlignUp(signaling, moeDispatchBufferAlignmentBytes)
	total += signalingAligned * 2

	return moeDispatchAlignUp(total, moeDispatchBufferAlignmentBytes)
}

// MoEDispatchLowLatencyQPDepth returns the minimum NVSHMEM queue-pair depth the
// low-latency mode requires: (numMaxDispatchTokensPerRank + 1) * 2, the bound
// DeepEP asserts on its nvshmem_qp_depth.
func MoEDispatchLowLatencyQPDepth(numMaxDispatchTokensPerRank int) int {
	return (numMaxDispatchTokensPerRank + 1) * 2
}

// MoEDispatchPlanForPhase computes the two-mode dispatch plan for a single
// phase: it selects the mode by phase and fills the mode's per-mode buffer
// accounting. It fails closed on an unknown phase.
func MoEDispatchPlanForPhase(phase MoEDispatchPhase, numRanks, numSMS, hiddenBytes, numMaxDispatchTokensPerRank, numExperts, numMaxNVLChunkedRecvTokens, numMaxRDMACheckedRecvTokens int) (MoEDispatchPlan, error) {
	mode, err := MoEDispatchModeForPhase(phase)
	if err != nil {
		return MoEDispatchPlan{}, err
	}
	var acc MoEDispatchBufferAccounting
	switch mode {
	case MoEDispatchModeNormal:
		acc.NVLinkBytes = MoEDispatchNormalNVLinkBytesHint(numRanks, numSMS, hiddenBytes, numMaxNVLChunkedRecvTokens)
		acc.RDMABytes = MoEDispatchNormalRDMABytesHint(numRanks, numSMS, hiddenBytes, numMaxRDMACheckedRecvTokens)
		acc.TotalBytes = acc.NVLinkBytes + acc.RDMABytes
	case MoEDispatchModeLowLatency:
		// Low-latency mode uses the RDMA buffer for both the intra- and
		// internode hops, so it carries no separate NVLink buffer.
		acc.RDMABytes = MoEDispatchLowLatencyRDMABytesHint(numMaxDispatchTokensPerRank, hiddenBytes, numRanks, numExperts)
		acc.TotalBytes = acc.RDMABytes
	}
	return MoEDispatchPlan{Mode: mode, Phase: phase, Accounting: acc}, nil
}

// MoEDispatchPlanForPhases computes the two-mode dispatch plan for a run of
// phases. When the run is homogeneous (every phase identical) a single mode
// already covers the run, so the step is a no-op: it returns that mode's plan
// with NoOp set and no per-phase split. When the run is mixed it returns the
// first phase's plan with NoOp clear, signaling that the caller must split the
// run by phase and plan each homogeneous segment separately. It fails closed on
// an unknown phase anywhere in the run.
func MoEDispatchPlanForPhases(phases []MoEDispatchPhase, numRanks, numSMS, hiddenBytes, numMaxDispatchTokensPerRank, numExperts, numMaxNVLChunkedRecvTokens, numMaxRDMACheckedRecvTokens int) (MoEDispatchPlan, error) {
	if len(phases) == 0 {
		return MoEDispatchPlan{NoOp: true}, nil
	}
	// Validate every phase up front so a bad phase anywhere fails the whole
	// run closed.
	for _, p := range phases {
		if _, err := MoEDispatchModeForPhase(p); err != nil {
			return MoEDispatchPlan{}, err
		}
	}
	plan, err := MoEDispatchPlanForPhase(phases[0], numRanks, numSMS, hiddenBytes, numMaxDispatchTokensPerRank, numExperts, numMaxNVLChunkedRecvTokens, numMaxRDMACheckedRecvTokens)
	if err != nil {
		return MoEDispatchPlan{}, err
	}
	homogeneous := true
	for _, p := range phases[1:] {
		if p != phases[0] {
			homogeneous = false
			break
		}
	}
	plan.NoOp = homogeneous
	return plan, nil
}

// envMoEDispatchModeSplit is the default-off opt-in for the two-mode dispatch
// planning step. Unset (or any value that does not parse on) keeps the MoE
// dispatch path on its single historical mode, byte-for-byte.
const envMoEDispatchModeSplit = "FAK_NATIVE_MOE_DISPATCH_MODE_SPLIT"

// MoEDispatchModeSplitEnabled reports whether the two-mode dispatch planning
// step is opted in via FAK_NATIVE_MOE_DISPATCH_MODE_SPLIT.
func MoEDispatchModeSplitEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(envMoEDispatchModeSplit))) {
	case "1", "on", "true", "yes", "enable", "enabled":
		return true
	default:
		return false
	}
}

// MoEDispatchPlanStep is the default-off planning step. When the split is off
// (the default) it returns a no-op plan so the MoE dispatch path keeps its
// single historical mode. When the split is on it plans the run of phases,
// splitting a mixed run by phase and no-op'ing a homogeneous one. It fails
// closed on an unknown phase.
func MoEDispatchPlanStep(phases []MoEDispatchPhase, numRanks, numSMS, hiddenBytes, numMaxDispatchTokensPerRank, numExperts, numMaxNVLChunkedRecvTokens, numMaxRDMACheckedRecvTokens int) (MoEDispatchPlan, error) {
	if !MoEDispatchModeSplitEnabled() {
		return MoEDispatchPlan{NoOp: true}, nil
	}
	return MoEDispatchPlanForPhases(phases, numRanks, numSMS, hiddenBytes, numMaxDispatchTokensPerRank, numExperts, numMaxNVLChunkedRecvTokens, numMaxRDMACheckedRecvTokens)
}
