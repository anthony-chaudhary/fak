package compute

// EPLB logical-to-physical expert remap plus per-physical hit-count recording:
// the CPU reference of eplb_map_record.cu (ADAPT of vllm-project/vllm
// vllm/model_executor/layers/fused_moe/router/base_router.py at pin
// ae065918640a198a56f6f17eb45e251c3229fe41, Apache-2.0 — the fused Triton
// _eplb_map_and_record_i32_kernel). This file is pure Go on purpose: it compiles
// and is tested in every build, while the device kernel ships behind the cuda
// build tag once the successor policy-hook packet wires eplb_map_record.cu into
// the single libfakcuda translation unit. The cgo launch wrapper lands with that
// wiring; declaring the C ABI here would break every -tags cuda link against a
// libfakcuda.a that does not carry the symbols yet.
//
// Semantics mirror the pinned kernel entry by entry:
//   - the token index of flat panel entry i is i / NumActiveExperts;
//   - replica selection is Knuth's multiplicative hash of the token index:
//     uint32(uint64(token) * 2654435769) % max(replicaCount, 1);
//   - physical id = LogicalToPhysical[logical*MapSlots + replicaIdx];
//   - hit counts add one per mapped entry, gated by RecordEnabled and the
//     unpadded-token window, and only for physical ids in [0, NumPhysicalExperts).
//
// Deliberate divergence from upstream: an out-of-range logical id is a typed
// refusal (ErrEplbLogicalIDOutOfRange) instead of the device's -1 fallback. The
// host API fails fast because the V4.1 expert residency/admission policy consumes
// these ids without a second chance; the device kernel keeps the graceful
// fallback so a bad id mid-batch cannot abort the forward pass.

import (
	"errors"
	"fmt"
)

var (
	// ErrEplbGeometry is the closed refusal for EplbMapRecord inputs that violate
	// operand lengths or overflow the int32 device ABI.
	ErrEplbGeometry = errors.New("compute: eplb map-record geometry refused")
	// ErrEplbLogicalIDOutOfRange is the closed refusal for a topk entry whose
	// logical expert id falls outside [0, NumLogicalExperts).
	ErrEplbLogicalIDOutOfRange = errors.New("compute: eplb logical expert id out of range")
)

// EplbMapRecordArgs is one fused map-and-record call over a flat
// [numTokens][NumActiveExperts] logical expert-id panel.
type EplbMapRecordArgs struct {
	// TopkIDs is the flat logical expert-id panel, length Numel.
	TopkIDs []int32
	// NumActiveExperts is the panel's last dimension (top-k per token); the
	// token index of entry i is i / NumActiveExperts.
	NumActiveExperts int
	// NumLogicalExperts bounds valid logical ids: [0, NumLogicalExperts).
	NumLogicalExperts int
	// LogicalToPhysical is the [NumLogicalExperts][MapSlots] row-major device
	// map; entry [e][r] is physical replica r of logical expert e.
	LogicalToPhysical []int32
	// LogicalReplicaCount[e] is the live replica count of logical expert e;
	// values below 1 clamp to 1, exactly like the kernel's tl.maximum.
	LogicalReplicaCount []int32
	// MapSlots is the width of each LogicalToPhysical row.
	MapSlots int
	// NumPhysicalExperts is the hit-count vector length (upstream out_size).
	NumPhysicalExperts int
	// RecordEnabled gates hit accumulation (upstream record_enabled scalar).
	RecordEnabled bool
	// NumUnpaddedTokens limits recording to entries below
	// NumUnpaddedTokens*NumActiveExperts; -1 disables the gate entirely
	// (upstream HAS_NUM_UNPADDED=False); values below -1 are refused.
	NumUnpaddedTokens int
}

// EplbMapRecordResult is the CPU image of what fcuda_eplb_map_record produces on
// device for one EplbMapRecordArgs.
type EplbMapRecordResult struct {
	// PhysicalIDs[i] is the physical expert id for TopkIDs[i]; length len(TopkIDs).
	PhysicalIDs []int32
	// HitCounts[p] counts mapped entries that recorded onto physical expert p;
	// length NumPhysicalExperts. All zero when RecordEnabled is false.
	HitCounts []int32
}

// knuthMultiplier is floor(2^32/phi), the classic Knuth multiplicative hash
// multiplier the pinned kernel uses for replica selection.
const knuthMultiplier uint64 = 2654435769

// EplbMapRecord remaps a flat logical expert-id panel to physical expert ids and
// records per-physical hit counts, mirroring the pinned vLLM kernel. It returns
// either a fully populated result or an error — never a partial one.
func EplbMapRecord(args EplbMapRecordArgs) (*EplbMapRecordResult, error) {
	numel := len(args.TopkIDs)
	if args.NumActiveExperts <= 0 || args.NumLogicalExperts <= 0 || args.MapSlots <= 0 || args.NumPhysicalExperts <= 0 {
		return nil, fmt.Errorf("%w: active/logical/slots/physical=%d/%d/%d/%d",
			ErrEplbGeometry, args.NumActiveExperts, args.NumLogicalExperts, args.MapSlots, args.NumPhysicalExperts)
	}
	const maxInt32 = int32(1<<31 - 1)
	if numel > int(maxInt32) {
		return nil, fmt.Errorf("%w: numel=%d overflows the int32 ABI", ErrEplbGeometry, numel)
	}
	if len(args.LogicalReplicaCount) != args.NumLogicalExperts {
		return nil, fmt.Errorf("%w: logicalReplicaCount length %d want %d",
			ErrEplbGeometry, len(args.LogicalReplicaCount), args.NumLogicalExperts)
	}
	mapLen := int64(args.NumLogicalExperts) * int64(args.MapSlots)
	if mapLen < 0 || mapLen > 1<<62 {
		return nil, fmt.Errorf("%w: logical*slots overflows", ErrEplbGeometry)
	}
	if len(args.LogicalToPhysical) != int(mapLen) {
		return nil, fmt.Errorf("%w: logicalToPhysical length %d want %d",
			ErrEplbGeometry, len(args.LogicalToPhysical), mapLen)
	}
	if args.NumUnpaddedTokens < -1 {
		return nil, fmt.Errorf("%w: numUnpaddedTokens=%d must be -1 (disabled) or >= 0",
			ErrEplbGeometry, args.NumUnpaddedTokens)
	}
	windowDisabled := args.NumUnpaddedTokens < 0
	var windowEnd int64 // exclusive bound on entry offsets that may record
	if !windowDisabled {
		if args.NumUnpaddedTokens > int(maxInt32) {
			return nil, fmt.Errorf("%w: numUnpaddedTokens=%d overflows the int32 ABI",
				ErrEplbGeometry, args.NumUnpaddedTokens)
		}
		windowEnd = int64(args.NumUnpaddedTokens) * int64(args.NumActiveExperts)
	}

	ne := int32(args.NumLogicalExperts) // validated above
	for i, id := range args.TopkIDs {
		if id < 0 || id >= ne {
			return nil, fmt.Errorf("%w: entry %d has logical id %d, want [0,%d)",
				ErrEplbLogicalIDOutOfRange, i, id, args.NumLogicalExperts)
		}
	}

	phys := make([]int32, numel)
	hits := make([]int32, args.NumPhysicalExperts)
	for i, id := range args.TopkIDs {
		replicas := args.LogicalReplicaCount[id]
		if replicas < 1 {
			replicas = 1 // upstream tl.maximum(replica_count, 1)
		}
		token := int64(i) / int64(args.NumActiveExperts)
		hashed := uint32(knuthMultiplier * uint64(token))
		replicaIdx := int(hashed % uint32(replicas))
		pid := args.LogicalToPhysical[int(id)*args.MapSlots+replicaIdx]
		phys[i] = pid
		if args.RecordEnabled && (windowDisabled || int64(i) < windowEnd) &&
			pid >= 0 && pid < int32(args.NumPhysicalExperts) {
			hits[pid]++
		}
	}
	return &EplbMapRecordResult{PhysicalIDs: phys, HitCounts: hits}, nil
}
