package bench

// deepseekinventory.go — the immutable DeepSeek V4 Pro provisioning inventory and
// its typed single-node admission verdict (#4788; consumed by #4781).
//
// #4788 is a provisioning spine, not a kernel task: pin the official V4 Pro artifact
// to an immutable revision, then decide — BEFORE transferring ~805 GiB — whether any
// sanctioned node can actually hold it. The upstream inference path documents no
// CPU/NVMe offload recipe, so the artifact must fit aggregate device HBM or it does
// not run at all. That reduces the feasibility question to arithmetic over two
// witnessed numbers (artifact bytes vs a node's aggregate HBM), which a checked-in
// leaf can answer honestly with no GPU in the loop — the admission refusal is
// derived here, not asserted by a worker's narration.
//
// Honesty posture (the #4788 confusion fence, load-bearing):
//
//   - This file records PROVENANCE and a typed FEASIBILITY verdict. It is NOT a
//     runtime witness: nothing here loads weights, runs inference, or reports
//     throughput, and no number below is a fak-authored saving.
//   - A refusal is a first-class recorded RESULT, not a failure to try. #4788's DoD
//     explicitly accepts "a runnable quantization/parallel plan, OR preserves a typed
//     infeasibility result" — this leaf is the second branch, preserved in a form
//     #4781 can consume rather than re-derive.
//   - The pure-fak synthetic DeepSeek scorecard/KV/MoE seams elsewhere in the tree
//     (and internal/deepseekbench's HOSTED provider scorecard) are a DIFFERENT
//     artifact class. Neither is V4 Pro weights; none of the three may be conflated.
//
// The DoD items this leaf does NOT satisfy stay open: the artifact is transfer-refused
// until #4801 reserves non-evicting multi-node capacity, so deterministic inference and
// the #4781 runtime handoff remain unwitnessed. Provisioning facts are recorded from the
// upstream repository metadata and the private-bridge node read-backs on #4788, scrubbed
// to node-class labels — no hosts, channels, tokens, or private paths.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// DeepSeekInventorySchema tags the inventory artifact (bumped on a breaking field
// change), so #4781 can pin the shape it consumes.
const DeepSeekInventorySchema = "deepseek-v4-pro-inventory/1"

// bytesPerGiB is the binary gibibyte used for every HBM figure below, matching the
// aggregate arithmetic the #4788 read-backs quote (8 x 80 GiB => 640 GiB).
const bytesPerGiB int64 = 1 << 30

// AdmissionVerdict is the closed set of typed feasibility outcomes for placing the
// artifact on one node. A closed vocabulary keeps a refusal checkable — a consumer
// switches on the verdict instead of parsing prose.
type AdmissionVerdict string

const (
	// AdmissionFits — the node's aggregate HBM holds the artifact and enough ranks
	// are free to place it without evicting a peer.
	AdmissionFits AdmissionVerdict = "FITS_SINGLE_NODE"
	// AdmissionInsufficientHBM — even with every rank on the node free, aggregate
	// HBM is below the artifact size. A reservation cannot fix this; only a smaller
	// artifact (quantization) or more nodes can.
	AdmissionInsufficientHBM AdmissionVerdict = "INSUFFICIENT_AGGREGATE_HBM"
	// AdmissionNeedsReservation — aggregate HBM would hold the artifact, but the
	// currently free ranks do not. Reservation-solvable WITHOUT eviction; this is
	// the case #4801 owns.
	AdmissionNeedsReservation AdmissionVerdict = "INSUFFICIENT_FREE_HBM_NEEDS_RESERVATION"
	// AdmissionNoGPU — the node exposes no GPU device at all.
	AdmissionNoGPU AdmissionVerdict = "NO_GPU_DEVICE"
	// AdmissionNotSingleNode — the rollup across every witnessed node when none of
	// them admits the artifact on its own.
	AdmissionNotSingleNode AdmissionVerdict = "NOT_SINGLE_NODE_ADMISSIBLE"
)

// NodeCapacity is one sanctioned node's device capacity as witnessed by a private
// bridge read-back, reduced to a scrubbed node-class label plus counts. Deliberately
// carries no host, address, mount, or channel.
type NodeCapacity struct {
	Name           string `json:"name"`              // scrubbed node-class label
	GPUModel       string `json:"gpu_model"`         // device model as reported by the node
	GPUCount       int    `json:"gpu_count"`         // total ranks present
	FreeGPUCount   int    `json:"free_gpu_count"`    // ranks with no peer allocation
	HBMBytesPerGPU int64  `json:"hbm_bytes_per_gpu"` // nominal per-rank HBM
	Note           string `json:"note,omitempty"`    // why ranks are unavailable, if any
}

// AggregateHBMBytes is the node's total device memory across every rank — the hard
// physical ceiling, reachable only by evicting peers.
func (n NodeCapacity) AggregateHBMBytes() int64 { return int64(n.GPUCount) * n.HBMBytesPerGPU }

// FreeHBMBytes is the device memory on ranks currently free of peer allocations —
// the ceiling reachable WITHOUT disturbing another workload (#4788 forbids eviction).
func (n NodeCapacity) FreeHBMBytes() int64 { return int64(n.FreeGPUCount) * n.HBMBytesPerGPU }

// RuntimeRequirement pins the upstream inference path's documented minima and its
// documented parallel layout. Recorded so #4781 does not re-derive them, and so a
// runtime-compatibility regression is separable from a placement refusal.
//
// The *Min fields carry a bare version ("2.10.0"), not a prose constraint (">=2.10.0"):
// the bound is already "minimum" by the field's name, and a bare version is what a
// consumer can actually compare against an installed version.
type RuntimeRequirement struct {
	TorchMin        string `json:"torch_min"`
	TransformersMin string `json:"transformers_min"`
	SafetensorsMin  string `json:"safetensors_min"`
	Experts         int    `json:"experts"`         // documented EXPERTS= for conversion
	TensorParallel  int    `json:"tensor_parallel"` // documented MP= / --nproc-per-node
	MultiNode       bool   `json:"multi_node"`      // upstream documents --nnodes/--node-rank/--master-addr
	CPUOffload      bool   `json:"cpu_offload"`     // upstream documents NO CPU/NVMe offload recipe
}

// AdmissionResult is the typed feasibility outcome for one node — the derived
// artifact #4788's DoD calls "a typed infeasibility result".
type AdmissionResult struct {
	Node              string           `json:"node"`
	Verdict           AdmissionVerdict `json:"verdict"`
	Admissible        bool             `json:"admissible"`
	WeightBytes       int64            `json:"weight_bytes"`
	AggregateHBMBytes int64            `json:"aggregate_hbm_bytes"`
	FreeHBMBytes      int64            `json:"free_hbm_bytes"`
	// ShortfallBytes is how far the reachable ceiling falls short of the artifact,
	// measured against AGGREGATE HBM (the physical ceiling), so it is 0 whenever the
	// node could hold the artifact at all. It is the number a quantization or
	// multi-node plan must close.
	ShortfallBytes int64  `json:"shortfall_bytes"`
	Why            string `json:"why"`
}

// DeepSeekInventory is the immutable, scrubbed provisioning inventory of the official
// DeepSeek V4 Pro artifact plus the derived admission verdict per witnessed node —
// the machine-readable handoff #4788 owes #4781.
type DeepSeekInventory struct {
	Schema string `json:"schema"`

	// Provenance — the immutable identity of the artifact (#4788 DoD item 1).
	ModelID        string `json:"model_id"`
	Revision       string `json:"revision"` // immutable upstream commit, never a moving tag
	License        string `json:"license"`
	Gated          bool   `json:"gated"`
	Source         string `json:"source"`
	TotalSizeBytes int64  `json:"total_size_bytes"` // from the weight index's total_size

	// Architecture — as reported by the pinned revision's config.
	ModelType       string `json:"model_type"`
	Layers          int    `json:"layers"`
	RoutedExperts   int    `json:"routed_experts"`
	ExpertsPerToken int    `json:"experts_per_token"`
	HiddenSize      int    `json:"hidden_size"`
	Precision       string `json:"precision"`
	TotalParams     string `json:"total_params"`
	ActivatedParams string `json:"activated_params"`

	Runtime RuntimeRequirement `json:"runtime"`

	// Admission — the derived feasibility read (#4788 DoD items 2 and 3).
	WitnessedNodes   []NodeCapacity    `json:"witnessed_nodes"`
	Admission        []AdmissionResult `json:"admission"`
	AdmissionSummary AdmissionVerdict  `json:"admission_summary"`

	// RuntimeWitnessed stays false until a deterministic inference actually runs.
	// #4781 must refuse to headline throughput while this is false.
	RuntimeWitnessed bool `json:"runtime_witnessed"`

	// Issue bindings, so a consumer can route a blocker without re-reading history.
	ProvisionIssue int `json:"provision_issue"` // this spine
	PlacementIssue int `json:"placement_issue"` // owns the no-eviction reservation
	ConsumerIssue  int `json:"consumer_issue"`  // waits on the runtime handle

	// Digest is the SHA-256 of the canonical inventory with Digest cleared, so the
	// artifact self-verifies after transport (same posture as cpumemstress).
	Digest string `json:"digest"`
}

// admitNode derives one node's typed verdict from witnessed capacity. The order of
// checks is the order the constraints bind: no device at all, then the physical
// aggregate ceiling (which no reservation can lift), then the free-rank ceiling (which
// a reservation CAN lift without evicting a peer).
func admitNode(n NodeCapacity, weightBytes int64) AdmissionResult {
	agg, free := n.AggregateHBMBytes(), n.FreeHBMBytes()
	r := AdmissionResult{
		Node:              n.Name,
		WeightBytes:       weightBytes,
		AggregateHBMBytes: agg,
		FreeHBMBytes:      free,
	}
	switch {
	case n.GPUCount == 0:
		r.Verdict, r.ShortfallBytes = AdmissionNoGPU, weightBytes
		r.Why = "node exposes no GPU device; the artifact has no device memory to occupy"
	case agg < weightBytes:
		r.Verdict, r.ShortfallBytes = AdmissionInsufficientHBM, weightBytes-agg
		r.Why = "aggregate HBM across every rank is below the artifact size; no reservation " +
			"can lift this ceiling — it needs a smaller artifact or more nodes"
	case free < weightBytes:
		r.Verdict = AdmissionNeedsReservation
		r.Why = "aggregate HBM would hold the artifact but the free ranks do not; " +
			"reservation-solvable without evicting a peer"
	default:
		r.Verdict, r.Admissible = AdmissionFits, true
		r.Why = "free ranks hold the artifact without disturbing a peer workload"
	}
	return r
}

// summarizeAdmission rolls per-node verdicts into the fleet-level read: any node that
// admits the artifact makes the fleet admissible; otherwise the artifact is refused
// single-node placement everywhere witnessed.
func summarizeAdmission(results []AdmissionResult) AdmissionVerdict {
	for _, r := range results {
		if r.Admissible {
			return AdmissionFits
		}
	}
	return AdmissionNotSingleNode
}

// witnessedNodes is the sanctioned-node capacity read back over the private bridge for
// #4788, scrubbed to node-class labels. HBM is the nominal per-rank figure (the
// aggregate arithmetic the read-backs quote); a rank's usable HBM is slightly lower, so
// a refusal derived from the nominal figure is the CONSERVATIVE direction — the real
// shortfall is larger, never smaller.
//
// GPUModel carries the device CLASS and its nominal capacity only. The board-interconnect
// SKU suffix that tools/scrub_hardware_names.py treats as an unconditional lab tell is
// dropped: it identifies the operator's private box without changing any number derived
// below, and TestDeepSeekInventoryScrubbed keeps it out mechanically.
func witnessedNodes() []NodeCapacity {
	return []NodeCapacity{
		{
			Name:           "node-a",
			GPUModel:       "NVIDIA A100-80GB",
			GPUCount:       8,
			FreeGPUCount:   6,
			HBMBytesPerGPU: 80 * bytesPerGiB,
			Note:           "two ranks carry an unrelated peer serve; #4788 forbids evicting it",
		},
		{
			Name:           "node-b",
			GPUModel:       "NVIDIA A100-40GB",
			GPUCount:       8,
			FreeGPUCount:   0,
			HBMBytesPerGPU: 40 * bytesPerGiB,
			Note:           "every rank holds a peer allocation despite ~0% instantaneous utilization",
		},
		{
			Name:           "node-c",
			GPUModel:       "NVIDIA A100-40GB",
			GPUCount:       8,
			FreeGPUCount:   0,
			HBMBytesPerGPU: 40 * bytesPerGiB,
			Note:           "every rank holds a peer allocation despite ~0% instantaneous utilization",
		},
		{
			Name:           "node-d",
			GPUModel:       "",
			GPUCount:       0,
			FreeGPUCount:   0,
			HBMBytesPerGPU: 0,
			Note:           "CPU/DC node; no GPU device",
		},
	}
}

// DeepSeekV4ProInventory returns the pinned inventory with admission derived from the
// witnessed node capacity and the digest computed over the result. It is a pure
// function — no network, no device, no key — so it runs anywhere `go test` runs and
// #4781 can import it directly rather than re-deriving the provisioning read.
func DeepSeekV4ProInventory() DeepSeekInventory {
	inv := DeepSeekInventory{
		Schema: DeepSeekInventorySchema,

		ModelID:        "deepseek-ai/DeepSeek-V4-Pro",
		Revision:       "b5968e9190ef611bbf34a7229255be88a0e937c1",
		License:        "MIT",
		Gated:          false,
		Source:         "huggingface",
		// The one pinned artifact size for the whole package: the placement seam
		// (#4801) already exports it as ArtifactBytes and measures its reservation bars
		// against it. Re-typing the literal here would let the two sibling records drift
		// onto different artifacts while both still claim to describe this revision.
		TotalSizeBytes: ArtifactBytes,

		ModelType:       "deepseek_v4",
		Layers:          61,
		RoutedExperts:   384,
		ExpertsPerToken: 6,
		HiddenSize:      7168,
		Precision:       "mixed: MoE expert parameters FP4, most other parameters FP8",
		TotalParams:     "1.6T",
		ActivatedParams: "49B",

		Runtime: RuntimeRequirement{
			TorchMin:        "2.10.0",
			TransformersMin: "5.0.0",
			SafetensorsMin:  "0.7.0",
			Experts:         384,
			TensorParallel:  8,
			MultiNode:       true,
			CPUOffload:      false,
		},

		RuntimeWitnessed: false,

		ProvisionIssue: 4788,
		PlacementIssue: 4801,
		ConsumerIssue:  4781,
	}

	inv.WitnessedNodes = witnessedNodes()
	for _, n := range inv.WitnessedNodes {
		inv.Admission = append(inv.Admission, admitNode(n, inv.TotalSizeBytes))
	}
	inv.AdmissionSummary = summarizeAdmission(inv.Admission)
	inv.Digest = inv.computeDigest()
	return inv
}

// computeDigest hashes the canonical inventory with Digest cleared, so the artifact
// verifies itself without a sidecar.
func (inv DeepSeekInventory) computeDigest() string {
	c := inv
	c.Digest = ""
	b, err := json.Marshal(c)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// VerifyDigest reports whether the inventory's recorded digest matches its content —
// the check a consumer runs after reading the artifact from disk or the wire.
func (inv DeepSeekInventory) VerifyDigest() bool {
	return inv.Digest != "" && inv.Digest == inv.computeDigest()
}

// Admissible reports whether any witnessed node can hold the artifact today. #4781
// must gate its capture on this: while it is false there is no runtime handle to take,
// and any V4 Pro throughput number would be unbacked by a real placement.
func (inv DeepSeekInventory) Admissible() bool {
	return inv.AdmissionSummary == AdmissionFits
}

// BlockingIssue names the issue that owns the next move when the artifact is not
// admissible — the placement reservation. It returns 0 once placement admits.
func (inv DeepSeekInventory) BlockingIssue() int {
	if inv.Admissible() {
		return 0
	}
	return inv.PlacementIssue
}
