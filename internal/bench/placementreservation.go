package bench

// placementreservation.go — the typed multi-node placement reservation gate (#4801;
// gates #4788's transfer, hands topology to #4781).
//
// #4801 is the placement prerequisite in front of #4788: BEFORE any ~805-GiB transfer
// and BEFORE any peer is disturbed, decide whether a sanctioned, operator-approved,
// non-evicting reservation actually exists that can hold the pinned artifact plus its
// runtime headroom. The sibling single-node inventory read (#4788) already refuses
// single-node placement; this leaf owns the question that refusal defers: can a
// *reserved multi-node set* admit it, and if not, exactly which resource is missing and
// who must act next.
//
// Honesty posture (load-bearing — this is a safety fence, not a capacity optimist):
//
//   - A reservation is NEVER assumed. `OperatorApproved` is an input, not an inference:
//     no probe, no read-back, and no free-rank arithmetic may synthesize approval. The
//     issue's two private-channel requests went unanswered, so the approval bit is false
//     and the gate refuses — that refusal is the correct RESULT, not a failure to try.
//   - This is not a runtime witness. Nothing here reserves, transfers, evicts, or runs a
//     collective. `CollectiveWitnessed` records whether a real NCCL smoke test passed on
//     the exact reserved ranks; while false, no topology may be handed to #4781.
//   - Refusing is the conservative direction. Every witnessed figure below is nominal
//     (per-rank usable HBM is slightly lower, allocator/CUDA/NCCL overhead is real), so a
//     refusal derived from nominal numbers understates the true shortfall, never overstates
//     the true headroom.
//
// The DoD items this leaf does NOT satisfy stay open by construction: items 1-6 each
// require an operator-approved reservation and a collective run on the reserved ranks,
// neither of which exists. This leaf is DoD item 7's documented else-branch — "emit a
// typed refusal with the missing resource and next operator action" — preserved in a form
// #4788 can gate on and an operator can act on, rather than prose on a thread.
//
// Envelope figures are scrubbed to node-class labels and aggregate counts, per the
// GPU-server private boundary: no hosts, channels, tokens, mounts, or transcripts.

import "fmt"

// PlacementReservationSchema tags the reservation verdict artifact (bumped on a breaking
// field change), so #4788 can pin the shape it gates on.
const PlacementReservationSchema = "deepseek-v4-pro-placement-reservation/1"

// reservationGiB is the binary gibibyte used for every figure below, matching the
// aggregate arithmetic the #4801 read-backs quote (8 x 80 GiB => 640 GiB).
const reservationGiB int64 = 1 << 30

// The required operating envelope, transcribed from #4801's "Target operating envelope".
// These are the bars a reservation must clear before the artifact may move; the memory
// bar already carries runtime headroom above the artifact's own bytes (see
// RequiredHeadroomBytes).
const (
	// RequiredUsableHBMBytes is aggregate USABLE accelerator memory across the reserved
	// ranks — artifact bytes plus runtime headroom for KV, activations, the allocator,
	// and CUDA/NCCL reserves.
	RequiredUsableHBMBytes int64 = 900 * reservationGiB
	// RequiredStagingStorageBytes is staging storage for the artifact on the reserved set.
	RequiredStagingStorageBytes int64 = 1000 * reservationGiB
	// RequiredWindowHours is the reserved validation window. A reservation with no
	// explicit window is not a reservation — a peer may reclaim the ranks mid-transfer.
	RequiredWindowHours = 12
	// RequiredCollectiveRanks is the rank count the NCCL smoke test must span before the
	// transfer is admissible.
	RequiredCollectiveRanks = 12
)

// ArtifactBytes is the pinned DeepSeek V4 Pro artifact size from the weight index's
// total_size (864,704,792,696 bytes = 805.319 GiB), the number every bar below is
// measured against.
const ArtifactBytes int64 = 864704792696

// RequiredHeadroomBytes is the runtime headroom the memory bar carries above the
// artifact itself. Stated as a derived constant so a consumer can see that the 900-GiB
// bar is not arbitrary: it is the artifact plus this margin.
const RequiredHeadroomBytes = RequiredUsableHBMBytes - ArtifactBytes

// ReservationVerdict is the closed set of typed placement outcomes. A closed vocabulary
// keeps a refusal checkable — #4788 switches on the verdict instead of parsing prose.
type ReservationVerdict string

const (
	// ReservationAdmitted — an operator-approved, collective-witnessed reservation clears
	// every bar. Only this verdict makes #4788 transfer-admissible.
	ReservationAdmitted ReservationVerdict = "RESERVATION_ADMITTED"
	// ReservationNoOperatorApproval — no independently confirmed non-eviction approval
	// exists for the named ranks. Capacity arithmetic cannot substitute for it.
	ReservationNoOperatorApproval ReservationVerdict = "NO_OPERATOR_APPROVAL"
	// ReservationInsufficientHBM — the reservation's usable accelerator memory is below
	// the artifact plus runtime headroom.
	ReservationInsufficientHBM ReservationVerdict = "INSUFFICIENT_NONEVICTING_HBM"
	// ReservationInsufficientStorage — staging storage on the reserved set cannot hold
	// the artifact.
	ReservationInsufficientStorage ReservationVerdict = "INSUFFICIENT_STAGING_STORAGE"
	// ReservationWindowTooShort — the reserved window is shorter than the validation run
	// needs, so a peer could reclaim the ranks mid-transfer.
	ReservationWindowTooShort ReservationVerdict = "RESERVATION_WINDOW_TOO_SHORT"
	// ReservationCollectiveUnwitnessed — every resource bar clears, but no NCCL collective
	// has been proven across the exact reserved ranks. #4801's acceptance gate requires
	// that smoke test BEFORE the transfer begins.
	ReservationCollectiveUnwitnessed ReservationVerdict = "COLLECTIVE_WITNESS_MISSING"
)

// MissingResource names one unmet bar, the size of the gap, and the operator action that
// would close it. This is the "missing resource and next operator action" pair DoD item 7
// requires, in a form a consumer can enumerate rather than parse.
type MissingResource struct {
	Verdict ReservationVerdict `json:"verdict"`
	// Have and Need are in the resource's own unit (bytes for memory/storage, hours for
	// the window, ranks for the collective). Unit names it.
	Have int64  `json:"have"`
	Need int64  `json:"need"`
	Unit string `json:"unit"`
	// Shortfall is Need-Have, clamped at 0 — the gap an operator action must close.
	Shortfall    int64  `json:"shortfall"`
	Why          string `json:"why"`
	NextOperator string `json:"next_operator_action"`
}

// ReservationEnvelope is a candidate reservation's witnessed capacity, reduced to
// scrubbed aggregate figures. Deliberately carries no host, address, mount, or channel.
//
// OperatorApproved and CollectiveWitnessed are INPUTS carrying independent evidence, not
// values this package may derive. A caller that sets either from its own narration has
// defeated the gate.
type ReservationEnvelope struct {
	// Name is a scrubbed node-class label for the candidate node set — never a host.
	Name string `json:"name"`
	// UsableHBMBytes is aggregate usable accelerator memory across ranks the reservation
	// would hold WITHOUT evicting a peer.
	UsableHBMBytes int64 `json:"usable_hbm_bytes"`
	// StagingStorageBytes is staging storage reachable from the reserved set.
	StagingStorageBytes int64 `json:"staging_storage_bytes"`
	// WindowHours is the explicitly reserved validation window. Zero means "no window
	// reserved", which is a refusal, not a long window.
	WindowHours int64 `json:"window_hours"`
	// CollectiveRanks is the rank count a passing NCCL smoke test actually spanned.
	CollectiveRanks int64 `json:"collective_ranks"`
	// OperatorApproved records an independently confirmed no-eviction/no-preemption
	// approval from the operator for these exact ranks and this window.
	OperatorApproved bool `json:"operator_approved"`
	// CollectiveWitnessed records that a real collective passed on the reserved ranks.
	CollectiveWitnessed bool `json:"collective_witnessed"`
	// Note records peer workloads or why ranks are unavailable, if any.
	Note string `json:"note,omitempty"`
}

// ReservationResult is the typed placement outcome for one candidate envelope — the
// derived artifact #4801's DoD calls "a typed refusal with the missing resource and next
// operator action", doubling as the dry-run placement table (DoD item 4).
type ReservationResult struct {
	Schema   string             `json:"schema"`
	Envelope string             `json:"envelope"`
	Verdict  ReservationVerdict `json:"verdict"`
	// Admissible is true only when EVERY bar clears. #4788 must gate its transfer on it.
	Admissible bool `json:"admissible"`

	ArtifactBytes  int64 `json:"artifact_bytes"`
	HeadroomBytes  int64 `json:"headroom_bytes"`
	RequiredHBM    int64 `json:"required_usable_hbm_bytes"`
	ActualHBM   int64 `json:"actual_usable_hbm_bytes"`
	RequiredStore  int64 `json:"required_staging_storage_bytes"`
	ActualStore int64 `json:"actual_staging_storage_bytes"`

	// Missing enumerates EVERY unmet bar, not just the binding one, so an operator sees
	// the whole gap in one read instead of discovering it one refusal at a time. Verdict
	// carries the first (most-binding) entry.
	Missing []MissingResource `json:"missing,omitempty"`
	// AbortThreshold and RollbackCommand are DoD item 6: what makes the placement abort,
	// and the exact command that undoes a partial staging.
	AbortThreshold  string `json:"abort_threshold"`
	RollbackCommand string `json:"rollback_command"`
}

// shortfall returns need-have clamped at zero.
func shortfall(have, need int64) int64 {
	if have >= need {
		return 0
	}
	return need - have
}

// AdmitPlacement derives a candidate reservation's typed verdict. It is a pure function —
// no network, no device, no key, and above all no side effect on the fleet: it cannot
// reserve, transfer, or evict. It only decides, so it runs anywhere `go test` runs.
//
// The checks are ordered by the order the constraints BIND, which is also the order an
// operator can act on them:
//
//  1. Operator approval — the safety fence. Without it there is no reservation at all,
//     and no amount of free capacity may be read as consent to occupy it.
//  2. Usable HBM, then staging storage, then the window — the resource bars.
//  3. The collective witness — last, because proving NCCL across ranks only means
//     anything once those exact ranks are actually reserved.
//
// Every unmet bar lands in Missing; Verdict reports the first.
func AdmitPlacement(e ReservationEnvelope) ReservationResult {
	r := ReservationResult{
		Schema:         PlacementReservationSchema,
		Envelope:       e.Name,
		ArtifactBytes:  ArtifactBytes,
		HeadroomBytes:  RequiredHeadroomBytes,
		RequiredHBM:    RequiredUsableHBMBytes,
		ActualHBM:   e.UsableHBMBytes,
		RequiredStore:  RequiredStagingStorageBytes,
		ActualStore: e.StagingStorageBytes,
		AbortThreshold: fmt.Sprintf(
			"abort staging if usable HBM drops below %d bytes, staging storage below %d bytes, "+
				"a peer allocation appears on a reserved rank, or the collective witness regresses",
			RequiredUsableHBMBytes, RequiredStagingStorageBytes),
		RollbackCommand: "remove the staged artifact tree from the reserved set's staging path and " +
			"release the reservation; no peer workload is touched because none was ever evicted",
	}

	if !e.OperatorApproved {
		r.Missing = append(r.Missing, MissingResource{
			Verdict: ReservationNoOperatorApproval,
			Have:    0, Need: 1, Unit: "approval", Shortfall: 1,
			Why: "no independently confirmed no-eviction/no-preemption approval exists for these " +
				"exact ranks and window; a reservation is never assumed from free capacity alone",
			NextOperator: "approve (or refuse) an explicit node set and reservation window on the " +
				"private control channel, naming the exact ranks and the no-eviction guarantee",
		})
	}
	if s := shortfall(e.UsableHBMBytes, RequiredUsableHBMBytes); s > 0 {
		r.Missing = append(r.Missing, MissingResource{
			Verdict: ReservationInsufficientHBM,
			Have:    e.UsableHBMBytes, Need: RequiredUsableHBMBytes, Unit: "bytes", Shortfall: s,
			Why: "usable accelerator memory reachable without evicting a peer is below the " +
				"artifact plus runtime headroom; the artifact has no documented CPU/NVMe offload " +
				"recipe, so it must fit device memory or it does not run at all",
			NextOperator: "approve additional non-evicting ranks (draining a peer workload is an " +
				"operator decision, never an agent's), raise cloud accelerator quota to a set that " +
				"clears the bar, or approve a smaller quantized artifact",
		})
	}
	if s := shortfall(e.StagingStorageBytes, RequiredStagingStorageBytes); s > 0 {
		r.Missing = append(r.Missing, MissingResource{
			Verdict: ReservationInsufficientStorage,
			Have:    e.StagingStorageBytes, Need: RequiredStagingStorageBytes, Unit: "bytes", Shortfall: s,
			Why:     "staging storage reachable from the reserved set cannot hold the artifact",
			NextOperator: "point staging at a filesystem with the required free space, or approve " +
				"a streaming placement that never materializes the whole artifact",
		})
	}
	if s := shortfall(e.WindowHours, RequiredWindowHours); s > 0 {
		r.Missing = append(r.Missing, MissingResource{
			Verdict: ReservationWindowTooShort,
			Have:    e.WindowHours, Need: RequiredWindowHours, Unit: "hours", Shortfall: s,
			Why: "the reserved validation window is shorter than the run needs; a peer could " +
				"reclaim the ranks mid-transfer and strand a partial artifact",
			NextOperator: "extend the reserved window to at least the required hours, or split the " +
				"validation into checkpointed segments that each fit the shorter window",
		})
	}
	if !e.CollectiveWitnessed || e.CollectiveRanks < RequiredCollectiveRanks {
		r.Missing = append(r.Missing, MissingResource{
			Verdict: ReservationCollectiveUnwitnessed,
			Have:    e.CollectiveRanks, Need: RequiredCollectiveRanks, Unit: "ranks",
			Shortfall: shortfall(e.CollectiveRanks, RequiredCollectiveRanks),
			Why: "no NCCL collective has been proven across the exact reserved ranks; " +
				"cross-node reachability and the runtime topology stay unverified until it passes",
			NextOperator: "run a small collective smoke test across the reserved ranks and record " +
				"its scrubbed result before any artifact transfer begins",
		})
	}

	if len(r.Missing) == 0 {
		r.Verdict, r.Admissible = ReservationAdmitted, true
		return r
	}
	r.Verdict = r.Missing[0].Verdict
	return r
}

// witnessedReservationEnvelope is the candidate reservation as directly witnessed for
// #4801, transcribed from the issue's "Directly witnessed envelope" and the node
// read-backs recorded on the thread, scrubbed to a node-class label.
//
// The figures are the NON-EVICTING reach: the lab's physical aggregate across all three
// GPU nodes is larger, but the other two nodes' ranks each carry a peer allocation and
// #4801 forbids evicting them. The gap between "physically present" and "reservable
// without disturbing a peer" is exactly what this envelope records.
func witnessedReservationEnvelope() ReservationEnvelope {
	return ReservationEnvelope{
		Name:                "lab-nonevicting-free-ranks",
		UsableHBMBytes:      480 * reservationGiB,
		StagingStorageBytes: 792 * reservationGiB,
		WindowHours:         0,
		CollectiveRanks:     0,
		OperatorApproved:    false,
		CollectiveWitnessed: false,
		Note: "six free A100-80GB ranks on one node; the remaining ranks there and every rank " +
			"on the other two GPU nodes carry peer allocations that may not be evicted. Two " +
			"private-channel reservation requests are unanswered, so no window is reserved.",
	}
}

// WitnessedPlacement is the current typed placement verdict for #4801 — the artifact
// #4788 gates its transfer on and #4781 waits behind. It refuses today, naming every
// missing resource and the operator action that would close it.
func WitnessedPlacement() ReservationResult { return AdmitPlacement(witnessedReservationEnvelope()) }

// TransferAdmissible reports whether #4788 may begin the artifact transfer. While it is
// false there is no approved placement, and any V4 Pro runtime number would be unbacked.
func TransferAdmissible() bool { return WitnessedPlacement().Admissible }
