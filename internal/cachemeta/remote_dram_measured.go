package cachemeta

// remote_dram_measured.go — the ingest half of the 2-node RDMA hardware witness (#5066).
//
// #4306 shipped the MODELED advantage (remote_dram.go): the representative profiles say a
// random KV page-in from a neighbor's borrowed DRAM over RDMA beats a local-NVMe random
// read, and a registered lender grows the KV pool by its lent region. #5066 is the run on
// real hardware that turns MODELED into MEASURED — and it cannot run on a fak dev box:
// it needs two RDMA-connected nodes with one-sided READ/WRITE, so it is dispatched to
// sanctioned lab compute (docs/fleet-compute-nodes.md) over the private control channel,
// which returns only SCRUBBED numbers to this repo.
//
// What this file adds is the half that needs no fabric: the typed record of what the lab
// returns, and the reconciliation that decides whether those numbers CONFIRM or REFUTE
// the paging preference the model already encodes. It stays inside the package charter —
// no payload, no bytes, no I/O; a returned measurement is plain data folded into a typed
// verdict, exactly like every other cachemeta record.
//
// Fail-closed is the whole point (Law A2: a modeled advantage is never quietly reported as
// a measured one). Until a COMPLETE measurement is folded in, the verdict is PENDING and
// the provenance stays MODELED — an absent, partial, or unattributed run can never read as
// a confirmed hardware witness, and the borrowed pool-bytes it would have proven stay 0.

// RemoteDRAMWitnessVerdict is the state of the #5066 hardware witness for the peer-DRAM-
// over-RDMA paging rung: not yet run, or run and adjudicated against the model.
type RemoteDRAMWitnessVerdict string

const (
	// RemoteDRAMWitnessPending is the fail-closed default: no complete measurement has
	// been returned from the lab, so the rung's advantage remains MODELED only.
	RemoteDRAMWitnessPending RemoteDRAMWitnessVerdict = "PENDING"
	// RemoteDRAMWitnessConfirms: the measured page-in from borrowed peer DRAM beat the
	// local-NVMe baseline, so the hardware agrees with the paging order the model encodes.
	RemoteDRAMWitnessConfirms RemoteDRAMWitnessVerdict = "CONFIRMS"
	// RemoteDRAMWitnessRefutes: the measured page-in did NOT beat local NVMe. The rung's
	// preference over disk is not supported by this hardware and must be demoted rather
	// than kept on the model's word.
	RemoteDRAMWitnessRefutes RemoteDRAMWitnessVerdict = "REFUTES"
)

// RemoteDRAMPageInMeasurement is the scrubbed result the authorized lab operator returns
// from the 2-node RDMA run: one-sided RDMA READ page-in latency against the lender, the
// local-NVMe random-read baseline on the starved node at the SAME page size, and the size
// of the lent region that was registered. Commit and Source carry the attribution every
// returned hardware result owes (docs/fleet-compute-nodes.md: "identify the tested commit
// or module revision ... and artifact location"); Source is the scrubbed machine class or
// runbook, never a private node identity.
//
// Latencies are whole-transfer page-in costs for PageBytes — the same quantity the modeled
// stageNanos computes — so the two sides are comparable without rescaling.
type RemoteDRAMPageInMeasurement struct {
	// PageBytes is the KV page size both paths were measured at (the representative run
	// uses 256 KiB).
	PageBytes int64
	// RemoteDRAMP50Nanos / RemoteDRAMP99Nanos are the one-sided RDMA READ page-in
	// latencies from the lender's borrowed DRAM. p50 is required; p99 is optional and,
	// when supplied, is held to the same "must beat NVMe" bar so a fabric that wins on
	// median while losing on the tail cannot pass.
	RemoteDRAMP50Nanos int64
	RemoteDRAMP99Nanos int64
	// LocalNVMeP50Nanos / LocalNVMeP99Nanos are the local-NVMe random-read baseline on the
	// starved node at the same page size (the tier the rung displaces).
	LocalNVMeP50Nanos int64
	LocalNVMeP99Nanos int64
	// BorrowedBytesGained is the measured "effective KV-pool size gained": the bytes of
	// the lent region actually registered on the lender. Required — a latency result with
	// no pool number answers only half of what #5066 asks.
	BorrowedBytesGained int64
	// Commit is the tested commit or module revision the run was taken against.
	Commit string
	// Source is the scrubbed origin of the run (machine class / runbook), never a private
	// node identity, host name, or credential.
	Source string
}

// RemoteDRAMPageInWitness is the adjudicated #5066 record: the MODELED advantage the
// placement policy already acts on, the returned measurement, the measured ratios, and the
// verdict that says whether the hardware backs the model.
type RemoteDRAMPageInWitness struct {
	// Verdict and Reason are the adjudication. Reason names what is missing while PENDING,
	// and which comparison decided a CONFIRMS/REFUTES.
	Verdict RemoteDRAMWitnessVerdict
	Reason  string
	// Provenance is MEASURED only once a complete, attributed measurement was folded in;
	// otherwise it stays MODELED so a pending witness can never be read as hardware proof.
	Provenance string
	// Modeled is the deterministic prediction at the measured page size — the same record
	// ModelRemoteDRAMPageInAdvantage returns, recomputed here so the two ratios sit side by
	// side. A divergence between Modeled.SpeedupX and MeasuredSpeedupX is REPORTED, not
	// adjudicated: the paging policy depends only on which tier wins, not by how much.
	Modeled RemoteDRAMPageInAdvantage
	// Measured is the returned record, verbatim.
	Measured RemoteDRAMPageInMeasurement
	// MeasuredSpeedupX is local-NVMe p50 / remote-DRAM p50 — how many times faster the
	// borrowed-RAM page-in measured. > 1 means the rung earned its place above disk.
	MeasuredSpeedupX float64
	// MeasuredTailSpeedupX is the same ratio at p99, or 0 when the run reported no tail.
	MeasuredTailSpeedupX float64
	// BorrowedBytesGained is the WITNESSED pool gain: the measured lent region once the
	// witness is complete, and 0 while PENDING (an unproven borrow gains nothing).
	BorrowedBytesGained int64
}

// ReconcileRemoteDRAMPageIn folds a returned 2-node RDMA measurement against the modeled
// advantage for the same page size and adjudicates the #5066 witness. probe supplies the
// lender registration the modeled side reads, exactly as ModelRemoteDRAMPageInAdvantage
// takes it.
//
// The zero measurement — the state of this repo until the lab run returns — yields a
// PENDING witness whose Reason names the first missing quantity, with provenance MODELED
// and zero pool gain. A complete measurement is MEASURED and confirms only when borrowed
// peer DRAM actually beat local NVMe at p50 and, when reported, at p99.
func ReconcileRemoteDRAMPageIn(m RemoteDRAMPageInMeasurement, probe CapacityProbe) RemoteDRAMPageInWitness {
	w := RemoteDRAMPageInWitness{
		Verdict:    RemoteDRAMWitnessPending,
		Provenance: "MODELED",
		Modeled:    ModelRemoteDRAMPageInAdvantage(m.PageBytes, probe),
		Measured:   m,
	}
	if missing := m.missingQuantity(); missing != "" {
		w.Reason = "hardware witness not returned: " + missing
		return w
	}
	w.Provenance = "MEASURED"
	w.BorrowedBytesGained = m.BorrowedBytesGained
	w.MeasuredSpeedupX = speedupMilliX(m.LocalNVMeP50Nanos, m.RemoteDRAMP50Nanos)
	if m.RemoteDRAMP99Nanos > 0 && m.LocalNVMeP99Nanos > 0 {
		w.MeasuredTailSpeedupX = speedupMilliX(m.LocalNVMeP99Nanos, m.RemoteDRAMP99Nanos)
	}
	switch {
	case w.MeasuredSpeedupX <= 1:
		w.Verdict = RemoteDRAMWitnessRefutes
		w.Reason = "measured page-in from borrowed peer DRAM did not beat local NVMe"
	case w.MeasuredTailSpeedupX != 0 && w.MeasuredTailSpeedupX <= 1:
		w.Verdict = RemoteDRAMWitnessRefutes
		w.Reason = "measured p99 page-in from borrowed peer DRAM did not beat local NVMe"
	default:
		w.Verdict = RemoteDRAMWitnessConfirms
		w.Reason = "measured page-in beats local NVMe, as the paging order models"
	}
	return w
}

// missingQuantity names the first quantity #5066's done-condition requires that this record
// does not carry, or "" when the record is complete. Attribution counts as a quantity: an
// unattributed number is not a witness, so it can never promote provenance to MEASURED.
func (m RemoteDRAMPageInMeasurement) missingQuantity() string {
	switch {
	case m.PageBytes <= 0:
		return "no page size"
	case m.RemoteDRAMP50Nanos <= 0:
		return "no remote-DRAM p50 page-in latency"
	case m.LocalNVMeP50Nanos <= 0:
		return "no local-NVMe p50 read latency"
	case m.BorrowedBytesGained <= 0:
		return "no borrowed KV-pool bytes gained"
	case m.Commit == "":
		return "no tested commit"
	case m.Source == "":
		return "no scrubbed source"
	}
	return ""
}

// speedupMilliX is the page-in ratio shared by the MODELED and MEASURED halves: how many
// times faster the fast path is than the slow one, truncated to milli-x so the two are
// rounded identically, compare directly, and stay stable across runs without pulling in a
// math dependency. A non-positive input yields 0 — no claim — never a divide-by-zero.
func speedupMilliX(slowNanos, fastNanos int64) float64 {
	if slowNanos <= 0 || fastNanos <= 0 {
		return 0
	}
	return float64((slowNanos*1000)/fastNanos) / 1000
}
