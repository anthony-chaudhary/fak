package swebench

import (
	"sort"
	"time"
)

// This file is the PROVENANCE SEAM for reproducing Ornith-9B on SWE-bench
// Verified — the data structure that keeps the vendor's reported score and a
// fak-served score in two physically separate fields so the two can never be
// confused. Ornith ships a headline pass-rate on SWE-bench Verified; that number
// is OBSERVED — relayed from the vendor's own harness, authored by nobody on this
// box. The whole point of this seam is to turn that OBSERVED number into a
// WITNESSED one: a pass/fail fak itself produced by serving the model and running
// the instance through the harness. No such run happens here — this is the typed
// record + the honesty machinery, NOT a benchmark run.
//
// The load-bearing invariant: the witnessed field starts nil and may ONLY be set
// from a real fak-served run. Copying the vendor's per-ticket result into the
// witnessed field would defeat the seam, so the constructor cannot do it — a
// vendor result only ever lands in VendorObservedPass.

// OrnithProvenance is the honesty tag on an Ornith reproduction record's
// strongest available evidence for a single ticket.
const (
	// OrnithObservedOnly: the only result we hold is the vendor's reported
	// pass/fail. fak has NOT served the model on this ticket. This is the
	// state of every freshly-constructed record.
	OrnithObservedOnly = "observed_only"
	// OrnithWitnessed: a fak-served run produced a pass/fail for this ticket.
	// This label is reachable ONLY by recording a real run via SetWitnessed.
	OrnithWitnessed = "witnessed"
)

// OrnithRecord is one SWE-bench Verified ticket in an Ornith-9B reproduction.
//
// It carries two independent results that MUST NOT be conflated:
//
//   - VendorObservedPass — the pass/fail the Ornith vendor reported. OBSERVED:
//     relayed from the vendor's harness, authored by no one on this box.
//   - FakWitnessedPass    — the pass/fail a fak-served run produced. WITNESSED.
//     Nil until a real run sets it. A value here MUST come from fak actually
//     serving the model and running the harness on this ticket — it is NEVER
//     copied from the vendor's number. Copying the observed value in would be a
//     provenance lie; the constructor refuses to do it and only SetWitnessed,
//     fed a real run's outcome, may populate it.
type OrnithRecord struct {
	TicketID           string     `json:"ticket_id"`
	VendorObservedPass bool       `json:"vendor_observed_pass"`         // OBSERVED — vendor's reported result
	FakWitnessedPass   *bool      `json:"fak_witnessed_pass,omitempty"` // WITNESSED — nil until a real fak-served run
	WitnessedAt        *time.Time `json:"witnessed_at,omitempty"`       // when the fak run produced the witnessed result
	RunRef             string     `json:"run_ref,omitempty"`            // optional handle to the producing fak run (e.g. predictions path / run id)
}

// NewOrnithRecord builds a record from the vendor's OBSERVED result only. The
// witnessed field is left nil on purpose: the constructor cannot witness, and a
// vendor result must never masquerade as a fak-served one.
func NewOrnithRecord(ticketID string, vendorObservedPass bool) OrnithRecord {
	return OrnithRecord{
		TicketID:           ticketID,
		VendorObservedPass: vendorObservedPass,
		FakWitnessedPass:   nil,
	}
}

// SetWitnessed records the pass/fail from a REAL fak-served harness run on this
// ticket, flipping the record's provenance to WITNESSED. runRef is an optional
// handle to the producing run (predictions path, run id) so the witness is
// auditable. Callers must only invoke this with an outcome fak actually produced
// — never with the vendor's observed value.
func (r *OrnithRecord) SetWitnessed(pass bool, at time.Time, runRef string) {
	r.FakWitnessedPass = &pass
	r.WitnessedAt = &at
	r.RunRef = runRef
}

// Provenance reports the strongest evidence the record holds: OrnithWitnessed
// once a fak-served run has set the witnessed result, OrnithObservedOnly while
// the vendor's number is all we have.
func (r OrnithRecord) Provenance() string {
	if r.FakWitnessedPass != nil {
		return OrnithWitnessed
	}
	return OrnithObservedOnly
}

// Witnessed reports whether a fak-served run has produced a result for this
// ticket (i.e. Provenance() == OrnithWitnessed).
func (r OrnithRecord) Witnessed() bool {
	return r.FakWitnessedPass != nil
}

// OrnithRepro is a set of reproduction records for an Ornith-9B SWE-bench
// Verified run, plus the labels that keep the two scores honest.
type OrnithRepro struct {
	Model   string          `json:"model"`   // the vendor model under reproduction, e.g. "ornith-9b"
	Dataset string          `json:"dataset"` // e.g. "swebench-verified"
	Records []*OrnithRecord `json:"records"`
}

// NewOrnithRepro builds an empty reproduction set for a model/dataset pair.
func NewOrnithRepro(model, dataset string) *OrnithRepro {
	return &OrnithRepro{Model: model, Dataset: dataset}
}

// Add appends a ticket's vendor-observed result to the reproduction set and
// returns a pointer to the stored record so a later fak run can witness it.
func (o *OrnithRepro) Add(ticketID string, vendorObservedPass bool) *OrnithRecord {
	r := NewOrnithRecord(ticketID, vendorObservedPass)
	o.Records = append(o.Records, &r)
	return &r
}

// OrnithStatus summarizes how much of a reproduction set fak has actually
// witnessed versus how much is still vendor-OBSERVED only — the honest headline
// for the seam.
type OrnithStatus struct {
	Total            int     `json:"total"`              // tickets in the set
	Witnessed        int     `json:"witnessed"`          // tickets a fak run has produced a result for
	ObservedOnly     int     `json:"observed_only"`      // tickets with only the vendor's number
	VendorObservedOK int     `json:"vendor_observed_ok"` // vendor-reported passes (OBSERVED)
	FakWitnessedOK   int     `json:"fak_witnessed_ok"`   // fak-served passes (WITNESSED) — 0 until real runs land
	WitnessedFrac    float64 `json:"witnessed_frac"`     // Witnessed / Total
}

// Status folds the records into the OBSERVED-vs-WITNESSED headline.
func (o *OrnithRepro) Status() OrnithStatus {
	st := OrnithStatus{Total: len(o.Records)}
	for _, r := range o.Records {
		if r == nil {
			st.ObservedOnly++
			continue
		}
		if r.VendorObservedPass {
			st.VendorObservedOK++
		}
		if r.Witnessed() {
			st.Witnessed++
			if *r.FakWitnessedPass {
				st.FakWitnessedOK++
			}
		} else {
			st.ObservedOnly++
		}
	}
	if st.Total > 0 {
		st.WitnessedFrac = float64(st.Witnessed) / float64(st.Total)
	}
	return st
}

// FullyWitnessed reports whether every ticket in the set has a fak-served result
// — i.e. the vendor's OBSERVED score has been fully reproduced as a WITNESSED one.
func (o *OrnithRepro) FullyWitnessed() bool {
	if len(o.Records) == 0 {
		return false
	}
	for _, r := range o.Records {
		if r == nil {
			return false
		}
		if !r.Witnessed() {
			return false
		}
	}
	return true
}

// TicketIDs returns the reproduction set's ticket ids in sorted order (stable
// output for reporting/diffing).
func (o *OrnithRepro) TicketIDs() []string {
	ids := make([]string, len(o.Records))
	for i, r := range o.Records {
		if r == nil {
			continue
		}
		ids[i] = r.TicketID
	}
	sort.Strings(ids)
	return ids
}
