package metrics

import (
	"sort"
	"strings"
)

const DefaultBroadcastContaminationThreshold = 2

// BroadcastDirective is an emit-time directive plus the workspace circuits that
// would consume it. NegframeFlagged is supplied by the negframe boundary: this
// detector measures blast radius, rather than attempting a second classification.
type BroadcastDirective struct {
	ID              string
	NegframeFlagged bool
	Consumers       []string
}

// BroadcastContamination is one detection-only blast-radius observation.
type BroadcastContamination struct {
	DirectiveID string   `json:"directive_id"`
	Consumers   []string `json:"consumers"`
	Radius      int      `json:"radius"`
	Threshold   int      `json:"threshold"`
	Flagged     bool     `json:"flagged"`
	High        bool     `json:"high_blast_radius"`
}

// BlastRadius joins a negframe emit-time flag to the distinct downstream
// workspace consumers. Unflagged directives deliberately score zero: this is a
// severity detector for already-detected negatives, not a content classifier.
func BlastRadius(d BroadcastDirective, threshold int) BroadcastContamination {
	if threshold <= 0 {
		threshold = DefaultBroadcastContaminationThreshold
	}
	row := BroadcastContamination{
		DirectiveID: strings.TrimSpace(d.ID),
		Threshold:   threshold,
		Flagged:     d.NegframeFlagged,
	}
	if !d.NegframeFlagged {
		return row
	}
	seen := make(map[string]struct{}, len(d.Consumers))
	for _, consumer := range d.Consumers {
		consumer = strings.TrimSpace(consumer)
		if consumer != "" {
			seen[consumer] = struct{}{}
		}
	}
	row.Consumers = make([]string, 0, len(seen))
	for consumer := range seen {
		row.Consumers = append(row.Consumers, consumer)
	}
	sort.Strings(row.Consumers)
	row.Radius = len(row.Consumers)
	row.High = row.Radius >= threshold
	return row
}

// BroadcastContaminationRecorder is the soak-period metrics series. It records
// observations only; callers decide policy after evidence establishes a floor.
type BroadcastContaminationRecorder struct {
	threshold int
	rows      []BroadcastContamination
}

func NewBroadcastContaminationRecorder(threshold int) *BroadcastContaminationRecorder {
	if threshold <= 0 {
		threshold = DefaultBroadcastContaminationThreshold
	}
	return &BroadcastContaminationRecorder{threshold: threshold}
}

func (r *BroadcastContaminationRecorder) Record(d BroadcastDirective) BroadcastContamination {
	if r == nil {
		return BlastRadius(d, DefaultBroadcastContaminationThreshold)
	}
	row := BlastRadius(d, r.threshold)
	if row.Flagged {
		r.rows = append(r.rows, row)
	}
	return row
}

// Rows returns a copy suitable for a JSON/control-plane view.
func (r *BroadcastContaminationRecorder) Rows() []BroadcastContamination {
	if r == nil {
		return nil
	}
	return append([]BroadcastContamination(nil), r.rows...)
}

func (r *BroadcastContaminationRecorder) Report() BroadcastContaminationReport {
	var report BroadcastContaminationReport
	if r == nil {
		return report
	}
	report.Threshold = r.threshold
	for _, row := range r.rows {
		report.Observed++
		if row.High {
			report.High++
		} else {
			report.Low++
		}
		if row.Radius > report.MaxRadius {
			report.MaxRadius = row.Radius
		}
	}
	return report
}

type BroadcastContaminationReport struct {
	Threshold int    `json:"threshold"`
	Observed  uint64 `json:"observed"`
	Low       uint64 `json:"low_blast_radius"`
	High      uint64 `json:"high_blast_radius"`
	MaxRadius int    `json:"max_radius"`
}

// Prometheus exposes a bounded aggregate view; per-directive IDs remain in Rows
// so user-controlled identifiers do not create unbounded metric labels.
func (r BroadcastContaminationReport) Prometheus() string {
	return "fak_broadcast_contamination_total{severity=\"low\"} " + utoa(r.Low) + "\n" +
		"fak_broadcast_contamination_total{severity=\"high\"} " + utoa(r.High) + "\n" +
		"fak_broadcast_contamination_max_radius " + itoaNonnegative(r.MaxRadius) + "\n" +
		"fak_broadcast_contamination_threshold " + itoaNonnegative(r.Threshold) + "\n"
}

func itoaNonnegative(n int) string {
	if n <= 0 {
		return "0"
	}
	return utoa(uint64(n))
}
