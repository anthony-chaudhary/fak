package modelaccept

import (
	"sort"
	"strings"
	"time"
)

const InventorySchema = "fak.modelaccept.inventory/1"

type InventoryOptions struct {
	Artifact         string
	ArtifactRevision string
	ExpectedCorpusID string
	AsOf             time.Time
	MaxEvidenceAge   time.Duration
}

type InventoryRow struct {
	Model            string   `json:"model"`
	Family           string   `json:"family"`
	Generation       string   `json:"generation"`
	Lifecycle        string   `json:"lifecycle"`
	EvalReason       string   `json:"eval_reason,omitempty"`
	CapabilityGate   Verdict  `json:"capability_gate"`
	RequestedTier    int      `json:"requested_tier"`
	WitnessedTier    *int     `json:"witnessed_tier,omitempty"`
	CorpusID         string   `json:"corpus_id"`
	DeclaredAt       string   `json:"declared_at"`
	ObservedFirst    string   `json:"observed_first,omitempty"`
	ObservedLast     string   `json:"observed_last,omitempty"`
	Samples          int      `json:"samples"`
	Artifact         string   `json:"artifact"`
	ArtifactRevision string   `json:"artifact_revision"`
	Reasons          []string `json:"reasons,omitempty"`
}

type Inventory struct {
	Schema   string         `json:"schema"`
	Verdict  Verdict        `json:"verdict"`
	CorpusID string         `json:"corpus_id"`
	Rows     []InventoryRow `json:"rows"`
	Reasons  []string       `json:"reasons,omitempty"`
}

// BuildInventory joins an acceptance fold to exact model IDs. It never borrows
// evidence across IDs: only a PASS decision for the same exact model can relax
// that row from HOLD.
func BuildInventory(in Input, opts InventoryOptions) Inventory {
	out := Inventory{Schema: InventorySchema, Verdict: Pass, CorpusID: in.Corpus.ID, Rows: []InventoryRow{}}
	decision := Evaluate(in)
	global := append([]string(nil), decision.Reasons...)
	if strings.TrimSpace(opts.Artifact) == "" {
		global = append(global, "acceptance artifact path is missing")
	}
	if strings.TrimSpace(opts.ArtifactRevision) == "" {
		global = append(global, "acceptance artifact revision is missing")
	}
	if want := strings.TrimSpace(opts.ExpectedCorpusID); want != "" && want != in.Corpus.ID {
		global = append(global, "acceptance corpus ID does not match inventory expectation")
	}
	asOf := opts.AsOf
	if asOf.IsZero() {
		asOf = time.Now().UTC()
	}
	maxAge := opts.MaxEvidenceAge
	if maxAge <= 0 {
		maxAge = 30 * 24 * time.Hour
	}

	byDecision := make(map[string]ModelDecision, len(decision.Models))
	for _, md := range decision.Models {
		byDecision[md.Model] = md
	}
	models := append([]ModelRequest(nil), in.Models...)
	sort.Slice(models, func(i, j int) bool { return models[i].Model < models[j].Model })
	for _, req := range models {
		row := InventoryRow{Model: req.Model, Family: req.Family, Generation: req.Generation, Lifecycle: req.Lifecycle, CapabilityGate: Hold, RequestedTier: req.RequestedTier, CorpusID: in.Corpus.ID, DeclaredAt: in.Corpus.DeclaredAt, Artifact: opts.Artifact, ArtifactRevision: opts.ArtifactRevision}
		row.Reasons = append(row.Reasons, global...)
		md, found := byDecision[req.Model]
		if !found {
			row.Reasons = append(row.Reasons, "exact model has no acceptance decision")
		} else {
			row.Samples = md.Samples
			row.EvalReason = md.EvalReason
			if md.Verdict == Skip {
				row.CapabilityGate = Skip
			} else if md.Verdict != Pass {
				row.Reasons = append(row.Reasons, md.Reasons...)
			} else {
				tier := md.RequestedTier
				row.WitnessedTier = &tier
			}
		}
		first, last := time.Time{}, time.Time{}
		for _, run := range in.Runs {
			if run.Model != req.Model {
				continue
			}
			observed, err := time.Parse(time.RFC3339, run.ObservedAt)
			if err != nil {
				continue
			}
			if first.IsZero() || observed.Before(first) {
				first = observed
			}
			if last.IsZero() || observed.After(last) {
				last = observed
			}
		}
		if !first.IsZero() {
			row.ObservedFirst = first.Format(time.RFC3339)
		}
		if !last.IsZero() {
			row.ObservedLast = last.Format(time.RFC3339)
		}
		if last.IsZero() && row.CapabilityGate != Skip {
			row.Reasons = append(row.Reasons, "exact model has no dated observations")
		} else if last.After(asOf) {
			row.Reasons = append(row.Reasons, "exact model acceptance evidence postdates inventory as-of")
		} else if asOf.Sub(last) > maxAge {
			row.Reasons = append(row.Reasons, "exact model acceptance evidence is stale")
		}
		if md.Verdict == Skip {
			row.CapabilityGate = Skip
		} else if md.Verdict == Pass && len(row.Reasons) == 0 {
			row.CapabilityGate = Pass
		} else {
			out.Verdict = Hold
		}
		out.Rows = append(out.Rows, row)
	}
	if len(out.Rows) == 0 {
		out.Verdict = Hold
		out.Reasons = append(out.Reasons, "inventory contains no exact model rows")
	}
	return out
}
