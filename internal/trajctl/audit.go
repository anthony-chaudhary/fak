package trajctl

import "sort"

// AuditSchema is the pinned schema for a provenance-audit report.
const AuditSchema = "fak-trajctl-audit/1"

// EvidenceStatus is the resolved state of one evidence pointer at read time.
type EvidenceStatus string

const (
	// EvidenceVerified means the pointer resolves: the commit exists, the
	// transcript span is present, or the verdict blob hash matches.
	EvidenceVerified EvidenceStatus = "verified"
	// EvidenceDangling means the pointer does not resolve — a stale SHA, a
	// missing span, or a hash mismatch. A row leaning on dangling evidence is
	// stale and must not gate.
	EvidenceDangling EvidenceStatus = "dangling"
	// EvidenceUnknown means no resolver could speak to this evidence kind, so
	// the pointer is neither confirmed nor refuted and does not demote a row.
	EvidenceUnknown EvidenceStatus = "unknown"
)

// EvidenceResolver resolves one evidence pointer to a status. Injecting the
// resolver keeps the audit deterministic and host-independent: the git,
// transcript, and verdict-hash resolvers live at the call site while the fold
// and its tests drive a fixture resolver. A nil resolver treats every pointer
// as unknown, so a missing resolver never silently demotes a row.
type EvidenceResolver func(EvidenceRef) EvidenceStatus

// EvidenceAudit is one resolved evidence pointer.
type EvidenceAudit struct {
	Ref    EvidenceRef    `json:"ref"`
	Status EvidenceStatus `json:"status"`
}

// RowAudit is the audit verdict for one score row in append order.
type RowAudit struct {
	Index       int             `json:"index"`
	ObjectiveID string          `json:"objective_id"`
	Method      string          `json:"method"`
	Value       float64         `json:"value"`
	Witness     WitnessRung     `json:"witness"`
	Stale       bool            `json:"stale"`
	Dangling    int             `json:"dangling"`
	Evidence    []EvidenceAudit `json:"evidence,omitempty"`
}

// AuditReport folds an evidence audit over every score row in a ledger and lists
// the stale rows worst-first.
type AuditReport struct {
	Schema    string     `json:"schema"`
	Scores    int        `json:"scores"`
	StaleRows int        `json:"stale_rows"`
	Dangling  int        `json:"dangling"`
	Stale     []RowAudit `json:"stale,omitempty"`
}

// auditRow re-resolves one score row's evidence. A row is stale iff it carries
// at least one evidence pointer that resolves to dangling; unknown pointers are
// conservative and never make a row stale.
func auditRow(idx int, s ScoreRow, resolve EvidenceResolver) RowAudit {
	ra := RowAudit{
		Index:       idx,
		ObjectiveID: s.ObjectiveID,
		Method:      s.Method,
		Value:       s.Value,
		Witness:     s.Witness,
	}
	for _, ev := range s.Evidence {
		st := EvidenceUnknown
		if resolve != nil {
			st = resolve(ev)
		}
		ra.Evidence = append(ra.Evidence, EvidenceAudit{Ref: ev, Status: st})
		if st == EvidenceDangling {
			ra.Dangling++
		}
	}
	ra.Stale = ra.Dangling > 0
	return ra
}

// Audit walks every score row in the folded ledger, re-resolves each evidence
// pointer through resolve, and reports the rows whose evidence no longer
// verifies, worst-first: higher-value rows first, then the stronger claimed
// witness rung, then more dangling pointers, then append order.
func Audit(rows []Row, resolve EvidenceResolver) AuditReport {
	st := Fold(rows)
	rep := AuditReport{Schema: AuditSchema, Scores: len(st.Scores)}
	for i, s := range st.Scores {
		ra := auditRow(i, s, resolve)
		if !ra.Stale {
			continue
		}
		rep.StaleRows++
		rep.Dangling += ra.Dangling
		rep.Stale = append(rep.Stale, ra)
	}
	sort.SliceStable(rep.Stale, func(i, j int) bool {
		a, b := rep.Stale[i], rep.Stale[j]
		if a.Value != b.Value {
			return a.Value > b.Value
		}
		if wa, wb := witnessStrength(a.Witness), witnessStrength(b.Witness); wa != wb {
			return wa > wb
		}
		if a.Dangling != b.Dangling {
			return a.Dangling > b.Dangling
		}
		return a.Index < b.Index
	})
	return rep
}

// FoldVerified folds the ledger like Fold, but re-verifies every score row's
// evidence through resolve and demotes any row leaning on dangling evidence to
// W0 so a stale pointer can no longer gate automated steering. A clean ledger
// folds identically to Fold.
func FoldVerified(rows []Row, resolve EvidenceResolver) State {
	st := Fold(rows)
	for i := range st.Scores {
		if auditRow(i, st.Scores[i], resolve).Stale {
			st.Scores[i].Witness = W0
		}
	}
	return st
}

// witnessStrength ranks the witness rungs so the audit can order a dangling W3
// (which would gate) ahead of a dangling W0 (which never gates alone).
func witnessStrength(w WitnessRung) int {
	switch w {
	case W3:
		return 3
	case W2:
		return 2
	case W1:
		return 1
	default:
		return 0
	}
}
