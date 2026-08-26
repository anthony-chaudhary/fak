package trajectoryassurance

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

const (
	TrajctlCurveSchema    = "fak-trajctl-curve/1"
	TrajectoryAuditSchema = "fak-trajectory-audit/1"
	DojoIterationSchema   = "fak-dojo-rsi/1"
	EffectReceiptSchema   = "fak.orchestration_effect_receipt.v1"

	ReasonSourceUnavailable    = "ASSURANCE_SOURCE_UNAVAILABLE"
	ReasonSourceStale          = "ASSURANCE_SOURCE_STALE"
	ReasonSchemaUnsupported    = "ASSURANCE_SCHEMA_UNSUPPORTED"
	ReasonIdentityMismatch     = "ASSURANCE_IDENTITY_MISMATCH"
	ReasonAccountingIncomplete = "DOJO_ACCOUNTING_INCOMPLETE"
	ReasonEffectReadbackFailed = "DELEGATION_EFFECT_READBACK_FAILED"
)

// UnavailableInput preserves a declared source as typed UNKNOWN instead of
// turning a temporarily missing receipt into authority-free evidence.
func UnavailableInput(kind, detail string) Input {
	reason := "declared receipt is unavailable"
	if strings.TrimSpace(detail) != "" {
		reason += ": " + detail
	}
	switch kind {
	case "trajctl":
		return Input{ObjectiveProgress: unknownObservation(TrajctlCurveSchema, "curve-receipt", ReasonSourceUnavailable, reason)}
	case "audit":
		return Input{DeterministicFloor: []DeterministicCheck{{Name: "trajectory_audit", Evidence: unknownEvidence(TrajectoryAuditSchema, "audit-jsonl", ReasonSourceUnavailable, reason)}}}
	case "dojo":
		return Input{Efficiency: EfficiencyInput{Evidence: unknownEvidence(DojoIterationSchema, "iteration-receipt", ReasonSourceUnavailable, reason)}}
	case "effects":
		return Input{DelegationIntegrity: unknownObservation(EffectReceiptSchema, "effect-readback", ReasonSourceUnavailable, reason)}
	case "ultracode":
		return Input{DelegationIntegrity: unknownObservation(UltracodeStatusSchema, "status-receipt", ReasonSourceUnavailable, reason)}
	default:
		return Input{}
	}
}

type trajctlCurveReceipt struct {
	Schema     string `json:"schema"`
	Objectives []struct {
		ObjectiveID string  `json:"objective_id"`
		Signal      string  `json:"signal"`
		Latest      float64 `json:"latest"`
		Delta       float64 `json:"delta"`
		Detail      string  `json:"detail"`
		Methods     []struct {
			Points []struct {
				UnixMillis int64  `json:"unix_millis"`
				SessionID  string `json:"session_id"`
				RunID      string `json:"run_id"`
			} `json:"points"`
		} `json:"methods"`
	} `json:"objectives"`
}

// DecodeTrajctlCurve adapts the pinned, payload-free progress receipt. A curve
// is fresh relative to its newest score, never relative to file metadata.
func DecodeTrajctlCurve(r io.Reader, now time.Time, maxAge time.Duration) (Input, error) {
	var receipt trajctlCurveReceipt
	if err := decodeOne(r, &receipt); err != nil {
		return Input{}, fmt.Errorf("trajectory assurance: decode trajctl curve: %w", err)
	}
	unknown := func(token, reason string) Input {
		return Input{ObjectiveProgress: unknownObservation(TrajctlCurveSchema, "trajctl-curve", token, reason)}
	}
	if receipt.Schema != TrajctlCurveSchema {
		return unknown(ReasonSchemaUnsupported, "unsupported trajctl curve schema"), nil
	}
	if len(receipt.Objectives) != 1 || strings.TrimSpace(receipt.Objectives[0].ObjectiveID) == "" {
		return unknown(ReasonIdentityMismatch, "trajctl receipt must contain exactly one identified objective"), nil
	}
	objective := receipt.Objectives[0]
	var newest int64
	var sessionID, runID string
	for _, method := range objective.Methods {
		for _, point := range method.Points {
			if point.UnixMillis >= newest {
				newest, sessionID, runID = point.UnixMillis, point.SessionID, point.RunID
			}
		}
	}
	freshness := freshnessWindow(newest, now, maxAge)
	state, token, reason := Pass, "OBJECTIVE_PROGRESS_WITNESSED", "trajctl reports witnessed objective progress"
	switch objective.Signal {
	case "STALL", "DRIFT", "DETOUR_OVERRUN":
		state, token, reason = Warn, "OBJECTIVE_PROGRESS_AT_RISK", "trajctl reports "+objective.Signal
	case "HEALTHY":
	default:
		state, token, reason = Unknown, ReasonSchemaUnsupported, "unsupported trajctl signal"
	}
	if freshness == "stale" {
		state, token, reason = Unknown, ReasonSourceStale, "trajctl progress receipt is stale"
	}
	return Input{ObjectiveID: objective.ObjectiveID, TrajectoryID: firstNonEmpty(runID, sessionID), SessionID: sessionID, RunID: runID, ObservationWindow: freshness, ObjectiveProgress: Observation{State: state, Evidence: Evidence{Source: TrajctlCurveSchema, Provenance: "curve-receipt", Authority: "trajctl", Freshness: freshness, Reason: reason, ReasonToken: token}}}, nil
}

type auditRow struct {
	Schema            string `json:"schema"`
	Kind              string `json:"kind"`
	Source            string `json:"source"`
	SessionID         string `json:"session_id"`
	UsageRecords      int    `json:"usage_records"`
	UsageRecordsExact int    `json:"usage_records_exact"`
	RefusedRecords    int    `json:"refused_records"`
}

// DecodeTrajectoryAudit adapts fak-trajectory-audit/1 JSONL without reading a transcript.
func DecodeTrajectoryAudit(r io.Reader, trajectoryID string) (Input, error) {
	scan := bufio.NewScanner(r)
	found := false
	refused := 0
	exact := 0
	for scan.Scan() {
		var row auditRow
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			return Input{}, fmt.Errorf("trajectory assurance: decode trajectory audit: %w", err)
		}
		if row.Schema != TrajectoryAuditSchema {
			return Input{DeterministicFloor: []DeterministicCheck{{Name: "trajectory_audit", Passed: nil, Evidence: unknownEvidence(TrajectoryAuditSchema, "audit-jsonl", ReasonSchemaUnsupported, "unsupported trajectory audit schema")}}}, nil
		}
		if row.Kind == "refusal" {
			refused++
		}
		if row.Kind == "session" && (trajectoryID == "" || row.SessionID == trajectoryID) {
			found = true
			exact += row.UsageRecords
		}
		if row.Kind == "summary" {
			refused += row.RefusedRecords
		}
	}
	if err := scan.Err(); err != nil {
		return Input{}, err
	}
	state, token, reason := Pass, "TRAJECTORY_AUDIT_EXACT", "trajectory audit diagnostics are exact"
	if refused > 0 {
		state, token, reason = Unknown, "TRAJECTORY_AUDIT_REFUSED", "trajectory audit contains refused records"
	} else if !found || exact == 0 {
		state, token, reason = Unknown, ReasonSourceUnavailable, "trajectory audit has no exact matching session"
	}
	return Input{TrajectoryID: trajectoryID, DeterministicFloor: []DeterministicCheck{{Name: "trajectory_audit", Passed: checkPassed(state), Evidence: Evidence{Source: TrajectoryAuditSchema, Provenance: "audit-jsonl", Authority: "trajectory-audit", Freshness: "declared-window", Reason: reason, ReasonToken: token}}}}, nil
}

type dojoReceipt struct {
	Schema  string `json:"schema"`
	Kept    bool   `json:"kept"`
	Reason  string `json:"reason"`
	Witness struct {
		OK                   bool    `json:"ok"`
		Outcome              *bool   `json:"outcome"`
		ConstraintsSatisfied *bool   `json:"constraints_satisfied"`
		ParentUnits          *int64  `json:"parent_units"`
		ChildUnits           []int64 `json:"child_units"`
		AccountingComplete   bool    `json:"accounting_complete"`
	} `json:"witness"`
}

func DecodeDojoIteration(r io.Reader) (Input, error) {
	var d dojoReceipt
	if err := decodeOne(r, &d); err != nil {
		return Input{}, fmt.Errorf("trajectory assurance: decode dojo receipt: %w", err)
	}
	e := Evidence{Source: DojoIterationSchema, Provenance: "iteration-receipt", Authority: "dojo-rsi", Freshness: "declared-window", Reason: d.Reason}
	if d.Schema != DojoIterationSchema {
		e.ReasonToken, e.Reason = ReasonSchemaUnsupported, "unsupported dojo receipt schema"
		return Input{Efficiency: EfficiencyInput{Evidence: e}}, nil
	}
	outcome := d.Witness.Outcome
	if outcome == nil {
		v := d.Kept
		outcome = &v
	}
	constraints := d.Witness.ConstraintsSatisfied
	if constraints == nil && d.Witness.OK {
		v := true
		constraints = &v
	}
	complete := d.Witness.AccountingComplete && d.Witness.ParentUnits != nil
	e.ReasonToken = "DOJO_EFFICIENCY_CALIBRATED"
	if !complete {
		e.ReasonToken, e.Reason = ReasonAccountingIncomplete, "dojo receipt lacks complete parent+child accounting"
	}
	return Input{Efficiency: EfficiencyInput{Outcome: outcome, ConstraintsSatisfied: constraints, ParentUnits: d.Witness.ParentUnits, ChildUnits: d.Witness.ChildUnits, AccountingComplete: complete, Evidence: e}}, nil
}

type effectReceipt struct {
	Schema         string `json:"schema"`
	RunID          string `json:"run_id"`
	ChildID        string `json:"child_id"`
	State          string `json:"state"`
	Reconciliation string `json:"reconciliation"`
	ObservedAt     string `json:"observed_at"`
	Witness        struct {
		AuthorityID   string `json:"authority_id"`
		AuthorChildID string `json:"author_child_id"`
	} `json:"witness"`
}

func DecodeEffectReceipts(r io.Reader, now time.Time, maxAge time.Duration) (Input, error) {
	dec := json.NewDecoder(r)
	count := 0
	runID := ""
	state := Pass
	token, reason := "DELEGATION_RECONCILED", "independently witnessed effects reconciled"
	for {
		var e effectReceipt
		err := dec.Decode(&e)
		if err == io.EOF {
			break
		}
		if err != nil {
			return Input{}, fmt.Errorf("trajectory assurance: decode effect receipt: %w", err)
		}
		count++
		if e.Schema != EffectReceiptSchema {
			state, token, reason = Unknown, ReasonSchemaUnsupported, "unsupported effect receipt schema"
			continue
		}
		if runID == "" {
			runID = e.RunID
		} else if runID != e.RunID {
			state, token, reason = Unknown, ReasonIdentityMismatch, "effect receipts disagree on run identity"
		}
		observed, err := time.Parse(time.RFC3339Nano, e.ObservedAt)
		if err != nil || (maxAge > 0 && now.Sub(observed) > maxAge) {
			state, token, reason = Unknown, ReasonSourceStale, "effect receipt is stale or has invalid time"
		}
		if e.State != "VERIFIED" || e.Reconciliation != "RECONCILED" || e.Witness.AuthorityID == "" || e.Witness.AuthorChildID == e.ChildID {
			state, token, reason = Fail, ReasonEffectReadbackFailed, "effect readback or independent witness failed"
		}
	}
	if count == 0 {
		state, token, reason = Unknown, ReasonSourceUnavailable, "no effect receipts were declared"
	}
	return Input{RunID: runID, TrajectoryID: runID, DelegationIntegrity: Observation{State: state, Evidence: Evidence{Source: EffectReceiptSchema, Provenance: "effect-readback", Authority: "independent-observer", Freshness: "declared-window", Reason: reason, ReasonToken: token}}}, nil
}

func decodeOne(r io.Reader, v any) error {
	dec := json.NewDecoder(r)
	if err := dec.Decode(v); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}
func freshnessWindow(ms int64, now time.Time, maxAge time.Duration) string {
	if ms <= 0 {
		return "unknown"
	}
	if maxAge > 0 && now.Sub(time.UnixMilli(ms)) > maxAge {
		return "stale"
	}
	return "current"
}
func unknownEvidence(source, provenance, token, reason string) Evidence {
	return Evidence{Source: source, Provenance: provenance, Authority: "declared-receipt", Freshness: "unknown", Reason: reason, ReasonToken: token}
}
func unknownObservation(source, provenance, token, reason string) Observation {
	return Observation{State: Unknown, Evidence: unknownEvidence(source, provenance, token, reason)}
}
func checkPassed(state State) *bool {
	if state == Unknown {
		return nil
	}
	v := state == Pass
	return &v
}
func firstNonEmpty(v ...string) string {
	for _, s := range v {
		if s != "" {
			return s
		}
	}
	return ""
}
