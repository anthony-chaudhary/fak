package ultracoderesume

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/workflow"
)

type Disposition string

const (
	DispositionSkip   Disposition = "skip"
	DispositionRerun  Disposition = "rerun"
	DispositionRefuse Disposition = "refuse"
)

const (
	ReasonMissingReceipt    = "missing_receipt"
	ReasonIncompleteAttempt = "incomplete_attempt"
	ReasonFailedAttempt     = "failed_attempt"
	ReasonInvalidReceipt    = "invalid_receipt"
	ReasonDependencyInvalid = "dependency_invalidated"
	ReasonDependencyRefused = "dependency_refused"
	ReasonWitnessMissing    = "witness_missing"
	ReasonWitnessUnverified = "witness_unverified"
	ReasonAmbiguousEffect   = "ambiguous_effect"
	ReasonLeaseUnavailable  = "lease_unavailable"
	ReasonLeaseNotFresh     = "lease_not_fresh"
	ReasonCapabilityExpired = "capability_expired"
	ReasonRunnerUnavailable = "runner_unavailable"
	ReasonRunnerFailed      = "runner_failed"
	ReasonPersistenceFailed = "persistence_failed"
)

// Corroborate resolves an opaque witness key against evidence the controller did
// not author. A nil resolver never confirms a completion.
type Corroborate func(ctx context.Context, nodeID, witnessKey string) (source string, ok bool)

// Acquire obtains fresh authority for one effectful rerun. The returned lease
// must differ from the previous attempt's lease and its capability must be live.
type Acquire func(ctx context.Context, node GraphNode, previousLeaseID string) (Admission, error)

// Runner performs only the node work selected for rerun. The receipt it returns
// is not accepted as completion until Corroborate confirms its witness key.
type Runner func(ctx context.Context, node GraphNode, admission Admission) (ExecutionReceipt, error)

type Options struct {
	NowMS       int64
	Corroborate Corroborate
	Acquire     Acquire
	Runner      Runner
}

// NodeReport is the deterministic, redacted resume verdict for one graph node.
type NodeReport struct {
	NodeID                 string      `json:"node_id"`
	NodeIdentity           string      `json:"node_identity"`
	Kind                   NodeKind    `json:"kind"`
	Disposition            Disposition `json:"disposition"`
	Reason                 string      `json:"reason,omitempty"`
	Status                 string      `json:"status,omitempty"`
	PreviousLeaseID        string      `json:"previous_lease_id,omitempty"`
	CurrentLeaseID         string      `json:"current_lease_id,omitempty"`
	PriorEffectReceiptID   string      `json:"prior_effect_receipt_id,omitempty"`
	CurrentEffectReceiptID string      `json:"current_effect_receipt_id,omitempty"`
	PriorReceiptKey        string      `json:"prior_receipt_key,omitempty"`
	PriorWitnessKey        string      `json:"prior_witness_key,omitempty"`
	ReceiptKey             string      `json:"receipt_key,omitempty"`
	WitnessKey             string      `json:"witness_key,omitempty"`
	WitnessSource          string      `json:"witness_source,omitempty"`
	Attempt                int         `json:"attempt,omitempty"`
}

type Report struct {
	Schema     string       `json:"schema"`
	GraphID    string       `json:"graph_id,omitempty"`
	GraphEpoch string       `json:"graph_epoch,omitempty"`
	Nodes      []NodeReport `json:"nodes,omitempty"`
	Skipped    int          `json:"skipped"`
	Rerun      int          `json:"rerun"`
	Refused    int          `json:"refused"`
	Refusal    *Refusal     `json:"refusal,omitempty"`
}

// StableJSON renders a report without maps or wall-clock data, so identical
// receipts and evidence produce byte-identical machine output.
func StableJSON(report Report) ([]byte, error) {
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

type foldedAttempts struct {
	Latest    map[string]AttemptReceipt
	Invalid   map[string]AttemptReceipt
	Max       map[string]int
	Completed map[string][]AttemptReceipt
}

type reconstruction struct {
	graph   GraphReceipt
	flow    *workflow.Graph
	folded  foldedAttempts
	report  Report
	outputs map[string]string
}

// Inspect reconstructs resume decisions without acquiring leases, executing
// nodes, or appending receipts.
func Inspect(ctx context.Context, store Store, plan orchestration.WorkflowPlan, corroborate Corroborate) (Report, error) {
	reconstruction, err := reconstruct(ctx, store, plan, corroborate)
	if err != nil {
		return reconstruction.report, err
	}
	return reconstruction.report, nil
}

// Resume reconstructs the graph, reruns only admitted nodes, and persists a
// write-ahead attempt row before each runner call plus a witnessed completion
// row and generic workflow entry afterward.
func Resume(ctx context.Context, store Store, plan orchestration.WorkflowPlan, options Options) (Report, error) {
	rec, err := reconstruct(ctx, store, plan, options.Corroborate)
	if err != nil {
		return rec.report, err
	}
	nowMS := options.NowMS
	if nowMS == 0 {
		nowMS = time.Now().UnixMilli()
	}

	indexByID := make(map[string]int, len(rec.graph.Nodes))
	for i, node := range rec.graph.Nodes {
		indexByID[node.ID] = i
	}
	for i, node := range rec.graph.Nodes {
		row := &rec.report.Nodes[i]
		if row.Disposition == DispositionSkip {
			rec.outputs[node.ID] = row.ReceiptKey
			continue
		}
		if row.Disposition == DispositionRefuse {
			continue
		}
		dependencyFailed := false
		for _, dependency := range node.Needs {
			upstream := rec.report.Nodes[indexByID[dependency]]
			if upstream.Disposition == DispositionRefuse || upstream.Status == "failed" || upstream.Status == string(AttemptEffectUnknown) {
				dependencyFailed = true
				break
			}
		}
		if dependencyFailed {
			row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonDependencyRefused, "refused"
			continue
		}
		if options.Runner == nil {
			row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonRunnerUnavailable, "refused"
			continue
		}

		admission := Admission{}
		if node.Kind == NodeEffect {
			if options.Acquire == nil {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonLeaseUnavailable, "refused"
				continue
			}
			admission, err = options.Acquire(ctx, node, row.PreviousLeaseID)
			if err != nil || !opaqueKey(admission.LeaseID) || !opaqueKey(admission.EffectReceiptID) || !opaqueKey(admission.CapabilityID) {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonLeaseUnavailable, "refused"
				continue
			}
			if admission.LeaseID == row.PreviousLeaseID {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonLeaseNotFresh, "refused"
				continue
			}
			if admission.EffectReceiptID == row.PriorEffectReceiptID {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonLeaseNotFresh, "refused"
				continue
			}
			if admission.CapabilityExpiresMS <= nowMS {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonCapabilityExpired, "refused"
				continue
			}
			row.CurrentLeaseID = admission.LeaseID
			row.CurrentEffectReceiptID = admission.EffectReceiptID
		}

		attempt := rec.folded.Max[node.ID] + 1
		_, makeErr := store.RecordAttempt(rec.graph, node.ID, AttemptRecord{
			Attempt: attempt, Status: AttemptStarted, Admission: admission, TSMS: nowMS,
		}, nil)
		if makeErr != nil {
			row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonPersistenceFailed, "refused"
			continue
		}
		row.Attempt = attempt
		result, runErr := options.Runner(ctx, node, admission)
		if runErr != nil {
			_, appendErr := store.RecordAttempt(rec.graph, node.ID, AttemptRecord{
				Attempt: attempt, Status: AttemptFailed, Admission: admission, TSMS: nowMS,
			}, nil)
			if appendErr != nil {
				return failReport(rec.report, RefusalPersistence, "persist failed attempt")
			}
			row.Reason, row.Status = ReasonRunnerFailed, "failed"
			continue
		}

		source, witnessed := corroborate(options.Corroborate, ctx, node.ID, result.WitnessKey)
		if !opaqueKey(result.ReceiptKey) || !opaqueKey(result.WitnessKey) || !witnessed {
			status := AttemptFailed
			if node.Kind == NodeEffect {
				status = AttemptEffectUnknown
				if !opaqueKey(result.ReceiptKey) {
					result.ReceiptKey = digestValue("untrusted-receipt", result.ReceiptKey)
				}
				if !opaqueKey(result.WitnessKey) {
					result.WitnessKey = ""
				}
			}
			terminalRecord := AttemptRecord{
				Attempt: attempt, Status: status, Admission: admission, Result: result, TSMS: nowMS,
			}
			_, terminalErr := NewAttemptReceipt(rec.graph, node.ID, terminalRecord)
			if terminalErr != nil {
				terminalRecord = AttemptRecord{
					Attempt: attempt, Status: AttemptFailed, Admission: admission, TSMS: nowMS,
				}
			}
			if _, appendErr := store.RecordAttempt(rec.graph, node.ID, terminalRecord, nil); appendErr != nil {
				return failReport(rec.report, RefusalPersistence, "persist unwitnessed attempt")
			}
			row.Disposition, row.Reason = DispositionRefuse, ReasonWitnessUnverified
			if node.Kind == NodeEffect {
				row.Reason = ReasonAmbiguousEffect
			}
			row.Status = string(status)
			row.ReceiptKey, row.WitnessKey = result.ReceiptKey, result.WitnessKey
			continue
		}

		completedRecord := AttemptRecord{
			Attempt: attempt, Status: AttemptCompleted, Admission: admission, Result: result, TSMS: nowMS,
		}
		dependencyReceipts := make(map[string]string, len(node.Needs))
		for _, dependency := range node.Needs {
			dependencyReceipts[dependency] = rec.outputs[dependency]
		}
		if _, err := store.RecordAttempt(rec.graph, node.ID, completedRecord, dependencyReceipts); err != nil {
			// The completed attempt remains sufficient for deterministic recovery;
			// surface the failed mirror write without undoing its witnessed receipt.
			return failReport(rec.report, RefusalPersistence, "persist workflow journal mirror")
		}
		rec.outputs[node.ID] = result.ReceiptKey
		row.Status = "completed"
		row.ReceiptKey, row.WitnessKey, row.WitnessSource = result.ReceiptKey, result.WitnessKey, source
	}
	recount(&rec.report)
	return rec.report, nil
}

func reconstruct(ctx context.Context, store Store, plan orchestration.WorkflowPlan, corr Corroborate) (reconstruction, error) {
	rec := reconstruction{report: Report{Schema: ResumeReportSchema}}
	current, err := ReceiptForPlan(plan)
	if err != nil {
		return rec, attachRefusal(&rec.report, err)
	}
	stored, err := store.ReadGraph()
	if err != nil {
		return rec, attachRefusal(&rec.report, err)
	}
	rec.graph, rec.flow = stored, stored.workflowGraph()
	rec.report.GraphID, rec.report.GraphEpoch = stored.GraphID, stored.GraphEpoch
	if current.GraphID != stored.GraphID {
		err := refuse(RefusalGraphIdentity, "stored %s current %s", stored.GraphID, current.GraphID)
		return rec, attachRefusal(&rec.report, err)
	}
	if current.GraphEpoch != stored.GraphEpoch {
		err := refuse(RefusalGraphEpoch, "stored %s current %s", stored.GraphEpoch, current.GraphEpoch)
		return rec, attachRefusal(&rec.report, err)
	}
	attempts, err := store.ReadAttempts()
	if err != nil {
		return rec, attachRefusal(&rec.report, err)
	}
	rec.folded, err = foldAttemptReceipts(stored, attempts)
	if err != nil {
		return rec, attachRefusal(&rec.report, err)
	}
	journal, err := store.ReadWorkflowJournal()
	if err != nil {
		return rec, attachRefusal(&rec.report, err)
	}
	if err := validateWorkflowJournal(stored, rec.folded, journal); err != nil {
		return rec, attachRefusal(&rec.report, err)
	}

	synthetic, completedOutputs := syntheticWorkflowRows(stored, rec.folded)
	state, err := workflow.Fold(synthetic, 0)
	if err != nil {
		err := refuse(RefusalWorkflowCorrupt, "reconstructed workflow state is invalid")
		return rec, attachRefusal(&rec.report, err)
	}
	resumption := workflow.Resume(ctx, rec.flow, state, stored.GraphEpoch, func(ctx context.Context, nodeID, witnessKey string) (string, bool) {
		return corroborate(corr, ctx, nodeID, witnessKey)
	})

	rec.outputs = make(map[string]string, len(stored.Nodes))
	byID := make(map[string]workflow.StepVerdict, len(resumption.Steps))
	for _, verdict := range resumption.Steps {
		byID[verdict.Step] = verdict
	}
	reports := make(map[string]NodeReport, len(stored.Nodes))
	for _, node := range stored.Nodes {
		latest, hasLatest := rec.folded.Latest[node.ID]
		invalid, hasInvalid := rec.folded.Invalid[node.ID]
		prior := latest
		if hasInvalid && (!hasLatest || invalid.Attempt >= latest.Attempt) {
			prior = invalid
		}
		row := NodeReport{
			NodeID: node.ID, NodeIdentity: node.Identity, Kind: node.Kind,
			PreviousLeaseID: prior.LeaseID, PriorEffectReceiptID: prior.EffectReceiptID,
			PriorReceiptKey: prior.ReceiptKey, PriorWitnessKey: prior.WitnessKey,
		}
		verdict := byID[node.ID]
		if verdict.Disposition == workflow.DispSkip {
			row.Disposition, row.Status = DispositionSkip, "completed"
			row.ReceiptKey, row.WitnessKey, row.WitnessSource = latest.ReceiptKey, latest.WitnessKey, verdict.Source
			rec.outputs[node.ID] = latest.ReceiptKey
		} else {
			row.Disposition, row.Reason = DispositionRerun, resumeReason(verdict.Reason, latest, hasLatest, hasInvalid)
			if hasInvalid && node.Kind == NodeEffect && (invalid.Status == AttemptCompleted || invalid.Status == AttemptEffectUnknown) {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonAmbiguousEffect, "refused"
			}
			if hasLatest && node.Kind == NodeEffect && (latest.Status == AttemptEffectUnknown ||
				(latest.Status == AttemptCompleted && (verdict.Reason == workflow.ReasonClaimMissing || verdict.Reason == workflow.ReasonClaimUnverified))) {
				row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonAmbiguousEffect, "refused"
			}
			for _, dependency := range node.Needs {
				if reports[dependency].Disposition == DispositionRefuse {
					row.Disposition, row.Reason, row.Status = DispositionRefuse, ReasonDependencyRefused, "refused"
					break
				}
			}
		}
		if completedOutputs[node.ID] != "" && row.PriorReceiptKey == "" {
			row.PriorReceiptKey = completedOutputs[node.ID]
		}
		reports[node.ID] = row
		rec.report.Nodes = append(rec.report.Nodes, row)
	}
	recount(&rec.report)
	return rec, nil
}

func foldAttemptReceipts(graph GraphReceipt, rows []AttemptReceipt) (foldedAttempts, error) {
	folded := foldedAttempts{
		Latest: make(map[string]AttemptReceipt), Invalid: make(map[string]AttemptReceipt),
		Max: make(map[string]int), Completed: make(map[string][]AttemptReceipt),
	}
	for _, row := range rows {
		if row.Schema != AttemptReceiptSchema {
			return folded, refuse(RefusalAttemptSchema, "got %q want %q", row.Schema, AttemptReceiptSchema)
		}
		node, ok := graph.node(row.NodeID)
		if !ok || row.GraphID != graph.GraphID || row.GraphEpoch != graph.GraphEpoch || row.NodeIdentity != node.Identity {
			return folded, refuse(RefusalAttemptIdentity, "attempt identity is incompatible with graph")
		}
		if row.Attempt > folded.Max[row.NodeID] {
			folded.Max[row.NodeID] = row.Attempt
		}
		if err := verifyAttempt(row, node); err != nil {
			markInvalid(&folded, row)
			continue
		}
		previous, exists := folded.Latest[row.NodeID]
		validTransition := !exists && row.Status == AttemptStarted
		if exists && row.Attempt > previous.Attempt && row.Status == AttemptStarted {
			validTransition = true
		}
		if exists && row.Attempt == previous.Attempt && previous.Status == AttemptStarted && row.Status != AttemptStarted &&
			row.LeaseID == previous.LeaseID && row.EffectReceiptID == previous.EffectReceiptID && row.CapabilityID == previous.CapabilityID &&
			row.CapabilityExpiresMS == previous.CapabilityExpiresMS && row.TSMS >= previous.TSMS {
			validTransition = true
		}
		if !validTransition {
			markInvalid(&folded, row)
			continue
		}
		folded.Latest[row.NodeID] = row
		if invalid, ok := folded.Invalid[row.NodeID]; ok && invalid.Attempt <= row.Attempt {
			delete(folded.Invalid, row.NodeID)
		}
		if row.Status == AttemptCompleted {
			folded.Completed[row.NodeID] = append(folded.Completed[row.NodeID], row)
		}
	}
	return folded, nil
}

func markInvalid(folded *foldedAttempts, row AttemptReceipt) {
	previous, ok := folded.Invalid[row.NodeID]
	if !ok || row.Attempt >= previous.Attempt {
		folded.Invalid[row.NodeID] = row
	}
}

func validateWorkflowJournal(graph GraphReceipt, folded foldedAttempts, rows []workflow.Entry) error {
	if _, err := workflow.Fold(rows, 0); err != nil {
		return refuse(RefusalWorkflowCorrupt, "workflow journal rows are invalid")
	}
	for _, row := range rows {
		if row.Schema != workflow.JournalSchema || row.Run != graph.GraphID || row.EpochHash != graph.GraphEpoch || row.Kind != workflow.StepEffectful {
			return refuse(RefusalWorkflowCorrupt, "workflow row identity is incompatible")
		}
		if _, ok := graph.node(row.Step); !ok || row.InputsHash == "" || row.OutputHash != workflow.HashOutput(row.Output) || row.Claim == "" {
			return refuse(RefusalWorkflowCorrupt, "workflow row is malformed")
		}
		matched := false
		for _, completed := range folded.Completed[row.Step] {
			if completed.ReceiptKey == row.Output && completed.WitnessKey == row.Claim {
				matched = true
				break
			}
		}
		if !matched {
			return refuse(RefusalWorkflowCorrupt, "workflow row lacks a matching attempt receipt")
		}
	}
	return nil
}

func syntheticWorkflowRows(graph GraphReceipt, folded foldedAttempts) ([]workflow.Entry, map[string]string) {
	var rows []workflow.Entry
	outputs := make(map[string]string, len(graph.Nodes))
	hashes := make(map[string]string, len(graph.Nodes))
	flow := graph.workflowGraph()
	for i, node := range graph.Nodes {
		latest, ok := folded.Latest[node.ID]
		if invalid, bad := folded.Invalid[node.ID]; bad && (!ok || invalid.Attempt >= latest.Attempt) {
			continue
		}
		if !ok || latest.Status != AttemptCompleted {
			continue
		}
		depHashes := make(map[string]string, len(node.Needs))
		for _, dependency := range node.Needs {
			depHashes[dependency] = hashes[dependency]
		}
		entry := workflow.Entry{
			Run: graph.GraphID, Step: node.ID, Kind: workflow.StepEffectful,
			InputsHash: workflow.StepInputsHash(flow.Nodes[i], depHashes), EpochHash: graph.GraphEpoch,
			OutputHash: workflow.HashOutput(latest.ReceiptKey), Output: latest.ReceiptKey,
			Claim: latest.WitnessKey, TSMS: latest.TSMS,
		}
		rows = append(rows, entry)
		outputs[node.ID], hashes[node.ID] = latest.ReceiptKey, entry.OutputHash
	}
	return rows, outputs
}

func resumeReason(workflowReason string, latest AttemptReceipt, hasLatest, hasInvalid bool) string {
	if hasInvalid {
		return ReasonInvalidReceipt
	}
	if workflowReason == workflow.ReasonUpstreamRerun || workflowReason == workflow.ReasonInputsDrift {
		return ReasonDependencyInvalid
	}
	if workflowReason == workflow.ReasonClaimMissing {
		return ReasonWitnessMissing
	}
	if workflowReason == workflow.ReasonClaimUnverified {
		return ReasonWitnessUnverified
	}
	if workflowReason == workflow.ReasonOutputMismatch {
		return ReasonInvalidReceipt
	}
	if !hasLatest {
		return ReasonMissingReceipt
	}
	switch latest.Status {
	case AttemptStarted:
		return ReasonIncompleteAttempt
	case AttemptFailed:
		return ReasonFailedAttempt
	case AttemptEffectUnknown:
		return ReasonAmbiguousEffect
	default:
		return ReasonMissingReceipt
	}
}

func corroborate(corr Corroborate, ctx context.Context, nodeID, witnessKey string) (string, bool) {
	if corr == nil || !opaqueKey(witnessKey) {
		return "", false
	}
	source, ok := corr(ctx, nodeID, witnessKey)
	if !ok {
		return "", false
	}
	if source == "" {
		source = "independent:" + witnessKey
	}
	return source, true
}

func recount(report *Report) {
	report.Skipped, report.Rerun, report.Refused = 0, 0, 0
	for _, row := range report.Nodes {
		switch row.Disposition {
		case DispositionSkip:
			report.Skipped++
		case DispositionRerun:
			report.Rerun++
		case DispositionRefuse:
			report.Refused++
		}
	}
}

func attachRefusal(report *Report, err error) error {
	var refusal *Refusal
	if !errors.As(err, &refusal) {
		refusal = refuse(RefusalPersistence, "resume reconstruction failed")
	}
	report.Refusal = refusal
	return refusal
}

func failReport(report Report, reason RefusalReason, detail string) (Report, error) {
	err := refuse(reason, "%s", detail)
	report.Refusal = err
	return report, err
}

// NodeReportsByID returns a copy so callers cannot mutate the report's
// deterministic topological order.
func NodeReportsByID(report Report) map[string]NodeReport {
	out := make(map[string]NodeReport, len(report.Nodes))
	for _, row := range report.Nodes {
		out[row.NodeID] = row
	}
	return out
}
