package model

// V41WorkloadRows is observed route work, not request or output tokens.
// Replayed full folds contribute their executed rows again. Failed partial work
// is not counted, except committed cohort lanes survive later cleanup errors.
// These values must not divide timings that include unobserved failed work.
type V41WorkloadRows struct {
	ObservedOperations uint64 `json:"observed_operations"`
	ExecutedRows       uint64 `json:"executed_rows"`
	AppendedRows       uint64 `json:"appended_rows"`
}

// V41WorkloadLifetime is a fixed, bounded companion to the existing lifetime
// phase ledger. It carries no prompt, request ID, filename, timer or metric label.
type V41WorkloadLifetime struct {
	ColdFull     V41WorkloadRows `json:"cold_full_prefill"`
	SuffixSerial V41WorkloadRows `json:"suffix_incremental_prefill"`
	SuffixBatch  V41WorkloadRows `json:"suffix_ubatch_prefill"`
	SuffixRefold V41WorkloadRows `json:"suffix_refold_prefill"`
	SerialDecode V41WorkloadRows `json:"serial_incremental_decode"`
	DecodeRefold V41WorkloadRows `json:"serial_refold_decode"`
	CohortDecode V41WorkloadRows `json:"cohort_decode"`
	Overflow     bool            `json:"overflow"`
}

type v41WorkloadRoute uint8

const (
	v41WorkloadCold v41WorkloadRoute = iota
	v41WorkloadSuffixSerial
	v41WorkloadSuffixBatch
	v41WorkloadSuffixRefold
	v41WorkloadDecode
	v41WorkloadDecodeRefold
	v41WorkloadCohort
)

// Explicit route arguments avoid using the shared mutable phase selector for
// this new dimension. The expert counters retain their existing scope warning.
func (m *Model) v41NoteWorkload(route v41WorkloadRoute, executed, appended int) {
	l := v41ExpertFaultLedgerOf(m, false)
	if l == nil || executed < 0 || appended < 0 || appended > executed {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	var row *V41WorkloadRows
	switch route {
	case v41WorkloadCold:
		row = &l.workload.ColdFull
	case v41WorkloadSuffixSerial:
		row = &l.workload.SuffixSerial
	case v41WorkloadSuffixBatch:
		row = &l.workload.SuffixBatch
	case v41WorkloadSuffixRefold:
		row = &l.workload.SuffixRefold
	case v41WorkloadDecode:
		row = &l.workload.SerialDecode
	case v41WorkloadDecodeRefold:
		row = &l.workload.DecodeRefold
	case v41WorkloadCohort:
		row = &l.workload.CohortDecode
	default:
		return
	}
	max := ^uint64(0)
	if row.ObservedOperations == max || row.ExecutedRows > max-uint64(executed) || row.AppendedRows > max-uint64(appended) {
		l.workload.Overflow = true
		return
	}
	row.ObservedOperations++
	row.ExecutedRows += uint64(executed)
	row.AppendedRows += uint64(appended)
}

// V41WorkloadCoverage states what the exported lifetime rows do NOT establish.
// Unavailable values are explicit nulls, not zero measurements or inferred IDs.
type V41WorkloadCoverage struct {
	Scope                  string  `json:"scope"`
	CohortFinalResult      string  `json:"cohort_final_result"`
	OperationBasis         string  `json:"operation_basis"`
	SerialDecodeSelection  string  `json:"serial_decode_selection"`
	Basis                  string  `json:"basis"`
	Unit                   string  `json:"unit"`
	Denominator            string  `json:"denominator"`
	RequestIsolation       string  `json:"request_isolation"`
	CounterBinding         string  `json:"counter_binding"`
	FailedPartialWork      string  `json:"failed_partial_work"`
	RestoredPrefixTokens   *uint64 `json:"restored_prefix_tokens"`
	RestoredPrefixReason   string  `json:"restored_prefix_reason"`
	EffectiveTimestampMode string  `json:"effective_timestamp_mode"`
	RunIdentity            string  `json:"run_identity"`
}

func V41WorkloadCoverageDefault() V41WorkloadCoverage {
	return V41WorkloadCoverage{
		Scope: "model_lifetime", Basis: "observed_control_flow", Unit: "token_rows",
		Denominator:           "observed_route_work_not_request_or_output_tokens",
		OperationBasis:        "successful_forward_or_committed_cohort_lane_including_cleanup_error",
		CohortFinalResult:     "unavailable_observed_before_final_cleanup_not_success",
		SerialDecodeSelection: "lookahead_opt_in_and_async_selection_unavailable_combined",
		RequestIsolation:      "unsupported", CounterBinding: "not_bound_to_expert_or_device_timing",
		FailedPartialWork:      "unavailable_except_committed_cohort_lanes",
		RestoredPrefixReason:   "model_history_does_not_identify_restore_provenance",
		EffectiveTimestampMode: "unavailable_at_model_ledger",
		RunIdentity:            "unavailable_at_model_ledger_use_existing_request_receipt_identity",
	}
}
