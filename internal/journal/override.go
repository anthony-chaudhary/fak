package journal

// KindSecurityOverride marks an auditable security override row (IFC sink override or
// quarantine result override) in the hash-chained journal.
const KindSecurityOverride = "SECURITY_OVERRIDE"

// AppendSecurityOverride records an operator or model-requested security override
// (IFC sink override or quarantine result override) as a durable, chained row.
func (j *Journal) AppendSecurityOverride(overrideType, target, traceID, reason, justification, by string) Row {
	if j == nil {
		return Row{}
	}
	row := Row{
		Kind:         KindSecurityOverride,
		Tool:         target,
		TraceID:      traceID,
		Verdict:      "ALLOW",
		Reason:       reason,
		By:           by,
		Witness:      justification,
		OverrideType: overrideType,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
