package session

// ResetTransactionSchema is the stable schema token for a budget-reset audit row.
const ResetTransactionSchema = "fak.session.reset_transaction.v1"

// ResetBudgetRearm records the fresh budget a reset armed on the child trace.
type ResetBudgetRearm struct {
	TurnsLeft         int `json:"turns_left"`
	TokensLeft        int `json:"tokens_left"`
	ContextTokensLeft int `json:"context_tokens_left,omitempty"`
	ContextTokensCap  int `json:"context_tokens_cap,omitempty"`
}

// ResetOmittedSpan is a payload-free pointer to transcript bytes that did NOT land
// verbatim in the carryover seed. Digest is the replay handle; it is over role+text,
// not a model-authored claim.
type ResetOmittedSpan struct {
	Index  int    `json:"index"`
	Role   string `json:"role,omitempty"`
	Digest string `json:"digest"`
	Reason string `json:"reason,omitempty"`
}

// ResetTransaction is the replayable row for one context-budget reset. The table
// owns the old/new trace and budget re-arm; sessionreset fills SeedDigest,
// Contributors, OmittedSpans, and WarmPrefixDigest when a carryover seed exists.
type ResetTransaction struct {
	Schema           string             `json:"schema,omitempty"`
	OldTrace         string             `json:"old_trace,omitempty"`
	NewTrace         string             `json:"new_trace,omitempty"`
	SeedDigest       string             `json:"seed_digest,omitempty"`
	Contributors     []string           `json:"contributors,omitempty"`
	OmittedSpans     []ResetOmittedSpan `json:"omitted_spans,omitempty"`
	BudgetRearm      ResetBudgetRearm   `json:"budget_rearm,omitempty,omitzero"`
	WarmPrefixDigest string             `json:"warm_prefix_digest,omitempty"`
}

// IsZero reports whether no transaction was recorded.
func (tx ResetTransaction) IsZero() bool {
	return tx.Schema == "" && tx.OldTrace == "" && tx.NewTrace == "" &&
		tx.SeedDigest == "" && len(tx.Contributors) == 0 && len(tx.OmittedSpans) == 0 &&
		tx.BudgetRearm == (ResetBudgetRearm{}) && tx.WarmPrefixDigest == ""
}

// NewResetTransaction records the table-owned half of a reset transaction.
func NewResetTransaction(parent, child string, fresh Budget) ResetTransaction {
	return normalizeResetTransaction(ResetTransaction{}, parent, child, fresh)
}

func normalizeResetTransaction(tx ResetTransaction, parent, child string, fresh Budget) ResetTransaction {
	if tx.Schema == "" {
		tx.Schema = ResetTransactionSchema
	}
	tx.OldTrace = parent
	tx.NewTrace = child
	tx.BudgetRearm = resetBudgetRearm(fresh.withContextCap())
	return tx
}

func resetBudgetRearm(b Budget) ResetBudgetRearm {
	return ResetBudgetRearm{
		TurnsLeft:         b.TurnsLeft,
		TokensLeft:        b.TokensLeft,
		ContextTokensLeft: b.ContextTokensLeft,
		ContextTokensCap:  b.ContextTokensCap,
	}
}
