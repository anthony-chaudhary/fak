package model

import (
	"time"
)

// Qwen35MTPRunReceipt measures one production invocation. Stage times partition
// TotalNanoseconds; transaction timings overlap those stages and must not be added
// to them. Memory and artifact identity belong to the enclosing process runner.
type Qwen35MTPRunReceipt struct {
	Engine           string                      `json:"engine"`
	Depth            int                         `json:"depth"`
	TotalNanoseconds int64                       `json:"total_nanoseconds"`
	Stages           map[string]int64            `json:"stage_nanoseconds"`
	Transactions     []TargetVerificationReceipt `json:"transactions"`
	Error            string                      `json:"error,omitempty"`
}

type qwen35MTPRunMeter struct {
	receipt     Qwen35MTPRunReceipt
	start, last time.Time
	stage       string
}

func (m *qwen35MTPRunMeter) enter(stage string) {
	if m == nil {
		return
	}
	now := time.Now()
	m.receipt.Stages[m.stage] += now.Sub(m.last).Nanoseconds()
	m.last, m.stage = now, stage
}

func (m *qwen35MTPRunMeter) record(tx *qwen35MTPTargetTransaction) {
	if m != nil && tx != nil {
		m.receipt.Transactions = append(m.receipt.Transactions, tx.VerificationReceipt())
	}
}
