package microagent

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Task is the bounded, typed input admitted for one child computation.
type Task struct {
	Goal         string     `json:"goal"`
	ArtifactRefs []string   `json:"artifact_refs"`
	Authority    []string   `json:"authority"`
	Budget       TaskBudget `json:"budget"`
}

// TaskBudget is the child's declared local envelope. Aggregate lineage budgets
// remain a host admission concern rather than agent-authored task data.
type TaskBudget struct {
	MaxTurns int `json:"max_turns"`
}

// Receipt is the only child material intended to fold into a parent context.
type Receipt struct {
	Decision            string        `json:"decision"`
	Evidence            []EvidenceRef `json:"evidence"`
	UnresolvedQuestions []string      `json:"unresolved_questions"`
}

// EvidenceRef points at independently readable proof without embedding a child transcript.
type EvidenceRef struct {
	Kind string `json:"kind"`
	Ref  string `json:"ref"`
}

var (
	ErrInvalidTask    = errors.New("microagent: invalid child task")
	ErrInvalidReceipt = errors.New("microagent: invalid child receipt")
)

// DecodeTaskJSON rejects schema drift before validating the task envelope.
func DecodeTaskJSON(raw []byte) (Task, error) {
	var task Task
	if err := decodeStrictJSON(raw, &task); err != nil {
		return Task{}, fmt.Errorf("%w: %v", ErrInvalidTask, err)
	}
	if err := task.Validate(); err != nil {
		return Task{}, err
	}
	return task, nil
}

// Validate checks the minimum contract required for host admission.
func (t Task) Validate() error {
	if strings.TrimSpace(t.Goal) == "" {
		return fmt.Errorf("%w: goal is required", ErrInvalidTask)
	}
	if t.Budget.MaxTurns < 1 || t.Budget.MaxTurns > 3 {
		return fmt.Errorf("%w: budget.max_turns must be within 1..3", ErrInvalidTask)
	}
	return nil
}

// DecodeReceiptJSON rejects schema drift and receipts without checkable proof.
func DecodeReceiptJSON(raw []byte) (Receipt, error) {
	var receipt Receipt
	if err := decodeStrictJSON(raw, &receipt); err != nil {
		return Receipt{}, fmt.Errorf("%w: %v", ErrInvalidReceipt, err)
	}
	if err := receipt.Validate(); err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}

// Validate keeps an unsupported child conclusion out of the parent context.
func (r Receipt) Validate() error {
	if strings.TrimSpace(r.Decision) == "" {
		return fmt.Errorf("%w: decision is required", ErrInvalidReceipt)
	}
	if len(r.Evidence) == 0 {
		return fmt.Errorf("%w: evidence is required", ErrInvalidReceipt)
	}
	for i, evidence := range r.Evidence {
		if strings.TrimSpace(evidence.Kind) == "" || strings.TrimSpace(evidence.Ref) == "" {
			return fmt.Errorf("%w: evidence[%d] requires kind and ref", ErrInvalidReceipt, i)
		}
	}
	return nil
}

func decodeStrictJSON(raw []byte, dst any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}
