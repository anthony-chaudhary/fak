package fabricmap

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

type HopState string

const (
	HopPending  HopState = "pending"
	HopComplete HopState = "complete"
	HopFailed   HopState = "failed"
)

type Integrity struct {
	Algorithm string `json:"algorithm"`
	Digest    string `json:"digest"`
}
type HopReceipt struct {
	HopIndex    int       `json:"hop_index"`
	LinkID      string    `json:"link_id"`
	From        string    `json:"from"`
	To          string    `json:"to"`
	Transport   string    `json:"transport"`
	Bytes       uint64    `json:"bytes"`
	StartedAt   time.Time `json:"started_at"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	Integrity   Integrity `json:"integrity,omitempty"`
	State       HopState  `json:"state"`
	Error       string    `json:"error,omitempty"`
}
type ExecutionState string

const (
	ExecutionComplete ExecutionState = "complete"
	ExecutionPartial  ExecutionState = "partial"
	ExecutionFailed   ExecutionState = "failed"
)

type RecoveryAction string

const (
	RecoveryNone           RecoveryAction = "none"
	RecoveryRetry          RecoveryAction = "retry"
	RecoveryAlternateRoute RecoveryAction = "replan"
	RecoveryOperator       RecoveryAction = "operator"
)

type ExecutionReceipt struct {
	From      string         `json:"from"`
	To        string         `json:"to"`
	Operation Operation      `json:"operation"`
	State     ExecutionState `json:"state"`
	Recovery  RecoveryAction `json:"recovery"`
	Hops      []HopReceipt   `json:"hops"`
}

// Executor executes exactly one directed link and independently witnesses its
// effect. A successful return without complete, matching receipt fields is not
// accepted as evidence.
type Executor interface {
	ExecuteHop(context.Context, AuthorizationRequest, uint64) (HopReceipt, error)
}
type ExecutorFunc func(context.Context, AuthorizationRequest, uint64) (HopReceipt, error)

func (f ExecutorFunc) ExecuteHop(ctx context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
	return f(ctx, r, n)
}

// Execute runs an authorized route serially and stops at the first failed hop.
// It never reports complete unless every hop returned a valid receipt.
func Execute(ctx context.Context, authorized AuthorizedRoute, bytes uint64, executor Executor) (ExecutionReceipt, error) {
	result := ExecutionReceipt{From: authorized.Route.From, To: authorized.Route.To, Operation: authorized.Access.Operation, State: ExecutionFailed, Recovery: RecoveryNone, Hops: make([]HopReceipt, 0, len(authorized.Route.Links))}
	if bytes == 0 {
		return result, errors.New("execution byte count must be positive")
	}
	if executor == nil {
		return result, errors.New("fabric executor is required")
	}
	for i, link := range authorized.Route.Links {
		request := AuthorizationRequest{Access: cloneAccess(authorized.Access), HopIndex: i, Link: cloneLink(link)}
		receipt, err := executor.ExecuteHop(ctx, request, bytes)
		if err != nil {
			failed := HopReceipt{HopIndex: i, LinkID: link.ID, From: link.From, To: link.To, Transport: link.Transport, Bytes: bytes, State: HopFailed, Error: "transport failed"}
			result.Hops = append(result.Hops, failed)
			result.State = executionFailureState(i)
			result.Recovery = recoveryFor(authorized.Access.Operation, i)
			return result, fmt.Errorf("execute hop %d link %q: %w", i, link.ID, err)
		}
		if err := validateHopReceipt(receipt, i, link, bytes); err != nil {
			receipt.State = HopFailed
			receipt.Error = "invalid receipt"
			result.Hops = append(result.Hops, receipt)
			result.State = executionFailureState(i)
			result.Recovery = RecoveryOperator
			return result, fmt.Errorf("witness hop %d link %q: %w", i, link.ID, err)
		}
		result.Hops = append(result.Hops, receipt)
	}
	result.State = ExecutionComplete
	result.Recovery = RecoveryNone
	return result, nil
}

func validateHopReceipt(r HopReceipt, index int, link Link, bytes uint64) error {
	if r.HopIndex != index || r.LinkID != link.ID || r.From != link.From || r.To != link.To || r.Transport != link.Transport {
		return errors.New("receipt identity does not match directed hop")
	}
	if r.State != HopComplete {
		return fmt.Errorf("receipt state %q is not complete", r.State)
	}
	if r.Bytes != bytes {
		return fmt.Errorf("receipt bytes %d do not match requested %d", r.Bytes, bytes)
	}
	if r.StartedAt.IsZero() || r.CompletedAt.IsZero() || r.CompletedAt.Before(r.StartedAt) {
		return errors.New("receipt timing is invalid")
	}
	if strings.TrimSpace(r.Integrity.Algorithm) == "" || strings.TrimSpace(r.Integrity.Digest) == "" {
		return errors.New("receipt integrity evidence is required")
	}
	return nil
}
func executionFailureState(index int) ExecutionState {
	if index == 0 {
		return ExecutionFailed
	}
	return ExecutionPartial
}
func recoveryFor(operation Operation, completed int) RecoveryAction {
	if completed == 0 {
		return RecoveryRetry
	}
	switch operation {
	case OperationRead, OperationCopy:
		return RecoveryAlternateRoute
	case OperationMove, OperationWrite, OperationWriteBack:
		return RecoveryOperator
	default:
		return RecoveryOperator
	}
}
