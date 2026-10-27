package remotemaint

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

type Prepared struct {
	authority Authority
	grant     Grant
	mu        sync.Mutex
	executed  bool
}

func Prepare(ctx context.Context, authority Authority, req Request) (*Prepared, error) {
	if authority == nil {
		return nil, fmt.Errorf("%w: missing authority", ErrUnavailable)
	}
	if err := validateRequest(req); err != nil {
		return nil, err
	}
	grant, err := authority.Resolve(ctx, req)
	if err != nil {
		return nil, boundedWrap(ErrAuthority, err)
	}
	grant = normalizeGrant(grant)
	if err := validateGrant(grant, time.Now().UTC()); err != nil {
		return nil, err
	}
	if grant.ProfileID != req.ProfileID || grant.OperationDigest != req.OperationDigest {
		return nil, fmt.Errorf("%w: request identity mismatch", ErrChanged)
	}
	if err := authority.Validate(ctx, cloneGrant(grant)); err != nil {
		return nil, boundedWrap(ErrAuthority, err)
	}
	return &Prepared{authority: authority, grant: cloneGrant(grant)}, nil
}

func (p *Prepared) Grant() Grant {
	if p == nil {
		return Grant{}
	}
	return cloneGrant(p.grant)
}

func (p *Prepared) Execute(ctx context.Context) (Receipt, error) {
	if p == nil || p.authority == nil {
		return Receipt{}, fmt.Errorf("%w: missing prepared authority", ErrUnavailable)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.executed {
		return Receipt{}, fmt.Errorf("%w: prepared grant already executed", ErrChanged)
	}
	grant := cloneGrant(p.grant)
	if err := validateGrant(grant, time.Now().UTC()); err != nil {
		return Receipt{}, err
	}
	if err := p.authority.Validate(ctx, cloneGrant(grant)); err != nil {
		return Receipt{}, boundedWrap(ErrAuthority, err)
	}
	startedAt := time.Now().UTC()
	if err := validateGrant(grant, startedAt); err != nil {
		return Receipt{}, err
	}
	// Consume the prepared grant at the last pre-effect boundary. Once Execute is
	// entered, an error or malformed receipt cannot prove that no effect occurred.
	p.executed = true
	receipt, err := p.authority.Execute(ctx, cloneGrant(grant))
	receipt.InstalledBoot = cloneInstalledBootObservation(receipt.InstalledBoot)
	if err != nil {
		if validateFailureReceipt(receipt, grant, time.Now().UTC()) == nil {
			switch receipt.Outcome {
			case OutcomeUnknownEffect:
				return receipt, fmt.Errorf("%w: authority reported uncertain effect", ErrUnknownEffect)
			case OutcomeRollbackFailed:
				return receipt, fmt.Errorf("%w: authority reported rollback failure", ErrRollbackFailed)
			}
		}
		return unknownEffectReceipt(grant, startedAt, time.Now().UTC()), fmt.Errorf("%w: terminal execution state unavailable", ErrUnknownEffect)
	}
	if err := validateReceipt(receipt, grant, time.Now().UTC()); err != nil {
		return unknownEffectReceipt(grant, startedAt, time.Now().UTC()), fmt.Errorf("%w: invalid terminal execution receipt", ErrUnknownEffect)
	}
	return receipt, nil
}

func unknownEffectReceipt(g Grant, startedAt, endedAt time.Time) Receipt {
	if endedAt.Before(startedAt) {
		endedAt = startedAt
	}
	return Receipt{
		Schema: ReceiptSchema, GrantDigest: GrantDigest(g), ProfileID: g.ProfileID,
		OperationDigest: g.OperationDigest, ContextID: g.ContextID, Target: g.Target,
		HostKey: g.HostKey, Operation: g.Operation, BootID: g.BootID,
		StartedAt: startedAt.UTC(), EndedAt: endedAt.UTC(), Outcome: OutcomeUnknownEffect,
		Verified: false, BackupDisposition: BackupUnknown,
	}
}

func boundedWrap(sentinel, err error) error {
	if err == nil {
		return sentinel
	}
	for _, known := range []error{ErrUnavailable, ErrInvalidRequest, ErrScope, ErrExpired, ErrChanged, ErrAuthority, ErrUnknownEffect, ErrRollbackFailed} {
		if errors.Is(err, known) {
			return fmt.Errorf("%w: authority operation denied", known)
		}
	}
	return fmt.Errorf("%w: authority operation denied", sentinel)
}
