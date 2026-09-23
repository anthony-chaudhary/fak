package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostgrant"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	guardHostGrantPathEnv      = "FAK_GUARD_HOSTGRANT_PATH"
	guardHostGrantCapacityEnv  = "FAK_GUARD_HOSTGRANT_CAPACITY"
	guardHostGrantRequestIDEnv = "FAK_GUARD_HOSTGRANT_REQUEST_ID"
	guardHostGrantTTL          = 24 * time.Hour
	guardHostGrantLockTimeout  = 5 * time.Second
)

// startGuardChildWithHostGrant starts child under the ordinary managed-job
// containment boundary. When host admission is configured, it reserves one
// process seat before start and transfers the fencing token to the child
// process identity before returning. The caller must invoke release only after
// child.Wait and job.Close; release is idempotent.
func startGuardChildWithHostGrant(ctx context.Context, child *exec.Cmd, cfg windowgate.ManagedJobConfig) (job *windowgate.JobObject, release func() error, err error) {
	store, enabled, err := guardHostGrantStoreFromEnv()
	if err != nil {
		return nil, nil, err
	}
	if !enabled {
		job, err = windowgate.StartManagedAgentInNewJob(child, cfg)
		return job, func() error { return nil }, err
	}

	parentOwner, err := guardHostGrantOwner("parent", os.Getpid())
	if err != nil {
		return nil, nil, err
	}
	requestID, err := guardHostGrantRequestID()
	if err != nil {
		return nil, nil, err
	}
	operationCtx, cancel := guardHostGrantOperationContext(ctx)
	grant, err := store.TryAcquire(operationCtx, hostgrant.Request{
		ID:    requestID,
		Owner: parentOwner,
		Cost:  hostgrant.Vector{Processes: 1},
		TTL:   guardHostGrantTTL,
	})
	cancel()
	if err != nil {
		return nil, nil, err
	}

	releaseGrant := func(g hostgrant.Grant) func() error {
		var once sync.Once
		var releaseErr error
		return func() error {
			once.Do(func() {
				releaseCtx, cancel := context.WithTimeout(context.Background(), guardHostGrantLockTimeout)
				defer cancel()
				releaseErr = store.Release(releaseCtx, g)
			})
			return releaseErr
		}
	}
	parentRelease := releaseGrant(grant)
	job, err = windowgate.StartManagedAgentInNewJob(child, cfg)
	if err != nil {
		return nil, nil, errors.Join(err, parentRelease())
	}

	failStarted := func(cause error) (*windowgate.JobObject, func() error, error) {
		closeErr := job.Close()
		var treeKillErr error
		if closeErr != nil && child.Process != nil {
			if ok, detail := guardChildTreeKill(child.Process.Pid); !ok {
				treeKillErr = fmt.Errorf("fak guard: tree kill PID %d failed after job close: %s", child.Process.Pid, detail)
			}
		}
		wait := make(chan error, 1)
		go func() { wait <- child.Wait() }()
		var waitErr error
		select {
		case waitErr = <-wait:
		case <-time.After(5 * time.Second):
			waitErr = errors.New("fak guard: managed child did not join within 5s")
		}
		if closeErr != nil || treeKillErr != nil || child.ProcessState == nil {
			return nil, nil, errors.Join(cause, closeErr, treeKillErr, waitErr)
		}
		return nil, nil, errors.Join(cause, waitErr, parentRelease())
	}
	if child.Process == nil {
		return failStarted(errors.New("fak guard: managed child started without process identity"))
	}
	childOwner, err := guardHostGrantOwner("child", child.Process.Pid)
	if err != nil {
		return failStarted(err)
	}
	operationCtx, cancel = guardHostGrantOperationContext(ctx)
	grant, err = store.Transfer(operationCtx, grant, childOwner)
	cancel()
	if err != nil {
		return failStarted(err)
	}
	return job, releaseGrant(grant), nil
}

func guardHostGrantStoreFromEnv() (hostgrant.Store, bool, error) {
	path := strings.TrimSpace(os.Getenv(guardHostGrantPathEnv))
	capacityText := strings.TrimSpace(os.Getenv(guardHostGrantCapacityEnv))
	requestID := strings.TrimSpace(os.Getenv(guardHostGrantRequestIDEnv))
	if path == "" && capacityText == "" && requestID == "" {
		return hostgrant.Store{}, false, nil
	}
	if path == "" || capacityText == "" {
		return hostgrant.Store{}, false, fmt.Errorf("fak guard: %s and %s must be configured together", guardHostGrantPathEnv, guardHostGrantCapacityEnv)
	}
	if !filepath.IsAbs(path) {
		return hostgrant.Store{}, false, fmt.Errorf("fak guard: %s must be an absolute path", guardHostGrantPathEnv)
	}
	capacity, err := strconv.ParseUint(capacityText, 10, 64)
	if err != nil || capacity == 0 {
		return hostgrant.Store{}, false, fmt.Errorf("fak guard: %s must be a positive base-10 integer", guardHostGrantCapacityEnv)
	}
	return hostgrant.Store{Path: filepath.Clean(path), Capacity: hostgrant.Vector{Processes: capacity}}, true, nil
}

func guardHostGrantOwner(kind string, pid int) (hostgrant.Owner, error) {
	startedAt, ok := processalive.StartTime(pid)
	if !ok || startedAt.IsZero() {
		return hostgrant.Owner{}, fmt.Errorf("fak guard: process start time unavailable for %s PID %d", kind, pid)
	}
	return hostgrant.Owner{ID: fmt.Sprintf("fak-guard-%s:%d", kind, pid), PID: pid, StartedAt: startedAt}, nil
}

func newGuardHostGrantID() (string, error) {
	var entropy [16]byte
	if _, err := rand.Read(entropy[:]); err != nil {
		return "", fmt.Errorf("fak guard: create host grant id: %w", err)
	}
	return "guard:" + hex.EncodeToString(entropy[:]), nil
}

func guardHostGrantRequestID() (string, error) {
	if id := strings.TrimSpace(os.Getenv(guardHostGrantRequestIDEnv)); id != "" {
		return id, nil
	}
	return newGuardHostGrantID()
}

func guardHostGrantOperationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= guardHostGrantLockTimeout {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, guardHostGrantLockTimeout)
}

type guardHostGrantCleanupError struct {
	err error
}

func (e *guardHostGrantCleanupError) Error() string {
	return "fak guard: host grant cleanup: " + e.err.Error()
}
func (e *guardHostGrantCleanupError) Unwrap() error { return e.err }

func finishGuardChildHostGrant(job *windowgate.JobObject, release func() error) error {
	if err := job.Close(); err != nil {
		return &guardHostGrantCleanupError{err: err}
	}
	if err := release(); err != nil {
		return &guardHostGrantCleanupError{err: err}
	}
	return nil
}

func isGuardHostGrantCleanupError(err error) bool {
	var target *guardHostGrantCleanupError
	return errors.As(err, &target)
}
