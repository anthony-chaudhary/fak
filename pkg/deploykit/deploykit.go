// Package deploykit is the shared contract every fak deployable plugs into: one
// descriptor, one stage enum, one executor seam, one hooks struct, and one
// receipt schema (fak.deploy.receipt/v1).
//
// The lifecycle it names is
//
//	resolve → acquire → verify → preflight → drain → stage → swap → activate → probe → commit | rollback → receipt
//
// This package holds types and a fail-closed stage parser only. It performs no
// swap, stamp, artifact, service, or probe work of its own; those facades wrap
// the existing public primitives (internal/selfinstall and friends) in their own
// subpackages, and the plan/apply driver consumes all of them.
package deploykit

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// Deployable describes one thing fak installs, updates, and rolls back. A new
// deployable should need only a descriptor plus optional Hooks, never new
// lifecycle code.
type Deployable struct {
	// Name identifies the deployable in plans and receipts (e.g. "fak", "router").
	Name string `json:"name"`
	// Source names where the candidate artifact comes from.
	Source Source `json:"source"`
	// Targets are the installed paths the swap replaces, one row per file.
	Targets []Target `json:"targets"`
	// Service optionally names the service unit that runs the deployable.
	// Empty means the deployable is not service-managed.
	Service string `json:"service,omitempty"`
	// Drain optionally names the endpoint that drains traffic before the swap.
	// Empty means no drain step.
	Drain string `json:"drain,omitempty"`
	// Probes are the post-activation checks that decide commit versus rollback.
	Probes []Probe `json:"probes,omitempty"`
	// Rollback bounds how many prior generations are kept for rollback.
	Rollback RollbackPolicy `json:"rollback"`
}

// Source is the artifact source of a deployable. Kind is an opaque tag here
// (for example "build" or "fetch"); the artifact facade gives it meaning.
type Source struct {
	Kind string `json:"kind"`
}

// Target is one installed file a deployable replaces.
type Target struct {
	Path string `json:"path"`
	Role string `json:"role"`
}

// Probe names one post-activation check. The probe facade interprets it.
type Probe struct {
	Name     string `json:"name"`
	Endpoint string `json:"endpoint"`
}

// RollbackPolicy bounds the rollback slots a deployable keeps after commit.
type RollbackPolicy struct {
	KeepSlots int `json:"keep_slots"`
}

// Stage is one step of the deployable lifecycle. The zero Stage is not a
// valid stage; ParseStage never returns it without an error.
type Stage string

const (
	StageResolve   Stage = "resolve"
	StageAcquire   Stage = "acquire"
	StageVerify    Stage = "verify"
	StagePreflight Stage = "preflight"
	StageDrain     Stage = "drain"
	StageStage     Stage = "stage"
	StageSwap      Stage = "swap"
	StageActivate  Stage = "activate"
	StageProbe     Stage = "probe"
	StageCommit    Stage = "commit"
	StageRollback  Stage = "rollback"
	StageReceipt   Stage = "receipt"
)

// stageOrder is the canonical lifecycle order. Commit and rollback are the two
// exits of probe; both end in receipt.
var stageOrder = [...]Stage{
	StageResolve,
	StageAcquire,
	StageVerify,
	StagePreflight,
	StageDrain,
	StageStage,
	StageSwap,
	StageActivate,
	StageProbe,
	StageCommit,
	StageRollback,
	StageReceipt,
}

// ErrUnknownStage is returned (wrapped) by ParseStage for any token that is
// not exactly one of the lifecycle stage names.
var ErrUnknownStage = errors.New("deploykit: unknown stage")

// Stages returns every lifecycle stage in canonical order.
func Stages() []Stage {
	return append([]Stage(nil), stageOrder[:]...)
}

// ParseStage maps a token to its Stage. Matching is exact and case-sensitive:
// an empty, unknown, or case-variant token returns an error wrapping
// ErrUnknownStage.
func ParseStage(s string) (Stage, error) {
	for _, st := range stageOrder {
		if string(st) == s {
			return st, nil
		}
	}
	return "", fmt.Errorf("%w %q", ErrUnknownStage, s)
}

// Valid reports whether s is one of the lifecycle stages.
func (s Stage) Valid() bool {
	_, err := ParseStage(string(s))
	return err == nil
}

// String returns the stage token.
func (s Stage) String() string { return string(s) }

// MarshalText refuses to encode a stage that is not a lifecycle stage.
func (s Stage) MarshalText() ([]byte, error) {
	if _, err := ParseStage(string(s)); err != nil {
		return nil, err
	}
	return []byte(s), nil
}

// UnmarshalText decodes a stage token and fails closed on unknown tokens.
func (s *Stage) UnmarshalText(b []byte) error {
	st, err := ParseStage(string(b))
	if err != nil {
		return err
	}
	*s = st
	return nil
}

// Executor runs one command where the deployable lives and returns combined
// output plus whether it succeeded. ok=false means the command failed or could
// not be started. The signature matches selfinstall.Runner so local and remote
// executors are interchangeable.
type Executor interface {
	Run(ctx context.Context, dir, name string, args ...string) (out string, ok bool)
}

// Local returns the executor for the current host. It wraps
// selfinstall.RealRunner, so child processes stay windowless on Windows.
func Local() Executor { return localExecutor{} }

type localExecutor struct{}

func (localExecutor) Run(ctx context.Context, dir, name string, args ...string) (string, bool) {
	return selfinstall.RealRunner(ctx, dir, name, args...)
}

// Hooks are the extension points a deployable uses to add checks without
// writing lifecycle code. A nil hook is skipped. A non-nil error from
// Preflight stops the run before any mutation; one from Probe sends the run to
// rollback.
type Hooks struct {
	Preflight func(ctx context.Context, x Executor, d Deployable) error
	Probe     func(ctx context.Context, x Executor, d Deployable) error
}
