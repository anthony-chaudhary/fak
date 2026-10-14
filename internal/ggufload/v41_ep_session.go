package ggufload

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// V41ExpertParallelSessionRequest binds one owner to an authenticated checkpoint
// and an already-formed process group. ExpectedShardSetSHA256 must come from a
// trusted manifest, independently of the files opened here. SourceSHA256 is a
// caller-declared lowercase SHA-256 used for peer agreement, not build attestation.
// Backend must explicitly support device memory, mixed packed Q2_K/Q3_K weights
// and the routed SwiGLU seam. A capability declaration is not physical proof.
type V41ExpertParallelSessionRequest struct {
	Shards                 []ShardIdentity
	ExpectedShardSetSHA256 string
	SourceSHA256           string
	Group                  *model.DistComm
	Backend                compute.Backend
	LoadProfiler           *LoadProfiler
}

// V41ExpertParallelLoadRetirement describes only construction-owned releases.
// Available && Completed on a preflight refusal means no owned work began.
// An unsuccessful opaque factory leaves the observation unavailable unless an
// observed outer release failure establishes Completed=false. Err contains only
// those observed release failures, never an inferred failure of hidden cleanup.
type V41ExpertParallelLoadRetirement struct {
	Available bool
	Completed bool
	Err       error
}

// V41ExpertParallelSessionResult preserves a failed construction's retirement
// evidence even when the compatibility API must rethrow its original panic.
// Panicked is independent of the payload, including a recovered nil payload.
// Successful ownership transfer has no factory-retirement observation; the
// caller still owns Session and its model/readers. Neither state proves native
// device execution or cleanup hidden inside an unsuccessful opaque factory.
type V41ExpertParallelSessionResult struct {
	Session              *model.Session
	Coordinator          *model.EPDecodeCoordinator
	Err                  error
	Panicked             bool
	PrimaryPanic         any
	OwnershipTransferred bool
	Retirement           V41ExpertParallelLoadRetirement
}

// V41ExpertParallelLoadPanicError exposes a captured construction or release
// panic without replacing its payload or requiring error-string parsing.
type V41ExpertParallelLoadPanicError struct {
	Operation string
	value     any
}

func (e *V41ExpertParallelLoadPanicError) Error() string {
	return fmt.Sprintf("gguf: V4.1 EP %s panic: %v", e.Operation, e.value)
}

func (e *V41ExpertParallelLoadPanicError) PanicValue() any { return e.value }

func (e *V41ExpertParallelLoadPanicError) Unwrap() error {
	err, _ := e.value.(error)
	return err
}

// LoadV41ExpertParallelSession authenticates retained readers, materializes only
// this group's owner band, binds the exact collective, and creates a checked HAL
// Session. The returned coordinator is non-nil only on rank zero. No forward or
// transport exchange is performed; first-forward geometry, packed-owner and
// device-only admission remain authoritative. Production serve stays unqualified.
//
// Session.M owns the checkpoint: use Session.Close and Model.CloseWeights, and
// rank zero must call coordinator.Shutdown after the last request. Further
// sessions use that model's NewBackendSessionChecked with the same backend.
// Keep the model/configuration and checkpoint files immutable until teardown.
//
// Group and Backend are borrowed. This factory never closes or aborts Group;
// pre-binding refusal leaves it untouched. Native failures after binding retain
// their existing terminal-group semantics. The caller must coordinate peer
// cancellation and join running workers before closing its group. ctx cancels
// construction only, after admitted loader work drains; it does not supervise
// returned sessions or idle follower waits.
func LoadV41ExpertParallelSession(ctx context.Context, req V41ExpertParallelSessionRequest) (*model.Session, *model.EPDecodeCoordinator, error) {
	return loadV41ExpertParallelSession(ctx, req, OpenWeightsAuthenticated)
}

func loadV41ExpertParallelSession(ctx context.Context, req V41ExpertParallelSessionRequest,
	open func(context.Context, []ShardIdentity, string) (*WeightSource, error),
) (*model.Session, *model.EPDecodeCoordinator, error) {
	result := loadV41ExpertParallelSessionWithResult(ctx, req, open)
	if result.Panicked {
		panic(result.PrimaryPanic)
	}
	return result.Session, result.Coordinator, result.Err
}

// LoadV41ExpertParallelSessionWithResult uses the same authenticated loader as
// LoadV41ExpertParallelSession, but captures construction panics explicitly and
// retains observed outer cleanup failures. It does not supervise returned users.
// Opaque factory failures cannot establish complete retirement of their private
// allocations. Group and Backend remain borrowed on every outcome.
func LoadV41ExpertParallelSessionWithResult(ctx context.Context, req V41ExpertParallelSessionRequest) V41ExpertParallelSessionResult {
	return loadV41ExpertParallelSessionWithResult(ctx, req, OpenWeightsAuthenticated)
}

func loadV41ExpertParallelSessionWithResult(ctx context.Context, req V41ExpertParallelSessionRequest,
	open func(context.Context, []ShardIdentity, string) (*WeightSource, error),
) (result V41ExpertParallelSessionResult) {
	var ws *WeightSource
	var m *model.Model
	var session *model.Session
	modelOwnsSource, opaqueIncomplete := false, false
	result.Err, result.PrimaryPanic, result.Panicked = v41EPLoadInvoke("construction", func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		be := req.Backend
		if be == nil || !be.Caps().DeviceMemory || !be.Caps().UploadDtype {
			return fmt.Errorf("gguf: V4.1 EP requires an explicit device-memory packed-upload backend")
		}
		for _, dtype := range []compute.Dtype{compute.Q2_K, compute.Q3_K} {
			if !compute.BackendSupportsDeviceWeightDtype(be, dtype) {
				return fmt.Errorf("gguf: V4.1 EP backend %s cannot execute %v", be.Name(), dtype)
			}
		}
		routed, ok := be.(compute.V41RoutedSwiGLUBackend)
		if !ok || !routed.V41RoutedSwiGLUAvailable() {
			return fmt.Errorf("gguf: V4.1 EP backend %s requires the fused routed SwiGLU operation", be.Name())
		}
		req.Shards = slices.Clone(req.Shards)
		if err := validateShardManifest(req.Shards, req.ExpectedShardSetSHA256); err != nil {
			return err
		}
		// Agreement precedes authentication; it is not artifact attestation.
		collective, err := model.NewV41ExpertParallelCollective(req.Group, model.V41ExpertParallelBinding{
			ArtifactSHA256: strings.ToLower(req.ExpectedShardSetSHA256), SourceSHA256: req.SourceSHA256,
		})
		if err != nil {
			return err
		}
		// An error/panic from an opaque constructor leaves its unreturned
		// internal allocations unobserved. Returned non-nil partial resources
		// are still owned here and receive the same ordered cleanup.
		opaqueIncomplete = true
		ws, err = open(ctx, req.Shards, req.ExpectedShardSetSHA256)
		if err != nil {
			return err
		}
		if ws == nil {
			return errors.New("gguf: V4.1 EP opener returned no source")
		}
		opaqueIncomplete = false
		if err := ctx.Err(); err != nil {
			return err
		}
		cfg, err := ws.File.Config()
		if err != nil {
			return err
		}
		if !cfg.IsDeepSeekV41() {
			return fmt.Errorf("gguf: V4.1 EP requires a DeepSeek V4.1 checkpoint, got %q", cfg.ModelType)
		}
		band, err := ExpertShardForRank(cfg.NumExperts, req.Group.Size(), req.Group.Rank())
		if err != nil {
			return err
		}
		if err := model.ValidateBackendForwardConfig(cfg, be); err != nil {
			return err
		}
		opaqueIncomplete = true
		m, err = ws.QuantModelQ4KProfileOptionsContext(ctx, req.LoadProfiler, WithExpertShard(band.Lo, band.Hi))
		if err != nil {
			return err
		}
		if m == nil {
			return errors.New("gguf: V4.1 EP materializer returned no model")
		}
		opaqueIncomplete = false
		m.SetWeightCloser(ws)
		modelOwnsSource = true
		if err := ctx.Err(); err != nil {
			return err
		}
		m.SetExpertParallelRanks(req.Group.Size())
		m.SetExpertParallelRank(req.Group.Rank())
		m.SetExpertParallelCollective(collective)
		var coord *model.EPDecodeCoordinator
		if req.Group.Rank() == 0 {
			coord, err = model.NewEPDecodeCoordinator(req.Group)
			if err != nil {
				return err
			}
			m.SetEPDecodeCoordinator(coord)
		}
		opaqueIncomplete = true
		session, err = m.NewBackendSessionChecked(be)
		if err != nil {
			return err
		}
		if session == nil {
			return errors.New("gguf: V4.1 EP constructor returned no Session")
		}
		opaqueIncomplete = false
		if err := ctx.Err(); err != nil {
			return err
		}
		result.Session, result.Coordinator = session, coord
		result.OwnershipTransferred = true
		return nil
	})
	if result.OwnershipTransferred {
		return result // Retirement is unavailable/not applicable, not success.
	}
	result.Retirement.Available = !opaqueIncomplete
	result.Retirement.Completed = !opaqueIncomplete
	var failures [3]error // One per outer release stage; never unbounded.
	count := 0
	cleanup := func(operation string, fn func() error) {
		err, payload, panicked := v41EPLoadInvoke(operation, fn)
		if err == nil {
			return
		}
		if panicked && !result.Panicked && result.Err == nil {
			result.Panicked, result.PrimaryPanic = true, payload
		}
		failures[count] = fmt.Errorf("gguf: V4.1 EP retirement %s: %w", operation, err)
		count++
		result.Retirement.Available, result.Retirement.Completed = true, false
	}
	if session != nil {
		cleanup("Session.Close", func() error { session.Close(); return nil })
	}
	if m != nil {
		cleanup("Model.CloseWeights", m.CloseWeights)
	}
	if ws != nil && !modelOwnsSource {
		cleanup("WeightSource.Close", ws.Close)
	}
	if count != 0 {
		result.Retirement.Err = errors.Join(failures[:count]...)
		if result.Err == nil {
			result.Err = result.Retirement.Err
		} else {
			result.Err = errors.Join(result.Err, result.Retirement.Err)
		}
	}
	return result
}

// Completion-state detection keeps panic presence separate from a nil payload.
// runtime.Goexit never returns a result and is outside this factory contract.
func v41EPLoadInvoke(operation string, fn func() error) (err error, payload any, panicked bool) {
	completed := false
	defer func() {
		if !completed {
			payload, panicked = recover(), true
			err = &V41ExpertParallelLoadPanicError{Operation: operation, value: payload}
		}
	}()
	err = fn()
	completed = true
	return
}
