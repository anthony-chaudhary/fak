package ggufload

import (
	"context"
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
) (_ *model.Session, _ *model.EPDecodeCoordinator, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	be := req.Backend
	if be == nil || !be.Caps().DeviceMemory || !be.Caps().UploadDtype {
		return nil, nil, fmt.Errorf("gguf: V4.1 EP requires an explicit device-memory packed-upload backend")
	}
	for _, dtype := range []compute.Dtype{compute.Q2_K, compute.Q3_K} {
		if !compute.BackendSupportsDeviceWeightDtype(be, dtype) {
			return nil, nil, fmt.Errorf("gguf: V4.1 EP backend %s cannot execute %v", be.Name(), dtype)
		}
	}
	routed, ok := be.(compute.V41RoutedSwiGLUBackend)
	if !ok || !routed.V41RoutedSwiGLUAvailable() {
		return nil, nil, fmt.Errorf("gguf: V4.1 EP backend %s requires the fused routed SwiGLU operation", be.Name())
	}
	req.Shards = slices.Clone(req.Shards)
	if err := validateShardManifest(req.Shards, req.ExpectedShardSetSHA256); err != nil {
		return nil, nil, err
	}
	// Manifest agreement is checked above; file authentication below is still
	// required before this identity can be attached to a loaded model.
	collective, err := model.NewV41ExpertParallelCollective(req.Group, model.V41ExpertParallelBinding{
		ArtifactSHA256: strings.ToLower(req.ExpectedShardSetSHA256), SourceSHA256: req.SourceSHA256,
	})
	if err != nil {
		return nil, nil, err
	}
	ws, err := open(ctx, req.Shards, req.ExpectedShardSetSHA256)
	if err != nil {
		return nil, nil, err
	}
	var m *model.Model
	var session *model.Session
	modelOwnsSource, ready := false, false
	defer func() {
		primary := recover()
		if !ready {
			// Attempt each ordered release even if a backend cleanup panics,
			// preserving the original construction error or panic identity.
			cleanup := func(f func()) {
				defer func() {
					if cause := recover(); cause != nil && primary == nil && err == nil {
						primary = cause
					}
				}()
				f()
			}
			if session != nil {
				cleanup(session.Close)
			}
			if m != nil {
				cleanup(func() { _ = m.CloseWeights() })
			}
			if !modelOwnsSource {
				cleanup(func() { _ = ws.Close() })
			}
		}
		if primary != nil {
			panic(primary)
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	cfg, err := ws.File.Config()
	if err != nil {
		return nil, nil, err
	}
	if !cfg.IsDeepSeekV41() {
		return nil, nil, fmt.Errorf("gguf: V4.1 EP requires a DeepSeek V4.1 checkpoint, got %q", cfg.ModelType)
	}
	band, err := ExpertShardForRank(cfg.NumExperts, req.Group.Size(), req.Group.Rank())
	if err != nil {
		return nil, nil, err
	}
	if err := model.ValidateBackendForwardConfig(cfg, be); err != nil {
		return nil, nil, err
	}
	m, err = ws.QuantModelQ4KProfileOptionsContext(ctx, req.LoadProfiler, WithExpertShard(band.Lo, band.Hi))
	if err != nil {
		return nil, nil, err
	}
	m.SetWeightCloser(ws)
	modelOwnsSource = true
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	m.SetExpertParallelRanks(req.Group.Size())
	m.SetExpertParallelRank(req.Group.Rank())
	m.SetExpertParallelCollective(collective)
	var coord *model.EPDecodeCoordinator
	if req.Group.Rank() == 0 {
		coord, err = model.NewEPDecodeCoordinator(req.Group)
		if err != nil {
			return nil, nil, err
		}
		m.SetEPDecodeCoordinator(coord)
	}
	session, err = m.NewBackendSessionChecked(be)
	if err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	ready = true
	return session, coord, nil
}
