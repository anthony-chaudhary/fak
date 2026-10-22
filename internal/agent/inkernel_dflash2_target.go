package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// DFlash2TargetSpec is a trusted, ordered identity supplied by the operator.
// ExpectedShardSetSHA256 must come from independent provenance, not a fresh hash
// of arbitrary files. All checkpoint files must remain immutable until Close.
// Retained readers prevent pathname replacement, not concurrent in-place writes.
type DFlash2TargetSpec struct {
	Shards                 []ggufload.ShardIdentity
	ExpectedShardSetSHA256 string
	TokenizerID            string
	RendererID             string
	SidecarPath            string
}

// DFlash2Target owns the authenticated source and loaded quantized target.
// Model and Tokenizer are borrowed immutable handles for gateway configuration.
// Only this owner may retire the model; all serving aliases must be registered
// before requests start. Direct Session users are outside this serving contract.
type DFlash2Target struct {
	mu          sync.Mutex
	options     DFlash2TargetOptions
	spec        DFlash2TargetSpec
	source      *ggufload.WeightSource
	m           *model.Model
	tok         *tokenizer.Tokenizer
	backend     compute.Backend
	domain      *dflash2Domain
	fingerprint [32]byte
	stops       map[int]bool
	leases      map[*dflash2Lease]struct{}
	closed      bool
	closeErr    error
	factory     func(context.Context, *dflash2Lease) error // private test seam
}

func dflash2Digest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && s == strings.ToLower(s)
}

func validateDFlash2Spec(s DFlash2TargetSpec) error {
	if len(s.Shards) == 0 || len(s.Shards) > 128 || !dflash2Digest(s.ExpectedShardSetSHA256) ||
		!strings.HasPrefix(s.TokenizerID, "sha256:") || !dflash2Digest(strings.TrimPrefix(s.TokenizerID, "sha256:")) ||
		(s.RendererID != "qwen-chatml-v1:thinking" && s.RendererID != "qwen-chatml-v1:no-thinking" &&
			s.RendererID != "ornith-qwen35-chatml-v1:thinking" && s.RendererID != "ornith-qwen35-chatml-v1:no-thinking") ||
		!filepath.IsAbs(s.SidecarPath) || filepath.Clean(s.SidecarPath) != s.SidecarPath {
		return fmt.Errorf("%w: shard, tokenizer, renderer and absolute sidecar identities required", ErrDFlash2Identity)
	}
	return nil
}

func dflash2TargetFingerprint(m *model.Model, tok *tokenizer.Tokenizer) ([32]byte, error) {
	var zero [32]byte
	if m == nil || tok == nil {
		return zero, errors.New("dflash2: model and tokenizer required")
	}
	// Config's normal JSON omits BOS/EOS, so bind them explicitly.
	b, err := json.Marshal(struct {
		Config    model.Config
		BOS       *int
		EOS       int
		EOSList   []int
		Tokenizer string
	}{m.Cfg, m.Cfg.BOSTokenID, m.Cfg.EOSTokenID, m.Cfg.EOSTokenIDs, tok.Identity()})
	if err != nil {
		return zero, err
	}
	return sha256.Sum256(b), nil
}

func dflash2StopIDs(tok *tokenizer.Tokenizer, cfg model.Config) map[int]bool {
	stops := StopIDs(tok, cfg)
	if len(cfg.EOSTokenIDs) > 0 {
		for _, id := range cfg.EOSTokenIDs {
			if id >= 0 {
				stops[id] = true
			}
		}
	} else if cfg.EOSTokenID >= 0 {
		stops[cfg.EOSTokenID] = true
	}
	return stops
}

// NewDFlash2Target authenticates and loads the exact GGUF readers. Its fixed
// packed/Q8 load policy never expands the entire 27B target into F32 weights.
// It does not download weights or assert a hardware memory/speed qualification.
func NewDFlash2Target(ctx context.Context, spec DFlash2TargetSpec, backend compute.Backend) (*DFlash2Target, error) {
	return newDFlash2Target(ctx, spec, backend, ggufload.OpenWeightsAuthenticated)
}

func newDFlash2Target(ctx context.Context, spec DFlash2TargetSpec, backend compute.Backend,
	open func(context.Context, []ggufload.ShardIdentity, string) (*ggufload.WeightSource, error)) (out *DFlash2Target, err error) {
	return newDFlash2TargetWithOptions(ctx, spec, backend, DFlash2TargetOptions{}, open)
}

func newDFlash2TargetWithOptions(ctx context.Context, spec DFlash2TargetSpec, backend compute.Backend, options DFlash2TargetOptions,
	open func(context.Context, []ggufload.ShardIdentity, string) (*ggufload.WeightSource, error)) (out *DFlash2Target, err error) {
	if err = options.validate(backend); err != nil {
		return nil, err
	}
	if ctx == nil {
		return nil, errors.New("dflash2: context required")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if err = validateDFlash2Spec(spec); err != nil {
		return nil, err
	}
	domain, err := registerDFlash2Domain(backend)
	if err != nil {
		return nil, err
	}
	release, err := domain.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer release()
	spec.Shards = slices.Clone(spec.Shards)
	target := &DFlash2Target{spec: spec, backend: backend, domain: domain, options: options, leases: make(map[*dflash2Lease]struct{})}
	completed, openEntered, openVerified := false, false, false
	defer func() {
		if !completed {
			// The loader may have acquired opaque state before interrupting. Retain the
			// whole attempt and refuse the domain; do not infer that unwind retired it.
			domain.quarantine(errors.New("dflash2: target construction interrupted"), target)
		} else if openEntered && !openVerified {
			domain.quarantine(errors.Join(errors.New("dflash2: acquisition retirement unverified"), err), target)
		} else if err != nil {
			err = errors.Join(err, dflash2CleanupCall("target construction retirement", target.retire))
		}
	}()
	if open == nil {
		completed = true
		return nil, errors.New("dflash2: source opener required")
	}
	openEntered = true
	target.source, err = open(ctx, spec.Shards, spec.ExpectedShardSetSHA256)
	if err != nil {
		completed = true
		return nil, err
	}
	if target.source == nil || target.source.File == nil {
		completed = true
		return nil, errors.New("dflash2: opener returned no authenticated source")
	}
	openVerified = true
	cfg, err := target.source.File.Config()
	if err != nil {
		completed = true
		return nil, err
	}
	if !cfg.IsQwen35Hybrid() || cfg.IsDeepSeekV41() || cfg.NumExperts != 0 ||
		cfg.NumLayers != 64 || cfg.HiddenSize != 5120 || cfg.VocabSize != 248320 || cfg.EnableResidualHook {
		completed = true
		return nil, errors.New("dflash2: exact dense Qwen3.8-27B target geometry required")
	}
	gt, ok := target.source.File.GGMLTokenizer()
	if !ok {
		completed = true
		return nil, errors.New("dflash2: authenticated embedded tokenizer required")
	}
	target.tok, err = tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		completed = true
		return nil, err
	}
	if target.tok.Identity() != spec.TokenizerID {
		completed = true
		return nil, errors.New("dflash2: tokenizer identity mismatch")
	}
	embedding, ok := target.source.Tensor("token_embd.weight")
	if !ok {
		completed = true
		return nil, errors.New("dflash2: authenticated token embedding missing")
	}
	loadOptions, err := dflash2LoadOptions(embedding.Type)
	if err != nil {
		completed = true
		return nil, err
	}
	target.m, err = target.source.QuantModelQ4KProfileOptionsContext(ctx, nil, loadOptions...)
	if err != nil {
		completed = true
		return nil, err
	}
	target.fingerprint, err = dflash2TargetFingerprint(target.m, target.tok)
	if err != nil {
		completed = true
		return nil, err
	}
	target.stops = dflash2StopIDs(target.tok, target.m.Cfg)
	if err = ctx.Err(); err != nil {
		completed = true
		return nil, err
	}
	completed = true
	return target, nil
}

func newDFlash2TargetSession(m *model.Model, backend compute.Backend) (*model.Session, error) {
	s, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		return nil, err
	}
	s.Quant = true
	s.Q4K = true
	return s, nil
}

func (t *DFlash2Target) Model() *model.Model {
	if t == nil {
		return nil
	}
	return t.m
}
func (t *DFlash2Target) Tokenizer() *tokenizer.Tokenizer {
	if t == nil {
		return nil
	}
	return t.tok
}

func (t *DFlash2Target) ValidateBinding(m *model.Model, tok *tokenizer.Tokenizer, backend compute.Backend, q4k, metal bool) error {
	if t == nil {
		return errors.New("dflash2: authenticated target required")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed || t.source == nil || t.m != m || t.tok != tok || !q4k || metal || backend == nil ||
		!reflect.TypeOf(backend).Comparable() || t.backend != backend {
		return errors.New("dflash2: target lifetime or planner identity mismatch")
	}
	owner, err := dflash2Owner(backend)
	if err != nil || t.domain == nil || owner != t.domain.owner {
		return errors.New("dflash2: execution owner changed")
	}
	fp, err := dflash2TargetFingerprint(m, tok)
	if err != nil || fp != t.fingerprint || tok.Identity() != t.spec.TokenizerID ||
		!reflect.DeepEqual(t.stops, dflash2StopIDs(tok, m.Cfg)) || m.Cfg.EnableResidualHook {
		return errors.New("dflash2: authenticated target metadata changed")
	}
	return nil
}

// retire is called only while the separate domain gate is held. A model close
// failure retains the source: otherwise mapped backing could outlive its reader.
func (t *DFlash2Target) retire() (err error) {
	t.mu.Lock()
	if t.closed {
		err = t.closeErr
		t.mu.Unlock()
		return err
	}
	t.closed = true
	t.closeErr = errors.New("dflash2: target retirement incomplete")
	t.mu.Unlock()
	completed := false
	defer func() {
		t.mu.Lock()
		if completed {
			t.closeErr = err
		}
		failure := t.closeErr
		t.mu.Unlock()
		if failure != nil {
			t.domain.quarantine(failure, t)
		}
	}()
	if t.m != nil {
		if err = t.m.CloseWeights(); err != nil {
			completed = true
			return err
		}
	}
	if t.source != nil {
		err = t.source.Close()
	}
	completed = true
	return err
}

func (t *DFlash2Target) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	if t.closed {
		err := t.closeErr
		t.mu.Unlock()
		return err
	}
	busy := len(t.leases) != 0
	t.mu.Unlock()
	if busy {
		return ErrDFlash2Busy
	}
	release, err := t.domain.acquire(context.Background())
	if err != nil {
		return err
	}
	defer release()
	return t.retire()
}

type dflash2Lease struct {
	target   *DFlash2Target
	session  *model.Session
	request  dflash2RequestOwner
	closed   bool
	closeErr error
}

func (t *DFlash2Target) newRequest(ctx context.Context) (lease *dflash2Lease, err error) {
	if err = t.ValidateBinding(t.m, t.tok, t.backend, true, false); err != nil {
		return nil, err
	}
	l := &dflash2Lease{target: t}
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil, errors.New("dflash2: target closed")
	}
	t.leases[l] = struct{}{}
	t.mu.Unlock()
	completed := false
	defer func() {
		if !completed {
			t.domain.quarantine(errors.New("dflash2: request construction interrupted"), l)
		}
		if completed && err != nil {
			err = errors.Join(err, dflash2CleanupCall("failed request close", l.close))
		}
	}()
	if t.factory != nil {
		err = t.factory(ctx, l)
	} else {
		l.session, err = newDFlash2TargetSession(t.m, t.backend)
		if err == nil {
			var q *model.Qwen38DFlash2Request
			q, err = model.NewQwen38DFlash2RequestWithOptions(ctx, l.session, t.spec.SidecarPath, t.options.Request)
			if err == nil {
				l.request = dflash2NativeRequest{q}
				l.session = nil
			}
		}
	}
	if err == nil && l.request == nil {
		err = errors.New("dflash2: constructor returned no request")
	}
	completed = true
	if err != nil {
		return nil, err
	}
	return l, nil
}

func (l *dflash2Lease) close() (err error) {
	if l == nil {
		return nil
	}
	if l.closed {
		return l.closeErr
	}
	l.closed = true
	l.closeErr = errors.New("dflash2: request retirement incomplete")
	completed := false
	defer func() {
		if completed {
			l.closeErr = err
		}
		if l.closeErr != nil {
			l.target.domain.quarantine(l.closeErr, l)
			return
		}
		l.target.mu.Lock()
		delete(l.target.leases, l)
		l.target.mu.Unlock()
	}()
	if l.request != nil {
		err = l.request.Close()
	} else if l.session != nil {
		l.session.Close()
	}
	completed = true
	return err
}

// String contains only pinned provenance, never local checkpoint paths.
func (t *DFlash2Target) String() string {
	if t == nil {
		return "dflash2:unbound"
	}
	return fmt.Sprintf("dflash2:%s:%s", t.spec.ExpectedShardSetSHA256, t.spec.TokenizerID)
}

func dflash2LoadOptions(dtype ggufload.TensorType) ([]ggufload.Q4KLoadOption, error) {
	opts := []ggufload.Q4KLoadOption{ggufload.WithMTPRetention(false), ggufload.WithNativeLinearAttnRows(false), ggufload.WithDenseKQuantResident(false)}
	switch dtype {
	case ggufload.TensorQ4_K:
		return append(opts, ggufload.WithQ4KEmbeddingResident(true)), nil
	case ggufload.TensorQ2_K:
		return append(opts, ggufload.WithQ2KEmbeddingResident(true)), nil
	default:
		return nil, fmt.Errorf("dflash2: unsupported packed embedding dtype %s", dtype)
	}
}
