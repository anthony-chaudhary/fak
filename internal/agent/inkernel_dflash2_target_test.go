package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type dflash2GateBackend struct {
	compute.Backend
	owner *sync.Mutex
}

func (b *dflash2GateBackend) ForwardExecutionOwner() *sync.Mutex { return b.owner }

func dflash2ServingFixture(t *testing.T) (*InKernelPlanner, *DFlash2Target) {
	t.Helper()
	tok, err := tokenizer.FromGGML([]string{"a", "b", "c", "<|im_end|>"}, nil, []int32{1, 1, 1, 3}, "qwen2")
	if err != nil {
		t.Fatal(err)
	}
	be := &dflash2GateBackend{owner: new(sync.Mutex)}
	domain, err := registerDFlash2Domain(be)
	if err != nil {
		t.Fatal(err)
	}
	m := &model.Model{Cfg: model.Config{EOSTokenID: 3, VocabSize: 4}}
	target := &DFlash2Target{m: m, tok: tok, backend: be, domain: domain, source: &ggufload.WeightSource{},
		spec:   DFlash2TargetSpec{TokenizerID: tok.Identity(), RendererID: inKernelPromptRendererID(m.Cfg, SampleParams{})},
		leases: make(map[*dflash2Lease]struct{})}
	target.fingerprint, err = dflash2TargetFingerprint(m, tok)
	if err != nil {
		t.Fatal(err)
	}
	target.stops = dflash2StopIDs(tok, m.Cfg)
	// This fixture admits only the private lifecycle/driver seam. It never calls
	// the public artifact constructor and makes no stock-weight or math claim.
	return &InKernelPlanner{m: m, tok: tok, backend: be, q4k: true, dflash2: target}, target
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TargetIdentityRefusals(t *testing.T) {
	p, target := dflash2ServingFixture(t)
	if err := target.ValidateBinding(p.m, p.tok, p.backend, true, false); err != nil {
		t.Fatal(err)
	}
	if err := target.ValidateBinding(&model.Model{}, p.tok, p.backend, true, false); err == nil {
		t.Fatal("different model admitted")
	}
	if err := target.ValidateBinding(p.m, p.tok, p.backend, false, false); err == nil {
		t.Fatal("nonquantized target admitted")
	}
	if err := target.ValidateBinding(p.m, p.tok, p.backend, true, true); err == nil {
		t.Fatal("native Metal admitted")
	}
	old := p.m.Cfg.EOSTokenID
	p.m.Cfg.EOSTokenID = 0
	if err := target.ValidateBinding(p.m, p.tok, p.backend, true, false); err == nil {
		t.Fatal("EOS mutation escaped tokenizer identity")
	}
	p.m.Cfg.EOSTokenID = old
	bos := 0
	p.m.Cfg.BOSTokenID = &bos
	if err := target.ValidateBinding(p.m, p.tok, p.backend, true, false); err == nil {
		t.Fatal("BOS mutation escaped binding")
	}
	p.m.Cfg.BOSTokenID = nil
	p.backend.(*dflash2GateBackend).owner = new(sync.Mutex)
	if err := target.ValidateBinding(p.m, p.tok, p.backend, true, false); err == nil {
		t.Fatal("mutable owner escaped registration")
	}
	if _, err := p.acquireDFlash2Domain(context.Background()); err == nil {
		t.Fatal("mutable alias gate admitted")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TargetStopContract(t *testing.T) {
	p, _ := dflash2ServingFixture(t)
	cfg := p.m.Cfg
	cfg.EOSTokenID = 0
	if !dflash2StopIDs(p.tok, cfg)[0] {
		t.Fatal("authenticated zero EOS omitted")
	}
	cfg.EOSTokenIDs = []int{1, 2}
	got := dflash2StopIDs(p.tok, cfg)
	// StopIDs includes the legacy scalar; the exact contract pins that union as
	// well as the authoritative EOS list, rather than relying on tokenizer hash.
	if !got[1] || !got[2] || !got[3] {
		t.Fatalf("stop IDs %v", got)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TargetConstructorRefusesBeforeIO(t *testing.T) {
	spec := dflash2ValidSpec()
	calls := 0
	poison := func(context.Context, []ggufload.ShardIdentity, string) (*ggufload.WeightSource, error) {
		calls++
		panic("unexpected opener")
	}
	for _, bad := range []DFlash2TargetSpec{{}, {Shards: spec.Shards, ExpectedShardSetSHA256: spec.ExpectedShardSetSHA256, TokenizerID: spec.TokenizerID, RendererID: spec.RendererID, SidecarPath: "relative"}} {
		if _, err := newDFlash2Target(context.Background(), bad, &dflash2GateBackend{owner: new(sync.Mutex)}, poison); !errors.Is(err, ErrDFlash2Identity) {
			t.Fatalf("identity refusal %v", err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := newDFlash2Target(ctx, spec, &dflash2GateBackend{owner: new(sync.Mutex)}, poison); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel %v", err)
	}
	if _, err := newDFlash2Target(context.Background(), spec, &dflash2GateBackend{}, poison); err == nil {
		t.Fatal("nil owner admitted")
	}
	if calls != 0 {
		t.Fatalf("early refusal performed %d opens", calls)
	}
}

func dflash2ValidSpec() DFlash2TargetSpec {
	return DFlash2TargetSpec{Shards: []ggufload.ShardIdentity{{Path: "/not-opened.gguf", Size: 1, RawFileSHA256: strings.Repeat("0", 64)}},
		ExpectedShardSetSHA256: strings.Repeat("0", 64), TokenizerID: "sha256:" + strings.Repeat("0", 64),
		RendererID: "qwen-chatml-v1:no-thinking", SidecarPath: "/not-opened.safetensors"}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2DomainAliasesAndBusy(t *testing.T) {
	p, target := dflash2ServingFixture(t)
	alias := &InKernelPlanner{backend: &dflash2GateBackend{owner: target.domain.owner}}
	release, err := p.acquireDFlash2Domain(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := alias.acquireDFlash2Domain(context.Background()); !errors.Is(err, ErrDFlash2Busy) {
		t.Fatalf("alias concurrency %v", err)
	}
	if !target.domain.owner.TryLock() {
		t.Fatal("request gate held backend execution mutex")
	}
	target.domain.owner.Unlock()
	release()
	errCause := errors.New("incomplete close")
	target.domain.quarantine(errCause, target)
	if _, err := alias.Complete(context.Background(), nil, nil); !errors.Is(err, ErrDFlash2Quarantined) || !errors.Is(err, errCause) {
		t.Fatalf("ordinary alias escaped quarantine: %v", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2LeaseIncompleteClose(t *testing.T) {
	for _, fault := range []string{"error", "panic", "nil-panic", "goexit"} {
		t.Run(fault, func(t *testing.T) {
			if fault == "nil-panic" {
				t.Setenv("GODEBUG", "panicnil=1")
			}
			_, target := dflash2ServingFixture(t)
			r := &dflash2ScriptRequest{closeFault: func() error {
				switch fault {
				case "error":
					return errors.New("close sentinel")
				case "panic":
					panic("close sentinel")
				case "nil-panic":
					panic(nil)
				default:
					runtime.Goexit()
				}
				return nil
			}}
			l := &dflash2Lease{target: target, request: r}
			target.leases[l] = struct{}{}
			finished := make(chan struct{})
			go func() { defer close(finished); defer func() { _ = recover() }(); _ = l.close() }()
			<-finished
			if l.closeErr == nil || l.close() == nil || !l.closed || r.closes != 1 {
				t.Fatalf("false retirement: %+v closes=%d", l, r.closes)
			}
			if _, ok := target.leases[l]; !ok {
				t.Fatal("lost incomplete owner")
			}
			if _, err := target.domain.acquire(context.Background()); !errors.Is(err, ErrDFlash2Quarantined) {
				t.Fatalf("quarantine %v", err)
			}
			if err := target.Close(); !errors.Is(err, ErrDFlash2Busy) {
				t.Fatalf("active owner retirement %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TargetConstructorFailureRetainsOwner(t *testing.T) {
	_, target := dflash2ServingFixture(t)
	cause, cleanup := errors.New("factory"), errors.New("cleanup")
	target.factory = func(_ context.Context, l *dflash2Lease) error {
		l.request = &dflash2ScriptRequest{closeFault: func() error { return cleanup }}
		return cause
	}
	if _, err := target.newRequest(context.Background()); !errors.Is(err, cause) || !errors.Is(err, cleanup) {
		t.Fatalf("failure %v", err)
	}
	if len(target.leases) != 1 || target.domain.failure == nil {
		t.Fatal("failed constructor lost ownership")
	}
}

type dflash2CloseFunc func() error

func (f dflash2CloseFunc) Close() error { return f() }

// fak-test:runtime fast est=1s lane=default
func TestDFlash2TargetRetirementFailureIsDurable(t *testing.T) {
	for _, fault := range []string{"error", "panic", "nil-panic", "goexit"} {
		t.Run(fault, func(t *testing.T) {
			if fault == "nil-panic" {
				t.Setenv("GODEBUG", "panicnil=1")
			}
			_, target := dflash2ServingFixture(t)
			calls := 0
			target.m.SetWeightCloser(dflash2CloseFunc(func() error {
				calls++
				switch fault {
				case "error":
					return errors.New("weight close")
				case "panic":
					panic("weight close")
				case "nil-panic":
					panic(nil)
				default:
					runtime.Goexit()
				}
				return nil
			}))
			done := make(chan struct{})
			go func() { defer close(done); defer func() { _ = recover() }(); _ = target.Close() }()
			<-done
			if !target.closed || target.closeErr == nil || target.Close() == nil || calls != 1 || target.source == nil {
				t.Fatalf("lost retirement state calls=%d error=%v", calls, target.closeErr)
			}
			if target.domain.failure == nil {
				t.Fatal("failed retirement did not quarantine")
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2ConstructorInterruptionRetainsAttempt(t *testing.T) {
	for _, fault := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(fault, func(t *testing.T) {
			if fault == "nil-panic" {
				t.Setenv("GODEBUG", "panicnil=1")
			}
			_, target := dflash2ServingFixture(t)
			owner := &dflash2ScriptRequest{}
			target.factory = func(_ context.Context, l *dflash2Lease) error {
				l.request = owner
				switch fault {
				case "panic":
					panic("constructor")
				case "nil-panic":
					panic(nil)
				default:
					runtime.Goexit()
				}
				return nil
			}
			done := make(chan struct{})
			go func() {
				defer close(done)
				defer func() { _ = recover() }()
				_, _ = target.newRequest(context.Background())
			}()
			<-done
			if len(target.leases) != 1 || target.domain.failure == nil || len(target.domain.retained) == 0 {
				t.Fatal("interrupted constructor lost ownership")
			}
			if owner.closes != 0 {
				t.Fatal("unknown construction state was blindly retired")
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2PackedLoadOptionsAreExclusive(t *testing.T) {
	for _, dtype := range []ggufload.TensorType{ggufload.TensorQ2_K, ggufload.TensorQ4_K} {
		opts, err := dflash2LoadOptions(dtype)
		if err != nil {
			t.Fatal(err)
		}
		got := ggufload.ApplyQ4KLoadOptions(append([]ggufload.Q4KLoadOption{ggufload.WithDenseKQuantResident(true), ggufload.WithMTPRetention(true)}, opts...))
		if got.Q2KEmbeddingResident != (dtype == ggufload.TensorQ2_K) || got.Q4KEmbeddingResident != (dtype == ggufload.TensorQ4_K) || got.MTPRetention || got.DenseKQuantResident {
			t.Fatalf("dtype=%s effects=%+v", dtype, got)
		}
	}
	for _, dtype := range []ggufload.TensorType{ggufload.TensorF32, ggufload.TensorQ6_K, ggufload.TensorQ8_0} {
		if _, err := dflash2LoadOptions(dtype); err == nil {
			t.Fatalf("unsupported dtype %s admitted", dtype)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2SourceAcquisitionUnverifiedRetirement(t *testing.T) {
	for _, fault := range []string{"error", "panic", "nil-panic", "goexit"} {
		t.Run(fault, func(t *testing.T) {
			if fault == "nil-panic" {
				t.Setenv("GODEBUG", "panicnil=1")
			}
			path := filepath.Join(t.TempDir(), "reader")
			if err := os.WriteFile(path, []byte("retained reader"), 0600); err != nil {
				t.Fatal(err)
			}
			primary, cleanup := errors.New("open failed after acquisition"), errors.New("reader close reported failure")
			spec := dflash2ValidSpec()
			original := spec.Shards[0]
			be := &dflash2GateBackend{owner: new(sync.Mutex)}
			opens, closes := 0, 0
			var reader *os.File
			var result error
			open := func(_ context.Context, shards []ggufload.ShardIdentity, _ string) (*ggufload.WeightSource, error) {
				opens++
				var err error
				reader, err = os.Open(path)
				if err != nil {
					return nil, err
				}
				// Mutating the opener's copy cannot change the caller's trusted manifest.
				shards[0].Path = "/mutated-copy"
				if err = reader.Close(); err != nil {
					return nil, err
				}
				closes++
				switch fault {
				case "error":
					return nil, errors.Join(primary, cleanup)
				case "panic":
					panic(primary)
				case "nil-panic":
					panic(nil)
				default:
					runtime.Goexit()
				}
				return nil, nil
			}
			done := make(chan struct{})
			go func() {
				defer close(done)
				defer func() { _ = recover() }()
				_, result = newDFlash2Target(context.Background(), spec, be, open)
			}()
			<-done
			if opens != 1 || closes != 1 || spec.Shards[0] != original {
				t.Fatalf("open=%d close=%d manifest=%v", opens, closes, spec.Shards)
			}
			if _, err := reader.Read(make([]byte, 1)); !errors.Is(err, os.ErrClosed) {
				t.Fatalf("physical reader closure %v", err)
			}
			if fault == "error" && (!errors.Is(result, primary) || !errors.Is(result, cleanup)) {
				t.Fatalf("acquisition causes %v", result)
			}
			d, err := registeredDFlash2Domain(be)
			if err != nil || d == nil {
				t.Fatal(err)
			}
			if d.failure == nil || len(d.retained) == 0 {
				t.Fatal("unverified acquisition was declared retired")
			}
			if _, err := (&InKernelPlanner{backend: be}).Complete(context.Background(), nil, nil); !errors.Is(err, ErrDFlash2Quarantined) {
				t.Fatalf("ordinary alias escaped: %v", err)
			}
		})
	}
}

type dflash2UncomparableAlias struct {
	compute.Backend
	owner *sync.Mutex
	extra []byte
}

func (b dflash2UncomparableAlias) ForwardExecutionOwner() *sync.Mutex { return b.owner }

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2NonComparableAliasRefusesComplete(t *testing.T) {
	p, target := dflash2ServingFixture(t)
	alias := &InKernelPlanner{backend: dflash2UncomparableAlias{owner: target.domain.owner, extra: []byte{1}}}
	release, err := p.acquireDFlash2Domain(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := alias.Complete(context.Background(), nil, nil); err == nil || !strings.Contains(err.Error(), "non-comparable") {
		t.Fatalf("busy alias %v", err)
	}
	release()
	target.domain.quarantine(errors.New("incomplete"), target)
	if _, err := alias.Complete(context.Background(), nil, nil); err == nil || !strings.Contains(err.Error(), "non-comparable") {
		t.Fatalf("quarantined alias %v", err)
	}
}

// dflash2AuthenticatedProbe writes an independently specified tiny GGUF. The
// manifest hashes the constructed bytes before opening them; no stock geometry
// or loaded target is fabricated by this fixture.
func dflash2AuthenticatedProbe(t *testing.T) DFlash2TargetSpec {
	t.Helper()
	t.Setenv("FAK_GGUF_MMAP", "0")
	var b bytes.Buffer
	put := func(v any) {
		t.Helper()
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	str := func(s string) { put(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	put(uint32(3))
	put(uint64(1))
	put(uint64(7))
	str("general.architecture")
	put(uint32(8))
	str("llama")
	for _, kv := range []struct {
		k string
		v uint32
	}{
		{"llama.embedding_length", 32}, {"llama.block_count", 1}, {"llama.attention.head_count", 1}, {"llama.feed_forward_length", 64},
	} {
		str(kv.k)
		put(uint32(4))
		put(kv.v)
	}
	str("llama.attention.layer_norm_rms_epsilon")
	put(uint32(6))
	put(float32(1e-5))
	str("tokenizer.ggml.tokens")
	put(uint32(9))
	put(uint32(8))
	put(uint64(2))
	str("a")
	str("b")
	str("probe.weight")
	put(uint32(1))
	put(uint64(1))
	put(uint32(ggufload.TensorF32))
	put(uint64(0))
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	put(float32(7))
	raw := b.Bytes()
	digest := sha256.Sum256(raw)
	path := filepath.Join(t.TempDir(), "probe.gguf")
	spec := dflash2ValidSpec()
	spec.Shards = []ggufload.ShardIdentity{{Path: path, Size: int64(len(raw)), RawFileSHA256: hex.EncodeToString(digest[:])}}
	encoded, err := json.Marshal(spec.Shards)
	if err != nil {
		t.Fatal(err)
	}
	set := sha256.Sum256(encoded)
	spec.ExpectedShardSetSHA256 = hex.EncodeToString(set[:])
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	return spec
}

func dflash2RequireFileReader(t *testing.T, source *ggufload.WeightSource) {
	t.Helper()
	// The public mmap gate is process-memoized. Refuse an already-enabled mapping
	// before attempting any post-close probe, rather than dereferencing stale
	// mmap data. This is read-only test inspection, not a production access seam.
	value := reflect.ValueOf(source).Elem()
	isFile := func(reader reflect.Value) bool {
		return reader.IsValid() && reader.Kind() == reflect.Interface && !reader.IsNil() && reader.Elem().Type() == reflect.TypeOf((*os.File)(nil))
	}
	if !isFile(value.FieldByName("r")) {
		t.Fatal("reader-retirement control requires the explicitly selected os.File reader")
	}
	routes := value.FieldByName("readerFor")
	if !routes.IsValid() || routes.Len() != 1 || !isFile(routes.Index(0)) {
		t.Fatal("probe tensor did not route to the guarded os.File reader")
	}
}

func dflash2RequireProbe(t *testing.T, source *ggufload.WeightSource) {
	t.Helper()
	got, _, err := source.TensorF32("probe.weight")
	if err != nil || len(got) != 1 || got[0] != 7 {
		t.Fatalf("real probe=%v err=%v", got, err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2AuthenticatedSourceEntersConstructor(t *testing.T) {
	spec := dflash2AuthenticatedProbe(t)
	var source *ggufload.WeightSource
	opens := 0
	open := func(ctx context.Context, shards []ggufload.ShardIdentity, set string) (*ggufload.WeightSource, error) {
		opens++
		var err error
		source, err = ggufload.OpenWeightsAuthenticated(ctx, shards, set)
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() {
			if err := source.Close(); err != nil {
				t.Error(err)
			}
		})
		dflash2RequireFileReader(t, source)
		dflash2RequireProbe(t, source)
		return source, nil
	}
	be := &dflash2GateBackend{owner: new(sync.Mutex)}
	target, err := newDFlash2Target(context.Background(), spec, be, open)
	if target != nil || err == nil || !strings.Contains(err.Error(), "exact dense Qwen3.8-27B target geometry") || opens != 1 {
		t.Fatalf("target=%v error=%v opens=%d", target, err, opens)
	}
	if _, _, err := source.TensorF32("probe.weight"); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("refused source was not retired: %v", err)
	}
	domain, err := registeredDFlash2Domain(be)
	if err != nil || domain == nil {
		t.Fatal(err)
	}
	release, err := domain.acquire(context.Background())
	if err != nil {
		t.Fatalf("verified retirement quarantined: %v", err)
	}
	release()
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2ModelRetiresBeforeAuthenticatedReader(t *testing.T) {
	spec := dflash2AuthenticatedProbe(t)
	source, err := ggufload.OpenWeightsAuthenticated(context.Background(), spec.Shards, spec.ExpectedShardSetSHA256)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := source.Close(); err != nil {
			t.Error(err)
		}
	})
	dflash2RequireFileReader(t, source)
	dflash2RequireProbe(t, source)
	_, target := dflash2ServingFixture(t)
	target.source = source
	calls := 0
	target.m.SetWeightCloser(dflash2CloseFunc(func() error {
		calls++
		got, _, err := source.TensorF32("probe.weight")
		if err != nil || len(got) != 1 || got[0] != 7 {
			return errors.New("reader retired before model backing")
		}
		return nil
	}))
	if err := target.Close(); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("model retirement calls=%d", calls)
	}
	if _, _, err := source.TensorF32("probe.weight"); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("reader still available after retirement: %v", err)
	}
	if err := target.Close(); err != nil || calls != 1 {
		t.Fatalf("repeated close error=%v calls=%d", err, calls)
	}
}
