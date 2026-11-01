package modelengine

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/fabricmap"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// NativePDFabricMaterialize is the receiver-local dense KV realization step.
// It describes ToKVCache, not a remote memory address or device kernel.
const NativePDFabricMaterialize = "native-pd-materialize"

// NativePDFabricReceiver binds one configured decode worker to a consumer plan.
// PlanID must match the ready link's kv.plan_id label in the provider snapshot.
type NativePDFabricReceiver struct {
	Landing string
	Ready   string
	PlanID  string
	Backend model.KVTransferBackend
}

// NativePDFabricConfig is receiver-owned configuration, never request routing
// authority. Providers describe capabilities; the separate Authorizer must allow
// both copy hops. Only the existing KV transport plus local ToKVCache is executed.
type NativePDFabricConfig struct {
	Provider   fabricmap.Provider
	ProviderID string
	Source     string
	Receivers  map[string]NativePDFabricReceiver
	Access     fabricmap.Access
	Authorizer fabricmap.Authorizer
}

type nativePDFabricState struct {
	mu        sync.Mutex // held through execution, generation recheck and scheduler admission
	config    NativePDFabricConfig
	snapshots fabricmap.SnapshotSet

	// Separate from the receiver transaction: scrapes must not wait behind
	// the very admission/transfer stall they are observing.
	timingMu sync.Mutex
	timings  nativePDFabricMetricValues
}

// WithNativePDReceiverFabric enables receiver-ready admission for this cluster.
// Nil leaves the legacy same-layout path unchanged. Invalid non-nil configuration
// fails closed at Admit; it never silently falls back to a one-hop transfer.
func WithNativePDReceiverFabric(config *NativePDFabricConfig) NativePDOption {
	return func(c *NativePDCluster) {
		if config == nil {
			c.fabric = nil
			return
		}
		copyConfig := *config
		copyConfig.Receivers = make(map[string]NativePDFabricReceiver, len(config.Receivers))
		for name, receiver := range config.Receivers {
			copyConfig.Receivers[name] = receiver
		}
		copyConfig.Access.Data = make(map[string]string, len(config.Access.Data))
		for key, value := range config.Access.Data {
			copyConfig.Access.Data[key] = value
		}
		c.fabric = &nativePDFabricState{config: copyConfig}
	}
}

func (f *nativePDFabricState) backend(worker string, requested model.KVTransferBackend) (model.KVTransferBackend, error) {
	receiver, ok := f.config.Receivers[worker]
	if !ok || receiver.Backend == "" || (requested != "" && requested != receiver.Backend) {
		return "", fmt.Errorf("%w: receiver backend is not configured for %s", cachemeta.ErrKVFabricNotReady, worker)
	}
	return receiver.Backend, nil
}

// snapshot is called only while mu is held. Retaining SnapshotSet makes a
// later rollback or equal-generation content change a refusal, not a new plan.
func (f *nativePDFabricState) snapshot(ctx context.Context) (fabricmap.Snapshot, fabricmap.Graph, error) {
	if err := ctx.Err(); err != nil {
		return fabricmap.Snapshot{}, fabricmap.Graph{}, err
	}
	if f.config.Provider == nil || f.config.ProviderID == "" {
		return fabricmap.Snapshot{}, fabricmap.Graph{}, errors.New("native P/D fabric provider is required")
	}
	snapshot, err := f.config.Provider.Snapshot(ctx)
	if err != nil {
		return snapshot, fabricmap.Graph{}, err
	}
	if err := ctx.Err(); err != nil {
		return snapshot, fabricmap.Graph{}, err
	}
	if snapshot.Provider != f.config.ProviderID || snapshot.ValidUntil.IsZero() {
		return snapshot, fabricmap.Graph{}, errors.New("native P/D fabric provider identity or expiry is missing")
	}
	now := time.Now()
	if err := f.snapshots.Update(snapshot, now); err != nil {
		return snapshot, fabricmap.Graph{}, err
	}
	graph, err := f.snapshots.Graph(now)
	return snapshot, graph, err
}

func (f *nativePDFabricState) transfer(ctx context.Context, worker string, cfg model.Config, src *model.PagedKV, descriptor cachemeta.KVTransfer, transport model.KVTransport, timing *nativePDFabricTiming) (received model.PagedKVTransferReceipt, cache *model.KVCache, err error) {
	receiver := f.config.Receivers[worker]
	proof := &cachemeta.KVFabricReceipt{
		PlanID: receiver.PlanID, Provider: f.config.ProviderID, Landing: receiver.Landing,
		Route:     fabricmap.Route{From: f.config.Source, To: receiver.Ready},
		Execution: fabricmap.ExecutionReceipt{State: fabricmap.ExecutionFailed, Recovery: fabricmap.RecoveryAlternateRoute},
	}
	received.Transfer = descriptor
	defer func() {
		// Reply metadata cannot replace the receiver's trusted descriptor,
		// including on partial/error paths. KV ownership stays on received.KV.
		wire := received.Transfer
		received.Transfer = descriptor
		received.Transfer.Backend = string(receiver.Backend)
		received.Transfer.BytesMoved = wire.BytesMoved
		received.Transfer.Outcome = wire.Outcome
		received.Transfer.Fabric = proof
		proof.SpanDigest = descriptor.SpanDigest
		if err != nil {
			received.Transfer.Outcome = cachemeta.KVTransferFault
			received.Transfer.FaultReason = "consumer-ready fabric execution failed"
		}
		received.Entry = cachemeta.FromKVTransfer(received.Transfer)
	}()
	expected, logicalBytes, err := nativePDFabricPagedDigest(src, cfg)
	if err != nil {
		return received, nil, err
	}
	descriptor.SpanDigest = expected
	received.Transfer = descriptor
	snapshot, graph, err := f.snapshot(ctx)
	if err != nil {
		return received, nil, err
	}
	route, err := graph.Plan(fabricmap.Request{From: f.config.Source, To: receiver.Ready, Bytes: logicalBytes})
	if err != nil {
		return received, nil, err
	}
	if len(route.Links) != 2 || receiver.PlanID == "" || receiver.Landing == "" ||
		route.Links[0].To != receiver.Landing || route.Links[1].From != receiver.Landing ||
		route.Links[0].Transport != string(receiver.Backend) ||
		route.Links[1].Transport != NativePDFabricMaterialize ||
		route.Links[1].Labels["kv.plan_id"] != receiver.PlanID {
		return received, nil, fmt.Errorf("%w: unsupported consumer realization", cachemeta.ErrKVFabricNotReady)
	}
	access := f.config.Access
	if access.Operation != fabricmap.OperationCopy {
		return received, nil, errors.New("native P/D fabric requires explicit copy authority")
	}
	access.Data = make(map[string]string, len(f.config.Access.Data)+2)
	for key, value := range f.config.Access.Data {
		access.Data[key] = value
	}
	access.Data["kv.content_digest"] = expected
	access.Data["kv.plan_id"] = receiver.PlanID
	authorized, err := fabricmap.AuthorizeRoute(route, access, f.config.Authorizer)
	if err != nil {
		return received, nil, err
	}
	proof = &cachemeta.KVFabricReceipt{
		PlanID: receiver.PlanID, Provider: snapshot.Provider, Generation: snapshot.Generation,
		ValidUntil: snapshot.ValidUntil, Landing: receiver.Landing, ContentDigest: expected,
		LogicalBytes: logicalBytes, Route: authorized.Route,
	}
	descriptor.Fabric = proof
	execution, execErr := fabricmap.Execute(ctx, authorized, logicalBytes, fabricmap.ExecutorFunc(func(ctx context.Context, request fabricmap.AuthorizationRequest, bytes uint64) (fabricmap.HopReceipt, error) {
		if err := ctx.Err(); err != nil {
			return fabricmap.HopReceipt{}, err
		}
		start := time.Now()
		if !start.Before(proof.ValidUntil) {
			return fabricmap.HopReceipt{}, fmt.Errorf("%w: consumer plan expired before hop", cachemeta.ErrKVFabricNotReady)
		}
		switch request.HopIndex {
		case 0:
			timing.next(nativePDFabricLanding)
		case 1:
			timing.next(nativePDFabricMaterializing)
		}
		defer timing.next(nativePDFabricPrepare)
		var actual string
		var count uint64
		var hopErr error
		switch request.HopIndex {
		case 0:
			received, hopErr = transport.Send(src, descriptor, 0, src.Len())
			if hopErr == nil {
				if received.KV == nil || received.Transfer.Outcome != cachemeta.KVTransferOK || received.Transfer.SpanDigest != expected || received.Transfer.Tokens != int64(src.Len()) || received.Transfer.SerializerID != model.PagedKVTransferSerializerID || received.Transfer.Backend != string(receiver.Backend) || len(received.Positions) != src.Len() {
					hopErr = errors.New("native P/D transport returned an incomplete receiver span")
				} else {
					for i, position := range received.Positions {
						if position != i {
							hopErr = errors.New("native P/D receiver positions are not the complete prefix")
							break
						}
					}
				}
			}
			if hopErr == nil {
				actual, count, hopErr = nativePDFabricPagedDigest(received.KV, cfg)
			}
		case 1:
			cache = received.KV.ToKVCache(cfg)
			actual, count, hopErr = nativePDFabricCacheDigest(cache, cfg)
		default:
			hopErr = errors.New("unsupported native P/D fabric hop")
		}
		if hopErr != nil {
			return fabricmap.HopReceipt{}, hopErr
		}
		if actual != expected || count != bytes {
			return fabricmap.HopReceipt{}, errors.New("native P/D receiver content witness mismatch")
		}
		link := request.Link
		return fabricmap.HopReceipt{
			HopIndex: request.HopIndex, LinkID: link.ID, From: link.From, To: link.To, Transport: link.Transport,
			Bytes: bytes, StartedAt: start, CompletedAt: time.Now(), State: fabricmap.HopComplete,
			Integrity: fabricmap.Integrity{Algorithm: cachemeta.KVFabricIntegrity, Digest: actual},
		}, nil
	}))
	proof.Execution = execution
	if execErr == nil {
		proof.ReadyAt = time.Now()
	}
	return received, cache, execErr
}

// admitReady must run immediately before AdmitImported while the same selection
// lock remains held. A newer valid plan is not evidence for an older execution.
func (f *nativePDFabricState) admitReady(ctx context.Context, transfer cachemeta.KVTransfer) error {
	proof := transfer.Fabric
	snapshot, _, err := f.snapshot(ctx)
	if err != nil {
		return err
	}
	if proof == nil || proof.Provider != snapshot.Provider || proof.Generation != snapshot.Generation || !proof.ValidUntil.Equal(snapshot.ValidUntil) {
		return fmt.Errorf("%w: consumer plan changed during transfer", cachemeta.ErrKVFabricNotReady)
	}
	return proof.Validate(transfer, time.Now())
}

// The witness covers logical prefix positions and framed layer/plane extents,
// not unused page tails. Both receiver stages hash their actual rows independently.
func nativePDFabricDigest(cfg model.Config, tokens int, row func(layer, plane int) []float32) (string, uint64, error) {
	if tokens <= 0 || cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return "", 0, errors.New("invalid native P/D fabric geometry")
	}
	count := uint64(tokens)
	for _, factor := range []int{cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, 3, 4} {
		if uint64(factor) > ^uint64(0)/count {
			return "", 0, errors.New("native P/D fabric logical size overflows")
		}
		count *= uint64(factor)
	}
	width := uint64(tokens) * uint64(cfg.NumKVHeads) * uint64(cfg.HeadDim)
	if width > uint64(^uint(0)>>1) {
		return "", 0, errors.New("native P/D fabric row extent overflows")
	}
	h := sha256.New()
	_, _ = h.Write([]byte("fak.native-pd.logical-dense-f32/v1\x00"))
	for _, value := range []uint64{uint64(tokens), uint64(cfg.NumLayers), uint64(cfg.NumKVHeads), uint64(cfg.HeadDim), 3} {
		nativePDFabricHashUint(h, value)
	}
	for position := 0; position < tokens; position++ {
		nativePDFabricHashUint(h, uint64(position))
	}
	var bits [4]byte
	for layer := 0; layer < cfg.NumLayers; layer++ {
		for plane := 0; plane < 3; plane++ {
			values := row(layer, plane)
			if uint64(len(values)) != width {
				return "", 0, errors.New("native P/D fabric has incomplete logical KV rows")
			}
			nativePDFabricHashUint(h, uint64(layer))
			nativePDFabricHashUint(h, uint64(plane))
			nativePDFabricHashUint(h, uint64(len(values)))
			for _, value := range values {
				binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
				_, _ = h.Write(bits[:])
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil)), count, nil
}

func nativePDFabricHashUint(h hash.Hash, value uint64) {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	_, _ = h.Write(encoded[:])
}

func nativePDFabricPagedDigest(seq *model.PagedKV, cfg model.Config) (string, uint64, error) {
	if seq == nil {
		return "", 0, errors.New("native P/D fabric has no paged span")
	}
	return nativePDFabricDigest(cfg, seq.Len(), func(layer, plane int) []float32 {
		switch plane {
		case 0:
			return seq.GatherK(layer)
		case 1:
			return seq.GatherKraw(layer)
		default:
			return seq.GatherV(layer)
		}
	})
}

func nativePDFabricCacheDigest(cache *model.KVCache, cfg model.Config) (string, uint64, error) {
	if cache == nil || len(cache.K) != cfg.NumLayers || len(cache.Kraw) != cfg.NumLayers || len(cache.V) != cfg.NumLayers {
		return "", 0, errors.New("native P/D fabric has no complete materialization")
	}
	return nativePDFabricDigest(cfg, cache.Len(), func(layer, plane int) []float32 {
		switch plane {
		case 0:
			return cache.K[layer]
		case 1:
			return cache.Kraw[layer]
		default:
			return cache.V[layer]
		}
	})
}

// Receiver telemetry is opt-in with WithNativePDReceiverFabric, not enabled on
// the legacy nil-config path. The five bins partition initialized selected Admit
// calls: preparation (including prefill, source witness and plan work), receiver
// mutex wait, landing (Send plus received-content validation), materialization
// (ToKVCache plus its content validation), and finalization (scheduler admission,
// registration and deferred cleanup). Early exits retain cleanup in their last
// reached phase. No phase is a hardware-kernel timer.
type nativePDFabricPhase uint8

const (
	nativePDFabricPrepare nativePDFabricPhase = iota
	nativePDFabricWait
	nativePDFabricLanding
	nativePDFabricMaterializing
	nativePDFabricFinalize
	nativePDFabricPhaseCount
)

var nativePDFabricPhaseNames = [nativePDFabricPhaseCount]string{
	"prepare", "receiver_wait", "landing", "materialize", "finalize",
}

type nativePDFabricMetricValues struct {
	inflight     [nativePDFabricPhaseCount]int64
	count        [2]uint64
	seconds      [2]float64
	phaseCount   [2][nativePDFabricPhaseCount]uint64
	phaseSeconds [2][nativePDFabricPhaseCount]float64
}

type nativePDFabricTiming struct {
	owner    *nativePDFabricState
	started  time.Time
	last     time.Time
	phase    nativePDFabricPhase
	seen     [nativePDFabricPhaseCount]bool
	seconds  [nativePDFabricPhaseCount]float64
	finished bool
}

func (f *nativePDFabricState) beginTiming() *nativePDFabricTiming {
	now := time.Now()
	timing := &nativePDFabricTiming{owner: f, started: now, last: now}
	timing.seen[nativePDFabricPrepare] = true
	f.timingMu.Lock()
	f.timings.inflight[nativePDFabricPrepare]++
	f.timingMu.Unlock()
	return timing
}

func (t *nativePDFabricTiming) next(phase nativePDFabricPhase) {
	if t == nil || t.finished || t.phase == phase {
		return
	}
	now := time.Now() // Sub retains Go's in-process monotonic clock.
	t.seconds[t.phase] += now.Sub(t.last).Seconds()
	t.owner.timingMu.Lock()
	t.owner.timings.inflight[t.phase]--
	t.owner.timings.inflight[phase]++
	t.owner.timingMu.Unlock()
	t.phase, t.last = phase, now
	t.seen[phase] = true
}

func (t *nativePDFabricTiming) finish(admitted bool) {
	if t == nil || t.finished {
		return
	}
	now := time.Now()
	t.seconds[t.phase] += now.Sub(t.last).Seconds()
	outcome := 0 // refused, including partial transfers and scheduler refusal
	if admitted {
		outcome = 1
	}
	t.owner.timingMu.Lock()
	t.owner.timings.inflight[t.phase]--
	t.owner.timings.count[outcome]++
	t.owner.timings.seconds[outcome] += now.Sub(t.started).Seconds()
	for phase, seen := range t.seen {
		if seen {
			t.owner.timings.phaseCount[outcome][phase]++
			t.owner.timings.phaseSeconds[outcome][phase] += t.seconds[phase]
		}
	}
	t.owner.timingMu.Unlock()
	t.finished = true
}

func (f *nativePDFabricState) writeMetrics(b *strings.Builder) {
	f.timingMu.Lock()
	values := f.timings
	f.timingMu.Unlock()
	b.WriteString("# HELP fak_native_pd_receiver_attempt_seconds Completed selected Admit wall time through owned cleanup, excluding final metric publication. Counts separate admitted and refused calls; pending calls appear only in phase_inflight.\n")
	b.WriteString("# TYPE fak_native_pd_receiver_attempt_seconds summary\n")
	b.WriteString("# HELP fak_native_pd_receiver_phase_seconds Non-overlapping selected Admit phase wall time in the same completed-call population. A phase count is calls that entered that phase, not hop or retry count.\n")
	b.WriteString("# TYPE fak_native_pd_receiver_phase_seconds summary\n")
	b.WriteString("# HELP fak_native_pd_receiver_phase_inflight Currently selected Admit calls in each phase, including blocked receiver mutex waits. Pending durations are not included in completed-call sums.\n")
	b.WriteString("# TYPE fak_native_pd_receiver_phase_inflight gauge\n")
	for outcome, name := range [2]string{"refused", "admitted"} {
		fmt.Fprintf(b, "fak_native_pd_receiver_attempt_seconds_count{outcome=%q} %d\n", name, values.count[outcome])
		fmt.Fprintf(b, "fak_native_pd_receiver_attempt_seconds_sum{outcome=%q} %.9f\n", name, values.seconds[outcome])
		for phase, phaseName := range nativePDFabricPhaseNames {
			fmt.Fprintf(b, "fak_native_pd_receiver_phase_seconds_count{phase=%q,outcome=%q} %d\n", phaseName, name, values.phaseCount[outcome][phase])
			fmt.Fprintf(b, "fak_native_pd_receiver_phase_seconds_sum{phase=%q,outcome=%q} %.9f\n", phaseName, name, values.phaseSeconds[outcome][phase])
		}
	}
	for phase, phaseName := range nativePDFabricPhaseNames {
		fmt.Fprintf(b, "fak_native_pd_receiver_phase_inflight{phase=%q} %d\n", phaseName, values.inflight[phase])
	}
}
