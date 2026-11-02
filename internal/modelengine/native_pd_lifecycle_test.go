package modelengine

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/fabricmap"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func waitNativePDClusterClosed(t *testing.T, cluster *NativePDCluster) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := cluster.WaitClosed(ctx); err != nil {
		t.Errorf("native P/D completion: %v", err)
	}
}

type pdLifecycleIndex struct {
	onObserve func()
	observes  int
}

func (*pdLifecycleIndex) Overlap(string, []string) int { return 0 }
func (p *pdLifecycleIndex) Observe(string, []string) {
	p.observes++
	if p.onObserve != nil {
		p.onObserve()
	}
}

// fak-test:runtime fast est=2s lane=default
func TestNativePDSelectedModeCompatibility(t *testing.T) {
	for _, order := range []string{"frame-first", "logical-first"} {
		t.Run(order, func(t *testing.T) {
			tx := &pdReceiverCopy{}
			receiver, provider := receiverFixture(tx)
			frame := WithNativePDReceiverFabric(map[string]*NativePDReceiverFabric{"decode-0": receiver})
			logical := WithNativePDLogicalFabric(&NativePDFabricConfig{})
			opts := []NativePDOption{frame, logical}
			if order == "logical-first" {
				opts = []NativePDOption{logical, frame}
			}
			cluster := NewNativePDClusterWithResidency(model.NewSynthetic(SyntheticConfig()), 1, nil, opts...)
			defer waitNativePDClusterClosed(t, cluster)
			_, err := cluster.Admit(context.Background(), NativePDRequest{Prompt: []int{1, 2}})
			if !errors.Is(err, ErrNativePDModeConflict) {
				t.Fatalf("ambiguous modes: %v", err)
			}
			if provider.reads != 0 || tx.closes != 0 || len(cluster.selectedActive) != 0 ||
				cluster.prefill.Stats().PrefillRequests != 0 || cluster.decode[0].pool.TotalBlocks() != 0 {
				t.Fatal("ambiguous modes produced receiver/admission effects")
			}
		})
	}
	// A nil frame map is the peer API's no-op, not an instruction to clear a
	// previously selected map. A nil logical option still clears logical mode.
	cluster := NewNativePDClusterWithResidency(model.NewSynthetic(SyntheticConfig()), 1, nil,
		WithNativePDReceiverFabric(map[string]*NativePDReceiverFabric{}),
		WithNativePDReceiverFabric(nil),
		WithNativePDLogicalFabric(&NativePDFabricConfig{}),
		WithNativePDLogicalFabric(nil))
	defer waitNativePDClusterClosed(t, cluster)
	if cluster.receiverFabric == nil || cluster.fabric != nil {
		t.Fatal("typed nil option semantics changed")
	}
	_, err := cluster.Admit(context.Background(), NativePDRequest{Prompt: []int{1, 2}})
	if !errors.Is(err, ErrNativePDReceiver) {
		t.Fatalf("empty selected map silently fell back: %v", err)
	}
	legacy := NewNativePDCluster(model.NewSynthetic(SyntheticConfig()), 1)
	legacy.Close()
	if legacy.selectedClosing || legacy.selectedDrained != nil {
		t.Fatal("nil legacy Close entered selected lifecycle")
	}
	var absent *NativePDCluster
	if err := absent.WaitClosed(context.Background()); err != nil {
		t.Fatalf("nil cluster is not already complete: %v", err)
	}
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer cancelExpired()
	if err := cluster.WaitClosed(expired); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired selected completion context: %v", err)
	}
}

// This pauses the real scheduler after successful AdmitImported but before the
// actual selected registration helper. Close wins registration; cancellation
// leaves the Session with the executing scheduler and frees only receiver pages.
// fak-test:runtime fast est=2s lane=default
func TestNativePDSelectedCloseLosesImportAndJoinsScheduler(t *testing.T) {
	for _, mode := range []string{"frame", "logical"} {
		t.Run(mode, func(t *testing.T) {
			m := model.NewSynthetic(SyntheticConfig())
			opt := WithNativePDReceiverFabric(map[string]*NativePDReceiverFabric{})
			if mode == "logical" {
				opt = WithNativePDLogicalFabric(&NativePDFabricConfig{})
			}
			index := &pdLifecycleIndex{}
			cluster := NewNativePDClusterWithResidency(m, 1, index, opt)
			defer waitNativePDClusterClosed(t, cluster)
			admitCtx, owner, err := cluster.beginSelectedAdmission(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			active := true
			defer func() {
				if active {
					cluster.endSelectedAdmission(owner)
				}
			}()
			decode := cluster.decode[0]
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			var enterOnce sync.Once
			decode.sched.beforeModelExecute = func(nativeSchedulerEventKind, *schedLane) {
				enterOnce.Do(func() { close(entered) })
				<-release
			}
			sess := m.NewSession()
			prompt := []int{1, 2, 3}
			logits := sess.Prefill(prompt)
			span, err := model.KVCacheToPaged(decode.pool, sess.Cache)
			if err != nil {
				sess.Close()
				t.Fatal(err)
			}
			defer span.Free()
			receipt := model.PagedKVTransferReceipt{KV: span, Transfer: cachemeta.KVTransfer{SpanDigest: "paused-import"}}
			request, err := decode.sched.AdmitImported(context.Background(), nil, ImportedSequence{
				Session: sess, Logits: logits, Prompt: prompt, Tool: "selected-close-loser",
			})
			if err != nil {
				sess.Close()
				t.Fatal(err)
			}
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("real imported scheduler did not enter execution")
			}
			cluster.Close()
			if err := cluster.commitSelectedImport(admitCtx, decode, receipt, request); !errors.Is(err, ErrNativePDClosing) {
				t.Fatalf("late import won shutdown: %v", err)
			}
			// The receiver owns this separate paged copy, not the scheduler's cache.
			span.Free()
			cluster.endSelectedAdmission(owner)
			active = false
			if got := decode.Stats(); got.DecodeImports != 0 || got.SchedulerAdmits != 0 || index.observes != 0 {
				t.Fatalf("late import published success: %+v observes=%d", got, index.observes)
			}
			if len(decode.spans) != 0 || decode.pool.FreeBlocks() != decode.pool.TotalBlocks() {
				t.Fatal("late import leaked receiver pages")
			}
			waitCtx, cancelWait := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancelWait()
			if err := cluster.WaitClosed(waitCtx); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("completion failed to join executing scheduler: %v", err)
			}
			decode.sched.mu.Lock()
			lane := request.(*schedLane)
			retained := lane.inflightSess == sess || lane.sess == sess
			terminal := lane.terminal
			decode.sched.mu.Unlock()
			if !retained || terminal {
				t.Fatal("loser cleanup stole the executing scheduler Session")
			}
			unblock()
			waitNativePDClusterClosed(t, cluster)
			select {
			case <-lane.done:
			default:
				t.Fatal("WaitClosed returned before scheduler retirement")
			}
			if !lane.Reclaimed() {
				t.Fatal("completed scheduler did not reclaim the request")
			}
			for range request.Tokens() {
			}
			if _, err := request.Result(); !errors.Is(err, context.Canceled) {
				t.Fatalf("late request was not canceled: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=2s lane=default
func TestNativePDSelectedReentrantClose(t *testing.T) {
	for _, where := range []string{"provider", "open", "adapter-close", "residency", "wait-background", "wait-deadline"} {
		t.Run(where, func(t *testing.T) {
			tx := &pdReceiverCopy{}
			receiver, provider := receiverFixture(tx)
			index := &pdLifecycleIndex{}
			cluster := NewNativePDClusterWithResidency(model.NewSynthetic(SyntheticConfig()), 1, index,
				WithNativePDReceiverFabric(map[string]*NativePDReceiverFabric{"decode-0": receiver}))
			defer waitNativePDClusterClosed(t, cluster)
			var callbackWait error
			stop := func() {
				cluster.Close()
				if where == "wait-background" {
					callbackWait = cluster.WaitClosed(context.Background())
				}
				if where == "wait-deadline" {
					ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
					defer cancel()
					callbackWait = cluster.WaitClosed(ctx)
				}
			}
			switch where {
			case "provider":
				provider.onRead = func(n int) {
					if n == 1 {
						stop()
					}
				}
			case "open":
				receiver.Open = func(_ context.Context, frame []byte) (NativePDReceiverTransaction, error) {
					tx.source = frame
					stop()
					return tx, nil
				}
			case "adapter-close", "wait-background", "wait-deadline":
				tx.onClose = stop
			case "residency":
				index.onObserve = stop
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var admission NativePDAdmission
			var admitErr error
			done := make(chan struct{})
			go func() {
				admission, admitErr = cluster.Admit(ctx, NativePDRequest{Prompt: []int{1, 2, 3}})
				close(done)
			}()
			select {
			case <-done:
			case <-ctx.Done():
				t.Fatal("reentrant Close blocked its own selected admission")
			}
			if where == "residency" {
				if admitErr != nil || admission.Request == nil || index.observes != 1 {
					t.Fatalf("pre-Close committed admission: %v observes=%d", admitErr, index.observes)
				}
			} else if !errors.Is(admitErr, ErrNativePDReceiver) || admission.Request != nil || index.observes != 0 {
				t.Fatalf("closed receiver published admission: %v observes=%d", admitErr, index.observes)
			}
			if where == "wait-background" && !errors.Is(callbackWait, ErrNativePDWaitDeadline) {
				t.Fatalf("unbounded self-wait was not refused: %v", callbackWait)
			}
			if where == "wait-deadline" && !errors.Is(callbackWait, context.DeadlineExceeded) {
				t.Fatalf("bounded self-wait did not expire: %v", callbackWait)
			}
			waitNativePDClusterClosed(t, cluster)
			if len(tx.source) > 0 && tx.closes != 1 {
				t.Fatalf("adapter cleanup count %d", tx.closes)
			}
			decode := cluster.decode[0]
			if len(decode.spans) != 0 || decode.pool.FreeBlocks() != decode.pool.TotalBlocks() || len(receiver.slot) != 0 {
				t.Fatal("completion retained pages or receiver gate")
			}
			stats := decode.Stats()
			if where != "residency" && (stats.DecodeImports != 0 || stats.SchedulerAdmits != 0) {
				t.Fatalf("refused admission counted scheduler success: %+v", stats)
			}
			before := index.observes
			_, err := cluster.Admit(context.Background(), NativePDRequest{Prompt: []int{1, 2, 3}})
			if !errors.Is(err, ErrNativePDClosing) || index.observes != before {
				t.Fatalf("post-completion admission reached residency: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativePDLogicalReentrantCloseFinishesMetrics(t *testing.T) {
	cfg := SyntheticConfig()
	m := model.NewSynthetic(cfg)
	backend := model.KVTransferBackend("test-logical-reentrant-close")
	model.RegisterKVTransferBackend(model.KVTransferBackendSpec{Backend: backend}, func(request model.KVTransportOpenRequest) (model.KVTransportHandle, error) {
		return model.KVTransportHandle{Transport: nativePDFabricTestTransport{pool: request.Pool, cfg: cfg}}, nil
	})
	defer model.RegisterKVTransferBackend(model.KVTransferBackendSpec{Backend: backend}, nil)
	now := time.Now()
	snapshot := fabricmap.Snapshot{
		Provider: "receiver", Generation: 1, ObservedAt: now.Add(-time.Second), ValidUntil: now.Add(time.Minute),
		Endpoints: []fabricmap.Endpoint{{ID: "source"}, {ID: "landing"}, {ID: "ready"}},
		Links: []fabricmap.Link{
			{ID: "copy", From: "source", To: "landing", Transport: string(backend)},
			{ID: "realize", From: "landing", To: "ready", Transport: NativePDFabricMaterialize, Labels: map[string]string{"kv.plan_id": "dense"}},
		},
	}
	var cluster *NativePDCluster
	config := &NativePDFabricConfig{
		Provider: nativePDFabricTestProvider(func(context.Context) (fabricmap.Snapshot, error) {
			cluster.Close()
			return snapshot, nil
		}),
		ProviderID: "receiver", Source: "source",
		Receivers:  map[string]NativePDFabricReceiver{"decode-0": {Landing: "landing", Ready: "ready", PlanID: "dense", Backend: backend}},
		Access:     fabricmap.Access{Principal: "receiver", Tenant: "test", Operation: fabricmap.OperationCopy},
		Authorizer: fabricmap.AuthorizerFunc(func(fabricmap.AuthorizationRequest) (bool, string) { return true, "" }),
	}
	cluster = NewNativePDClusterWithResidency(m, 1, nil, WithNativePDLogicalFabric(config))
	defer waitNativePDClusterClosed(t, cluster)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	done := make(chan struct{})
	var admitErr error
	go func() {
		_, admitErr = cluster.Admit(ctx, NativePDRequest{Prompt: []int{1, 2, 3}})
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("logical callback Close blocked its own receiver mutex")
	}
	if admitErr == nil {
		t.Fatal("closed logical receiver was admitted")
	}
	waitNativePDClusterClosed(t, cluster)
	cluster.fabric.timingMu.Lock()
	metrics := cluster.fabric.timings
	cluster.fabric.timingMu.Unlock()
	if metrics.count[0] != 1 || metrics.count[1] != 0 {
		t.Fatal("completion preceded refused timing cleanup")
	}
	for _, active := range metrics.inflight {
		if active != 0 {
			t.Fatal("completion preceded in-flight timing cleanup")
		}
	}
	after := cluster.fabric.timings
	if !reflect.DeepEqual(metrics, after) {
		t.Fatal("timing changed after completed shutdown")
	}
}

// Close may win before or during acquire. This witnesses preservation of the
// foreign gate owner, not deterministic time spent blocked inside acquire.
// fak-test:runtime fast est=1s lane=default
func TestNativePDSelectedClosePreservesForeignReceiverGate(t *testing.T) {
	tx := &pdReceiverCopy{}
	receiver, provider := receiverFixture(tx)
	if err := receiver.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	held := true
	defer func() {
		if held {
			<-receiver.slot
		}
	}()
	cluster := NewNativePDClusterWithResidency(model.NewSynthetic(SyntheticConfig()), 1, nil,
		WithNativePDReceiverFabric(map[string]*NativePDReceiverFabric{"decode-0": receiver}))
	defer waitNativePDClusterClosed(t, cluster)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started := make(chan struct{})
	done := make(chan struct{})
	var admissionErr error
	go func() {
		// The public call is counted before it can wait on the foreign gate.
		close(started)
		_, admissionErr = cluster.Admit(ctx, NativePDRequest{Prompt: []int{1, 2, 3}})
		close(done)
	}()
	<-started
	// Wait for actual selected registration; no channel-timing assumption.
	for {
		cluster.selectedMu.Lock()
		active := len(cluster.selectedActive) == 1
		cluster.selectedMu.Unlock()
		if active {
			break
		}
		select {
		case <-done:
			t.Fatal("admission completed through a held receiver gate")
		case <-ctx.Done():
			t.Fatal("admission did not register")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cluster.Close()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("Close did not cancel selected admission with a foreign gate held")
	}
	if !errors.Is(admissionErr, ErrNativePDReceiver) || tx.closes != 0 || provider.reads != 0 || len(receiver.slot) != 1 {
		t.Fatalf("canceled admission disturbed foreign receiver owner: err=%v", admissionErr)
	}
	waitNativePDClusterClosed(t, cluster)
	<-receiver.slot
	held = false
}
