package modelengine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/fabricmap"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type nativePDFabricTestProvider func(context.Context) (fabricmap.Snapshot, error)

func (f nativePDFabricTestProvider) Snapshot(ctx context.Context) (fabricmap.Snapshot, error) {
	return f(ctx)
}

type nativePDFabricTestTransport struct {
	pool    *model.PagedKVPool
	cfg     model.Config
	failure string
}

func (t nativePDFabricTestTransport) Send(src *model.PagedKV, descriptor cachemeta.KVTransfer, from, n int) (model.PagedKVTransferReceipt, error) {
	receipt, err := (model.LocalKVTransport{Pool: t.pool}).Send(src, descriptor, from, n)
	if err != nil {
		return receipt, err
	}
	if t.failure == "partial-send" {
		receipt.Transfer.ModelID = "untrusted-reply"
		receipt.Transfer.Owner = "untrusted-reply"
		receipt.Transfer.AdmittedBy = "untrusted-reply"
		return receipt, errors.New("send failed after receiver allocated pages")
	}
	if t.failure == "corrupt-landing" {
		// Keep sender metadata intact while changing the actual received rows.
		rows := make([][]float32, t.cfg.NumLayers)
		for layer := range rows {
			rows[layer] = make([]float32, t.cfg.NumKVHeads*t.cfg.HeadDim)
		}
		receipt.KV.AppendRaw(rows, rows, rows)
	}
	return receipt, nil
}

// fak-test:runtime fast est=2s
func TestNativePDReceiverReadyAdmission(t *testing.T) {
	for _, name := range []string{
		"complete", "wait-observed", "legacy", "landing-only", "expired", "generation-changed",
		"unauthorized", "partial-send", "corrupt-landing", "scheduler-closed", "close-serialized",
	} {
		t.Run(name, func(t *testing.T) {
			cfg := SyntheticConfig()
			m := model.NewSynthetic(cfg)
			backend := model.KVTransferBackend("test-native-pd-fabric-" + name)
			model.RegisterKVTransferBackend(model.KVTransferBackendSpec{Backend: backend}, func(request model.KVTransportOpenRequest) (model.KVTransportHandle, error) {
				return model.KVTransportHandle{Transport: nativePDFabricTestTransport{pool: request.Pool, cfg: cfg, failure: name}}, nil
			})
			t.Cleanup(func() {
				model.RegisterKVTransferBackend(model.KVTransferBackendSpec{Backend: backend}, nil)
			})
			now := time.Now()
			snapshot := fabricmap.Snapshot{
				Provider: "receiver", Generation: 7, ObservedAt: now.Add(-time.Second), ValidUntil: now.Add(time.Minute),
				Endpoints: []fabricmap.Endpoint{{ID: "source"}, {ID: "landing"}, {ID: "ready"}},
				Links: []fabricmap.Link{
					{ID: "copy", From: "source", To: "landing", Transport: string(backend)},
					{ID: "realize", From: "landing", To: "ready", Transport: NativePDFabricMaterialize, Labels: map[string]string{"kv.plan_id": "opaque-native-dense"}},
				},
			}
			if name == "landing-only" {
				snapshot.Links = snapshot.Links[:1]
			}
			if name == "expired" {
				snapshot.ValidUntil = now.Add(-time.Millisecond)
			}
			reads := 0
			provider := nativePDFabricTestProvider(func(context.Context) (fabricmap.Snapshot, error) {
				reads++
				current := snapshot
				if name == "generation-changed" && reads > 1 {
					current.Generation++
				}
				return current, nil
			})
			config := &NativePDFabricConfig{
				Provider: provider, ProviderID: "receiver", Source: "source",
				Receivers: map[string]NativePDFabricReceiver{
					"decode-0": {Landing: "landing", Ready: "ready", PlanID: "opaque-native-dense", Backend: backend},
				},
				Access: fabricmap.Access{Principal: "native-receiver", Tenant: "test", Operation: fabricmap.OperationCopy},
				Authorizer: fabricmap.AuthorizerFunc(func(request fabricmap.AuthorizationRequest) (bool, string) {
					return name != "unauthorized" || request.HopIndex != 1, "receiver denies local realization"
				}),
			}
			if name == "legacy" {
				config = nil
			}
			cluster := NewNativePDClusterWithResidency(m, 1, nil, WithNativePDReceiverFabric(config))
			t.Cleanup(cluster.Close)
			decode := cluster.decode[0]
			if name == "close-serialized" {
				// Hold the same transaction boundary as selected admission.
				// Close must wait for registration, then retire those owned pages.
				cluster.fabric.mu.Lock()
				locked := true
				defer func() {
					if locked {
						cluster.fabric.mu.Unlock()
					}
				}()
				started, closed := make(chan struct{}), make(chan struct{})
				go func() {
					close(started)
					cluster.Close()
					close(closed)
				}()
				<-started
				select {
				case <-closed:
					t.Fatal("Close bypassed the receiver admission transaction")
				case <-time.After(20 * time.Millisecond):
				}
				rows := make([][]float32, cfg.NumLayers)
				for layer := range rows {
					rows[layer] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
				}
				span := decode.pool.NewSequence()
				span.AppendRaw(rows, rows, rows)
				decode.addImport(model.PagedKVTransferReceipt{KV: span, Transfer: cachemeta.KVTransfer{SpanDigest: "held-import"}})
				cluster.fabric.mu.Unlock()
				locked = false
				select {
				case <-closed:
				case <-time.After(time.Second):
					t.Fatal("Close did not finish after receiver transaction")
				}
				if len(decode.spans) != 0 || decode.pool.FreeBlocks() != decode.pool.TotalBlocks() {
					t.Fatal("Close left receiver pages registered after shutdown")
				}
				return
			}
			if name == "scheduler-closed" {
				decode.sched.Close()
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			prompt := []int{3, 7, 11, 5, 17, 19}
			var admission NativePDAdmission
			var err error
			if name == "wait-observed" {
				// Observe a real Admit blocked on the receiver transaction, not
				// a synthetic sleep or a standalone timing helper.
				cluster.fabric.mu.Lock()
				locked := true
				unlock := func() {
					if locked {
						cluster.fabric.mu.Unlock()
						locked = false
					}
				}
				done := make(chan struct{})
				go func() {
					admission, err = cluster.Admit(ctx, NativePDRequest{Prompt: prompt})
					close(done)
				}()
				defer func() {
					unlock()
					cancel()
					select {
					case <-done:
					case <-time.After(time.Second):
						t.Error("timed admission did not join after receiver unlock")
					}
				}()
				deadline := time.Now().Add(time.Second)
				for {
					cluster.fabric.timingMu.Lock()
					waiting := cluster.fabric.timings.inflight[nativePDFabricWait] == 1
					cluster.fabric.timingMu.Unlock()
					if waiting {
						break
					}
					if time.Now().After(deadline) {
						t.Fatal("actual Admit never exposed its receiver wait")
					}
					runtime.Gosched()
				}
				scraped := make(chan string, 1)
				go func() { scraped <- cluster.Metrics() }()
				select {
				case text := <-scraped:
					if !strings.Contains(text, "fak_native_pd_receiver_phase_inflight{phase=\"receiver_wait\"} 1\n") {
						t.Fatal("live receiver wait is missing from the existing metrics surface")
					}
				case <-time.After(time.Second):
					unlock()
					<-scraped
					t.Fatal("metrics scrape waited behind the receiver transaction")
				}
				unlock()
				select {
				case <-done:
				case <-ctx.Done():
					t.Fatal("observed receiver wait did not complete after unlock")
				}
			} else {
				admission, err = cluster.Admit(ctx, NativePDRequest{Prompt: prompt})
			}
			wantSuccess := name == "complete" || name == "wait-observed" || name == "legacy"
			if (err == nil) != wantSuccess {
				t.Fatalf("Admit error=%v, want success=%v", err, wantSuccess)
			}
			if cluster.fabric != nil {
				cluster.fabric.timingMu.Lock()
				observed := cluster.fabric.timings
				cluster.fabric.timingMu.Unlock()
				outcome := 0
				outcomeName := "refused"
				if wantSuccess {
					outcome, outcomeName = 1, "admitted"
				}
				if observed.count[outcome] != 1 || observed.count[1-outcome] != 0 {
					t.Fatal("receiver denominator did not count exactly this completed call")
				}
				var phaseSeconds float64
				for phase, active := range observed.inflight {
					if active != 0 {
						t.Fatalf("receiver phase %d leaked an in-flight gauge", phase)
					}
					phaseSeconds += observed.phaseSeconds[outcome][phase]
					if wantSuccess && observed.phaseCount[outcome][phase] != 1 {
						t.Fatalf("actual successful Admit did not enter phase %d exactly once per call", phase)
					}
				}
				if observed.seconds[outcome] <= 0 || math.Abs(phaseSeconds-observed.seconds[outcome]) > 1e-9 {
					t.Fatal("receiver phases do not partition the completed-call wall-time denominator")
				}
				if !strings.Contains(cluster.Metrics(), fmt.Sprintf("fak_native_pd_receiver_attempt_seconds_count{outcome=%q} 1\n", outcomeName)) {
					t.Fatal("completed receiver call is absent from the existing metrics surface")
				}
			} else if strings.Contains(cluster.Metrics(), "fak_native_pd_receiver_") {
				t.Fatal("nil fabric configuration changed legacy metrics")
			}
			if !wantSuccess {
				if got := decode.Stats(); got.DecodeImports != 0 || got.SchedulerAdmits != 0 {
					t.Fatalf("refused span was published: %+v", got)
				}
				if len(decode.entries) != 0 || len(decode.spans) != 0 || decode.pool.FreeBlocks() != decode.pool.TotalBlocks() {
					t.Fatalf("refusal retained imported ownership: entries=%d spans=%d free=%d total=%d", len(decode.entries), len(decode.spans), decode.pool.FreeBlocks(), decode.pool.TotalBlocks())
				}
				if admission.Transfer.Transfer.ModelID != "native-pd" || admission.Transfer.Transfer.Owner != "decode-0" || admission.Transfer.Transfer.AdmittedBy != "native-pd" {
					t.Fatal("failure event adopted transport reply identity/security")
				}
				if admission.Transfer.Verdict().Kind != cachemeta.LookupFault {
					t.Fatalf("refusal lost typed fault: %+v", admission.Transfer.Entry)
				}
				return
			}
			got := drainPDRequest(t, admission.Request)
			if want := serialGreedyTokens(m, prompt); !reflect.DeepEqual(got, want) {
				t.Fatalf("receiver-ready decode changed tokens: got %v want %v", got, want)
			}
			if admission.Transfer.Verdict().Kind != cachemeta.LookupHit {
				t.Fatal("complete import did not publish a KV hit")
			}
			proof := admission.Transfer.Transfer.Fabric
			if name == "legacy" {
				if proof != nil {
					t.Fatal("nil configuration changed legacy transfer evidence")
				}
				return
			}
			if reads != 2 || proof == nil || len(proof.Execution.Hops) != 2 {
				t.Fatalf("missing full route/generation witness: reads=%d proof=%+v", reads, proof)
			}
			if err := proof.Validate(admission.Transfer.Transfer, time.Now()); err != nil {
				t.Fatal(err)
			}
			if proof.LogicalBytes >= uint64(admission.Transfer.Transfer.BytesMoved) {
				t.Fatal("logical materialization count was confused with padded wire bytes")
			}
		})
	}
}
