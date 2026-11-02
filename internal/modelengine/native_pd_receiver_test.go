package modelengine

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/fabricmap"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type pdReceiverProvider struct {
	snapshot fabricmap.Snapshot
	reads    int
	onRead   func(int)
}

func (p *pdReceiverProvider) Snapshot(context.Context) (fabricmap.Snapshot, error) {
	p.reads++
	if p.onRead != nil {
		p.onRead(p.reads)
	}
	return p.snapshot, nil
}

type pdReceiverIndex struct{ observes int }

func (*pdReceiverIndex) Overlap(string, []string) int { return 0 }
func (p *pdReceiverIndex) Observe(string, []string)   { p.observes++ }

// This fixture performs two actual host byte copies. Its receipts witness those
// copies only, never RDMA/GPU execution or end-to-end performance qualification.
type pdReceiverCopy struct {
	source, landing, ready []byte
	corrupt                bool
	closes                 int
	closeErr               error
	onClose                func()
}

func (x *pdReceiverCopy) ExecuteHop(ctx context.Context, r fabricmap.AuthorizationRequest, n uint64) (fabricmap.HopReceipt, error) {
	if err := ctx.Err(); err != nil {
		return fabricmap.HopReceipt{}, err
	}
	start := time.Now()
	var b []byte
	switch r.HopIndex {
	case 0:
		x.landing = append([]byte(nil), x.source...)
		b = x.landing
	case 1:
		x.ready = append([]byte(nil), x.landing...)
		b = x.ready
	default:
		return fabricmap.HopReceipt{}, errors.New("unexpected hop")
	}
	h := sha256.Sum256(b)
	return fabricmap.HopReceipt{HopIndex: r.HopIndex, LinkID: r.Link.ID, From: r.Link.From, To: r.Link.To, Transport: r.Link.Transport, Bytes: uint64(len(b)), StartedAt: start, CompletedAt: time.Now(), State: fabricmap.HopComplete, Integrity: fabricmap.Integrity{Algorithm: "sha256", Digest: hex.EncodeToString(h[:])}}, nil
}
func (x *pdReceiverCopy) Materialize(context.Context) ([]byte, error) {
	b := append([]byte(nil), x.ready...)
	if x.corrupt && len(b) > 0 {
		b[len(b)-1] ^= 1
	}
	return b, nil
}
func (x *pdReceiverCopy) Close() error {
	x.closes++
	if x.onClose != nil {
		x.onClose()
	}
	return x.closeErr
}

func receiverFixture(tx *pdReceiverCopy) (*NativePDReceiverFabric, *pdReceiverProvider) {
	p := &pdReceiverProvider{snapshot: fabricmap.Snapshot{Provider: "receiver-owned", Generation: 1, ObservedAt: time.Now().Add(-time.Second), ValidUntil: time.Now().Add(time.Minute), Endpoints: []fabricmap.Endpoint{{ID: "source"}, {ID: "landing"}, {ID: "ready"}}, Links: []fabricmap.Link{{ID: "land", From: "source", To: "landing", Transport: "host-copy"}, {ID: "realize", From: "landing", To: "ready", Transport: "host-copy"}}}}
	r := &NativePDReceiverFabric{Provider: p, Authorizer: fabricmap.AuthorizerFunc(func(fabricmap.AuthorizationRequest) (bool, string) { return true, "" }), SourceEndpoint: "source", LandingEndpoint: "landing", ReadyEndpoint: "ready", Access: fabricmap.Access{Principal: "receiver", Tenant: "fixture", Operation: fabricmap.OperationCopy}, Open: func(_ context.Context, frame []byte) (NativePDReceiverTransaction, error) {
		tx.source = frame
		return tx, nil
	}}
	return r, p
}

// fak-test:runtime fast est=200ms lane=default
func TestNativePDReceiverReadyAdmission(t *testing.T) {
	for _, mode := range []string{"ready", "wrong-materialization", "denied-hop", "cleanup-failure", "route-changed", "missing-receiver", "request-override", "governance-deny", "open-error-resource", "missing-expiry", "expired-plan", "generation-changed", "cancel-after-decode", "provider-rollover"} {
		t.Run(mode, func(t *testing.T) {
			m := model.NewSynthetic(SyntheticConfig())
			tx := &pdReceiverCopy{corrupt: mode == "wrong-materialization"}
			r, provider := receiverFixture(tx)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if mode == "open-error-resource" {
				r.Open = func(_ context.Context, frame []byte) (NativePDReceiverTransaction, error) {
					tx.source = frame
					return tx, errors.New("open failed with owned resource")
				}
			}
			if mode == "missing-expiry" {
				provider.snapshot.ValidUntil = time.Time{}
			}
			if mode == "expired-plan" {
				provider.snapshot.ValidUntil = time.Now().Add(-time.Second)
			}
			if mode == "generation-changed" {
				tx.onClose = func() { provider.snapshot.Generation++ }
			}
			if mode == "provider-rollover" {
				tx.onClose = func() {
					provider.snapshot.Provider = "replacement"
					provider.snapshot.Generation = 1
					provider.snapshot.Links = nil
					provider.snapshot.Endpoints = nil
				}
			}

			if mode == "cancel-after-decode" {
				provider.onRead = func(n int) {
					if n == 3 {
						cancel()
					}
				}
			}

			if mode == "denied-hop" {
				r.Authorizer = fabricmap.AuthorizerFunc(func(a fabricmap.AuthorizationRequest) (bool, string) { return a.HopIndex == 0, "second hop denied" })
			}
			if mode == "cleanup-failure" {
				tx.closeErr = errors.New("receiver cleanup failed")
			}
			if mode == "route-changed" {
				tx.onClose = func() { provider.snapshot.Generation++; provider.snapshot.Links[1].Transport = "changed" }
			}
			receivers := map[string]*NativePDReceiverFabric{"decode-0": r}
			if mode == "missing-receiver" {
				delete(receivers, "decode-0")
			}
			index := &pdReceiverIndex{}
			cluster := NewNativePDClusterWithResidency(m, 1, index, WithNativePDReceiverFabric(receivers))
			defer cluster.Close()
			req := NativePDRequest{Prompt: []int{3, 7, 11, 5}, ModelID: "synthetic", TokenizerID: "byte", Lease: "test-receiver", AdmissionVerdict: cachemeta.AdmissionAllow}
			if mode == "request-override" {
				req.TransferBackend = model.KVTransferBackendSHM
			}
			if mode == "governance-deny" {
				req.AdmissionVerdict = cachemeta.AdmissionDeny
			}
			admission, err := cluster.Admit(ctx, req)
			if mode != "ready" {
				if !errors.Is(err, ErrNativePDReceiver) {
					t.Fatalf("selected receiver refusal = %v", err)
				}
				if mode == "cancel-after-decode" && !errors.Is(err, context.Canceled) {
					t.Fatalf("final admission did not preserve cancellation: %v", err)
				}
				if admission.Request != nil || admission.ReceiverFabric != nil || index.observes != 0 {
					t.Fatal("refusal published scheduler or residency admission")
				}
				stats := cluster.decode[0].Stats()
				if stats.DecodeImports != 0 || stats.SchedulerAdmits != 0 {
					t.Fatalf("refusal counted import: %+v", stats)
				}
				if len(cluster.decode[0].spans) != 0 {
					t.Fatal("refusal retained imported KV")
				}
				if len(tx.source) > 0 && tx.closes != 1 {
					t.Fatalf("transaction cleanup count %d", tx.closes)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := provider.snapshot.Endpoints[0].ID; got != "source" {
				t.Fatalf("receiver admission mutated provider endpoint order: first=%q", got)
			}
			if admission.ReceiverFabric == nil || admission.ReceiverFabric.State != fabricmap.ExecutionComplete || len(admission.ReceiverFabric.Hops) != 2 {
				t.Fatal("missing observed two-hop receipt")
			}
			if admission.ReceiverFabric.Snapshot.Provider != "receiver-owned" || admission.ReceiverFabric.Snapshot.Generation != 1 || admission.ReceiverFabric.Snapshot.ValidUntil.IsZero() || admission.ReceiverFabric.Snapshot.SHA256 == "" || admission.ReceiverFabric.FrameSHA256 == "" {
				t.Fatal("receiver receipt lost observed plan/frame binding")
			}
			if tx.closes != 1 || index.observes != 1 {
				t.Fatalf("cleanup=%d residency=%d", tx.closes, index.observes)
			}
			if got, want := drainPDRequest(t, admission.Request), serialGreedyTokens(m, req.Prompt); !reflect.DeepEqual(got, want) {
				t.Fatalf("receiver tokens %v differ from serial %v", got, want)
			}
			stats := cluster.decode[0].Stats()
			if stats.DecodeImports != 1 || stats.SchedulerAdmits != 1 || stats.RePrefillOnDecode != 0 {
				t.Fatalf("consumer stats %+v", stats)
			}
			labels := admission.Transfer.Entry.Labels
			if labels["receiver_provider"] != admission.ReceiverFabric.Snapshot.Provider || labels["receiver_generation"] != "1" || labels["receiver_snapshot_sha256"] != admission.ReceiverFabric.Snapshot.SHA256 || labels["receiver_frame_sha256"] != admission.ReceiverFabric.FrameSHA256 || labels["receiver_ready_endpoint"] != "ready" || labels["receiver_snapshot_valid_until"] == "" || labels["receiver_wire_bytes"] == "" {
				t.Fatal("normalized entry lost receiver-ready binding")
			}
			if admission.Transfer.Entry.ID.Digest == "" || admission.Transfer.Entry.Residency.Owner != "decode-0" {
				t.Fatal("existing cachemeta transfer binding missing")
			}
			if admission.Transfer.Transfer.BytesMoved >= int64(admission.ReceiverFabric.Hops[0].Bytes) {
				t.Fatal("wire frame bytes were conflated with native KV payload accounting")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativePDReceiverCanceledOwnerWait(t *testing.T) {
	tx := &pdReceiverCopy{}
	r, _ := receiverFixture(tx)
	if err := r.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer func() { <-r.slot }()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := r.acquire(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled admission: %v", err)
	}
	if len(r.slot) != 1 {
		t.Fatal("canceled waiter released foreign receiver owner")
	}
}
