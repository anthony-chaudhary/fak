package modelengine

import (
	"context"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativePDRoleSplitTransfersKVIntoDecodeScheduler(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 2)
	defer cluster.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5, 17, 19}
	req := NativePDRequest{
		Call:             inlineCall("issue28_pd", `{"prompt":"alpha"}`),
		Prompt:           prompt,
		ModelID:          "synthetic-test",
		TokenizerID:      "byte",
		Lease:            "lease-28",
		Taint:            abi.TaintTrusted,
		Scope:            abi.ScopeFleet,
		AdmissionVerdict: cachemeta.AdmissionAllow,
		AdmittedBy:       "admission-test",
	}

	admit, err := cluster.Admit(ctx, req)
	if err != nil {
		t.Fatalf("Admit native P/D: %v", err)
	}
	if admit.PrefillWorker == "" || admit.DecodeWorker == "" || admit.PrefillWorker == admit.DecodeWorker {
		t.Fatalf("workers not split: prefill=%q decode=%q", admit.PrefillWorker, admit.DecodeWorker)
	}
	if admit.LocalityHit {
		t.Fatal("first route should be a cold receive, not a locality hit")
	}
	if admit.Transfer.Transfer.SerializerID != model.PagedKVTransferSerializerID {
		t.Fatalf("serializer = %q, want %q", admit.Transfer.Transfer.SerializerID, model.PagedKVTransferSerializerID)
	}
	if admit.Transfer.Transfer.BytesMoved <= 0 {
		t.Fatalf("transfer moved no KV bytes: %+v", admit.Transfer.Transfer)
	}

	got := drainPDRequest(t, admit.Request)
	want := serialGreedyTokens(m, prompt)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("P/D tokens changed:\n got %v\nwant %v", got, want)
	}

	prefillStats := cluster.PrefillWorker().Stats()
	if prefillStats.PrefillRequests != 1 {
		t.Fatalf("prefill stats = %+v, want one prefill request", prefillStats)
	}
	decode, ok := cluster.DecodeWorker(admit.DecodeWorker)
	if !ok {
		t.Fatalf("decode worker %q not found", admit.DecodeWorker)
	}
	decodeStats := decode.Stats()
	if decodeStats.DecodeImports != 1 || decodeStats.SchedulerAdmits != 1 || decodeStats.RePrefillOnDecode != 0 {
		t.Fatalf("decode stats = %+v, want one import/admit and no decode-side prefill", decodeStats)
	}
	if peak := decode.Scheduler().MaxObservedRunning(); peak == 0 {
		t.Fatalf("decode scheduler peak running = %d, want imported lane to run through scheduler", peak)
	}

	entry, ok := decode.TransferEntry(admit.Transfer.Transfer.SpanDigest)
	if !ok {
		t.Fatalf("decode worker cannot query transfer entry for digest %q", admit.Transfer.Transfer.SpanDigest)
	}
	if entry.Security.Taint != abi.TaintTrusted ||
		entry.Security.Scope != abi.ScopeFleet ||
		entry.Security.AdmissionVerdict != cachemeta.AdmissionAllow ||
		entry.Security.AdmittedBy != "admission-test" {
		t.Fatalf("trust descriptor not preserved: %+v", entry.Security)
	}
	if entry.Residency.Lease != "lease-28" {
		t.Fatalf("lease not preserved: %+v", entry.Residency)
	}

	assertExactEvictAfterPDImport(t, m, prompt, admit.ImportedCache, 4, 2)

	second, err := cluster.Admit(ctx, req)
	if err != nil {
		t.Fatalf("second Admit native P/D: %v", err)
	}
	if !second.LocalityHit {
		t.Fatal("repeat prefix should route by KV locality")
	}
	if second.DecodeWorker != admit.DecodeWorker {
		t.Fatalf("repeat prefix routed to %q, want resident holder %q", second.DecodeWorker, admit.DecodeWorker)
	}
	_ = drainPDRequest(t, second.Request)

	for _, wantMetric := range []string{
		`fak_native_pd_worker_requests_total{role="prefill",worker="prefill-0"} 2`,
		`fak_native_pd_worker_requests_total{role="decode",worker="` + admit.DecodeWorker + `"} 2`,
		`fak_native_pd_route_total{result="locality_hit"} 1`,
	} {
		if metrics := cluster.Metrics(); !strings.Contains(metrics, wantMetric) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", wantMetric, metrics)
		}
	}
}

func TestNativePDRoutesThroughSharedResidencyIndex(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	idx := &testPDResidency{held: map[string][][]string{
		"decode-1": {{"tenant-a", "shared-prefix"}},
	}}
	cluster := NewNativePDClusterWithResidency(m, 2, idx)
	defer cluster.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	prompt := []int{13, 21, 34, 55}
	admit, err := cluster.Admit(ctx, NativePDRequest{
		Call:             inlineCall("issue28_shared_index", `{"prompt":"shared"}`),
		Prompt:           prompt,
		PrefixSegments:   []string{"tenant-a", "shared-prefix"},
		Taint:            abi.TaintTrusted,
		Scope:            abi.ScopeFleet,
		AdmissionVerdict: cachemeta.AdmissionAllow,
	})
	if err != nil {
		t.Fatalf("Admit native P/D with shared residency: %v", err)
	}
	if !admit.LocalityHit {
		t.Fatal("shared residency match should be reported as a locality hit")
	}
	if admit.DecodeWorker != "decode-1" {
		t.Fatalf("decode worker = %q, want decode-1 from shared residency index", admit.DecodeWorker)
	}
	if got, want := drainPDRequest(t, admit.Request), serialGreedyTokens(m, prompt); !reflect.DeepEqual(got, want) {
		t.Fatalf("P/D tokens changed:\n got %v\nwant %v", got, want)
	}
	if idx.Overlap("decode-1", []string{"tenant-a", "shared-prefix"}) != 2 {
		t.Fatal("decode worker residency was not retained in shared index")
	}
}

func TestNativePDDecodeWorkerBatchesImportedSequences(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 1)
	defer cluster.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	prompts := [][]int{
		{3, 1, 4, 1, 5},
		{9, 2, 6, 5, 3},
	}
	admissions := make([]NativePDAdmission, len(prompts))
	for i, prompt := range prompts {
		admit, err := cluster.Admit(ctx, NativePDRequest{
			Call:             inlineCall("issue28_batch", `{"i":`+strconv.Itoa(i)+`}`),
			Prompt:           prompt,
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: cachemeta.AdmissionAllow,
		})
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		admissions[i] = admit
	}

	got := make([][]int, len(admissions))
	var wg sync.WaitGroup
	for i, admit := range admissions {
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for tok := range r.Tokens() {
				got[i] = append(got[i], tok.ID)
			}
		}(i, admit.Request)
	}
	wg.Wait()
	for i, admit := range admissions {
		if _, err := admit.Request.Result(); err != nil {
			t.Fatalf("Result %d: %v", i, err)
		}
		want := serialGreedyTokens(m, prompts[i])
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("imported lane %d tokens changed:\n got %v\nwant %v", i, got[i], want)
		}
	}
	decode, _ := cluster.DecodeWorker("decode-0")
	if peak := decode.Scheduler().MaxObservedRunning(); peak < 2 {
		t.Fatalf("decode scheduler peak running = %d, want imported lanes co-batched", peak)
	}
}

type testPDResidency struct {
	mu   sync.Mutex
	held map[string][][]string
}

func (r *testPDResidency) Overlap(worker string, prefix []string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	best := 0
	for _, held := range r.held[worker] {
		if n := nativePDSegmentPrefixLen(held, prefix); n > best {
			best = n
		}
	}
	return best
}

func (r *testPDResidency) Observe(worker string, prefix []string) {
	if worker == "" || len(prefix) == 0 {
		return
	}
	r.mu.Lock()
	r.held[worker] = append(r.held[worker], append([]string(nil), prefix...))
	r.mu.Unlock()
}

func drainPDRequest(t *testing.T, r abi.EngineRequest) []int {
	t.Helper()
	var out []int
	for tok := range r.Tokens() {
		out = append(out, tok.ID)
	}
	res, err := r.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("Result = %+v, want StatusOK", res)
	}
	return out
}

func assertExactEvictAfterPDImport(t *testing.T, m *model.Model, prompt []int, imported *model.KVCache, from, n int) {
	t.Helper()
	if imported == nil {
		t.Fatal("imported cache is nil")
	}
	got := imported.Clone()
	if removed := got.Evict(from, n); removed != n {
		t.Fatalf("imported Evict removed %d, want %d", removed, n)
	}
	kept := append([]int(nil), prompt[:from]...)
	kept = append(kept, prompt[from+n:]...)
	wantSess := m.NewSession()
	wantSess.Prefill(kept)
	assertKVRowsEqual(t, "post-P/D evict", wantSess.Cache, got)
}

func assertKVRowsEqual(t *testing.T, label string, want, got *model.KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s: Len = %d, want %d", label, got.Len(), want.Len())
	}
	for l := range want.K {
		if !reflect.DeepEqual(got.K[l], want.K[l]) {
			t.Fatalf("%s: layer %d K differs", label, l)
		}
		if !reflect.DeepEqual(got.Kraw[l], want.Kraw[l]) {
			t.Fatalf("%s: layer %d Kraw differs", label, l)
		}
		if !reflect.DeepEqual(got.V[l], want.V[l]) {
			t.Fatalf("%s: layer %d V differs", label, l)
		}
	}
}
