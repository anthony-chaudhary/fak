package gateway

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestFeatureTrackerIncompleteResponseDoesNotCertifyFinalUse(t *testing.T) {
	tracker := NewFeatureActivationTracker(FeatureCatalog{})
	ctx := WithFeatureActivationTracker(context.Background(), tracker)
	base := httptest.NewRecorder()
	wrapped, finish := newFeatureActivationResponseWriter(base, tracker)
	tracker.RecordActivation(FeaturePolicyFloor, FeatureOutcomeUsed)
	io.WriteString(wrapped, "data: first\n\n")
	wrapped.(http.Flusher).Flush()
	markFeatureActivationIncomplete(ctx)
	snapshot, first := finish()
	if !first || !reflect.DeepEqual(snapshot.Used, []ServeFeature{FeaturePolicyFloor}) {
		t.Fatalf("observed partial evidence lost: snapshot=%+v first=%v", snapshot, first)
	}
	if tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed) {
		t.Fatal("late worker mutated finalized evidence")
	}
	response := base.Result()
	defer response.Body.Close()
	if got := response.Header.Get(HeaderFeaturesUsed); got != "policy_floor" {
		t.Fatalf("initial evidence=%q", got)
	}
	if got := response.Trailer.Get(HeaderFeaturesUsedFinal); got != "" {
		t.Fatalf("unfinished worker falsely certified final use: %q", got)
	}
}

func TestFeatureTracker_ResponseHeaders(t *testing.T) {
	tracker := NewFeatureActivationTracker(FeatureCatalog{Schema: FeatureSchema, Features: []FeatureStatus{
		{Feature: FeatureVDSO, State: FeatureConfiguredActive},
		{Feature: FeaturePolicyFloor, State: FeatureConfiguredActive},
		{Feature: FeatureElideResults, State: FeatureConfiguredStandby},
	}})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	finished := make(chan FeatureActivationSnapshot, 1)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Trailer", "X-Other-Witness")
		wrapped, finish := newFeatureActivationResponseWriter(w, tracker)
		defer func() { snapshot, _ := finish(); finished <- snapshot }()
		tracker.RecordActivation(FeaturePolicyFloor, FeatureOutcomeUsed)
		wrapped.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(wrapped, "data: first\n\n")
		wrapped.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed)
		tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed)
		io.WriteString(wrapped, "data: [DONE]\n\n")
		wrapped.Header().Set("X-Other-Witness", "preserved")
	}))
	defer func() { unblock(); ts.Close() }()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	line, err := reader.ReadString('\n')
	if err != nil || line != "data: first\n" {
		t.Fatalf("first streaming event=%q err=%v", line, err)
	}
	if got := resp.Header.Get(HeaderFeaturesUsed); got != "policy_floor" {
		t.Errorf("initial Used=%q", got)
	}
	if got := resp.Header.Get(HeaderFeaturesEnabled); got != "elide_results,policy_floor,vdso" {
		t.Errorf("Enabled=%q", got)
	}
	unblock()
	if _, err := io.ReadAll(reader); err != nil {
		t.Fatal(err)
	}
	if got := resp.Header.Get(HeaderFeaturesUsed); got != "policy_floor" {
		t.Errorf("initial header retrospectively changed to %q", got)
	}
	if got := resp.Trailer.Get(HeaderFeaturesUsedFinal); got != "policy_floor,vdso" {
		t.Errorf("late activation trailer=%q", got)
	}
	if got := resp.Trailer.Get("X-Other-Witness"); got != "preserved" {
		t.Errorf("unrelated trailer lost: %q", got)
	}
	if got := <-finished; !reflect.DeepEqual(got.Used, []ServeFeature{FeaturePolicyFloor, FeatureVDSO}) {
		t.Errorf("final used=%v", got.Used)
	}
}

type featureHeaderRecorder struct {
	header http.Header
	codes  []int
}

func (w *featureHeaderRecorder) Header() http.Header         { return w.header }
func (w *featureHeaderRecorder) WriteHeader(code int)        { w.codes = append(w.codes, code) }
func (w *featureHeaderRecorder) Write(p []byte) (int, error) { return len(p), nil }

func TestFeatureTrackerWriterCapabilitiesAndInformationals(t *testing.T) {
	base := &featureHeaderRecorder{header: make(http.Header)}
	tracker := NewFeatureActivationTracker(FeatureCatalog{})
	wrapped, finish := newFeatureActivationResponseWriter(base, tracker)
	if _, ok := wrapped.(http.Flusher); ok {
		t.Fatal("invented Flusher capability")
	}
	wrapped.WriteHeader(http.StatusEarlyHints)
	tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed)
	wrapped.WriteHeader(http.StatusOK)
	if !reflect.DeepEqual(base.codes, []int{103, 200}) || wrapped.Header().Get(HeaderFeaturesUsed) != "vdso" {
		t.Fatalf("informational froze final headers: codes=%v used=%q", base.codes, wrapped.Header().Get(HeaderFeaturesUsed))
	}
	if !strings.Contains(wrapped.Header().Get("Trailer"), HeaderFeaturesUsedFinal) {
		t.Fatal("final trailer not declared")
	}
	if _, first := finish(); !first {
		t.Fatal("first completion rejected")
	}
	if _, first := finish(); first {
		t.Fatal("completion repeated")
	}
	if got, _ := newFeatureActivationResponseWriter(base, nil); got != base {
		t.Fatal("nil tracker changed writer identity")
	}
}
