package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime medium est=1s
func TestReplicaHTTPSampleOptions(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	t.Setenv("FAK_PROVIDER_MAX_TOKENS", "0")
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	for _, mode := range []string{"buffered", "stream", "hedge", "retarget", "stream_retarget", "canceled", "unhealthy", "defaults", "explicit_zero", "capped"} {
		t.Run(mode, func(t *testing.T) {
			policy := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
			type observed struct {
				wire, reserved int
				temperature    float64
				err            error
			}
			seen := make(chan observed, 4)
			var hits atomic.Int32
			stop := make(chan struct{})
			var endpoints []string
			var servers []*httptest.Server
			for _, name := range []string{"a", "b"} {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body struct {
						MaxTokens   int     `json:"max_tokens"`
						Stream      bool    `json:"stream"`
						Temperature float64 `json:"temperature"`
					}
					decodeErr := json.NewDecoder(r.Body).Decode(&body)
					reserved := -1
					for _, decision := range policy.DecodeFootprintDecisions() {
						if decision.Worker == name && !decision.Released {
							reserved = decision.RequestedOutputTokens
						}
					}
					seen <- observed{body.MaxTokens, reserved, body.Temperature, decodeErr}
					n := hits.Add(1)
					if strings.Contains(mode, "retarget") && n == 1 {
						w.WriteHeader(http.StatusBadGateway)
						return
					}
					if mode == "hedge" && n == 1 {
						select {
						case <-r.Context().Done():
						case <-stop:
						}
						return
					}
					if body.Stream {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
						return
					}
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprint(w, `{"model":"fleet-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
				}))
				t.Cleanup(server.Close)
				servers = append(servers, server)
				endpoints = append(endpoints, name+"="+server.URL)
			}
			t.Cleanup(func() { close(stop) })
			planner, err := newProxyPlanner(Config{Provider: "openai"}, "fleet-model", endpoints)
			if err != nil {
				t.Fatal(err)
			}
			router := planner.(*ReplicaDispatch).WithPickPolicy(policy)
			for i, replica := range router.replicas {
				httpPlanner := replica.Planner.(*agent.HTTPPlanner)
				httpPlanner.Client = servers[i].Client()
				httpPlanner.MaxTokens, httpPlanner.Temperature = 73, 0.75
				if mode == "capped" {
					httpPlanner.MaxTokensCap = 32
				}
			}
			var receipt HedgeReceipt
			if mode == "hedge" {
				router.Hedge = eligibleHedgePolicy(func(r HedgeReceipt) { receipt = r })
				router.Hedge.Delay = 25 * time.Millisecond
				router.Hedge.Deadline = 2 * time.Second
				router.Hedge.DrainTimeout = time.Second
			}
			if mode == "unhealthy" {
				router.WithMembership(router.FleetMembership()) // Unprobed roster refuses dispatch.
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if mode == "canceled" {
				cancel()
			}
			var calls atomic.Int32
			stateful := func(sp *agent.SampleParams) {
				n := 64 * int(calls.Add(1))
				if mode == "defaults" {
					return
				}
				if mode == "explicit_zero" {
					n = 0
				}
				zero := 0.0
				sp.MaxTokens, sp.Temperature = &n, &zero
			}
			var comp *agent.Completion
			var streamed strings.Builder
			messages := []agent.Message{{Role: agent.RoleUser, Content: "same resolved request"}}
			if strings.HasPrefix(mode, "stream") {
				comp, err = router.CompleteStream(ctx, func(delta string) error {
					streamed.WriteString(delta)
					return nil
				}, messages, nil, stateful)
			} else {
				comp, err = router.Complete(ctx, messages, nil, stateful)
			}
			if calls.Load() != 1 {
				t.Fatalf("caller SampleOpt ran %d times, want once per dispatch", calls.Load())
			}
			wantHits := 1
			switch mode {
			case "canceled":
				wantHits = 0
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("canceled dispatch = %v", err)
				}
			case "unhealthy":
				wantHits = 0
				if !errors.Is(err, ErrNoHealthyWorker) {
					t.Fatalf("unhealthy dispatch = %v", err)
				}
			default:
				if err != nil || comp == nil || comp.Message.Content != "ok" {
					t.Fatalf("HTTP dispatch = %+v, %v", comp, err)
				}
				if strings.HasPrefix(mode, "stream") && streamed.String() != "ok" {
					t.Fatalf("stream = %q", streamed.String())
				}
				if mode == "hedge" || strings.Contains(mode, "retarget") {
					wantHits = 2
				}
			}
			if got := int(hits.Load()); got != wantHits {
				t.Fatalf("HTTP attempts = %d, want %d", got, wantHits)
			}
			for range wantHits {
				got := <-seen
				wantWire, wantReserved, wantTemperature := 64, 64, 0.0
				switch mode {
				case "defaults":
					wantWire, wantReserved, wantTemperature = 73, 0, 0.75
				case "explicit_zero":
					wantWire, wantReserved = 73, 0
				case "capped":
					wantWire = 32
				}
				if got.err != nil || got.wire != wantWire || got.reserved != wantReserved || got.temperature != wantTemperature {
					t.Fatalf("wire/reservation/default contract changed: %+v, want %d/%d/%g", got, wantWire, wantReserved, wantTemperature)
				}
			}
			if mode == "hedge" && (receipt.RealizedMode != "hedged" || !receipt.Primary.Opened || !receipt.Hedge.Opened) {
				t.Fatalf("real eligible hedge was not exercised: %+v", receipt)
			}
			if active := policy.DecodeFootprintActiveByWorker(); len(active) != 0 {
				t.Fatalf("dispatch leaked bookings: %+v", active)
			}
		})
	}
}

// fak-test:runtime fast est=50ms
func TestReplicaHTTPSampleOptionCopies(t *testing.T) {
	router, err := NewReplicaDispatch("m", []PlannerReplica{{Name: "a", Planner: agent.NewHTTPPlanner("http://unused.invalid", "m", "")}})
	if err != nil {
		t.Fatal(err)
	}
	var input agent.SampleParams
	fillReplicaMutableOptions(t, reflect.ValueOf(&input).Elem())
	var observed int
	input.DecodeTokenObserver = func(string, string) { observed++ }
	var calls int
	opts := router.resolveHTTPSampleOpts([]agent.SampleOpt{nil, func(sp *agent.SampleParams) { calls++; *sp = input }})
	var first, second agent.SampleParams
	opts[0](&first)
	opts[0](&second)
	if calls != 1 {
		t.Fatalf("caller ran %d times", calls)
	}
	assertReplicaOptionsDetached(t, reflect.ValueOf(input), reflect.ValueOf(first), "input")
	assertReplicaOptionsDetached(t, reflect.ValueOf(first), reflect.ValueOf(second), "replay")
	first.DecodeTokenObserver("piece", "text")
	second.DecodeTokenObserver("piece", "text")
	if observed != 2 {
		t.Fatal("opaque observer capability was not preserved")
	}
	var omitted agent.SampleParams
	router.resolveHTTPSampleOpts([]agent.SampleOpt{nil})[0](&omitted)
	if !reflect.DeepEqual(omitted, agent.SampleParams{}) {
		t.Fatal("omitted overrides were not preserved")
	}
	empty := agent.SampleParams{Stop: []string{}, ResponseFormat: json.RawMessage{}, GuidedDecode: map[string]json.RawMessage{}}
	if copied := cloneReplicaSampleParams(empty); !reflect.DeepEqual(copied, empty) {
		t.Fatal("explicit empty containers changed to omitted ones")
	}
	// Relative options still compose in caller order against HTTPPlanner's zero baseline.
	var ordered agent.SampleParams
	router.resolveHTTPSampleOpts([]agent.SampleOpt{agent.WithMaxTokens(10), func(sp *agent.SampleParams) { *sp.MaxTokens += 7 }})[0](&ordered)
	if ordered.MaxTokens == nil || *ordered.MaxTokens != 17 {
		t.Fatalf("ordered fold = %+v", ordered.MaxTokens)
	}
	// A custom planner may seed its own defaults before applying overrides.
	router.replicas[0].Planner = &replicaDispatchTestPlanner{name: "custom"}
	custom := router.resolveHTTPSampleOpts([]agent.SampleOpt{func(sp *agent.SampleParams) { *sp.MaxTokens += 2 }})
	n := 11
	customParams := agent.SampleParams{MaxTokens: &n}
	custom[0](&customParams)
	if n != 13 {
		t.Fatal("custom planner's relative-option semantics changed")
	}
}

// Populate every mutable exported member so a new reference-bearing field cannot
// silently escape the copy regression. Functions are intentionally opaque.
func fillReplicaMutableOptions(t *testing.T, value reflect.Value) {
	t.Helper()
	for i := 0; i < value.NumField(); i++ {
		field := value.Field(i)
		if !field.CanSet() {
			continue
		}
		switch field.Kind() {
		case reflect.Pointer:
			field.Set(reflect.New(field.Type().Elem()))
			if field.Elem().Kind() == reflect.Struct {
				fillReplicaMutableOptions(t, field.Elem())
			} else {
				fillReplicaOptionScalar(field.Elem())
			}
		case reflect.Slice:
			field.Set(reflect.MakeSlice(field.Type(), 1, 1))
			fillReplicaOptionScalar(field.Index(0))
		case reflect.Map:
			field.Set(reflect.MakeMap(field.Type()))
			item := reflect.New(field.Type().Elem()).Elem()
			if item.Kind() == reflect.Slice {
				item.Set(reflect.MakeSlice(item.Type(), 1, 1))
				fillReplicaOptionScalar(item.Index(0))
			} else {
				fillReplicaOptionScalar(item)
			}
			field.SetMapIndex(reflect.Zero(field.Type().Key()), item)
		case reflect.Interface, reflect.Chan, reflect.UnsafePointer:
			t.Fatalf("unreviewed mutable option type: %s", value.Type().Field(i).Name)
		default:
			fillReplicaOptionScalar(field)
		}
	}
}

func fillReplicaOptionScalar(value reflect.Value) {
	switch value.Kind() {
	case reflect.Bool:
		value.SetBool(true)
	case reflect.Int, reflect.Int64:
		value.SetInt(7)
	case reflect.Uint8:
		value.SetUint(7)
	case reflect.Float64:
		value.SetFloat(0.25)
	case reflect.String:
		value.SetString("fixture")
	}
}

func assertReplicaOptionsDetached(t *testing.T, a, b reflect.Value, path string) {
	t.Helper()
	switch a.Kind() {
	case reflect.Struct:
		for i := 0; i < a.NumField(); i++ {
			if a.Type().Field(i).IsExported() {
				assertReplicaOptionsDetached(t, a.Field(i), b.Field(i), path+"."+a.Type().Field(i).Name)
			}
		}
	case reflect.Pointer:
		if a.IsNil() || b.IsNil() || a.Pointer() == b.Pointer() {
			t.Fatalf("option pointer aliases or disappeared: %s", path)
		}
		assertReplicaOptionsDetached(t, a.Elem(), b.Elem(), path)
	case reflect.Slice, reflect.Map:
		if a.IsNil() || b.IsNil() || a.Pointer() == b.Pointer() || !reflect.DeepEqual(a.Interface(), b.Interface()) {
			t.Fatalf("option container aliases or changed: %s", path)
		}
		if a.Kind() == reflect.Map && a.Type().Elem().Kind() == reflect.Slice {
			for _, key := range a.MapKeys() {
				assertReplicaOptionsDetached(t, a.MapIndex(key), b.MapIndex(key), path+"[value]")
			}
		}
	case reflect.Func:
		if a.Pointer() != b.Pointer() {
			t.Fatalf("callback changed: %s", path)
		}
	default:
		if !reflect.DeepEqual(a.Interface(), b.Interface()) {
			t.Fatalf("option scalar changed: %s", path)
		}
	}
}
