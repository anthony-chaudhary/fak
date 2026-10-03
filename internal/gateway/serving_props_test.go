package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// serving_props_test.go — the /props and /slots contract witness.
//
// The two tests here are deliberately different kinds of witness:
//
//   - TestServingProps exercises the HANDLERS and asserts the wire contract a
//     llama-server-shaped consumer parses: 200, and every required key present
//     with a type it can decode. It is table-driven over server SHAPES because
//     the honest answer differs by shape — an engine with no admission
//     controller must OMIT total_slots rather than report a zero, because
//     "unreported" and "reported zero" are different facts and only the absent
//     key says the first.
//   - TestPropsRouteReachability asserts the REGISTRATION, reading the real
//     http.go so it fails if the route is deleted from routeTable() (a handler
//     that exists but is unrouted is exactly the 404 this work set out to fix),
//     and then drives the live mux to prove the pattern really dispatches.
//
// The honesty assertions are the load-bearing ones: the admission cases pin
// total_slots to a policy value chosen by the TEST, so a handler that returned
// an invented constant could not pass them, and the no-admission case pins the
// field's ABSENCE.

// propsKeys decodes a JSON object into raw messages so a test can assert both
// that a key is present and that it carries a JSON number rather than a quoted
// string — the distinction a flexUint consumer cares about.
func propsKeys(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var out map[string]json.RawMessage
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("decode body: %v\n--- body ---\n%s", err, body)
	}
	return out
}

func servingPropsGet(t *testing.T, srv *Server) (int, []byte) {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.handleProps(rec, httptest.NewRequest(http.MethodGet, "/props", nil))
	return rec.Code, rec.Body.Bytes()
}

func servingSlotsGet(t *testing.T, srv *Server) (int, []byte) {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.handleSlots(rec, httptest.NewRequest(http.MethodGet, "/slots", nil))
	return rec.Code, rec.Body.Bytes()
}

// requireJSONNumber asserts a key decodes as a bare JSON number, which is the
// only form a flexUint / int consumer will read as a value.
func requireJSONNumber(t *testing.T, obj map[string]json.RawMessage, key string) float64 {
	t.Helper()
	raw, ok := obj[key]
	if !ok {
		t.Fatalf("key %q is absent; an unreported field must be omitted, not defaulted", key)
	}
	var n float64
	if err := json.Unmarshal(raw, &n); err != nil {
		t.Fatalf("key %q is not a JSON number (%s): %v", key, raw, err)
	}
	return n
}

// TestServingProps asserts the GET /props and GET /slots wire contract across
// the server shapes the gateway actually runs in. Table-driven over shape, as
// the sibling serving-metrics tests are over emitter fixture.
func TestServingProps(t *testing.T) {
	t.Run("props", func(t *testing.T) {
		for _, tt := range []struct {
			name string
			srv  *Server
			// wantTotalSlots is the admission MaxNumSeqs the test pinned into the
			// policy, or the sentinel -1 meaning the key must be ABSENT. The
			// sentinel is explicit on every case because the zero value is a
			// legitimate capacity and would otherwise be indistinguishable from
			// "this case did not say".
			wantTotalSlots int
		}{
			{
				// The pure-proxy shape: no admission controller is installed, so
				// there is no in-kernel running set and NO slot count to report.
				// Emitting 0 here would read as "an engine with zero slots", which
				// is a claim this server cannot make.
				name:           "no_admission_controller_omits_total_slots",
				srv:            &Server{},
				wantTotalSlots: -1,
			},
			{
				// A pinned, non-default MaxNumSeqs. A handler returning an invented
				// constant cannot pass this: the number must come from the policy.
				name:           "reports_the_live_admission_running_set_cap",
				srv:            &Server{admissionCtl: NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 7})},
				wantTotalSlots: 7,
			},
			{
				// MaxNumSeqs <= 0 DISABLES the seq cap, so there is genuinely no
				// bound to report — the same absent-key posture as no controller.
				name:           "uncapped_running_set_omits_total_slots",
				srv:            &Server{admissionCtl: NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 0})},
				wantTotalSlots: -1,
			},
			{
				// The shipping default, read back from the policy itself rather
				// than retyped, so the test and the handler cannot drift.
				name:           "shipping_default_cap",
				srv:            &Server{admissionCtl: NewAdmissionController(DefaultAdmissionPolicy())},
				wantTotalSlots: DefaultAdmissionPolicy().MaxNumSeqs,
			},
		} {
			t.Run(tt.name, func(t *testing.T) {
				code, body := servingPropsGet(t, tt.srv)
				if code != http.StatusOK {
					t.Fatalf("GET /props status = %d, want 200", code)
				}
				fields := propsKeys(t, body)

				// The consumer-required shape, whatever the server shape.
				if _, ok := fields["default_generation_settings"]; !ok {
					t.Error("default_generation_settings absent; the consumer requires the object")
				}
				if _, ok := fields["build_info"]; !ok {
					t.Error("build_info absent; the consumer requires the string")
				}
				for _, key := range []string{"endpoint_slots", "endpoint_metrics"} {
					var b bool
					raw, ok := fields[key]
					if !ok {
						t.Errorf("%s absent; the consumer requires the bool", key)
						continue
					}
					if err := json.Unmarshal(raw, &b); err != nil {
						t.Errorf("%s is not a JSON bool (%s): %v", key, raw, err)
						continue
					}
					if !b {
						t.Errorf("%s = false, want true: this build serves the route", key)
					}
				}

				// total_slots: present and equal to the live policy, or absent.
				if tt.wantTotalSlots < 0 {
					if raw, ok := fields["total_slots"]; ok {
						t.Errorf("total_slots = %s, want the key ABSENT: this server shape has no running-set cap to report", raw)
					}
				} else {
					if got := requireJSONNumber(t, fields, "total_slots"); int(got) != tt.wantTotalSlots {
						t.Errorf("total_slots = %v, want the live admission MaxNumSeqs %d", got, tt.wantTotalSlots)
					}
					if strings.TrimSpace(string(fields["fak_total_slots_source"])) == "" {
						t.Error("fak_total_slots_source is empty; a declared cap must name the field it was read from")
					}
				}

				// Nothing in this table installs an in-kernel planner, so n_ctx
				// has NO live source and must be omitted rather than defaulted
				// to a zero-token engine.
				var dgs map[string]json.RawMessage
				if raw, ok := fields["default_generation_settings"]; ok {
					if err := json.Unmarshal(raw, &dgs); err != nil {
						t.Fatalf("default_generation_settings is not an object (%s): %v", raw, err)
					}
					if raw, ok := dgs["n_ctx"]; ok {
						t.Errorf("n_ctx = %s, want ABSENT: no in-kernel model declared a context window", raw)
					}
				}
			})
		}
	})

	t.Run("slots", func(t *testing.T) {
		// The process-global cacheobs tap may carry counts from sibling tests, so
		// swap in a fresh observer — the house idiom (vcache_score_test.go) — and
		// assert against a genuinely idle tap.
		restore := swapCacheObserver(cacheobs.New())
		defer restore()

		// Nothing has been observed by the tap, so every counter must be a
		// truthful zero AND the scope disclosure must be present — a bare zero
		// with no scope reads as "this engine reuses nothing".
		srv := &Server{}
		code, body := servingSlotsGet(t, srv)
		if code != http.StatusOK {
			t.Fatalf("GET /slots status = %d, want 200", code)
		}
		var slots []map[string]json.RawMessage
		if err := json.Unmarshal(body, &slots); err != nil {
			t.Fatalf("GET /slots body is not a JSON array: %v\n--- body ---\n%s", err, body)
		}
		if len(slots) != 1 {
			t.Fatalf("GET /slots returned %d entries, want exactly 1: fak's in-kernel path has ONE shared resident KV prefix, so there is no per-session slot list to enumerate", len(slots))
		}
		slot := slots[0]
		for _, key := range []string{"n_prompt_tokens", "n_prompt_tokens_processed", "n_prompt_tokens_cache"} {
			if got := requireJSONNumber(t, slot, key); got != 0 {
				t.Errorf("%s = %v, want 0 on an idle tap", key, got)
			}
		}
		// The honesty disclosure must be on the wire, not only in a comment.
		var scope string
		if err := json.Unmarshal(slot["fak_scope"], &scope); err != nil || strings.TrimSpace(scope) == "" {
			t.Errorf("fak_scope must state that these are process-lifetime totals, got %q", string(slot["fak_scope"]))
		}
		var model string
		if err := json.Unmarshal(slot["fak_slot_model"], &model); err != nil || strings.TrimSpace(model) == "" {
			t.Errorf("fak_slot_model must state fak's shared-prefix KV topology, got %q", string(slot["fak_slot_model"]))
		}
		var source map[string]string
		if err := json.Unmarshal(slot["fak_source"], &source); err != nil {
			t.Fatalf("fak_source is not an object: %v", err)
		}
		if got := source["n_prompt_tokens_cache"]; got != "fak_gateway_kv_prefix_reused_tokens_total" {
			t.Errorf("fak_source[n_prompt_tokens_cache] = %q, want the /metrics family the counter mirrors", got)
		}
	})

	t.Run("slots_mirrors_the_metrics_tap", func(t *testing.T) {
		// The whole point of this surface: /slots must read the SAME live state
		// /metrics renders, not a second tally. Feeding the tap and asserting
		// both the wire and the scrape carry the identical integers is what pins
		// that. swapCacheObserver keeps the process-global tap hermetic for the
		// rest of the package.
		restore := swapCacheObserver(cacheobs.New())
		defer restore()
		cacheobs.Default.Observe(900, 256)

		scrape := newTestServer(t).renderMetrics()
		for _, want := range []string{
			"fak_gateway_kv_prefix_prompt_tokens_total 900",
			"fak_gateway_kv_prefix_reused_tokens_total 256",
		} {
			if !strings.Contains(scrape, want) {
				t.Fatalf("scrape does not carry the fed total %q\n--- metrics ---\n%s", want, scrape)
			}
		}

		code, body := servingSlotsGet(t, &Server{})
		if code != http.StatusOK {
			t.Fatalf("GET /slots status = %d, want 200", code)
		}
		var slots []map[string]json.RawMessage
		if err := json.Unmarshal(body, &slots); err != nil {
			t.Fatalf("decode slots: %v", err)
		}
		if got := requireJSONNumber(t, slots[0], "n_prompt_tokens"); got != 900 {
			t.Errorf("n_prompt_tokens = %v, want the live 900", got)
		}
		if got := requireJSONNumber(t, slots[0], "n_prompt_tokens_cache"); got != 256 {
			t.Errorf("n_prompt_tokens_cache = %v, want the live 256", got)
		}
	})
}

// TestPropsRouteReachability is the registration witness: it fails if the
// /props or /slots route is deleted from routeTable(). A handler that exists but
// is unrouted is exactly the 404 that made a Fak-native engine unobservable to a
// llama-server-shaped consumer, so the registration is part of the deliverable.
//
// It reads the REAL http.go (the house idiom, as in windowgate_test.go and
// cmd/fak/guard_managed_cache_test.go) rather than calling routeTable(), because
// a routeTable()-driven assertion would keep passing if the registration moved
// out of the table the way the mux builds it. Then it drives the live mux to
// prove the pattern genuinely dispatches.
func TestPropsRouteReachability(t *testing.T) {
	src, err := os.ReadFile("http.go")
	if err != nil {
		t.Fatalf("read http.go: %v", err)
	}
	text := string(src)

	for _, want := range []string{
		`{"/props", s.handleProps}`,
		`{"/slots", s.handleSlots}`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("http.go must register %s in routeTable(): without the registration the handler is unreachable and a llama-server-shaped consumer reads a 404", want)
		}
	}
	// The read-scoped auth floor must cover the pair, exactly as it covers
	// /metrics: a consumer allowed to read the metrics that explain the engine
	// must be allowed to read the introspection that describes it.
	if !strings.Contains(text, `"/props", "/slots":`) {
		t.Error("http.go must admit /props and /slots in readScopedPath alongside /metrics")
	}

	// Registration present AND dispatching: drive the real mux.
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for path, wantKey := range map[string]string{
		"/props": "default_generation_settings",
		"/slots": "n_prompt_tokens",
	} {
		resp, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		body := readAndCloseBody(t, resp)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s status = %d, want 200 (the route is registered but not serving)", path, resp.StatusCode)
			continue
		}
		if !strings.Contains(string(body), wantKey) {
			t.Errorf("GET %s body does not carry %q:\n%s", path, wantKey, body)
		}
	}
}

// readAndCloseBody drains and closes a probe response body.
func readAndCloseBody(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return body
}
