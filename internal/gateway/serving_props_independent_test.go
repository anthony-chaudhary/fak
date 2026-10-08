package gateway

// serving_props_independent_test.go — the ADVERSARIAL second witness for
// GET /props and GET /slots.
//
// This file is written against the CLAIMS serving_props.go makes about itself, not
// against its output. Every test here is built so that a handler which emitted an
// invented constant, a fabricated row, or a plausible-but-second-source number
// would go RED. Where the author's own serving_props_test.go already pins a
// contract shape, these tests deliberately attack a DIFFERENT seam:
//
//	claim 1  total_slots is the LIVE admission cap, not a constant and not MaxBatch
//	claim 2  /slots emits ONE row, and MaxNumSeqs never multiplies it
//	claim 3  /slots counters are the SAME integers /metrics publishes
//	claim 4  endpoint_slots/endpoint_metrics are a projection of the route table
//	claim 5  fak_prefix_cache_hit_rate is absent on an idle process, real after a turn
//	reachability  /props and /slots are REGISTERED in routeTable(), read from the
//	              production source by AST (so deleting a registration is red)
//	              AND the AST agrees with the live runtime table.
//
// House idiom follows serving_metrics_test.go / cachevalue_evict_witness_test.go:
// httptest, no network beyond loopback, no sleeps, deterministic. The package has
// no t.Parallel() anywhere (newTestServer mutates the global ABI registry and
// swapCacheObserver mutates a process-global tap), so only the pure
// total_slots table — which touches no mutable global — runs parallel.

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/pkg/gatewayauth"
)

// ---------------------------------------------------------------------------
// helpers (names prefixed ind* so this file never collides with the author's)
// ---------------------------------------------------------------------------

// indPropsFields decodes a /props body into raw messages so a test can assert a
// key is ABSENT (not defaulted to 0, not a quoted string).
func indPropsFields(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var out map[string]json.RawMessage
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("decode /props body: %v\n--- body ---\n%s", err, body)
	}
	return out
}

// indPropsGet drives the real handler (not the mux) and returns status+body.
func indPropsGet(t *testing.T, srv *Server) (int, []byte) {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.handleProps(rec, httptest.NewRequest(http.MethodGet, "/props", nil))
	return rec.Code, rec.Body.Bytes()
}

// indSlotsGet drives the real handler and decodes the array form.
func indSlotsGet(t *testing.T, srv *Server) (int, []byte) {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.handleSlots(rec, httptest.NewRequest(http.MethodGet, "/slots", nil))
	return rec.Code, rec.Body.Bytes()
}

// indSlotRow decodes a /slots body into raw rows.
func indSlotRow(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var rows []map[string]json.RawMessage
	if err := json.Unmarshal(body, &rows); err != nil {
		t.Fatalf("decode /slots body: %v\n--- body ---\n%s", err, body)
	}
	if len(rows) != 1 {
		t.Fatalf("decoded %d /slots rows, want exactly 1: %s", len(rows), body)
	}
	return rows[0]
}

// indUint reads a JSON key as an exact uint64. Deliberately NOT float64: a
// float64 round-trip would let a value above 2^53 pass a fabricated assertion.
func indUint(t *testing.T, obj map[string]json.RawMessage, key string) uint64 {
	t.Helper()
	raw, ok := obj[key]
	if !ok {
		t.Fatalf("key %q is ABSENT; an unreported field must be omitted, not defaulted", key)
	}
	var n uint64
	if err := json.Unmarshal(raw, &n); err != nil {
		t.Fatalf("key %q is not a JSON uint (%s): %v", key, raw, err)
	}
	return n
}

// indFloat reads a JSON key as a float64 and fails if the key is absent.
func indFloat(t *testing.T, obj map[string]json.RawMessage, key string) float64 {
	t.Helper()
	raw, ok := obj[key]
	if !ok {
		t.Fatalf("key %q is ABSENT", key)
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("key %q is not a JSON number (%s): %v", key, raw, err)
	}
	return f
}

// indScrapeUint parses an UNLABELLED Prometheus counter sample out of a scrape
// and returns it as an exact int64. It fails when the family is missing, so a
// handler cannot "agree" with a family /metrics never published.
func indScrapeUint(t *testing.T, scrape, family string) int64 {
	t.Helper()
	line := metricLine(scrape, family)
	if line == "" {
		t.Fatalf("scrape carries no %q sample:\n--- metrics ---\n%s", family, scrape)
	}
	v, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, family)), 10, 64)
	if err != nil {
		t.Fatalf("parse %q sample %q: %v", family, line, err)
	}
	return v
}

// indScrapeFloat parses a labelled Prometheus gauge sample out of a scrape.
func indScrapeFloat(t *testing.T, scrape, familyWithLabels string) float64 {
	t.Helper()
	line := metricLine(scrape, familyWithLabels)
	if line == "" {
		t.Fatalf("scrape carries no %q sample:\n--- metrics ---\n%s", familyWithLabels, scrape)
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(line, familyWithLabels)), 64)
	if err != nil {
		t.Fatalf("parse %q sample %q: %v", familyWithLabels, line, err)
	}
	return v
}

// indAbsent fails when key is present at all.
func indAbsent(t *testing.T, obj map[string]json.RawMessage, key, why string) {
	t.Helper()
	if raw, ok := obj[key]; ok {
		t.Fatalf("key %q = %s, want ABSENT: %s", key, raw, why)
	}
}

// ---------------------------------------------------------------------------
// CLAIM 1 — total_slots is the LIVE admission cap, not an invented constant
// ---------------------------------------------------------------------------

// TestIndependentPropsTotalSlotsIsReadLiveNotConstant attacks the fabrication
// risk directly: if the handler emitted ANY constant — the shipping default 256,
// the batch default 32, or a literal 1 — at least one of these five policies is
// wrong for it. The four values are chosen so no two are equal and none equals
// DefaultAdmissionPolicy().MaxNumSeqs or DefaultBatchPolicy().MaxBatch.
//
// It then proves the read is LIVE (not a value frozen at construction) by
// re-pointing the SAME server at a second controller through the production
// SetAdmissionController seam and requiring the wire to follow.
func TestIndependentPropsTotalSlotsIsReadLiveNotConstant(t *testing.T) {
	t.Parallel() // pure: reads only its own admission controllers

	// The two constants an implementation would most plausibly fabricate from.
	shippedSeqs := DefaultAdmissionPolicy().MaxNumSeqs
	batchCap := DefaultBatchPolicy().MaxBatch

	for _, maxSeqs := range []int{1, 5, 37, 4096, 65537} {
		if maxSeqs == shippedSeqs || maxSeqs == batchCap {
			t.Fatalf("test fixture %d collides with a plausible fabricated constant", maxSeqs)
		}
		ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: maxSeqs})
		srv := &Server{admissionCtl: ctl}

		code, body := indPropsGet(t, srv)
		if code != http.StatusOK {
			t.Fatalf("MaxNumSeqs=%d: GET /props status = %d, want 200", maxSeqs, code)
		}
		fields := indPropsFields(t, body)

		raw, ok := fields["total_slots"]
		if !ok {
			t.Fatalf("MaxNumSeqs=%d: total_slots ABSENT, want the live cap %d read from AdmissionPolicy.MaxNumSeqs", maxSeqs, maxSeqs)
		}
		var got int
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("MaxNumSeqs=%d: total_slots is not a JSON int (%s): %v", maxSeqs, raw, err)
		}
		if got != maxSeqs {
			t.Fatalf("MaxNumSeqs=%d: total_slots = %d, want the live admission cap: a constant (shippedSeqs=%d, batchCap=%d) cannot pass this table",
				maxSeqs, got, shippedSeqs, batchCap)
		}
		// The disclosure must name the source, or "equal to the policy" is
		// indistinguishable from "equal to a value that happens to match".
		var src string
		if err := json.Unmarshal(fields["fak_total_slots_source"], &src); err != nil || !strings.Contains(src, "MaxNumSeqs") {
			t.Fatalf("MaxNumSeqs=%d: fak_total_slots_source = %q, want it to name AdmissionPolicy.MaxNumSeqs", maxSeqs, string(fields["fak_total_slots_source"]))
		}
	}

	// LIVE, not construction-time: re-point the same Server at a controller
	// carrying a different cap. A handler that cached the first policy, or that
	// emitted a constant, fails here.
	srv := &Server{admissionCtl: NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 11})}
	srv.SetAdmissionController(NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 90210}))
	_, body := indPropsGet(t, srv)
	fields := indPropsFields(t, body)
	var got int
	if err := json.Unmarshal(fields["total_slots"], &got); err != nil {
		t.Fatalf("total_slots is not a JSON int (%s): %v", fields["total_slots"], err)
	}
	if got != 90210 {
		t.Fatalf("after SetAdmissionController(MaxNumSeqs=90210): total_slots = %d, want 90210: the read is not tracking the live admission controller", got)
	}

	// Detaching the controller must retract the claim, not keep the last number.
	srv.SetAdmissionController(nil)
	_, body = indPropsGet(t, srv)
	indAbsent(t, indPropsFields(t, body), "total_slots",
		"the admission controller was detached, so there is no running-set cap to report")
}

// TestIndependentPropsTotalSlotsAbsentWithoutACap pins the two OMIT cases: no
// controller at all (the pure-proxy shape) and a non-positive MaxNumSeqs, which
// admission.go documents as "disables the seq cap" — genuinely uncapped, so a 0
// would read as "an engine with zero slots" and a negative would read as a cap
// nobody can enforce.
func TestIndependentPropsTotalSlotsAbsentWithoutACap(t *testing.T) {
	t.Parallel() // pure

	cases := []struct {
		name string
		srv  *Server
	}{
		{"no_admission_controller", &Server{}},
		{"max_num_seqs_zero_means_uncapped", &Server{admissionCtl: NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 0})}},
		{"negative_max_num_seqs_also_uncapped", &Server{admissionCtl: NewAdmissionController(AdmissionPolicy{MaxNumSeqs: -4})}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			code, body := indPropsGet(t, tc.srv)
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200", code)
			}
			fields := indPropsFields(t, body)
			indAbsent(t, fields, "total_slots", "an uncapped/unattached running set has no slot count to report")
			indAbsent(t, fields, "fak_total_slots_source", "no cap was read, so nothing may name a source")
		})
	}
}

// TestIndependentPropsNeverReadsTheUnwiredBatchPolicy pins the author's stated
// REASON for using MaxNumSeqs: batchsched.go's own HONEST FENCE says its
// composition policy "is not yet wired into the live serve request path", so
// reading BatchPolicy.MaxBatch as an engine slot capacity would be a fabricated
// claim. This asserts that at the SOURCE level — an AST walk, so a mention in a
// comment (the file's header discusses MaxBatch at length) cannot satisfy it.
func TestIndependentPropsNeverReadsTheUnwiredBatchPolicy(t *testing.T) {
	t.Parallel() // reads a source file only

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "serving_props.go", nil, 0)
	if err != nil {
		t.Fatalf("parse serving_props.go: %v", err)
	}
	var offenders []string
	ast.Inspect(file, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if sel.Sel.Name == "MaxBatch" || sel.Sel.Name == "BatchPolicy" {
			offenders = append(offenders, fset.Position(sel.Pos()).String())
		}
		return true
	})
	if len(offenders) > 0 {
		t.Fatalf("serving_props.go reads the unwired batch composition policy at %v: batchsched.go's HONEST FENCE states it is not wired into the live serve path, so reporting it as an engine slot capacity would be a fabricated claim",
			offenders)
	}
}

// ---------------------------------------------------------------------------
// CLAIM 2 — /slots emits exactly ONE row; MaxNumSeqs never multiplies it
// ---------------------------------------------------------------------------

// TestIndependentPropsSlotsRowCountIsNotMaxNumSeqs proves the architectural
// claim that the single-row shape is not a shortcut: at MaxNumSeqs 65537 the
// document still carries exactly one row. A "one row per admission slot"
// implementation — the obvious fabrication — emits 65537 rows here and is red.
//
// (Assessment, not assertion: one row is the honest shape for THIS document.
// Every counter on the row is a process-lifetime tap total, so N rows would be N
// copies of the same global number under N different ids. See the report.)
func TestIndependentPropsSlotsRowCountIsNotMaxNumSeqs(t *testing.T) {
	for _, maxSeqs := range []int{0, 1, 256, 65537} {
		t.Run("max_num_seqs_"+strconv.Itoa(maxSeqs), func(t *testing.T) {
			restore := swapCacheObserver(cacheobs.New())
			defer restore()
			cacheobs.Default.Observe(4096, 2048) // a live turn: counters are non-zero

			srv := &Server{admissionCtl: NewAdmissionController(AdmissionPolicy{MaxNumSeqs: maxSeqs})}
			code, body := indSlotsGet(t, srv)
			if code != http.StatusOK {
				t.Fatalf("GET /slots status = %d, want 200", code)
			}
			var rows []json.RawMessage
			if err := json.Unmarshal(body, &rows); err != nil {
				t.Fatalf("/slots is not a JSON array: %v\n--- body ---\n%s", err, body)
			}
			if len(rows) != 1 {
				t.Fatalf("/slots emitted %d rows at MaxNumSeqs=%d, want exactly 1: fak's in-kernel path has ONE shared resident radix KV prefix, so a per-admission-slot row would be fabricated per-slot KV state",
					len(rows), maxSeqs)
			}
			row := indSlotRow(t, body)
			if got := indUint(t, row, "id"); got != 0 {
				t.Fatalf("the single slot id = %d, want 0 (the one shared KV prefix)", got)
			}
			// The topology disclosure is what stops a reader from reading one row
			// as "one concurrent request"; without it the row count is misleading.
			var model string
			if err := json.Unmarshal(row["fak_slot_model"], &model); err != nil ||
				!strings.Contains(model, "shared") || !strings.Contains(model, "not one slot per request") {
				t.Fatalf("fak_slot_model = %q, want the shared-prefix topology disclosure", string(row["fak_slot_model"]))
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CLAIM 3 — /slots mirrors /metrics; it is not a second source
// ---------------------------------------------------------------------------

// TestIndependentPropsSlotsCountersEqualTheMetricsScrape drives the SAME
// process-global cacheobs tap /metrics renders and then compares the WIRE to the
// SCRAPE numerically — not by string containment against a literal. A handler
// keeping its own tally, reading a different field, or summing per-request
// instead of process-lifetime produces different integers and is red.
//
// The fed turns are deliberately unequal (prompt != reused, four distinct
// running ratios) so an off-by-one accumulation, a swapped field, or a
// "processed = prompt - reused" derivation all diverge.
func TestIndependentPropsSlotsCountersEqualTheMetricsScrape(t *testing.T) {
	restore := swapCacheObserver(cacheobs.New())
	defer restore()

	// Deliberately four turns with distinct realized ratios: 0, 0.45,
	// 0.4666666666666667, 0.48. Cumulative: prompt 5000, reused 2400, turns 4.
	for _, turn := range [][2]int{{1000, 0}, {1000, 900}, {1000, 500}, {2000, 1000}} {
		cacheobs.Default.Observe(turn[0], turn[1])
	}

	srv := newTestServer(t)
	scrape := srv.renderMetrics()

	wantPrompt := indScrapeUint(t, scrape, "fak_gateway_kv_prefix_prompt_tokens_total")
	wantReused := indScrapeUint(t, scrape, "fak_gateway_kv_prefix_reused_tokens_total")
	wantTurns := indScrapeUint(t, scrape, "fak_gateway_kv_prefix_turns_total")
	if wantPrompt != 5000 || wantReused != 2400 || wantTurns != 4 {
		t.Fatalf("scrape totals = prompt %d / reused %d / turns %d, want 5000 / 2400 / 4",
			wantPrompt, wantReused, wantTurns)
	}

	code, body := indSlotsGet(t, srv)
	if code != http.StatusOK {
		t.Fatalf("GET /slots status = %d, want 200", code)
	}
	row := indSlotRow(t, body)

	if got := indUint(t, row, "n_prompt_tokens"); got != uint64(wantPrompt) {
		t.Fatalf("n_prompt_tokens = %d, want the exact /metrics integer %d (fak_gateway_kv_prefix_prompt_tokens_total)", got, wantPrompt)
	}
	if got := indUint(t, row, "n_prompt_tokens_cache"); got != uint64(wantReused) {
		t.Fatalf("n_prompt_tokens_cache = %d, want the exact /metrics integer %d (fak_gateway_kv_prefix_reused_tokens_total)", got, wantReused)
	}
	if got := indUint(t, row, "fak_kv_prefix_turns_total"); got != uint64(wantTurns) {
		t.Fatalf("fak_kv_prefix_turns_total = %d, want the exact /metrics integer %d", got, wantTurns)
	}
	// The reuse ratio must equal the very float /metrics publishes, to the bit.
	wantRatio := indScrapeFloat(t, scrape, "fak_gateway_kv_prefix_reuse_ratio")
	if got := indFloat(t, row, "fak_kv_prefix_reuse_ratio"); got != wantRatio {
		t.Fatalf("fak_kv_prefix_reuse_ratio = %v, want the exact /metrics float %v", got, wantRatio)
	}

	// DIVERGENCE CANNOT ACCUMULATE SILENTLY: read the wire, feed more turns,
	// re-read both surfaces. If /slots held its own tally it would lag.
	cacheobs.Default.Observe(800, 800)
	scrape2 := srv.renderMetrics()
	row2 := indSlotRow(t, func() []byte { _, b := indSlotsGet(t, srv); return b }())
	if got, want := indUint(t, row2, "n_prompt_tokens"), uint64(indScrapeUint(t, scrape2, "fak_gateway_kv_prefix_prompt_tokens_total")); got != want {
		t.Fatalf("after a fifth turn n_prompt_tokens = %d, want %d: the wire and the scrape have diverged", got, want)
	}
	if got, want := indUint(t, row2, "n_prompt_tokens_cache"), uint64(indScrapeUint(t, scrape2, "fak_gateway_kv_prefix_reused_tokens_total")); got != want {
		t.Fatalf("after a fifth turn n_prompt_tokens_cache = %d, want %d: the wire and the scrape have diverged", got, want)
	}

	// Every mirrored field must name the family it mirrors, on the wire.
	var src map[string]string
	if err := json.Unmarshal(row["fak_source"], &src); err != nil {
		t.Fatalf("fak_source is not an object: %v", err)
	}
	for field, family := range map[string]string{
		"n_prompt_tokens":           "fak_gateway_kv_prefix_prompt_tokens_total",
		"n_prompt_tokens_cache":     "fak_gateway_kv_prefix_reused_tokens_total",
		"fak_kv_prefix_turns_total": "fak_gateway_kv_prefix_turns_total",
	} {
		if got := src[field]; got != family {
			t.Errorf("fak_source[%q] = %q, want %q", field, got, family)
		}
		// The named family must be a family this scrape really published, or the
		// provenance disclosure points at nothing.
		if metricLine(scrape, family) == "" {
			t.Errorf("fak_source[%q] names %q, but this scrape publishes no such sample", field, family)
		}
	}
}

// ---------------------------------------------------------------------------
// CLAIM 4 — endpoint_slots / endpoint_metrics project the LIVE route table
// ---------------------------------------------------------------------------

// TestIndependentPropsEndpointFlagsTrackTheRouteTable proves the flags are a
// MEMBERSHIP TEST against routeTable(), not two hardcoded true booleans. The
// positive side asserts every pattern the live table registers projects true;
// the negative side asserts a spread of patterns the table does NOT register
// project false. An implementation that returned true unconditionally, or
// hardcoded {"/props", "/metrics"}, is red on the negative side.
//
// It also cross-checks the flag the wire actually emits against the table the
// test enumerates itself, so the boolean on the wire is pinned to the table
// rather than to the handler's own opinion.
func TestIndependentPropsEndpointFlagsTrackTheRouteTable(t *testing.T) {
	srv := newTestServer(t)

	registered := map[string]bool{}
	for _, rt := range srv.routeTable() {
		registered[rt.pattern] = true
	}

	// The invariant, stated without a precondition so a REMOVED route fails the
	// real assertion rather than a guard: for every pattern, the projection and
	// the wire must equal live membership in routeTable(). A route deletion flips
	// membership, and a hardcoded true cannot follow it down.
	for _, pattern := range []string{"/props", "/slots", "/metrics"} {
		want := registered[pattern]
		if got := srv.servingRouteServed(pattern); got != want {
			t.Errorf("servingRouteServed(%q) = %v, want %v (membership in the live route table)", pattern, got, want)
		}
	}

	// Negative: patterns the table does not register must project false. A
	// hardcoded-true or hardcoded-pair implementation fails here.
	for _, absent := range []string{
		"/propsx", "/slot", "/slots/", "/v2/props", "/v2/slots",
		"/PROPS", "/Slots", "/metrics/", "/not-a-fak-route",
	} {
		if registered[absent] {
			t.Fatalf("precondition: %q IS registered in the live route table", absent)
		}
		if srv.servingRouteServed(absent) {
			t.Errorf("servingRouteServed(%q) = true, want false: %q is NOT in routeTable(), so the flag is not a projection of the live table", absent, absent)
		}
	}

	// On the wire, both flags must equal the membership the test computed.
	_, body := indPropsGet(t, srv)
	fields := indPropsFields(t, body)
	for key, pattern := range map[string]string{"endpoint_slots": "/slots", "endpoint_metrics": "/metrics"} {
		var b bool
		if err := json.Unmarshal(fields[key], &b); err != nil {
			t.Fatalf("%s is not a JSON bool (%s): %v", key, fields[key], err)
		}
		if want := registered[pattern]; b != want {
			t.Errorf("%s = %v, want %v (membership of %q in the live route table): the endpoint claim did not follow the route table", key, b, want, pattern)
		}
	}
}

// TestIndependentPropsEndpointFlagsAreComputedNotHardcoded closes the one hole
// the runtime assertions above cannot reach: with /slots and /metrics always
// registered, a wire field hardcoded to `true` agrees with the membership test on
// every live case. So this pins the SOURCE SHAPE: in servingProps' struct
// literal, endpoint_slots and endpoint_metrics must be CALL expressions (the
// route-table projection), never bare literals.
//
// A mutation run confirmed the hole is real — overwriting
// `EndpointSlots: s.servingRouteServed("/slots")` with `EndpointSlots: true`
// passes every wire-level assertion; this test is red for it.
func TestIndependentPropsEndpointFlagsAreComputedNotHardcoded(t *testing.T) {
	t.Parallel() // reads a source file only

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "serving_props.go", nil, 0)
	if err != nil {
		t.Fatalf("parse serving_props.go: %v", err)
	}

	fields := map[string]ast.Expr{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "servingProps" || fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			kv, ok := n.(*ast.KeyValueExpr)
			if !ok {
				return true
			}
			key, ok := kv.Key.(*ast.Ident)
			if !ok {
				return true
			}
			if key.Name == "EndpointSlots" || key.Name == "EndpointMetrics" {
				fields[key.Name] = kv.Value
			}
			return true
		})
	}
	if len(fields) != 2 {
		t.Fatalf("found %d of {EndpointSlots, EndpointMetrics} assignments in servingProps: %v; this oracle is not reading what it claims to", len(fields), fields)
	}
	for _, name := range []string{"EndpointSlots", "EndpointMetrics"} {
		expr := fields[name]
		if _, ok := expr.(*ast.CallExpr); !ok {
			t.Errorf("%s = %T (%s), want a CALL expression reading the live route table: a hardcoded literal makes the endpoint claim survive the route being removed, which is exactly the fabrication this surface must not ship",
				name, expr, fset.Position(expr.Pos()))
		}
	}
}

// ---------------------------------------------------------------------------
// CLAIM 5 — no phantom ratio on an idle process
// ---------------------------------------------------------------------------

// TestIndependentPropsHitRateAbsentWhenIdleRealAfterATurn pins the difference
// between "unreported" and "reported zero". On a cold tap the key must be
// ABSENT — a 0.0 would be a fabricated reading of a cache that has never served.
// After a turn it must be PRESENT and carry the tap's exact realized ratio, and
// it must TRACK the tap across several distinct ratios so a hardcoded 0.0 (or a
// hardcoded 1.0, or "present whenever a controller exists") is red.
func TestIndependentPropsHitRateAbsentWhenIdleRealAfterATurn(t *testing.T) {
	// --- idle: with and without an admission controller, the key is absent ---
	for _, tc := range []struct {
		name string
		srv  *Server
	}{
		{"idle_no_controller", &Server{}},
		{"idle_with_controller", &Server{admissionCtl: NewAdmissionController(DefaultAdmissionPolicy())}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			restore := swapCacheObserver(cacheobs.New())
			defer restore()

			code, body := indPropsGet(t, tc.srv)
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200", code)
			}
			indAbsent(t, indPropsFields(t, body), "fak_prefix_cache_hit_rate",
				"no turn has been observed, so a reported 0.0 would be a fabricated cache reading")
		})
	}

	// --- after real turns: present, exact, and tracking ---
	t.Run("present_and_tracks_the_tap", func(t *testing.T) {
		restore := swapCacheObserver(cacheobs.New())
		defer restore()

		srv := &Server{admissionCtl: NewAdmissionController(DefaultAdmissionPolicy())}
		// Cumulative realized ratios 0.4, 0.65, 0.6, 0.56 — four DISTINCT nonzero
		// readings. A fabricated 0.0, a fabricated 1.0, a hardcoded constant,
		// and a value that stops tracking the tap each fail a different step; the
		// bit-exact comparison against Snapshot() catches every one of them.
		for i, turn := range [][2]int{{1000, 400}, {1000, 900}, {1000, 500}, {2000, 1000}} {
			cacheobs.Default.Observe(turn[0], turn[1])
			_, body := indPropsGet(t, srv)
			got := indFloat(t, indPropsFields(t, body), "fak_prefix_cache_hit_rate")
			want := cacheobs.Default.Snapshot().ReuseRatio
			if got != want {
				t.Fatalf("turn %d: fak_prefix_cache_hit_rate = %v, want the live tap ratio %v", i+1, got, want)
			}
			if got <= 0 {
				t.Fatalf("turn %d: fak_prefix_cache_hit_rate = %v, want a positive realized ratio: the tap has booked %d reused of %d prompt tokens, so 0 is a fabricated reading",
					i+1, got, turn[1], turn[0])
			}
		}
	})
}

// TestIndependentPropsHitRateEqualsTheServingGauge pins the specific family the
// handler names for this field: the value must be the SAME float
// fak_serving_prefix_cache_hit_rate publishes. That family only renders when a
// native serving row exists, so the test installs an admission controller (the
// production path that makes nativeServingMetricRow present) on a real server.
func TestIndependentPropsHitRateEqualsTheServingGauge(t *testing.T) {
	restore := swapCacheObserver(cacheobs.New())
	defer restore()
	cacheobs.Default.Observe(1200, 900) // ratio 0.75

	srv := newTestServer(t)
	srv.SetAdmissionController(NewAdmissionController(DefaultAdmissionPolicy()))

	scrape := srv.renderMetrics()
	const labels = `{worker="local",engine="test",model="test-model"}`
	want := indScrapeFloat(t, scrape, "fak_serving_prefix_cache_hit_rate"+labels)
	if want != 0.75 {
		t.Fatalf("precondition: fak_serving_prefix_cache_hit_rate = %v, want the 0.75 the tap earned", want)
	}

	_, body := indPropsGet(t, srv)
	got := indFloat(t, indPropsFields(t, body), "fak_prefix_cache_hit_rate")
	if got != want {
		t.Fatalf("fak_prefix_cache_hit_rate = %v, want the exact /metrics value %v (fak_serving_prefix_cache_hit_rate): the two surfaces disagree", got, want)
	}
}

// ---------------------------------------------------------------------------
// REACHABILITY — both routes are REGISTERED, read from the production source
// ---------------------------------------------------------------------------

// indRegisteredRoutes parses internal/gateway/http.go and returns the
// pattern -> handler-selector pairs registered by routeTable(). Reading the
// production SOURCE (not calling routeTable()) is deliberate: a routeTable()
// call would keep passing if the registration moved out of the slice the mux is
// built from, which is the exact failure mode this asserts against.
func indRegisteredRoutes(t *testing.T) map[string]string {
	t.Helper()
	src, err := os.ReadFile("http.go")
	if err != nil {
		t.Fatalf("read http.go: %v", err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "http.go", src, 0)
	if err != nil {
		t.Fatalf("parse http.go: %v", err)
	}

	out := map[string]string{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "routeTable" || fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			lit, ok := n.(*ast.CompositeLit)
			if !ok {
				return true
			}
			for _, elt := range lit.Elts {
				// routeTable writes its entries POSITIONALLY
				// ({"pattern", s.handler}, not {pattern: ..., handler: ...}),
				// so each element is itself a CompositeLit. Accept the keyed
				// form too so a reformat cannot silently disarms this oracle.
				var patternExpr ast.Expr
				var handlerExpr ast.Expr
				switch e := elt.(type) {
				case *ast.CompositeLit:
					if len(e.Elts) == 2 {
						patternExpr = e.Elts[0]
						handlerExpr = e.Elts[1]
					}
				case *ast.KeyValueExpr:
					patternExpr = e.Key
					handlerExpr = e.Value
				default:
					continue
				}
				var pattern string
				switch e := patternExpr.(type) {
				case *ast.BasicLit:
					if e.Kind != token.STRING {
						continue
					}
					pattern, err = strconv.Unquote(e.Value)
					if err != nil {
						continue
					}
				case *ast.SelectorExpr:
					pkg, ok := e.X.(*ast.Ident)
					if !ok || pkg.Name != "gatewayauth" || e.Sel.Name != "KeyProofPath" {
						continue
					}
					pattern = gatewayauth.KeyProofPath
				default:
					continue
				}
				handler := "<not-a-method-value>"
				if sel, ok := handlerExpr.(*ast.SelectorExpr); ok {
					if recv, ok := sel.X.(*ast.Ident); ok {
						handler = recv.Name + "." + sel.Sel.Name
					}
				}
				out[pattern] = handler
			}
			return true
		})
	}
	if len(out) == 0 {
		t.Fatal("no route registrations parsed out of routeTable() in http.go; the AST walk is broken and this test proves nothing")
	}
	return out
}

// TestPropsRouteRegistrationIsReachableInSource is the reachability oracle,
// independent of the author's: it parses http.go, requires BOTH registrations to
// name the right handler, requires the runtime table to agree with the parsed
// source (so a stale-file match cannot pass), and then drives the LIVE mux over
// loopback to prove each pattern dispatches its own document rather than falling
// through to the "/" catch-all.
//
// Deleting either registration from routeTable() turns this red twice: the AST
// check fails AND the live GET 404s.
func TestPropsRouteRegistrationIsReachableInSource(t *testing.T) {
	parsed := indRegisteredRoutes(t)

	for pattern, handler := range map[string]string{
		"/props": "s.handleProps",
		"/slots": "s.handleSlots",
	} {
		got, ok := parsed[pattern]
		if !ok {
			t.Errorf("http.go routeTable() does not register %q: the handler exists but is unreachable, which is the 404 this surface exists to remove", pattern)
			continue
		}
		if got != handler {
			t.Errorf("http.go registers %q -> %s, want %s", pattern, got, handler)
		}
	}

	// The read-scoped auth floor must admit both, checked by CALLING the
	// production predicate rather than by grepping its source.
	for _, path := range []string{"/props", "/slots"} {
		if !readScopedPath(httptest.NewRequest(http.MethodGet, path, nil)) {
			t.Errorf("readScopedPath(%q) = false, want true: a consumer admitted to /metrics must be admitted to the introspection that explains it", path)
		}
	}

	// The parsed source must equal the LIVE table — otherwise the AST check
	// could be satisfied by a registration that is no longer built into the mux.
	srv := newTestServer(t)
	live := map[string]string{}
	for _, rt := range srv.routeTable() {
		live[rt.pattern] = "<method-value>"
	}
	for pattern := range parsed {
		if _, ok := live[pattern]; !ok {
			t.Errorf("http.go registers %q but the live routeTable() does not contain it: the mux and the source have drifted", pattern)
		}
	}
	for pattern := range live {
		if _, ok := parsed[pattern]; !ok {
			t.Errorf("live routeTable() serves %q but http.go's routeTable() does not register it: the source and the served surface have drifted", pattern)
		}
	}

	// Registered AND dispatching: drive the real mux over loopback.
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for path, wantKey := range map[string]string{
		"/props": `"fak_schema":"` + servingPropsSchema + `"`,
		"/slots": `"fak_scope":"`,
	} {
		resp, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		body := func() []byte {
			defer resp.Body.Close()
			b, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("read %s body: %v", path, err)
			}
			return b
		}()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s status = %d, want 200 (registered but not serving; a fall-through to the / catch-all means the route is unrouted)", path, resp.StatusCode)
			continue
		}
		if !strings.Contains(string(body), wantKey) {
			t.Errorf("GET %s body does not carry %q:\n%s", path, wantKey, body)
		}
		if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "application/json") {
			t.Errorf("GET %s Content-Type = %q, want application/json (a non-JSON body means the pattern dispatched to the wrong handler)", path, ct)
		}
	}
}
