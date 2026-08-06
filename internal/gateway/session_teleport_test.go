package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// armFixture is the re-arm state the teleport tests carry: a session part-way
// through a bounded allotment, already tainted, on its second re-arm.
func armFixture() TeleportArm {
	return TeleportArm{
		Budget:         SessionBudget{TurnsLeft: 4, TokensLeft: 120_000, ContextTokensLeft: 64_000},
		TaintHighWater: "tainted",
		Generation:     2,
	}
}

// seedHostA builds the source host's ledger: one established session with a few
// turns of history on it. Returns the ledger and the trace.
func seedHostA(t *testing.T) (*sessionledger.Ledger, string) {
	t.Helper()
	l := sessionledger.Memory()
	const trace = "sess:teleport-1"
	for _, e := range []struct{ kind, content string }{
		{sessionledger.KindEstablish, `{"surface":"cli","conversation":"c1"}`},
		{"turn_start", `{"messages":3,"raw_bytes":812}`},
		{"turn_complete", `{"output_tokens":140}`},
		{"turn_start", `{"messages":5,"raw_bytes":1904}`},
	} {
		if _, err := l.Append(trace, e.kind, []byte(e.content)); err != nil {
			t.Fatalf("seed %s: %v", e.kind, err)
		}
	}
	return l, trace
}

// TestSessionTeleportRoundTrip is issue #2419's witness: export the session on
// gateway A, import it on gateway B, and require B's next-turn resident window to
// be BYTE-IDENTICAL to the one A would have sent. Then require that a tampered
// bundle is refused rather than imported.
func TestSessionTeleportRoundTrip(t *testing.T) {
	hostA, trace := seedHostA(t)
	arm := armFixture()

	// What host A would put in front of the model on its next turn.
	windowA, err := TeleportWindow(hostA, trace, arm)
	if err != nil {
		t.Fatalf("host A window: %v", err)
	}

	bundle, err := ExportTeleport(hostA, trace, arm)
	if err != nil {
		t.Fatalf("export on host A: %v", err)
	}
	if bundle.Schema != TeleportSchema || bundle.TraceID != trace {
		t.Fatalf("bundle identity = %q/%q, want %q/%q", bundle.Schema, bundle.TraceID, TeleportSchema, trace)
	}
	if bundle.Head != hostA.Head(trace) {
		t.Fatalf("bundle head = %s, want host A's head %s", bundle.Head, hostA.Head(trace))
	}
	if len(bundle.Entries) != 4 {
		t.Fatalf("bundle carries %d entries, want the whole 4-entry closure", len(bundle.Entries))
	}

	// The bundle crosses the host boundary as bytes, not as a Go value.
	wire, err := json.Marshal(bundle)
	if err != nil {
		t.Fatalf("marshal bundle: %v", err)
	}
	var arrived TeleportBundle
	if err := json.Unmarshal(wire, &arrived); err != nil {
		t.Fatalf("unmarshal bundle: %v", err)
	}

	// Host B has never seen this session.
	hostB := sessionledger.Memory()
	if hostB.Head(trace) != "" {
		t.Fatalf("host B is not empty")
	}
	rederived, err := ImportTeleport(hostB, arrived)
	if err != nil {
		t.Fatalf("import on host B: %v", err)
	}
	if rederived.Head != bundle.Head {
		t.Fatalf("host B re-derived head %s, want %s", rederived.Head, bundle.Head)
	}
	if rederived.Closure != bundle.Closure {
		t.Fatalf("host B closure %s != host A closure %s", rederived.Closure, bundle.Closure)
	}

	// THE ACCEPTANCE: B's next-turn resident window is byte-identical to A's.
	windowB, err := TeleportWindow(hostB, trace, arrived.Arm)
	if err != nil {
		t.Fatalf("host B window: %v", err)
	}
	if !bytes.Equal(windowA, windowB) {
		t.Fatalf("resident window diverged across the hop:\n A: %s\n B: %s", windowA, windowB)
	}

	// The re-arm state survived: budget is not refunded, taint is not laundered.
	if rederived.Arm != arm {
		t.Fatalf("re-arm state = %+v, want %+v", rederived.Arm, arm)
	}
}

// TestSessionTeleportRefusesTamperedBundle is the other half of #2419's acceptance:
// chain verification failure on a tampered bundle refuses import. Each case edits
// one field of a valid bundle; every one must be refused, and must leave the
// receiving host untouched.
func TestSessionTeleportRefusesTamperedBundle(t *testing.T) {
	hostA, trace := seedHostA(t)
	good, err := ExportTeleport(hostA, trace, armFixture())
	if err != nil {
		t.Fatalf("export: %v", err)
	}

	// clone deep-copies the bundle so one case's edit cannot leak into the next.
	clone := func() TeleportBundle {
		var c TeleportBundle
		b, err := json.Marshal(good)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if err := json.Unmarshal(b, &c); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		return c
	}

	cases := []struct {
		name   string
		tamper func(*TeleportBundle)
		want   string
	}{
		{"content edited", func(b *TeleportBundle) {
			b.Entries[1].Content = json.RawMessage(`{"messages":3,"raw_bytes":9999}`)
		}, "chain verification failed"},
		{"kind edited", func(b *TeleportBundle) {
			b.Entries[2].Kind = "turn_forged"
		}, "chain verification failed"},
		{"entry dropped from the middle", func(b *TeleportBundle) {
			b.Entries = append(b.Entries[:1], b.Entries[2:]...)
		}, "chain verification failed"},
		{"hash restamped to match edited content", func(b *TeleportBundle) {
			// The strongest case: an editor who also rewrites the hash it broke. The
			// re-stamped entry no longer chains to its successor's parent.
			b.Entries[1].Content = json.RawMessage(`{"messages":3,"raw_bytes":9999}`)
			b.Entries[1].Hash = sessionledger.Hash(strings.Repeat("a", 64))
		}, "chain verification failed"},
		{"head redirected", func(b *TeleportBundle) {
			b.Head = sessionledger.Hash(strings.Repeat("b", 64))
		}, "closure seal"},
		{"budget inflated", func(b *TeleportBundle) {
			b.Arm.Budget.TokensLeft = 1 << 30
		}, "closure seal"},
		{"taint laundered clean", func(b *TeleportBundle) {
			b.Arm.TaintHighWater = "trusted"
		}, "closure seal"},
		{"schema downgraded", func(b *TeleportBundle) {
			b.Schema = "fak.session.teleport.v0"
		}, "unknown schema"},
		{"emptied", func(b *TeleportBundle) {
			b.Entries = nil
		}, "no entries"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := clone()
			tc.tamper(&b)
			if err := b.Verify(); err == nil {
				t.Fatalf("Verify accepted a tampered bundle")
			} else if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Verify error = %v, want it to mention %q", err, tc.want)
			}
			hostB := sessionledger.Memory()
			if _, err := ImportTeleport(hostB, b); err == nil {
				t.Fatalf("ImportTeleport accepted a tampered bundle")
			}
			if hostB.Head(trace) != "" || hostB.NodeCount() != 0 {
				t.Fatalf("a refused import still wrote to the receiving host (head %q, %d nodes)",
					hostB.Head(trace), hostB.NodeCount())
			}
		})
	}

	// The untampered bundle still imports, so the cases above are refusing the edit
	// and not something structural about the fixture.
	if _, err := ImportTeleport(sessionledger.Memory(), good); err != nil {
		t.Fatalf("the untampered bundle must still import: %v", err)
	}
}

// TestSessionTeleportImportRefusesOccupiedTrace pins the other refusal: import
// re-derives from the root, which is only truthful onto an empty trace.
func TestSessionTeleportImportRefusesOccupiedTrace(t *testing.T) {
	hostA, trace := seedHostA(t)
	b, err := ExportTeleport(hostA, trace, armFixture())
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if _, err := ImportTeleport(hostA, b); err == nil {
		t.Fatalf("import onto the trace's own live host must be refused")
	} else if !strings.Contains(err.Error(), "already holds history") {
		t.Fatalf("error = %v, want the resident-trace refusal", err)
	}
}

// TestSessionTeleportForkSharesPrefix pins the fork half of #2419: a fork mints a
// NEW trace pointing at the shared prefix, copies no entry, and two forks of one
// session are two sessions.
func TestSessionTeleportForkSharesPrefix(t *testing.T) {
	l, trace := seedHostA(t)
	head, nodes := l.Head(trace), l.NodeCount()

	first, err := ForkTeleport(l, trace, "")
	if err != nil {
		t.Fatalf("fork: %v", err)
	}
	if first.SharedPrefix != head {
		t.Fatalf("shared prefix = %s, want the source head %s", first.SharedPrefix, head)
	}
	if first.ForkTraceID == trace || first.ForkTraceID == "" {
		t.Fatalf("fork trace id = %q, want a fresh id", first.ForkTraceID)
	}
	if l.Head(first.ForkTraceID) != head {
		t.Fatalf("fork head = %s, want the shared prefix %s", l.Head(first.ForkTraceID), head)
	}
	if l.NodeCount() != nodes {
		t.Fatalf("fork copied entries: node count %d -> %d", nodes, l.NodeCount())
	}
	// The source is untouched — the original keeps its place.
	if l.Head(trace) != head {
		t.Fatalf("fork moved the source head to %s", l.Head(trace))
	}

	// Two forks of one session are two sessions.
	second, err := ForkTeleport(l, trace, "")
	if err != nil {
		t.Fatalf("second fork: %v", err)
	}
	if second.ForkTraceID == first.ForkTraceID {
		t.Fatalf("both forks minted the same trace %q", first.ForkTraceID)
	}
	if second.SharedPrefix != head {
		t.Fatalf("second fork shares %s, want %s", second.SharedPrefix, head)
	}

	// A fork is exportable in its own right, and its window matches the source's at
	// the shared prefix under the same arm — the continuation splice.
	arm := armFixture()
	srcWindow, err := TeleportWindow(l, trace, arm)
	if err != nil {
		t.Fatalf("source window: %v", err)
	}
	forkWindow, err := TeleportWindow(l, first.ForkTraceID, arm)
	if err != nil {
		t.Fatalf("fork window: %v", err)
	}
	// The windows differ only in the trace name: same head, same entries.
	if bytes.Equal(srcWindow, forkWindow) {
		t.Fatalf("the fork must be a DISTINCT session, not a byte-identical alias")
	}
	if !bytes.Equal(
		bytes.Replace(forkWindow, []byte(first.ForkTraceID), []byte(trace), 1),
		srcWindow,
	) {
		t.Fatalf("fork window diverges beyond the trace name:\n src:  %s\n fork: %s", srcWindow, forkWindow)
	}

	if _, err := ForkTeleport(l, "sess:absent", ""); err == nil {
		t.Fatalf("forking an unknown trace must be refused")
	}
	if _, err := ForkTeleport(l, trace, trace); err == nil {
		t.Fatalf("forking a trace onto itself must be refused")
	}
	if _, err := ForkTeleport(l, trace, first.ForkTraceID); err == nil {
		t.Fatalf("forking onto an occupied trace must be refused")
	}
}

// TestSessionTeleportExportRefusesUnrootedChain pins the export-side refusal: a
// chain whose oldest entries have aged out cannot be rooted, so it is refused at
// export rather than failing on the receiving host after the operator committed.
func TestSessionTeleportExportRefusesUnrootedChain(t *testing.T) {
	l := sessionledger.Memory()
	if _, err := ExportTeleport(l, "sess:never-seen", TeleportArm{}); err == nil {
		t.Fatalf("exporting an unknown trace must be refused")
	}
	if _, err := ExportTeleport(l, "", TeleportArm{}); err == nil {
		t.Fatalf("exporting an empty trace must be refused")
	}
	if _, err := ExportTeleport(nil, "sess:x", TeleportArm{}); err == nil {
		t.Fatalf("exporting without a ledger must be refused")
	}
}

// TestSessionTeleportExportRefusesNonCanonicalContent pins the JSON-hop trap: an
// entry whose content carries insignificant whitespace hashes one way in the ledger
// and another after an encoder re-flows it, so it is refused at export — on the
// source host, before the operator commits to the move — rather than failing on the
// receiving host.
func TestSessionTeleportExportRefusesNonCanonicalContent(t *testing.T) {
	l := sessionledger.Memory()
	const trace = "sess:spaced"
	if _, err := l.Append(trace, "turn_start", []byte(`{"messages": 3, "raw_bytes": 812}`)); err != nil {
		t.Fatalf("append: %v", err)
	}
	_, err := ExportTeleport(l, trace, TeleportArm{})
	if err == nil {
		t.Fatalf("exporting non-canonical content must be refused")
	}
	if !strings.Contains(err.Error(), "canonical compact form") {
		t.Fatalf("error = %v, want the canonical-form refusal", err)
	}

	// The same content, appended through an encoder, exports cleanly — so the check
	// is refusing the encoding and not the payload.
	compact, err := json.Marshal(map[string]int{"messages": 3, "raw_bytes": 812})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	l2 := sessionledger.Memory()
	if _, err := l2.Append(trace, "turn_start", compact); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := ExportTeleport(l2, trace, TeleportArm{}); err != nil {
		t.Fatalf("canonical content must export: %v", err)
	}
}

// TestSessionTeleportControlPlane pins the wire: the three verbs are reachable on
// the existing /v1/fak/session/{id}/{verb} control plane and answer the documents
// the CLI reads, including the refusal codes.
func TestSessionTeleportControlPlane(t *testing.T) {
	hostA, trace := seedHostA(t)
	hostB := sessionledger.Memory()

	// New() always installs a logger; the control verbs log an accepted move, so a
	// hand-built Server needs one too.
	srv := &Server{logf: func(string, ...any) {}}
	serve := func(l *sessionledger.Ledger, method, path string, body any) *httptest.ResponseRecorder {
		t.Helper()
		prev := teleportLedger
		teleportLedger = func() (*sessionledger.Ledger, error) { return l, nil }
		defer func() { teleportLedger = prev }()
		var rdr *bytes.Reader
		if body != nil {
			raw, err := json.Marshal(body)
			if err != nil {
				t.Fatalf("marshal body: %v", err)
			}
			rdr = bytes.NewReader(raw)
		} else {
			rdr = bytes.NewReader(nil)
		}
		req := httptest.NewRequest(method, path, rdr)
		rec := httptest.NewRecorder()
		srv.handleFakSession(rec, req)
		return rec
	}

	// export
	rec := serve(hostA, http.MethodPost, "/v1/fak/session/"+trace+"/export", armFixture())
	if rec.Code != http.StatusOK {
		t.Fatalf("export status = %d, body %s", rec.Code, rec.Body)
	}
	var bundle TeleportBundle
	if err := json.Unmarshal(rec.Body.Bytes(), &bundle); err != nil {
		t.Fatalf("decode exported bundle: %v", err)
	}
	if bundle.Head != hostA.Head(trace) || len(bundle.Entries) != 4 {
		t.Fatalf("exported bundle = head %s / %d entries", bundle.Head, len(bundle.Entries))
	}

	// import onto host B
	rec = serve(hostB, http.MethodPost, "/v1/fak/session/"+trace+"/import", bundle)
	if rec.Code != http.StatusOK {
		t.Fatalf("import status = %d, body %s", rec.Code, rec.Body)
	}
	if hostB.Head(trace) != bundle.Head {
		t.Fatalf("host B head = %s, want %s", hostB.Head(trace), bundle.Head)
	}

	// re-import is the resident-trace conflict, not a silent overwrite
	rec = serve(hostB, http.MethodPost, "/v1/fak/session/"+trace+"/import", bundle)
	if rec.Code != http.StatusConflict {
		t.Fatalf("re-import status = %d, want 409; body %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "teleport_trace_resident") {
		t.Fatalf("re-import body = %s, want the resident reason code", rec.Body)
	}

	// a tampered bundle is refused on the wire too
	bad := bundle
	bad.Arm.Budget.TokensLeft = 1 << 30
	rec = serve(sessionledger.Memory(), http.MethodPost, "/v1/fak/session/"+trace+"/import", bad)
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("tampered import status = %d, want 422; body %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "teleport_unverified") {
		t.Fatalf("tampered import body = %s, want the unverified reason code", rec.Body)
	}

	// fork
	rec = serve(hostA, http.MethodPost, "/v1/fak/session/"+trace+"/fork", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("fork status = %d, body %s", rec.Code, rec.Body)
	}
	var f TeleportFork
	if err := json.Unmarshal(rec.Body.Bytes(), &f); err != nil {
		t.Fatalf("decode fork: %v", err)
	}
	if f.SharedPrefix != hostA.Head(trace) || f.ForkTraceID == "" || f.TraceID != trace {
		t.Fatalf("fork response = %+v", f)
	}

	// forking an unknown trace is a 404, not a mint
	rec = serve(hostA, http.MethodPost, "/v1/fak/session/sess:absent/fork", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("absent fork status = %d, want 404; body %s", rec.Code, rec.Body)
	}

	// The generic control path is untouched: an unrelated verb still needs the
	// injected control implementation, so teleport did not swallow it.
	rec = serve(hostA, http.MethodPost, "/v1/fak/session/"+trace+"/run", map[string]any{"run": "paused"})
	if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), "session control is not configured") {
		t.Fatalf("the generic control path changed shape: %d %s", rec.Code, rec.Body)
	}
}
