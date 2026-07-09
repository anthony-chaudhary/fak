package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestDoomloopTickWiresCorrection drives the full shell path: repeated
// burning-flat ticks must trip DOOM_LOOP, write an accountability-ledger row,
// and (with --correct) queue a reversible re-anchor nudge to the outbox.
func TestDoomloopTickWiresCorrection(t *testing.T) {
	store := t.TempDir()
	var lastOut bytes.Buffer

	// Five observations: effort climbs every tick, progress pinned flat.
	efforts := []string{"10", "20", "30", "40", "50"}
	for i, e := range efforts {
		lastOut.Reset()
		var errb bytes.Buffer
		argv := []string{"tick",
			"--session", "w1",
			"--effort", e,
			"--progress", "7",
			"--alive",
			"--correct",
			"--store", store,
			"--now", dlItoa(int64(1000 + i)),
		}
		if rc := runDoomloop(&lastOut, &errb, strings.NewReader(""), argv); rc != 0 {
			t.Fatalf("tick %d rc=%d stderr=%s", i, rc, errb.String())
		}
	}

	if !strings.Contains(lastOut.String(), string("DOOM_LOOP")) {
		t.Fatalf("final tick did not report DOOM_LOOP:\n%s", lastOut.String())
	}

	// Accountability ledger exists and its last row is a nudge-queued doom loop.
	ledger := filepath.Join(store, "decisions.jsonl")
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	lines := dlNonEmptyLines(string(raw))
	if len(lines) != len(efforts) {
		t.Fatalf("ledger has %d rows, want %d", len(lines), len(efforts))
	}
	var last dlDecision
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatalf("decode last decision: %v", err)
	}
	if last.Verdict != "DOOM_LOOP" {
		t.Fatalf("last verdict = %q, want DOOM_LOOP", last.Verdict)
	}
	if last.Applied != "nudge-queued" {
		t.Fatalf("last applied = %q, want nudge-queued", last.Applied)
	}

	// The reversible correction artifact landed in the outbox.
	outbox := filepath.Join(store, "outbox")
	des, err := os.ReadDir(outbox)
	if err != nil || len(des) == 0 {
		t.Fatalf("expected a queued nudge packet in %s (err=%v, n=%d)", outbox, err, len(des))
	}
	pktRaw, err := os.ReadFile(filepath.Join(outbox, des[0].Name()))
	if err != nil {
		t.Fatalf("read nudge: %v", err)
	}
	var pkt nudgePacket
	if err := json.Unmarshal(pktRaw, &pkt); err != nil {
		t.Fatalf("decode nudge: %v", err)
	}
	if !pkt.Reversible || pkt.Kind != "reanchor" || pkt.Session != "w1" {
		t.Fatalf("nudge packet malformed: %+v", pkt)
	}
}

// TestDoomloopScanFoldsStore verifies scan classifies each stored worker.
func TestDoomloopScanFoldsStore(t *testing.T) {
	store := t.TempDir()
	// Healthy worker: effort and progress both climb.
	for i := 0; i < 4; i++ {
		var o, e bytes.Buffer
		argv := []string{"tick", "--session", "healthy",
			"--effort", dlItoa(int64(10 * (i + 1))), "--progress", dlItoa(int64(i + 1)),
			"--alive", "--store", store, "--now", dlItoa(int64(2000 + i))}
		if rc := runDoomloop(&o, &e, strings.NewReader(""), argv); rc != 0 {
			t.Fatalf("healthy tick rc=%d %s", rc, e.String())
		}
	}
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"scan", "--store", store, "--json"}); rc != 0 {
		t.Fatalf("scan rc=%d %s", rc, e.String())
	}
	if !strings.Contains(o.String(), "healthy") || !strings.Contains(o.String(), "HEALTHY") {
		t.Fatalf("scan output missing healthy worker verdict:\n%s", o.String())
	}
}

// TestDoomloopClassifyStdin verifies the one-shot path over a samples stream.
func TestDoomloopClassifyStdin(t *testing.T) {
	in := strings.Join([]string{
		`{"unix_millis":0,"effort":5,"progress":2,"alive":true}`,
		`{"unix_millis":1,"effort":15,"progress":2,"alive":true}`,
		`{"unix_millis":2,"effort":25,"progress":2,"alive":true}`,
		`{"unix_millis":3,"effort":35,"progress":2,"alive":true}`,
	}, "\n")
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(in), []string{"classify"}); rc != 0 {
		t.Fatalf("classify rc=%d %s", rc, e.String())
	}
	if !strings.Contains(o.String(), "DOOM_LOOP") {
		t.Fatalf("classify did not detect doom loop:\n%s", o.String())
	}
}

func TestDoomloopTickRequiresSession(t *testing.T) {
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"tick", "--effort", "1"}); rc != 2 {
		t.Fatalf("rc = %d, want 2 for missing --session", rc)
	}
}

func dlNonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

func dlItoa(n int64) string { return strconv.FormatInt(n, 10) }

// TestDoomloopDrainReportsButKeeps proves the OBSERVE-only output drainer: it
// enumerates a queued nudge with its steer destination, exits 3 (actionable, not
// enacted), and removes nothing.
func TestDoomloopDrainReportsButKeeps(t *testing.T) {
	store := t.TempDir()
	// Queue a real nudge: repeated burning-flat ticks with --correct.
	for i := 0; i < 5; i++ {
		var o, e bytes.Buffer
		argv := []string{"tick", "--session", "w1",
			"--effort", dlItoa(int64(10 * (i + 1))), "--progress", "7",
			"--alive", "--correct", "--store", store, "--now", dlItoa(int64(4000 + i))}
		if rc := runDoomloop(&o, &e, strings.NewReader(""), argv); rc != 0 {
			t.Fatalf("tick %d rc=%d %s", i, rc, e.String())
		}
	}
	outbox := filepath.Join(store, "outbox")
	before, err := os.ReadDir(outbox)
	if err != nil || len(before) == 0 {
		t.Fatalf("precondition: expected a queued nudge (err=%v n=%d)", err, len(before))
	}

	// JSON mode surfaces the queued packet and its steer destination.
	var jo, je bytes.Buffer
	if rc := runDoomloop(&jo, &je, strings.NewReader(""), []string{"drain", "--store", store, "--json"}); rc != 0 {
		t.Fatalf("drain --json rc=%d %s", rc, je.String())
	}
	var rep drainReport
	if err := json.Unmarshal(jo.Bytes(), &rep); err != nil {
		t.Fatalf("decode drain report: %v\n%s", err, jo.String())
	}
	// A sustained doom loop queues one nudge per tripped tick, so drain must
	// report every packet actually in the outbox - not a hard-coded count.
	if rep.Pending != len(before) || len(rep.Nudges) != len(before) {
		t.Fatalf("drain pending=%d nudges=%d, want %d (all queued packets)", rep.Pending, len(rep.Nudges), len(before))
	}
	if rep.Pending < 1 {
		t.Fatalf("precondition: expected at least one queued nudge to drain")
	}
	for _, n := range rep.Nudges {
		if n.Session != "w1" || !strings.Contains(n.Destination, "/steer") {
			t.Fatalf("drain row malformed: %+v", n)
		}
	}

	// Text mode exits 3 (pending, not enacted) and says it delivered nothing.
	var to, te bytes.Buffer
	rc := runDoomloop(&to, &te, strings.NewReader(""), []string{"drain", "--store", store})
	if rc != 3 {
		t.Fatalf("drain text rc = %d, want 3 (actionable, not enacted); out=%s", rc, to.String())
	}
	if !strings.Contains(to.String(), "observe-only") || !strings.Contains(to.String(), "w1") {
		t.Fatalf("drain text output missing observe-only / session:\n%s", to.String())
	}

	// OBSERVE-only: the packet is still queued (nothing removed).
	after, err := os.ReadDir(outbox)
	if err != nil || len(after) != len(before) {
		t.Fatalf("observe-only violated: outbox changed from %d to %d packet(s) (err=%v)", len(before), len(after), err)
	}
}

// dlWriteNudge writes one nudge packet directly into <store>/outbox/<file> so a delivery
// test can control the exact queue without driving a full tick sequence.
func dlWriteNudge(t *testing.T, outbox, file string, pkt nudgePacket) {
	t.Helper()
	if err := os.MkdirAll(outbox, 0o755); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(pkt)
	if err := os.WriteFile(filepath.Join(outbox, file), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestDoomloopDrainDeliverPostsAndArchives: --deliver POSTs a queued nudge onto the steer
// bus as the doomloop-guard machine principal; a 202 archives the packet out of the outbox
// (idempotent re-drain) and reports it delivered with exit 0.
func TestDoomloopDrainDeliverPostsAndArchives(t *testing.T) {
	store := t.TempDir()
	outbox := filepath.Join(store, "outbox")
	dlWriteNudge(t, outbox, "w-live-1.json", nudgePacket{
		Session: "w-live", Kind: "reanchor", Reason: "DOOM_LOOP", Message: "step back and re-read the goal", Reversible: true, Streak: 5,
	})

	var gotPrincipal, gotText, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var sr struct {
			Text      string `json:"text"`
			Principal string `json:"principal"`
		}
		_ = json.NewDecoder(r.Body).Decode(&sr)
		gotPrincipal, gotText, gotPath = sr.Principal, sr.Text, r.URL.Path
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("{}"))
	}))
	defer srv.Close()

	var o, e bytes.Buffer
	rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"drain", "--deliver", "--store", store, "--addr", srv.URL})
	if rc != 0 {
		t.Fatalf("drain --deliver rc = %d, want 0 (all delivered); stderr=%s out=%s", rc, e.String(), o.String())
	}
	if gotPrincipal != doomloopSteerPrincipal {
		t.Fatalf("delivered steer principal = %q, want %q (machine attribution, not operator)", gotPrincipal, doomloopSteerPrincipal)
	}
	if gotText != "step back and re-read the goal" {
		t.Fatalf("delivered steer text = %q, want the nudge message", gotText)
	}
	if !strings.Contains(gotPath, "/session/w-live/steer") {
		t.Fatalf("delivered to path %q, want the session's steer endpoint", gotPath)
	}
	// Archived: gone from the outbox, present under delivered/ (idempotent re-drain).
	if des, err := os.ReadDir(outbox); err != nil || len(des) != 0 {
		t.Fatalf("delivered packet not removed from outbox: n=%d err=%v", len(des), err)
	}
	if des, err := os.ReadDir(filepath.Join(store, "delivered")); err != nil || len(des) != 1 {
		t.Fatalf("delivered packet not archived: n=%d err=%v", len(des), err)
	}
}

// TestDoomloopDrainDeliverRefusedKeepsPacket: a 409 STEER_NO_OWNED_LOOP (the honest #3528
// refusal — target owns no loop) leaves the packet in the outbox, reports it as refused (not
// a silent success), and exits 3 so a scheduler sees the residual.
func TestDoomloopDrainDeliverRefusedKeepsPacket(t *testing.T) {
	store := t.TempDir()
	outbox := filepath.Join(store, "outbox")
	dlWriteNudge(t, outbox, "w-proxy-1.json", nudgePacket{
		Session: "w-proxy", Kind: "reanchor", Reason: "DOOM_LOOP", Message: "re-anchor", Reversible: true, Streak: 4,
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"error":{"code":"steer_no_owned_loop","message":"STEER_NO_OWNED_LOOP: this serve owns no loop; start it with --native"}}`))
	}))
	defer srv.Close()

	var o, e bytes.Buffer
	rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"drain", "--deliver", "--store", store, "--addr", srv.URL, "--json"})
	if rc != 3 {
		t.Fatalf("drain --deliver rc = %d, want 3 (residual undelivered); out=%s", rc, o.String())
	}
	var rep drainReport
	if err := json.Unmarshal(o.Bytes(), &rep); err != nil {
		t.Fatalf("decode deliver report: %v\n%s", err, o.String())
	}
	if rep.Delivered != 0 || rep.Refused != 1 || rep.Pending != 1 || len(rep.Nudges) != 1 {
		t.Fatalf("deliver report = %+v, want delivered=0 refused=1 pending=1 nudges=1", rep)
	}
	if rep.Nudges[0].Delivered || rep.Nudges[0].Outcome != "refused-no-owned-loop" {
		t.Fatalf("refused nudge row = %+v, want delivered=false outcome=refused-no-owned-loop", rep.Nudges[0])
	}
	// Fail-closed: the packet is STILL in the outbox for a later drain.
	if des, err := os.ReadDir(outbox); err != nil || len(des) != 1 {
		t.Fatalf("refused packet must stay queued: n=%d err=%v", len(des), err)
	}
	if des, err := os.ReadDir(filepath.Join(store, "delivered")); err == nil && len(des) != 0 {
		t.Fatalf("refused packet must NOT be archived: found %d in delivered/", len(des))
	}
}

// TestDoomloopDrainEmptyOutbox: a store with no queued nudges drains clean.
func TestDoomloopDrainEmptyOutbox(t *testing.T) {
	store := t.TempDir()
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"drain", "--store", store}); rc != 0 {
		t.Fatalf("drain empty rc = %d, want 0; stderr=%s", rc, e.String())
	}
	if !strings.Contains(o.String(), "empty") {
		t.Fatalf("drain empty output = %q, want mention of an empty outbox", o.String())
	}
}

// dlWriteLines (over)writes the file at path with n non-empty JSONL-ish lines -
// a transcript stand-in whose line count is the effort axis.
func dlWriteLines(t *testing.T, path string, n int) {
	t.Helper()
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteString(`{"turn":`)
		b.WriteString(dlItoa(int64(i)))
		b.WriteString("}\n")
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
}

func dlNonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

func dlItoa(n int64) string { return strconv.FormatInt(n, 10) }
