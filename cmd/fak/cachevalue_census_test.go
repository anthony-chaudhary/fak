package main

// cachevalue_census_test.go — `fak cachevalue census` (#3650): the scrape half of the fleet
// managed-cache posture census. The FOLD is proved pure in internal/cachevaluereport; what is
// only provable here is what this file decides BEFORE the fold, which is where the census can
// lie:
//
//   - a live worker whose /debug/vars read fails must land in UNKNOWN, never in PASSIVE, so an
//     unreachable slice of the fleet can never manufacture a low adoption number;
//   - an ended or stale session must be dropped before the fold and never dialed at all.
//
// Both are asserted over the REAL cross-process read path (fetchSessionIndexStatus against a
// live httptest gateway, presenting each row's read-scoped bearer) rather than through the
// censusVarsFetch seam, so the transport's own failure modes — a refused status, a body that is
// not the gateway's JSON document — are the ones being classified.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// censusLivePID is the one pid the injected process-relations snapshot reports running, so
// liveness is decided by the fixture rather than by whatever this box happens to be doing.
const censusLivePID = 424242

// censusDeadPID is absent from that snapshot: a row carrying it is a crashed session (no
// tombstone, no process), which is the "stale" state.
const censusDeadPID = 424243

// censusFleetBodies maps each worker's gateway path to the /debug/vars answer it gives. Three
// shapes matter: a published managed_cache block, an answer with NO block (the producer's
// affirmative "lever off, nothing observed"), and a body that is not the gateway's document at
// all. A path absent from this map is served 503 — a live worker whose gateway refuses.
var censusFleetBodies = map[string]string{
	"/w1-active-fired": `{"fak_gateway":{"drive":"running"},"managed_cache":{"active":true,"inert":false,"upgraded":3}}`,
	"/w2-active-quiet": `{"fak_gateway":{"drive":"running"},"managed_cache":{"active":true,"inert":true,"upgraded":0}}`,
	"/w3-active-codex": `{"fak_gateway":{"drive":"running"},"managed_cache":{"active":true,"inert":false,"upgraded":0,"wire":"openai-responses"}}`,
	"/w4-passive":      `{"fak_gateway":{"drive":"running"}}`,
	"/w6-torn":         `<html><body>502 Bad Gateway</body></html>`,
}

// censusFleet stands up one gateway serving every worker in the fixture fleet and records the
// matching guard-session index under a fresh registry dir. It returns the registry dir and a
// reader for every /debug/vars path the census actually dialed — that second half is what
// proves a dropped session was never contacted, not merely excluded from the totals.
//
// The fleet is built to make each pre-fold decision separable:
//
//	w1 ACTIVE, upgrade fired                w5 live, gateway REFUSES the read (503)
//	w2 ACTIVE, nothing fired                w6 live, answers with a non-gateway body
//	w3 ACTIVE on the codex wire (no 1h-TTL lever to fire)
//	w4 answered, NO managed_cache block -> affirmative PASSIVE
//	w7 live but published no gateway at all
//	w8 ended (clean tombstone)              w9 stale (pid gone)
func censusFleet(t *testing.T) (regDir string, dialed func() []string) {
	t.Helper()
	var mu sync.Mutex
	var paths []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		worker := strings.TrimSuffix(r.URL.Path, "/debug/vars")
		// The census must present each row's OWN read-scoped bearer; a worker it
		// authenticated wrongly would read as dark and silently leave the ratios.
		if r.Header.Get("Authorization") != "Bearer read-"+strings.TrimPrefix(worker, "/") {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		body, ok := censusFleetBodies[worker]
		if !ok {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(ts.Close)

	withSessionQueryRelations(t, []procguard.Proc{{PID: censusLivePID, Name: "claude"}})
	regDir = t.TempDir()
	started := time.Now().Add(-30 * time.Minute)
	record := func(handle string, pid int, gwPath, endedAt string) {
		row := guardsessions.Row{
			Handle:    handle,
			TraceID:   "trace-" + handle,
			Agent:     "claude",
			PID:       pid,
			StartedAt: started.UTC().Format(time.RFC3339),
			EndedAt:   endedAt,
		}
		if gwPath != "" {
			row = row.WithGateway(ts.URL+gwPath, "read-"+handle)
		}
		recordSessionQueryRow(t, regDir, row)
	}
	record("w1-active-fired", censusLivePID, "/w1-active-fired", "")
	record("w2-active-quiet", censusLivePID, "/w2-active-quiet", "")
	record("w3-active-codex", censusLivePID, "/w3-active-codex", "")
	record("w4-passive", censusLivePID, "/w4-passive", "")
	record("w5-refused", censusLivePID, "/w5-refused", "")
	record("w6-torn", censusLivePID, "/w6-torn", "")
	record("w7-no-gateway", censusLivePID, "", "")
	record("w8-ended", censusLivePID, "/w8-ended", time.Now().UTC().Format(time.RFC3339))
	record("w9-stale", censusDeadPID, "/w9-stale", "")

	return regDir, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), paths...)
	}
}

// censusJSON runs the verb's JSON path over a registry dir and decodes the folded census.
func censusJSON(t *testing.T, regDir string) cachevaluereport.CensusReport {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if code := runCachevalueCensus(&stdout, &stderr, []string{"--reg-dir", regDir, "--json"}); code != 0 {
		t.Fatalf("census --json exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var rep cachevaluereport.CensusReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("census JSON did not decode: %v\n%s", err, stdout.String())
	}
	return rep
}

// censusStates maps each folded row's worker label to its census state.
func censusStates(rep cachevaluereport.CensusReport) map[string]string {
	states := make(map[string]string, len(rep.Rows))
	for _, r := range rep.Rows {
		states[r.Worker] = r.State()
	}
	return states
}

// The central honesty rule of the scrape: EVERY way a live worker's read can fail lands in
// UNKNOWN and leaves both ratios, while the one worker that genuinely answered without a
// managed_cache block counts as an affirmative PASSIVE.
//
// The stakes are exactly the verdict. Three of these seven workers are unreadable; folding
// them in as PASSIVE would put ACTIVE at 3/7 (43%) and flip the headline to MOSTLY_PASSIVE —
// sending an operator to re-check a launcher posture that is in fact fine, when the real
// finding is that three workers cannot be read at all.
func TestCachevalueCensusCountsAnUnreadableWorkerAsUnknownNotPassive(t *testing.T) {
	regDir, _ := censusFleet(t)
	rep := censusJSON(t, regDir)

	if rep.Schema != cachevaluereport.CensusSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, cachevaluereport.CensusSchema)
	}
	if rep.Workers != 7 || rep.Reached != 4 || rep.Unreached != 3 {
		t.Fatalf("coverage = workers %d / reached %d / unreached %d, want 7 / 4 / 3",
			rep.Workers, rep.Reached, rep.Unreached)
	}
	if rep.Active != 3 || rep.Passive != 1 {
		t.Fatalf("ACTIVE %d / PASSIVE %d, want 3 / 1 — only w4 answered without a managed_cache block",
			rep.Active, rep.Passive)
	}
	if rep.ActivePct == nil || *rep.ActivePct != 75 {
		t.Fatalf("ACTIVE share = %v, want 75%% of the REACHED workers", rep.ActivePct)
	}
	if rep.Verdict != cachevaluereport.VerdictAdopted {
		t.Fatalf("verdict = %q, want %q — three unreadable workers must not drag the headline down",
			rep.Verdict, cachevaluereport.VerdictAdopted)
	}

	// Each unreadable worker by name, so a regression that rescued one of them into PASSIVE
	// names which failure mode leaked.
	states := censusStates(rep)
	for worker, want := range map[string]string{
		"w1-active-fired": cachevaluereport.StateActive,
		"w2-active-quiet": cachevaluereport.StateActive,
		"w3-active-codex": cachevaluereport.StateActive,
		"w4-passive":      cachevaluereport.StatePassive,
		"w5-refused":      cachevaluereport.StateUnknown, // dialed; the gateway refused (503)
		"w6-torn":         cachevaluereport.StateUnknown, // dialed; answered a non-gateway body
		"w7-no-gateway":   cachevaluereport.StateUnknown, // nothing published to dial
	} {
		if got := states[worker]; got != want {
			t.Fatalf("worker %s = %q, want %q", worker, got, want)
		}
	}

	// The finding says the unreachable workers left the ratios, rather than leaving the reader
	// to infer it from a percentage whose denominator silently shrank.
	if !strings.Contains(rep.Finding, "3 unreachable worker(s) excluded from every ratio") {
		t.Fatalf("finding should disclose the excluded workers: %s", rep.Finding)
	}
}

// An ACTIVE worker on the codex wire has no 1h-TTL lever to fire, so it is excluded from the
// upgrade denominator instead of counted as a worker that failed to fire. Counting it would
// report 1/3 (33%) and read as a fleet-wide upgrade-wiring failure that is really one wire
// without the lever.
func TestCachevalueCensusExcludesALeverlessWireFromTheUpgradeRatio(t *testing.T) {
	regDir, _ := censusFleet(t)
	rep := censusJSON(t, regDir)

	if rep.ActiveWithLever != 2 || rep.ActiveLeverless != 1 {
		t.Fatalf("ACTIVE with lever %d / leverless %d, want 2 / 1 (w3 runs the codex wire)",
			rep.ActiveWithLever, rep.ActiveLeverless)
	}
	if rep.UpgradeFired != 1 || rep.UpgradeFiredPct == nil || *rep.UpgradeFiredPct != 50 {
		t.Fatalf("upgrade fired %d (%v), want 1 of the 2 ACTIVE-with-lever workers (50%%)",
			rep.UpgradeFired, rep.UpgradeFiredPct)
	}
	if !strings.Contains(rep.Finding, "1 ACTIVE worker(s) on a wire with no 1h-TTL lever excluded from the upgrade ratio") {
		t.Fatalf("finding should disclose the leverless exclusion: %s", rep.Finding)
	}
	// The wire that decided the exclusion rides on the row, so the breakdown can be audited
	// without re-scraping the fleet.
	for _, r := range rep.Rows {
		if r.Worker == "w3-active-codex" && r.Wire != "openai-responses" {
			t.Fatalf("w3 row wire = %q, want the resolved codex wire", r.Wire)
		}
	}
}

// Ended and stale sessions are dropped BEFORE the fold and — the part only this test can see —
// are never dialed at all. Their recorded ports may since have been reused by an unrelated
// process, which is why the drop has to happen on the index row rather than on the answer.
func TestCachevalueCensusDropsEndedAndStaleSessionsWithoutDialingThem(t *testing.T) {
	regDir, dialed := censusFleet(t)
	rep := censusJSON(t, regDir)

	if rep.Workers != 7 {
		t.Fatalf("workers = %d, want 7 — the ended and stale rows are not fleet workers", rep.Workers)
	}
	states := censusStates(rep)
	for _, gone := range []string{"w8-ended", "w9-stale"} {
		if st, ok := states[gone]; ok {
			t.Fatalf("%s folded into the census as %q; a recorded-but-dead session would dilute every ratio forever", gone, st)
		}
	}

	got := dialed()
	sort.Strings(got)
	want := []string{
		"/w1-active-fired/debug/vars", "/w2-active-quiet/debug/vars", "/w3-active-codex/debug/vars",
		"/w4-passive/debug/vars", "/w5-refused/debug/vars", "/w6-torn/debug/vars",
	}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("dialed %v, want exactly the six live workers that published a gateway %v", got, want)
	}
}

// censusRenderedState reads one worker's state out of the rendered breakdown, so the assertion
// survives a column-width change without restating the renderer's format string.
func censusRenderedState(t *testing.T, out, worker string) string {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == worker {
			return fields[1]
		}
	}
	t.Fatalf("rendered breakdown has no row for %s:\n%s", worker, out)
	return ""
}

// The rendered block is the operator-facing half: it must carry the coverage line, both
// headlines, and a per-worker breakdown addressable back to a `fak session status <handle>`.
func TestCachevalueCensusRendersCoverageBothHeadlinesAndEveryWorker(t *testing.T) {
	regDir, _ := censusFleet(t)

	var stdout, stderr bytes.Buffer
	if code := runCachevalueCensus(&stdout, &stderr, []string{"--reg-dir", regDir}); code != 0 {
		t.Fatalf("census exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"fleet managed-cache posture census — " + cachevaluereport.VerdictAdopted,
		"workers 7 (reached 4, unreachable 3)",
		"ACTIVE 3 · PASSIVE 1 · ACTIVE share 75%",
		"upgrade fired 1/2 ACTIVE-with-lever (50%)",
		"1 ACTIVE on a wire with no 1h-TTL lever",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered census missing %q:\n%s", want, out)
		}
	}
	for worker, want := range map[string]string{
		"w1-active-fired": cachevaluereport.StateActive,
		"w4-passive":      cachevaluereport.StatePassive,
		"w5-refused":      cachevaluereport.StateUnknown,
		"w7-no-gateway":   cachevaluereport.StateUnknown,
	} {
		if got := censusRenderedState(t, out, worker); got != want {
			t.Fatalf("rendered row for %s reads %q, want %q", worker, got, want)
		}
	}
	if strings.Contains(out, "w8-ended") || strings.Contains(out, "w9-stale") {
		t.Fatalf("rendered breakdown lists a dead session:\n%s", out)
	}
}

// A box with no live guard sessions is a valid answer, not a failure: exit 0, the honest
// INSUFFICIENT census rather than a fabricated 0% adoption, and a pointer at the index file
// that was actually consulted so the operator can tell "empty fleet" from "wrong --reg-dir".
func TestCachevalueCensusOnAnEmptyFleetIsInsufficientNotZeroAdoption(t *testing.T) {
	withSessionQueryRelations(t, []procguard.Proc{{PID: censusLivePID, Name: "claude"}})
	regDir := t.TempDir()

	var stdout, stderr bytes.Buffer
	if code := runCachevalueCensus(&stdout, &stderr, []string{"--reg-dir", regDir}); code != 0 {
		t.Fatalf("census on an empty fleet exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, cachevaluereport.VerdictInsufficient) || strings.Contains(out, "ACTIVE share 0%") {
		t.Fatalf("an empty fleet must read INSUFFICIENT, never 0%% adoption:\n%s", out)
	}
	if !strings.Contains(out, guardsessions.IndexPath(regDir)) {
		t.Fatalf("the empty-fleet hint should name the index it consulted:\n%s", out)
	}

	rep := censusJSON(t, regDir)
	if rep.Workers != 0 || rep.ActivePct != nil || rep.UpgradeFiredPct != nil {
		t.Fatalf("empty census = %+v, want no workers and NIL ratios (nothing to divide by is not a measured zero)", rep)
	}
	if !rep.OK {
		t.Fatalf("the census is a diagnostic loop, not a gate: OK must stay true, got %+v", rep)
	}
}

// A bad flag is a usage error the operator can fix (2), and it stops before any dial.
func TestCachevalueCensusRefusesAnUnknownFlagWithoutScraping(t *testing.T) {
	regDir, dialed := censusFleet(t)

	var stdout, stderr bytes.Buffer
	if code := runCachevalueCensus(&stdout, &stderr, []string{"--reg-dir", regDir, "--nope"}); code != 2 {
		t.Fatalf("exit = %d, want 2 for an unknown flag; stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if got := dialed(); len(got) != 0 {
		t.Fatalf("a refused invocation still dialed %d worker(s): %v", len(got), got)
	}
	if stdout.Len() != 0 {
		t.Fatalf("a refused invocation wrote a census to stdout: %s", stdout.String())
	}
}
