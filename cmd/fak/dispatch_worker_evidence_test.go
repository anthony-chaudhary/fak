package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/canon"
)

// plantWedgedResolveWorker writes the sidecars of a worker that served partial output
// and then WEDGED: its transcript carries real streamed work (turns, a tool call,
// route-health/quota markers, and a leaked secret), but its log mtime is pinned old so
// the in-flight age reads as a stalled, over-five-minute request. The pid is THIS
// process so the liveness gate passes hermetically.
func plantWedgedResolveWorker(t *testing.T, runsDir string, issue int, lane string, mtime time.Time) string {
	t.Helper()
	stem := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-20000101-000000", issue))
	var body strings.Builder
	fmt.Fprintf(&body, "# fak-spawn 20000101-000000 issue=%d lane=%s backend=opencode argv0=fak\n", issue, lane)
	body.WriteString("=== Turn 1 ===\nrouting request\n")
	body.WriteString("=== Turn 2 ===\ntool_call: read_file\n")
	body.WriteString("route_health: degraded\n")
	body.WriteString("=== Turn 3 ===\ntool_use web_search\n")
	body.WriteString("rate_limit: waiting\n")
	// A real leaked credential in the partial output — the artifact MUST scrub it.
	body.WriteString("Authorization: Bearer xoxb-1234567890-abcdefghijklmnop\n")
	body.WriteString(strings.Repeat("...streaming partial output...\n", 30))
	if err := os.WriteFile(stem+".log", []byte(body.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".pid", []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".backend", []byte("opencode"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(stem+".log", mtime, mtime); err != nil {
		t.Fatal(err)
	}
	return stem
}

func TestDispatchWorkerEvidenceWedgedWorkerLeavesScrubbedProof(t *testing.T) {
	runsDir := t.TempDir()
	// now = 2000-01-01 00:10:00; log mtime = 00:04:00 → 360s in-flight (> 5 min wedge).
	now := time.Date(2000, 1, 1, 0, 10, 0, 0, time.UTC)
	mtime := time.Date(2000, 1, 1, 0, 4, 0, 0, time.UTC)
	stem := plantWedgedResolveWorker(t, runsDir, 3037, "tools", mtime)

	out, errb, code := runDispatchAt("evidence", "--runs-dir", runsDir,
		"--materialize", "--now", fmt.Sprint(now.Unix()), "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var snap dispatchWorkerEvidenceSnapshot
	if err := json.Unmarshal([]byte(out), &snap); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out)
	}
	if snap.Schema != dispatchWorkerEvidenceSnapshotSchema {
		t.Fatalf("schema = %q, want %q", snap.Schema, dispatchWorkerEvidenceSnapshotSchema)
	}
	if !snap.Materialized || snap.LiveWorkerCount != 1 {
		t.Fatalf("want 1 materialized worker, got materialized=%v count=%d", snap.Materialized, snap.LiveWorkerCount)
	}
	w := snap.Workers[0]

	// The load-bearing wedge signal: in-flight age from the mtime gap.
	if w.InFlightAgeSeconds < 300 {
		t.Fatalf("in_flight_age_seconds = %d, want >= 300 (a wedge)", w.InFlightAgeSeconds)
	}
	// "When known" fields parsed from the partial transcript.
	if w.LastFlushedTurn != 3 {
		t.Fatalf("last_flushed_turn = %d, want 3", w.LastFlushedTurn)
	}
	if w.LastTool != "web_search" {
		t.Fatalf("last_tool = %q, want web_search", w.LastTool)
	}
	if w.RouteHealth != "degraded" {
		t.Fatalf("route_health = %q, want degraded", w.RouteHealth)
	}
	if w.QuotaState != "waiting" {
		t.Fatalf("quota_state = %q, want waiting", w.QuotaState)
	}
	if w.Backend != "opencode" {
		t.Fatalf("backend = %q, want opencode", w.Backend)
	}
	if w.TranscriptPath != stem+".log" {
		t.Fatalf("transcript_path = %q, want %q", w.TranscriptPath, stem+".log")
	}

	// The artifact must be secret-scrubbed and safe to attach.
	if !w.SecretScrubbed || w.SecretsMasked < 1 {
		t.Fatalf("want scrubbed with >=1 masked, got scrubbed=%v masked=%d", w.SecretScrubbed, w.SecretsMasked)
	}
	if strings.Contains(w.TranscriptTail, "xoxb-1234567890-abcdefghijklmnop") {
		t.Fatalf("transcript_tail leaked the secret:\n%s", w.TranscriptTail)
	}
	if !strings.Contains(w.TranscriptTail, "[redacted:secret") {
		t.Fatalf("transcript_tail missing the redaction mark:\n%s", w.TranscriptTail)
	}

	// The durable proof sidecar must exist, be readable JSON, and stay scrubbed on disk —
	// enough to file a route-health / tool-health / harness-writer issue without touching
	// the live process.
	if w.ArtifactPath != stem+dispatchWorkerEvidenceSidecarSuffix {
		t.Fatalf("artifact_path = %q, want %q", w.ArtifactPath, stem+dispatchWorkerEvidenceSidecarSuffix)
	}
	diskBytes, err := os.ReadFile(w.ArtifactPath)
	if err != nil {
		t.Fatalf("read proof sidecar: %v", err)
	}
	if strings.Contains(string(diskBytes), "xoxb-1234567890-abcdefghijklmnop") {
		t.Fatalf("proof sidecar leaked the secret on disk:\n%s", diskBytes)
	}
	var disk workerPartialEvidence
	if err := json.Unmarshal(diskBytes, &disk); err != nil {
		t.Fatalf("proof sidecar is not readable JSON: %v", err)
	}
	if disk.Schema != dispatchWorkerEvidenceArtifactSchema {
		t.Fatalf("artifact schema = %q, want %q", disk.Schema, dispatchWorkerEvidenceArtifactSchema)
	}
	if disk.Issue != 3037 || disk.LastFlushedTurn != 3 || disk.LastTool != "web_search" {
		t.Fatalf("artifact lost evidence: %+v", disk)
	}
	// The on-disk artifact is never self-referential.
	if disk.ArtifactPath != "" {
		t.Fatalf("on-disk artifact should not embed its own path, got %q", disk.ArtifactPath)
	}
}

func TestDispatchWorkerEvidenceSealsObfuscatedSecret(t *testing.T) {
	runsDir := t.TempDir()
	stem := filepath.Join(runsDir, "resolve-4242-20000101-000000")
	// A base64-obfuscated secret has no raw span RedactSecrets can mask, so the tail must
	// be SEALED (emitted empty) rather than risk a leak.
	obf := "c2stYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXowMTIzNDU2Nzg5" // base64("sk-…")-shaped
	body := "# fak-spawn 20000101-000000 issue=4242 lane=tools backend=opencode argv0=fak\n" +
		strings.Repeat("streamed line\n", 40) + "leak=" + obf + "\n"
	if err := os.WriteFile(stem+".log", []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".pid", []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatal(err)
	}

	// The fixture is only an obfuscated-secret case if canon flags it via de-obfuscation
	// but cannot locate a raw span; if this canon build treats it as raw-clean, the seal
	// branch is not exercised and there is nothing to prove here.
	if canon.RawSecretComplete([]byte(body)) {
		t.Skip("fixture secret is raw-locatable (or unflagged) on this canon build; seal branch not exercised")
	}

	scope := dispatchLiveScope{Issue: 4242, Lane: "tools", Log: stem + ".log", PID: os.Getpid(), Worker: filepath.Base(stem)}
	ev := collectWorkerPartialEvidence(scope, time.Now())
	if ev.SecretScrubbed || ev.TranscriptTail != "" {
		t.Fatalf("obfuscated secret must seal the tail: scrubbed=%v tail=%q", ev.SecretScrubbed, ev.TranscriptTail)
	}
}

func TestDispatchWorkerEvidenceCarriesExactCanonicalBlocker(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "issue-7105.log")
	if err := os.WriteFile(logPath, []byte("turn=3 tool=go_test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	old := now.Add(-2 * time.Minute)
	if err := os.Chtimes(logPath, old, old); err != nil {
		t.Fatal(err)
	}
	ev := collectWorkerPartialEvidence(dispatchLiveScope{Issue: 7105, Log: logPath}, now)
	if ev.Delivery == nil {
		t.Fatal("missing delivery blocker")
	}
	if ev.Delivery.UnitID != "issue-7105" || ev.Delivery.Stage != "runtime-observation" || ev.Delivery.Bottleneck != "unknown-irreducible" {
		t.Fatalf("delivery=%+v", ev.Delivery)
	}
	rendered := renderWorkerEvidence(dispatchWorkerEvidenceSnapshot{Workers: []workerPartialEvidence{ev}, LiveWorkerCount: 1})
	if !strings.Contains(rendered, "blocked-unit=issue-7105 stage=runtime-observation bottleneck=unknown-irreducible") {
		t.Fatalf("render=%q", rendered)
	}
}

func TestDispatchWorkerEvidenceDoesNotBlockFreshActiveWorker(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "issue-7105.log")
	if err := os.WriteFile(logPath, []byte("turn=3 tool=go_test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	recent := now.Add(-time.Second)
	if err := os.Chtimes(logPath, recent, recent); err != nil {
		t.Fatal(err)
	}
	ev := collectWorkerPartialEvidence(dispatchLiveScope{Issue: 7105, Log: logPath}, now)
	if ev.Delivery != nil {
		t.Fatalf("fresh worker falsely blocked: %+v", ev.Delivery)
	}
}
