package main

// dispatch_worker_evidence.go — `fak dispatch evidence`, worker supervision's
// partial-evidence materialization for in-flight harness workers (#3037).
//
// When a harness worker WEDGES — an in-flight request older than five minutes with no
// flushed proof file — the operator needs durable, secret-scrubbed evidence to classify
// the failure AFTER the fact (provider latency, quota/rate wait, a stuck tool call, a
// transcript-writer failure, a guard refusal, a harness deadlock) WITHOUT reading
// volatile process memory. This verb folds the runs directory into one partial-evidence
// record per live worker and, with --materialize, writes each as a durable
// `<worker>.partial-evidence.json` sidecar that is safe to attach to an issue or a
// closure audit.
//
// The load-bearing wedge signal is the transcript's IN-FLIGHT AGE: a wedged worker's log
// stops advancing, so `now - transcript mtime` is how long the worker has been stuck
// mid-request. Everything else — last flushed turn, last tool, route-health, quota/rate
// state — is parsed best-effort from the transcript itself and reported only WHEN KNOWN:
// the transcript is the raw harness stdout/stderr, so a field stays empty/zero when the
// harness emits no recognizable marker. The transcript tail is always secret-scrubbed
// through the same canon matcher the rest of the kernel uses, so detection and redaction
// never drift; a tail carrying an obfuscated (non-raw-locatable) secret is SEALED rather
// than risk a leak.
//
//	fak dispatch evidence                 # human card over the live workers
//	fak dispatch evidence --json          # machine snapshot (fleet-dispatch-worker-evidence/1)
//	fak dispatch evidence --materialize   # + write the scrubbed proof sidecars
//
// It reuses the already-wired liveResolutionScopes scanner (resolve-*.log/.pid sidecars)
// so it sees exactly the workers `fak dispatch status` sees, and unlike a bare status
// fold it is the write path — the periodic materialization worker supervision runs to
// turn a wedged process into a diagnosable artifact.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

const dispatchWorkerEvidenceSnapshotSchema = "fleet-dispatch-worker-evidence/1"
const dispatchWorkerEvidenceArtifactSchema = "fak-worker-partial-evidence/1"
const dispatchWorkerEvidenceSidecarSuffix = ".partial-evidence.json"

// dispatchWorkerEvidenceTailBytes bounds the scrubbed transcript tail carried in the
// artifact: enough partial output to classify the wedge, small enough to attach to an
// issue without pasting a whole session log.
const dispatchWorkerEvidenceTailBytes = 4096

// workerPartialEvidence is the durable, secret-scrubbed proof a wedged worker leaves
// behind. The structural fields (paths, pid, in-flight age) are always present; the
// marker fields (last flushed turn, last tool, route-health, quota state) are populated
// only WHEN KNOWN — parsed from the transcript, empty/zero otherwise.
type workerPartialEvidence struct {
	Schema             string `json:"schema"`
	Issue              int    `json:"issue"`
	Worker             string `json:"worker"`
	Lane               string `json:"lane"`
	PID                int    `json:"pid"`
	Backend            string `json:"backend,omitempty"`
	TranscriptPath     string `json:"transcript_path"`
	GuardAuditPath     string `json:"guard_audit_path,omitempty"`
	InFlightAgeSeconds int    `json:"in_flight_age_seconds"`
	LastFlushedTurn    int    `json:"last_flushed_turn"`
	LastTool           string `json:"last_tool,omitempty"`
	RouteHealth        string `json:"route_health,omitempty"`
	QuotaState         string `json:"quota_state,omitempty"`
	TranscriptBytes    int64  `json:"transcript_bytes"`
	SecretsMasked      int    `json:"secrets_masked"`
	SecretScrubbed     bool   `json:"secret_scrubbed"`
	TranscriptTail     string `json:"transcript_tail,omitempty"`
	// ArtifactPath is set only in the snapshot after --materialize wrote the sidecar; it
	// is never embedded in the on-disk artifact itself (which would be self-referential).
	ArtifactPath string                           `json:"artifact_path,omitempty"`
	Delivery     *workdelivery.AdapterObservation `json:"delivery,omitempty"`
}

// dispatchWorkerEvidenceSnapshot is the fleet-dispatch-worker-evidence/1 payload.
type dispatchWorkerEvidenceSnapshot struct {
	Schema          string                  `json:"schema"`
	RunsDir         string                  `json:"runs_dir"`
	NowUnix         int64                   `json:"now_unix"`
	Materialized    bool                    `json:"materialized"`
	LiveWorkerCount int                     `json:"live_worker_count"`
	Workers         []workerPartialEvidence `json:"workers"`
}

// Best-effort transcript markers. fak does not inject these — the transcript is the raw
// harness stdout/stderr — so each matcher reads whatever the harness itself printed and
// the last match wins (the most recent flushed state). A field stays empty/zero when no
// marker is present, which is the honest "when known" degradation.
var (
	transcriptTurnRE        = regexp.MustCompile(`(?i)\bturn[ =#:]+(\d+)\b`)
	transcriptToolRE        = regexp.MustCompile(`(?i)(?:\btool[_ ]?(?:call|use|name)\b|"tool"|"tool_name"|\btool=)["'\s:=]*([A-Za-z0-9_.\-]{2,})`)
	transcriptRouteHealthRE = regexp.MustCompile(`(?i)\broute[_ -]?health[ =:]+([A-Za-z0-9_.\-]+)`)
	transcriptQuotaRE       = regexp.MustCompile(`(?i)\b(?:quota|rate[_ -]?limit)[ =:]+([A-Za-z0-9_.\-]+)`)
)

func runDispatchWorkerEvidence(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch evidence", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", dispatchProgressRunsDir, "directory of dispatch worker logs")
	asJSON := fs.Bool("json", false, "emit the fleet-dispatch-worker-evidence/1 JSON snapshot")
	materialize := fs.Bool("materialize", false, "write a scrubbed <worker>.partial-evidence.json proof sidecar per live worker")
	nowUnix := fs.Int64("now", 0, "override wall-clock (unix seconds) for hermetic in-flight-age tests")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch evidence: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	now := time.Now()
	if *nowUnix > 0 {
		now = time.Unix(*nowUnix, 0)
	}

	snap := dispatchWorkerEvidenceScan(*runsDir, now)
	if *materialize {
		for i := range snap.Workers {
			path, err := materializeWorkerEvidence(snap.Workers[i])
			if err != nil {
				fmt.Fprintf(stderr, "fak dispatch evidence: write proof for #%d: %v\n", snap.Workers[i].Issue, err)
				return 1
			}
			snap.Workers[i].ArtifactPath = path
		}
		snap.Materialized = true
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, snap, "fak dispatch evidence")
	}
	fmt.Fprint(stdout, renderWorkerEvidence(snap))
	return 0
}

// dispatchWorkerEvidenceScan folds the runs directory into one partial-evidence record
// per live worker. It is a pure read over the filesystem given `now`, so a test drives it
// hermetically by planting sidecars and pinning `now` via --now.
func dispatchWorkerEvidenceScan(runsDir string, now time.Time) dispatchWorkerEvidenceSnapshot {
	scopes := liveResolutionScopes(runsDir)
	workers := make([]workerPartialEvidence, 0, len(scopes))
	for _, s := range scopes {
		workers = append(workers, collectWorkerPartialEvidence(s, now))
	}
	return dispatchWorkerEvidenceSnapshot{
		Schema:          dispatchWorkerEvidenceSnapshotSchema,
		RunsDir:         runsDir,
		NowUnix:         now.Unix(),
		LiveWorkerCount: len(workers),
		Workers:         workers,
	}
}

// collectWorkerPartialEvidence reads one live worker's transcript + sidecars into a
// durable, secret-scrubbed evidence record. It never fabricates a field: the in-flight
// age is the mtime gap (the load-bearing wedge signal), and every marker is empty/zero
// unless the transcript actually carries it.
func collectWorkerPartialEvidence(scope dispatchLiveScope, now time.Time) workerPartialEvidence {
	ev := workerPartialEvidence{
		Schema:         dispatchWorkerEvidenceArtifactSchema,
		Issue:          scope.Issue,
		Worker:         scope.Worker,
		Lane:           scope.Lane,
		PID:            scope.PID,
		TranscriptPath: scope.Log,
	}
	stem := strings.TrimSuffix(scope.Log, filepath.Ext(scope.Log))
	if b, err := os.ReadFile(stem + ".backend"); err == nil {
		ev.Backend = strings.TrimSpace(string(b))
	}
	ev.GuardAuditPath = firstExistingPath(stem+".guard-audit.json", stem+".guard-audit.jsonl", stem+".guard-audit")

	raw, err := os.ReadFile(scope.Log)
	if err != nil {
		return ev
	}
	ev.TranscriptBytes = int64(len(raw))

	// In-flight age: how long since the transcript last advanced. A wedged worker stops
	// writing, so this is the durable stand-in for "in-flight request older than N".
	if st, statErr := os.Stat(scope.Log); statErr == nil {
		age := int(now.Sub(st.ModTime()).Seconds())
		if age < 0 {
			age = 0
		}
		ev.InFlightAgeSeconds = age
	}

	ev.LastFlushedTurn = lastTranscriptTurn(raw)
	ev.LastTool = lastTranscriptMarker(raw, transcriptToolRE)
	ev.RouteHealth = lastTranscriptMarker(raw, transcriptRouteHealthRE)
	ev.QuotaState = lastTranscriptMarker(raw, transcriptQuotaRE)
	if ev.InFlightAgeSeconds > 0 {
		unitID := fmt.Sprintf("issue-%d", ev.Issue)
		next := "inspect the worker transcript and split the failing check"
		if delivery, deliveryErr := workdelivery.BlockedObservation(workdelivery.AdapterFleet, unitID, "runtime-observation", "capacity-seat", []workdelivery.Evidence{{Kind: "transcript", Reference: ev.TranscriptPath, Witnessed: true}}, next); deliveryErr == nil {
			ev.Delivery = &delivery
		}
	}

	// Scrub a line-aligned tail. Start at a clean line boundary so the cut never splits a
	// credential mid-token, then mask every raw secret span. If the tail carries an
	// obfuscated secret that has no raw span (RawSecretComplete=false), SEAL the tail
	// rather than emit bytes a raw redactor can't reach — the artifact must be safe to
	// attach to a public issue.
	tail := lineAlignedTranscriptTail(raw)
	if canon.RawSecretComplete(tail) {
		scrubbed, masked := canon.RedactSecrets(tail)
		ev.SecretsMasked = masked
		ev.SecretScrubbed = true
		ev.TranscriptTail = string(scrubbed)
	} else {
		ev.SecretScrubbed = false
		ev.TranscriptTail = ""
	}
	return ev
}

// lineAlignedTranscriptTail returns at most dispatchWorkerEvidenceTailBytes of raw's tail,
// advanced past the first newline inside the window so the cut lands on a LINE boundary and
// can never split a credential mid-token. Input shorter than the window is returned whole.
// Shared by the evidence sidecar and `fak dispatch sessions --tail`, which must agree: both
// hand the result to canon.RawSecretComplete, and a mid-token cut is exactly what would make
// a raw secret span unrecognizable to the redactor.
func lineAlignedTranscriptTail(raw []byte) []byte {
	tail := raw
	if len(tail) > dispatchWorkerEvidenceTailBytes {
		tail = tail[len(tail)-dispatchWorkerEvidenceTailBytes:]
		if i := bytes.IndexByte(tail, '\n'); i >= 0 && i+1 <= len(tail) {
			tail = tail[i+1:]
		}
	}
	return tail
}

// materializeWorkerEvidence writes the scrubbed evidence record next to its transcript as
// `<worker>.partial-evidence.json` and returns the path. The written artifact never
// embeds its own ArtifactPath — that is a snapshot-only field.
func materializeWorkerEvidence(ev workerPartialEvidence) (string, error) {
	ev.ArtifactPath = ""
	stem := strings.TrimSuffix(ev.TranscriptPath, filepath.Ext(ev.TranscriptPath))
	path := stem + dispatchWorkerEvidenceSidecarSuffix
	b, err := json.MarshalIndent(ev, "", "  ")
	if err != nil {
		return "", err
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func lastTranscriptTurn(raw []byte) int {
	m := transcriptTurnRE.FindAllSubmatch(raw, -1)
	if len(m) == 0 {
		return 0
	}
	n, err := strconv.Atoi(string(m[len(m)-1][1]))
	if err != nil {
		return 0
	}
	return n
}

func lastTranscriptMarker(raw []byte, re *regexp.Regexp) string {
	m := re.FindAllSubmatch(raw, -1)
	if len(m) == 0 {
		return ""
	}
	return string(m[len(m)-1][1])
}

func firstExistingPath(paths ...string) string {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func renderWorkerEvidence(snap dispatchWorkerEvidenceSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch worker evidence — %d live worker(s)\n", snap.LiveWorkerCount)
	fmt.Fprintf(&b, "runs-dir: %s\n", snap.RunsDir)
	if snap.Materialized {
		fmt.Fprintln(&b, "materialized: wrote scrubbed partial-evidence sidecars")
	}
	if len(snap.Workers) == 0 {
		fmt.Fprint(&b, "no live issue-resolution workers\n")
		return b.String()
	}
	for _, w := range snap.Workers {
		lane := w.Lane
		if lane == "" {
			lane = "(no lane)"
		}
		fmt.Fprintf(&b, "  #%d  lane=%s  pid=%d  in-flight=%ds  turn=%d  tool=%s\n",
			w.Issue, lane, w.PID, w.InFlightAgeSeconds, w.LastFlushedTurn, firstString(w.LastTool, "(unknown)"))
		fmt.Fprintf(&b, "        transcript=%s\n", w.TranscriptPath)
		if w.GuardAuditPath != "" {
			fmt.Fprintf(&b, "        guard-audit=%s\n", w.GuardAuditPath)
		}
		if w.RouteHealth != "" {
			fmt.Fprintf(&b, "        route-health=%s\n", w.RouteHealth)
		}
		if w.QuotaState != "" {
			fmt.Fprintf(&b, "        quota=%s\n", w.QuotaState)
		}
		if w.Delivery != nil {
			fmt.Fprintf(&b, "        blocked-unit=%s stage=%s bottleneck=%s next=%s\n", w.Delivery.UnitID, w.Delivery.Stage, w.Delivery.Bottleneck, w.Delivery.NextAction)
		}
		if !w.SecretScrubbed {
			fmt.Fprint(&b, "        (transcript tail sealed — obfuscated secret not raw-maskable)\n")
		}
		if w.ArtifactPath != "" {
			fmt.Fprintf(&b, "        proof=%s\n", w.ArtifactPath)
		}
	}
	return b.String()
}
