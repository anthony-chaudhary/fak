// cron_fire.go adds the FIRE WITNESS to `fak cron` (#2886, the durable-job side of
// the long-run epic #2382). `fak cron emit` (cron.go, #765) projects the schedule
// DOWN to an OS scheduler unit; the OS owns wall-clock firing but nothing proved a
// fire landed exactly once. This closes that gap: `fak cron fire` records each fire
// in an append-only JSONL ledger under a compare-and-set keyed by (job, slot), so a
// duplicate or overlapping tick for a slot that already fired is recorded as a
// structured `deduped` event instead of double-delivering; `fak cron audit` rolls
// the ledger up per job into fired / missed / deduped counts.
//
// This mirrors Hermes' Chronos reliability core (studied for epic #2871): its
// at-most-once = store compare-and-set + a `.tick.lock` file lock against duplicate
// ticks. Here the CAS is a scan of the fire ledger for an existing `fired` record
// at the same (job, slot); the `.tick.lock` is cronTickLock, a blocking O_EXCL
// lockfile that serializes concurrent ticks so the read-modify-write is atomic
// across processes. The slot key is the tick quantized to its --interval boundary,
// so every tick within one period maps to ONE slot — the dedup unit that makes a
// late catch-up tick land at most once. fak-does-it-better: the fire is a queryable
// ledger event, not a silent gap.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/exclusivefile"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	cronFireSchema     = "fak-cron-fire/1" // ledger row schema tag
	cronOutcomeFired   = "fired"           // the CAS admitted this (job, slot); run the job
	cronOutcomeDeduped = "deduped"         // a dup/overlap for an already-fired slot; skip

	// cronExitDeduped is `fak cron fire`'s exit code when the tick deduped, so a
	// caller can gate the job on it: `fak cron fire ... && <run job>` runs the job
	// only on a fresh fire (exit 0) and skips a duplicate (exit 3) — at-most-once
	// delivery at the shell seam.
	cronExitDeduped = 3

	cronTickWait = 5 * time.Second // how long a tick blocks for a live lock holder
	cronTickTTL  = 2 * time.Minute // a lockfile older than this is a crashed tick's leftover
	cronTickPoll = 5 * time.Millisecond
)

// cronFireRecord is one witnessed cron fire in the append-only ledger. Slot is the
// tick quantized to its schedule boundary (the CAS dedup key with Job); Outcome
// partitions every recorded tick into exactly one queryable class.
type cronFireRecord struct {
	Schema   string `json:"schema"`
	Job      string `json:"job"`
	Slot     string `json:"slot"`       // RFC3339 UTC, tick truncated to the interval
	Interval int64  `json:"interval_s"` // cadence in seconds (0 = one-shot)
	Outcome  string `json:"outcome"`    // cronOutcomeFired | cronOutcomeDeduped
	FiredAt  string `json:"fired_at"`   // RFC3339 UTC witnessed wall-clock of the tick
}

// cronJobAudit is the per-job rollup `fak cron audit` reports: distinct slots that
// fired, expected slots missed (a gap in the observed cadence), and duplicate ticks
// deduped to preserve single delivery.
type cronJobAudit struct {
	Job     string `json:"job"`
	Fired   int    `json:"fired"`
	Missed  int    `json:"missed"`
	Deduped int    `json:"deduped"`
}

// runCronFire records one fire under the dup-tick lock + (job, slot) compare-and-set.
// Exit 0 = fired (fresh slot, run the job); exit cronExitDeduped = deduped (an
// already-fired slot, skip); exit 2 = usage/IO error.
func runCronFire(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron fire", flag.ContinueOnError)
	fs.SetOutput(stderr)
	job := fs.String("job", "", "job/loop id this fire belongs to (required)")
	ledger := fs.String("ledger", "", "fire-witness ledger path, JSONL (required)")
	interval := fs.Duration("interval", 0, "firing cadence; the tick is quantized to this slot (0 = one-shot)")
	at := fs.String("at", "", "wall-clock tick time (RFC3339); default now — injectable for tests")
	slot := fs.String("slot", "", "override the computed slot key directly (default: --at truncated to --interval)")
	quiet := fs.Bool("quiet", false, "suppress the outcome line on stdout")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*job) == "" {
		fmt.Fprintln(stderr, "fak cron fire: --job is required")
		return 2
	}
	if strings.TrimSpace(*ledger) == "" {
		fmt.Fprintln(stderr, "fak cron fire: --ledger is required")
		return 2
	}

	fireAt, slotKey, ok := resolveCronTimeAndSlot(stderr, "fak cron fire", *at, *slot, *interval)
	if !ok {
		return 2
	}

	// The dup-tick lock makes the ledger read-modify-write atomic across concurrent
	// ticks (Chronos' .tick.lock). Held past the wait budget is an error, not a
	// silent double-fire — fail closed rather than race the CAS.
	release, err := cronTickLock(*ledger+".tick.lock", cronTickWait, cronTickTTL)
	if err != nil {
		fmt.Fprintf(stderr, "fak cron fire: %v\n", err)
		return 2
	}
	defer func() { _ = release() }()

	fires, err := cronReadFires(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak cron fire: read ledger: %v\n", err)
		return 2
	}
	outcome := cronOutcomeFired
	for _, r := range fires {
		if r.Job == *job && r.Slot == slotKey && r.Outcome == cronOutcomeFired {
			outcome = cronOutcomeDeduped // CAS miss: this slot already fired
			break
		}
	}
	rec := cronFireRecord{
		Schema:   cronFireSchema,
		Job:      *job,
		Slot:     slotKey,
		Interval: int64(interval.Seconds()),
		Outcome:  outcome,
		FiredAt:  fireAt.Format(time.RFC3339),
	}
	if err := cronAppendFire(*ledger, rec); err != nil {
		fmt.Fprintf(stderr, "fak cron fire: append ledger: %v\n", err)
		return 2
	}
	if !*quiet {
		fmt.Fprintf(stdout, "%s %s %s\n", outcome, *job, slotKey)
	}
	if outcome == cronOutcomeDeduped {
		return cronExitDeduped
	}
	return 0
}

// runCronAudit rolls the fire ledger up per job into fired/missed/deduped.
func runCronAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "fire-witness ledger path, JSONL (required)")
	jobFilter := fs.String("job", "", "restrict the report to one job id")
	asJSON := fs.Bool("json", false, "emit the per-job summary as JSON instead of a table")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*ledger) == "" {
		fmt.Fprintln(stderr, "fak cron audit: --ledger is required")
		return 2
	}
	fires, err := cronReadFires(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak cron audit: read ledger: %v\n", err)
		return 2
	}
	summ := cronAuditSummary(fires)
	if jf := strings.TrimSpace(*jobFilter); jf != "" {
		kept := make([]cronJobAudit, 0, len(summ))
		for _, s := range summ {
			if s.Job == jf {
				kept = append(kept, s)
			}
		}
		summ = kept
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summ); err != nil {
			fmt.Fprintf(stderr, "fak cron audit: encode: %v\n", err)
			return 2
		}
		return 0
	}
	if len(summ) == 0 {
		fmt.Fprintln(stdout, "no cron fires recorded")
		return 0
	}
	fmt.Fprintf(stdout, "%-24s %8s %8s %8s\n", "JOB", "FIRED", "MISSED", "DEDUPED")
	for _, s := range summ {
		fmt.Fprintf(stdout, "%-24s %8d %8d %8d\n", s.Job, s.Fired, s.Missed, s.Deduped)
	}
	return 0
}

// cronFireSlot quantizes a tick to its schedule slot: a positive interval truncates
// the tick to the interval boundary (every tick within one period → the SAME slot
// key, the CAS dedup unit); a zero/negative interval is a one-shot whose slot is the
// tick instant itself.
func cronFireSlot(at time.Time, interval time.Duration) string {
	at = at.UTC()
	if interval > 0 {
		at = at.Truncate(interval)
	}
	return at.Format(time.RFC3339)
}

// cronTickLock is the dup-tick file lock (Chronos' .tick.lock): an O_CREATE|O_EXCL
// lockfile whose create IS the atomic acquire, so a concurrent tick serializes on
// the ledger's read-modify-write. It blocks up to wait for a live holder and
// reclaims a lockfile older than ttl (a crashed tick's leftover — TTL-not-pid
// staleness, the same policy internal/resume.TryTickLock uses to avoid a
// procguard import and the os.Kill-on-Windows footgun).
func cronTickLock(path string, wait, ttl time.Duration) (release func() error, err error) {
	deadline := time.Now().Add(wait)
	for {
		cerr := exclusivefile.CreatePIDTime(path)
		if cerr == nil {
			return func() error {
				if rerr := os.Remove(path); rerr != nil && !os.IsNotExist(rerr) {
					return rerr
				}
				return nil
			}, nil
		}
		if !os.IsExist(cerr) {
			return nil, fmt.Errorf("tick lock %s: %w", path, cerr)
		}
		// Held. Reclaim if stale (past ttl), else wait and retry until the deadline.
		if fi, serr := os.Stat(path); serr == nil && time.Since(fi.ModTime()) >= ttl {
			_ = os.Remove(path)
			continue
		}
		if !time.Now().Before(deadline) {
			return nil, fmt.Errorf("tick lock %s held (waited %s)", path, wait)
		}
		time.Sleep(cronTickPoll)
	}
}

// cronReadFires loads well-formed fire rows from the ledger. A missing ledger is an
// empty history, not an error.
func cronReadFires(path string) ([]cronFireRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return jsonlledger.Parse(string(b), func(r cronFireRecord) bool {
		return r.Schema == cronFireSchema && r.Job != "" && r.Slot != ""
	}), nil
}

// cronAppendFire appends one fire row as a JSONL line, creating the ledger (and its
// dir) on first write.
func cronAppendFire(path string, rec cronFireRecord) error {
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}

// cronAuditSummary rolls fire rows up per job, sorted by job id for a stable report.
func cronAuditSummary(fires []cronFireRecord) []cronJobAudit {
	type acc struct {
		firedSlots map[string]bool
		deduped    int
		interval   time.Duration
	}
	byJob := map[string]*acc{}
	for _, r := range fires {
		a := byJob[r.Job]
		if a == nil {
			a = &acc{firedSlots: map[string]bool{}}
			byJob[r.Job] = a
		}
		switch r.Outcome {
		case cronOutcomeFired:
			a.firedSlots[r.Slot] = true
			if r.Interval > 0 {
				a.interval = time.Duration(r.Interval) * time.Second
			}
		case cronOutcomeDeduped:
			a.deduped++
		}
	}
	jobs := make([]string, 0, len(byJob))
	for j := range byJob {
		jobs = append(jobs, j)
	}
	sort.Strings(jobs)
	out := make([]cronJobAudit, 0, len(jobs))
	for _, j := range jobs {
		a := byJob[j]
		out = append(out, cronJobAudit{
			Job:     j,
			Fired:   len(a.firedSlots),
			Missed:  cronCountMissed(a.firedSlots, a.interval),
			Deduped: a.deduped,
		})
	}
	return out
}

// cronCountMissed walks the observed fired slots from earliest to latest in
// interval-sized steps and counts each expected slot with no fired record — a gap
// in the cadence. It is bounded between the first and last OBSERVED fire, so a slot
// not yet due is never counted missed. A zero interval (a one-shot, or fires that
// carried no cadence) has no cadence to miss → 0.
func cronCountMissed(firedSlots map[string]bool, interval time.Duration) int {
	if interval <= 0 || len(firedSlots) < 2 {
		return 0
	}
	slots := make([]time.Time, 0, len(firedSlots))
	for s := range firedSlots {
		t, err := time.Parse(time.RFC3339, s)
		if err != nil {
			continue
		}
		slots = append(slots, t.UTC())
	}
	if len(slots) < 2 {
		return 0
	}
	sort.Slice(slots, func(i, j int) bool { return slots[i].Before(slots[j]) })
	present := make(map[string]bool, len(slots))
	for _, t := range slots {
		present[t.Format(time.RFC3339)] = true
	}
	first, last := slots[0], slots[len(slots)-1]
	// Defensive iteration cap: the observed span / interval bounds the real count,
	// so a pathological interval can never spin here.
	const maxSteps = 1 << 20
	missed, steps := 0, 0
	for t := first; !t.After(last) && steps < maxSteps; t = t.Add(interval) {
		if !present[t.Format(time.RFC3339)] {
			missed++
		}
		steps++
	}
	return missed
}
