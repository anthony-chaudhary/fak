package modelroute

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// IssueAuditLoopScanCapMax is the batch policy's hard ceiling: the producer
// scans at most this many issues per planning run. A discoverer that would
// return more is a bug, not a silent truncation, so the tick refuses it.
const IssueAuditLoopScanCapMax = 500

const (
	// IssueAuditLoopCursorSchema stamps the durable scheduler cursor.
	IssueAuditLoopCursorSchema = "fak-crossaudit-loop-cursor/v1"
	// IssueAuditLoopReportSchema stamps the typed per-tick decision.
	IssueAuditLoopReportSchema = "fak-crossaudit-loop-report/v1"
)

// IssueAuditLoopState is the closed vocabulary of typed next decisions a tick
// reports. It is deliberately small so an operator (or a supervisor) can act on
// the loop's health without reading receipts:
//
//	ADVANCING — at least one eligible subject reached a terminal (PASS/REFUTE)
//	            audit this tick; the cursor moved forward.
//	WAIT      — no terminal progress this tick, but eligible work is only
//	            transiently blocked (backoff, provider cooldown, a contended
//	            per-subject lease, or the per-tick cap) OR everything eligible is
//	            already settled. Nothing is wrong; come back on the next trigger.
//	STALLED   — eligible pending work exists and ALL of it has exhausted its
//	            retry budget (dead-lettered). Nothing progresses without operator
//	            intervention.
//	DARK      — discovery surfaced no eligible subjects at all (empty snapshot or
//	            everything filtered out by eligibility/risk). A potential blind
//	            spot: the loop cannot see auditable work.
type IssueAuditLoopState string

const (
	IssueAuditLoopAdvancing IssueAuditLoopState = "ADVANCING"
	IssueAuditLoopWait      IssueAuditLoopState = "WAIT"
	IssueAuditLoopStalled   IssueAuditLoopState = "STALLED"
	IssueAuditLoopDark      IssueAuditLoopState = "DARK"
)

func (s IssueAuditLoopState) Valid() bool {
	switch s {
	case IssueAuditLoopAdvancing, IssueAuditLoopWait, IssueAuditLoopStalled, IssueAuditLoopDark:
		return true
	default:
		return false
	}
}

// IssueAuditLoopSubject is one closed-issue candidate in a discovery snapshot.
// MarkerKey is the fak-crossaudit marker (one audit per key); the loop dedupes
// its at-most-once cursor by IssueNumber, which the ledger's receipts also
// carry, so crash recovery can reconcile the two without a marker lookup.
type IssueAuditLoopSubject struct {
	IssueNumber      int       `json:"issue_number"`
	MarkerKey        string    `json:"marker_key,omitempty"`
	Risk             AuditRisk `json:"risk,omitempty"`
	Eligible         bool      `json:"eligible"`
	IneligibleReason string    `json:"ineligible_reason,omitempty"`
}

// IssueAuditLoopDiscoverer snapshots eligible closed issues. It must return at
// most scanCap rows; the tick enforces the ceiling either way.
type IssueAuditLoopDiscoverer interface {
	DiscoverAuditSubjects(ctx context.Context, scanCap int) ([]IssueAuditLoopSubject, error)
}

type IssueAuditLoopDiscovererFunc func(context.Context, int) ([]IssueAuditLoopSubject, error)

func (f IssueAuditLoopDiscovererFunc) DiscoverAuditSubjects(ctx context.Context, scanCap int) ([]IssueAuditLoopSubject, error) {
	return f(ctx, scanCap)
}

// IssueAuditLoopAuditor performs one subject's cross-model audit and returns its
// receipt. UNAVAILABLE (provider/host failure) is returned as a verdict, not an
// error, when the spine produced a receipt; a hard error (evidence fetch
// failure) is returned as a non-nil error and the tick treats it as UNAVAILABLE
// for retry accounting without appending anything.
type IssueAuditLoopAuditor interface {
	AuditSubject(ctx context.Context, subject IssueAuditLoopSubject) (IssueAuditReceipt, error)
}

type IssueAuditLoopAuditorFunc func(context.Context, IssueAuditLoopSubject) (IssueAuditReceipt, error)

func (f IssueAuditLoopAuditorFunc) AuditSubject(ctx context.Context, subject IssueAuditLoopSubject) (IssueAuditReceipt, error) {
	return f(ctx, subject)
}

// IssueAuditLoopLease admits at most one worker per subject. The lease is held
// only for the duration of that subject's audit — never a single global lease
// across every slow model call — so a contended subject is skipped this tick
// while its siblings still make progress.
type IssueAuditLoopLease interface {
	AcquireSubjectLease(subjectKey string) (release func(), acquired bool, err error)
}

// IssueAuditLoopClock injects wall-clock time so ticks are deterministic under
// test. A nil clock defaults to time.Now.
type IssueAuditLoopClock func() time.Time

// IssueAuditLoopConfig is one tick's complete input. The three seams
// (Discoverer, Auditor, Lease) are injected so the tick is a pure function of
// its inputs plus the on-disk ledger and cursor.
type IssueAuditLoopConfig struct {
	LedgerPath string
	CursorPath string

	ScanCap  int // discovery ceiling (<= IssueAuditLoopScanCapMax; 0 => max)
	BatchCap int // audits attempted per tick (> 0; caps work per tick)

	MaxAttempts      int           // retries before a subject is dead-lettered (default 3)
	BackoffBase      time.Duration // first retry delay; doubles per attempt (default 1m)
	BackoffCap       time.Duration // maximum retry delay (default 1h)
	ProviderCooldown time.Duration // pause a provider after an UNAVAILABLE (default 5m)

	DryRun       bool  // plan only: no lease, no audit, no ledger/cursor write
	ReplayIssues []int // force re-audit of these issue numbers even if settled

	Now        IssueAuditLoopClock
	Discoverer IssueAuditLoopDiscoverer
	Auditor    IssueAuditLoopAuditor
	Lease      IssueAuditLoopLease
}

func (cfg IssueAuditLoopConfig) scanCap() int {
	if cfg.ScanCap <= 0 || cfg.ScanCap > IssueAuditLoopScanCapMax {
		return IssueAuditLoopScanCapMax
	}
	return cfg.ScanCap
}

func (cfg IssueAuditLoopConfig) maxAttempts() int {
	if cfg.MaxAttempts <= 0 {
		return 3
	}
	return cfg.MaxAttempts
}

func (cfg IssueAuditLoopConfig) backoffBase() time.Duration {
	if cfg.BackoffBase <= 0 {
		return time.Minute
	}
	return cfg.BackoffBase
}

func (cfg IssueAuditLoopConfig) backoffCap() time.Duration {
	if cfg.BackoffCap <= 0 {
		return time.Hour
	}
	if cfg.BackoffCap < cfg.backoffBase() {
		return cfg.backoffBase()
	}
	return cfg.BackoffCap
}

func (cfg IssueAuditLoopConfig) providerCooldown() time.Duration {
	if cfg.ProviderCooldown <= 0 {
		return 5 * time.Minute
	}
	return cfg.ProviderCooldown
}

func (cfg IssueAuditLoopConfig) backoff(attempts int) time.Duration {
	base, cap := cfg.backoffBase(), cfg.backoffCap()
	d := base
	for i := 1; i < attempts && d < cap; i++ {
		d *= 2
	}
	if d > cap {
		d = cap
	}
	return d
}

// issueAuditLoopSettled is one at-most-once cursor row: a subject whose terminal
// receipt is in the ledger. The verdict and audit key mirror the ledger so a
// later tick can skip the subject without re-running its expensive audit.
type issueAuditLoopSettled struct {
	IssueNumber int               `json:"issue_number"`
	AuditKey    string            `json:"audit_key"`
	Verdict     CrossAuditVerdict `json:"verdict"`
}

// issueAuditLoopDeadLetter tracks a subject whose audit did not reach a terminal
// verdict. UNAVAILABLE/INCONCLUSIVE never advance the cursor — they accumulate
// bounded retries here so an unavailable row is preserved, not silently lost.
type issueAuditLoopDeadLetter struct {
	IssueNumber          int    `json:"issue_number"`
	Attempts             int    `json:"attempts"`
	LastClass            string `json:"last_class"`
	LastReason           string `json:"last_reason,omitempty"`
	Provider             string `json:"provider,omitempty"`
	NextEligibleUnixNano int64  `json:"next_eligible_unix_nano"`
	DeadLettered         bool   `json:"dead_lettered"`
}

// IssueAuditLoopCursor is the durable scheduler state that survives restarts.
// The receipt ledger remains the source of truth for at-most-once; this cursor
// is the cheap index that lets a tick skip settled subjects and preserve retry
// backoff/cooldown timing across a crash. LedgerRows/LedgerHead pin the ledger
// prefix the cursor was last reconciled against (truncation witness).
type IssueAuditLoopCursor struct {
	Schema            string                           `json:"schema"`
	UpdatedAtUnixNano int64                            `json:"updated_at_unix_nano"`
	LedgerRows        uint64                           `json:"ledger_rows"`
	LedgerHead        string                           `json:"ledger_head,omitempty"`
	Settled           map[int]issueAuditLoopSettled    `json:"settled,omitempty"`
	DeadLetters       map[int]issueAuditLoopDeadLetter `json:"dead_letters,omitempty"`
	Cooldowns         map[string]int64                 `json:"cooldowns,omitempty"`
	Hash              string                           `json:"hash"`
}

func newIssueAuditLoopCursor() IssueAuditLoopCursor {
	return IssueAuditLoopCursor{
		Schema:      IssueAuditLoopCursorSchema,
		Settled:     map[int]issueAuditLoopSettled{},
		DeadLetters: map[int]issueAuditLoopDeadLetter{},
		Cooldowns:   map[string]int64{},
	}
}

func (c IssueAuditLoopCursor) recomputeHash() string {
	c.Hash = ""
	return hashJSON(c)
}

// LoadIssueAuditLoopCursor reads and verifies the durable cursor. A missing file
// is a valid empty cursor (first run / fresh install), never an error.
func LoadIssueAuditLoopCursor(path string) (IssueAuditLoopCursor, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return IssueAuditLoopCursor{}, fmt.Errorf("modelroute: audit loop cursor path is required")
	}
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return newIssueAuditLoopCursor(), nil
	}
	if err != nil {
		return IssueAuditLoopCursor{}, fmt.Errorf("modelroute: read audit loop cursor: %w", err)
	}
	var cursor IssueAuditLoopCursor
	if err := json.Unmarshal(b, &cursor); err != nil {
		return IssueAuditLoopCursor{}, fmt.Errorf("modelroute: parse audit loop cursor: %w", err)
	}
	if cursor.Schema != IssueAuditLoopCursorSchema {
		return IssueAuditLoopCursor{}, fmt.Errorf("modelroute: audit loop cursor schema %q, want %q", cursor.Schema, IssueAuditLoopCursorSchema)
	}
	if want := cursor.recomputeHash(); cursor.Hash != want {
		return IssueAuditLoopCursor{}, fmt.Errorf("modelroute: audit loop cursor hash mismatch: stamped %s, recomputed %s", cursor.Hash, want)
	}
	if cursor.Settled == nil {
		cursor.Settled = map[int]issueAuditLoopSettled{}
	}
	if cursor.DeadLetters == nil {
		cursor.DeadLetters = map[int]issueAuditLoopDeadLetter{}
	}
	if cursor.Cooldowns == nil {
		cursor.Cooldowns = map[string]int64{}
	}
	return cursor, nil
}

func saveIssueAuditLoopCursor(path string, cursor IssueAuditLoopCursor, now time.Time) error {
	cursor.Schema = IssueAuditLoopCursorSchema
	cursor.UpdatedAtUnixNano = now.UTC().UnixNano()
	cursor.Hash = ""
	cursor.Hash = cursor.recomputeHash()
	b, err := json.Marshal(cursor)
	if err != nil {
		return fmt.Errorf("modelroute: encode audit loop cursor: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("modelroute: create audit loop cursor directory: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return fmt.Errorf("modelroute: write audit loop cursor: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("modelroute: commit audit loop cursor: %w", err)
	}
	return nil
}

// IssueAuditLoopReport is the typed tick outcome. State is the headline; the
// counts explain it and every deferred/dead subject is named so nothing is a
// silent gap.
type IssueAuditLoopReport struct {
	Schema          string              `json:"schema"`
	State           IssueAuditLoopState `json:"state"`
	DryRun          bool                `json:"dry_run"`
	ScanCap         int                 `json:"scan_cap"`
	BatchCap        int                 `json:"batch_cap"`
	Discovered      int                 `json:"discovered"`
	Eligible        int                 `json:"eligible"`
	AlreadySettled  int                 `json:"already_settled"`
	Planned         int                 `json:"planned"`
	Audited         int                 `json:"audited"`
	Passed          int                 `json:"passed"`
	Refuted         int                 `json:"refuted"`
	Inconclusive    int                 `json:"inconclusive"`
	Unavailable     int                 `json:"unavailable"`
	LeaseConflicts  int                 `json:"lease_conflicts"`
	Deferred        int                 `json:"deferred"`
	DeadLettered    int                 `json:"dead_lettered"`
	CapDeferred     int                 `json:"cap_deferred"`
	DarkSubjects    []int               `json:"dark_subjects,omitempty"`
	PlannedSubjects []int               `json:"planned_subjects,omitempty"`
	DeadLetterQueue []int               `json:"dead_letter_queue,omitempty"`
	LedgerRows      uint64              `json:"ledger_rows"`
	LedgerHead      string              `json:"ledger_head,omitempty"`
	Notes           []string            `json:"notes,omitempty"`
}

// RunIssueAuditLoopTick runs exactly one deterministic, idempotent producer
// tick. It never launches a live worker: it fetches a bounded snapshot, leases
// each eligible unsettled subject per-subject, invokes the injected auditor,
// appends only terminal (PASS/REFUTE) receipts at-most-once, records
// UNAVAILABLE/INCONCLUSIVE as bounded retries (so no unavailable row is lost),
// advances the durable cursor, and returns a typed next decision.
func RunIssueAuditLoopTick(ctx context.Context, cfg IssueAuditLoopConfig) (IssueAuditLoopReport, error) {
	if strings.TrimSpace(cfg.LedgerPath) == "" {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop needs a ledger path")
	}
	if strings.TrimSpace(cfg.CursorPath) == "" {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop needs a cursor path")
	}
	if cfg.BatchCap <= 0 {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop batch cap must be positive")
	}
	if cfg.Discoverer == nil {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop needs a discoverer")
	}
	if !cfg.DryRun && cfg.Auditor == nil {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop needs an auditor")
	}
	nowFn := cfg.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	now := nowFn().UTC()

	scanCap := cfg.scanCap()
	subjects, err := cfg.Discoverer.DiscoverAuditSubjects(ctx, scanCap)
	if err != nil {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop discovery: %w", err)
	}
	if len(subjects) > scanCap {
		return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop discoverer returned %d subjects over scan cap %d", len(subjects), scanCap)
	}

	cursor, err := LoadIssueAuditLoopCursor(cfg.CursorPath)
	if err != nil {
		return IssueAuditLoopReport{}, err
	}
	verification, err := reconcileIssueAuditLoopCursor(&cursor, cfg.LedgerPath)
	if err != nil {
		return IssueAuditLoopReport{}, err
	}
	replay := make(map[int]bool, len(cfg.ReplayIssues))
	for _, n := range cfg.ReplayIssues {
		replay[n] = true
	}

	report := IssueAuditLoopReport{
		Schema:     IssueAuditLoopReportSchema,
		DryRun:     cfg.DryRun,
		ScanCap:    scanCap,
		BatchCap:   cfg.BatchCap,
		Discovered: len(subjects),
		LedgerRows: verification.Cursor.Rows,
		LedgerHead: verification.Cursor.HeadHash,
	}

	// Partition the snapshot: ineligible rows are DARK candidates; eligible rows
	// split into already-settled, retry-deferred, and ready-to-plan.
	var ready []IssueAuditLoopSubject
	for _, subject := range subjects {
		if !subject.Eligible {
			report.DarkSubjects = append(report.DarkSubjects, subject.IssueNumber)
			continue
		}
		report.Eligible++
		if replay[subject.IssueNumber] {
			delete(cursor.Settled, subject.IssueNumber)
			delete(cursor.DeadLetters, subject.IssueNumber)
		}
		if _, ok := cursor.Settled[subject.IssueNumber]; ok {
			report.AlreadySettled++
			continue
		}
		if dl, ok := cursor.DeadLetters[subject.IssueNumber]; ok {
			eligibleAt := issueAuditLoopRetryEligibleAt(dl, cursor)
			if dl.DeadLettered || now.UnixNano() < eligibleAt {
				report.Deferred++
				if dl.DeadLettered {
					report.DeadLettered++
					report.DeadLetterQueue = append(report.DeadLetterQueue, subject.IssueNumber)
				}
				continue
			}
		}
		ready = append(ready, subject)
	}
	sort.Slice(ready, func(i, j int) bool { return ready[i].IssueNumber < ready[j].IssueNumber })

	batch := ready
	if len(batch) > cfg.BatchCap {
		batch = ready[:cfg.BatchCap]
		report.CapDeferred = len(ready) - cfg.BatchCap
	}
	report.Planned = len(batch)
	for _, subject := range batch {
		report.PlannedSubjects = append(report.PlannedSubjects, subject.IssueNumber)
	}

	if cfg.DryRun {
		report.State = classifyIssueAuditLoopState(report, ready)
		sort.Ints(report.DeadLetterQueue)
		return report, nil
	}

	for _, subject := range batch {
		if ctx.Err() != nil {
			report.Notes = append(report.Notes, "stop signal: "+ctx.Err().Error())
			break
		}
		release, acquired, leaseErr := acquireIssueAuditLoopLease(cfg.Lease, subject)
		if leaseErr != nil {
			return IssueAuditLoopReport{}, fmt.Errorf("modelroute: audit loop lease issue #%d: %w", subject.IssueNumber, leaseErr)
		}
		if !acquired {
			report.LeaseConflicts++
			report.Notes = append(report.Notes, fmt.Sprintf("issue #%d: lease busy, deferred", subject.IssueNumber))
			continue
		}
		receipt, auditErr := cfg.Auditor.AuditSubject(ctx, subject)
		release()

		verdict := receipt.Verdict
		if auditErr != nil && verdict == "" {
			verdict = CrossAuditUnavailable
		}
		switch verdict {
		case CrossAuditPass, CrossAuditRefute:
			appendResult, err := AppendAuditReceiptLedger(cfg.LedgerPath, receipt)
			if err != nil {
				// A conflicting duplicate is surfaced, not swallowed, but it does
				// not abort the tick: other subjects still make progress.
				report.Notes = append(report.Notes, fmt.Sprintf("issue #%d: ledger append refused: %v", subject.IssueNumber, err))
				continue
			}
			cursor.Settled[subject.IssueNumber] = issueAuditLoopSettled{
				IssueNumber: subject.IssueNumber,
				AuditKey:    appendResult.Row.AuditKey,
				Verdict:     verdict,
			}
			delete(cursor.DeadLetters, subject.IssueNumber)
			cursor.LedgerRows = appendResult.Cursor.Rows
			cursor.LedgerHead = appendResult.Cursor.HeadHash
			report.Audited++
			if verdict == CrossAuditPass {
				report.Passed++
			} else {
				report.Refuted++
			}
		default:
			// UNAVAILABLE / INCONCLUSIVE / anything non-terminal: never advance the
			// cursor, never append — retain the row as a bounded retry so it is not
			// lost as a permanent blind spot.
			class := strings.ToLower(string(verdict))
			reason := strings.TrimSpace(receipt.Reason)
			provider := strings.TrimSpace(receipt.Auditor.Provider)
			if auditErr != nil {
				class = "unavailable"
				if reason == "" {
					reason = auditErr.Error()
				}
			}
			if verdict == CrossAuditInconclusive {
				report.Inconclusive++
			} else {
				report.Unavailable++
			}
			issueAuditLoopRecordRetry(&cursor, subject.IssueNumber, class, reason, provider, verdict, now, cfg)
		}
	}

	report.LedgerRows = cursor.LedgerRows
	report.LedgerHead = cursor.LedgerHead
	report.State = classifyIssueAuditLoopState(report, ready)
	sort.Ints(report.DeadLetterQueue)

	if err := saveIssueAuditLoopCursor(cfg.CursorPath, cursor, now); err != nil {
		return IssueAuditLoopReport{}, err
	}
	return report, nil
}

// reconcileIssueAuditLoopCursor makes the ledger authoritative for at-most-once.
// Every terminal receipt in the ledger is folded into the cursor's settled set
// by issue number, so a subject that reached a terminal receipt but whose cursor
// write was lost to a crash is still treated as settled on the next tick.
func reconcileIssueAuditLoopCursor(cursor *IssueAuditLoopCursor, ledgerPath string) (AuditReceiptLedgerVerification, error) {
	if cursor.Settled == nil {
		cursor.Settled = map[int]issueAuditLoopSettled{}
	}
	if cursor.DeadLetters == nil {
		cursor.DeadLetters = map[int]issueAuditLoopDeadLetter{}
	}
	if cursor.Cooldowns == nil {
		cursor.Cooldowns = map[string]int64{}
	}
	if _, err := os.Stat(ledgerPath); os.IsNotExist(err) {
		return AuditReceiptLedgerVerification{Cursor: AuditReceiptLedgerCursor{Schema: AuditReceiptLedgerCursorSchema}}, nil
	}
	verification, err := VerifyAuditReceiptLedger(ledgerPath)
	if err != nil {
		return AuditReceiptLedgerVerification{}, fmt.Errorf("modelroute: audit loop ledger: %w", err)
	}
	f, err := os.Open(ledgerPath)
	if err != nil {
		return AuditReceiptLedgerVerification{}, fmt.Errorf("modelroute: open audit loop ledger: %w", err)
	}
	defer f.Close()
	rows, err := ParseAuditReceiptLedger(f)
	if err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	for _, row := range rows {
		verdict := row.Receipt.Verdict
		if verdict != CrossAuditPass && verdict != CrossAuditRefute {
			continue
		}
		issue := row.Receipt.Subject.IssueNumber
		if issue <= 0 {
			continue
		}
		cursor.Settled[issue] = issueAuditLoopSettled{IssueNumber: issue, AuditKey: row.AuditKey, Verdict: verdict}
		delete(cursor.DeadLetters, issue)
	}
	cursor.LedgerRows = verification.Cursor.Rows
	cursor.LedgerHead = verification.Cursor.HeadHash
	return verification, nil
}

func issueAuditLoopRecordRetry(cursor *IssueAuditLoopCursor, issue int, class, reason, provider string, verdict CrossAuditVerdict, now time.Time, cfg IssueAuditLoopConfig) {
	dl := cursor.DeadLetters[issue]
	dl.IssueNumber = issue
	dl.Attempts++
	dl.LastClass = class
	dl.LastReason = reason
	if provider != "" {
		dl.Provider = provider
	}
	backoff := cfg.backoff(dl.Attempts)
	dl.NextEligibleUnixNano = now.Add(backoff).UnixNano()
	if verdict == CrossAuditUnavailable && dl.Provider != "" {
		cooldownUntil := now.Add(cfg.providerCooldown()).UnixNano()
		if existing := cursor.Cooldowns[dl.Provider]; existing > cooldownUntil {
			cooldownUntil = existing
		}
		cursor.Cooldowns[dl.Provider] = cooldownUntil
	}
	if dl.Attempts >= cfg.maxAttempts() {
		dl.DeadLettered = true
	}
	cursor.DeadLetters[issue] = dl
}

// issueAuditLoopRetryEligibleAt is the effective earliest retry time: the later
// of the subject's own backoff and any active cooldown on its last provider, so
// a provider-wide outage pushes every one of its subjects out together.
func issueAuditLoopRetryEligibleAt(dl issueAuditLoopDeadLetter, cursor IssueAuditLoopCursor) int64 {
	at := dl.NextEligibleUnixNano
	if dl.Provider != "" {
		if cooldown, ok := cursor.Cooldowns[dl.Provider]; ok && cooldown > at {
			at = cooldown
		}
	}
	return at
}

func acquireIssueAuditLoopLease(lease IssueAuditLoopLease, subject IssueAuditLoopSubject) (func(), bool, error) {
	if lease == nil {
		return func() {}, true, nil
	}
	return lease.AcquireSubjectLease(fmt.Sprintf("issue-%d", subject.IssueNumber))
}

func classifyIssueAuditLoopState(report IssueAuditLoopReport, ready []IssueAuditLoopSubject) IssueAuditLoopState {
	if report.Audited > 0 {
		return IssueAuditLoopAdvancing
	}
	// Anything still workable later (a deferred/backing-off subject, a contended
	// lease, or a subject the per-tick cap held back) means the loop is merely
	// waiting, not stuck.
	pendingTransient := report.CapDeferred > 0 || report.LeaseConflicts > 0 ||
		(report.Deferred-report.DeadLettered) > 0 || len(ready) > 0
	if report.Eligible == 0 {
		return IssueAuditLoopDark
	}
	if pendingTransient {
		return IssueAuditLoopWait
	}
	if report.DeadLettered > 0 {
		return IssueAuditLoopStalled
	}
	// Every eligible subject is settled and nothing is pending: caught up.
	return IssueAuditLoopWait
}

// FlockIssueAuditLoopLease is the default per-subject lease: one advisory file
// lock per subject key under Dir. It is held only across a single subject's
// audit, never as a global lock across the whole batch.
type FlockIssueAuditLoopLease struct {
	Dir string
}

func (l FlockIssueAuditLoopLease) AcquireSubjectLease(subjectKey string) (func(), bool, error) {
	dir := strings.TrimSpace(l.Dir)
	if dir == "" {
		return nil, false, fmt.Errorf("modelroute: audit loop lease dir is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, false, fmt.Errorf("modelroute: create audit loop lease dir: %w", err)
	}
	safe := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, subjectKey)
	lockPath := filepath.Join(dir, safe+".lease")
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, false, fmt.Errorf("modelroute: open audit loop lease: %w", err)
	}
	if err := flock.TryLock(f); err != nil {
		f.Close()
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("modelroute: lock audit loop lease: %w", err)
	}
	return func() {
		flock.Unlock(f)
		f.Close()
	}, true, nil
}
