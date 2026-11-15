package gateway

// usage_ledger.go — the billing-grade per-request usage ledger keyed by the
// authenticated principal on every inference route (issue #14451).
//
// THE GAP IT CLOSES. The gateway_inference_turn / gateway_usage_record lines
// answer "what did this turn cost", but they attribute tenancy through the
// trace-owner table — a context-restore primitive filled only on some routes,
// keyed on client-chosen trace ids, first-writer-wins, wiped at 8192 entries.
// Usage could land on another tenant's name and nothing durable proved who paid
// for what: uninvoicable, unauditable, forgeable. This file closes the gap:
//
//   - PRINCIPAL FROM THE AUTHENTICATED KEY: the funnel receives the principal
//     stamped onto the request context by withAuth's keyset match (#5332) —
//     never a header or body field, so a caller presenting key-b can never be
//     attributed to tenant-a even with a spoofed X-Fak-Principal or a
//     pre-claimed X-Trace-Id. The legacy trace-owner binding is consulted ONLY
//     when no authenticated principal is present (single-tenant / unkeyed
//     deployments), preserving today's attribution there.
//   - DURABLE + TAMPER-EVIDENT: append-only JSONL, one sha256-chained row per
//     request (the same discipline as internal/usagelog: prev-hash chain,
//     monotonic seq, wall-clock anchor, per-row flush). Verify re-reads the
//     file and recomputes every link; a flipped byte breaks the chain at that
//     row. A corrupted chain refuses to extend (fail-closed at Open).
//   - CANCELLED / PARTIAL STREAMS RECORDED: the terminal finish reason maps to
//     a closed status vocabulary (completed / cancelled / error); a stream the
//     client abandoned mid-turn still lands its partial-usage row.
//   - EXPORT: GET /v1/fak/usage/ledger returns the last N rows plus whole-file
//     per-principal token sums and the chain verdict, so an operator reconciles
//     invoices against the ledger without parsing JSONL.
//
// Law A2: purely observational. The ledger never feeds the request path; an
// unset ledger or an append failure degrades silently (dropped counter) instead
// of failing the served request.

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// UsageLedgerSchema is the on-disk schema tag stamped into every row, so a
// reader tells this ledger apart from any other JSONL file and refuses to fold
// a row whose schema it does not understand.
const UsageLedgerSchema = "fak.gateway.usage-ledger.v1"

// usageLedgerGenesisHash is the prev_hash of the first row of a fresh chain.
const usageLedgerGenesisHash = "0000000000000000000000000000000000000000000000000000000000000000"

// usageLedgerDefaultN bounds the exported tail absent ?n=.
const usageLedgerDefaultN = 50

// Closed status vocabulary for a ledger row's terminal outcome.
const (
	UsageLedgerStatusCompleted = "completed"
	UsageLedgerStatusCancelled = "cancelled"
	UsageLedgerStatusError     = "error"
)

// UsageLedgerRow is one durable per-request billing row. Token axes are the
// provider's own counters (OBSERVED), normalized exactly like UsageRecord so
// every provider shape reads identically.
type UsageLedgerRow struct {
	Schema             string `json:"schema"`
	Seq                uint64 `json:"seq"`
	UnixMillis         int64  `json:"unix_millis"`
	RequestID          string `json:"request_id"` // session/trace family
	Principal          string `json:"principal"`  // authenticated keyset principal; "unauthenticated" when none
	Wire               string `json:"wire"`
	Model              string `json:"model,omitempty"`
	Status             string `json:"status"`
	PromptTokens       int64  `json:"prompt_tokens"`
	CompletionTokens   int64  `json:"completion_tokens"`
	CachedPromptTokens int64  `json:"cached_prompt_tokens"`
	TotalTokens        int64  `json:"total_tokens"`
	PrevHash           string `json:"prev_hash"`
	Hash               string `json:"hash"`
}

// usageLedgerPreimage is the hashed content of a row: every field except Hash
// itself (PrevHash included, so the chain links forward). Struct-field ordering
// makes the marshaling deterministic.
func (r UsageLedgerRow) usageLedgerPreimage() []byte {
	held := r.Hash
	r.Hash = ""
	b, _ := json.Marshal(r)
	r.Hash = held
	return b
}

func usageLedgerHash(row UsageLedgerRow) string {
	sum := sha256.Sum256(row.usageLedgerPreimage())
	return hex.EncodeToString(sum[:])
}

// UsageLedgerSum aggregates one principal's rows across the WHOLE file.
type UsageLedgerSum struct {
	Requests           int   `json:"requests"`
	PromptTokens       int64 `json:"prompt_tokens"`
	CompletionTokens   int64 `json:"completion_tokens"`
	CachedPromptTokens int64 `json:"cached_prompt_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
}

// UsageLedger appends hash-chained rows to a single JSONL file. One gateway
// process owns a ledger file; the mutex serializes head + write.
type UsageLedger struct {
	mu       sync.Mutex
	path     string
	f        *os.File
	seq      uint64
	lastHash string
	dropped  atomic.Uint64
	closed   bool
}

// ReadUsageLedger parses every row of the ledger file (oldest first). A missing
// file reads as zero rows, not an error.
func ReadUsageLedger(path string) ([]UsageLedgerRow, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var rows []UsageLedgerRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	lineNo := 0
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		lineNo++
		var row UsageLedgerRow
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("usage ledger %s: line %d: %w", path, lineNo, err)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("usage ledger %s: %w", path, err)
	}
	return rows, nil
}

// VerifyUsageLedger recomputes the hash chain over parsed rows: seq must run
// 1..N, each prev_hash must equal the previous row's hash (genesis zeros for
// the first), and each hash must equal the recomputed digest. Any violation
// fails the check — a tamper is detectable, not preventable.
func VerifyUsageLedger(rows []UsageLedgerRow) error {
	prev := usageLedgerGenesisHash
	for i, row := range rows {
		n := i + 1
		if row.Schema != UsageLedgerSchema {
			return fmt.Errorf("usage ledger: row %d: schema %q, want %q", n, row.Schema, UsageLedgerSchema)
		}
		if row.Seq != uint64(n) {
			return fmt.Errorf("usage ledger: row %d: seq %d, want %d", n, row.Seq, uint64(n))
		}
		if row.PrevHash != prev {
			return fmt.Errorf("usage ledger: row %d: prev_hash mismatch (chain broken)", n)
		}
		if got := usageLedgerHash(row); got != row.Hash {
			return fmt.Errorf("usage ledger: row %d: hash mismatch (content altered)", n)
		}
		prev = row.Hash
	}
	return nil
}

// VerifyUsageLedgerFile reads and verifies the whole chain on disk.
func VerifyUsageLedgerFile(path string) error {
	rows, err := ReadUsageLedger(path)
	if err != nil {
		return err
	}
	return VerifyUsageLedger(rows)
}

// OpenUsageLedger opens (creating if absent) the ledger at path and recovers
// the chain head from disk: it re-verifies the existing chain and resumes
// seq/hash from the last valid row. A corrupt or truncated chain REFUSES to
// open — extending a broken chain would launder the tamper into the tail, so
// the file must be repaired or rotated first. The parent directory must exist.
func OpenUsageLedger(path string) (*UsageLedger, error) {
	rows, err := ReadUsageLedger(path)
	if err != nil {
		return nil, err
	}
	if err := VerifyUsageLedger(rows); err != nil {
		return nil, fmt.Errorf("usage ledger %s: %w", path, err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("usage ledger: open %s: %w", path, err)
	}
	l := &UsageLedger{path: path, f: f, lastHash: usageLedgerGenesisHash}
	if len(rows) > 0 {
		l.seq = rows[len(rows)-1].Seq
		l.lastHash = rows[len(rows)-1].Hash
	}
	return l, nil
}

// Append records one served request. It stamps Schema/Seq/UnixMillis/PrevHash/
// Hash, writes the row as a single O_APPEND line, and fsyncs. Callers populate
// every other field. A failed write increments Dropped and returns the error;
// the caller (the observation funnel) treats it as non-fatal to the request.
// A returned fsync error leaves the row written and chained (durability risk,
// not loss).
func (l *UsageLedger) Append(in UsageLedgerRow) error {
	if l == nil {
		return errors.New("usage ledger: nil")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed || l.f == nil {
		l.dropped.Add(1)
		return errors.New("usage ledger: closed")
	}
	row := in
	row.Schema = UsageLedgerSchema
	row.Seq = l.seq + 1
	if row.UnixMillis <= 0 {
		row.UnixMillis = time.Now().UnixMilli()
	}
	row.PrevHash = l.lastHash
	row.Hash = usageLedgerHash(row)
	b, err := json.Marshal(row)
	if err != nil {
		l.dropped.Add(1)
		return err
	}
	b = append(b, '\n')
	if _, err := l.f.Write(b); err != nil {
		l.dropped.Add(1)
		return err
	}
	l.seq = row.Seq
	l.lastHash = row.Hash
	return l.f.Sync()
}

// Close flushes and releases the file. Idempotent.
func (l *UsageLedger) Close() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if l.f == nil {
		return nil
	}
	err := l.f.Close()
	l.f = nil
	return err
}

// Dropped counts rows lost to write failures since Open.
func (l *UsageLedger) Dropped() uint64 {
	if l == nil {
		return 0
	}
	return l.dropped.Load()
}

// Path returns the ledger file path.
func (l *UsageLedger) Path() string {
	if l == nil {
		return ""
	}
	return l.path
}

// resolveUsageLedgerPrincipal picks the row's principal: the authenticated
// keyset principal (stamped by withAuth) is AUTHORITATIVE and never yields to
// anything client-supplied; the legacy trace-owner binding is consulted only
// when no authenticated principal is present; neither yields "unauthenticated".
func resolveUsageLedgerPrincipal(authenticated, traceOwner string) string {
	if p := strings.TrimSpace(authenticated); p != "" {
		return p
	}
	if o := strings.TrimSpace(traceOwner); o != "" {
		return o
	}
	return "unauthenticated"
}

// usageLedgerStatus maps the terminal finish reason onto the closed ledger
// status vocabulary. Cancelled/partial streams keep their row with the partial
// usage observed up to the abort.
func usageLedgerStatus(finishReason string) string {
	switch strings.ToLower(strings.TrimSpace(finishReason)) {
	case "cancelled", "canceled", "abort", "aborted", "client_disconnected":
		return UsageLedgerStatusCancelled
	case "error", "upstream_error", "server_error":
		return UsageLedgerStatusError
	default:
		return UsageLedgerStatusCompleted
	}
}

// appendUsageLedgerRow lands the billing-grade ledger row for one served turn
// (#14451). Observational only: a nil/unset ledger or an append failure never
// affects the request.
func (s *Server) appendUsageLedgerRow(traceID, wire, model string, usage agent.Usage, unixMillis int64, finishReason, principal string) {
	if s == nil {
		return
	}
	l := s.UsageLedger()
	if l == nil {
		return
	}
	owner, _ := s.traceOwnerOf(traceID)
	row := UsageLedgerRow{
		UnixMillis:         unixMillis,
		RequestID:          strings.TrimSpace(traceID),
		Principal:          resolveUsageLedgerPrincipal(principal, owner),
		Wire:               wire,
		Model:              model,
		Status:             usageLedgerStatus(finishReason),
		PromptTokens:       int64(clampNonNeg(usage.PromptTokens)),
		CompletionTokens:   int64(clampNonNeg(usage.CompletionTokens)),
		CachedPromptTokens: int64(clampNonNeg(usage.CachedPromptTokens())),
		TotalTokens:        int64(clampNonNeg(usage.TotalTokens)),
	}
	if err := l.Append(row); err != nil && s.logf != nil {
		s.logf("gateway: usage ledger append failed (dropped=%d): %v", l.Dropped(), err)
	}
}

// SetUsageLedger enables (or disables, with nil) the billing-grade usage
// ledger. Safe for concurrent use; the request path is byte-for-byte unchanged
// while it is unset.
func (s *Server) SetUsageLedger(l *UsageLedger) {
	if s == nil {
		return
	}
	s.usageLedger.Store(l)
}

// UsageLedger returns the enabled ledger, or nil when off.
func (s *Server) UsageLedger() *UsageLedger {
	if s == nil {
		return nil
	}
	return s.usageLedger.Load()
}

// usageLedgerReport is the GET /v1/fak/usage/ledger response: the last N rows,
// whole-file per-principal token sums, and the chain verdict. Counts only —
// never prompts or content.
type usageLedgerReport struct {
	Schema          string                    `json:"schema"`
	Enabled         bool                      `json:"enabled"`
	ChainOK         bool                      `json:"chain_ok"`
	Rows            []UsageLedgerRow          `json:"rows"`
	SumsByPrincipal map[string]UsageLedgerSum `json:"sums_by_principal"`
	Dropped         uint64                    `json:"dropped"`
}

// usageLedgerReport folds the ledger file into the export report. Sums cover
// the WHOLE file regardless of N; Rows carries the last N.
func (s *Server) usageLedgerReport(n int) usageLedgerReport {
	rep := usageLedgerReport{Schema: UsageLedgerSchema, SumsByPrincipal: map[string]UsageLedgerSum{}}
	if s == nil {
		return rep
	}
	l := s.UsageLedger()
	if l == nil {
		return rep
	}
	rep.Enabled = true
	rep.Dropped = l.Dropped()
	rows, err := ReadUsageLedger(l.Path())
	if err != nil {
		return rep
	}
	rep.ChainOK = VerifyUsageLedger(rows) == nil
	for _, row := range rows {
		sum := rep.SumsByPrincipal[row.Principal]
		sum.Requests++
		sum.PromptTokens += row.PromptTokens
		sum.CompletionTokens += row.CompletionTokens
		sum.CachedPromptTokens += row.CachedPromptTokens
		sum.TotalTokens += row.TotalTokens
		rep.SumsByPrincipal[row.Principal] = sum
	}
	if n > 0 && len(rows) > n {
		rows = rows[len(rows)-n:]
	}
	rep.Rows = rows
	return rep
}

// handleFakUsageLedger serves the billing-grade usage export (#14451): the
// last N ledger rows (?n=, default 50), whole-file per-principal token sums,
// and the chain verdict. Read-only GET, counts only.
func (s *Server) handleFakUsageLedger(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	n := usageLedgerDefaultN
	if parsed, err := strconv.Atoi(strings.TrimSpace(r.URL.Query().Get("n"))); err == nil && parsed > 0 {
		n = parsed
	}
	writeJSON(w, http.StatusOK, s.usageLedgerReport(n))
}
