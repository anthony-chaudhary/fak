package slackoutbox

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Spool/state file names inside the outbox directory. Both are append-only JSONL: the
// spool holds message rows (what to send), the state file holds transitions (what
// happened to each nonce). Replaying state over spool yields the effective queue, so a
// restart resumes exactly where the last process stopped.
const (
	spoolFile = "spool.jsonl"
	stateFile = "state.jsonl"
	lockFile  = "drain.lock"
)

// Row is one enqueued message. UpdateTS empty means a new post (threaded when ThreadTS
// is set); UpdateTS set means a chat.update of that card. CardKey groups update rows for
// coalescing (defaulted to channel+update_ts at enqueue when empty).
type Row struct {
	Nonce      string `json:"nonce"`
	Channel    string `json:"channel"`
	Text       string `json:"text"`
	Blocks     []any  `json:"blocks,omitempty"`
	ThreadTS   string `json:"thread_ts,omitempty"`
	UpdateTS   string `json:"update_ts,omitempty"`
	CardKey    string `json:"card_key,omitempty"`
	Source     string `json:"source,omitempty"`      // producing surface, for status/dead reporting
	EnqueuedAt string `json:"enqueued_at,omitempty"` // RFC3339 UTC
}

// Row states. Absent = pending. sending is the pre-send intent marker that closes the
// crash window (see Drain); failed keeps a row pending with an attempt count; posted,
// refused, and superseded are terminal forever; dead is terminal until an operator
// Retry re-arms it.
const (
	statePending    = ""
	stateSending    = "sending"
	stateFailed     = "failed"
	statePosted     = "posted"
	stateDead       = "dead"
	stateRefused    = "refused"
	stateSuperseded = "superseded"
	stateRetry      = "retry"      // operator re-arm transition (dead -> pending)
	stateDrainPass  = "drain_pass" // heartbeat transition (Nonce == "")
)

// transition is one state-file row: nonce N moved to State at At. Attempts and
// Ambiguous ride on failed/sending transitions; TS rides on posted.
type transition struct {
	Nonce     string `json:"nonce"`
	State     string `json:"state"`
	TS        string `json:"ts,omitempty"`     // posted message ts (posts) or card ts (updates)
	Reason    string `json:"reason,omitempty"` // failure/refusal reason
	Attempts  int    `json:"attempts,omitempty"`
	Ambiguous bool   `json:"ambiguous,omitempty"` // the attempt may have half-succeeded (transport error after send)
	At        string `json:"at,omitempty"`        // RFC3339 UTC
}

// rowState is the folded effective state of one nonce after replaying its transitions.
type rowState struct {
	State     string
	TS        string
	Reason    string
	Attempts  int
	Ambiguous bool
}

// terminal reports whether the state accepts no further sends. dead counts as terminal
// here because only an explicit Retry transition (not a drain) may move it.
func (s rowState) terminal() bool {
	switch s.State {
	case statePosted, stateDead, stateRefused, stateSuperseded:
		return true
	}
	return false
}

// apply folds one transition into the state, later rows winning — the replay order IS
// the file order, so the newest line decides.
func (s rowState) apply(t transition) rowState {
	switch t.State {
	case stateRetry:
		// Re-arm a dead row; a terminal-forever state (posted/refused/superseded) stays.
		if s.State == stateDead {
			return rowState{}
		}
		return s
	case stateFailed, stateSending:
		s.State = t.State
		s.Reason = t.Reason
		s.Attempts = t.Attempts
		s.Ambiguous = t.Ambiguous
		return s
	case statePosted, stateDead, stateRefused, stateSuperseded:
		s.State = t.State
		s.TS = t.TS
		s.Reason = t.Reason
		if t.Attempts > 0 {
			s.Attempts = t.Attempts
		}
		return s
	}
	return s // unknown transition kinds are ignored (forward compatibility)
}

// Outbox is one spool directory. It holds no open handles — every append opens
// O_APPEND, writes one line, syncs, closes — so any number of producer processes can
// enqueue concurrently while one drainer runs.
type Outbox struct {
	dir string
	now func() time.Time // injected in tests

	// appendStateSeam, when non-nil, replaces the state append — the test seam that
	// simulates dying between a successful post and its record (the crash window the
	// nonce probe exists to close).
	appendStateSeam func(transition) error
}

// Open ensures dir exists and returns an Outbox over it.
func Open(dir string) (*Outbox, error) {
	if dir == "" {
		return nil, fmt.Errorf("slackoutbox: empty spool dir")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Outbox{dir: dir, now: time.Now}, nil
}

// Dir returns the spool directory (for diagnostics).
func (o *Outbox) Dir() string { return o.dir }

// NewNonce returns a fresh 128-bit hex idempotency nonce.
func NewNonce() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failing is a broken host; fall back to a time-derived nonce
		// rather than refusing to enqueue a durable message.
		return fmt.Sprintf("t-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// Enqueue validates the row, stamps nonce/card-key/enqueued-at defaults, and appends it
// durably to the spool. It returns the row's nonce once the row is on disk — producers
// treat that return as "the message will be delivered or dead-lettered, never lost".
// No network happens here; the leak fence runs at drain (send) time.
func (o *Outbox) Enqueue(r Row) (string, error) {
	if r.Channel == "" {
		return "", fmt.Errorf("slackoutbox: enqueue: channel is required")
	}
	if r.Text == "" {
		return "", fmt.Errorf("slackoutbox: enqueue: text is required (Slack needs the notification fallback)")
	}
	if r.UpdateTS != "" && r.ThreadTS != "" {
		return "", fmt.Errorf("slackoutbox: enqueue: update_ts and thread_ts are mutually exclusive (an update edits an existing message)")
	}
	if r.Nonce == "" {
		r.Nonce = NewNonce()
	}
	if r.UpdateTS != "" && r.CardKey == "" {
		r.CardKey = r.Channel + "\x00" + r.UpdateTS
	}
	if r.EnqueuedAt == "" {
		r.EnqueuedAt = o.now().UTC().Format(time.RFC3339)
	}
	if err := appendJSONL(filepath.Join(o.dir, spoolFile), r); err != nil {
		return "", err
	}
	return r.Nonce, nil
}

// appendState records one transition durably.
func (o *Outbox) appendState(t transition) error {
	if t.At == "" {
		t.At = o.now().UTC().Format(time.RFC3339)
	}
	if o.appendStateSeam != nil {
		return o.appendStateSeam(t)
	}
	return appendJSONL(filepath.Join(o.dir, stateFile), t)
}

// appendJSONL appends v as one JSON line, fsyncing before close — the durability
// contract Enqueue's "return means it will not be lost" rests on. Opened O_APPEND per
// call (no held handle) so concurrent producers append whole lines, the
// gatewayusageledger/cachevalueledger writer idiom.
func appendJSONL(path string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(append(b, '\n')); err != nil {
		return err
	}
	return f.Sync()
}

// Snapshot is the folded view of the outbox: every spool row plus its effective state,
// in spool (enqueue) order.
type Snapshot struct {
	Rows    []Row
	States  map[string]rowState // by nonce; missing key = pending with zero attempts
	Corrupt int                 // unparseable lines across both files (counted, never fatal)

	LastDrainAt time.Time // zero when no drain_pass heartbeat exists yet
}

// state returns the folded state for a nonce (zero value = pending).
func (s *Snapshot) state(nonce string) rowState { return s.States[nonce] }

// Load replays spool + state from disk. A malformed line increments Corrupt and is
// skipped — one corrupt row must not wedge the whole outbox.
func (o *Outbox) Load() (*Snapshot, error) {
	snap := &Snapshot{States: map[string]rowState{}}
	seen := map[string]bool{}

	err := readJSONL(filepath.Join(o.dir, spoolFile), func(line []byte) {
		var r Row
		if json.Unmarshal(line, &r) != nil || r.Nonce == "" || r.Channel == "" {
			snap.Corrupt++
			return
		}
		if seen[r.Nonce] {
			// A duplicate nonce in the spool would double-send under one identity;
			// first write wins, the duplicate is counted as corrupt.
			snap.Corrupt++
			return
		}
		seen[r.Nonce] = true
		snap.Rows = append(snap.Rows, r)
	})
	if err != nil {
		return nil, err
	}

	err = readJSONL(filepath.Join(o.dir, stateFile), func(line []byte) {
		var t transition
		if json.Unmarshal(line, &t) != nil || t.State == "" {
			snap.Corrupt++
			return
		}
		if t.State == stateDrainPass {
			if at, err := time.Parse(time.RFC3339, t.At); err == nil && at.After(snap.LastDrainAt) {
				snap.LastDrainAt = at
			}
			return
		}
		if t.Nonce == "" {
			snap.Corrupt++
			return
		}
		snap.States[t.Nonce] = snap.States[t.Nonce].apply(t)
	})
	if err != nil {
		return nil, err
	}
	return snap, nil
}

// readJSONL streams a JSONL file line by line; a missing file is an empty file.
func readJSONL(path string, fn func(line []byte)) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024) // a blocks payload can be large
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		fn(line)
	}
	return sc.Err()
}

// DeadRow is one dead-lettered message, the health/status view of a delivery failure.
type DeadRow struct {
	Nonce      string `json:"nonce"`
	Channel    string `json:"channel"`
	Source     string `json:"source,omitempty"`
	Reason     string `json:"reason"`
	Attempts   int    `json:"attempts"`
	EnqueuedAt string `json:"enqueued_at,omitempty"`
}

// Status is the fold `fak slack outbox status` and the health rung read. Ages are in
// seconds so the JSON is arithmetic-free for the watchdog; -1 means "not applicable".
type Status struct {
	Pending           int       `json:"pending"`
	Posted            int       `json:"posted"`
	Dead              int       `json:"dead"`
	Refused           int       `json:"refused"`
	Superseded        int       `json:"superseded"`
	Corrupt           int       `json:"corrupt"`
	OldestPendingAgeS int64     `json:"oldest_pending_age_s"` // -1 when nothing is pending
	LastDrainAgeS     int64     `json:"last_drain_age_s"`     // -1 when no drain has ever run
	DeadRows          []DeadRow `json:"dead_rows,omitempty"`
}

// Status folds the snapshot into counts + ages at `now`.
func (o *Outbox) Status(now time.Time) (*Status, error) {
	snap, err := o.Load()
	if err != nil {
		return nil, err
	}
	st := &Status{Corrupt: snap.Corrupt, OldestPendingAgeS: -1, LastDrainAgeS: -1}
	for _, r := range snap.Rows {
		rs := snap.state(r.Nonce)
		switch rs.State {
		case statePosted:
			st.Posted++
		case stateDead:
			st.Dead++
			st.DeadRows = append(st.DeadRows, DeadRow{
				Nonce: r.Nonce, Channel: r.Channel, Source: r.Source,
				Reason: rs.Reason, Attempts: rs.Attempts, EnqueuedAt: r.EnqueuedAt,
			})
		case stateRefused:
			st.Refused++
		case stateSuperseded:
			st.Superseded++
		default: // pending / sending / failed — all still owed a delivery
			st.Pending++
			if at, err := time.Parse(time.RFC3339, r.EnqueuedAt); err == nil {
				if age := int64(now.Sub(at) / time.Second); st.OldestPendingAgeS < age {
					st.OldestPendingAgeS = age
				}
			}
		}
	}
	if !snap.LastDrainAt.IsZero() {
		st.LastDrainAgeS = int64(now.Sub(snap.LastDrainAt) / time.Second)
	}
	return st, nil
}

// Dead lists the dead-lettered rows (the `fak slack outbox dead` fold).
func (o *Outbox) Dead() ([]DeadRow, error) {
	st, err := o.Status(o.now())
	if err != nil {
		return nil, err
	}
	return st.DeadRows, nil
}

// Retry re-arms dead rows: the given nonce, or every dead row when nonce is "". It
// returns the nonces re-armed. Posted/refused/superseded rows are never resurrected —
// a refused body must be re-authored, a posted one is done.
func (o *Outbox) Retry(nonce string) ([]string, error) {
	snap, err := o.Load()
	if err != nil {
		return nil, err
	}
	var armed []string
	for _, r := range snap.Rows {
		if nonce != "" && r.Nonce != nonce {
			continue
		}
		if snap.state(r.Nonce).State != stateDead {
			continue
		}
		if err := o.appendState(transition{Nonce: r.Nonce, State: stateRetry}); err != nil {
			return armed, err
		}
		armed = append(armed, r.Nonce)
	}
	if nonce != "" && len(armed) == 0 {
		return nil, fmt.Errorf("slackoutbox: retry: nonce %s is not a dead row", nonce)
	}
	return armed, nil
}
