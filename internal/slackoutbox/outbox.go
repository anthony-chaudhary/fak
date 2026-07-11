package slackoutbox

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/randhex"
)

// Spool/state file names inside the outbox directory. Both are append-only JSONL: the
// spool holds message rows (what to send), the state file holds transitions (what
// happened to each nonce). Replaying state over spool yields the effective queue, so a
// restart resumes exactly where the last process stopped.
//
// Each stream is a two-segment log: a live HEAD that lock-free producers append to
// (spool.jsonl / state.jsonl) and a compacted ARCHIVE that only Compact rewrites
// (spool.arch.jsonl / state.arch.jsonl). Compaction seals the head to a transient SEAL
// segment (spool.seal.jsonl / state.seal.jsonl) so it can be read without racing an
// appender; a crash may leave a seal behind, which Load simply folds as an older segment.
// Load replays archive → seal → head (oldest → newest) so the transition fold's
// "last line wins" and the spool's "first write per nonce wins" both still hold.
const (
	spoolFile     = "spool.jsonl"
	spoolSealFile = "spool.seal.jsonl"
	spoolArchFile = "spool.arch.jsonl"
	stateFile     = "state.jsonl"
	stateSealFile = "state.seal.jsonl"
	stateArchFile = "state.arch.jsonl"
	lockFile      = "drain.lock"
)

// spoolLayers / stateLayers are the segments Load folds, oldest → newest.
var (
	spoolLayers = []string{spoolArchFile, spoolSealFile, spoolFile}
	stateLayers = []string{stateArchFile, stateSealFile, stateFile}
)

// Row is one enqueued message. UpdateTS empty means a new post (threaded when ThreadTS
// is set); UpdateTS set means a chat.update of that card. CardKey groups update rows for
// coalescing (defaulted to channel+update_ts at enqueue when empty). ParentNonce threads
// a reply to another row in THIS outbox: it is resolved to the parent's posted ts at
// drain, so a root and its replies can be enqueued together before the root has a ts.
type Row struct {
	Nonce       string `json:"nonce"`
	Channel     string `json:"channel"`
	Text        string `json:"text"`
	Blocks      []any  `json:"blocks,omitempty"`
	ThreadTS    string `json:"thread_ts,omitempty"`
	UpdateTS    string `json:"update_ts,omitempty"`
	ParentNonce string `json:"parent_nonce,omitempty"` // deferred thread parent, resolved to its posted ts at drain
	CardKey     string `json:"card_key,omitempty"`
	Source      string `json:"source,omitempty"`         // producing surface, for status/dead reporting
	EnqueuedAt  string `json:"enqueued_at,omitempty"`    // RFC3339 UTC
	DeleteAfterS int   `json:"delete_after_s,omitempty"` // >0: reap this message this many seconds after its last activity (per-row ephemeral TTL; 0 => channel/opts default, see reap.go)
}

// Row states. Absent = pending. sending is the pre-send intent marker that closes the
// crash window (see Drain); failed keeps a row pending with an attempt count; posted,
// refused, and superseded are terminal forever; dead is terminal until an operator
// Retry re-arms it.
const (
	statePending     = ""
	stateSending     = "sending"
	stateFailed      = "failed"
	statePosted      = "posted"
	stateDead        = "dead"
	stateRefused     = "refused"
	stateSuperseded  = "superseded"
	stateUnchanged   = "unchanged"    // no-op update (identical to the card's last posted body), suppressed pre-send
	stateReaped      = "reaped"       // the posted message was chat.delete'd by the ephemeral reaper (terminal forever; see reap.go)
	stateRetry       = "retry"        // operator re-arm transition (dead -> pending)
	stateDrainPass   = "drain_pass"   // heartbeat transition (Nonce == "")
	stateCompactPass = "compact_pass" // heartbeat transition marking the last Compact (Nonce == "")
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
	Hash      string `json:"hash,omitempty"`      // payload hash on a posted UPDATE — feeds no-op suppression
}

// rowState is the folded effective state of one nonce after replaying its transitions.
// At is the timestamp of the winning (latest) transition — zero when unset or unparseable;
// compaction reads it to judge how long a row has been terminal.
type rowState struct {
	State     string
	TS        string
	Reason    string
	Attempts  int
	Ambiguous bool
	At        time.Time
	Hash      string // payload hash carried on a posted UPDATE (see transition.Hash)
}

// terminal reports whether the state accepts no further sends. dead counts as terminal
// here because only an explicit Retry transition (not a drain) may move it.
func (s rowState) terminal() bool {
	switch s.State {
	case statePosted, stateDead, stateRefused, stateSuperseded, stateUnchanged, stateReaped:
		return true
	}
	return false
}

// apply folds one transition into the state, later rows winning — the replay order IS
// the file order, so the newest line decides.
func (s rowState) apply(t transition) rowState {
	// The transition's own timestamp; zero on empty/malformed (only compaction reads it).
	at, _ := time.Parse(time.RFC3339, t.At)
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
		s.At = at
		return s
	case statePosted, stateDead, stateRefused, stateSuperseded, stateUnchanged, stateReaped:
		s.State = t.State
		s.TS = t.TS
		s.Reason = t.Reason
		if t.Attempts > 0 {
			s.Attempts = t.Attempts
		}
		s.At = at
		s.Hash = t.Hash // non-empty only on a posted UPDATE; feeds Snapshot.CardHashes
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

	// compactSleep, when non-nil, replaces the compaction seal-quiesce wait so tests
	// witness compaction without real sleeps.
	compactSleep func(time.Duration)
}

// Open ensures dir exists and returns an Outbox over it.
func Open(dir string) (*Outbox, error) {
	o := &Outbox{dir: dir, now: time.Now}
	if err := o.prepare(); err != nil {
		return nil, err
	}
	return o, nil
}

func (o *Outbox) prepare() error {
	if o.dir == "" {
		return fmt.Errorf("slackoutbox: empty spool dir")
	}
	return pathutil.EnsureDir(o.dir)
}

// Dir returns the spool directory (for diagnostics).
func (o *Outbox) Dir() string { return o.dir }

// NewNonce returns a fresh 128-bit hex idempotency nonce.
func NewNonce() string {
	if nonce, ok := randhex.String(16); ok {
		return nonce
	}
	// crypto/rand failing is a broken host; fall back to a time-derived nonce
	// rather than refusing to enqueue a durable message.
	return fmt.Sprintf("t-%d", time.Now().UnixNano())
}

// payloadHash is a stable fingerprint of a row's VISIBLE payload — the fallback text plus
// any block payload. Two rows with the same hash render identically in Slack, so an update
// whose hash equals its card's last posted body is a no-op the drain can drop without a
// chat.update. Content-only by design: nonce, ts, and enqueue time are excluded so the same
// card content always hashes the same across the fresh nonce each tick enqueues.
func payloadHash(r Row) string {
	h := sha256.New()
	h.Write([]byte(r.Text))
	h.Write([]byte{0}) // domain separator between text and blocks
	if len(r.Blocks) > 0 {
		if b, err := json.Marshal(r.Blocks); err == nil {
			h.Write(b)
		}
	}
	return hex.EncodeToString(h.Sum(nil)[:16]) // 128-bit prefix — collisions are not a practical risk
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
	if r.ParentNonce != "" && r.UpdateTS != "" {
		return "", fmt.Errorf("slackoutbox: enqueue: parent_nonce and update_ts are mutually exclusive (a reply is a new post, not an edit)")
	}
	if r.ParentNonce != "" && r.ThreadTS != "" {
		return "", fmt.Errorf("slackoutbox: enqueue: parent_nonce and thread_ts are mutually exclusive (choose a deferred parent nonce or a literal thread_ts)")
	}
	if r.Nonce == "" {
		r.Nonce = NewNonce()
	}
	if r.ParentNonce != "" && r.ParentNonce == r.Nonce {
		return "", fmt.Errorf("slackoutbox: enqueue: parent_nonce cannot equal the row's own nonce")
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

	// CardHashes is the payload hash of the LATEST posted update per CardKey — the read the
	// drain uses to suppress a no-op edit (an update whose body equals the card's last posted
	// body). Empty for any card that has never had a hash-carrying posted update.
	CardHashes map[string]string

	LastDrainAt   time.Time // zero when no drain_pass heartbeat exists yet
	LastCompactAt time.Time // zero when Compact has never run (no compact_pass heartbeat)

	drainPasses int // drain_pass heartbeat lines folded (Compact reports how many it collapsed)
}

// state returns the folded state for a nonce (zero value = pending).
func (s *Snapshot) state(nonce string) rowState { return s.States[nonce] }

// PostedTS returns the Slack message ts a nonce posted at, or "" if it has not posted yet
// (or reached a non-posted terminal state). It is the read a deferred producer uses to
// thread or edit a row it enqueued earlier without holding the ts in memory.
func (s *Snapshot) PostedTS(nonce string) string {
	st := s.States[nonce]
	if st.State != statePosted {
		return ""
	}
	return st.TS
}

// Load replays spool + state from disk. A malformed line increments Corrupt and is
// skipped — one corrupt row must not wedge the whole outbox.
func (o *Outbox) Load() (*Snapshot, error) {
	return o.foldFiles(spoolLayers, stateLayers)
}

// foldFiles replays the given spool segments then the given state segments, oldest →
// newest, into one snapshot. Load folds all three layers; Compact folds archive+seal only
// (the segments it is about to rewrite), leaving the live head to keep accumulating.
func (o *Outbox) foldFiles(spoolNames, stateNames []string) (*Snapshot, error) {
	snap := &Snapshot{States: map[string]rowState{}, CardHashes: map[string]string{}}
	byNonce := map[string]Row{} // nonce -> row, to join a posted transition back to its CardKey

	// Spool: fold archive → seal → head. A repeated nonce WITHIN a segment is the
	// double-send signal (counted corrupt, first write wins); the same nonce ACROSS
	// segments is the benign overlap a leftover seal leaves after a crashed compaction,
	// so it is skipped without inflating Corrupt.
	seen := map[string]bool{}
	for _, name := range spoolNames {
		seenThis := map[string]bool{}
		err := readJSONL(filepath.Join(o.dir, name), func(line []byte) {
			var r Row
			if json.Unmarshal(line, &r) != nil || r.Nonce == "" || r.Channel == "" {
				snap.Corrupt++
				return
			}
			if seenThis[r.Nonce] {
				snap.Corrupt++
				return
			}
			seenThis[r.Nonce] = true
			if seen[r.Nonce] {
				return // cross-segment overlap (leftover seal) — already have this row
			}
			seen[r.Nonce] = true
			snap.Rows = append(snap.Rows, r)
			byNonce[r.Nonce] = r
		})
		if err != nil {
			return nil, err
		}
	}

	// State: replay archive → seal → head so the latest transition per nonce wins.
	for _, name := range stateNames {
		err := readJSONL(filepath.Join(o.dir, name), func(line []byte) {
			var t transition
			if json.Unmarshal(line, &t) != nil || t.State == "" {
				snap.Corrupt++
				return
			}
			switch t.State {
			case stateDrainPass:
				snap.drainPasses++
				if at, err := time.Parse(time.RFC3339, t.At); err == nil && at.After(snap.LastDrainAt) {
					snap.LastDrainAt = at
				}
				return
			case stateCompactPass:
				if at, err := time.Parse(time.RFC3339, t.At); err == nil && at.After(snap.LastCompactAt) {
					snap.LastCompactAt = at
				}
				return
			}
			if t.Nonce == "" {
				snap.Corrupt++
				return
			}
			snap.States[t.Nonce] = snap.States[t.Nonce].apply(t)
			// Track the latest posted-update body hash per card key (state file order =
			// newest last), so the drain can drop a byte-identical re-edit without a call.
			if t.State == statePosted && t.Hash != "" {
				if ck := byNonce[t.Nonce].CardKey; ck != "" {
					snap.CardHashes[ck] = t.Hash
				}
			}
		})
		if err != nil {
			return nil, err
		}
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
	Reaped            int       `json:"reaped,omitempty"`     // posted messages the ephemeral reaper deleted (see `outbox reap`)
	Suppressed        int       `json:"suppressed,omitempty"` // no-op update edits dropped pre-send (see `outbox calls`)
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
		case stateReaped:
			st.Reaped++ // posted then deleted by the ephemeral reaper — gone from the channel
		case stateUnchanged:
			st.Suppressed++ // terminal no-op edit — not owed a delivery, not a real post
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
