package leasequeue

// store.go is the ONE I/O half of this leaf — the durable trace that stops a refusal from
// evaporating. Plan (leasequeue.go) stays pure; everything that touches a disk is here, the same
// leaf/shell split regionadmit uses for LoadTaxonomy.
//
// SUBSTRATE AND ITS HONEST BOUNDARY. A ticket is ONE small JSON file under
// <git-dir>/fak/leasequeue/<id>.json. One file per waiter is deliberate: two different waiters
// never touch the same path, so minting needs no lock and no read-modify-write. It lives beside
// the lease fabric's own home (internal/leaseref writes refs/fak/locks/* in the same git dir) but
// it is NOT a ref, so — unlike a lease — a ticket does NOT ride git fetch/push between clones.
// This is a SAME-MACHINE waiter plane. Cross-clone queueing would need the ref substrate and is
// deliberately not claimed here.
//
// A ticket is never reaped by a daemon: it carries a TTL and Ticket.Lapsed drops it on READ, so
// an abandoned waiter cannot reserve a region forever and no reaper process has to exist.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// DefaultTTLSeconds is how long a ticket survives without a re-attempt. It is the seatpark poll
// schedule's own budget with headroom: the documented backoff runs base 30s doubling to a 5m cap
// over 5 parks (see internal/seatpark), so a waiter still polling on schedule renews far inside
// this window, while one that stopped polling is gone within it.
const DefaultTTLSeconds int64 = 1800

// TicketID is the waiter's STABLE identity: a digest of (actor, lane, resolved tree). Stability is
// the entire mechanism — the same caller re-asking the same question refreshes ONE ticket instead
// of minting a new one, so its enqueue clock survives the retry and the order it earns is by
// arrival rather than by who polled first.
func TicketID(actor, lane string, tree []string) string {
	norm := make([]string, 0, len(tree))
	for _, g := range tree {
		g = strings.TrimSpace(strings.ReplaceAll(g, "\\", "/"))
		if g != "" {
			norm = append(norm, g)
		}
	}
	sort.Strings(norm)
	h := sha256.Sum256([]byte(strings.Join(append([]string{
		strings.TrimSpace(actor), strings.TrimSpace(lane),
	}, norm...), "\n")))
	return hex.EncodeToString(h[:8])
}

// QueueDir is the waiter-plane directory for a repo root: <git-dir>/fak/leasequeue. It resolves a
// worktree/submodule `.git` FILE (which holds a `gitdir: <path>` line) as well as an ordinary
// `.git` directory, so a worker worktree queues in its own git dir rather than silently nowhere.
// A root that is not a repo yields an error — the caller then simply has no queue, which is a
// degraded report, never a changed verdict.
func QueueDir(root string) (string, error) {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	dot := filepath.Join(root, ".git")
	info, err := os.Stat(dot)
	if err != nil {
		return "", fmt.Errorf("resolve git dir: %w", err)
	}
	if info.IsDir() {
		return filepath.Join(dot, "fak", "leasequeue"), nil
	}
	raw, err := os.ReadFile(dot)
	if err != nil {
		return "", fmt.Errorf("read git dir pointer: %w", err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "gitdir:"); ok {
			p := strings.TrimSpace(rest)
			if p == "" {
				break
			}
			if !filepath.IsAbs(p) {
				p = filepath.Join(root, p)
			}
			return filepath.Join(p, "fak", "leasequeue"), nil
		}
	}
	return "", fmt.Errorf("no gitdir pointer in %s", dot)
}

// Store is the durable waiter journal rooted at one queue directory.
type Store struct{ dir string }

// NewStore opens the journal at dir (created on first mint).
func NewStore(dir string) *Store { return &Store{dir: dir} }

// OpenStore opens the journal for a repo root.
func OpenStore(root string) (*Store, error) {
	dir, err := QueueDir(root)
	if err != nil {
		return nil, err
	}
	return NewStore(dir), nil
}

// Dir reports the journal directory, so a report can name where the queue is inspectable.
func (s *Store) Dir() string { return s.dir }

// path is the file one ticket lives at. The id is a hex digest from TicketID, so it is already a
// safe basename; anything else is rejected rather than sanitized, so a caller can never be talked
// into writing outside the journal.
func (s *Store) path(id string) (string, error) {
	if id == "" || strings.ContainsAny(id, `/\.:`) {
		return "", fmt.Errorf("invalid ticket id %q", id)
	}
	return filepath.Join(s.dir, id+".json"), nil
}

// Mint records the waiter's place in line, and is the whole point of this package: it is what a
// refusal does INSTEAD of evaporating.
//
// It is create-or-REFRESH. On a first refusal the ticket is created with EnqueuedUnix = now and
// Parks = 1. On every later refusal by the same waiter the existing ticket's EnqueuedUnix is
// PRESERVED — only RenewedUnix, LastParkUnix and Parks advance. That preservation is what turns
// a re-race into a queue: the waiter keeps the position it earned by waiting.
//
// A ticket found already LAPSED is treated as a fresh arrival (its holder stopped polling and
// gave up its place), so a stale file can never grant an unearned seniority.
func (s *Store) Mint(t Ticket, now time.Time) (Ticket, error) {
	if t.ID == "" {
		t.ID = TicketID(t.Actor, t.Lane, t.Tree)
	}
	p, err := s.path(t.ID)
	if err != nil {
		return Ticket{}, err
	}
	nowUnix := now.Unix()
	if t.TTLSeconds <= 0 {
		t.TTLSeconds = DefaultTTLSeconds
	}
	t.EnqueuedUnix = nowUnix
	t.Parks = 1
	if prev, err := readTicket(p); err == nil && !prev.Lapsed(nowUnix) {
		t.EnqueuedUnix = prev.EnqueuedUnix
		t.Parks = prev.Parks + 1
	}
	t.RenewedUnix = nowUnix
	t.LastParkUnix = nowUnix
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return Ticket{}, fmt.Errorf("create queue dir: %w", err)
	}
	blob, err := json.Marshal(t)
	if err != nil {
		return Ticket{}, err
	}
	if err := writeFileAtomic(p, blob); err != nil {
		return Ticket{}, err
	}
	return t, nil
}

// Drop removes a waiter's ticket — what a caller does once it finally holds the region, so it
// stops reserving a place it no longer needs. A missing ticket is not an error.
func (s *Store) Drop(id string) error {
	p, err := s.path(id)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Live reads every ticket that has not lapsed at now, in stable id order. A file that cannot be
// parsed is SKIPPED rather than failing the read: one corrupt ticket must not blind the whole
// queue, and the worst case of skipping it is that its owner re-mints on its next poll.
func (s *Store) Live(now time.Time) ([]Ticket, error) {
	ents, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // an empty queue is not an error
		}
		return nil, err
	}
	nowUnix := now.Unix()
	out := make([]Ticket, 0, len(ents))
	for _, ent := range ents {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".json") {
			continue
		}
		t, err := readTicket(filepath.Join(s.dir, ent.Name()))
		if err != nil || t.ID == "" || t.Lapsed(nowUnix) {
			continue
		}
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func readTicket(path string) (Ticket, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Ticket{}, err
	}
	var t Ticket
	if err := json.Unmarshal(raw, &t); err != nil {
		return Ticket{}, err
	}
	return t, nil
}

// writeFileAtomic writes via a temp file in the same directory and renames over the target, so a
// concurrent reader sees either the old ticket or the new one and never a half-written file.
func writeFileAtomic(path string, blob []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(blob); err != nil {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return err
	}
	return nil
}
