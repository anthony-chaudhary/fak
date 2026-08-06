// Package gitbroker is rung 3 of epic #5619 (#5622): a resident, per-repo git
// query broker that separate short-lived client processes share over one
// Unix-domain socket.
//
// WHY A BROKER AND NOT JUST A POOL. Rung 2 (#5621) batches git reads WITHIN one
// process. The churn this fleet actually generates comes from MANY short-lived
// ones — eight resident sessions firing per-turn hooks, plus N python
// dispatchers on timers — and an in-process pool shares nothing across them. A
// resident broker is the only place a warm git backend can be shared BETWEEN
// separate client processes. AF_UNIX is the transport because it already works
// on this Windows fleet: cmd/fak/guard_lifecycle_ipc.go net.Listen("unix", …)s
// in production, and Go supports AF_UNIX on Windows 10+.
//
// CLASS A ONLY. This rung caches exactly one thing: content-addressed reads,
// keyed by a full object ID. Git objects are immutable, so a Class A entry can
// never go stale and this cache needs no invalidation at all — which is exactly
// why it is the first thing to land. Working-tree state (Class B/C) is mutable,
// needs real invalidation, and is deliberately NOT cached here (#5623). IsOID is
// the gate: a key that is not a full OID is answered live, every time.
//
// FAIL-OPEN IS THE WHOLE SAFETY ARGUMENT. #4603 (`core.fsmonitor=true` pointed
// at a dead daemon) is the standing lesson that a resident thing which dies must
// not take the fleet down with it. Every Client call is deadlined, and any
// failure — no broker, a wedged broker, a bad token, a truncated reply — falls
// back to spawning git in the caller through the SAME Runner the broker itself
// uses. A broker that is down changes latency and provenance; it never changes
// bytes.
//
// PROVENANCE ON EVERY ANSWER. Per the internal/stallscan lesson, "the broker is
// down" must never be spelled the same way as "the tree is clean". Every Result
// carries Broker, Cache, or FallbackSpawn, and no path can produce a Result
// without one — an answer whose provenance is missing or unrecognized is treated
// as a broker failure and re-derived by spawning.
package gitbroker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Provenance names where an answer came from. It is mandatory on every response
// and every Result: the operator-visible difference between "a warm broker
// answered", "the broker's immutable-object cache answered", and "no broker was
// usable, so the caller spawned git exactly like it did before this package
// existed".
type Provenance string

const (
	// Broker — the resident broker answered from its live git backend.
	Broker Provenance = "broker"
	// Cache — the resident broker answered from its content-addressed cache
	// without touching git at all. Only ever produced for a full-OID key.
	Cache Provenance = "cache"
	// FallbackSpawn — no usable broker, so the client spawned git itself. This
	// is the pre-broker path, byte-for-byte; only latency differs.
	FallbackSpawn Provenance = "fallback-spawn"
)

// Valid reports whether p is one of the three known tags. A response carrying
// anything else is not trusted — see the package doc's provenance rule.
func (p Provenance) Valid() bool {
	return p == Broker || p == Cache || p == FallbackSpawn
}

// Object is one git object exactly as `git cat-file --batch` reports it: the
// resolved OID, the type, the payload size git declared, and the payload bytes.
type Object struct {
	OID  string `json:"oid"`
	Type string `json:"type"`
	Size int64  `json:"size"`
	Data []byte `json:"data,omitempty"`
}

// Result is an Object plus the provenance of the answer that produced it.
type Result struct {
	Object
	Provenance Provenance `json:"provenance"`
}

// ErrMissingObject is returned when git reports the key as missing. It is a real
// answer about the repository, not a broker failure.
var ErrMissingObject = errors.New("gitbroker: object not found")

// Runner reads one git object out of a repository.
//
// It is the seam the rung-2 warm `cat-file --batch` pool (#5621) plugs into: the
// broker multiplexes every client onto whichever Runner it was built with, and
// the client uses the same interface for its spawn fallback. Today the only
// implementation is SpawnRunner, which is precisely the pre-broker path — that
// identity is what makes the fail-open guarantee byte-exact rather than
// aspirational.
type Runner interface {
	Object(ctx context.Context, rev string) (Object, error)
}

// SpawnRunner answers each query with its own `git cat-file --batch` process
// against RepoRoot. One spawn yields type, size, and payload together, so this
// is the cheapest honest single-object read git offers.
type SpawnRunner struct{ RepoRoot string }

// Object implements Runner by spawning git. rev may be a full OID or any
// revision expression git accepts; resolution is git's job, not ours.
func (r SpawnRunner) Object(ctx context.Context, rev string) (Object, error) {
	rev = strings.TrimSpace(rev)
	if rev == "" {
		return Object{}, errors.New("gitbroker: empty object key")
	}
	// A newline in the key would inject a second batch record; refuse rather
	// than silently answer a different question.
	if strings.ContainsAny(rev, "\r\n") {
		return Object{}, fmt.Errorf("gitbroker: invalid object key %q", rev)
	}
	cmd := exec.CommandContext(ctx, "git", "-C", r.RepoRoot, "cat-file", "--batch")
	// The whole point of this package is that a spawn happens on EVERY fallback
	// read, under background automation, on a Windows fleet. Without the
	// no-window hook each one flashes a console; the DESKTOP_POPUP_REGRESSION
	// gate refuses a push that reintroduces that.
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdin = strings.NewReader(rev + "\n")
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return Object{}, fmt.Errorf("gitbroker: git cat-file --batch: %w: %s", err, strings.TrimSpace(errb.String()))
	}
	return parseBatchRecord(out.Bytes())
}

// parseBatchRecord decodes one `git cat-file --batch` record:
//
//	<oid> SP <type> SP <size> LF <payload> LF
//
// A two-field header (`<key> SP missing`, or `ambiguous`) is git telling us the
// key did not resolve, which is ErrMissingObject and not a transport failure.
func parseBatchRecord(b []byte) (Object, error) {
	nl := bytes.IndexByte(b, '\n')
	if nl < 0 {
		return Object{}, errors.New("gitbroker: truncated cat-file record (no header)")
	}
	header := string(b[:nl])
	fields := strings.Fields(header)
	if len(fields) == 2 {
		return Object{}, fmt.Errorf("%w: %s", ErrMissingObject, header)
	}
	if len(fields) != 3 {
		return Object{}, fmt.Errorf("gitbroker: unparseable cat-file header %q", header)
	}
	size, err := strconv.ParseInt(fields[2], 10, 64)
	if err != nil || size < 0 {
		return Object{}, fmt.Errorf("gitbroker: unparseable cat-file size in %q", header)
	}
	payload := b[nl+1:]
	if int64(len(payload)) < size {
		return Object{}, fmt.Errorf("gitbroker: truncated cat-file payload (%d of %d bytes)", len(payload), size)
	}
	// Copy: payload aliases the whole command buffer, and the Object outlives it
	// in the cache.
	data := make([]byte, size)
	copy(data, payload[:size])
	return Object{OID: fields[0], Type: fields[1], Size: size, Data: data}, nil
}

// IsOID reports whether key is a full, unabbreviated object ID — the ONLY key
// shape this rung will cache.
//
// This is the Class A gate in one function. A full OID names immutable content,
// so an entry keyed by it can never need invalidation. An abbreviated OID can
// become ambiguous as the repo grows, and a revision expression (`HEAD`,
// `main:go.mod`, `@{u}`) names something mutable — neither is content-addressed,
// so neither is cached here. Both SHA-1 (40) and SHA-256 (64) hex are accepted;
// git spells OIDs in lowercase hex, and so must a cache key, or the same object
// would occupy two entries.
func IsOID(key string) bool {
	if len(key) != 40 && len(key) != 64 {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// Rendezvous is the per-repo socket + token-file pair a broker publishes and a
// client discovers. Both halves compute it from the repo root ALONE, which is
// what makes a live broker findable with no env hand-wiring: a client that knows
// which repo it is asking about already knows where to knock.
type Rendezvous struct {
	Socket string
	Token  string
}

// RendezvousIn computes the pair under dir (empty = the OS temp dir).
//
// The name is a hash of the canonical repo root rather than the path itself for
// two reasons: an AF_UNIX path is length-bounded, and a repo path contains
// separators and (on Windows) a drive letter. Canonicalization folds case on
// Windows so C:\work\fak and c:\work\fak are one broker, not two.
func RendezvousIn(dir, repoRoot string) Rendezvous {
	if dir == "" {
		dir = os.TempDir()
	}
	sum := sha256.Sum256([]byte(canonicalRoot(repoRoot)))
	stem := filepath.Join(dir, "fak-gitd-"+hex.EncodeToString(sum[:])[:16])
	return Rendezvous{Socket: stem + ".sock", Token: stem + ".token"}
}

// RendezvousFor is RendezvousIn under the default directory — the production
// spelling both `fak gitd` and its clients use.
func RendezvousFor(repoRoot string) Rendezvous { return RendezvousIn("", repoRoot) }

func canonicalRoot(repoRoot string) string {
	abs, err := filepath.Abs(repoRoot)
	if err != nil {
		abs = repoRoot
	}
	abs = filepath.Clean(abs)
	if runtime.GOOS == "windows" {
		abs = strings.ToLower(abs)
	}
	return abs
}

// Stats is the operator-facing counter set: how many queries the broker served,
// how they split between the live backend and the immutable-object cache, and
// how much the cache is holding. `fak gitd --status` reads it over the same
// socket, so "is the broker actually saving spawns?" is answerable from
// evidence.
//
// THE THREE SAVINGS ARE COUNTED SEPARATELY ON PURPOSE (#5623). Hits is the Class
// A object cache, TreeHits is the Class B working-tree cache, and Coalesced is
// single-flight — three different mechanisms with three different correctness
// arguments. Bundled into one "saved" number, neither contribution would be
// attributable and a later regression could not be localized to the mechanism
// that caused it.
type Stats struct {
	Served    int64 `json:"served"`
	Hits      int64 `json:"cache_hits"`
	Misses    int64 `json:"cache_misses"`
	Uncached  int64 `json:"uncacheable"`
	Entries   int   `json:"entries"`
	CacheSize int64 `json:"cache_bytes"`

	// Coalesced counts queries answered by JOINING an execution already in
	// flight — the single-flight saving, with no cache involved at all.
	Coalesced int64 `json:"coalesced"`

	// The Class B working-tree cache, counted apart from Class A.
	TreeHits   int64 `json:"tree_cache_hits"`
	TreeMisses int64 `json:"tree_cache_misses"`
	// TreeFresh counts working-tree queries answered by a fresh execution that
	// was never eligible for the cache: a decisional (Class C) caller, or a tree
	// too recently written to be keyed safely.
	TreeFresh int64 `json:"tree_fresh"`
	TreeEntry bool  `json:"tree_entry"`
}

// cache is the Class A store: OID -> Object, bounded by total payload bytes.
// There is no invalidation path and there must never be one — the day this
// caches something mutable is the day it needs one, and that is #5623's problem
// in #5623's package.
type cache struct {
	mu      sync.Mutex
	m       map[string]Object
	bytes   int64
	limit   int64
	maxItem int64
}

func newCache(limit int64) *cache {
	if limit <= 0 {
		limit = DefaultCacheBytes
	}
	return &cache{m: map[string]Object{}, limit: limit, maxItem: limit / 8}
}

func (c *cache) get(oid string) (Object, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	o, ok := c.m[oid]
	return o, ok
}

// put stores obj under oid unless it alone would dominate the budget, evicting
// other entries until the total fits. Eviction order is unspecified (Go map
// order); the contract this cache owes is boundedness and correctness, and
// because every entry is immutable content, evicting the "wrong" one costs one
// re-read and can never cost correctness.
func (c *cache) put(oid string, obj Object) {
	if int64(len(obj.Data)) > c.maxItem {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, dup := c.m[oid]; dup {
		return
	}
	c.m[oid] = obj
	c.bytes += int64(len(obj.Data))
	for c.bytes > c.limit {
		evicted := false
		for k, v := range c.m {
			if k == oid {
				continue
			}
			delete(c.m, k)
			c.bytes -= int64(len(v.Data))
			evicted = true
			break
		}
		if !evicted {
			return // only the new entry is left; it is under maxItem by construction
		}
	}
}

func (c *cache) sizes() (entries int, bytes int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.m), c.bytes
}
