package gitbroker

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Defaults. DefaultCacheBytes bounds the Class A store; MaxServedBytes caps what
// the broker will ship over the socket at all (a client asking for something
// larger falls back to spawning, which streams it without a JSON envelope);
// defaultHandleTimeout only stops a single connection from leaking, since the
// deadline that actually protects a caller lives on the client side.
const (
	DefaultCacheBytes    int64 = 64 << 20
	MaxServedBytes       int64 = 32 << 20
	defaultHandleTimeout       = 10 * time.Second
	maxRequestBytes      int64 = 8 << 10
)

// Wire ops: read one content-addressed object, read working-tree state, and
// report counters. There is no write op in this package and there is not meant
// to be one.
const (
	opObject = "object"
	opStats  = "stats"
	opTree   = "tree"
)

type wireRequest struct {
	Token string `json:"token"`
	Op    string `json:"op"`
	Rev   string `json:"rev,omitempty"`
	// Class is what the CALLER declares the answer is for. Absent — an older
	// client, or a caller that did not think about it — decodes to "", which
	// Class.Decisional treats as Class C: fresh every time. The unsafe direction
	// is never the default.
	Class Class `json:"class,omitempty"`
}

// wireResponse always carries a Provenance on success. The field is not
// omitempty-optional by accident: a reply with no provenance is rejected by the
// client, so a future op cannot quietly ship an untagged answer.
type wireResponse struct {
	Provenance Provenance `json:"provenance,omitempty"`
	Object     *Object    `json:"object,omitempty"`
	Tree       *TreeState `json:"tree,omitempty"`
	Stats      *Stats     `json:"stats,omitempty"`
	Error      string     `json:"error,omitempty"`
}

// Config builds a Server. Runner is the backend every client is multiplexed onto
// — nil means SpawnRunner{RepoRoot}, and #5621's warm pool drops in here without
// touching anything else in this package.
type Config struct {
	RepoRoot   string
	Dir        string // rendezvous directory; empty = the OS temp dir
	Runner     Runner
	CacheBytes int64

	// Tree is the working-tree backend; nil means SpawnTreeRunner{RepoRoot}.
	Tree TreeRunner
	// TreeRaceWindow is the assumed filesystem mtime granularity that bounds the
	// Class B cache's stale-read budget; <=0 means DefaultTreeRaceWindow. See
	// the budget stated at the top of treestate.go.
	TreeRaceWindow time.Duration
}

// Server is the resident per-repo broker.
type Server struct {
	rv       Rendezvous
	ln       net.Listener
	repoRoot string
	runner   Runner
	tree     TreeRunner
	token    string
	cache    *cache
	trees    *treeCache
	timeout  time.Duration
	window   time.Duration

	// Single-flight, one group per query kind. These are what collapse the
	// concurrent fan-in from eight sessions; they store nothing between calls.
	objFlight  flightGroup[Object]
	treeFlight flightGroup[TreeState]

	served     atomic.Int64
	hits       atomic.Int64
	misses     atomic.Int64
	uncached   atomic.Int64
	treeHits   atomic.Int64
	treeMisses atomic.Int64
	treeFresh  atomic.Int64

	done chan struct{}
	wg   sync.WaitGroup
	once sync.Once
}

// Serve binds the per-repo socket, publishes the auth token beside it, and
// starts accepting. It returns as soon as the broker is reachable, so a caller
// that gets a nil error may immediately tell clients the broker is up.
func Serve(cfg Config) (*Server, error) {
	if cfg.RepoRoot == "" {
		return nil, errors.New("gitbroker: no repo root")
	}
	runner := cfg.Runner
	if runner == nil {
		runner = SpawnRunner{RepoRoot: cfg.RepoRoot}
	}
	tree := cfg.Tree
	if tree == nil {
		tree = SpawnTreeRunner{RepoRoot: cfg.RepoRoot}
	}
	window := cfg.TreeRaceWindow
	if window <= 0 {
		window = DefaultTreeRaceWindow
	}
	rv := RendezvousIn(cfg.Dir, cfg.RepoRoot)
	ln, err := listenFresh(rv.Socket)
	if err != nil {
		return nil, err
	}
	token, err := newToken()
	if err != nil {
		_ = ln.Close()
		_ = os.Remove(rv.Socket)
		return nil, err
	}
	// 0600: the token is the only thing between a local process and this
	// repo's object store, so it is readable by its owner alone.
	if err := os.WriteFile(rv.Token, []byte(token), 0o600); err != nil {
		_ = ln.Close()
		_ = os.Remove(rv.Socket)
		return nil, fmt.Errorf("gitbroker: publish token: %w", err)
	}
	_ = os.Chmod(rv.Socket, 0o600) // best-effort; AF_UNIX on Windows has no mode

	s := &Server{
		rv:       rv,
		ln:       ln,
		repoRoot: cfg.RepoRoot,
		runner:   runner,
		tree:     tree,
		token:    token,
		cache:    newCache(cfg.CacheBytes),
		trees:    &treeCache{},
		timeout:  defaultHandleTimeout,
		window:   window,
		done:     make(chan struct{}),
	}
	s.wg.Add(1)
	go s.accept()
	return s, nil
}

// listenFresh binds path, clearing a CORPSE socket file first. A crashed broker
// leaves its socket node behind and bind fails with EADDRINUSE forever after; a
// LIVE broker also fails to bind, and stealing that rendezvous would silently
// split the fleet across two brokers. The two are told apart by knocking: only a
// live one answers the dial.
func listenFresh(path string) (net.Listener, error) {
	ln, err := net.Listen("unix", path)
	if err == nil {
		return ln, nil
	}
	if c, derr := net.DialTimeout("unix", path, 200*time.Millisecond); derr == nil {
		_ = c.Close()
		return nil, fmt.Errorf("gitbroker: a broker is already serving %s", path)
	}
	if rmErr := os.Remove(path); rmErr != nil && !os.IsNotExist(rmErr) {
		return nil, fmt.Errorf("gitbroker: listen %s: %w", path, err)
	}
	ln, err2 := net.Listen("unix", path)
	if err2 != nil {
		return nil, fmt.Errorf("gitbroker: listen %s: %w", path, err2)
	}
	return ln, nil
}

func newToken() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("gitbroker: token: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

// Rendezvous reports where this broker is listening.
func (s *Server) Rendezvous() Rendezvous { return s.rv }

// Token is the shared secret a client must present. Clients normally read it
// from the published file; this accessor exists for in-process callers.
func (s *Server) Token() string { return s.token }

// Stats snapshots the operator counters.
func (s *Server) Stats() Stats {
	entries, bytes := s.cache.sizes()
	return Stats{
		Served:     s.served.Load(),
		Hits:       s.hits.Load(),
		Misses:     s.misses.Load(),
		Uncached:   s.uncached.Load(),
		Entries:    entries,
		CacheSize:  bytes,
		Coalesced:  s.objFlight.Coalesced() + s.treeFlight.Coalesced(),
		TreeHits:   s.treeHits.Load(),
		TreeMisses: s.treeMisses.Load(),
		TreeFresh:  s.treeFresh.Load(),
		TreeEntry:  s.trees.held(),
	}
}

// Close stops accepting, waits for in-flight connections, and removes the
// rendezvous so the next broker binds cleanly. It is safe to call twice.
func (s *Server) Close() error {
	s.once.Do(func() {
		close(s.done)
		_ = s.ln.Close()
		s.wg.Wait()
		_ = os.Remove(s.rv.Socket)
		_ = os.Remove(s.rv.Token)
	})
	return nil
}

func (s *Server) accept() {
	defer s.wg.Done()
	for {
		conn, err := s.ln.Accept()
		if err != nil {
			return
		}
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.handle(conn)
		}()
	}
}

// handle serves exactly one request per connection, mirroring the
// guard_lifecycle_ipc.go precedent. Connection-per-query is fine: the win this
// rung is after is the warm backend and the shared cache, not socket reuse.
func (s *Server) handle(conn net.Conn) {
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(s.timeout))
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()

	var req wireRequest
	if err := json.NewDecoder(io.LimitReader(conn, maxRequestBytes)).Decode(&req); err != nil {
		writeResponse(conn, wireResponse{Error: "malformed request"})
		return
	}
	if subtle.ConstantTimeCompare([]byte(req.Token), []byte(s.token)) != 1 {
		writeResponse(conn, wireResponse{Error: "unauthorized"})
		return
	}
	switch req.Op {
	case opStats:
		st := s.Stats()
		writeResponse(conn, wireResponse{Provenance: Broker, Stats: &st})
	case opObject:
		obj, prov, err := s.object(ctx, req.Rev)
		if err != nil {
			writeResponse(conn, wireResponse{Error: err.Error()})
			return
		}
		if obj.Size > MaxServedBytes {
			// Too big for the JSON envelope. Say so plainly; the client
			// re-derives it by spawning, which is the same bytes.
			writeResponse(conn, wireResponse{Error: fmt.Sprintf("object %s exceeds the broker envelope (%d bytes)", obj.OID, obj.Size)})
			return
		}
		s.served.Add(1)
		writeResponse(conn, wireResponse{Provenance: prov, Object: &obj})
	case opTree:
		st, prov, err := s.treeState(ctx, req.Class)
		if err != nil {
			writeResponse(conn, wireResponse{Error: err.Error()})
			return
		}
		s.served.Add(1)
		writeResponse(conn, wireResponse{Provenance: prov, Tree: &st})
	default:
		writeResponse(conn, wireResponse{Error: fmt.Sprintf("unknown op %q", req.Op)})
	}
}

// object is the Class A decision point, and the only place the object cache is
// consulted. A full-OID key may be answered from the immutable-object cache;
// anything else goes to the backend every single time and is never stored.
//
// Every backend read — cached-key or not — goes through single-flight, so N
// concurrent callers asking the same question cost one git invocation. That is
// safe for BOTH branches here without any invalidation argument: a full OID is
// immutable, and a non-OID read is still computed fresh, just once.
func (s *Server) object(ctx context.Context, rev string) (Object, Provenance, error) {
	if !IsOID(rev) {
		s.uncached.Add(1)
		obj, _, err := s.objFlight.Do("obj\x00"+rev, func() (Object, error) {
			return s.runner.Object(ctx, rev)
		})
		if err != nil {
			return Object{}, "", err
		}
		return obj, Broker, nil
	}
	if obj, ok := s.cache.get(rev); ok {
		s.hits.Add(1)
		return obj, Cache, nil
	}
	s.misses.Add(1)
	obj, _, err := s.objFlight.Do("obj\x00"+rev, func() (Object, error) {
		o, err := s.runner.Object(ctx, rev)
		if err == nil {
			s.cache.put(rev, o)
		}
		return o, err
	})
	if err != nil {
		return Object{}, "", err
	}
	return obj, Broker, nil
}

// treeState is the Class B/C decision point, and the ONLY function in this
// package permitted to touch the working-tree cache. Read it as three gates in
// order:
//
//  1. DECISIONAL: a caller whose answer feeds a commit gate, a mutation, or a
//     refusal gets a fresh execution — no cache read, no cache write, and no
//     joining someone else's in-flight query either, so its snapshot is never
//     older than its own call. This is the correctness line of the whole epic.
//  2. UNKEYABLE / UNSETTLED: if the repository cannot be keyed, or the tree was
//     written so recently that filesystem mtime granularity could hide a peer's
//     write behind our sample, the answer is computed fresh and not stored.
//  3. KEYED: sample the key, serve on an exact match, otherwise compute and
//     store — but only if the key is UNCHANGED across the computation, so a
//     write that landed while `git status` was running can never be recorded as
//     the state of the tree after it.
func (s *Server) treeState(ctx context.Context, class Class) (TreeState, Provenance, error) {
	if class.Decisional() {
		s.treeFresh.Add(1)
		st, err := s.tree.TreeState(ctx)
		if err != nil {
			return TreeState{}, "", err
		}
		st.Key, _ = sampleStateKey(s.repoRoot, s.window)
		return st, Broker, nil
	}

	before, settled := sampleStateKey(s.repoRoot, s.window)
	if !settled {
		s.treeFresh.Add(1)
		st, _, err := s.treeFlight.Do("tree\x00fresh", func() (TreeState, error) {
			return s.tree.TreeState(ctx)
		})
		if err != nil {
			return TreeState{}, "", err
		}
		st.Key = before
		return st, Broker, nil
	}
	if st, ok := s.trees.lookup(before); ok {
		s.treeHits.Add(1)
		return st, Cache, nil
	}
	s.treeMisses.Add(1)
	st, _, err := s.treeFlight.Do("tree\x00"+treeFlightKey(before), func() (TreeState, error) {
		out, err := s.tree.TreeState(ctx)
		if err != nil {
			return TreeState{}, err
		}
		out.Key = before
		after, stillSettled := sampleStateKey(s.repoRoot, s.window)
		if stillSettled && after == before {
			s.trees.store(before, out)
		}
		return out, nil
	})
	if err != nil {
		return TreeState{}, "", err
	}
	return st, Broker, nil
}

// treeFlightKey renders a StateKey as a single-flight key. Two callers coalesce
// only when they are asking about the SAME tree state; a peer's write between
// them puts them in different flights rather than sharing one answer.
func treeFlightKey(k StateKey) string {
	return fmt.Sprintf("%d\x00%d\x00%s\x00%d", k.IndexMod, k.IndexSize, k.HeadOID, k.RefsMod)
}

func writeResponse(conn net.Conn, resp wireResponse) {
	_ = json.NewEncoder(conn).Encode(resp)
}
