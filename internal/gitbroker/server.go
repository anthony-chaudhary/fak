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

// Wire ops. Deliberately two: read one content-addressed object, and report
// counters. There is no write op in this rung and there is not meant to be one.
const (
	opObject = "object"
	opStats  = "stats"
)

type wireRequest struct {
	Token string `json:"token"`
	Op    string `json:"op"`
	Rev   string `json:"rev,omitempty"`
}

// wireResponse always carries a Provenance on success. The field is not
// omitempty-optional by accident: a reply with no provenance is rejected by the
// client, so a future op cannot quietly ship an untagged answer.
type wireResponse struct {
	Provenance Provenance `json:"provenance,omitempty"`
	Object     *Object    `json:"object,omitempty"`
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
}

// Server is the resident per-repo broker.
type Server struct {
	rv      Rendezvous
	ln      net.Listener
	runner  Runner
	token   string
	cache   *cache
	timeout time.Duration

	served   atomic.Int64
	hits     atomic.Int64
	misses   atomic.Int64
	uncached atomic.Int64

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
		rv:      rv,
		ln:      ln,
		runner:  runner,
		token:   token,
		cache:   newCache(cfg.CacheBytes),
		timeout: defaultHandleTimeout,
		done:    make(chan struct{}),
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
		Served:    s.served.Load(),
		Hits:      s.hits.Load(),
		Misses:    s.misses.Load(),
		Uncached:  s.uncached.Load(),
		Entries:   entries,
		CacheSize: bytes,
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
	default:
		writeResponse(conn, wireResponse{Error: fmt.Sprintf("unknown op %q", req.Op)})
	}
}

// object is the Class A decision point, and the only place the cache is
// consulted. A full-OID key may be answered from the immutable-object cache;
// anything else goes to the backend every single time and is never stored.
func (s *Server) object(ctx context.Context, rev string) (Object, Provenance, error) {
	if !IsOID(rev) {
		s.uncached.Add(1)
		obj, err := s.runner.Object(ctx, rev)
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
	obj, err := s.runner.Object(ctx, rev)
	if err != nil {
		return Object{}, "", err
	}
	s.cache.put(rev, obj)
	return obj, Broker, nil
}

func writeResponse(conn net.Conn, resp wireResponse) {
	_ = json.NewEncoder(conn).Encode(resp)
}
