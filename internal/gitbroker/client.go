package gitbroker

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"strings"
	"time"
)

// DefaultTimeout is the per-call client deadline. A warm broker on a local
// socket answers in well under a millisecond, so this is not a performance knob
// — it is the fuse. #4603 (`core.fsmonitor=true` against a dead daemon) is the
// standing lesson: a resident helper that hangs must cost the caller a bounded
// wait and nothing else.
const DefaultTimeout = 500 * time.Millisecond

const maxResponseBytes int64 = 64 << 20 // MaxServedBytes, with room for base64 + envelope

// Client is the caller-side half. It is safe to construct per call — that is the
// point, since the callers this exists for are short-lived processes.
//
// The invariant every method upholds: the broker can make an answer FASTER, and
// it can change the Provenance tag. It can never change the bytes, and it can
// never make a call take longer than Timeout plus the cost of the spawn the
// caller would have paid anyway.
type Client struct {
	RepoRoot string
	Dir      string        // rendezvous directory; empty = the OS temp dir
	Timeout  time.Duration // per-call broker deadline; <=0 = DefaultTimeout
	Runner   Runner        // fallback backend; nil = SpawnRunner{RepoRoot}
	// TreeRunner is the working-tree fallback backend; nil =
	// SpawnTreeRunner{RepoRoot}. It is a separate seam from Runner because the two
	// answer different questions with different git invocations.
	TreeRunner TreeRunner
}

func (c *Client) runner() Runner {
	if c.Runner != nil {
		return c.Runner
	}
	return SpawnRunner{RepoRoot: c.RepoRoot}
}

func (c *Client) treeRunner() TreeRunner {
	if c.TreeRunner != nil {
		return c.TreeRunner
	}
	return SpawnTreeRunner{RepoRoot: c.RepoRoot}
}

func (c *Client) timeout() time.Duration {
	if c.Timeout > 0 {
		return c.Timeout
	}
	return DefaultTimeout
}

// Object answers one read, preferring the resident broker and falling back to
// spawning git.
//
// EVERY broker-side problem — no socket, no token, a bad token, a wedged broker,
// a truncated or untagged reply, an object too large for the envelope, even the
// broker reporting the object missing — routes to the same fallback. That is
// deliberate: routing only SOME failures back to the spawn path would leave a
// class of query whose answer depends on whether a background daemon happened to
// be healthy, which is exactly the coupling this rung promises not to introduce.
// The cost is one redundant spawn on a genuinely-missing object; the purchase is
// that a client's answer never depends on the broker at all.
func (c *Client) Object(ctx context.Context, rev string) (Result, error) {
	if res, ok := c.viaBroker(ctx, rev); ok {
		return res, nil
	}
	obj, err := c.runner().Object(ctx, rev)
	if err != nil {
		return Result{}, err
	}
	return Result{Object: obj, Provenance: FallbackSpawn}, nil
}

// Tree answers working-tree state — is the tree dirty, and the porcelain status
// that says why — preferring the resident broker and falling back to spawning
// `git status` exactly as the caller would have.
//
// THE CLASS IS THE CALLER'S TO DECLARE, and this signature makes declaring it
// unavoidable. Only the caller knows whether the answer will be REPORTED to a
// human (ClassB, which may be served from the broker's keyed cache or share
// another caller's in-flight execution) or will DECIDE something — a commit gate,
// a mutation, a refusal (ClassC, which is always computed fresh). Guessing that
// on the caller's behalf is the one mistake this epic cannot afford, so the
// broker never guesses: an absent or unrecognized class is decisional. See the
// classes and the stale-read budget in treestate.go.
func (c *Client) Tree(ctx context.Context, class Class) (TreeResult, error) {
	if res, ok := c.treeViaBroker(ctx, class); ok {
		return res, nil
	}
	st, err := c.treeRunner().TreeState(ctx)
	if err != nil {
		return TreeResult{}, err
	}
	return TreeResult{TreeState: st, Provenance: FallbackSpawn}, nil
}

// treeViaBroker attempts the socket path. Beyond the mandatory-provenance rule
// every reply obeys, it refuses one shape the object path has no analogue for: a
// CACHED answer to a decisional query.
//
// That check is redundant against a correct broker — Server.treeState never
// consults the cache for a Class C caller. It is here anyway because a client
// must not inherit its correctness from a resident process it cannot see: the
// broker on the socket may be an older or regressed build, and it must cost this
// caller a spawn, not a stale gate decision. Refusing routes to the same fallback
// as every other broker problem.
func (c *Client) treeViaBroker(ctx context.Context, class Class) (TreeResult, bool) {
	resp, ok := c.roundTrip(ctx, wireRequest{Op: opTree, Class: class})
	if !ok || resp.Tree == nil {
		return TreeResult{}, false
	}
	if resp.Provenance != Broker && resp.Provenance != Cache {
		return TreeResult{}, false
	}
	if class.Decisional() && resp.Provenance != Broker {
		return TreeResult{}, false
	}
	return TreeResult{TreeState: *resp.Tree, Provenance: resp.Provenance}, true
}

// Stats asks a live broker for its counters. ok is false when no broker
// answered, which is how an operator surface tells "the broker is down" apart
// from "the broker is up and has served nothing" — the stallscan rule, applied
// to this package's own status output.
func (c *Client) Stats(ctx context.Context) (Stats, bool) {
	resp, ok := c.roundTrip(ctx, wireRequest{Op: opStats})
	if !ok || resp.Stats == nil {
		return Stats{}, false
	}
	return *resp.Stats, true
}

// viaBroker attempts the socket path. It returns ok=false for every failure
// mode; the caller's job is to fall back, never to interpret why.
func (c *Client) viaBroker(ctx context.Context, rev string) (Result, bool) {
	resp, ok := c.roundTrip(ctx, wireRequest{Op: opObject, Rev: rev})
	if !ok || resp.Object == nil {
		return Result{}, false
	}
	// Provenance is mandatory. An answer the broker could not tag is an answer
	// this package will not hand back — see the package doc.
	if resp.Provenance != Broker && resp.Provenance != Cache {
		return Result{}, false
	}
	// A payload that does not match the size git declared is a truncated or
	// corrupted transport, not an object. Re-derive it.
	if int64(len(resp.Object.Data)) != resp.Object.Size {
		return Result{}, false
	}
	return Result{Object: *resp.Object, Provenance: resp.Provenance}, true
}

// roundTrip performs the whole deadlined exchange: read the published token,
// dial, write one request, read one response. Every step shares ONE deadline, so
// a broker that is slow at any stage — accept, git, write — is cut off by the
// same fuse.
func (c *Client) roundTrip(ctx context.Context, req wireRequest) (wireResponse, bool) {
	rv := RendezvousIn(c.Dir, c.RepoRoot)
	tok, err := os.ReadFile(rv.Token)
	if err != nil {
		return wireResponse{}, false
	}
	req.Token = strings.TrimSpace(string(tok))

	deadline := time.Now().Add(c.timeout())
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	dialWait := time.Until(deadline)
	if dialWait <= 0 {
		return wireResponse{}, false
	}
	conn, err := net.DialTimeout("unix", rv.Socket, dialWait)
	if err != nil {
		return wireResponse{}, false
	}
	defer conn.Close()
	_ = conn.SetDeadline(deadline)

	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return wireResponse{}, false
	}
	var resp wireResponse
	if err := json.NewDecoder(io.LimitReader(conn, maxResponseBytes)).Decode(&resp); err != nil {
		return wireResponse{}, false
	}
	if resp.Error != "" {
		return wireResponse{}, false
	}
	return resp, true
}
