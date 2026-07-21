package gateway

// leasewrite.go is the multi-node dev-server WRITE plane (#2299, epic #2254, plane 1 —
// the atomicity closure). It is the write twin of leaseplane.go's read plane: where
// GET /v1/leases only OBSERVES the coordinator clone's refs/fak/locks/* view, this
// surface MUTATES it, so a fenced acquire/renew/release routed through the coordinator
// becomes a SINGLE-ARBITER operation.
//
// WHY A SINGLE ARBITER IS THE ONLY HONEST CLOSURE. internal/leaseref's boundary is
// explicit (fence.go): refs/fak/locks/* is DISTRIBUTION / VISIBILITY, not atomic
// acquisition — two clones can both write a lease in the same fetch window, and git's
// merge converges the SET of refs without arbitrating a winner. leaseref.AcquireFenced
// closes that only for SAME-HOST racers (an update-ref old-value compare-and-swap). The
// remaining cross-host final race is structural for a leaderless git substrate; the ONE
// honest way to close it is to funnel every node's acquire through a SINGLE process that
// owns the store. The gateway is already the fleet's front door (fak guard), so it is
// that arbiter for nodes that can reach it. Nodes that cannot degrade to plane 0 and are
// exactly as safe as before — this surface only ADDS an arbiter, it removes no fallback.
//
// HOW THE ARBITER IS ATOMIC. Every write here is serialized through leaseWriteMu before
// it touches the injected store, so two acquires for the same id that arrive concurrently
// at the coordinator are ORDERED: the first takes the lease (fresh generation), and the
// second — reading the now-live lease held by a different holder — is refused LEASE_HELD.
// Serialization turns the same-host CAS's racy LEASE_CONTENDED into a deterministic
// LEASE_HELD, which is the whole point of a single arbiter: one winner, no retry storm.
//
// DENY-AS-VALUE, exactly as leaseref.FenceVerdict promises. A refusal — LEASE_HELD,
// STALE_LEASE, LEASE_CONTENDED, NO_LEASE — is a 200 response carrying ok:false and the
// closed reason, never an HTTP error. HTTP status is reserved for TRANSPORT/PROTOCOL
// facts only: 404 when the plane is unwired or the verb is unknown, 405 for a non-POST,
// 400 for a malformed body, 500 for an infrastructure failure (git not executable). A
// calling loop routes on the verdict's reason, never on the status class.
//
// LEASEREF-BLIND, like the read plane. This package never imports leaseref: the host CLI
// injects a single write function via SetLeaseWriteFunc, and the verdict crosses the seam
// as this package's own LeaseWriteResult (the same shapes leaseref.FenceVerdict/Record
// carry, re-declared here so the git-shelling substrate stays out of the gateway's import
// graph). A deployment that never wired the function returns 404 — the same fail-closed
// posture as every other injected surface.

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"time"
)

// leaseWriteSource names where an accepted write landed and against what. Parallel to
// leasePlaneSource: the value is not a client claim, it is the coordinator clone's own
// ref store after the arbitrated compare-and-swap.
const leaseWriteSource = "refs/fak/locks/* on the coordinator clone (single-arbiter fenced write, serialized through the gateway)"

// leaseWriteTimeout bounds one arbitrated write. The write function shells to git under
// the serialization lock; a wedged git must time out the request rather than pin the
// arbiter (and the lock) for every waiting node.
const leaseWriteTimeout = 15 * time.Second

// The fenced-write verbs, the trailing path segment under /v1/leases/. A closed set —
// any other segment is an unknown route (404), never a silent no-op.
const (
	leaseOpAcquire = "acquire"
	leaseOpRenew   = "renew"
	leaseOpRelease = "release"
)

// LeaseWriteRequest is the POST body for /v1/leases/{acquire,renew,release}. It carries
// the same fields leaseref.Record's write side reasons about: the lease id, the holder
// identity, the trees an acquire covers, the TTL window, and — for a renew or release —
// the Generation the caller believes it holds (the fencing token it must present so the
// arbiter can refuse a stale writer). TreeGlobs/TTLSeconds/Description are meaningful only
// on acquire; renew and release read id + holder + generation.
type LeaseWriteRequest struct {
	ID          string   `json:"id"`
	Holder      string   `json:"holder"`
	TreeGlobs   []string `json:"tree_globs,omitempty"`
	TTLSeconds  int64    `json:"ttl_seconds,omitempty"`
	Generation  int64    `json:"generation,omitempty"`
	Description string   `json:"description,omitempty"`
}

// LeaseWriteResult is the deny-as-value verdict of one fenced write, the wire mirror of
// leaseref.FenceVerdict plus the assigned token. OK is the only admit state; a non-empty
// Reason names the closed refusal class (LEASE_HELD / STALE_LEASE / LEASE_CONTENDED /
// NO_LEASE). Generation is the assigned fencing token on an accepted acquire/renew (the
// value the caller must present on every later write); CurrentGeneration and Holder name
// what the live lease actually is, so a refusal states who owns it. Op echoes the verb;
// ObservedUnix/Source stamp when and against what the arbiter decided.
type LeaseWriteResult struct {
	OK                bool     `json:"ok"`
	Reason            string   `json:"reason,omitempty"`
	Op                string   `json:"op"`
	ID                string   `json:"id"`
	Holder            string   `json:"holder,omitempty"`
	Generation        int64    `json:"generation"`
	CurrentGeneration int64    `json:"current_generation"`
	TreeGlobs         []string `json:"tree_globs,omitempty"`
	Detail            string   `json:"detail,omitempty"`
	ObservedUnix      int64    `json:"observed_unix"`
	Source            string   `json:"source"`
}

// LeaseWriteFunc is the host-injected arbiter body: given the verb and the decoded
// request, it performs the fenced write against the coordinator's leaseref store and
// returns the deny-as-value verdict. It returns a non-nil error ONLY for an infrastructure
// failure (git not executable, an unreadable record) — every POLICY outcome, including a
// refusal, is an ok:false LeaseWriteResult with a non-nil error. The gateway serializes
// calls to this function through leaseWriteMu, so an implementation need not lock itself.
type LeaseWriteFunc func(ctx context.Context, op string, req LeaseWriteRequest) (LeaseWriteResult, error)

// The host-injected arbiter. nil (a deployment that never wired it) keeps the whole
// /v1/leases/ subtree 404 — inert, exactly like the read plane's providers.
var leaseWriteFn LeaseWriteFunc

// leaseWriteMu SERIALIZES every arbitrated write through this coordinator: it is the
// mechanism that makes the gateway a single arbiter rather than one more racing clone.
// Held across the whole write function call so two acquires for one id are ordered — the
// first wins the lease, the second reads it live and is refused deterministically.
var leaseWriteMu sync.Mutex

// SetLeaseWriteFunc installs (or, with nil, clears) the host-injected single-arbiter
// fenced-write function behind POST /v1/leases/{acquire,renew,release}. cmd/fak wires it
// over the same leaseref store its CLI verbs use; this package never imports leaseref, so
// the git-shelling substrate stays out of the gateway's import graph.
func SetLeaseWriteFunc(fn LeaseWriteFunc) { leaseWriteFn = fn }

// handleLeaseWrite serves POST /v1/leases/{acquire,renew,release} — the write half of the
// dev-server lease plane. It routes on the trailing path segment (the verb), rejects any
// non-POST with 405, and 404s an unknown verb (the verb set is closed). The decoded
// request is handed to the injected arbiter UNDER leaseWriteMu, so the coordinator serves
// as a single arbiter; the verdict comes back as a 200 deny-as-value body (even a refusal),
// with HTTP error status reserved for transport/infrastructure facts.
func (s *Server) handleLeaseWrite(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	fn := leaseWriteFn
	if fn == nil {
		writeErr(w, http.StatusNotFound, "lease write plane is not configured for this deployment")
		return
	}
	// The verb is the single path segment after /v1/leases/. A trailing slash, an empty
	// segment, or a sub-path (a slash inside the segment) is not a known verb.
	op := strings.TrimPrefix(r.URL.Path, "/v1/leases/")
	if op == "" || strings.Contains(op, "/") || !knownLeaseOp(op) {
		writeErr(w, http.StatusNotFound, "unknown lease write verb (use acquire, renew, or release)")
		return
	}

	var req LeaseWriteRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	if req.ID == "" {
		writeErr(w, http.StatusBadRequest, "lease write requires a non-empty id")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), leaseWriteTimeout)
	defer cancel()

	// The single-arbiter serialization point: hold the lock across the whole write so two
	// concurrent acquires for one id are ORDERED into a clean winner + a LEASE_HELD loser,
	// not a racing pair that both attempt the same-host CAS.
	leaseWriteMu.Lock()
	res, err := fn(ctx, op, req)
	leaseWriteMu.Unlock()
	if err != nil {
		// Infrastructure failure only — the local git error text never crosses to a
		// possibly-unauthenticated caller. A POLICY refusal never lands here (it is an
		// ok:false result with a nil error).
		s.logf("gateway: lease write plane (%s %s): %v", op, req.ID, err)
		writeErr(w, http.StatusInternalServerError, "lease write plane failed to arbitrate against the local ref store")
		return
	}

	// Stamp the observation fields the arbiter itself does not fill, so every response
	// carries when and against what it was decided — the same witnessed-status discipline
	// the read plane applies.
	res.Op = op
	if res.ID == "" {
		res.ID = req.ID
	}
	if res.ObservedUnix == 0 {
		res.ObservedUnix = time.Now().Unix()
	}
	res.Source = leaseWriteSource
	// Deny-as-value: an accepted OR refused verdict is a 200. Only transport/infrastructure
	// facts (above) change the status.
	writeJSON(w, http.StatusOK, res)
}

// knownLeaseOp reports whether op is one of the closed fenced-write verbs. An unknown
// segment is a 404 route, never a silent accept.
func knownLeaseOp(op string) bool {
	switch op {
	case leaseOpAcquire, leaseOpRenew, leaseOpRelease:
		return true
	}
	return false
}
