package gateway

// leaseplane.go is the multi-node dev-server READ plane (#2297, epic #2254, plane 1):
// GET /v1/leases and GET /v1/sessions serve the coordinator clone's refs/fak/locks/*
// view at HTTP latency, so a node on the same LAN/Tailscale as this gateway can consult
// live lease and presence state fresh, instead of waiting for a plane-0 git fetch
// window (`fak leaseref sync`). Nodes that cannot reach this gateway degrade to plane 0
// and are exactly as safe as before — this surface only ADDS freshness, it moves no
// admission decision: the arbiter stays wherever it runs today and just gets fresher
// input. The write side (single-arbiter fenced acquire) is the separate child #2299.
//
// The gateway stays leaseref-BLIND: like the tasks snapshot (fak_tasks.go), the host
// CLI injects providers via SetLeasePlaneProviders, and the row bytes cross the seam as
// pre-marshaled JSON — the SAME canonical shapes `fak leaseref live` / `fak leaseref
// liveness` emit, so the HTTP plane and the CLI can never drift apart. A deployment
// that did not wire the providers returns 404, never a silent empty reading — the same
// fail-closed posture as the other injected surfaces.
//
// Every value here is OBSERVED: read from this gateway's own clone at ObservedUnix,
// never relayed from a client's claim (the witnessed-status contract,
// dos.toml RUN_STATUS_CLAIMED_FIELD — a status a peer consumes must carry evidence,
// not self-report). The source field states the qualifier explicitly.

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// leasePlaneSource names where every value in a lease-plane response was read from.
// The OBSERVED qualifier is load-bearing: these are values this gateway read from its
// own clone's ref store at observed_unix, not a relayed claim.
const leasePlaneSource = "refs/fak/locks/* on the coordinator clone (observed at observed_unix, not client-claimed)"

// leasePlaneTimeout bounds one observation of the local ref store. The providers shell
// to git (off the adjudication hot path — this is an operational read surface); a
// wedged git must time out the request, never pin the handler.
const leasePlaneTimeout = 10 * time.Second

// LeasePlaneView is one observation of the lock-lease namespace, as returned by the
// host-injected leases provider: the dos_arbitrate live_leases projection
// (leaseref.LiveLeases → [{lane, lane_kind, tree}]) pre-marshaled to JSON, plus the
// unix instant it was read.
type LeasePlaneView struct {
	ObservedUnix int64
	LiveLeases   json.RawMessage
}

// LeasePresenceView is one observation of the presence namespace: the live session
// descriptors (leaseref.LiveSessions) and each live lock lease classified by its owning
// session's liveness (leaseref.ClassifyLive: self|peer-live|peer-dead|peer-unknown),
// both pre-marshaled to JSON.
type LeasePresenceView struct {
	ObservedUnix     int64
	Sessions         json.RawMessage
	ClassifiedLeases json.RawMessage
}

// The host-injected readers. nil (a deployment that never wired them) keeps both
// routes 404 — inert, exactly like the tasks snapshot provider.
var (
	leasePlaneLeases   func(ctx context.Context) (LeasePlaneView, error)
	leasePlanePresence func(ctx context.Context) (LeasePresenceView, error)
)

// SetLeasePlaneProviders installs (or, with nils, clears) the host-injected lease and
// presence readers behind GET /v1/leases and GET /v1/sessions. cmd/fak wires these over
// the same leaseref store its CLI verbs use; this package never imports leaseref, so
// the git-shelling substrate stays out of the gateway's import graph.
func SetLeasePlaneProviders(
	leases func(ctx context.Context) (LeasePlaneView, error),
	presence func(ctx context.Context) (LeasePresenceView, error),
) {
	leasePlaneLeases = leases
	leasePlanePresence = presence
}

// LeasesResponse is the GET /v1/leases body: the live (non-expired) lock leases
// projected into the exact live_leases shape a dos_arbitrate-style admission kernel
// consumes — a caller can feed live_leases straight into its arbiter.
type LeasesResponse struct {
	ObservedUnix int64           `json:"observed_unix"`
	Source       string          `json:"source"`
	LiveLeases   json.RawMessage `json:"live_leases"`
}

// LeaseSessionsResponse is the GET /v1/sessions body: the live session descriptors
// (presence/heartbeat) plus every live lock lease classified by its owning session's
// liveness. Classification is from an ANONYMOUS reader's view — nothing classifies
// `self`; each caller re-keys reclaim decisions on its own session id.
type LeaseSessionsResponse struct {
	ObservedUnix     int64           `json:"observed_unix"`
	Source           string          `json:"source"`
	Sessions         json.RawMessage `json:"sessions"`
	ClassifiedLeases json.RawMessage `json:"classified_leases"`
}

// handleLeases serves GET /v1/leases — the read half of the dev-server lease plane.
func (s *Server) handleLeases(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	provider := leasePlaneLeases
	if provider == nil {
		writeErr(w, http.StatusNotFound, "lease read plane is not configured for this deployment")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), leasePlaneTimeout)
	defer cancel()
	v, err := provider(ctx)
	if err != nil {
		// Log the detail for the operator; the wire gets a generic message (the local
		// git error text never crosses to a possibly-unauthenticated caller).
		s.logf("gateway: lease read plane: %v", err)
		writeErr(w, http.StatusInternalServerError, "lease read plane failed to observe the local ref store")
		return
	}
	writeJSON(w, http.StatusOK, LeasesResponse{
		ObservedUnix: v.ObservedUnix,
		Source:       leasePlaneSource,
		LiveLeases:   orEmptyJSONArray(v.LiveLeases),
	})
}

// handleLeaseSessions serves GET /v1/sessions — the presence half of the dev-server
// lease plane. Distinct from /v1/fak/sessions (the served-session DRIVE-state
// snapshot): this one is the fleet's cross-machine guard-session descriptors and the
// lease-liveness classification built on them.
func (s *Server) handleLeaseSessions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	provider := leasePlanePresence
	if provider == nil {
		writeErr(w, http.StatusNotFound, "presence read plane is not configured for this deployment")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), leasePlaneTimeout)
	defer cancel()
	v, err := provider(ctx)
	if err != nil {
		s.logf("gateway: presence read plane: %v", err)
		writeErr(w, http.StatusInternalServerError, "presence read plane failed to observe the local ref store")
		return
	}
	writeJSON(w, http.StatusOK, LeaseSessionsResponse{
		ObservedUnix:     v.ObservedUnix,
		Source:           leasePlaneSource,
		Sessions:         orEmptyJSONArray(v.Sessions),
		ClassifiedLeases: orEmptyJSONArray(v.ClassifiedLeases),
	})
}

// orEmptyJSONArray keeps the wire contract "an empty view is []" — a nil RawMessage
// would otherwise encode as null, which a consumer folding live_leases must not have
// to special-case (the same rule leaseref.LiveLeases applies to its own slice).
func orEmptyJSONArray(m json.RawMessage) json.RawMessage {
	if len(m) == 0 {
		return json.RawMessage("[]")
	}
	return m
}
