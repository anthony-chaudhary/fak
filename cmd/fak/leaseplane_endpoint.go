package main

import (
	"context"
	"encoding/json"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// init wires the multi-node dev-server READ plane (#2297, epic #2254 plane 1) —
// GET /v1/leases and GET /v1/sessions — over the same leaseref store the
// `fak leaseref` CLI verbs use. Installing here is inert for every subcommand:
// the store is only consulted when a served gateway request actually arrives, and
// each observation is one bounded read of the local clone's refs/fak/locks/*
// namespace (for-each-ref + per-record cat-file), off the adjudication hot path.
func init() {
	gateway.SetLeasePlaneProviders(serveLeasePlaneLeases, serveLeasePlanePresence)
}

// leasePlaneDir resolves the repo whose refs/fak/locks/* namespace the read plane
// serves: FAK_LEASEPLANE_DIR when the coordinator gateway runs outside the clone,
// else git discovery from the process cwd — the same default as the CLI verbs.
func leasePlaneDir() string { return os.Getenv("FAK_LEASEPLANE_DIR") }

// serveLeasePlaneLeases is the gateway leases provider: the live (non-expired) lock
// leases projected into the dos_arbitrate live_leases shape, observed now.
func serveLeasePlaneLeases(ctx context.Context) (gateway.LeasePlaneView, error) {
	now := time.Now()
	leases, err := leaseref.NewInDir(leasePlaneDir()).LiveLeases(ctx, now)
	if err != nil {
		return gateway.LeasePlaneView{}, err
	}
	raw, err := json.Marshal(leases)
	if err != nil {
		return gateway.LeasePlaneView{}, err
	}
	return gateway.LeasePlaneView{ObservedUnix: now.Unix(), LiveLeases: raw}, nil
}

// serveLeasePlanePresence is the gateway presence provider: the live session
// descriptors plus every live lock lease classified by its owning session's
// liveness. selfSession is empty — the coordinator answers as an ANONYMOUS reader
// (nothing classifies `self`); each caller re-keys reclaim decisions on its own
// session id, and reclaiming still goes through the fenced acquire.
func serveLeasePlanePresence(ctx context.Context) (gateway.LeasePresenceView, error) {
	now := time.Now()
	store := leaseref.NewInDir(leasePlaneDir())
	live, _, err := store.LiveSessions(ctx, now)
	if err != nil {
		return gateway.LeasePresenceView{}, err
	}
	if live == nil {
		live = []leaseref.SessionDescriptor{}
	}
	classified, err := store.ClassifyLive(ctx, "", now)
	if err != nil {
		return gateway.LeasePresenceView{}, err
	}
	rawSessions, err := json.Marshal(live)
	if err != nil {
		return gateway.LeasePresenceView{}, err
	}
	rawClassified, err := json.Marshal(classified)
	if err != nil {
		return gateway.LeasePresenceView{}, err
	}
	return gateway.LeasePresenceView{
		ObservedUnix:     now.Unix(),
		Sessions:         rawSessions,
		ClassifiedLeases: rawClassified,
	}, nil
}
