package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
	gateway.SetWorkspaceLeaseAdmissionProvider(serveWorkspaceLeaseAdmission)
}

// leasePlaneDir resolves the repo whose refs/fak/locks/* namespace the read plane
// serves: FAK_LEASEPLANE_DIR when the coordinator gateway runs outside the clone,
// else git discovery from the process cwd — the same default as the CLI verbs.
func leasePlaneDir() string { return os.Getenv("FAK_LEASEPLANE_DIR") }

const workspaceLeaseAdmissionTimeout = 30 * time.Second

type dosLiveLease struct {
	Holder string   `json:"holder"`
	Lane   string   `json:"lane"`
	Tree   []string `json:"tree"`
}

// serveWorkspaceLeaseAdmission reads the DOS WAL through its canonical CLI.
// The fixed command is bounded and its workspace comes from this host process
// rather than the request wire. Lease holder/session text is not authenticated
// here and therefore never grants an ownership exemption.
func serveWorkspaceLeaseAdmission(ctx context.Context) (gateway.WorkspaceLeaseAdmissionView, error) {
	root := strings.TrimSpace(leasePlaneDir())
	if root == "" {
		root = "."
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return gateway.WorkspaceLeaseAdmissionView{}, fmt.Errorf("resolve DOS workspace: %w", err)
	}
	dosReadDeadline, cancel := context.WithTimeout(ctx, workspaceLeaseAdmissionTimeout)
	defer cancel()
	cmd := exec.CommandContext(dosReadDeadline, "dos", "lease-lane", "live")
	// DOS workspace discovery is cwd based. Passing an explicit workspace path
	// can select a different .dos store than the repository-local authority.
	cmd.Dir = absRoot
	output, err := cmd.Output()
	if err != nil {
		return gateway.WorkspaceLeaseAdmissionView{}, fmt.Errorf("read DOS live leases: %w", err)
	}
	var rows []dosLiveLease
	if err := json.Unmarshal(output, &rows); err != nil {
		return gateway.WorkspaceLeaseAdmissionView{}, fmt.Errorf("decode DOS live leases: %w", err)
	}
	if rows == nil {
		return gateway.WorkspaceLeaseAdmissionView{}, fmt.Errorf("decode DOS live leases: expected array")
	}
	view := gateway.WorkspaceLeaseAdmissionView{
		WorkspaceRoot: absRoot,
		OwnSession:    "",
		Leases:        make([]gateway.WorkspaceAdmissionLease, 0, len(rows)),
	}
	for _, row := range rows {
		if len(row.Tree) == 0 {
			return gateway.WorkspaceLeaseAdmissionView{}, fmt.Errorf("DOS live lease %q has no tree", row.Lane)
		}
		for _, tree := range row.Tree {
			if strings.TrimSpace(tree) == "" {
				return gateway.WorkspaceLeaseAdmissionView{}, fmt.Errorf("DOS live lease %q has an empty tree", row.Lane)
			}
		}
		view.Leases = append(view.Leases, gateway.WorkspaceAdmissionLease{
			TreeGlobs: append([]string(nil), row.Tree...),
			// DOS holder is caller-selected text, not an authenticated session
			// binding. It must never qualify for the own-session exemption.
			SessionID: "",
		})
	}
	return view, nil
}

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
