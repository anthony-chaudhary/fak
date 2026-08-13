package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/leasequeue"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

const microParentTaskID = "micro-selfcheck-parent"

type microChildReceipt struct {
	WorkUnitID   string `json:"work_unit_id"`
	LeaseID      string `json:"lease_id"`
	SessionID    string `json:"session_id"`
	State        string `json:"state"`
	EffectDigest string `json:"effect_digest"`
	Witnessed    bool   `json:"witnessed"`
}

func foldMicroSelfcheckChildren() ([]microChildReceipt, error) {
	taxonomy := regionadmit.Taxonomy{Trees: map[string][]string{
		"micro-alpha": {"pinned/alpha"},
		"micro-beta":  {"pinned/beta"},
	}}
	tickets := []leasequeue.Ticket{
		{ID: "lease-micro-alpha", Actor: "value-000", Lane: "micro-alpha", Tree: taxonomy.Trees["micro-alpha"], Class: "active", EnqueuedUnix: 1},
		{ID: "lease-micro-beta", Actor: "value-001", Lane: "micro-beta", Tree: taxonomy.Trees["micro-beta"], Class: "active", EnqueuedUnix: 1},
	}
	plan := leasequeue.Plan(tickets, nil, taxonomy, leasequeue.Params{NowUnix: 1})
	children := make([]microChildReceipt, 0, len(tickets))
	for i, ticket := range tickets {
		entry, ok := plan.Find(ticket.ID)
		if !ok || !entry.Grant {
			return nil, fmt.Errorf("lease %s was not granted", ticket.ID)
		}
		children = append(children, microChildReceipt{WorkUnitID: ticket.Lane, LeaseID: ticket.ID, SessionID: fmt.Sprintf("value-%03d", i), State: "queued"})
	}
	return children, nil
}

// digestMicroReadback reads the retired runner's observed answer after execution;
// the child agent does not author this digest or witnessed verdict.
func digestMicroReadback(answer string) (string, bool) {
	if answer == "" {
		return "", false
	}
	sum := sha256.Sum256([]byte(answer))
	return "sha256:" + hex.EncodeToString(sum[:]), true
}
