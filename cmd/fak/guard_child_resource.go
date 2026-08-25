package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const (
	guardResourcePollDefault   = time.Second
	guardTreeCommitDefault     = uint64(64) << 30
	guardSystemHeadroomDefault = uint64(16) << 30
)

type guardResourcePolicy struct {
	PollInterval      time.Duration
	MaxTreeCommit     uint64
	MinSystemHeadroom uint64
	Stop              <-chan struct{}
}

type guardResourceDecision struct {
	Stop              bool
	Reason            string
	Offender          procguard.CommitProcess
	TreeCommitBytes   uint64
	SystemCommitBytes uint64
	SystemCommitLimit uint64
	ThresholdBytes    uint64
	HeadroomBytes     uint64
	OwnedPIDs         []int
}

type guardResourceReceipt struct {
	Schema             string `json:"schema"`
	At                 string `json:"at"`
	TraceID            string `json:"trace_id"`
	Agent              string `json:"agent"`
	RootPID            int    `json:"root_pid"`
	OffenderPID        int    `json:"offender_pid"`
	OffenderPPID       int    `json:"offender_ppid"`
	OffenderName       string `json:"offender_name"`
	OffenderCommand    string `json:"offender_command,omitempty"`
	TreeCommitBytes    uint64 `json:"tree_commit_bytes"`
	SystemCommitBytes  uint64 `json:"system_commit_bytes"`
	SystemCommitLimit  uint64 `json:"system_commit_limit"`
	ThresholdBytes     uint64 `json:"threshold_bytes"`
	HeadroomBytes      uint64 `json:"headroom_bytes"`
	Reason             string `json:"reason"`
	Action             string `json:"action"`
	DescendantsSurvive bool   `json:"descendants_survive"`
	Detail             string `json:"detail,omitempty"`
}

func guardResourcePolicyFromEnv() guardResourcePolicy {
	p := guardResourcePolicy{PollInterval: guardResourcePollDefault, MaxTreeCommit: guardTreeCommitDefault, MinSystemHeadroom: guardSystemHeadroomDefault}
	if n, err := strconv.ParseUint(strings.TrimSpace(os.Getenv("FAK_CHILD_MAX_COMMIT_MB")), 10, 64); err == nil && n > 0 {
		p.MaxTreeCommit = n << 20
	}
	if n, err := strconv.ParseUint(strings.TrimSpace(os.Getenv("FAK_SYSTEM_COMMIT_HEADROOM_MB")), 10, 64); err == nil && n > 0 {
		p.MinSystemHeadroom = n << 20
	}
	if d, err := time.ParseDuration(strings.TrimSpace(os.Getenv("FAK_CHILD_RESOURCE_POLL"))); err == nil && d >= 100*time.Millisecond {
		p.PollInterval = d
	}
	return p
}

func decideGuardResource(p guardResourcePolicy, s procguard.CommitSnapshot) guardResourceDecision {
	d := guardResourceDecision{TreeCommitBytes: s.TreeCommitBytes, SystemCommitBytes: s.SystemCommitBytes, SystemCommitLimit: s.SystemCommitLimit, ThresholdBytes: p.MaxTreeCommit}
	for _, process := range s.Processes {
		d.OwnedPIDs = append(d.OwnedPIDs, process.PID)
	}
	if s.SystemCommitLimit >= s.SystemCommitBytes {
		d.HeadroomBytes = s.SystemCommitLimit - s.SystemCommitBytes
	}
	if len(s.Processes) > 0 {
		sort.Slice(s.Processes, func(i, j int) bool { return s.Processes[i].CommitBytes > s.Processes[j].CommitBytes })
		d.Offender = s.Processes[0]
	}
	if p.MaxTreeCommit > 0 && s.TreeCommitBytes >= p.MaxTreeCommit {
		d.Stop = true
		d.Reason = "CHILD_TREE_COMMIT_LIMIT"
		return d
	}
	if p.MinSystemHeadroom > 0 && s.SystemCommitLimit > 0 && d.HeadroomBytes <= p.MinSystemHeadroom {
		d.Stop = true
		d.Reason = "SYSTEM_COMMIT_HEADROOM"
	}
	return d
}

func appendGuardResourceReceipt(path string, r guardResourceReceipt) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("child resource receipt path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create child resource receipt directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open child resource receipt: %w", err)
	}
	if err = json.NewEncoder(f).Encode(r); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return fmt.Errorf("persist child resource receipt: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close child resource receipt: %w", closeErr)
	}
	return nil
}

func guardResourceReceiptPath() string {
	if p := strings.TrimSpace(os.Getenv("FAK_CHILD_RESOURCE_JOURNAL")); p != "" {
		return p
	}
	if base, err := os.UserConfigDir(); err == nil && strings.TrimSpace(base) != "" {
		return filepath.Join(base, "fak", "guard", "child-resource.jsonl")
	}
	return filepath.Join(os.TempDir(), "fak", "guard", "child-resource.jsonl")
}

func guardResourceReason(d guardResourceDecision) string {
	return fmt.Sprintf("%s tree_commit=%d threshold=%d system_commit=%d limit=%d headroom=%d offender_pid=%d", d.Reason, d.TreeCommitBytes, d.ThresholdBytes, d.SystemCommitBytes, d.SystemCommitLimit, d.HeadroomBytes, d.Offender.PID)
}

func startGuardChildResourceMonitor(rootPID int, traceID, agent string, policy guardResourcePolicy) <-chan guardChildWaitEvent {
	out := make(chan guardChildWaitEvent, 1)
	go func() {
		ticker := time.NewTicker(policy.PollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-policy.Stop:
				return
			case <-ticker.C:
			}
			snapshot, supported, detail := procguard.CollectCommitSnapshot(rootPID)
			if !supported {
				if runtime.GOOS == "windows" {
					out <- guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: "CHILD_RESOURCE_MONITOR_UNAVAILABLE: " + detail, Resource: &guardResourceDecision{Stop: true, Reason: "CHILD_RESOURCE_MONITOR_UNAVAILABLE", Offender: procguard.CommitProcess{PID: rootPID}, OwnedPIDs: []int{rootPID}}}
				}
				return
			}
			if detail != "" {
				if runtime.GOOS == "windows" {
					out <- guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: "CHILD_RESOURCE_MONITOR_ERROR: " + detail, Resource: &guardResourceDecision{Stop: true, Reason: "CHILD_RESOURCE_MONITOR_ERROR", Offender: procguard.CommitProcess{PID: rootPID}, OwnedPIDs: []int{rootPID}}}
				}
				return
			}
			decision := decideGuardResource(policy, snapshot)
			if !decision.Stop {
				continue
			}
			out <- guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: guardResourceReason(decision), Resource: &decision}
			return
		}
	}()
	return out
}

func guardWriteResourceReceipt(event guardChildWaitEvent, traceID, agent string, rootPID int) error {
	if event.Resource == nil {
		return errors.New("child resource receipt missing decision")
	}
	d := *event.Resource
	deadline := time.Now().Add(2 * time.Second)
	for {
		rows, detail := procguard.CollectRelations()
		if detail != "" {
			return fmt.Errorf("verify child resource reap: %s", detail)
		}
		alive := make(map[int]bool, len(rows))
		for _, row := range rows {
			alive[row.PID] = true
		}
		survives := false
		for _, pid := range d.OwnedPIDs {
			if alive[pid] {
				survives = true
				break
			}
		}
		if !survives {
			return appendGuardResourceReceipt(guardResourceReceiptPath(), guardResourceReceipt{Schema: "fak.guard.child-resource.v1", At: time.Now().UTC().Format(time.RFC3339Nano), TraceID: traceID, Agent: agent, RootPID: rootPID, OffenderPID: d.Offender.PID, OffenderPPID: d.Offender.PPID, OffenderName: d.Offender.Name, OffenderCommand: d.Offender.CommandLine, TreeCommitBytes: d.TreeCommitBytes, SystemCommitBytes: d.SystemCommitBytes, SystemCommitLimit: d.SystemCommitLimit, ThresholdBytes: d.ThresholdBytes, HeadroomBytes: d.HeadroomBytes, Reason: d.Reason, Action: "reap_tree", DescendantsSurvive: false})
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("verify child resource reap: owned processes still alive: %v", d.OwnedPIDs)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
