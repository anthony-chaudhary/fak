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
	guardTreeRSSFallback       = uint64(4) << 30
	guardTreeRSSMinimum        = uint64(1) << 30
)

type guardResourcePolicy struct {
	PollInterval      time.Duration
	Metric            procguard.MemoryMetric
	MaxTreeBytes      uint64
	MinSystemHeadroom uint64
	Stop              <-chan struct{}
}

type guardResourceDecision struct {
	Stop           bool
	Reason         string
	Metric         procguard.MemoryMetric
	Offender       procguard.MemoryProcess
	TreeBytes      uint64
	SystemBytes    uint64
	SystemLimit    uint64
	ThresholdBytes uint64
	HeadroomBytes  uint64
	OwnedPIDs      []int
}

type guardResourceReceipt struct {
	Schema             string  `json:"schema"`
	At                 string  `json:"at"`
	TraceID            string  `json:"trace_id"`
	Agent              string  `json:"agent"`
	RootPID            int     `json:"root_pid"`
	OffenderPID        int     `json:"offender_pid"`
	OffenderPPID       int     `json:"offender_ppid"`
	OffenderName       string  `json:"offender_name"`
	OffenderCommand    string  `json:"offender_command,omitempty"`
	MemoryMetric       string  `json:"memory_metric"`
	TreeMemoryBytes    uint64  `json:"tree_memory_bytes"`
	SystemMemoryBytes  uint64  `json:"system_memory_bytes,omitempty"`
	SystemMemoryLimit  uint64  `json:"system_memory_limit,omitempty"`
	TreeCommitBytes    *uint64 `json:"tree_commit_bytes,omitempty"`
	SystemCommitBytes  *uint64 `json:"system_commit_bytes,omitempty"`
	SystemCommitLimit  *uint64 `json:"system_commit_limit,omitempty"`
	TreeRSSBytes       *uint64 `json:"tree_rss_bytes,omitempty"`
	ThresholdBytes     uint64  `json:"threshold_bytes"`
	HeadroomBytes      uint64  `json:"headroom_bytes"`
	Reason             string  `json:"reason"`
	Action             string  `json:"action"`
	DescendantsSurvive bool    `json:"descendants_survive"`
	Detail             string  `json:"detail,omitempty"`
}

func guardResourcePolicyFromEnv() guardResourcePolicy {
	metric := procguard.MemoryMetricCommit
	maxTree := guardTreeCommitDefault
	headroom := guardSystemHeadroomDefault
	if runtime.GOOS == "darwin" {
		metric = procguard.MemoryMetricRSS
		hostBytes, _ := procguard.HostPhysicalMemoryBytes()
		maxTree = guardTreeRSSDefault(hostBytes)
		headroom = 0 // physical capacity is not a current system-RSS pressure sample.
	}
	p := guardResourcePolicy{PollInterval: guardResourcePollDefault, Metric: metric, MaxTreeBytes: maxTree, MinSystemHeadroom: headroom}
	// The generic name is preferred. Metric-specific names retain compatibility
	// with the Windows spine and give Darwin an RSS-honest override.
	override := strings.TrimSpace(os.Getenv("FAK_CHILD_MAX_MEMORY_MB"))
	if override == "" && metric == procguard.MemoryMetricRSS {
		override = strings.TrimSpace(os.Getenv("FAK_CHILD_MAX_RSS_MB"))
	}
	if override == "" {
		override = strings.TrimSpace(os.Getenv("FAK_CHILD_MAX_COMMIT_MB"))
	}
	if n, ok := parseGuardResourceMegabytes(override); ok {
		p.MaxTreeBytes = n
	}
	if n, ok := parseGuardResourceMegabytes(strings.TrimSpace(os.Getenv("FAK_SYSTEM_COMMIT_HEADROOM_MB"))); ok {
		p.MinSystemHeadroom = n
	}
	if d, err := time.ParseDuration(strings.TrimSpace(os.Getenv("FAK_CHILD_RESOURCE_POLL"))); err == nil && d >= 100*time.Millisecond {
		p.PollInterval = d
	}
	return p
}

func parseGuardResourceMegabytes(raw string) (uint64, bool) {
	n, err := strconv.ParseUint(raw, 10, 64)
	if err != nil || n == 0 || n > ^uint64(0)>>20 {
		return 0, false
	}
	return n << 20, true
}

func guardTreeRSSDefault(hostPhysicalBytes uint64) uint64 {
	if hostPhysicalBytes == 0 {
		return guardTreeRSSFallback
	}
	limit := hostPhysicalBytes / 4
	if limit < guardTreeRSSMinimum {
		limit = guardTreeRSSMinimum
	}
	if limit > guardTreeCommitDefault {
		limit = guardTreeCommitDefault
	}
	return limit
}

func decideGuardResource(p guardResourcePolicy, s procguard.MemorySnapshot) guardResourceDecision {
	d := guardResourceDecision{Metric: s.Metric, TreeBytes: s.TreeBytes, SystemBytes: s.SystemBytes, SystemLimit: s.SystemLimit, ThresholdBytes: p.MaxTreeBytes}
	for _, process := range s.Processes {
		d.OwnedPIDs = append(d.OwnedPIDs, process.PID)
	}
	if s.SystemLimit >= s.SystemBytes {
		d.HeadroomBytes = s.SystemLimit - s.SystemBytes
	}
	if len(s.Processes) > 0 {
		sort.Slice(s.Processes, func(i, j int) bool { return s.Processes[i].Bytes > s.Processes[j].Bytes })
		d.Offender = s.Processes[0]
	}
	if p.MaxTreeBytes > 0 && s.TreeBytes >= p.MaxTreeBytes {
		d.Stop = true
		d.Reason = "CHILD_TREE_" + strings.ToUpper(string(s.Metric)) + "_LIMIT"
		return d
	}
	if s.Metric == procguard.MemoryMetricCommit && p.MinSystemHeadroom > 0 && s.SystemLimit > 0 && d.HeadroomBytes <= p.MinSystemHeadroom {
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
	metric := d.Metric
	if metric == "" {
		metric = procguard.MemoryMetricCommit
	}
	if metric == procguard.MemoryMetricCommit {
		// Preserve the Windows v1 reason text byte-for-byte for existing journal
		// consumers while Darwin gets an RSS-honest label below.
		return fmt.Sprintf("%s tree_commit=%d threshold=%d system_commit=%d limit=%d headroom=%d offender_pid=%d", d.Reason, d.TreeBytes, d.ThresholdBytes, d.SystemBytes, d.SystemLimit, d.HeadroomBytes, d.Offender.PID)
	}
	return fmt.Sprintf("%s metric=%s tree_bytes=%d threshold=%d system_bytes=%d limit=%d headroom=%d offender_pid=%d", d.Reason, metric, d.TreeBytes, d.ThresholdBytes, d.SystemBytes, d.SystemLimit, d.HeadroomBytes, d.Offender.PID)
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
			snapshot, supported, detail := procguard.CollectMemorySnapshot(rootPID)
			if !supported {
				if runtime.GOOS == "windows" {
					out <- guardResourceMonitorFailure(rootPID, snapshot, "CHILD_RESOURCE_MONITOR_UNAVAILABLE", detail)
				}
				return
			}
			if detail != "" {
				out <- guardResourceMonitorFailure(rootPID, snapshot, "CHILD_RESOURCE_MONITOR_ERROR", detail)
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

func guardResourceMonitorFailure(rootPID int, snapshot procguard.MemorySnapshot, reason, detail string) guardChildWaitEvent {
	d := guardResourceDecision{Stop: true, Reason: reason, Metric: snapshot.Metric, TreeBytes: snapshot.TreeBytes, SystemBytes: snapshot.SystemBytes, SystemLimit: snapshot.SystemLimit, Offender: procguard.MemoryProcess{PID: rootPID}}
	for _, process := range snapshot.Processes {
		d.OwnedPIDs = append(d.OwnedPIDs, process.PID)
		if process.PID == rootPID {
			d.Offender = process
		}
	}
	if len(d.OwnedPIDs) == 0 {
		d.OwnedPIDs = []int{rootPID}
	}
	return guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: reason + ": " + detail, Resource: &d}
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
			return appendGuardResourceReceipt(guardResourceReceiptPath(), newGuardResourceReceipt(traceID, agent, rootPID, d))
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("verify child resource reap: owned processes still alive: %v", d.OwnedPIDs)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func newGuardResourceReceipt(traceID, agent string, rootPID int, d guardResourceDecision) guardResourceReceipt {
	receipt := guardResourceReceipt{Schema: "fak.guard.child-resource.v1", At: time.Now().UTC().Format(time.RFC3339Nano), TraceID: traceID, Agent: agent, RootPID: rootPID, OffenderPID: d.Offender.PID, OffenderPPID: d.Offender.PPID, OffenderName: d.Offender.Name, OffenderCommand: d.Offender.CommandLine, MemoryMetric: string(d.Metric), TreeMemoryBytes: d.TreeBytes, SystemMemoryBytes: d.SystemBytes, SystemMemoryLimit: d.SystemLimit, ThresholdBytes: d.ThresholdBytes, HeadroomBytes: d.HeadroomBytes, Reason: d.Reason, Action: "reap_tree", DescendantsSurvive: false}
	if d.Metric == procguard.MemoryMetricRSS {
		receipt.TreeRSSBytes = uint64Pointer(d.TreeBytes)
	} else {
		receipt.MemoryMetric = string(procguard.MemoryMetricCommit)
		receipt.TreeCommitBytes = uint64Pointer(d.TreeBytes)
		receipt.SystemCommitBytes = uint64Pointer(d.SystemBytes)
		receipt.SystemCommitLimit = uint64Pointer(d.SystemLimit)
	}
	return receipt
}

func uint64Pointer(v uint64) *uint64 { return &v }
