package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const (
	guardResourcePollDefault    = time.Second
	guardTreeCommitDefault      = uint64(64) << 30
	guardTreeRSSFallback        = uint64(4) << 30
	guardTreeRSSMinimum         = uint64(1) << 30
	guardResourceDetailMaxBytes = 512
)

var (
	guardResourceBearerPattern      = regexp.MustCompile(`(?i)\bbearer\s+\S+`)
	guardResourceSecretPattern      = regexp.MustCompile(`(?i)((?:token|secret|passw(?:or)?d|api[-_]?key|credential|authorization)\s*[=:]\s*)\S+`)
	guardResourcePrivateHostPattern = regexp.MustCompile(`(?i)\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:internal|corp|lan|local|intranet)\b`)
	guardResourcePrivateIPv4Pattern = regexp.MustCompile(`\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b`)
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
	Detail         string
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
	BuildCommit        string  `json:"build_commit,omitempty"`
	BuildModule        string  `json:"build_module,omitempty"`
	BuildDirty         bool    `json:"build_dirty,omitempty"`
	ActivationID       string  `json:"activation_id,omitempty"`
}

type guardResourceConfig struct {
	MaxMemoryMB  uint64
	PollInterval time.Duration
	ReceiptPath  string
}

var guardResourceConfigured guardResourceConfig

func setGuardResourceConfig(config guardResourceConfig) {
	guardResourceConfigured = config
}

func guardResourcePolicyConfigured() guardResourcePolicy {
	metric := procguard.MemoryMetricCommit
	maxTree := guardTreeCommitDefault
	headroom := procguard.RequiredSystemCommitHeadroom(os.Getenv)
	if runtime.GOOS == "darwin" {
		metric = procguard.MemoryMetricRSS
		hostBytes, _ := procguard.HostPhysicalMemoryBytes()
		maxTree = guardTreeRSSDefault(hostBytes)
		headroom = 0 // physical capacity is not a current system-RSS pressure sample.
	}
	p := guardResourcePolicy{PollInterval: guardResourcePollDefault, Metric: metric, MaxTreeBytes: maxTree, MinSystemHeadroom: headroom}
	if guardResourceConfigured.MaxMemoryMB > 0 && guardResourceConfigured.MaxMemoryMB <= ^uint64(0)>>20 {
		p.MaxTreeBytes = guardResourceConfigured.MaxMemoryMB << 20
	}
	if guardResourceConfigured.PollInterval >= 100*time.Millisecond {
		p.PollInterval = guardResourceConfigured.PollInterval
	}
	return p
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
	headroom := procguard.EvaluateSystemCommitHeadroom(s, p.MinSystemHeadroom)
	d.HeadroomBytes = headroom.ObservedBytes
	if len(s.Processes) > 0 {
		sort.Slice(s.Processes, func(i, j int) bool { return s.Processes[i].Bytes > s.Processes[j].Bytes })
		d.Offender = s.Processes[0]
	}
	if p.MaxTreeBytes > 0 && s.TreeBytes >= p.MaxTreeBytes {
		d.Stop = true
		d.Reason = "CHILD_TREE_" + strings.ToUpper(string(s.Metric)) + "_LIMIT"
		return d
	}
	if headroom.Refuse {
		d.Stop = true
		d.Reason = headroom.Reason
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
	if p := strings.TrimSpace(guardResourceConfigured.ReceiptPath); p != "" {
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

// scrubGuardResourceDetail makes collector diagnostics safe for the durable
// child-resource receipt. The detail remains useful for public invariants and
// PID lists, but never carries secret-shaped values, private hosts or absolute
// machine paths. It is normalized to one line and byte-bounded before any
// terminal or persistence surface sees it.
func scrubGuardResourceDetail(detail string) string {
	fields := strings.Fields(detail)
	for i, field := range fields {
		if strings.ContainsAny(field, `/\`) {
			fields[i] = "[path]"
		}
	}
	detail = strings.Join(fields, " ")
	detail = guardResourceBearerPattern.ReplaceAllString(detail, "Bearer [redacted]")
	detail = guardResourceSecretPattern.ReplaceAllString(detail, "${1}[redacted]")
	detail = guardResourcePrivateHostPattern.ReplaceAllString(detail, "[host]")
	detail = guardResourcePrivateIPv4Pattern.ReplaceAllString(detail, "[ip]")
	if len(detail) <= guardResourceDetailMaxBytes {
		return detail
	}
	detail = detail[:guardResourceDetailMaxBytes-3]
	for !utf8.ValidString(detail) {
		detail = detail[:len(detail)-1]
	}
	return strings.TrimSpace(detail) + "..."
}

func startGuardChildResourceMonitor(rootPID int, traceID, agent string, policy guardResourcePolicy) <-chan guardChildWaitEvent {
	return startGuardChildResourceMonitorWithCollector(rootPID, traceID, agent, policy, procguard.CollectMemorySnapshot)
}

func startGuardChildResourceMonitorWithCollector(rootPID int, traceID, agent string, policy guardResourcePolicy, collect func(int) (procguard.MemorySnapshot, bool, string)) <-chan guardChildWaitEvent {
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
			snapshot, supported, detail := collect(rootPID)
			if runtime.GOOS == "darwin" && snapshot.RootPID == 0 {
				// The root exited during Darwin's independent relation/RSS census.
				// Leave its exit to the normal child wait/crash path; there is no
				// resource violation to receipt or reap.
				return
			}
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
	detail = scrubGuardResourceDetail(detail)
	d := guardResourceDecision{Stop: true, Reason: reason, Metric: snapshot.Metric, TreeBytes: snapshot.TreeBytes, SystemBytes: snapshot.SystemBytes, SystemLimit: snapshot.SystemLimit, Offender: procguard.MemoryProcess{PID: rootPID}, Detail: detail}
	for _, process := range snapshot.Processes {
		d.OwnedPIDs = append(d.OwnedPIDs, process.PID)
		if process.PID == rootPID {
			d.Offender = process
		}
	}
	if len(d.OwnedPIDs) == 0 {
		d.OwnedPIDs = []int{rootPID}
	}
	eventReason := reason
	if detail != "" {
		eventReason += ": " + detail
	}
	return guardChildWaitEvent{Kind: guardChildResourceLimit, Reason: eventReason, Resource: &d}
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

var (
	guardResourceBuildIdentity = buildIdentityFromRuntime
	guardResourceActivationID  = newGuardResourceActivationID()
)

func newGuardResourceActivationID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("pid-%d-%d", os.Getpid(), time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}
func newGuardResourceReceipt(traceID, agent string, rootPID int, d guardResourceDecision) guardResourceReceipt {
	identity := guardResourceBuildIdentity()
	receipt := guardResourceReceipt{Schema: "fak.guard.child-resource.v1", At: time.Now().UTC().Format(time.RFC3339Nano), TraceID: traceID, Agent: agent, RootPID: rootPID, OffenderPID: d.Offender.PID, OffenderPPID: d.Offender.PPID, OffenderName: d.Offender.Name, MemoryMetric: string(d.Metric), TreeMemoryBytes: d.TreeBytes, SystemMemoryBytes: d.SystemBytes, SystemMemoryLimit: d.SystemLimit, ThresholdBytes: d.ThresholdBytes, HeadroomBytes: d.HeadroomBytes, Reason: d.Reason, Action: "reap_tree", DescendantsSurvive: false, Detail: scrubGuardResourceDetail(d.Detail), BuildCommit: identity.Commit, BuildModule: identity.ModuleVersion, BuildDirty: identity.Dirty, ActivationID: guardResourceActivationID}
	if receipt.BuildModule == "" {
		receipt.BuildModule = "cmd/fak"
	}
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
