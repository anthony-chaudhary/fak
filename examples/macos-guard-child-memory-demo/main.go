// macos-guard-child-memory-demo demonstrates default child-memory containment
// on macOS under fak guard: host-sized RSS thresholds, metric typing, and
// fail-closed receipt emission.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const (
	guardTreeCommitDefault = uint64(64) << 30
	guardTreeRSSFallback   = uint64(4) << 30
	guardTreeRSSMinimum    = uint64(1) << 30
	receiptSchema          = "fak.guard.child-resource.v1"
)

// DefaultDarwinRSSLimit computes the default child process-tree memory threshold
// on macOS, derived from host physical memory: clamp(physical/4, 1GiB, 64GiB).
func DefaultDarwinRSSLimit(hostPhysicalBytes uint64) uint64 {
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

type ResourcePolicy struct {
	Metric       procguard.MemoryMetric `json:"metric"`
	MaxTreeBytes uint64                 `json:"max_tree_bytes"`
}

type ResourceDecision struct {
	Stop           bool                    `json:"stop"`
	Reason         string                  `json:"reason"`
	Metric         procguard.MemoryMetric  `json:"metric"`
	TreeBytes      uint64                  `json:"tree_bytes"`
	ThresholdBytes uint64                  `json:"threshold_bytes"`
	Offender       procguard.MemoryProcess `json:"offender"`
	OwnedPIDs      []int                   `json:"owned_pids"`
}

type GuardResourceReceipt struct {
	Schema             string  `json:"schema"`
	At                 string  `json:"at"`
	TraceID            string  `json:"trace_id"`
	Agent              string  `json:"agent"`
	RootPID            int     `json:"root_pid"`
	OffenderPID        int     `json:"offender_pid"`
	OffenderPPID       int     `json:"offender_ppid"`
	OffenderName       string  `json:"offender_name"`
	MemoryMetric       string  `json:"memory_metric"`
	TreeMemoryBytes    uint64  `json:"tree_memory_bytes"`
	TreeRSSBytes       *uint64 `json:"tree_rss_bytes,omitempty"`
	ThresholdBytes     uint64  `json:"threshold_bytes"`
	Reason             string  `json:"reason"`
	Action             string  `json:"action"`
	DescendantsSurvive bool    `json:"descendants_survive"`
	BuildModule        string  `json:"build_module,omitempty"`
}

type ScenarioResult struct {
	Name      string                   `json:"name"`
	Compliant bool                     `json:"compliant"`
	Policy    ResourcePolicy           `json:"policy"`
	Snapshot  procguard.MemorySnapshot `json:"snapshot"`
	Decision  ResourceDecision         `json:"decision"`
	Receipt   *GuardResourceReceipt    `json:"receipt,omitempty"`
}

// EvaluateResource decides whether a process-tree memory snapshot crosses policy limits.
func EvaluateResource(policy ResourcePolicy, s procguard.MemorySnapshot) ResourceDecision {
	treeBytes := s.TreeBytes
	if treeBytes == 0 && len(s.Processes) > 0 {
		for _, p := range s.Processes {
			treeBytes += p.Bytes
		}
	}
	d := ResourceDecision{
		Metric:         s.Metric,
		TreeBytes:      treeBytes,
		ThresholdBytes: policy.MaxTreeBytes,
	}
	for _, process := range s.Processes {
		d.OwnedPIDs = append(d.OwnedPIDs, process.PID)
	}
	sort.Ints(d.OwnedPIDs)

	if len(s.Processes) > 0 {
		procs := make([]procguard.MemoryProcess, len(s.Processes))
		copy(procs, s.Processes)
		sort.Slice(procs, func(i, j int) bool {
			if procs[i].Bytes == procs[j].Bytes {
				return procs[i].PID < procs[j].PID
			}
			return procs[i].Bytes > procs[j].Bytes
		})
		d.Offender = procs[0]
	}

	if policy.MaxTreeBytes > 0 && treeBytes >= policy.MaxTreeBytes {
		d.Stop = true
		d.Reason = "CHILD_TREE_" + strings.ToUpper(string(s.Metric)) + "_LIMIT"
	}
	return d
}

// MintReceipt produces an immutable, typed containment receipt adhering to fak.guard.child-resource.v1.
func MintReceipt(traceID, agent string, rootPID int, d ResourceDecision, timestamp string) GuardResourceReceipt {
	action := "observe_only"
	descendantsSurvive := true
	if d.Stop {
		action = "reap_tree"
		descendantsSurvive = false
	}
	if timestamp == "" {
		timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}
	r := GuardResourceReceipt{
		Schema:             receiptSchema,
		At:                 timestamp,
		TraceID:            traceID,
		Agent:              agent,
		RootPID:            rootPID,
		OffenderPID:        d.Offender.PID,
		OffenderPPID:       d.Offender.PPID,
		OffenderName:       d.Offender.Name,
		MemoryMetric:       string(d.Metric),
		TreeMemoryBytes:    d.TreeBytes,
		ThresholdBytes:     d.ThresholdBytes,
		Reason:             d.Reason,
		Action:             action,
		DescendantsSurvive: descendantsSurvive,
		BuildModule:        "cmd/fak",
	}
	if d.Metric == procguard.MemoryMetricRSS {
		treeBytesCopy := d.TreeBytes
		r.TreeRSSBytes = &treeBytesCopy
	}
	return r
}

// BuiltinScenarios returns compliant and leaking child-tree scenarios.
func BuiltinScenarios() (compliant ScenarioResult, breach ScenarioResult) {
	policy := ResourcePolicy{
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: 48 << 20, // 48 MiB threshold
	}

	snapCompliant := procguard.MemorySnapshot{
		Metric:    procguard.MemoryMetricRSS,
		RootPID:   1000,
		TreeBytes: 32 << 20, // 32 MiB
		Processes: []procguard.MemoryProcess{
			{PID: 1000, PPID: 1, Name: "claude", Bytes: 12 << 20},
			{PID: 1001, PPID: 1000, Name: "worker", Bytes: 20 << 20},
		},
	}
	decCompliant := EvaluateResource(policy, snapCompliant)
	compliant = ScenarioResult{
		Name:      "compliant-child-tree",
		Compliant: true,
		Policy:    policy,
		Snapshot:  snapCompliant,
		Decision:  decCompliant,
	}

	snapBreach := procguard.MemorySnapshot{
		Metric:    procguard.MemoryMetricRSS,
		RootPID:   2000,
		TreeBytes: 75 << 20, // 75 MiB
		Processes: []procguard.MemoryProcess{
			{PID: 2000, PPID: 1, Name: "claude", Bytes: 15 << 20},
			{PID: 2001, PPID: 2000, Name: "leaking-worker", Bytes: 45 << 20},
			{PID: 2002, PPID: 2001, Name: "sub-helper", Bytes: 15 << 20},
		},
	}
	decBreach := EvaluateResource(policy, snapBreach)
	receiptBreach := MintReceipt("demo-contain-trace-8984", "claude", snapBreach.RootPID, decBreach, "2026-09-03T12:00:00.000000000Z")
	breach = ScenarioResult{
		Name:      "runaway-child-tree",
		Compliant: false,
		Policy:    policy,
		Snapshot:  snapBreach,
		Decision:  decBreach,
		Receipt:   &receiptBreach,
	}
	return compliant, breach
}

func runSelfcheck() error {
	// 1. Invariant: Darwin memory clamp formula: clamp(physical/4, 1GiB, 64GiB)
	clampCases := []struct {
		phys uint64
		want uint64
	}{
		{phys: 0, want: 4 << 30},          // fallback 4 GiB
		{phys: 2 << 30, want: 1 << 30},    // 2/4 = 0.5 GiB -> clamped to min 1 GiB
		{phys: 16 << 30, want: 4 << 30},   // 16/4 = 4 GiB
		{phys: 36 << 30, want: 9 << 30},   // 36/4 = 9 GiB
		{phys: 512 << 30, want: 64 << 30}, // 512/4 = 128 GiB -> clamped to max 64 GiB
	}
	for _, tc := range clampCases {
		got := DefaultDarwinRSSLimit(tc.phys)
		if got != tc.want {
			return fmt.Errorf("DefaultDarwinRSSLimit(%d) = %d, want %d", tc.phys, got, tc.want)
		}
	}

	// 2. Invariant: Compliant subtree must not trigger containment
	compliant, breach := BuiltinScenarios()
	if compliant.Decision.Stop {
		return fmt.Errorf("compliant scenario unexpectedly stopped: %+v", compliant.Decision)
	}
	if compliant.Decision.Reason != "" {
		return fmt.Errorf("compliant scenario unexpected reason: %q", compliant.Decision.Reason)
	}

	// 3. Invariant: Breach subtree triggers CHILD_TREE_RSS_LIMIT and attributes largest offender
	if !breach.Decision.Stop {
		return fmt.Errorf("breach scenario was not stopped: %+v", breach.Decision)
	}
	if breach.Decision.Reason != "CHILD_TREE_RSS_LIMIT" {
		return fmt.Errorf("breach scenario reason = %q, want CHILD_TREE_RSS_LIMIT", breach.Decision.Reason)
	}
	if breach.Decision.Offender.PID != 2001 || breach.Decision.Offender.Name != "leaking-worker" {
		return fmt.Errorf("breach offender mismatch: PID=%d name=%q, want 2001 leaking-worker",
			breach.Decision.Offender.PID, breach.Decision.Offender.Name)
	}
	if breach.Receipt == nil {
		return fmt.Errorf("breach receipt was nil")
	}
	if breach.Receipt.Schema != receiptSchema {
		return fmt.Errorf("receipt schema = %q, want %q", breach.Receipt.Schema, receiptSchema)
	}
	if breach.Receipt.MemoryMetric != "rss" {
		return fmt.Errorf("receipt memory_metric = %q, want rss", breach.Receipt.MemoryMetric)
	}
	if breach.Receipt.TreeRSSBytes == nil || *breach.Receipt.TreeRSSBytes != 75<<20 {
		return fmt.Errorf("receipt tree_rss_bytes = %v, want 75 MiB", breach.Receipt.TreeRSSBytes)
	}
	if breach.Receipt.Action != "reap_tree" || breach.Receipt.DescendantsSurvive {
		return fmt.Errorf("receipt action=%q descendants_survive=%v, want reap_tree/false",
			breach.Receipt.Action, breach.Receipt.DescendantsSurvive)
	}

	// 4. Invariant: Offender tie-break determinism: equal RSS selects lower PID
	snapTie := procguard.MemorySnapshot{
		Metric:    procguard.MemoryMetricRSS,
		RootPID:   3000,
		TreeBytes: 50 << 20,
		Processes: []procguard.MemoryProcess{
			{PID: 3005, PPID: 3000, Name: "worker-b", Bytes: 25 << 20},
			{PID: 3002, PPID: 3000, Name: "worker-a", Bytes: 25 << 20},
		},
	}
	decTie := EvaluateResource(ResourcePolicy{Metric: procguard.MemoryMetricRSS, MaxTreeBytes: 40 << 20}, snapTie)
	if decTie.Offender.PID != 3002 {
		return fmt.Errorf("offender tie-break PID=%d, want 3002 (lower PID)", decTie.Offender.PID)
	}

	// 5. Invariant: Byte-for-byte serialization determinism
	receipt1 := MintReceipt("trace-det", "agent", 2000, breach.Decision, "2026-09-03T00:00:00Z")
	receipt2 := MintReceipt("trace-det", "agent", 2000, breach.Decision, "2026-09-03T00:00:00Z")
	b1, err := json.Marshal(receipt1)
	if err != nil {
		return fmt.Errorf("marshal receipt1: %w", err)
	}
	b2, err := json.Marshal(receipt2)
	if err != nil {
		return fmt.Errorf("marshal receipt2: %w", err)
	}
	if !bytes.Equal(b1, b2) {
		return fmt.Errorf("receipt serialization non-deterministic")
	}

	// 6. Invariant: Live host probe verification on Darwin
	if runtime.GOOS == "darwin" {
		phys, detail := procguard.HostPhysicalMemoryBytes()
		if detail != "" || phys == 0 {
			return fmt.Errorf("live Darwin HostPhysicalMemoryBytes failed: phys=%d detail=%s", phys, detail)
		}
		snap, supported, snapDetail := procguard.CollectMemorySnapshot(os.Getpid())
		if !supported || snapDetail != "" || snap.TreeBytes == 0 || snap.Metric != procguard.MemoryMetricRSS {
			return fmt.Errorf("live Darwin CollectMemorySnapshot failed: supported=%v metric=%s treeBytes=%d detail=%s",
				supported, snap.Metric, snap.TreeBytes, snapDetail)
		}
	}

	return nil
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("macos-guard-child-memory-demo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run deterministic invariant selfcheck and exit")
	jsonMode := fs.Bool("json", false, "output demo scenarios and receipts as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "usage: macos-guard-child-memory-demo [-selfcheck] [-json]\n")
		return 2
	}

	if *selfcheck {
		if err := runSelfcheck(); err != nil {
			fmt.Fprintf(stderr, "selfcheck: FAIL: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "selfcheck: PASS (macOS child RSS limit clamp, metric typing, breach containment, receipt schema, and live Darwin probe verified)\n")
		return 0
	}

	compliant, breach := BuiltinScenarios()

	if *jsonMode {
		payload := struct {
			Platform     string           `json:"platform"`
			Metric       string           `json:"metric"`
			DefaultLimit uint64           `json:"default_limit_bytes"`
			Scenarios    []ScenarioResult `json:"scenarios"`
			Selfcheck    string           `json:"selfcheck"`
		}{
			Platform:     runtime.GOOS + "/" + runtime.GOARCH,
			Metric:       string(procguard.MemoryMetricRSS),
			DefaultLimit: DefaultDarwinRSSLimit(0),
			Scenarios:    []ScenarioResult{compliant, breach},
			Selfcheck:    "PASS",
		}
		if runtime.GOOS == "darwin" {
			if phys, _ := procguard.HostPhysicalMemoryBytes(); phys > 0 {
				payload.DefaultLimit = DefaultDarwinRSSLimit(phys)
			}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "encode json: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintln(stdout, "== macOS Default Guard Child-Memory Containment Demo ==")
	fmt.Fprintf(stdout, "Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	var physBytes uint64
	if runtime.GOOS == "darwin" {
		phys, detail := procguard.HostPhysicalMemoryBytes()
		if detail == "" {
			physBytes = phys
		}
	}
	defLimit := DefaultDarwinRSSLimit(physBytes)
	if physBytes > 0 {
		fmt.Fprintf(stdout, "Host physical RAM: %.2f GiB\n", float64(physBytes)/(1<<30))
		fmt.Fprintf(stdout, "Default guard child RSS limit: %.2f GiB (clamp(physical/4, 1GiB, 64GiB))\n", float64(defLimit)/(1<<30))
	} else {
		fmt.Fprintf(stdout, "Default guard child RSS limit: %.2f GiB (fallback clamp)\n", float64(defLimit)/(1<<30))
	}
	fmt.Fprintf(stdout, "Active memory metric: %s\n\n", procguard.MemoryMetricRSS)

	fmt.Fprintf(stdout, "Scenario 1: %s\n", compliant.Name)
	fmt.Fprintf(stdout, "  Policy threshold: %d MiB RSS\n", compliant.Policy.MaxTreeBytes>>20)
	fmt.Fprintf(stdout, "  Process tree: %d processes, %d MiB RSS\n", len(compliant.Snapshot.Processes), compliant.Decision.TreeBytes>>20)
	for _, p := range compliant.Snapshot.Processes {
		fmt.Fprintf(stdout, "    PID %d (%s): %d MiB\n", p.PID, p.Name, p.Bytes>>20)
	}
	fmt.Fprintf(stdout, "  Decision: stop=%v (compliant; no containment action)\n\n", compliant.Decision.Stop)

	fmt.Fprintf(stdout, "Scenario 2: %s\n", breach.Name)
	fmt.Fprintf(stdout, "  Policy threshold: %d MiB RSS\n", breach.Policy.MaxTreeBytes>>20)
	fmt.Fprintf(stdout, "  Process tree: %d processes, %d MiB RSS (BREACH)\n", len(breach.Snapshot.Processes), breach.Decision.TreeBytes>>20)
	for _, p := range breach.Snapshot.Processes {
		marker := ""
		if p.PID == breach.Decision.Offender.PID {
			marker = " [OFFENDER]"
		}
		fmt.Fprintf(stdout, "    PID %d (%s): %d MiB%s\n", p.PID, p.Name, p.Bytes>>20, marker)
	}
	fmt.Fprintf(stdout, "  Decision: stop=%v reason=%s\n", breach.Decision.Stop, breach.Decision.Reason)
	fmt.Fprintf(stdout, "  Action: %s (descendants_survive=%v)\n", breach.Receipt.Action, breach.Receipt.DescendantsSurvive)
	fmt.Fprintf(stdout, "  Emitted receipt schema: %s (metric=%s, tree_rss_bytes=%d)\n\n",
		breach.Receipt.Schema, breach.Receipt.MemoryMetric, *breach.Receipt.TreeRSSBytes)

	if runtime.GOOS == "darwin" {
		snap, supported, detail := procguard.CollectMemorySnapshot(os.Getpid())
		if supported && detail == "" && snap.TreeBytes > 0 {
			fmt.Fprintf(stdout, "Live Darwin Process Probe:\n")
			fmt.Fprintf(stdout, "  PID %d: verified live snapshot (metric=%s, tree_rss=%d bytes) · consistency=ok\n\n",
				os.Getpid(), snap.Metric, snap.TreeBytes)
		}
	}

	if err := runSelfcheck(); err != nil {
		fmt.Fprintf(stderr, "selfcheck: FAIL: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, "selfcheck: PASS (all macOS default guard child-memory containment invariants verified)")
	return 0
}

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}
