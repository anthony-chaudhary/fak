package sessiondiag

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/stringset"
)

const InventorySchema = "fak.sessiondiag.inventory.v1"

const (
	KindInteractiveTUI      = "interactive_tui"
	KindGuardedTUI          = "fak_guarded_tui"
	KindHeadlessExec        = "headless_exec_worker"
	KindResumeWrapper       = "resume_wrapper"
	KindSpawnedSubagent     = "spawned_subagent"
	KindGatewayServed       = "gateway_served_session"
	KindUnknown             = "unknown"
	HealthActive            = "ACTIVE"
	HealthFailedButLocked   = "FAILED_BUT_LOCKED"
	HealthOrphanProcess     = "ORPHAN_PROCESS"
	HealthStaleLock         = "STALE_LOCK"
	HealthReceiptOnly       = "RECEIPT_ONLY"
	HealthCompleted         = "COMPLETED"
	HealthUnknown           = "UNKNOWN"
	EndpointActive          = "active"
	EndpointStale           = "stale"
	EndpointCompleted       = "completed"
	EndpointUnknown         = "unknown"
	ReasonTurnInProgress    = "TURN_IN_PROGRESS_WITH_PROCESS"
	ReasonRecentThread      = "RECENT_THREAD_UPDATE_WITH_PROCESS_AND_LOCK"
	ReasonInteractiveWait   = "INTERACTIVE_PROCESS_WAITING_AFTER_COMPLETED_TURN"
	ReasonFailedLocked      = "FAILED_TURN_WITH_WRITER_LOCK"
	ReasonFailedProcess     = "FAILED_TURN_WITH_LIVE_PROCESS_TREE"
	ReasonTerminalProcess   = "TERMINAL_TURN_WITH_PROCESS_TREE"
	ReasonTerminalLock      = "TERMINAL_TURN_WITH_WRITER_LOCK"
	ReasonStaleLockAge      = "WRITER_LOCK_OLDER_THAN_STALE_THRESHOLD"
	ReasonLockNoProcess     = "WRITER_LOCK_WITHOUT_PROCESS_TREE"
	ReasonProcessNoLock     = "PROCESS_TREE_WITHOUT_WRITER_LOCK"
	ReasonProcessNoThread   = "PROCESS_TREE_WITHOUT_THREAD"
	ReasonReceiptNoProcess  = "LAUNCH_RECEIPT_WITHOUT_PROCESS_OR_LOCK"
	ReasonTerminalTurn      = "TERMINAL_TURN_WITHOUT_CURRENT_SIGNALS"
	ReasonNoCurrentEvidence = "NO_CORROBORATED_CURRENT_EVIDENCE"
	ReasonPIDReuse          = "PID_REUSE_PARENT_STARTS_AFTER_CHILD"
	ReasonMissingCommand    = "PROCESS_COMMAND_LINE_UNAVAILABLE"
	ReasonExplicitThread    = "PROCESS_THREAD_EXPLICIT_COMMAND_HINT"
	ReasonStartMatched      = "PROCESS_THREAD_START_TIME_MATCH"
	ReasonAmbiguousHints    = "PROCESS_MULTIPLE_THREAD_HINTS"
)

var threadIDPattern = regexp.MustCompile(`(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`)
var safeErrorTokenPattern = regexp.MustCompile(`^[A-Za-z0-9_.:-]{1,80}$`)

type ThreadEvidence struct {
	ThreadID      string
	Source        string
	ThreadSource  string
	CreatedAt     time.Time
	UpdatedAt     time.Time
	Archived      bool
	AgentNickname string
	AgentRole     string
	CWD           string
}

type TurnEvidence struct {
	ThreadID       string
	TurnID         string
	Status         string
	ErrorJSON      string
	StartedAt      time.Time
	CompletedAt    time.Time
	DurationMS     int64
	RolloutOrdinal int64
}

type WriterLockEvidence struct {
	ThreadID   string
	ModifiedAt time.Time
}

type GuardReceiptEvidence struct {
	ThreadID   string
	RecordedAt time.Time
}

type ProcessEvidence struct {
	PID         int
	ParentPID   int
	Name        string
	CommandLine string
	StartedAt   time.Time
	AgeSeconds  int64
}

type SpawnEdgeEvidence struct {
	ParentThreadID string
	ChildThreadID  string
	Status         string
}

type SourceError struct {
	Source string `json:"source"`
	Code   string `json:"code"`
}

type InventoryInput struct {
	Threads            []ThreadEvidence
	Turns              []TurnEvidence
	WriterLocks        []WriterLockEvidence
	GuardReceipts      []GuardReceiptEvidence
	Processes          []ProcessEvidence
	SpawnEdges         []SpawnEdgeEvidence
	Registrations      []sessionregistry.Record
	SourceErrors       []SourceError
	Window             time.Duration
	StaleAfter         time.Duration
	ProcessMatchWindow time.Duration
}

type InventorySources struct {
	ThreadMetadata int           `json:"thread_metadata"`
	Turns          int           `json:"turns"`
	WriterLocks    int           `json:"writer_locks"`
	GuardReceipts  int           `json:"guard_launch_receipts"`
	Processes      int           `json:"processes"`
	ProcessTrees   int           `json:"process_trees"`
	SpawnEdges     int           `json:"spawn_edges"`
	Registrations  int           `json:"registrations"`
	Errors         []SourceError `json:"errors,omitempty"`
}

type InventoryCounts struct {
	Total         int                    `json:"total"`
	Active        int                    `json:"active"`
	ByKind        map[string]int         `json:"by_kind"`
	ByHealth      map[string]int         `json:"by_health"`
	ProcessTrees  int                    `json:"process_trees"`
	SpawnEdges    int                    `json:"spawn_edges"`
	Registrations sessionregistry.Counts `json:"registrations"`
}

type CleanupAction struct {
	Artifact       string `json:"artifact"`
	Identity       string `json:"identity"`
	Action         string `json:"action"`
	Reason         string `json:"reason"`
	RegistrationID string `json:"registration_id,omitempty"`
}

type ThreadRecord struct {
	ID            string `json:"id"`
	Source        string `json:"source,omitempty"`
	ThreadSource  string `json:"thread_source,omitempty"`
	CreatedAt     string `json:"created_at,omitempty"`
	UpdatedAt     string `json:"updated_at,omitempty"`
	Archived      bool   `json:"archived,omitempty"`
	AgentNickname string `json:"agent_nickname,omitempty"`
	AgentRole     string `json:"agent_role,omitempty"`
	CWD           string `json:"cwd,omitempty"`
}

type TurnRecord struct {
	ID           string `json:"id,omitempty"`
	Status       string `json:"status,omitempty"`
	ErrorPresent bool   `json:"error_present"`
	ErrorCode    string `json:"error_code,omitempty"`
	StartedAt    string `json:"started_at,omitempty"`
	CompletedAt  string `json:"completed_at,omitempty"`
	DurationMS   int64  `json:"duration_ms,omitempty"`
}

type WriterLockSignal struct {
	EvidenceType   string `json:"evidence_type"`
	ModifiedAt     string `json:"modified_at"`
	AgeSeconds     int64  `json:"age_seconds"`
	ProvesLiveness bool   `json:"proves_liveness"`
}

type GuardReceiptSignal struct {
	EvidenceType   string `json:"evidence_type"`
	RecordedAt     string `json:"recorded_at"`
	AgeSeconds     int64  `json:"age_seconds"`
	ProvesLiveness bool   `json:"proves_liveness"`
}

type ProcessNode struct {
	PID                  int    `json:"pid"`
	ParentPID            int    `json:"parent_pid,omitempty"`
	Name                 string `json:"name"`
	Role                 string `json:"role"`
	StartedAt            string `json:"started_at,omitempty"`
	CommandLineAvailable bool   `json:"command_line_available"`
}

type ProcessTree struct {
	ID                   string        `json:"id"`
	RootPID              int           `json:"root_pid"`
	CodexPID             int           `json:"codex_pid,omitempty"`
	StartedAt            string        `json:"started_at,omitempty"`
	CodexStartedAt       string        `json:"codex_started_at,omitempty"`
	Nodes                []ProcessNode `json:"nodes"`
	HasCodex             bool          `json:"has_codex"`
	HasGuard             bool          `json:"has_fak_guard"`
	HasGateway           bool          `json:"has_fak_gateway"`
	HasExec              bool          `json:"has_exec"`
	HasResume            bool          `json:"has_resume"`
	CommandLinesComplete bool          `json:"command_lines_complete"`
	Association          string        `json:"association"`
	ThreadID             string        `json:"thread_id,omitempty"`
	Reasons              []string      `json:"reasons,omitempty"`
	ProvesLiveness       bool          `json:"proves_liveness"`
}

type SessionRecord struct {
	RecordID      string              `json:"record_id"`
	Harness       string              `json:"harness,omitempty"`
	HarnessSource string              `json:"harness_source,omitempty"`
	Thread        *ThreadRecord       `json:"thread,omitempty"`
	LatestTurn    *TurnRecord         `json:"latest_turn,omitempty"`
	WriterLock    *WriterLockSignal   `json:"writer_lock,omitempty"`
	GuardReceipt  *GuardReceiptSignal `json:"guard_launch_receipt,omitempty"`
	ProcessTrees  []ProcessTree       `json:"process_trees,omitempty"`
	ParentThreads []string            `json:"parent_threads,omitempty"`
	ChildThreads  []string            `json:"child_threads,omitempty"`
	Kind          string              `json:"kind"`
	Health        string              `json:"health"`
	Reasons       []string            `json:"reasons"`
}

type SpawnEndpoint struct {
	ThreadID string `json:"thread_id"`
	State    string `json:"state"`
}

type SpawnEdgeRecord struct {
	Parent SpawnEndpoint `json:"parent"`
	Child  SpawnEndpoint `json:"child"`
	Status string        `json:"status"`
}

type RegistrationRecord struct {
	RegistrationID       string                `json:"registration_id"`
	ParentRegistrationID string                `json:"parent_registration_id,omitempty"`
	ParentAttemptID      string                `json:"parent_attempt_id,omitempty"`
	RootRegistrationID   string                `json:"root_registration_id"`
	RootOutcome          string                `json:"root_outcome,omitempty"`
	RootIssue            string                `json:"root_issue,omitempty"`
	TaskID               string                `json:"task_id,omitempty"`
	AttemptID            string                `json:"attempt_id"`
	ResumeOfAttemptID    string                `json:"resume_of_attempt_id,omitempty"`
	LaunchKind           string                `json:"launch_kind"`
	Lane                 string                `json:"lane,omitempty"`
	LeaseID              string                `json:"lease_id,omitempty"`
	Runtime              string                `json:"runtime"`
	SessionID            string                `json:"session_id,omitempty"`
	ThreadID             string                `json:"thread_id,omitempty"`
	PID                  int                   `json:"pid,omitempty"`
	ProcessStartedAt     time.Time             `json:"process_started_at,omitempty"`
	State                sessionregistry.State `json:"state"`
	Reason               string                `json:"reason,omitempty"`
	WitnessRef           string                `json:"witness_ref,omitempty"`
	ProcessMatched       bool                  `json:"process_matched"`
	Health               string                `json:"health"`
}
type InventoryReport struct {
	Schema                     string                                 `json:"schema"`
	ObservedAt                 string                                 `json:"observed_at"`
	WindowSeconds              int64                                  `json:"window_seconds"`
	StaleAfterSeconds          int64                                  `json:"stale_after_seconds"`
	ReadOnly                   bool                                   `json:"read_only"`
	Sources                    InventorySources                       `json:"sources"`
	Counts                     InventoryCounts                        `json:"counts"`
	Sessions                   []SessionRecord                        `json:"sessions"`
	SpawnEdges                 []SpawnEdgeRecord                      `json:"spawn_edges"`
	Registrations              []RegistrationRecord                   `json:"registrations"`
	UnregisteredObserved       []sessionregistry.UnregisteredObserved `json:"unregistered_observed,omitempty"`
	RegistrationReconciliation []sessionregistry.Reconciliation       `json:"registration_reconciliation,omitempty"`
	CleanupActions             []CleanupAction                        `json:"cleanup_actions,omitempty"`
	Notice                     string                                 `json:"notice"`
}

type processTreeWork struct {
	report ProcessTree
	raw    []ProcessEvidence
}

func ReconcileInventory(in InventoryInput, now time.Time) InventoryReport {
	now = now.UTC()
	if in.Window <= 0 {
		in.Window = 24 * time.Hour
	}
	if in.StaleAfter <= 0 {
		in.StaleAfter = 10 * time.Minute
	}
	if in.ProcessMatchWindow <= 0 {
		in.ProcessMatchWindow = 3 * time.Second
	}
	indexed := indexInventoryEvidence(in)
	threads, turns := indexed.threads, indexed.turns
	harnessByThread, locks, receipts := indexed.harnessByThread, indexed.locks, indexed.receipts
	parentIDs, childIDs, childSet := indexed.parentIDs, indexed.childIDs, indexed.childSet

	trees := buildProcessTrees(in.Processes, threads)
	associateProcessTrees(trees, threads, in.ProcessMatchWindow)
	treesByThread := map[string][]ProcessTree{}
	var orphanTrees []ProcessTree
	for _, tree := range trees {
		if tree.report.ThreadID == "" {
			orphanTrees = append(orphanTrees, tree.report)
			continue
		}
		treesByThread[tree.report.ThreadID] = append(treesByThread[tree.report.ThreadID], tree.report)
	}

	cutoff := now.Add(-in.Window)
	ids := map[string]bool{}
	for id, t := range threads {
		if t.UpdatedAt.IsZero() || !t.UpdatedAt.Before(cutoff) {
			ids[id] = true
		}
	}
	for id := range locks {
		ids[id] = true
	}
	for id, receipt := range receipts {
		if receipt.RecordedAt.IsZero() || !receipt.RecordedAt.Before(cutoff) {
			ids[id] = true
		}
	}
	for id := range treesByThread {
		ids[id] = true
	}
	for child := range childSet {
		if t, ok := threads[child]; ok && (t.UpdatedAt.IsZero() || !t.UpdatedAt.Before(cutoff)) {
			ids[child] = true
		}
	}

	sessions := make([]SessionRecord, 0, len(ids)+len(orphanTrees))
	sessionByThread := map[string]SessionRecord{}
	for id := range ids {
		thread, hasThread := threads[id]
		turn, hasTurn := turns[id]
		lock, hasLock := locks[id]
		receipt, hasReceipt := receipts[id]
		record := buildSessionRecord(id, thread, hasThread, turn, hasTurn, lock, hasLock, receipt, hasReceipt, treesByThread[id], childSet[id], parentIDs[id], childIDs[id], now, in.StaleAfter)
		record.Harness = "codex"
		record.HarnessSource = "legacy_codex_inventory"
		if harness := harnessByThread[id]; harness != "" {
			record.Harness = harness
			record.HarnessSource = "session_registration"
		}
		sessions = append(sessions, record)
		sessionByThread[id] = record
	}
	for _, tree := range orphanTrees {
		kind := KindUnknown
		if tree.HasGateway {
			kind = KindGatewayServed
		} else if tree.HasResume {
			kind = KindResumeWrapper
		} else if tree.HasExec {
			kind = KindHeadlessExec
		}
		reasons := []string{ReasonProcessNoThread}
		reasons = append(reasons, tree.Reasons...)
		sessions = append(sessions, SessionRecord{
			RecordID:     tree.ID,
			ProcessTrees: []ProcessTree{tree},
			Kind:         kind,
			Health:       HealthOrphanProcess,
			Reasons:      uniqueSorted(reasons),
		})
	}

	edges := make([]SpawnEdgeRecord, 0, len(in.SpawnEdges))
	for _, edge := range in.SpawnEdges {
		parent, child := safeID(edge.ParentThreadID), safeID(edge.ChildThreadID)
		if parent == "" || child == "" {
			continue
		}
		edges = append(edges, SpawnEdgeRecord{
			Parent: SpawnEndpoint{ThreadID: parent, State: endpointState(parent, edge.Status, threads, turns, sessionByThread, now, in.Window)},
			Child:  SpawnEndpoint{ThreadID: child, State: endpointState(child, edge.Status, threads, turns, sessionByThread, now, in.Window)},
			Status: strings.ToLower(strings.TrimSpace(edge.Status)),
		})
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Parent.ThreadID != edges[j].Parent.ThreadID {
			return edges[i].Parent.ThreadID < edges[j].Parent.ThreadID
		}
		return edges[i].Child.ThreadID < edges[j].Child.ThreadID
	})
	sortSessions(sessions)

	registrationRecords, unregistered := inventoryRegistrations(in.Registrations, in.Processes)
	observedRegistrable := observedRegistrationProcesses(in.Processes)
	reconciliations := sessionregistry.ReconcileStale(in.Registrations, observedRegistrable, now, in.StaleAfter)
	registrationCounts := sessionregistry.Summarize(in.Registrations, len(unregistered))
	cleanupActions := buildCleanupActions(sessions, edges, reconciliations)

	counts := InventoryCounts{
		ByKind:        map[string]int{},
		ByHealth:      map[string]int{},
		ProcessTrees:  len(trees),
		SpawnEdges:    len(edges),
		Registrations: registrationCounts,
	}
	for _, session := range sessions {
		counts.Total++
		counts.ByKind[session.Kind]++
		counts.ByHealth[session.Health]++
		if session.Health == HealthActive {
			counts.Active++
		}
	}
	sourceErrors := append([]SourceError(nil), in.SourceErrors...)
	sort.Slice(sourceErrors, func(i, j int) bool {
		if sourceErrors[i].Source != sourceErrors[j].Source {
			return sourceErrors[i].Source < sourceErrors[j].Source
		}
		return sourceErrors[i].Code < sourceErrors[j].Code
	})
	return InventoryReport{
		Schema:            InventorySchema,
		ObservedAt:        stamp(now),
		WindowSeconds:     int64(in.Window.Seconds()),
		StaleAfterSeconds: int64(in.StaleAfter.Seconds()),
		ReadOnly:          true,
		Sources: InventorySources{
			ThreadMetadata: len(in.Threads),
			Turns:          len(in.Turns),
			WriterLocks:    len(in.WriterLocks),
			GuardReceipts:  len(in.GuardReceipts),
			Processes:      len(in.Processes),
			ProcessTrees:   len(trees),
			SpawnEdges:     len(edges),
			Registrations:  len(in.Registrations),
			Errors:         sourceErrors,
		},
		Counts:                     counts,
		Sessions:                   sessions,
		SpawnEdges:                 edges,
		Registrations:              registrationRecords,
		UnregisteredObserved:       unregistered,
		RegistrationReconciliation: reconciliations,
		CleanupActions:             cleanupActions,
		Notice:                     "guard receipts are launch receipts; writer locks and OS processes are presence signals; none is treated as liveness by itself",
	}
}

type inventoryEvidenceIndex struct {
	threads         map[string]ThreadEvidence
	turns           map[string]TurnEvidence
	harnessByThread map[string]string
	locks           map[string]WriterLockEvidence
	receipts        map[string]GuardReceiptEvidence
	parentIDs       map[string][]string
	childIDs        map[string][]string
	childSet        map[string]bool
}

func indexInventoryEvidence(in InventoryInput) inventoryEvidenceIndex {
	threads := make(map[string]ThreadEvidence, len(in.Threads))
	for _, t := range in.Threads {
		if id := safeID(t.ThreadID); id != "" {
			t.ThreadID = id
			threads[id] = t
		}
	}
	turns := latestTurns(in.Turns)
	harnessByThread := map[string]string{}
	for _, registration := range in.Registrations {
		harness := normalizedSessionHarness(registration.Identity.Runtime)
		if harness == "" {
			continue
		}
		if id := safeID(registration.Identity.ThreadID); id != "" {
			harnessByThread[id] = harness
		}
		if id := safeID(registration.Identity.SessionID); id != "" {
			harnessByThread[id] = harness
		}
	}
	locks := make(map[string]WriterLockEvidence, len(in.WriterLocks))
	for _, lock := range in.WriterLocks {
		if id := safeID(lock.ThreadID); id != "" {
			lock.ThreadID = id
			locks[id] = lock
		}
	}
	receipts := make(map[string]GuardReceiptEvidence, len(in.GuardReceipts))
	for _, receipt := range in.GuardReceipts {
		if id := safeID(receipt.ThreadID); id != "" {
			receipt.ThreadID = id
			receipts[id] = receipt
		}
	}
	parentIDs := map[string][]string{}
	childIDs := map[string][]string{}
	childSet := map[string]bool{}
	for _, edge := range in.SpawnEdges {
		parent, child := safeID(edge.ParentThreadID), safeID(edge.ChildThreadID)
		if parent == "" || child == "" {
			continue
		}
		childSet[child] = true
		parentIDs[child] = appendUnique(parentIDs[child], parent)
		childIDs[parent] = appendUnique(childIDs[parent], child)
	}
	return inventoryEvidenceIndex{
		threads:         threads,
		turns:           turns,
		harnessByThread: harnessByThread,
		locks:           locks,
		receipts:        receipts,
		parentIDs:       parentIDs,
		childIDs:        childIDs,
		childSet:        childSet,
	}
}

func latestTurns(turns []TurnEvidence) map[string]TurnEvidence {
	out := map[string]TurnEvidence{}
	for _, turn := range turns {
		id := safeID(turn.ThreadID)
		if id == "" {
			continue
		}
		turn.ThreadID = id
		prev, ok := out[id]
		if !ok || turn.RolloutOrdinal > prev.RolloutOrdinal ||
			(turn.RolloutOrdinal == prev.RolloutOrdinal && turn.StartedAt.After(prev.StartedAt)) {
			out[id] = turn
		}
	}
	return out
}

func buildSessionRecord(
	id string,
	thread ThreadEvidence,
	hasThread bool,
	turn TurnEvidence,
	hasTurn bool,
	lock WriterLockEvidence,
	hasLock bool,
	receipt GuardReceiptEvidence,
	hasReceipt bool,
	trees []ProcessTree,
	spawned bool,
	parents, children []string,
	now time.Time,
	staleAfter time.Duration,
) SessionRecord {
	record := SessionRecord{RecordID: "thread:" + id, ParentThreads: sortedCopy(parents), ChildThreads: sortedCopy(children)}
	if hasThread {
		record.Thread = &ThreadRecord{
			ID:            id,
			Source:        normalizedToken(thread.Source),
			ThreadSource:  normalizedToken(thread.ThreadSource),
			CreatedAt:     stamp(thread.CreatedAt),
			UpdatedAt:     stamp(thread.UpdatedAt),
			Archived:      thread.Archived,
			AgentNickname: safeLabel(thread.AgentNickname),
			AgentRole:     safeLabel(thread.AgentRole),
			CWD:           strings.TrimSpace(thread.CWD),
		}
	} else {
		record.Thread = &ThreadRecord{ID: id}
	}
	if hasTurn {
		record.LatestTurn = &TurnRecord{
			ID:           safeID(turn.TurnID),
			Status:       canonicalTurnStatus(turn.Status),
			ErrorPresent: strings.TrimSpace(turn.ErrorJSON) != "",
			ErrorCode:    safeTurnErrorCode(turn.ErrorJSON),
			StartedAt:    stamp(turn.StartedAt),
			CompletedAt:  stamp(turn.CompletedAt),
			DurationMS:   turn.DurationMS,
		}
	}
	if hasLock {
		record.WriterLock = &WriterLockSignal{
			EvidenceType:   "writer_lock",
			ModifiedAt:     stamp(lock.ModifiedAt),
			AgeSeconds:     ageSeconds(now, lock.ModifiedAt),
			ProvesLiveness: false,
		}
	}
	if hasReceipt {
		record.GuardReceipt = &GuardReceiptSignal{
			EvidenceType:   "launch_receipt",
			RecordedAt:     stamp(receipt.RecordedAt),
			AgeSeconds:     ageSeconds(now, receipt.RecordedAt),
			ProvesLiveness: false,
		}
	}
	record.ProcessTrees = append([]ProcessTree(nil), trees...)
	sort.Slice(record.ProcessTrees, func(i, j int) bool { return record.ProcessTrees[i].ID < record.ProcessTrees[j].ID })
	record.Kind = classifyKind(thread, hasThread, spawned, hasReceipt, trees)
	record.Health, record.Reasons = classifyHealth(thread, hasThread, turn, hasTurn, hasLock, lock, hasReceipt, len(trees) > 0, record.Kind, now, staleAfter)
	for _, tree := range trees {
		record.Reasons = append(record.Reasons, tree.Reasons...)
	}
	record.Reasons = uniqueSorted(record.Reasons)
	return record
}

func classifyKind(thread ThreadEvidence, hasThread, spawned, hasReceipt bool, trees []ProcessTree) string {
	if spawned {
		return KindSpawnedSubagent
	}
	for _, tree := range trees {
		if tree.HasResume {
			return KindResumeWrapper
		}
	}
	source := strings.ToLower(strings.TrimSpace(thread.Source))
	for _, tree := range trees {
		if tree.HasGateway && !tree.HasCodex {
			return KindGatewayServed
		}
		if source == "cli" && (tree.HasGuard || hasReceipt) {
			return KindGuardedTUI
		}
		if tree.HasExec {
			return KindHeadlessExec
		}
	}
	switch source {
	case "cli":
		if hasReceipt {
			return KindGuardedTUI
		}
		return KindInteractiveTUI
	case "exec":
		return KindHeadlessExec
	}
	if !hasThread {
		return KindUnknown
	}
	return KindUnknown
}

func classifyHealth(
	thread ThreadEvidence,
	hasThread bool,
	turn TurnEvidence,
	hasTurn bool,
	hasLock bool,
	lock WriterLockEvidence,
	hasReceipt bool,
	hasProcess bool,
	kind string,
	now time.Time,
	staleAfter time.Duration,
) (string, []string) {
	status := strings.ToLower(strings.TrimSpace(turn.Status))
	recentUpdate := hasThread && !thread.UpdatedAt.IsZero() && now.Sub(thread.UpdatedAt) <= staleAfter
	lockStale := hasLock && !lock.ModifiedAt.IsZero() && now.Sub(lock.ModifiedAt) > staleAfter
	switch status {
	case "failed":
		reasons := []string{}
		if hasLock {
			reasons = append(reasons, ReasonFailedLocked)
		}
		if hasProcess {
			reasons = append(reasons, ReasonFailedProcess)
		}
		if hasLock {
			return HealthFailedButLocked, reasons
		}
		if hasProcess {
			return HealthOrphanProcess, append(reasons, ReasonProcessNoLock)
		}
		return HealthCompleted, append(reasons, ReasonTerminalTurn)
	case "interrupted":
		if hasProcess {
			return HealthOrphanProcess, []string{ReasonTerminalProcess}
		}
		if hasLock {
			return HealthStaleLock, []string{ReasonTerminalLock}
		}
		return HealthCompleted, []string{ReasonTerminalTurn}
	case "inprogress", "in_progress", "running":
		if hasProcess && hasLock {
			return HealthActive, []string{ReasonTurnInProgress}
		}
		if hasProcess {
			return HealthOrphanProcess, []string{ReasonProcessNoLock}
		}
		if hasLock && lockStale {
			return HealthStaleLock, []string{ReasonStaleLockAge, ReasonLockNoProcess}
		}
		if hasLock {
			return HealthUnknown, []string{ReasonLockNoProcess}
		}
		return HealthUnknown, []string{ReasonNoCurrentEvidence}
	case "completed":
		if (kind == KindInteractiveTUI || kind == KindGuardedTUI) && hasProcess && hasLock && recentUpdate {
			return HealthActive, []string{ReasonInteractiveWait}
		}
		if hasProcess {
			return HealthOrphanProcess, []string{ReasonTerminalProcess}
		}
		if hasLock {
			return HealthStaleLock, []string{ReasonTerminalLock}
		}
		return HealthCompleted, []string{ReasonTerminalTurn}
	}
	if hasProcess && hasLock && recentUpdate {
		return HealthActive, []string{ReasonRecentThread}
	}
	if hasProcess {
		return HealthOrphanProcess, []string{ReasonProcessNoLock}
	}
	if hasLock {
		if lockStale {
			return HealthStaleLock, []string{ReasonStaleLockAge, ReasonLockNoProcess}
		}
		return HealthUnknown, []string{ReasonLockNoProcess}
	}
	if hasReceipt {
		return HealthReceiptOnly, []string{ReasonReceiptNoProcess}
	}
	if hasTurn {
		return HealthCompleted, []string{ReasonTerminalTurn}
	}
	return HealthUnknown, []string{ReasonNoCurrentEvidence}
}

func buildProcessTrees(processes []ProcessEvidence, threads map[string]ThreadEvidence) []*processTreeWork {
	byPID := map[int]ProcessEvidence{}
	for _, process := range processes {
		if process.PID <= 0 {
			continue
		}
		process.Name = normalizedProcessName(process.Name)
		byPID[process.PID] = process
	}
	type chain struct {
		root    int
		nodes   []ProcessEvidence
		reasons []string
	}
	chains := []chain{}
	for _, process := range byPID {
		role := processRole(process)
		if role != "codex" && role != "guard" && role != "gateway" {
			continue
		}
		nodes := []ProcessEvidence{process}
		reasons := []string{}
		cur := process
		seen := map[int]bool{process.PID: true}
		for cur.ParentPID > 0 {
			parent, ok := byPID[cur.ParentPID]
			if !ok || seen[parent.PID] || processRole(parent) == "" {
				break
			}
			if processPredatesParent(cur, parent) {
				reasons = append(reasons, ReasonPIDReuse)
				break
			}
			seen[parent.PID] = true
			nodes = append(nodes, parent)
			cur = parent
			if len(nodes) >= 8 {
				break
			}
		}
		chains = append(chains, chain{root: nodes[len(nodes)-1].PID, nodes: nodes, reasons: reasons})
	}
	grouped := map[int]*processTreeWork{}
	for _, c := range chains {
		work := grouped[c.root]
		if work == nil {
			work = &processTreeWork{}
			grouped[c.root] = work
		}
		work.report.Reasons = append(work.report.Reasons, c.reasons...)
		seen := map[int]bool{}
		for _, p := range work.raw {
			seen[p.PID] = true
		}
		for _, p := range c.nodes {
			if !seen[p.PID] {
				work.raw = append(work.raw, p)
				seen[p.PID] = true
			}
		}
	}
	out := make([]*processTreeWork, 0, len(grouped))
	for root, work := range grouped {
		sort.Slice(work.raw, func(i, j int) bool {
			if work.raw[i].StartedAt.Equal(work.raw[j].StartedAt) {
				return work.raw[i].PID < work.raw[j].PID
			}
			if work.raw[i].StartedAt.IsZero() {
				return false
			}
			if work.raw[j].StartedAt.IsZero() {
				return true
			}
			return work.raw[i].StartedAt.Before(work.raw[j].StartedAt)
		})
		work.report.RootPID = root
		work.report.ID = processTreeID(root, byPID[root].StartedAt)
		work.report.CommandLinesComplete = true
		hints := map[string]bool{}
		for _, process := range work.raw {
			role := processRole(process)
			node := ProcessNode{
				PID:                  process.PID,
				ParentPID:            process.ParentPID,
				Name:                 process.Name,
				Role:                 role,
				StartedAt:            stamp(process.StartedAt),
				CommandLineAvailable: strings.TrimSpace(process.CommandLine) != "",
			}
			work.report.Nodes = append(work.report.Nodes, node)
			if !node.CommandLineAvailable {
				work.report.CommandLinesComplete = false
				work.report.Reasons = append(work.report.Reasons, ReasonMissingCommand)
			}
			switch role {
			case "codex":
				work.report.HasCodex = true
				if work.report.CodexPID == 0 || process.StartedAt.Before(parseStamp(work.report.CodexStartedAt)) {
					work.report.CodexPID = process.PID
					work.report.CodexStartedAt = stamp(process.StartedAt)
				}
			case "guard":
				work.report.HasGuard = true
			case "gateway":
				work.report.HasGateway = true
			}
			cmd := strings.ToLower(process.CommandLine)
			if codexExecCommand(cmd) {
				work.report.HasExec = true
			}
			if strings.Contains(cmd, " resume ") || strings.Contains(cmd, " exec resume ") {
				work.report.HasResume = true
			}
			for _, hint := range threadIDPattern.FindAllString(process.CommandLine, -1) {
				hint = strings.ToLower(hint)
				if _, ok := threads[hint]; ok {
					hints[hint] = true
				}
			}
		}
		if rootProcess, ok := byPID[root]; ok {
			work.report.StartedAt = stamp(rootProcess.StartedAt)
		}
		switch len(hints) {
		case 1:
			for id := range hints {
				work.report.ThreadID = id
			}
			work.report.Association = "explicit_thread_id"
			work.report.Reasons = append(work.report.Reasons, ReasonExplicitThread)
		case 0:
			work.report.Association = "unmatched"
		default:
			work.report.Association = "ambiguous"
			work.report.Reasons = append(work.report.Reasons, ReasonAmbiguousHints)
		}
		work.report.Reasons = uniqueSorted(work.report.Reasons)
		out = append(out, work)
	}
	sort.Slice(out, func(i, j int) bool {
		ti, tj := processTreeStart(out[i].report), processTreeStart(out[j].report)
		if ti.Equal(tj) {
			return out[i].report.ID < out[j].report.ID
		}
		if ti.IsZero() {
			return false
		}
		if tj.IsZero() {
			return true
		}
		return ti.Before(tj)
	})
	return out
}

func associateProcessTrees(trees []*processTreeWork, threads map[string]ThreadEvidence, window time.Duration) {
	assignedThreads := map[string]bool{}
	for _, tree := range trees {
		if tree.report.ThreadID != "" {
			assignedThreads[tree.report.ThreadID] = true
		}
	}
	for _, source := range []string{"cli", "exec", ""} {
		groupTrees := []*processTreeWork{}
		for _, tree := range trees {
			if tree.report.ThreadID != "" || !tree.report.HasCodex || tree.report.Association == "ambiguous" {
				continue
			}
			expected := ""
			if tree.report.HasExec {
				expected = "exec"
			} else if tree.report.CommandLinesComplete {
				expected = "cli"
			}
			if expected == source {
				groupTrees = append(groupTrees, tree)
			}
		}
		groupThreads := []ThreadEvidence{}
		for id, thread := range threads {
			if assignedThreads[id] {
				continue
			}
			if strings.ToLower(strings.TrimSpace(thread.Source)) == source || source == "" {
				groupThreads = append(groupThreads, thread)
			}
		}
		sort.Slice(groupThreads, func(i, j int) bool {
			if groupThreads[i].CreatedAt.Equal(groupThreads[j].CreatedAt) {
				return groupThreads[i].ThreadID < groupThreads[j].ThreadID
			}
			return groupThreads[i].CreatedAt.Before(groupThreads[j].CreatedAt)
		})
		sort.Slice(groupTrees, func(i, j int) bool {
			ti, tj := processTreeStart(groupTrees[i].report), processTreeStart(groupTrees[j].report)
			if ti.Equal(tj) {
				return groupTrees[i].report.ID < groupTrees[j].report.ID
			}
			return ti.Before(tj)
		})
		for _, match := range orderedStartMatches(groupTrees, groupThreads, window) {
			tree := groupTrees[match.tree]
			id := groupThreads[match.thread].ThreadID
			tree.report.ThreadID = id
			tree.report.Association = "start_time"
			tree.report.Reasons = uniqueSorted(append(tree.report.Reasons, ReasonStartMatched))
			assignedThreads[id] = true
		}
	}
}

type startMatch struct {
	tree   int
	thread int
}

// orderedStartMatches maximizes the number of start-time joins, then minimizes
// their total timestamp delta, while preserving OS/thread creation order. The
// ordering fence matters for simultaneous `codex exec` launches: a nearest-pair
// greedy pass can cross two adjacent workers and attach each PID tree to the
// other's thread even though both individual deltas look plausible.
func orderedStartMatches(trees []*processTreeWork, threads []ThreadEvidence, window time.Duration) []startMatch {
	type score struct {
		matches int
		cost    int64
	}
	better := func(a score, aPriority int, b score, bPriority int) bool {
		if a.matches != b.matches {
			return a.matches > b.matches
		}
		if a.cost != b.cost {
			return a.cost < b.cost
		}
		return aPriority < bPriority
	}
	m, n := len(trees), len(threads)
	dp := make([][]score, m+1)
	action := make([][]byte, m+1)
	for i := range dp {
		dp[i] = make([]score, n+1)
		action[i] = make([]byte, n+1)
	}
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			best, bestAction, bestPriority := dp[i+1][j], byte('t'), 2
			if candidate := dp[i][j+1]; better(candidate, 1, best, bestPriority) {
				best, bestAction, bestPriority = candidate, 'h', 1
			}
			treeStart, threadStart := processTreeStart(trees[i].report), threads[j].CreatedAt
			if !treeStart.IsZero() && !threadStart.IsZero() {
				delta := absDuration(threadStart.Sub(treeStart))
				if delta <= window {
					candidate := dp[i+1][j+1]
					candidate.matches++
					candidate.cost += delta.Nanoseconds()
					if better(candidate, 0, best, bestPriority) {
						best, bestAction = candidate, 'm'
					}
				}
			}
			dp[i][j], action[i][j] = best, bestAction
		}
	}
	out := []startMatch{}
	for i, j := 0, 0; i < m && j < n; {
		switch action[i][j] {
		case 'm':
			out = append(out, startMatch{tree: i, thread: j})
			i++
			j++
		case 'h':
			j++
		default:
			i++
		}
	}
	return out
}

func endpointState(
	id, edgeStatus string,
	threads map[string]ThreadEvidence,
	turns map[string]TurnEvidence,
	sessions map[string]SessionRecord,
	now time.Time,
	window time.Duration,
) string {
	if session, ok := sessions[id]; ok {
		switch session.Health {
		case HealthActive:
			return EndpointActive
		case HealthCompleted:
			return EndpointCompleted
		case HealthFailedButLocked, HealthOrphanProcess, HealthStaleLock, HealthReceiptOnly:
			return EndpointStale
		}
	}
	thread, ok := threads[id]
	if !ok {
		return EndpointUnknown
	}
	if turn, ok := turns[id]; ok && terminalTurn(turn.Status) {
		return EndpointCompleted
	}
	if thread.Archived || strings.EqualFold(strings.TrimSpace(edgeStatus), "closed") {
		return EndpointCompleted
	}
	if !thread.UpdatedAt.IsZero() && now.Sub(thread.UpdatedAt) > window {
		return EndpointStale
	}
	return EndpointUnknown
}

func inventoryRegistrations(rows []sessionregistry.Record, processes []ProcessEvidence) ([]RegistrationRecord, []sessionregistry.UnregisteredObserved) {
	observed := make([]sessionregistry.ObservedProcess, 0, len(processes))
	for _, p := range processes {
		if isAgentProcess(p) {
			observed = append(observed, sessionregistry.ObservedProcess{PID: p.PID, ProcessStartedAt: p.StartedAt, Runtime: strings.TrimSuffix(strings.ToLower(filepath.Base(p.Name)), ".exe")})
		}
	}
	unregistered := sessionregistry.ReconcileObserved(rows, observed)
	out := make([]RegistrationRecord, 0, len(rows))
	for _, r := range rows {
		matched := false
		for _, p := range processes {
			if r.Identity.PID != 0 && r.Identity.PID == p.PID && !r.Identity.ProcessStartedAt.IsZero() && r.Identity.ProcessStartedAt.Equal(p.StartedAt.UTC()) {
				matched = true
				break
			}
		}
		health := "RECEIPT_ONLY"
		switch {
		case matched && r.State == sessionregistry.StateActive:
			health = "ACTIVE"
		case matched:
			health = "PROCESS_PRESENT"
		case r.State == sessionregistry.StateActive:
			health = "REGISTERED_ACTIVE_NO_PROCESS"
		case r.State == sessionregistry.StateUnknown:
			health = "UNKNOWN"
		case r.State == sessionregistry.StateRegistered:
			health = "REGISTERED_NOT_STARTED"
		default:
			health = "TERMINAL"
		}
		out = append(out, RegistrationRecord{RegistrationID: r.RegistrationID, ParentRegistrationID: r.ParentRegistrationID, ParentAttemptID: r.ParentAttemptID, RootRegistrationID: r.RootRegistrationID, RootOutcome: r.RootOutcome, RootIssue: r.RootIssue, TaskID: r.TaskID, AttemptID: r.AttemptID, ResumeOfAttemptID: r.ResumeOfAttemptID, LaunchKind: r.LaunchKind, Lane: r.Lane, LeaseID: r.LeaseID, Runtime: r.Identity.Runtime, SessionID: r.Identity.SessionID, ThreadID: r.Identity.ThreadID, PID: r.Identity.PID, ProcessStartedAt: r.Identity.ProcessStartedAt, State: r.State, Reason: r.Reason, WitnessRef: r.WitnessRef, ProcessMatched: matched, Health: health})
	}
	return out, unregistered
}
func normalizedSessionHarness(runtime string) string {
	switch strings.ToLower(strings.TrimSpace(runtime)) {
	case "claude", "codex", "opencode", "fak":
		return strings.ToLower(strings.TrimSpace(runtime))
	default:
		return ""
	}
}

func isAgentProcess(p ProcessEvidence) bool {
	n := strings.ToLower(filepath.Base(p.Name))
	c := strings.ToLower(p.CommandLine)
	return strings.Contains(n, "codex") || strings.Contains(n, "claude") || strings.Contains(n, "opencode") || strings.Contains(c, "dispatchworker") || strings.Contains(c, "fak guard")
}

func RenderInventory(w io.Writer, report InventoryReport) {
	fmt.Fprintf(w, "CODEX SESSION INVENTORY observed=%s active=%d total=%d process_trees=%d spawn_edges=%d\n",
		report.ObservedAt, report.Counts.Active, report.Counts.Total, report.Counts.ProcessTrees, report.Counts.SpawnEdges)
	fmt.Fprintf(w, "health: %s\n", renderCounts(report.Counts.ByHealth))
	fmt.Fprintf(w, "kinds:  %s\n", renderCounts(report.Counts.ByKind))
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "HEALTH\tKIND\tTHREAD\tTURN\tLOCK\tRECEIPT\tPROCS\tREASONS")
	for _, session := range report.Sessions {
		if session.Health == HealthCompleted && session.WriterLock == nil && len(session.ProcessTrees) == 0 {
			continue
		}
		threadID := "-"
		if session.Thread != nil {
			threadID = shortID(session.Thread.ID)
		}
		turn := "-"
		if session.LatestTurn != nil {
			turn = session.LatestTurn.Status
			if session.LatestTurn.ErrorCode != "" {
				turn += ":" + session.LatestTurn.ErrorCode
			}
		}
		lock := "-"
		if session.WriterLock != nil {
			lock = compactAge(session.WriterLock.AgeSeconds)
		}
		receipt := "-"
		if session.GuardReceipt != nil {
			receipt = "launch"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n",
			session.Health, session.Kind, threadID, turn, lock, receipt, len(session.ProcessTrees), strings.Join(session.Reasons, ","))
	}
	_ = tw.Flush()
	if len(report.CleanupActions) > 0 {
		fmt.Fprintln(w, "lifecycle reconciliation (dry-run):")
		fmt.Fprintln(w, "ARTIFACT\tIDENTITY\tACTION\tREASON")
		for _, action := range report.CleanupActions {
			fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", action.Artifact, action.Identity, action.Action, action.Reason)
		}
	}
	if len(report.Registrations) > 0 || len(report.UnregisteredObserved) > 0 {
		fmt.Fprintf(w, "REGISTRATIONS total=%d active=%d terminal=%d unknown=%d unregistered_observed=%d\n", report.Counts.Registrations.Total, report.Counts.Registrations.Active, report.Counts.Registrations.Terminal, report.Counts.Registrations.Unknown, report.Counts.Registrations.UnregisteredObserved)
		rtw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintln(rtw, "REGISTRATION\tPARENT\tROOT\tISSUE\tATTEMPT\tKIND\tPID\tSTATE\tHEALTH\tWITNESS")
		for _, r := range report.Registrations {
			fmt.Fprintf(rtw, "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n", inventoryValue(r.RegistrationID), inventoryValue(r.ParentRegistrationID), inventoryValue(r.RootRegistrationID), inventoryValue(r.RootIssue), inventoryValue(r.AttemptID), inventoryValue(r.LaunchKind), r.PID, r.State, r.Health, inventoryValue(r.WitnessRef))
		}
		for _, u := range report.UnregisteredObserved {
			fmt.Fprintf(rtw, "UNREGISTERED_OBSERVED\t-\t-\t-\t-\tobserved_process\t%d\t-\t%s\t-\n", u.Process.PID, u.Reason)
		}
		_ = rtw.Flush()
	}
	if len(report.SpawnEdges) > 0 {
		fmt.Fprintln(w, "spawn edges:")
		etw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintln(etw, "PARENT\tCHILD\tSTATUS\tPARENT_STATE\tCHILD_STATE")
		for _, edge := range report.SpawnEdges {
			fmt.Fprintf(etw, "%s\t%s\t%s\t%s\t%s\n",
				shortID(edge.Parent.ThreadID), shortID(edge.Child.ThreadID), edge.Status, edge.Parent.State, edge.Child.State)
		}
		_ = etw.Flush()
	}
	fmt.Fprintf(w, "note: %s\n", report.Notice)
}

func safeTurnErrorCode(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	lower := strings.ToLower(raw)
	for _, token := range []string{
		"invalid_id_prefix", "invalid_request_error", "rate_limit_exceeded", "context_length_exceeded",
		"authentication_error", "permission_denied", "server_error", "timeout", "cancelled",
	} {
		if strings.Contains(lower, token) {
			return token
		}
	}
	var value any
	if json.Unmarshal([]byte(raw), &value) == nil {
		if token := firstSafeErrorToken(value); token != "" {
			return strings.ToLower(token)
		}
	}
	return "error_present"
}

func firstSafeErrorToken(value any) string {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"code", "type", "codexErrorInfo", "reason"} {
			if token, ok := typed[key].(string); ok && safeErrorTokenPattern.MatchString(token) {
				return token
			}
		}
		for _, key := range []string{"error", "additionalDetails", "details"} {
			if token := firstSafeErrorToken(typed[key]); token != "" {
				return token
			}
		}
	case []any:
		for _, item := range typed {
			if token := firstSafeErrorToken(item); token != "" {
				return token
			}
		}
	}
	return ""
}

func processRole(process ProcessEvidence) string {
	name := normalizedProcessName(process.Name)
	cmd := strings.ToLower(strings.TrimSpace(process.CommandLine))
	switch name {
	case "codex":
		return "codex"
	case "node":
		return "node"
	case "fak":
		switch {
		case strings.Contains(cmd, " guard ") || strings.HasSuffix(cmd, " guard") || strings.Contains(cmd, " guard codex"):
			return "guard"
		case strings.Contains(cmd, " serve ") || strings.HasSuffix(cmd, " serve") || strings.Contains(cmd, " gateway "):
			return "gateway"
		}
	case "cmd", "pwsh", "powershell", "bash", "sh", "zsh":
		return "wrapper"
	}
	return ""
}

func codexExecCommand(cmd string) bool {
	if !strings.Contains(cmd, "codex") {
		return false
	}
	return strings.Contains(cmd, " exec ") || strings.HasSuffix(cmd, " exec") || strings.Contains(cmd, " exec --")
}

func processPredatesParent(child, parent ProcessEvidence) bool {
	return !child.StartedAt.IsZero() && !parent.StartedAt.IsZero() && child.StartedAt.Before(parent.StartedAt)
}

func processTreeID(root int, started time.Time) string {
	if started.IsZero() {
		return fmt.Sprintf("process:%d:unknown", root)
	}
	return fmt.Sprintf("process:%d:%d", root, started.UTC().UnixNano())
}

func processTreeStart(tree ProcessTree) time.Time {
	if t := parseStamp(tree.CodexStartedAt); !t.IsZero() {
		return t
	}
	return parseStamp(tree.StartedAt)
}

func parseStamp(value string) time.Time {
	t, _ := time.Parse(time.RFC3339Nano, value)
	return t
}

func normalizedProcessName(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	name = strings.TrimSuffix(name, ".exe")
	return name
}

func normalizedToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if safeErrorTokenPattern.MatchString(value) {
		return value
	}
	return ""
}

func canonicalTurnStatus(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "inprogress", "in_progress", "running":
		return "inProgress"
	case "completed":
		return "completed"
	case "failed":
		return "failed"
	case "interrupted":
		return "interrupted"
	case "cancelled", "canceled":
		return "cancelled"
	default:
		return normalizedToken(value)
	}
}

func safeLabel(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > 80 {
		return ""
	}
	for _, r := range value {
		if !(r == ' ' || r == '-' || r == '_' || r == '.' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			return ""
		}
	}
	return value
}

func terminalTurn(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "completed", "failed", "interrupted", "cancelled":
		return true
	}
	return false
}

func ageSeconds(now, then time.Time) int64 {
	if then.IsZero() {
		return -1
	}
	age := int64(now.Sub(then).Seconds())
	if age < 0 {
		return 0
	}
	return age
}

func absDuration(value time.Duration) time.Duration {
	if value < 0 {
		return -value
	}
	return value
}

func appendUnique(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func sortedCopy(values []string) []string {
	out := append([]string(nil), values...)
	sort.Strings(out)
	return out
}

func uniqueSorted(values []string) []string {
	set := map[string]bool{}
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			set[value] = true
		}
	}
	return stringset.Sorted(set)
}

func sortSessions(sessions []SessionRecord) {
	priority := map[string]int{
		HealthActive:          0,
		HealthFailedButLocked: 1,
		HealthOrphanProcess:   2,
		HealthStaleLock:       3,
		HealthReceiptOnly:     4,
		HealthUnknown:         5,
		HealthCompleted:       6,
	}
	sort.Slice(sessions, func(i, j int) bool {
		pi, pj := priority[sessions[i].Health], priority[sessions[j].Health]
		if pi != pj {
			return pi < pj
		}
		if sessions[i].Kind != sessions[j].Kind {
			return sessions[i].Kind < sessions[j].Kind
		}
		return sessions[i].RecordID < sessions[j].RecordID
	})
}

func renderCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, counts[key]))
	}
	return strings.Join(parts, " ")
}

func shortID(id string) string {
	if len(id) <= 16 {
		return id
	}
	return id[:8] + ".." + id[len(id)-4:]
}

func compactAge(seconds int64) string {
	switch {
	case seconds < 0:
		return "?"
	case seconds < 60:
		return fmt.Sprintf("%ds", seconds)
	case seconds < 3600:
		return fmt.Sprintf("%dm", seconds/60)
	default:
		return fmt.Sprintf("%dh", seconds/3600)
	}
}

func inventoryValue(v string) string {
	if strings.TrimSpace(v) == "" {
		return "-"
	}
	return v
}
