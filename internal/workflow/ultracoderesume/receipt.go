package ultracoderesume

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/workflow"
)

const (
	GraphReceiptSchema   = "fak-ultracode-graph-receipt/1"
	AttemptReceiptSchema = "fak-ultracode-node-attempt/1"
	ResumeReportSchema   = "fak-ultracode-resume/1"

	GraphFile           = "graph.json"
	AttemptsFile        = "attempts.jsonl"
	WorkflowJournalFile = "journal.jsonl"
)

type RefusalReason string

const (
	RefusalPlanSchema      RefusalReason = "PLAN_SCHEMA_INCOMPATIBLE"
	RefusalPlanContract    RefusalReason = "PLAN_CONTRACT_INCOMPATIBLE"
	RefusalGraphSchema     RefusalReason = "GRAPH_SCHEMA_INCOMPATIBLE"
	RefusalGraphIdentity   RefusalReason = "GRAPH_IDENTITY_MISMATCH"
	RefusalGraphEpoch      RefusalReason = "GRAPH_EPOCH_MISMATCH"
	RefusalAttemptSchema   RefusalReason = "ATTEMPT_SCHEMA_INCOMPATIBLE"
	RefusalAttemptIdentity RefusalReason = "ATTEMPT_IDENTITY_MISMATCH"
	RefusalGraphCorrupt    RefusalReason = "GRAPH_RECEIPT_CORRUPT"
	RefusalAttemptCorrupt  RefusalReason = "ATTEMPT_JOURNAL_CORRUPT"
	RefusalWorkflowCorrupt RefusalReason = "WORKFLOW_JOURNAL_CORRUPT"
	RefusalPersistence     RefusalReason = "PERSISTENCE_FAILED"
)

// Refusal is a fail-closed compatibility or persistence verdict. Detail is
// deliberately limited to schema tags, opaque identities, and stable labels.
type Refusal struct {
	Reason RefusalReason `json:"reason"`
	Detail string        `json:"detail,omitempty"`
}

// refusalString renders a closed-reason refusal the one way every *Refusal
// error in this package does: the reason token alone, or "REASON: detail"
// when a detail is attached. The string form is for logs and error plumbing;
// callers switch on Reason, never parse Detail (the Refusal doc).
func refusalString(reason, detail string) string {
	if detail == "" {
		return reason
	}
	return reason + ": " + detail
}

func (r *Refusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

func refuse(reason RefusalReason, format string, args ...any) *Refusal {
	return &Refusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

type NodeKind string

const (
	NodeObserve NodeKind = "observe"
	NodeEffect  NodeKind = "effect"
)

// GraphNode stores only stable ids and digests. Prompts, purposes, access paths,
// hostnames, and engine configuration remain outside the recovery journal.
type GraphNode struct {
	ID       string   `json:"id"`
	Identity string   `json:"identity"`
	Kind     NodeKind `json:"kind"`
	Needs    []string `json:"needs,omitempty"`
}

// GraphReceipt is the versioned, persisted reconstruction boundary for one
// resolved Ultracode plan.
type GraphReceipt struct {
	Schema       string      `json:"schema"`
	PlanSchema   string      `json:"plan_schema"`
	PlanIdentity string      `json:"plan_identity"`
	GraphID      string      `json:"graph_id"`
	GraphEpoch   string      `json:"graph_epoch"`
	Nodes        []GraphNode `json:"nodes"`
}

type AttemptStatus string

const (
	AttemptStarted       AttemptStatus = "started"
	AttemptCompleted     AttemptStatus = "completed"
	AttemptFailed        AttemptStatus = "failed"
	AttemptEffectUnknown AttemptStatus = "effect_unknown"
)

// Admission is the fresh authority attached before an effectful retry starts.
type Admission struct {
	LeaseID             string `json:"lease_id,omitempty"`
	EffectReceiptID     string `json:"effect_receipt_id,omitempty"`
	CapabilityID        string `json:"capability_id,omitempty"`
	CapabilityExpiresMS int64  `json:"capability_expires_ms,omitempty"`
}

// AdmissionFromEffectSuccessor maps the existing native orchestration launch
// receipt into the compact recovery identity. The resume journal stores only
// its opaque receipt and lease ids, never its envelope or path fields.
func AdmissionFromEffectSuccessor(nodeID string, receipt orchestration.EffectSuccessorReceipt, capabilityID string, capabilityExpiresMS int64) (Admission, error) {
	if receipt.Schema != orchestration.EffectSuccessorReceiptSchema {
		return Admission{}, refuse(RefusalAttemptSchema, "effect receipt schema %q is incompatible", receipt.Schema)
	}
	if receipt.NodeID != nodeID || !opaqueKey(receipt.ID) || !opaqueKey(receipt.LeaseID) {
		return Admission{}, refuse(RefusalAttemptIdentity, "effect receipt does not bind node %q", nodeID)
	}
	if !opaqueKey(capabilityID) || capabilityExpiresMS <= 0 {
		return Admission{}, refuse(RefusalPlanContract, "effect receipt lacks a bounded capability")
	}
	return Admission{
		LeaseID: receipt.LeaseID, EffectReceiptID: receipt.ID,
		CapabilityID: capabilityID, CapabilityExpiresMS: capabilityExpiresMS,
	}, nil
}

// ExecutionReceipt contains opaque receipt and independent-witness keys only.
// It intentionally cannot carry an output body, prompt, credential, path, or host.
type ExecutionReceipt struct {
	ReceiptKey string `json:"receipt_key,omitempty"`
	WitnessKey string `json:"witness_key,omitempty"`
}

// AttemptRecord is the caller-facing input used to mint a sealed attempt row.
type AttemptRecord struct {
	Attempt   int
	Status    AttemptStatus
	Admission Admission
	Result    ExecutionReceipt
	TSMS      int64
}

// AttemptReceipt is one write-ahead or terminal node-attempt row. ReceiptID is
// a digest of every other field, so a damaged row cannot silently become a hit.
type AttemptReceipt struct {
	Schema              string        `json:"schema"`
	GraphID             string        `json:"graph_id"`
	GraphEpoch          string        `json:"graph_epoch"`
	NodeID              string        `json:"node_id"`
	NodeIdentity        string        `json:"node_identity"`
	Attempt             int           `json:"attempt"`
	Status              AttemptStatus `json:"status"`
	LeaseID             string        `json:"lease_id,omitempty"`
	EffectReceiptID     string        `json:"effect_receipt_id,omitempty"`
	CapabilityID        string        `json:"capability_id,omitempty"`
	CapabilityExpiresMS int64         `json:"capability_expires_ms,omitempty"`
	ReceiptKey          string        `json:"receipt_key,omitempty"`
	WitnessKey          string        `json:"witness_key,omitempty"`
	TSMS                int64         `json:"ts_ms"`
	ReceiptID           string        `json:"receipt_id"`
}

// Store is the crash-safe file boundary. Every append is synced before the
// caller is allowed to cross the corresponding launch or completion boundary.
type Store struct {
	Dir string
}

func ReceiptForPlan(plan orchestration.WorkflowPlan) (GraphReceipt, error) {
	if plan.Schema != orchestration.SchemaVersion {
		return GraphReceipt{}, refuse(RefusalPlanSchema, "got %q want %q", plan.Schema, orchestration.SchemaVersion)
	}
	if plan.Profile != orchestration.ProfileUltracode || !plan.Leases.Required ||
		!plan.Witness.Independent || !plan.Witness.EffectReadback || !plan.Reconcile.Required {
		return GraphReceipt{}, refuse(RefusalPlanContract, "ultracode requires leases, independent effect readback, and reconciliation")
	}

	plan = clonePlan(plan)
	if err := orchestration.NormalizeWorkflowPlanAccess(&plan); err != nil {
		return GraphReceipt{}, refuse(RefusalPlanContract, "invalid node access contract")
	}
	for i := range plan.Roles {
		sort.Strings(plan.Roles[i].Access.Tools)
	}
	sort.Slice(plan.Roles, func(i, j int) bool { return plan.Roles[i].ID < plan.Roles[j].ID })
	sort.Slice(plan.DAG, func(i, j int) bool {
		if plan.DAG[i].From != plan.DAG[j].From {
			return plan.DAG[i].From < plan.DAG[j].From
		}
		return plan.DAG[i].To < plan.DAG[j].To
	})

	roleByID := make(map[string]orchestration.Role, len(plan.Roles))
	identityByID := make(map[string]string, len(plan.Roles))
	needs := make(map[string][]string, len(plan.Roles))
	for _, role := range plan.Roles {
		if strings.TrimSpace(role.ID) == "" {
			return GraphReceipt{}, refuse(RefusalPlanContract, "empty node id")
		}
		if _, exists := roleByID[role.ID]; exists {
			return GraphReceipt{}, refuse(RefusalPlanContract, "duplicate node id %q", role.ID)
		}
		roleByID[role.ID] = role
		identityByID[role.ID] = digestValue("node", role)
	}
	for _, edge := range plan.DAG {
		if _, ok := roleByID[edge.From]; !ok {
			return GraphReceipt{}, refuse(RefusalPlanContract, "edge source %q is unknown", edge.From)
		}
		if _, ok := roleByID[edge.To]; !ok {
			return GraphReceipt{}, refuse(RefusalPlanContract, "edge target %q is unknown", edge.To)
		}
		needs[edge.To] = append(needs[edge.To], edge.From)
	}

	tasks := make([]workflow.TaskSpec, 0, len(plan.Roles))
	for _, role := range plan.Roles {
		kind := nodeKind(role.Access.Mode)
		if kind == "" {
			return GraphReceipt{}, refuse(RefusalPlanContract, "node %q has unsupported access mode", role.ID)
		}
		sort.Strings(needs[role.ID])
		tasks = append(tasks, workflow.TaskSpec{ID: role.ID, Op: string(kind), Payload: identityByID[role.ID], Needs: needs[role.ID]})
	}
	compiled, err := workflow.Compile(workflow.Spec{Name: "ultracode", Tasks: tasks})
	if err != nil {
		return GraphReceipt{}, refuse(RefusalPlanContract, "invalid graph topology")
	}

	nodes := make([]GraphNode, 0, len(compiled.Nodes))
	for _, node := range compiled.Nodes {
		nodes = append(nodes, GraphNode{
			ID: node.ID, Identity: identityByID[node.ID], Kind: nodeKind(roleByID[node.ID].Access.Mode),
			Needs: append([]string(nil), node.Needs...),
		})
	}
	planIdentity := digestValue("plan", struct {
		Schema      string
		Profile     orchestration.Profile
		TaskID      string
		WorkClass   orchestration.WorkClass
		Nodes       []GraphNode
		Budget      orchestration.Budget
		Leases      orchestration.LeasePolicy
		Witness     orchestration.WitnessPolicy
		Reconcile   orchestration.ReconcilePolicy
		Interaction orchestration.InteractionPolicy
		EngineRef   string
		SOLRoute    orchestration.SOLRoute
	}{plan.Schema, plan.Profile, plan.TaskID, plan.WorkClass, nodes, plan.Budget, plan.Leases, plan.Witness, plan.Reconcile, plan.Interaction, plan.EngineRef, plan.SOLRoute})
	receipt := GraphReceipt{Schema: GraphReceiptSchema, PlanSchema: plan.Schema, PlanIdentity: planIdentity, Nodes: nodes}
	receipt.GraphID = graphDigest(receipt)
	receipt.GraphEpoch = workflow.GraphEpoch(receipt.workflowGraph(), receipt.GraphID)
	return receipt, nil
}

func (g GraphReceipt) workflowGraph() *workflow.Graph {
	nodes := make([]workflow.Node, 0, len(g.Nodes))
	for _, node := range g.Nodes {
		nodes = append(nodes, workflow.Node{
			ID: node.ID, Op: string(node.Kind), Payload: node.Identity, Needs: append([]string(nil), node.Needs...),
		})
	}
	return &workflow.Graph{Name: g.GraphID, Nodes: nodes}
}

func verifyGraph(g GraphReceipt) error {
	if g.Schema != GraphReceiptSchema {
		return refuse(RefusalGraphSchema, "got %q want %q", g.Schema, GraphReceiptSchema)
	}
	if g.PlanSchema != orchestration.SchemaVersion {
		return refuse(RefusalPlanSchema, "got %q want %q", g.PlanSchema, orchestration.SchemaVersion)
	}
	if g.PlanIdentity == "" || len(g.Nodes) == 0 {
		return refuse(RefusalGraphCorrupt, "missing graph identity or nodes")
	}
	tasks := make([]workflow.TaskSpec, 0, len(g.Nodes))
	seen := make(map[string]bool, len(g.Nodes))
	for _, node := range g.Nodes {
		if node.ID == "" || node.Identity == "" || (node.Kind != NodeObserve && node.Kind != NodeEffect) || seen[node.ID] {
			return refuse(RefusalGraphCorrupt, "invalid graph node")
		}
		seen[node.ID] = true
		tasks = append(tasks, workflow.TaskSpec{ID: node.ID, Op: string(node.Kind), Payload: node.Identity, Needs: node.Needs})
	}
	compiled, err := workflow.Compile(workflow.Spec{Name: g.GraphID, Tasks: tasks})
	if err != nil || len(compiled.Nodes) != len(g.Nodes) {
		return refuse(RefusalGraphCorrupt, "invalid graph topology")
	}
	for i := range compiled.Nodes {
		if compiled.Nodes[i].ID != g.Nodes[i].ID {
			return refuse(RefusalGraphCorrupt, "nodes are not in canonical order")
		}
	}
	if want := graphDigest(g); g.GraphID != want {
		return refuse(RefusalGraphIdentity, "stored %s derived %s", g.GraphID, want)
	}
	if want := workflow.GraphEpoch(g.workflowGraph(), g.GraphID); g.GraphEpoch != want {
		return refuse(RefusalGraphEpoch, "stored %s derived %s", g.GraphEpoch, want)
	}
	return nil
}

func graphDigest(g GraphReceipt) string {
	return digestValue("graph", struct {
		Schema       string
		PlanSchema   string
		PlanIdentity string
		Nodes        []GraphNode
	}{g.Schema, g.PlanSchema, g.PlanIdentity, g.Nodes})
}

func nodeKind(mode orchestration.ChildAccessMode) NodeKind {
	switch mode {
	case orchestration.ChildAccessObserve:
		return NodeObserve
	case orchestration.ChildAccessEffect:
		return NodeEffect
	default:
		return ""
	}
}

func clonePlan(plan orchestration.WorkflowPlan) orchestration.WorkflowPlan {
	plan.Roles = append([]orchestration.Role(nil), plan.Roles...)
	for i := range plan.Roles {
		access := &plan.Roles[i].Access
		access.ReadSet = append([]string(nil), access.ReadSet...)
		access.WriteSet = append([]string(nil), access.WriteSet...)
		access.Tools = append([]string(nil), access.Tools...)
	}
	plan.DAG = append([]orchestration.Edge(nil), plan.DAG...)
	return plan
}

func NewAttemptReceipt(graph GraphReceipt, nodeID string, record AttemptRecord) (AttemptReceipt, error) {
	if err := verifyGraph(graph); err != nil {
		return AttemptReceipt{}, err
	}
	node, ok := graph.node(nodeID)
	if !ok {
		return AttemptReceipt{}, refuse(RefusalAttemptIdentity, "unknown node %q", nodeID)
	}
	receipt := AttemptReceipt{
		Schema: AttemptReceiptSchema, GraphID: graph.GraphID, GraphEpoch: graph.GraphEpoch,
		NodeID: node.ID, NodeIdentity: node.Identity, Attempt: record.Attempt, Status: record.Status,
		LeaseID: record.Admission.LeaseID, EffectReceiptID: record.Admission.EffectReceiptID, CapabilityID: record.Admission.CapabilityID,
		CapabilityExpiresMS: record.Admission.CapabilityExpiresMS,
		ReceiptKey:          record.Result.ReceiptKey, WitnessKey: record.Result.WitnessKey, TSMS: record.TSMS,
	}
	if err := validateAttemptShape(receipt, node); err != nil {
		return AttemptReceipt{}, err
	}
	receipt.ReceiptID = attemptDigest(receipt)
	return receipt, nil
}

func validateAttemptShape(receipt AttemptReceipt, node GraphNode) error {
	if receipt.Attempt <= 0 || receipt.TSMS <= 0 {
		return errors.New("attempt and ts_ms must be positive")
	}
	if node.Kind == NodeEffect {
		if !opaqueKey(receipt.LeaseID) || !opaqueKey(receipt.EffectReceiptID) || !opaqueKey(receipt.CapabilityID) || receipt.CapabilityExpiresMS <= receipt.TSMS {
			return errors.New("effect attempt requires an effect receipt, live lease, and capability")
		}
	}
	switch receipt.Status {
	case AttemptStarted:
		if receipt.ReceiptKey != "" || receipt.WitnessKey != "" {
			return errors.New("started attempt cannot carry terminal keys")
		}
	case AttemptCompleted:
		if !opaqueKey(receipt.ReceiptKey) || !opaqueKey(receipt.WitnessKey) {
			return errors.New("completed attempt requires opaque receipt and witness keys")
		}
	case AttemptFailed:
		if receipt.ReceiptKey != "" || receipt.WitnessKey != "" {
			return errors.New("failed attempt cannot carry success keys")
		}
	case AttemptEffectUnknown:
		if node.Kind != NodeEffect || !opaqueKey(receipt.ReceiptKey) {
			return errors.New("effect_unknown requires an effect node and receipt key")
		}
		if receipt.WitnessKey != "" && !opaqueKey(receipt.WitnessKey) {
			return errors.New("effect_unknown witness key is not opaque")
		}
	default:
		return errors.New("unknown attempt status")
	}
	return nil
}

func verifyAttempt(receipt AttemptReceipt, node GraphNode) error {
	if err := validateAttemptShape(receipt, node); err != nil {
		return err
	}
	if receipt.ReceiptID == "" || receipt.ReceiptID != attemptDigest(receipt) {
		return errors.New("attempt receipt digest mismatch")
	}
	return nil
}

func attemptDigest(receipt AttemptReceipt) string {
	receipt.ReceiptID = ""
	return digestValue("attempt", receipt)
}

func opaqueKey(value string) bool {
	if value == "" || len(value) > 256 {
		return false
	}
	for _, r := range value {
		if unicode.IsSpace(r) || unicode.IsControl(r) || r == '\\' {
			return false
		}
	}
	return true
}

func digestValue(domain string, value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		panic(err) // all callers pass fixed JSON-safe structs
	}
	digest := sha256.Sum256(append([]byte(domain+"\x00"), raw...))
	return "sha256:" + hex.EncodeToString(digest[:])
}

func (g GraphReceipt) node(id string) (GraphNode, bool) {
	for _, node := range g.Nodes {
		if node.ID == id {
			return node, true
		}
	}
	return GraphNode{}, false
}

func (s Store) Initialize(plan orchestration.WorkflowPlan) (GraphReceipt, error) {
	want, err := ReceiptForPlan(plan)
	if err != nil {
		return GraphReceipt{}, err
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return GraphReceipt{}, refuse(RefusalPersistence, "create run directory")
	}
	path := filepath.Join(s.Dir, GraphFile)
	if _, err := os.Stat(path); err == nil {
		got, readErr := s.ReadGraph()
		if readErr != nil {
			return GraphReceipt{}, readErr
		}
		if got.GraphID != want.GraphID || got.GraphEpoch != want.GraphEpoch {
			return GraphReceipt{}, refuse(RefusalGraphIdentity, "stored %s current %s", got.GraphID, want.GraphID)
		}
		return got, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return GraphReceipt{}, refuse(RefusalPersistence, "inspect graph receipt")
	}
	raw, err := json.MarshalIndent(want, "", "  ")
	if err != nil {
		return GraphReceipt{}, refuse(RefusalPersistence, "encode graph receipt")
	}
	raw = append(raw, '\n')
	if err := writeAtomic(path, raw); err != nil {
		return GraphReceipt{}, refuse(RefusalPersistence, "persist graph receipt")
	}
	return want, nil
}

func (s Store) ReadGraph() (GraphReceipt, error) {
	f, err := os.Open(filepath.Join(s.Dir, GraphFile))
	if err != nil {
		return GraphReceipt{}, refuse(RefusalGraphCorrupt, "graph receipt is unreadable")
	}
	defer f.Close()
	var graph GraphReceipt
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&graph); err != nil || !jsonEOF(dec) {
		return GraphReceipt{}, refuse(RefusalGraphCorrupt, "graph receipt is not canonical JSON")
	}
	if err := verifyGraph(graph); err != nil {
		return GraphReceipt{}, err
	}
	return graph, nil
}

func (s Store) AppendAttempt(receipt AttemptReceipt) error {
	graph, err := s.ReadGraph()
	if err != nil {
		return err
	}
	node, ok := graph.node(receipt.NodeID)
	if !ok || receipt.Schema != AttemptReceiptSchema || receipt.GraphID != graph.GraphID ||
		receipt.GraphEpoch != graph.GraphEpoch || receipt.NodeIdentity != node.Identity {
		return refuse(RefusalAttemptIdentity, "attempt does not belong to persisted graph")
	}
	if err := verifyAttempt(receipt, node); err != nil {
		return refuse(RefusalAttemptCorrupt, "attempt receipt is invalid")
	}
	return appendJSONLine(filepath.Join(s.Dir, AttemptsFile), receipt)
}

// RecordAttempt is the integration seam for native launch adapters. It syncs
// the versioned attempt row before returning. A completed row is synced first,
// then mirrored into the generic workflow journal; a crash between those writes
// is recoverable from the independently witnessed attempt receipt.
func (s Store) RecordAttempt(graph GraphReceipt, nodeID string, record AttemptRecord, dependencyReceiptKeys map[string]string) (AttemptReceipt, error) {
	receipt, err := NewAttemptReceipt(graph, nodeID, record)
	if err != nil {
		return AttemptReceipt{}, err
	}
	var mirror *workflow.Entry
	if record.Status == AttemptCompleted {
		node, _ := graph.node(nodeID)
		depHashes := make(map[string]string, len(node.Needs))
		for _, dependency := range node.Needs {
			key := dependencyReceiptKeys[dependency]
			if !opaqueKey(key) {
				return AttemptReceipt{}, refuse(RefusalPersistence, "completed node lacks a dependency receipt key")
			}
			depHashes[dependency] = workflow.HashOutput(key)
		}
		var flowNode workflow.Node
		for _, candidate := range graph.workflowGraph().Nodes {
			if candidate.ID == nodeID {
				flowNode = candidate
				break
			}
		}
		mirror = &workflow.Entry{
			Run: graph.GraphID, Step: nodeID, Kind: workflow.StepEffectful,
			InputsHash: workflow.StepInputsHash(flowNode, depHashes), EpochHash: graph.GraphEpoch,
			OutputHash: workflow.HashOutput(receipt.ReceiptKey), Output: receipt.ReceiptKey,
			Claim: receipt.WitnessKey, TSMS: receipt.TSMS,
		}
	}
	if err := s.AppendAttempt(receipt); err != nil {
		return AttemptReceipt{}, err
	}
	if mirror != nil {
		if err := s.appendWorkflow(*mirror); err != nil {
			return receipt, err
		}
	}
	return receipt, nil
}

func (s Store) ReadAttempts() ([]AttemptReceipt, error) {
	f, err := os.Open(filepath.Join(s.Dir, AttemptsFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, refuse(RefusalAttemptCorrupt, "attempt journal is unreadable")
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	dec.DisallowUnknownFields()
	var rows []AttemptReceipt
	for {
		var row AttemptReceipt
		if err := dec.Decode(&row); errors.Is(err, io.EOF) {
			return rows, nil
		} else if err != nil {
			return nil, refuse(RefusalAttemptCorrupt, "attempt journal contains invalid JSON")
		}
		rows = append(rows, row)
	}
}

func (s Store) ReadWorkflowJournal() ([]workflow.Entry, error) {
	f, err := os.Open(filepath.Join(s.Dir, WorkflowJournalFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, refuse(RefusalWorkflowCorrupt, "workflow journal is unreadable")
	}
	defer f.Close()
	rows, err := workflow.ReadJournal(f)
	if err != nil {
		return nil, refuse(RefusalWorkflowCorrupt, "workflow journal contains invalid JSON")
	}
	return rows, nil
}

func (s Store) appendWorkflow(entry workflow.Entry) error {
	path := filepath.Join(s.Dir, WorkflowJournalFile)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return refuse(RefusalPersistence, "open workflow journal")
	}
	if err := workflow.AppendEntry(f, entry); err != nil {
		_ = f.Close()
		return refuse(RefusalPersistence, "append workflow journal")
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return refuse(RefusalPersistence, "sync workflow journal")
	}
	if err := f.Close(); err != nil {
		return refuse(RefusalPersistence, "close workflow journal")
	}
	return nil
}

func appendJSONLine(path string, value any) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return refuse(RefusalPersistence, "open attempt journal")
	}
	raw, err := json.Marshal(value)
	if err != nil {
		_ = f.Close()
		return refuse(RefusalPersistence, "encode attempt receipt")
	}
	if _, err := f.Write(append(raw, '\n')); err != nil {
		_ = f.Close()
		return refuse(RefusalPersistence, "append attempt journal")
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return refuse(RefusalPersistence, "sync attempt journal")
	}
	if err := f.Close(); err != nil {
		return refuse(RefusalPersistence, "close attempt journal")
	}
	return nil
}

func writeAtomic(path string, raw []byte) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".graph-*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	ok := false
	defer func() {
		if !ok {
			_ = os.Remove(tmp)
		}
	}()
	if _, err := io.Copy(f, bytes.NewReader(raw)); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	ok = true
	return nil
}

func jsonEOF(dec *json.Decoder) bool {
	var extra any
	return errors.Is(dec.Decode(&extra), io.EOF)
}
