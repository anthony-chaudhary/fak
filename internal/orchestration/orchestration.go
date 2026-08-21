package orchestration

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

const SchemaVersion = "fak-orchestration-plan/1"

type Profile string

const (
	ProfileOff       Profile = "off"
	ProfileAuto      Profile = "auto"
	ProfileUltracode Profile = "ultracode"
)

type SupportLevel string

const (
	SupportNative      SupportLevel = "native"
	SupportEmulated    SupportLevel = "emulated"
	SupportDegraded    SupportLevel = "degraded"
	SupportUnsupported SupportLevel = "unsupported"
)

type WorkClass string

const (
	WorkDefault WorkClass = "default"
	WorkGrind   WorkClass = "grind"
	WorkRigor   WorkClass = "rigor"
)

type OrchestrationProfile struct {
	Name           Profile `json:"name"`
	Strict         bool    `json:"strict,omitempty"`
	MaxWorkers     *int    `json:"max_workers,omitempty"`
	MaxTokens      *int64  `json:"max_tokens,omitempty"`
	Attended       *bool   `json:"attended,omitempty"`
	RequireWitness *bool   `json:"require_independent_witness,omitempty"`
}

type HarnessCapabilities struct {
	Concurrency        SupportLevel `json:"concurrency"`
	TaskMessaging      SupportLevel `json:"task_messaging"`
	Cancellation       SupportLevel `json:"cancellation"`
	Leases             SupportLevel `json:"leases"`
	IndependentWitness SupportLevel `json:"independent_witness"`
}

type TaskSpec struct {
	Schema     string    `json:"schema"`
	ID         string    `json:"id"`
	WorkClass  WorkClass `json:"work_class,omitempty"`
	Attended   *bool     `json:"attended,omitempty"`
	MaxWorkers *int      `json:"max_workers,omitempty"`
	MaxTokens  *int64    `json:"max_tokens,omitempty"`
	EngineRef  string    `json:"engine_ref,omitempty"`
}

type ChildAccessMode string

const (
	ChildAccessObserve ChildAccessMode = "observe"
	ChildAccessEffect  ChildAccessMode = "effect"
)

// ChildAccess is the provider-neutral capability declaration compiled by a
// native launch adapter. Observe carries no write footprint. Effect names the
// one bounded write region and tool set that the child may exercise.
type ChildAccess struct {
	Mode      ChildAccessMode `json:"mode"`
	Lane      string          `json:"lane,omitempty"`
	WriteTree string          `json:"write_tree,omitempty"`
	Tools     []string        `json:"tools,omitempty"`
}

type Role struct {
	ID      string      `json:"id"`
	Purpose string      `json:"purpose"`
	TaskID  string      `json:"task_id"`
	Access  ChildAccess `json:"access"`
}
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
}
type Budget struct {
	MaxWorkers int   `json:"max_workers"`
	MaxTokens  int64 `json:"max_tokens"`
}
type LeasePolicy struct {
	Scheduler string `json:"scheduler"`
	Required  bool   `json:"required"`
}
type WitnessPolicy struct {
	Independent    bool `json:"independent"`
	EffectReadback bool `json:"effect_readback"`
}
type ReconcilePolicy struct {
	Required bool   `json:"required"`
	Strategy string `json:"strategy"`
}
type InteractionPolicy struct {
	Attended      bool `json:"attended"`
	TaskMessaging bool `json:"task_messaging"`
	Cancellation  bool `json:"cancellation"`
}

type Degradation struct {
	Capability string       `json:"capability"`
	Required   SupportLevel `json:"required"`
	Available  SupportLevel `json:"available"`
	Reason     string       `json:"reason"`
}
type Provenance struct {
	Field  string `json:"field"`
	Source string `json:"source"`
	Value  any    `json:"value"`
}

type WorkflowPlan struct {
	Schema       string            `json:"schema"`
	Profile      Profile           `json:"profile"`
	TaskID       string            `json:"task_id"`
	WorkClass    WorkClass         `json:"work_class"`
	Roles        []Role            `json:"roles"`
	DAG          []Edge            `json:"dag"`
	Budget       Budget            `json:"budget"`
	Leases       LeasePolicy       `json:"leases"`
	Witness      WitnessPolicy     `json:"witness"`
	Reconcile    ReconcilePolicy   `json:"reconcile"`
	Interaction  InteractionPolicy `json:"interaction"`
	EngineRef    string            `json:"engine_ref"`
	SOLRoute     SOLRoute          `json:"sol_route"`
	Degradations []Degradation     `json:"degradations"`
	Explanation  []string          `json:"explanation"`
}

type Resolution struct {
	Schema       string               `json:"schema"`
	Requested    OrchestrationProfile `json:"requested"`
	Resolved     WorkflowPlan         `json:"resolved"`
	Overrides    []Provenance         `json:"overrides"`
	Degradations []Degradation        `json:"degradations"`
}

var ErrStrictDegradation = errors.New("strict orchestration rejects degradation")

// TaskFromText turns the current operator task into the smallest typed task
// accepted by the orchestration resolver. It intentionally classifies only the
// work shape the profile needs; the original text stays outside the receipt.
func TaskFromText(text string) (TaskSpec, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return TaskSpec{}, errors.New("task text is required")
	}
	words := strings.Fields(text)
	lower := strings.ToLower(text)
	workClass := WorkDefault
	trivial := len(words) <= 4 && !containsAny(lower, " and ", " then ", "multi-step", "parallel", "unattended", "overnight", "backlog", "workflow")
	if trivial {
		workClass = WorkDefault
	} else if containsAny(lower, "parallel", "concurrent", "independent", "fan out", "fanout", "fleet", "wave") {
		workClass = WorkGrind
	} else if containsAny(lower, "unattended", "overnight", "all night", "backlog", "drain") {
		workClass = WorkRigor
	} else if containsAny(lower, "multi-step", "multistep", "multiple steps", " and then ") || strings.Count(lower, " and ") >= 2 || isSerialActionList(lower) {
		workClass = WorkGrind
	}
	id := taskTextID(text)
	return TaskSpec{Schema: "fak-orchestration-task/1", ID: id, WorkClass: workClass}, nil
}

func isSerialActionList(text string) bool {
	clauses := strings.FieldsFunc(text, func(r rune) bool { return r == ',' || r == ';' })
	if len(clauses) < 3 {
		return false
	}
	actions := 0
	for _, clause := range clauses {
		clause = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(clause), "and "))
		verb, _, _ := strings.Cut(clause, " ")
		if containsAny(verb, "add", "audit", "build", "continue", "dogfood", "fix", "implement", "inspect", "measure", "produce", "run", "ship", "summarize", "test", "verify", "write") {
			actions++
		}
	}
	return actions >= 3
}

func containsAny(s string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

func taskTextID(text string) string {
	digest := sha256.Sum256([]byte(text))
	return "task-" + hex.EncodeToString(digest[:8])
}

func ParseTask(data []byte) (TaskSpec, error) {
	var t TaskSpec
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&t); err != nil {
		return t, fmt.Errorf("task fixture: %w", err)
	}
	if err := requireJSONEOF(d); err != nil {
		return t, fmt.Errorf("task fixture: %w", err)
	}
	if t.Schema != "fak-orchestration-task/1" {
		return t, fmt.Errorf("task fixture: unsupported schema %q", t.Schema)
	}
	if t.ID == "" {
		return t, errors.New("task fixture: id is required")
	}
	if t.WorkClass == "" {
		t.WorkClass = WorkDefault
	}
	if t.WorkClass != WorkDefault && t.WorkClass != WorkGrind && t.WorkClass != WorkRigor {
		return t, fmt.Errorf("task fixture: unknown work_class %q", t.WorkClass)
	}
	return t, nil
}

func ParseResolution(data []byte) (Resolution, error) {
	var r Resolution
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		return r, err
	}
	if err := requireJSONEOF(d); err != nil {
		return r, err
	}
	if r.Schema != SchemaVersion {
		return r, fmt.Errorf("unsupported schema %q", r.Schema)
	}
	return r, nil
}

func Resolve(req OrchestrationProfile, task TaskSpec, caps HarnessCapabilities) (Resolution, error) {
	if req.Name == "" {
		req.Name = ProfileAuto
	}
	if req.Name != ProfileOff && req.Name != ProfileAuto && req.Name != ProfileUltracode {
		return Resolution{}, fmt.Errorf("unknown profile %q", req.Name)
	}
	workers, tokens, attended, witness := 1, int64(4096), false, false
	prov := []Provenance{{"budget.max_workers", "base-default", workers}, {"budget.max_tokens", "base-default", tokens}, {"interaction.attended", "base-default", attended}, {"witness.independent", "base-default", witness}}
	resolvedProfile := req.Name
	if req.Name == ProfileAuto {
		switch task.WorkClass {
		case WorkGrind:
			workers, tokens = 4, 32768
			resolvedProfile = ProfileUltracode
		case WorkRigor:
			workers, tokens, witness = 3, 65536, true
			resolvedProfile = ProfileUltracode
		default:
			resolvedProfile = ProfileOff
		}
		prov = append(prov, Provenance{"profile", "task.work_class", resolvedProfile})
	} else if req.Name == ProfileUltracode {
		workers, tokens, witness = 4, 65536, true
		prov = append(prov, Provenance{"profile", "preset.ultracode", resolvedProfile})
	} else {
		resolvedProfile = ProfileOff
		prov = append(prov, Provenance{"profile", "preset.off", resolvedProfile})
	}
	if task.MaxWorkers != nil {
		workers = *task.MaxWorkers
		prov = append(prov, Provenance{"budget.max_workers", "task", workers})
	}
	if task.MaxTokens != nil {
		tokens = *task.MaxTokens
		prov = append(prov, Provenance{"budget.max_tokens", "task", tokens})
	}
	if task.Attended != nil {
		attended = *task.Attended
		prov = append(prov, Provenance{"interaction.attended", "task", attended})
	}
	if req.MaxWorkers != nil {
		workers = *req.MaxWorkers
		prov = append(prov, Provenance{"budget.max_workers", "operator", workers})
	}
	if req.MaxTokens != nil {
		tokens = *req.MaxTokens
		prov = append(prov, Provenance{"budget.max_tokens", "operator", tokens})
	}
	if req.Attended != nil {
		attended = *req.Attended
		prov = append(prov, Provenance{"interaction.attended", "operator", attended})
	}
	if req.RequireWitness != nil {
		witness = *req.RequireWitness
		prov = append(prov, Provenance{"witness.independent", "operator", witness})
	}
	if workers < 1 || tokens < 1 {
		return Resolution{}, errors.New("budgets must be positive")
	}
	multi := resolvedProfile != ProfileOff && workers > 1
	required := map[string]bool{"concurrency": multi, "task_messaging": multi, "cancellation": multi, "leases": multi, "independent_witness": witness}
	available := map[string]SupportLevel{"concurrency": caps.Concurrency, "task_messaging": caps.TaskMessaging, "cancellation": caps.Cancellation, "leases": caps.Leases, "independent_witness": caps.IndependentWitness}
	deg := []Degradation{}
	for _, name := range []string{"concurrency", "task_messaging", "cancellation", "leases", "independent_witness"} {
		if !required[name] {
			continue
		}
		level := available[name]
		if level == "" {
			level = SupportUnsupported
		}
		if level != SupportNative {
			deg = append(deg, Degradation{name, SupportNative, level, fmt.Sprintf("%s support is %s", strings.ReplaceAll(name, "_", " "), level)})
		}
	}
	if req.Strict && len(deg) > 0 {
		return Resolution{}, fmt.Errorf("%w: %s", ErrStrictDegradation, deg[0].Reason)
	}
	roles := []Role{{ID: "lead", Purpose: "decompose and reconcile", TaskID: task.ID, Access: ChildAccess{Mode: ChildAccessObserve}}}
	var dag []Edge
	if multi {
		for i := 1; i < workers; i++ {
			id := fmt.Sprintf("worker-%d", i)
			roles = append(roles, Role{ID: id, Purpose: "execute leased task", TaskID: task.ID, Access: ChildAccess{Mode: ChildAccessObserve}})
			dag = append(dag, Edge{id, "lead"})
		}
	}
	engine := task.EngineRef
	if engine == "" {
		engine = "executionroute:auto"
	}
	solRoute := SelectSOLRoute("", resolvedProfile, task.WorkClass, "gpt-5.6-sol")
	explain := []string{fmt.Sprintf("profile %s resolved from %s work", resolvedProfile, task.WorkClass), fmt.Sprintf("budget capped at %d workers and %d tokens", workers, tokens), fmt.Sprintf("task execution remains delegated to taskmgr with engine reference %s", engine)}
	for _, d := range deg {
		explain = append(explain, "degraded: "+d.Reason)
	}
	plan := WorkflowPlan{SchemaVersion, resolvedProfile, task.ID, task.WorkClass, roles, dag, Budget{workers, tokens}, LeasePolicy{"taskmgr", multi}, WitnessPolicy{witness, witness}, ReconcilePolicy{multi, "effect-readback"}, InteractionPolicy{attended, multi, multi}, engine, solRoute, deg, explain}
	return Resolution{SchemaVersion, req, plan, prov, deg}, nil
}

func StableJSON(r Resolution) ([]byte, error) {
	sort.Slice(r.Overrides, func(i, j int) bool {
		if r.Overrides[i].Field != r.Overrides[j].Field {
			return r.Overrides[i].Field < r.Overrides[j].Field
		}
		if r.Overrides[i].Source != r.Overrides[j].Source {
			return r.Overrides[i].Source < r.Overrides[j].Source
		}
		a, _ := json.Marshal(r.Overrides[i].Value)
		b, _ := json.Marshal(r.Overrides[j].Value)
		return bytes.Compare(a, b) < 0
	})
	return json.MarshalIndent(r, "", "  ")
}

func requireJSONEOF(d *json.Decoder) error {
	var extra any
	err := d.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return errors.New("trailing JSON")
}
