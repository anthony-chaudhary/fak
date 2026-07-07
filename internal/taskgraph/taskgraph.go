// Package taskgraph folds a shared task journal into a typed task table with
// lease-gated claims. It is the checkable replacement for the file-lock +
// only-mark-completed-when-done honor rule a team's shared task list rests on
// (#2437, part of the harness-native program #2387): tasks.jsonl is an
// append-only event journal (created / claimed / blocked / completed /
// abandoned) and Fold is a pure function to a typed table — same journal in,
// byte-identical table out, an injectable clock, no live state to corrupt.
//
// It is built on the toolproc discipline (internal/toolproc): a closed event
// vocabulary that fails closed at the boundary, a fold that tolerates documented
// benign races as findings and refuses impossible transitions, and a verdict
// vocabulary of closed reason tokens registered by the consumer. The two
// lease-gated refusals are the whole point — they turn a race into a closed
// reason a reader can witness from the journal alone:
//
//   - Claiming requires a LIVE lease admitted by the arbiter over the task's
//     declared tree. A claim under an already-dead lease is refused
//     (TASK_CLAIM_NO_LIVE_LEASE); a claim whose declared tree intersects a
//     task already live-claimed is refused (TASK_CLAIM_TREE_COLLISION), so two
//     workers structurally cannot hold colliding tasks.
//   - Completing a task with an open (not-yet-completed) blocker is refused
//     (TASK_COMPLETE_OPEN_BLOCKERS) instead of silently marking it done.
//
// The package stays a pure, init-free fold: it reserves a block in the
// registered ReasonCode range (above abi.ReasonCoreMax) but the CONSUMER
// registers the names (the `fak tasks` shell), the same egressfloor/toolproc
// pattern — internal/abi is human-owned and additive-only, so this leaf needs
// no defconfig entry.
package taskgraph

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The out-of-tree refusal codes this leaf's verdict tokens cite (above
// abi.ReasonCoreMax = 1023). egressfloor holds 1024 and toolproc holds
// 1040–1044; this leaf takes 1050–1052 and leaves the gap as headroom for the
// tool-lifecycle family.
const (
	ReasonTaskClaimNoLiveLease     abi.ReasonCode = 1050
	ReasonTaskCompleteOpenBlockers abi.ReasonCode = 1051
	ReasonTaskClaimTreeCollision   abi.ReasonCode = 1052
)

// The stable names for the codes above — the closed verdict vocabulary a finding
// cites. A finding never carries free text in Reason.
const (
	ReasonTaskClaimNoLiveLeaseName     = "TASK_CLAIM_NO_LIVE_LEASE"
	ReasonTaskCompleteOpenBlockersName = "TASK_COMPLETE_OPEN_BLOCKERS"
	ReasonTaskClaimTreeCollisionName   = "TASK_CLAIM_TREE_COLLISION"
)

// ReasonPair is one (code, name) row of this leaf's verdict vocabulary.
type ReasonPair struct {
	Code abi.ReasonCode
	Name string
}

// ReasonPairs lists the vocabulary for a consumer to abi.RegisterReason — kept
// as data so the registration call sites cannot drift from the constants.
func ReasonPairs() []ReasonPair {
	return []ReasonPair{
		{ReasonTaskClaimNoLiveLease, ReasonTaskClaimNoLiveLeaseName},
		{ReasonTaskCompleteOpenBlockers, ReasonTaskCompleteOpenBlockersName},
		{ReasonTaskClaimTreeCollision, ReasonTaskClaimTreeCollisionName},
	}
}

// EventKind is the CLOSED journal-event vocabulary. Parse fails closed: an
// unrecognized kind refuses the journal at the boundary, never coerces.
type EventKind string

const (
	EvCreated   EventKind = "created"   // a task enters the graph (declares its tree + blockers)
	EvClaimed   EventKind = "claimed"   // a worker claims a task under an arbiter-admitted lease
	EvBlocked   EventKind = "blocked"   // an edge added: this task now waits on more blockers
	EvCompleted EventKind = "completed" // the claimant marks the task done (refused if blockers are open)
	EvAbandoned EventKind = "abandoned" // the claimant releases the task back to the pool
)

// Event is one journal row. The journal is append-only JSONL: unknown FIELDS are
// tolerated (additive evolution), unknown enum TOKENS are refused (closed sets).
type Event struct {
	Kind EventKind `json:"kind"`
	Task string    `json:"task,omitempty"` // the task id this event acts on
	AtMS int64     `json:"at_unix_ms"`

	// created only — the declared task shape.
	Title string   `json:"title,omitempty"`
	Tree  []string `json:"tree,omitempty"` // declared file-tree globs the lease rides on

	// created + blocked — the blocker task ids this task waits on.
	BlockedBy []string `json:"blocked_by,omitempty"`

	// claimed only — the arbiter-admitted lease the claim rides on.
	Owner          string `json:"owner,omitempty"`                 // worker/session id
	Lease          string `json:"lease,omitempty"`                 // lease id admitted over the task's tree
	LeaseExpiresMS int64  `json:"lease_expires_unix_ms,omitempty"` // absolute lease expiry

	// completed / abandoned (optional) — a human note for the ledger.
	Note string `json:"note,omitempty"`
}

// Status is the CLOSED per-task lifecycle state after the fold.
type Status string

const (
	StatusOpen      Status = "OPEN"      // unblocked and unclaimed — dispatchable from evidence
	StatusBlocked   Status = "BLOCKED"   // at least one declared blocker is not yet completed
	StatusClaimed   Status = "CLAIMED"   // a live lease holds it (owner + unexpired lease)
	StatusCompleted Status = "COMPLETED" // the claimant marked it done with no open blockers
)

// Finding is one refusal verdict on one task: a closed reason token plus a
// deterministic human line (no wall-clock, only what the fold was given).
type Finding struct {
	Reason string `json:"reason"`
	Code   uint32 `json:"code"`
	Detail string `json:"detail"`
}

// Task is one folded row — what `fak tasks table` renders.
type Task struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`

	Status Status   `json:"status"`
	Tree   []string `json:"tree,omitempty"`

	// BlockedBy is every declared blocker; OpenBlockedBy is the subset not yet
	// completed at NowUnixMS — the "unblocked" evidence the dispatch reads.
	BlockedBy     []string `json:"blocked_by,omitempty"`
	OpenBlockedBy []string `json:"open_blocked_by,omitempty"`

	// The owner-lease column. Owner/Lease are set only while a claim is live;
	// LeaseLive says whether the lease is still admitted at NowUnixMS. A claim
	// whose lease has since aged out clears Owner and reads OPEN again — the
	// task is reclaimable, not silently held.
	Owner          string `json:"owner,omitempty"`
	Lease          string `json:"lease,omitempty"`
	LeaseExpiresMS int64  `json:"lease_expires_unix_ms,omitempty"`
	LeaseLive      bool   `json:"lease_live,omitempty"`

	CreatedMS int64 `json:"created_unix_ms"`
	UpdatedMS int64 `json:"updated_unix_ms,omitempty"`
	Abandons  int   `json:"abandons,omitempty"` // times a claim was released back to the pool

	Findings []Finding `json:"findings,omitempty"`
}

// Counts is the table's attention summary.
type Counts struct {
	Open      int `json:"open"`
	Blocked   int `json:"blocked"`
	Claimed   int `json:"claimed"`
	Completed int `json:"completed"`
	Refused   int `json:"refused"` // tasks carrying at least one refusal finding
}

// Config tunes the fold. Zero values are safe.
type Config struct {
	// LeaseGraceMS extends a lease's admitted window: a lease is live while
	// NowUnixMS < expiry + grace, and a claim is dead-on-arrival only when its
	// expiry + grace is already at or before the claim instant. 0 = no grace.
	LeaseGraceMS int64 `json:"lease_grace_ms,omitempty"`
}

// Table is the deterministic fold output: the shared task list at one instant.
type Table struct {
	Schema    string `json:"schema"`
	NowUnixMS int64  `json:"now_unix_ms"`
	Config    Config `json:"config"`
	Tasks     []Task `json:"tasks"`
	Counts    Counts `json:"counts"`
	// AttentionNeeded is true when any task carries a refusal finding — the
	// one-bit gate a supervisor loop or CI can key on.
	AttentionNeeded bool `json:"attention_needed"`
}

// TableSchema stamps the fold output.
const TableSchema = "fak.taskgraph-table.v1"

// ParseEvents reads a JSONL journal. Fail-closed: an unknown kind, a missing
// identity field, a bad enum token, or malformed JSON refuses the whole journal
// with the offending line number — fail at the boundary, do not guess.
func ParseEvents(r io.Reader) ([]Event, error) {
	var out []Event
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		var ev Event
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			return nil, fmt.Errorf("taskgraph: line %d: %v", line, err)
		}
		if err := ValidateEvent(ev); err != nil {
			return nil, fmt.Errorf("taskgraph: line %d: %v", line, err)
		}
		out = append(out, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("taskgraph: %v", err)
	}
	return out, nil
}

// ValidateEvent enforces the closed vocabulary and per-kind required fields.
func ValidateEvent(ev Event) error {
	if ev.AtMS <= 0 {
		return fmt.Errorf("event %q: at_unix_ms must be positive", ev.Kind)
	}
	if strings.TrimSpace(ev.Task) == "" {
		return fmt.Errorf("event %q: task id required", ev.Kind)
	}
	switch ev.Kind {
	case EvCreated:
		// tree/blockers are optional (a leaf task with no declared tree is legal).
	case EvClaimed:
		if strings.TrimSpace(ev.Owner) == "" {
			return fmt.Errorf("claimed %s: owner required", ev.Task)
		}
		if ev.LeaseExpiresMS < 0 {
			return fmt.Errorf("claimed %s: negative lease expiry", ev.Task)
		}
	case EvBlocked:
		if len(ev.BlockedBy) == 0 {
			return fmt.Errorf("blocked %s: blocked_by required", ev.Task)
		}
	case EvCompleted, EvAbandoned:
		// no extra required fields.
	default:
		return fmt.Errorf("unknown event kind %q", ev.Kind)
	}
	return nil
}

// task is the mutable fold accumulator for one task id.
type task struct {
	Task
	completed bool
	blockers  []string        // declared blockers, in first-seen order
	blockerAt map[string]bool // set membership for dedup
}

// Fold replays a validated journal into the task table at nowMS. It is a pure
// function: same events + same now + same config => byte-identical table.
//
// The fold is CHRONOLOGICAL: it walks the append-only journal in order, so a
// claim is admitted against the claims live at THAT instant and a completion is
// judged against the blockers open at THAT instant. Two refusals turn a race
// into a closed reason instead of corrupt state:
//
//   - a claim under a dead lease (empty lease id, or expiry already at/before the
//     claim instant) leaves the task unclaimed and flags TASK_CLAIM_NO_LIVE_LEASE;
//   - a claim whose declared tree intersects a task already live-claimed leaves
//     the task unclaimed and flags TASK_CLAIM_TREE_COLLISION;
//   - a completion while a declared blocker is still open leaves the task
//     un-completed and flags TASK_COMPLETE_OPEN_BLOCKERS.
//
// Tolerated benign events (documented, never guessed):
//   - a claim / block / complete / abandon on a task never created folds a
//     synthetic created row at that instant (the journal may be truncated at the
//     head); the task simply has no declared tree/title until one is seen.
//   - an abandon releases the current claim back to the pool (reclaimable),
//     counting one Abandon.
func Fold(events []Event, nowMS int64, cfg Config) (Table, error) {
	if nowMS <= 0 {
		return Table{}, fmt.Errorf("taskgraph: now_unix_ms must be positive")
	}
	if cfg.LeaseGraceMS < 0 {
		return Table{}, fmt.Errorf("taskgraph: negative lease grace")
	}

	tasks := map[string]*task{}
	var order []string

	ensure := func(id string, atMS int64) *task {
		t, ok := tasks[id]
		if !ok {
			t = &task{Task: Task{ID: id, CreatedMS: atMS}, blockerAt: map[string]bool{}}
			tasks[id] = t
			order = append(order, id)
		}
		return t
	}
	addBlockers := func(t *task, ids []string) {
		for _, b := range ids {
			b = strings.TrimSpace(b)
			if b == "" || b == t.ID || t.blockerAt[b] {
				continue
			}
			t.blockerAt[b] = true
			t.blockers = append(t.blockers, b)
		}
	}
	// leaseLiveAt reports whether an owner's lease still holds at instant ms.
	leaseLiveAt := func(t *task, ms int64) bool {
		return t.Owner != "" && t.LeaseExpiresMS+cfg.LeaseGraceMS > ms
	}

	for _, ev := range events {
		if err := ValidateEvent(ev); err != nil {
			return Table{}, fmt.Errorf("taskgraph: %v", err)
		}
		switch ev.Kind {
		case EvCreated:
			t := ensure(ev.Task, ev.AtMS)
			if ev.Title != "" {
				t.Title = ev.Title
			}
			if len(ev.Tree) > 0 {
				t.Tree = append([]string(nil), ev.Tree...)
			}
			addBlockers(t, ev.BlockedBy)
			t.UpdatedMS = ev.AtMS
		case EvBlocked:
			t := ensure(ev.Task, ev.AtMS)
			addBlockers(t, ev.BlockedBy)
			t.UpdatedMS = ev.AtMS
		case EvClaimed:
			t := ensure(ev.Task, ev.AtMS)
			t.UpdatedMS = ev.AtMS
			// Refuse a dead-on-arrival lease: no admitted lease can be live if its
			// window is already closed at the moment of claiming.
			if strings.TrimSpace(ev.Lease) == "" || ev.LeaseExpiresMS+cfg.LeaseGraceMS <= ev.AtMS {
				t.Findings = append(t.Findings, Finding{
					Reason: ReasonTaskClaimNoLiveLeaseName,
					Code:   uint32(ReasonTaskClaimNoLiveLease),
					Detail: fmt.Sprintf("claim by %s under an expired/absent lease (expires=%d, at=%d)", ev.Owner, ev.LeaseExpiresMS, ev.AtMS),
				})
				continue
			}
			// Refuse a claim whose declared tree collides with a task already
			// live-claimed at this instant — the arbiter would never admit two such
			// leases, so a journal that shows it is refused, not trusted.
			if collider := treeCollider(tasks, order, t.ID, ev.Tree, ev.AtMS, leaseLiveAt); collider != "" {
				t.Findings = append(t.Findings, Finding{
					Reason: ReasonTaskClaimTreeCollisionName,
					Code:   uint32(ReasonTaskClaimTreeCollision),
					Detail: fmt.Sprintf("claim by %s over %v collides with live-claimed task %s", ev.Owner, treeOrDeclared(ev.Tree, t.Tree), collider),
				})
				continue
			}
			if len(ev.Tree) > 0 {
				t.Tree = append([]string(nil), ev.Tree...)
			}
			t.Owner = ev.Owner
			t.Lease = ev.Lease
			t.LeaseExpiresMS = ev.LeaseExpiresMS
		case EvCompleted:
			t := ensure(ev.Task, ev.AtMS)
			t.UpdatedMS = ev.AtMS
			if open := openBlockers(tasks, t); len(open) > 0 {
				t.Findings = append(t.Findings, Finding{
					Reason: ReasonTaskCompleteOpenBlockersName,
					Code:   uint32(ReasonTaskCompleteOpenBlockers),
					Detail: fmt.Sprintf("complete refused: %d open blocker(s) %v", len(open), open),
				})
				continue
			}
			t.completed = true
			t.Owner, t.Lease, t.LeaseExpiresMS = "", "", 0
		case EvAbandoned:
			t := ensure(ev.Task, ev.AtMS)
			t.UpdatedMS = ev.AtMS
			if t.Owner != "" {
				t.Abandons++
			}
			t.Owner, t.Lease, t.LeaseExpiresMS = "", "", 0
		}
	}

	out := Table{Schema: TableSchema, NowUnixMS: nowMS, Config: cfg}
	for _, id := range order {
		t := tasks[id]
		finalizeTask(t, tasks, nowMS, leaseLiveAt)
		out.Tasks = append(out.Tasks, t.Task)
		switch t.Status {
		case StatusOpen:
			out.Counts.Open++
		case StatusBlocked:
			out.Counts.Blocked++
		case StatusClaimed:
			out.Counts.Claimed++
		case StatusCompleted:
			out.Counts.Completed++
		}
		if len(t.Findings) > 0 {
			out.Counts.Refused++
			out.AttentionNeeded = true
		}
	}
	return out, nil
}

// finalizeTask resolves a task's blocker lists, live-lease flag, and status at
// nowMS in a fixed precedence (completed > blocked > claimed > open) so the
// output is deterministic.
func finalizeTask(t *task, tasks map[string]*task, nowMS int64, leaseLiveAt func(*task, int64) bool) {
	t.BlockedBy = append([]string(nil), t.blockers...)
	t.OpenBlockedBy = openBlockers(tasks, t)

	// A claim whose lease has aged out at nowMS is no longer held: clear the
	// owner so the task reads reclaimable, but keep the recorded expiry visible.
	t.LeaseLive = leaseLiveAt(t, nowMS)
	if t.Owner != "" && !t.LeaseLive {
		t.Owner = ""
		t.Lease = ""
	}

	switch {
	case t.completed:
		t.Status = StatusCompleted
	case len(t.OpenBlockedBy) > 0:
		t.Status = StatusBlocked
	case t.LeaseLive && t.Owner != "":
		t.Status = StatusClaimed
	default:
		t.Status = StatusOpen
	}
}

// openBlockers returns the declared blockers of t that are not completed, in
// declared order. An unknown blocker id (never created in this journal) is
// treated as OPEN — the fold never completes past a blocker it cannot witness.
func openBlockers(tasks map[string]*task, t *task) []string {
	var open []string
	for _, b := range t.blockers {
		if bt, ok := tasks[b]; ok && bt.completed {
			continue
		}
		open = append(open, b)
	}
	return open
}

// treeCollider returns the id of a task, other than self, that is live-claimed
// at instant ms and whose tree intersects want (falling back to self's already
// declared tree when the claim declared none). "" means no collision. order is
// walked so the result is deterministic (first colliding task by creation order).
func treeCollider(tasks map[string]*task, order []string, self string, want []string, ms int64, leaseLiveAt func(*task, int64) bool) string {
	trees := want
	if len(trees) == 0 {
		if st, ok := tasks[self]; ok {
			trees = st.Tree
		}
	}
	if len(trees) == 0 {
		return "" // a claim with no declared tree cannot be proven to collide
	}
	for _, id := range order {
		if id == self {
			continue
		}
		ot := tasks[id]
		if !leaseLiveAt(ot, ms) || len(ot.Tree) == 0 {
			continue
		}
		if treesIntersect(trees, ot.Tree) {
			return id
		}
	}
	return ""
}

func treeOrDeclared(want, declared []string) []string {
	if len(want) > 0 {
		return want
	}
	return declared
}

// treesIntersect reports whether two file-tree glob sets can name a common path.
// It compares the fixed (non-wildcard) prefixes: two globs overlap when one
// prefix is a path-prefix of the other (e.g. "internal/**" covers
// "internal/taskgraph/**"). This is the same containment rule lane disjointness
// rests on and is intentionally conservative — it never misses a real overlap,
// and at worst flags a pair that a finer matcher would separate.
func treesIntersect(a, b []string) bool {
	for _, ga := range a {
		pa := globPrefix(ga)
		for _, gb := range b {
			pb := globPrefix(gb)
			if pathPrefix(pa, pb) || pathPrefix(pb, pa) {
				return true
			}
		}
	}
	return false
}

// globPrefix returns the fixed directory prefix of a glob — everything up to the
// first wildcard, trimmed to a path boundary and normalized to forward slashes.
func globPrefix(g string) string {
	g = strings.ReplaceAll(strings.TrimSpace(g), "\\", "/")
	if i := strings.IndexAny(g, "*?["); i >= 0 {
		g = g[:i]
	}
	if j := strings.LastIndex(g, "/"); j >= 0 {
		g = g[:j+1]
	} else {
		g = ""
	}
	return g
}

// pathPrefix reports whether a is a path-prefix of b (both slash-normalized,
// trailing-slash tolerant). An empty a (a glob rooted at the repo) prefixes
// everything.
func pathPrefix(a, b string) bool {
	a = strings.TrimSuffix(a, "/")
	b = strings.TrimSuffix(b, "/")
	if a == "" {
		return true
	}
	return a == b || strings.HasPrefix(b, a+"/")
}

// Dispatchable returns the ids of tasks that are OPEN — unblocked and unclaimed
// from the journal alone — in table order. This is the evidence the dispatch
// tick reads to route unblocked-and-unclaimed work (#2437).
func Dispatchable(t Table) []string {
	var out []string
	for _, tk := range t.Tasks {
		if tk.Status == StatusOpen {
			out = append(out, tk.ID)
		}
	}
	return out
}

// RenderText writes the human table `fak tasks table` prints: a summary line,
// then one row per task with the owner-lease, status, and blockedBy columns the
// acceptance names — from the folded journal alone. A refusal finding renders as
// a `!!` line under its task so a race-turned-reason is visible at a glance.
func RenderText(w io.Writer, t Table) {
	fmt.Fprintf(w, "tasks: now_unix_ms=%d open=%d blocked=%d claimed=%d completed=%d refused=%d attention=%t\n",
		t.NowUnixMS, t.Counts.Open, t.Counts.Blocked, t.Counts.Claimed,
		t.Counts.Completed, t.Counts.Refused, t.AttentionNeeded)
	fmt.Fprintf(w, "  %-16s %-9s %-22s %s\n", "TASK", "STATUS", "OWNER-LEASE", "BLOCKED-BY")
	for _, tk := range t.Tasks {
		owner := "-"
		if tk.LeaseLive && tk.Owner != "" {
			owner = tk.Owner + "@" + tk.Lease
		}
		blocked := "-"
		if len(tk.OpenBlockedBy) > 0 {
			blocked = strings.Join(tk.OpenBlockedBy, ",")
		}
		fmt.Fprintf(w, "  %-16s %-9s %-22s %s\n", tk.ID, string(tk.Status), owner, blocked)
		for _, f := range tk.Findings {
			fmt.Fprintf(w, "    !! %s: %s\n", f.Reason, f.Detail)
		}
	}
}

// Sample returns a deterministic built-in journal + fold instant that exercises
// every status and both lease refusals — the no-key, no-model, no-GPU proof
// `fak tasks sample` renders. Fixed epoch so the output is byte-stable.
func Sample() ([]Event, int64, Config) {
	const base int64 = 1_700_000_000_000
	now := base + 60_000
	events := []Event{
		// a completed root that unblocks its dependents.
		{Kind: EvCreated, Task: "spine", Title: "ship the spine", Tree: []string{"internal/taskgraph/**"}, AtMS: base},
		{Kind: EvClaimed, Task: "spine", Owner: "w1", Lease: "L-spine", LeaseExpiresMS: base + 300_000, AtMS: base + 1_000},
		{Kind: EvCompleted, Task: "spine", AtMS: base + 5_000},
		// open + dispatchable now its only blocker is done.
		{Kind: EvCreated, Task: "cli", Title: "fak tasks table", Tree: []string{"cmd/fak/**"}, BlockedBy: []string{"spine"}, AtMS: base + 500},
		// still blocked: waits on a task not yet complete.
		{Kind: EvCreated, Task: "wire", Title: "wire the fold into dispatch", Tree: []string{"cmd/fak/dispatch_tick_route.go"}, BlockedBy: []string{"cli"}, AtMS: base + 700},
		// live-claimed under a healthy lease.
		{Kind: EvCreated, Task: "docs", Title: "concept note", Tree: []string{"docs/**"}, AtMS: base + 800},
		{Kind: EvClaimed, Task: "docs", Owner: "w2", Lease: "L-docs", LeaseExpiresMS: base + 300_000, AtMS: base + 2_000},
		// refused: claim under an already-dead lease.
		{Kind: EvCreated, Task: "flaky", Title: "claim with a stale lease", Tree: []string{"internal/other/**"}, AtMS: base + 900},
		{Kind: EvClaimed, Task: "flaky", Owner: "w3", Lease: "L-old", LeaseExpiresMS: base + 950, AtMS: base + 3_000},
	}
	return events, now, Config{}
}
