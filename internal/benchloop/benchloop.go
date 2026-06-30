package benchloop

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
	"github.com/anthony-chaudhary/fak/internal/benchruns"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

const StatusSchema = "fak.bench-loop.status.v1"

type Options struct {
	Root        string
	OverlayPath string
	LedgerPath  string
	Now         time.Time
}

type Parts struct {
	Root        string
	Now         time.Time
	Benchmarks  []benchcatalog.Bench
	Catalog     benchruns.Catalog
	CatalogErr  error
	CatalogPath string
	Ledger      []nightrun.CollectRow
	LedgerPath  string
	Caps        nightrun.Capabilities
	Tasks       []nightrun.Task
	TaskErr     error
	Authority   nightrun.LedgerGapReport
}

type Report struct {
	Schema      string          `json:"schema"`
	GeneratedAt string          `json:"generated_at"`
	Root        string          `json:"root"`
	Registry    RegistryStatus  `json:"registry"`
	Catalog     CatalogStatus   `json:"catalog"`
	Ledger      LedgerStatus    `json:"ledger"`
	Local       LocalStatus     `json:"local"`
	Authority   AuthorityStatus `json:"authority"`
	NextAction  Action          `json:"next_action"`
	Walk        []Surface       `json:"walk"`
}

type RegistryStatus struct {
	Benchmarks int `json:"benchmarks"`
	Offline    int `json:"offline"`
	Weights    int `json:"weights"`
	Dataset    int `json:"dataset"`
	Serving    int `json:"serving"`
	E2E        int `json:"e2e"`
	Manual     int `json:"manual"`
}

type CatalogStatus struct {
	Path            string `json:"path"`
	Present         bool   `json:"present"`
	Error           string `json:"error,omitempty"`
	RunCount        int    `json:"run_count"`
	MachineCount    int    `json:"machine_count"`
	LatestRunID     string `json:"latest_run_id,omitempty"`
	LatestMachine   string `json:"latest_machine,omitempty"`
	LatestModel     string `json:"latest_model,omitempty"`
	LatestPrecision string `json:"latest_precision,omitempty"`
	LatestTimestamp string `json:"latest_timestamp,omitempty"`
}

type LedgerStatus struct {
	Path          string `json:"path"`
	Rows          int    `json:"rows"`
	CollectedRows int    `json:"collected_rows"`
	LastDate      string `json:"last_date,omitempty"`
	LastTaskID    string `json:"last_task_id,omitempty"`
	LastOutcome   string `json:"last_outcome,omitempty"`
	LastBox       string `json:"last_box,omitempty"`
}

type LocalStatus struct {
	Capabilities nightrun.Capabilities `json:"capabilities"`
	TaskCount    int                   `json:"task_count"`
	Feasible     int                   `json:"feasible"`
	Blocked      int                   `json:"blocked"`
	Saturated    int                   `json:"saturated"`
	Error        string                `json:"error,omitempty"`
	HasNext      bool                  `json:"has_next"`
	Next         *NextTask             `json:"next,omitempty"`
}

type NextTask struct {
	ID         string  `json:"id"`
	Title      string  `json:"title"`
	Source     string  `json:"source"`
	Value      string  `json:"value"`
	Score      float64 `json:"score"`
	Reason     string  `json:"reason"`
	Run        string  `json:"run"`
	Acceptance string  `json:"acceptance"`
	Manual     bool    `json:"manual,omitempty"`
}

type AuthorityStatus struct {
	Date           string `json:"date"`
	NewerThan      int    `json:"newer_than"`
	TotalRows      int    `json:"total_rows"`
	TotalCollected int    `json:"total_collected"`
}

type Action struct {
	Kind    string `json:"kind"`
	Command string `json:"command"`
	Detail  string `json:"detail"`
}

type Surface struct {
	Name    string `json:"name"`
	Command string `json:"command"`
	Use     string `json:"use"`
}

func Load(opts Options) Report {
	root := opts.Root
	if root == "" {
		root, _ = os.Getwd()
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	catalogPath := filepath.Join(root, filepath.FromSlash(benchruns.CatalogRel))
	catalog, catalogErr := benchruns.LoadCatalog(root)
	ledgerPath := opts.LedgerPath
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, filepath.FromSlash(nightrun.DefaultLedgerRel))
	}
	ledger := nightrun.ReadLedgerFile(ledgerPath)
	overlay := opts.OverlayPath
	if overlay == "" {
		def := filepath.Join(root, filepath.FromSlash(nightrun.DefaultOverlayRel))
		if _, err := os.Stat(def); err == nil {
			overlay = def
		}
	}
	tasks, taskErr := nightrun.Backlog(overlay)
	caps := nightrun.ProbeLocal(root)
	return StatusFromParts(Parts{
		Root:        root,
		Now:         now,
		Benchmarks:  benchcatalog.All(),
		Catalog:     catalog,
		CatalogErr:  catalogErr,
		CatalogPath: catalogPath,
		Ledger:      ledger,
		LedgerPath:  ledgerPath,
		Caps:        caps,
		Tasks:       tasks,
		TaskErr:     taskErr,
		Authority:   nightrun.CompareWithAuthority(ledger, root),
	})
}

func StatusFromParts(p Parts) Report {
	now := p.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	if p.CatalogPath == "" && p.Root != "" {
		p.CatalogPath = filepath.Join(p.Root, filepath.FromSlash(benchruns.CatalogRel))
	}
	if p.LedgerPath == "" && p.Root != "" {
		p.LedgerPath = filepath.Join(p.Root, filepath.FromSlash(nightrun.DefaultLedgerRel))
	}
	r := Report{
		Schema:      StatusSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Root:        p.Root,
		Registry:    registryStatus(p.Benchmarks),
		Catalog:     catalogStatus(p.CatalogPath, p.Catalog, p.CatalogErr),
		Ledger:      ledgerStatus(p.LedgerPath, p.Ledger),
		Authority: AuthorityStatus{
			Date:           p.Authority.AuthorityDate,
			NewerThan:      len(p.Authority.NewerThan),
			TotalRows:      p.Authority.TotalRows,
			TotalCollected: p.Authority.TotalCollected,
		},
		Walk: Walk(),
	}
	r.Local = localStatus(p.Caps, p.Tasks, p.Ledger, now, p.TaskErr)
	r.NextAction = chooseAction(r)
	return r
}

func Walk() []Surface {
	return []Surface{
		{Name: "status", Command: "fak bench-loop status", Use: "one read-only benchmark control-plane snapshot"},
		{Name: "enter", Command: "fak bench-loop next", Use: "print the single next benchmark-loop action"},
		{Name: "discover", Command: "fak benchmarks list --offline", Use: "find benchmark entry points and zero-asset starts"},
		{Name: "local-next", Command: "fak nightrun next", Use: "show the highest-value datum this box can collect"},
		{Name: "local-loop", Command: "fak bench-loop run --apply --loop", Use: "execute the feasible local queue through the nightrun runner and append observed ledger rows"},
		{Name: "runs", Command: "fak bench-runs summary", Use: "query recorded benchmark catalog runs"},
		{Name: "request", Command: "fak bench request --now <STAMP> --dry-run", Use: "post or preview the next-test-per-machine request"},
		{Name: "gap", Command: "fak nightrun gap", Use: "show collected rows newer than the benchmark authority"},
		{Name: "rollup", Command: "fak bench post --rollup latest --dry-run", Use: "post or preview the bench-channel latest-run rollup"},
	}
}

func registryStatus(list []benchcatalog.Bench) RegistryStatus {
	var s RegistryStatus
	s.Benchmarks = len(list)
	for _, b := range list {
		switch b.Need {
		case benchcatalog.NeedNone:
			s.Offline++
		case benchcatalog.NeedWeights:
			s.Weights++
		case benchcatalog.NeedDataset:
			s.Dataset++
		}
		switch b.Level {
		case benchcatalog.LevelServing:
			s.Serving++
		case benchcatalog.LevelE2E:
			s.E2E++
		}
		if b.Manual {
			s.Manual++
		}
	}
	return s
}

func catalogStatus(path string, cat benchruns.Catalog, err error) CatalogStatus {
	s := CatalogStatus{Path: filepath.ToSlash(path), Present: err == nil}
	if err != nil {
		s.Error = err.Error()
		return s
	}
	s.RunCount = len(cat.Runs)
	machines := map[string]bool{}
	var latest benchruns.Run
	for _, run := range cat.Runs {
		if m := runString(run, "machine_id"); m != "" {
			machines[m] = true
		}
		if latest == nil || runString(run, "timestamp") > runString(latest, "timestamp") {
			latest = run
		}
	}
	s.MachineCount = len(machines)
	if latest != nil {
		s.LatestRunID = runString(latest, "run_id")
		s.LatestMachine = runString(latest, "machine_id")
		s.LatestModel = runString(latest, "model")
		s.LatestPrecision = runString(latest, "precision")
		s.LatestTimestamp = runString(latest, "timestamp")
	}
	return s
}

func ledgerStatus(path string, rows []nightrun.CollectRow) LedgerStatus {
	s := LedgerStatus{Path: filepath.ToSlash(path), Rows: len(rows)}
	var latest *nightrun.CollectRow
	for i := range rows {
		row := rows[i]
		if nightrun.CollectedOutcome(nightrun.Outcome(row.Outcome)) {
			s.CollectedRows++
		}
		if latest == nil || rowLater(row, *latest) {
			r := row
			latest = &r
		}
	}
	if latest != nil {
		s.LastDate = latest.Date
		s.LastTaskID = latest.TaskID
		s.LastOutcome = latest.Outcome
		s.LastBox = latest.Box
	}
	return s
}

func localStatus(caps nightrun.Capabilities, tasks []nightrun.Task, ledger []nightrun.CollectRow, now time.Time, err error) LocalStatus {
	s := LocalStatus{Capabilities: caps, TaskCount: len(tasks)}
	if err != nil {
		s.Error = err.Error()
		return s
	}
	ranked := nightrun.Rank(tasks, caps, ledger, now)
	for _, row := range ranked {
		if row.Feasible {
			s.Feasible++
		} else {
			s.Blocked++
		}
		if row.Saturated {
			s.Saturated++
		}
		if s.Next == nil && row.Feasible && !row.Saturated {
			next := nextTask(row)
			s.Next = &next
			s.HasNext = true
		}
	}
	return s
}

func nextTask(s nightrun.Scored) NextTask {
	t := s.Task
	return NextTask{
		ID:         t.ID,
		Title:      t.Title,
		Source:     string(t.Source),
		Value:      string(t.Value),
		Score:      s.Score,
		Reason:     s.Reason,
		Run:        t.Run,
		Acceptance: t.Acceptance,
		Manual:     t.Manual,
	}
}

func chooseAction(r Report) Action {
	if r.Catalog.Error != "" {
		return Action{
			Kind:    "refresh_catalog",
			Command: "python tools/bench_catalog.py build",
			Detail:  "benchmark catalog is missing or unreadable: " + r.Catalog.Error,
		}
	}
	if r.Catalog.RunCount == 0 {
		return Action{
			Kind:    "populate_catalog",
			Command: "python tools/bench_catalog.py build",
			Detail:  "benchmark catalog has no runs yet",
		}
	}
	if r.Local.Error != "" {
		return Action{
			Kind:    "fix_backlog",
			Command: "fak nightrun plan",
			Detail:  "benchmark collection backlog could not be assembled: " + r.Local.Error,
		}
	}
	if r.Local.HasNext && r.Local.Next != nil {
		if r.Local.Next.Manual {
			return Action{
				Kind:    "manual_collect",
				Command: r.Local.Next.Run,
				Detail:  fmt.Sprintf("%s is an operator recipe; run it by hand, then record/fold the observed result", r.Local.Next.ID),
			}
		}
		return Action{
			Kind:    "collect_local",
			Command: "fak bench-loop run --apply",
			Detail:  fmt.Sprintf("collect %s on this box; add --loop to continue through the feasible queue", r.Local.Next.ID),
		}
	}
	if r.Authority.NewerThan > 0 {
		return Action{
			Kind:    "publish_authority",
			Command: "fak nightrun gap",
			Detail:  fmt.Sprintf("%d collected row(s) are newer than BENCHMARK-AUTHORITY.md", r.Authority.NewerThan),
		}
	}
	if r.Local.Feasible == 0 {
		return Action{
			Kind:    "resolve_capability",
			Command: "fak nightrun plan",
			Detail:  "no benchmark task is feasible on this box; inspect blocked capabilities",
		}
	}
	return Action{
		Kind:    "review_status",
		Command: "fak bench-loop walk",
		Detail:  "no unsaturated local datum is selected; inspect the benchmark surfaces",
	}
}

func rowLater(a, b nightrun.CollectRow) bool {
	if a.GeneratedAt != "" || b.GeneratedAt != "" {
		return a.GeneratedAt > b.GeneratedAt
	}
	return a.Date > b.Date
}

func runString(r benchruns.Run, key string) string {
	if r == nil {
		return ""
	}
	if v, ok := r[key].(string); ok {
		return v
	}
	return ""
}

func RenderStatus(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "benchmark super-loop status\n")
	fmt.Fprintf(&b, "generated: %s\n", r.GeneratedAt)
	fmt.Fprintf(&b, "root: %s\n\n", blank(r.Root))
	fmt.Fprintf(&b, "registry: %d benchmark(s), %d offline, %d weights, %d dataset, %d serving, %d e2e, %d manual\n",
		r.Registry.Benchmarks, r.Registry.Offline, r.Registry.Weights, r.Registry.Dataset, r.Registry.Serving, r.Registry.E2E, r.Registry.Manual)
	if r.Catalog.Error != "" {
		fmt.Fprintf(&b, "catalog: MISSING (%s)\n", r.Catalog.Error)
	} else {
		fmt.Fprintf(&b, "catalog: %d run(s) across %d machine(s); latest %s %s %s/%s\n",
			r.Catalog.RunCount, r.Catalog.MachineCount, blank(r.Catalog.LatestTimestamp),
			blank(r.Catalog.LatestMachine), blank(r.Catalog.LatestModel), blank(r.Catalog.LatestPrecision))
	}
	fmt.Fprintf(&b, "ledger: %d row(s), %d collected; last %s %s %s on %s\n",
		r.Ledger.Rows, r.Ledger.CollectedRows, blank(r.Ledger.LastDate), blank(r.Ledger.LastTaskID),
		blank(r.Ledger.LastOutcome), blank(r.Ledger.LastBox))
	fmt.Fprintf(&b, "local: box=%s gpu=%s weights=%s datasets=%s net=%s tasks=%d feasible=%d blocked=%d saturated=%d\n",
		blank(r.Local.Capabilities.Box), gpu(r.Local.Capabilities.GPU), yesno(r.Local.Capabilities.Weights),
		yesno(r.Local.Capabilities.Datasets), yesno(r.Local.Capabilities.Net),
		r.Local.TaskCount, r.Local.Feasible, r.Local.Blocked, r.Local.Saturated)
	if r.Local.Error != "" {
		fmt.Fprintf(&b, "local-error: %s\n", r.Local.Error)
	}
	fmt.Fprintf(&b, "authority: last-updated=%s; newer-collected=%d\n\n", blank(r.Authority.Date), r.Authority.NewerThan)
	fmt.Fprintf(&b, "next action: %s\n", r.NextAction.Kind)
	fmt.Fprintf(&b, "  command: %s\n", r.NextAction.Command)
	fmt.Fprintf(&b, "  detail : %s\n", r.NextAction.Detail)
	if r.Local.Next != nil {
		fmt.Fprintf(&b, "  task   : %s (%s, score %.3f)\n", r.Local.Next.ID, r.Local.Next.Value, r.Local.Next.Score)
		fmt.Fprintf(&b, "  why    : %s\n", r.Local.Next.Reason)
	}
	return b.String()
}

func RenderNext(a Action) string {
	return fmt.Sprintf("%s\ncommand: %s\ndetail: %s\n", a.Kind, a.Command, a.Detail)
}

func RenderWalk(surfaces []Surface) string {
	var b strings.Builder
	fmt.Fprintf(&b, "benchmark super-loop walk\n\n")
	width := 0
	for _, s := range surfaces {
		if len(s.Name) > width {
			width = len(s.Name)
		}
	}
	for _, s := range surfaces {
		fmt.Fprintf(&b, "%-*s  %s\n", width, s.Name, s.Command)
		fmt.Fprintf(&b, "%-*s  %s\n", width, "", s.Use)
	}
	return b.String()
}

func blank(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func yesno(v bool) string {
	if v {
		return "yes"
	}
	return "no"
}

func gpu(s string) string {
	if strings.TrimSpace(s) == "" {
		return "none"
	}
	return s
}
