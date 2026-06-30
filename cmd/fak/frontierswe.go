package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// frontiersweSampleTasks is the committed task tree `fak frontierswe describe`
// falls back to when no --tasks dir is given, so the advertised RUNNABLE-NOW
// entry point works with zero args and zero external assets — no GPU, no Modal
// account, no (large) Docker images. It mirrors the offline-fallback contract
// `fak swebench describe` has with its committed difficulty sample. The tree
// holds the small set of upstream task.toml fixtures committed to the repo; the
// catalog (the 17 canonical task names + scoring families) is always in-binary,
// so describe lists all 17 even where a per-task fixture is not yet committed.
const frontiersweSampleTasks = "testdata/frontierswe"

// fak frontierswe — FrontierSWE (Proximal Labs' time-to-solution benchmark of 17
// long-horizon engineering tasks) as a fak-native surface. The headline cost is
// the [agent] timeout_sec: a 20-hour (72000s) wall-clock budget per task. That is
// the number the value question hangs on — "how much of that 20h is re-prefill
// work fak eliminates?" — so describe puts it front and centre, before any number
// is claimed.
//
// Subcommands:
//
//	describe — load the task catalog (the 17 names + scoring family), overlay the
//	           20h budget + resource envelope from each committed task.toml, and
//	           print per task: name, category, difficulty, the 20h budget, the
//	           [environment] envelope (cpus/memory_mb/gpus/allow_internet), and the
//	           scoring gate class. Fully offline; needs no model, GPU, or network.
//	           RUNNABLE NOW.
func cmdFrontierswe(argv []string) {
	if len(argv) == 0 {
		os.Exit(runFrontierswe(os.Stdout, os.Stderr, []string{"describe"}))
	}
	os.Exit(runFrontierswe(os.Stdout, os.Stderr, argv))
}

func runFrontierswe(stdout, stderr io.Writer, argv []string) int {
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "describe", "show":
		return runFrontiersweDescribe(stdout, stderr, rest)
	case "-h", "--help", "help":
		frontiersweUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak frontierswe: unknown subcommand %q\n", sub)
		frontiersweUsage(stderr)
		return 2
	}
}

func frontiersweUsage(w io.Writer) {
	fmt.Fprint(w, `fak frontierswe — FrontierSWE, the 17-task time-to-solution benchmark

usage:
  fak frontierswe describe [--tasks DIR] [--json] [--out FILE]
        Load the FrontierSWE task catalog (the 17 task names + their scoring
        family) and print, per task: name, category, difficulty, the 20-hour
        [agent] timeout_sec (the HEADLINE cost), the [environment] envelope
        (cpus / memory_mb / gpus / allow_internet), and the scoring gate class
        (implementation / performance / ml_research). Fully offline: with no
        flags it reads the committed task fixtures (`+frontiersweSampleTasks+`) and
        announces it on stderr — no model, GPU, Modal account, or Docker image.
`)
}

// FrontierTaskRow is one row of the describe output: the catalog facts (always
// known: Name, ScoringCategory/GateClass) plus the per-task envelope overlaid
// from a committed task.toml when one exists. HasFixture is false when no
// task.toml is committed for this task yet — the catalog still lists it, but the
// budget/environment fields are not authored and stay zero.
type FrontierTaskRow struct {
	Name      string `json:"name"`
	GateClass string `json:"gate_class"` // the scoring family: implementation/performance/ml_research

	// Overlaid from task.toml when HasFixture is true.
	HasFixture     bool    `json:"has_fixture"`
	Difficulty     string  `json:"difficulty,omitempty"` // [metadata] difficulty (free-form: hard/very_hard/frontier)
	Category       string  `json:"category,omitempty"`   // [metadata] category (free-form blurb)
	AgentTimeoutS  int64   `json:"agent_timeout_sec"`    // [agent] timeout_sec — the 20h headline budget
	AgentTimeoutHr float64 `json:"agent_timeout_hours"`  // the same budget in hours, for the headline
	CPUs           int     `json:"cpus"`                 // [environment]
	MemoryMB       int     `json:"memory_mb"`            // [environment]
	GPUs           int     `json:"gpus"`                 // [environment]
	AllowInternet  bool    `json:"allow_internet"`       // [environment]
}

// FrontierDescribe is the full describe payload: every catalog task as a row,
// plus the headline (the canonical 20h budget and the task count).
type FrontierDescribe struct {
	Source        string            `json:"source"`
	TaskCount     int               `json:"task_count"`
	FixtureCount  int               `json:"fixture_count"`         // how many tasks have a committed task.toml
	HeadlineHours float64           `json:"headline_budget_hours"` // the canonical 20h per-task agent budget
	GateClasses   map[string]int    `json:"gate_classes"`          // count of tasks per scoring family
	Tasks         []FrontierTaskRow `json:"tasks"`
}

func runFrontiersweDescribe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe describe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tasks := fs.String("tasks", "", "task tree to overlay budgets/resources from (dir of <name>/task.toml); default: the committed "+frontiersweSampleTasks+" sample")
	asJSON := fs.Bool("json", false, "emit only the describe JSON on stdout (no human table on stderr)")
	out := fs.String("out", "", "write the describe JSON here (default: stdout JSON + human table on stderr)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// describe is the advertised RUNNABLE-NOW entry point: it must work with no
	// flags. When no --tasks dir is given, fall back to the committed task fixtures
	// so a newcomer sees the real budgets/resources on the first command. The
	// catalog (the 17 names + scoring families) is always in-binary, so describe
	// lists all 17 regardless of how many per-task fixtures are committed.
	tasksDir := *tasks
	if tasksDir == "" {
		tasksDir = frontiersweSampleTasks
		fmt.Fprintf(stderr, "fak frontierswe describe: no --tasks dir; using the committed fixtures %s (offline; no model, GPU, or Docker).\n", tasksDir)
	}

	d := buildFrontierDescribe(tasksDir)

	jb, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe describe: %v\n", err)
		return 1
	}
	if *out != "" {
		if err := os.WriteFile(*out, jb, 0o644); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe describe: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, string(jb))
	}

	if !*asJSON {
		printFrontierSummary(stderr, d, *out)
	}
	return 0
}

// buildFrontierDescribe folds the in-binary catalog (the 17 task names + scoring
// families) with whatever per-task task.toml fixtures live under tasksDir into the
// describe payload. The catalog is the source of truth for which tasks exist and
// their gate class; tasksDir supplies the budget/resource overlay where present.
func buildFrontierDescribe(tasksDir string) FrontierDescribe {
	names := frontierswe.TaskNames() // all 17, sorted
	d := FrontierDescribe{
		Source:        tasksDir,
		TaskCount:     len(names),
		HeadlineHours: 72000.0 / 3600.0, // the canonical 20h per-task agent budget
		GateClasses:   map[string]int{},
		Tasks:         make([]FrontierTaskRow, 0, len(names)),
	}

	for _, name := range names {
		cat, _ := frontierswe.CategoryOf(name) // every catalog name resolves
		row := FrontierTaskRow{
			Name:      name,
			GateClass: cat.String(),
		}
		d.GateClasses[cat.String()]++

		// Overlay the 20h budget + envelope from a committed task.toml when present.
		if task, err := frontierswe.LoadTask(filepath.Join(tasksDir, name)); err == nil {
			row.HasFixture = true
			row.Difficulty = task.Metadata.Difficulty
			row.Category = task.Metadata.Category
			row.AgentTimeoutS = int64(task.AgentTimeoutSec())
			row.AgentTimeoutHr = task.AgentTimeoutSec() / 3600.0
			row.CPUs = task.Environment.CPUs
			row.MemoryMB = task.Environment.MemoryMB
			row.GPUs = task.Environment.GPUs
			row.AllowInternet = task.Environment.AllowInternet
			d.FixtureCount++
		}

		d.Tasks = append(d.Tasks, row)
	}
	return d
}

// printFrontierSummary writes the human-readable describe table on stderr (so
// stdout stays clean JSON when piped) — the same stdout-JSON / stderr-table split
// `fak swebench describe` uses.
func printFrontierSummary(w io.Writer, d FrontierDescribe, out string) {
	fmt.Fprintf(w, "\n== fak frontierswe describe ==\n")
	fmt.Fprintf(w, "source        : %s\n", d.Source)
	fmt.Fprintf(w, "tasks         : %d  (%d with a committed budget/resource fixture)\n", d.TaskCount, d.FixtureCount)
	fmt.Fprintf(w, "headline cost : %.0fh per-task agent budget (the [agent] timeout_sec — the wall-clock that dominates a run)\n", d.HeadlineHours)
	fmt.Fprintf(w, "gate classes  : %s\n", sortedGateClasses(d.GateClasses))
	fmt.Fprintln(w)

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "TASK\tGATE CLASS\tDIFFICULTY\tBUDGET\tCPUS\tMEM(GB)\tGPUS\tNET")
	for _, t := range d.Tasks {
		if !t.HasFixture {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				t.Name, t.GateClass, "(no fixture)", "—", "—", "—", "—", "—")
			continue
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%.0fh\t%d\t%.0f\t%d\t%s\n",
			t.Name, t.GateClass, t.Difficulty, t.AgentTimeoutHr,
			t.CPUs, float64(t.MemoryMB)/1024.0, t.GPUs, internetLabel(t.AllowInternet))
	}
	_ = tw.Flush()

	fmt.Fprintf(w, "\n  BUDGET = the [agent] timeout_sec, the headline time-to-solution cost per task.\n")
	fmt.Fprintf(w, "  NET    = allow_internet from the [environment] envelope.\n")
	if out != "" {
		fmt.Fprintf(w, "\nDescribe JSON written: %s\n", out)
	}
}

// internetLabel renders allow_internet as a short table cell.
func internetLabel(allow bool) string {
	if allow {
		return "yes"
	}
	return "no"
}

// sortedGateClasses renders the gate-class -> count map in a deterministic order
// (sorted by class name) as "class=N" pairs.
func sortedGateClasses(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := ""
	for i, k := range keys {
		if i > 0 {
			out += "  "
		}
		out += fmt.Sprintf("%s=%d", k, m[k])
	}
	return out
}
