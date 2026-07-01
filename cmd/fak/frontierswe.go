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
	case "cache-witness":
		return runFrontiersweCacheWitness(stdout, stderr, rest)
	case "env-adapter", "environment":
		return runFrontiersweEnvAdapter(stdout, stderr, rest)
	case "smoke-contract":
		return runFrontiersweSmokeContract(stdout, stderr, rest)
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
                           [--tts [--reuse R] [--workers N,...] [--task NAME]]
        Load the FrontierSWE task catalog (the 17 task names + their scoring
        family) and print, per task: name, category, difficulty, the 20-hour
        [agent] timeout_sec (the HEADLINE cost), the [environment] envelope
        (cpus / memory_mb / gpus / allow_internet), and the scoring gate class
        (implementation / performance / ml_research). Fully offline: with no
        flags it reads the committed task fixtures (`+frontiersweSampleTasks+`) and
        announces it on stderr — no model, GPU, Modal account, or Docker image.

        --tts adds the deterministic time-to-solution PROJECTION block: per task,
        the projected TTS ratio T_fak/T_raw at cross-turn reuse rate --reuse plus
        the A/C work-elimination floor, and a --workers cross-trial sweep (the
        n_concurrent_trials prefix-sharing axis). --task NAME restricts it to one
        task. It is a deterministic floor (no model), NOT a measurement — the
        measured TTS is deferred to C14.

  fak frontierswe cache-witness [--metrics-dir DIR | --metrics-files A,B,...
                                | --gateway URL --interval SEC --samples N]
                                [--out TRACE.jsonl]
        Fold a FrontierSWE trial's periodic fak serve /metrics scrapes into the
        per-turn reused-prefill-token SERIES and the MEASURED cross-turn reuse rate
        r that C14 plugs into the TTS projection. fak's own KV-prefix reuse is
        WITNESSED; the provider cache_read is OBSERVED — kept separate, never summed.
        --metrics-dir / --metrics-files fold captured bodies offline (RUNNABLE NOW,
        no gateway); --gateway live-scrapes a co-resident gateway (needs C7).

  fak frontierswe env-adapter [--tasks DIR] [--task NAME] [--json] [--out FILE]
                              [--gateway-base-url URL] [--gateway-addr HOST:PORT]
                              [--upstream-base-url URL] [--pinned-hosts A,B]
        Emit the C7 co-resident environment adapter recipe: start fak serve inside
        the FrontierSWE task sandbox, wait on /healthz before turn 1, smoke one
        chat-completions request through the same /v1 base URL the C6 shim uses,
        then exec the FrontierSWE harness. If this host lacks Docker/GHCR/Modal,
        the result is honestly GATED and prints the exact remote command.

  fak frontierswe smoke-contract [--tasks DIR] [--task NAME] [--model MODEL]
                                  [--agent NAME] [--out contract.json]
                                  [--md contract.md]
        Emit the C10 raw-vs-fak pre-run contract. It fixes the same task, model,
        agent, budget, raw arm, fak-routed arm, official grader gate,
        score-parity gate, and TTS metric. It is offline and always
        result_claim_allowed=false until both arms run and the official scorer
        confirms score parity.
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

	// TTS is the optional time-to-solution PROJECTION block, present only when
	// --tts is given. One entry per (selected) task: the C4 TTSModel projection at
	// the chosen reuse rate plus the cross-trial sweep. Deterministic floor, not a
	// measurement (the measured TTS is deferred to C14).
	TTS []frontierswe.TTSProjection `json:"tts_projection,omitempty"`
}

func runFrontiersweDescribe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe describe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tasks := fs.String("tasks", "", "task tree to overlay budgets/resources from (dir of <name>/task.toml); default: the committed "+frontiersweSampleTasks+" sample")
	asJSON := fs.Bool("json", false, "emit only the describe JSON on stdout (no human table on stderr)")
	out := fs.String("out", "", "write the describe JSON here (default: stdout JSON + human table on stderr)")
	tts := fs.Bool("tts", false, "add the deterministic time-to-solution PROJECTION block: per-task projected TTS ratio + A/C work-elimination floor (offline; no model)")
	reuse := fs.Float64("reuse", frontierswe.DefaultReuseRate, "cross-turn reuse rate r in [0,1] for the --tts projection (the value-stack reuse dial; a PROJECTION, not a measurement)")
	workersArg := fs.String("workers", "", "comma-separated cross-trial counts to sweep for --tts (the n_concurrent_trials prefix-sharing axis); default: 1 + the task's declared trials")
	task := fs.String("task", "", "restrict the --tts projection to a single task by name (default: all catalog tasks)")
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

	// The --tts projection block overlays the C4 TTSModel: per task, the projected
	// time-to-solution ratio at reuse rate --reuse plus the A/C work-elimination
	// floor, and a cross-trial sweep. It is deterministic and offline (no model);
	// it is labeled a PROJECTION (floor) throughout, with the measured TTS deferred
	// to C14. The block rides on the same JSON payload so the stdout/stderr split is
	// unchanged.
	if *tts {
		trialSweep := parseIntList(*workersArg)
		d.TTS = buildFrontierTTS(d, tasksDir, *reuse, trialSweep, *task)
		if len(d.TTS) == 0 {
			fmt.Fprintf(stderr, "fak frontierswe describe --tts: no task matched --task %q\n", *task)
			return 1
		}
	}

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
		if *tts {
			printFrontierTTS(stderr, d, *reuse)
		}
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

// buildFrontierTTS builds the per-task time-to-solution projection block. For each
// catalog task (or just --task NAME when given), it loads the committed task.toml
// when present to derive the agent budget, projects the long-horizon geometry, and
// runs the C4 TTSModel at reuse rate r over the trial sweep. A task with no
// committed fixture still gets a projection from the catalog-default budget, marked
// HasFixture=false so the floor is honest about its input. Deterministic and
// offline — no model, no GPU, no network.
func buildFrontierTTS(d FrontierDescribe, tasksDir string, reuse float64, trialSweep []int, only string) []frontierswe.TTSProjection {
	out := make([]frontierswe.TTSProjection, 0, len(d.Tasks))
	for _, row := range d.Tasks {
		if only != "" && row.Name != only {
			continue
		}
		// Load the task so the projection sees the real budget + n_concurrent_trials
		// where a fixture exists. With no fixture, synthesize a Task carrying just
		// the canonical 20h headline budget so the catalog task still gets a floor.
		task, err := frontierswe.LoadTask(filepath.Join(tasksDir, row.Name))
		if err != nil {
			task = &frontierswe.Task{Name: row.Name}
			task.Agent.TimeoutSec = d.HeadlineHours * 3600.0
		}
		p := frontierswe.ProjectTTS(task, reuse, trialSweep)
		p.Name = row.Name
		p.HasFixture = row.HasFixture
		out = append(out, p)
	}
	return out
}

// printFrontierTTS writes the human-readable time-to-solution projection table on
// stderr (keeping stdout clean JSON). It is clearly labeled a deterministic
// PROJECTION (floor): the per-task projected TTS ratio + A/C work-elimination at
// the chosen reuse rate, then the cross-trial sweep for tasks that fan out.
func printFrontierTTS(w io.Writer, d FrontierDescribe, reuse float64) {
	fmt.Fprintf(w, "\n== time-to-solution PROJECTION (deterministic floor, no model) ==\n")
	fmt.Fprintf(w, "reuse rate    : r=%.2f (the cross-turn value-stack reuse dial; a PROJECTION, not a measurement)\n", reuse)
	fmt.Fprintf(w, "geometry      : turns projected from each task's [agent] timeout_sec (measured TTS deferred to C14)\n\n")

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "TASK\tTURNS\tA/B TURN-TAX\tA/C @r\tTTS RATIO @r")
	for _, p := range d.TTS {
		fmt.Fprintf(tw, "%s\t%d\t%.1fx\t%.1fx\t%.4f\n",
			p.Name, p.Geometry.Turns, p.Arms.AOverB, p.Arms.AOverC, p.Arms.TTSRatio)
	}
	_ = tw.Flush()

	fmt.Fprintf(w, "\n  A/B TURN-TAX  = re-prefill-every-turn vs KV persistence (structural, reuse-independent).\n")
	fmt.Fprintf(w, "  A/C @r        = net re-prefill work-elimination at reuse rate r (the value-stack floor).\n")
	fmt.Fprintf(w, "  TTS RATIO @r  = projected T_fak/T_raw = C(r)/A — fraction of the budget fak's run is projected to take.\n")

	// The cross-trial sweep: only meaningful where a task fans out to >1 concurrent
	// trial (n_concurrent_trials), so concurrent trials of the same task share the
	// identical prefix. Print it only when at least one task sweeps past 1 trial.
	if frontierTTSHasCrossTrial(d.TTS) {
		fmt.Fprintf(w, "\ncross-trial reuse (n_concurrent_trials share the identical prefix):\n")
		ttw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
		fmt.Fprintln(ttw, "TASK\tTRIALS\tA(all trials)\tD(shared prefix)\tA/D\tTTS RATIO")
		for _, p := range d.TTS {
			for _, ta := range p.TrialSweep {
				if ta.Trials <= 1 {
					continue
				}
				fmt.Fprintf(ttw, "%s\t%d\t%.0f\t%.0f\t%.1fx\t%.4f\n",
					p.Name, ta.Trials, ta.ATrials, ta.DShared, ta.ATrialsD, ta.TTSTrial)
			}
		}
		_ = ttw.Flush()
		fmt.Fprintf(w, "\n  A/D = cross-trial work-elimination: N trials re-prefilling vs sharing one prefix (bites at trials>1).\n")
	}
}

// frontierTTSHasCrossTrial reports whether any projection's trial sweep fans out
// past a single trial, so the cross-trial table is only printed when it carries a
// real cross-trial arm.
func frontierTTSHasCrossTrial(ps []frontierswe.TTSProjection) bool {
	for _, p := range ps {
		for _, ta := range p.TrialSweep {
			if ta.Trials > 1 {
				return true
			}
		}
	}
	return false
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
