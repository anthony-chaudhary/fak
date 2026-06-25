package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func cmdTask(argv []string) { os.Exit(runTask(os.Stdout, os.Stderr, argv)) }

func runTask(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		taskUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "sample":
		return runTaskSample(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		taskUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak task: unknown subcommand %q\n", argv[0])
		taskUsage(stderr)
		return 2
	}
}

func runTaskSample(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("task sample", flag.ContinueOnError)
	fs.SetOutput(stderr)
	taskID := fs.String("task", "process", "task id to stamp in the sample")
	title := fs.String("title", "process sample", "task title")
	stepID := fs.String("step", "snapshot", "step id to stamp in the sample")
	concept := fs.String("concept", "observe", "concept bucket for the step")
	done := fs.Float64("done", 0, "completed work units")
	total := fs.Float64("total", 0, "total work units, if known")
	unit := fs.String("unit", "", "work unit label")
	asJSON := fs.Bool("json", false, "emit the full JSON snapshot")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *done < 0 || *total < 0 {
		fmt.Fprintln(stderr, "fak task sample: --done and --total must be non-negative")
		return 2
	}

	m := taskmgr.NewManager()
	task, err := m.StartTask(taskmgr.TaskSpec{TaskID: *taskID, Title: *title, Total: *total, Unit: *unit})
	if err != nil {
		fmt.Fprintf(stderr, "fak task sample: %v\n", err)
		return 2
	}
	step, err := task.StartStep(taskmgr.StepSpec{StepID: *stepID, Title: "sample process resources", Concept: *concept, Total: *total, Unit: *unit})
	if err != nil {
		fmt.Fprintf(stderr, "fak task sample: %v\n", err)
		return 2
	}
	if *done > 0 || *total > 0 || *unit != "" {
		if err := task.SetProgress(*done, *total, *unit); err != nil {
			fmt.Fprintf(stderr, "fak task sample: %v\n", err)
			return 2
		}
		if err := step.SetProgress(*done, *total, *unit); err != nil {
			fmt.Fprintf(stderr, "fak task sample: %v\n", err)
			return 2
		}
	}

	snap := m.Snapshot()
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(snap); err != nil {
			fmt.Fprintf(stderr, "fak task sample: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderTaskSample(stdout, snap)
	return 0
}

func renderTaskSample(w io.Writer, snap taskmgr.Snapshot) {
	fmt.Fprintf(w, "process pid=%d uptime=%s cpu=%s heap=%s sys=%s goroutines=%d\n",
		snap.ProcessID,
		secondsText(snap.UptimeSeconds),
		secondsText(snap.Resource.CPUSeconds),
		bytesText(snap.Resource.HeapAllocBytes),
		bytesText(snap.Resource.SysBytes),
		snap.Resource.Goroutines,
	)
	for _, task := range snap.Tasks {
		fmt.Fprintf(w, "task %-18s %-8s liveness=%-8s runtime=%s progress=%s",
			task.TaskID, task.State, livenessText(task.LivenessClass), secondsText(task.RuntimeSeconds), progressText(task.Progress))
		if task.ETASeconds != nil {
			fmt.Fprintf(w, " eta=%s", secondsText(*task.ETASeconds))
		}
		fmt.Fprintln(w)
		for _, step := range task.Steps {
			concept := step.Concept
			if concept == "" {
				concept = "-"
			}
			fmt.Fprintf(w, "  step %-17s concept=%-10s %-8s liveness=%-8s runtime=%s cpu_delta=%s progress=%s",
				step.StepID, concept, step.State, livenessText(step.LivenessClass),
				secondsText(step.RuntimeSeconds), secondsText(step.Resource.Delta.CPUSeconds), progressText(step.Progress))
			if step.ETASeconds != nil {
				fmt.Fprintf(w, " eta=%s", secondsText(*step.ETASeconds))
			}
			fmt.Fprintln(w)
		}
	}
	if len(snap.Concepts) > 0 {
		fmt.Fprintln(w, "concepts:")
		for _, c := range snap.Concepts {
			fmt.Fprintf(w, "  %-12s steps=%d running=%d runtime=%s cpu=%s\n",
				c.Concept, c.Steps, c.RunningSteps, secondsText(c.RuntimeSeconds), secondsText(c.CPUSeconds))
		}
	}
}

func livenessText(class taskmgr.LivenessClass) string {
	if class == "" {
		return "-"
	}
	return string(class)
}

func progressText(p taskmgr.Progress) string {
	if p.Total <= 0 {
		if p.Done > 0 {
			if strings.TrimSpace(p.Unit) == "" {
				return trimFloat(p.Done)
			}
			return trimFloat(p.Done) + " " + strings.TrimSpace(p.Unit)
		}
		return "unknown"
	}
	base := trimFloat(p.Done) + "/" + trimFloat(p.Total)
	if p.Unit != "" {
		base += " " + p.Unit
	}
	if p.Percent != nil {
		base += " (" + trimFloat(*p.Percent) + "%)"
	}
	return base
}

func secondsText(v float64) string {
	return trimFloat(v) + "s"
}

func bytesText(v uint64) string {
	const unit = 1024
	if v < unit {
		return strconv.FormatUint(v, 10) + "B"
	}
	value := float64(v)
	for _, suffix := range []string{"KiB", "MiB", "GiB", "TiB"} {
		value /= unit
		if value < unit {
			return trimFloat(value) + suffix
		}
	}
	return trimFloat(value) + "PiB"
}

func trimFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', 3, 64)
}

func taskUsage(w io.Writer) {
	fmt.Fprint(w, `fak task - process-local task-manager snapshot

  fak task sample [--json] [--task ID] [--title T] [--step ID]
                  [--concept NAME] [--done N --total N --unit UNIT]

The sample command emits the same snapshot shape a long-running fak process can embed:
process resources, task/step wall time, concept runtime, progress, and ETA when known.
`)
}
