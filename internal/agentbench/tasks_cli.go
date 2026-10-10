package agentbench

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

const tasksUsage = "usage: fak bench agent tasks --endpoint URL --model ID --out DIR [--suite default|extended] [--include-heldout] [--reps N] [--concurrency N] [--task-limit N] [--temperature F] [--top-p F] [--top-k N] [--max-tokens N] [--seed N]"

type tasksOptions struct {
	endpoint, model, out, suite string
	includeHeldout              bool
	reps, concurrency, limit    int
	sampling                    taskrun.Sampling
}

// TasksSummary folds every rep of a task-only run. Receipts holds one
// taskrun receipt path per rep; Tasks lists per-rep acceptance by fixture.
type TasksSummary struct {
	Schema            string                 `json:"schema"`
	Endpoint          string                 `json:"endpoint"`
	Model             string                 `json:"model"`
	Suite             string                 `json:"suite"`
	IncludeHeldout    bool                   `json:"include_heldout"`
	Reps              int                    `json:"reps"`
	CompletedReps     int                    `json:"completed_reps"`
	Concurrency       int                    `json:"concurrency"`
	SamplingRequested *taskrun.Sampling      `json:"sampling_requested,omitempty"`
	SamplingObserved  []taskrun.Sampling     `json:"sampling_observed,omitempty"`
	Receipts          []string               `json:"receipts"`
	Tasks             []TaskRepRow           `json:"tasks"`
	PerRep            []taskrun.TrialSummary `json:"per_rep"`
	Aggregate         taskrun.TrialSummary   `json:"aggregate"`
	StartedAt         time.Time              `json:"started_at"`
	Duration          time.Duration          `json:"duration"`
	Error             string                 `json:"error,omitempty"`
}

type TaskRepRow struct {
	ID            string   `json:"id"`
	Family        string   `json:"family"`
	Accepted      []bool   `json:"accepted"`
	AcceptedCount int      `json:"accepted_count"`
	Loops         []bool   `json:"loop"`
	Errors        []string `json:"errors,omitempty"`
}

func runTasksCLI(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	opts, err := parseTasksOptions(stderr, args)
	if err != nil {
		if !errors.Is(err, flag.ErrHelp) {
			fmt.Fprintf(stderr, "agentbench: tasks: %v\n%s\n", err, tasksUsage)
		}
		return 2
	}
	executable, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: tasks: executable: %v\n", err)
		return 1
	}
	if err := os.MkdirAll(opts.out, 0700); err != nil {
		fmt.Fprintf(stderr, "agentbench: tasks: output: %v\n", err)
		return 1
	}
	summary := runTaskReps(ctx, opts, func(rep int, out string) (taskrun.Receipt, error) {
		return taskrun.Run(ctx, taskrun.Options{
			Executable: executable, Endpoint: strings.TrimSuffix(opts.endpoint, "/chat/completions"), Model: opts.model, OutDir: out,
			Concurrency: opts.concurrency, IncludeHeldout: opts.includeHeldout, TaskLimit: opts.limit,
			Suite: opts.suite, Sampling: opts.sampling,
		})
	})
	if err := writeSummaryJSON(filepath.Join(opts.out, "summary.json"), summary); err != nil {
		fmt.Fprintf(stderr, "agentbench: tasks: write summary: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(summary); err != nil {
		return 1
	}
	if summary.Error != "" {
		fmt.Fprintf(stderr, "agentbench: tasks: %s\n", summary.Error)
		return 1
	}
	return 0
}

type repRunner func(rep int, outDir string) (taskrun.Receipt, error)

func runTaskReps(ctx context.Context, opts tasksOptions, run repRunner) TasksSummary {
	started := time.Now()
	s := TasksSummary{Schema: "fak.agentbench.tasks-summary.v1", Endpoint: opts.endpoint, Model: opts.model, Suite: opts.suite, IncludeHeldout: opts.includeHeldout, Reps: opts.reps, Concurrency: opts.concurrency, StartedAt: started}
	if opts.sampling != (taskrun.Sampling{}) {
		sampling := opts.sampling
		s.SamplingRequested = &sampling
	}
	var pooled []taskrun.TaskReceipt
	rows := map[string]int{}
	seenSampling := map[string]bool{}
	for rep := 1; rep <= opts.reps; rep++ {
		if err := ctx.Err(); err != nil {
			s.Error = fmt.Sprintf("rep %d: %v", rep, err)
			break
		}
		out := filepath.Join(opts.out, fmt.Sprintf("rep-%02d", rep))
		receipt, err := run(rep, out)
		if err != nil {
			s.Error = fmt.Sprintf("rep %d: %v", rep, err)
			break
		}
		s.CompletedReps++
		s.Receipts = append(s.Receipts, filepath.Join(out, "receipt.json"))
		s.PerRep = append(s.PerRep, receipt.Aggregate)
		pooled = append(pooled, receipt.Tasks...)
		for _, task := range receipt.Tasks {
			index, ok := rows[task.ID]
			if !ok {
				index = len(s.Tasks)
				rows[task.ID] = index
				s.Tasks = append(s.Tasks, TaskRepRow{ID: task.ID, Family: task.Family})
			}
			row := &s.Tasks[index]
			row.Accepted = append(row.Accepted, task.Accepted)
			row.Loops = append(row.Loops, task.ToolCalls.Loop)
			if task.Accepted {
				row.AcceptedCount++
			}
			if task.Error != "" {
				row.Errors = append(row.Errors, fmt.Sprintf("rep %d: %s", rep, task.Error))
			}
			for _, sent := range task.Model.SamplingSent {
				key, _ := json.Marshal(sent)
				if !seenSampling[string(key)] {
					seenSampling[string(key)] = true
					s.SamplingObserved = append(s.SamplingObserved, sent)
				}
			}
		}
	}
	s.Aggregate = taskrun.SummarizeTrials(pooled)
	s.Duration = time.Since(started)
	return s
}

func parseTasksOptions(stderr io.Writer, args []string) (tasksOptions, error) {
	opts := tasksOptions{suite: taskfixture.SuiteDefault, reps: 1, concurrency: 1}
	flags := flag.NewFlagSet("bench agent tasks", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&opts.endpoint, "endpoint", "", "OpenAI-compatible base URL (for example http://host:8080/v1)")
	flags.StringVar(&opts.model, "model", "", "model identifier the endpoint must report")
	flags.StringVar(&opts.out, "out", "", "artifact directory; rep-NN/receipt.json per rep plus summary.json")
	flags.StringVar(&opts.suite, "suite", opts.suite, "fixture suite: default or extended")
	flags.BoolVar(&opts.includeHeldout, "include-heldout", false, "include heldout fixtures")
	flags.IntVar(&opts.reps, "reps", opts.reps, "repetitions of the whole suite")
	flags.IntVar(&opts.concurrency, "concurrency", opts.concurrency, "concurrent task children")
	flags.IntVar(&opts.limit, "task-limit", 0, "run only the first N fixtures (0 = all)")
	flags.Func("temperature", "sampling temperature (unset: planner default 0)", floatFlag(&opts.sampling.Temperature))
	flags.Func("top-p", "nucleus sampling top_p (unset: endpoint default)", floatFlag(&opts.sampling.TopP))
	flags.Func("top-k", "top_k (unset: endpoint default)", intFlag(&opts.sampling.TopK))
	flags.Func("max-tokens", "max_tokens per turn (unset: planner default 1024)", intFlag(&opts.sampling.MaxTokens))
	flags.Func("seed", "sampling seed (unset: endpoint default)", func(v string) error {
		n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return err
		}
		opts.sampling.Seed = &n
		return nil
	})
	if err := flags.Parse(args); err != nil {
		return opts, err
	}
	switch {
	case flags.NArg() != 0:
		return opts, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	case opts.endpoint == "" || opts.model == "" || opts.out == "":
		return opts, errors.New("--endpoint, --model and --out are required")
	case opts.reps <= 0 || opts.concurrency <= 0 || opts.limit < 0:
		return opts, errors.New("--reps and --concurrency must be positive and --task-limit non-negative")
	}
	if _, err := taskfixture.Suite(opts.suite, opts.includeHeldout); err != nil {
		return opts, err
	}
	if err := opts.sampling.Validate(); err != nil {
		return opts, err
	}
	return opts, nil
}

func floatFlag(dst **float64) func(string) error {
	return func(v string) error {
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			return err
		}
		*dst = &f
		return nil
	}
}

func intFlag(dst **int) func(string) error {
	return func(v string) error {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return err
		}
		*dst = &n
		return nil
	}
}

func writeSummaryJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}
