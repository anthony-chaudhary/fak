package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devcheckpoint"
)

type checkpointStringList []string

func (s *checkpointStringList) String() string     { return strings.Join(*s, ",") }
func (s *checkpointStringList) Set(v string) error { *s = append(*s, v); return nil }

// RunCheckpoint appends one repository-development milestone checkpoint.
func RunCheckpoint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak-dev checkpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var in devcheckpoint.Input
	var evidence, blockers checkpointStringList
	var path string
	var asJSON bool
	fs.StringVar(&in.Actor, "actor", "", "stable worker/session label (required)")
	fs.StringVar(&in.Scope, "scope", "", "issue, plan phase, job, or task (required)")
	state := fs.String("state", "", "started|progress|blocked|handoff|done (required)")
	fs.IntVar(&in.StageCurrent, "stage-current", 0, "current ordinal stage; required for progress")
	fs.IntVar(&in.StageTotal, "stage-total", 0, "total stage count; required for progress")
	fs.StringVar(&in.StageName, "stage-name", "", "human-readable current stage")
	fs.StringVar(&in.Summary, "summary", "", "one sentence describing the milestone (required)")
	fs.Var(&evidence, "evidence", "evidence path/test/commit/URL; repeat or comma-separate")
	fs.StringVar(&in.Next, "next", "", "one next action; required unless done")
	fs.Var(&blockers, "blocker", "blocker; repeat or comma-separate")
	fs.StringVar(&in.GitHub, "github", "", "canonical existing issue/PR URL")
	fs.StringVar(&path, "log", "", "JSONL path (default .fak/dev-status.jsonl)")
	fs.BoolVar(&asJSON, "json", false, "echo appended record as JSON")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: fak dev checkpoint [flags]")
		fmt.Fprintln(fs.Output(), "Append a structured repository-development milestone checkpoint.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		fmt.Fprintf(stderr, "dev checkpoint: %v\n", err)
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "dev checkpoint: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	in.State, in.Evidence, in.Blockers = devcheckpoint.State(*state), evidence, blockers
	if path == "" {
		path = filepath.Join(".fak", "dev-status.jsonl")
	}
	record, err := devcheckpoint.New(in, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "dev checkpoint: %v\n", err)
		return 2
	}
	if err := devcheckpoint.Append(path, record); err != nil {
		fmt.Fprintf(stderr, "dev checkpoint: append: %v\n", err)
		return 2
	}
	if asJSON {
		if err := json.NewEncoder(stdout).Encode(record); err != nil {
			fmt.Fprintf(stderr, "dev checkpoint: output: %v\n", err)
			return 2
		}
		return 0
	}
	stage := ""
	if record.Stage != nil {
		stage = fmt.Sprintf(" stage=%d/%d(%d%%)", record.Stage.Current, record.Stage.Total, record.Stage.Percent)
	}
	fmt.Fprintf(stdout, "checkpoint appended: %s %s%s — %s\n", record.Actor, record.State, stage, record.Summary)
	return 0
}
