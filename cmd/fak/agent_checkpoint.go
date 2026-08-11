package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentcheckpoint"
)

type checkpointStringList []string

func (s *checkpointStringList) String() string     { return strings.Join(*s, ",") }
func (s *checkpointStringList) Set(v string) error { *s = append(*s, v); return nil }

func cmdAgentCheckpoint(argv []string) {
	fs := flag.NewFlagSet("fak agent checkpoint", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var in agentcheckpoint.Input
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
	fs.StringVar(&path, "log", "", "JSONL path (default .fak/agent-status.jsonl)")
	fs.BoolVar(&asJSON, "json", false, "echo appended record as JSON")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: fak agent checkpoint [flags]")
		fmt.Fprintln(fs.Output(), "Append a structured agent milestone checkpoint.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return
		}
		agentCheckpointFatalf("agent checkpoint: %v", err)
	}
	if fs.NArg() != 0 {
		agentCheckpointFatalf("agent checkpoint: unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}
	in.State, in.Evidence, in.Blockers = agentcheckpoint.State(*state), evidence, blockers
	if path == "" {
		path = filepath.Join(".fak", "agent-status.jsonl")
	}
	record, err := agentcheckpoint.New(in, time.Now())
	if err != nil {
		agentCheckpointFatalf("agent checkpoint: %v", err)
	}
	if err := agentcheckpoint.Append(path, record); err != nil {
		agentCheckpointFatalf("agent checkpoint: append: %v", err)
	}
	if asJSON {
		if err := json.NewEncoder(os.Stdout).Encode(record); err != nil {
			agentCheckpointFatalf("agent checkpoint: output: %v", err)
		}
		return
	}
	stage := ""
	if record.Stage != nil {
		stage = fmt.Sprintf(" stage=%d/%d(%d%%)", record.Stage.Current, record.Stage.Total, record.Stage.Percent)
	}
	fmt.Printf("checkpoint appended: %s %s%s — %s\n", record.Actor, record.State, stage, record.Summary)
}

func agentCheckpointFatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
