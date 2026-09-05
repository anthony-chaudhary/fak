package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
	"github.com/anthony-chaudhary/fak/internal/stopgate"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

const (
	guardWitnessedDoneModeEnv = "FAK_GUARD_WITNESSED_DONE_MODE"
	guardWitnessedDoneMaxEnv  = "FAK_GUARD_WITNESSED_DONE_MAX"
	guardWitnessedDoneDefault = guardPreCompactModeShadow
	guardWitnessedDoneMax     = 3

	stopDispClaimUnwitnessedContinue guardStopDisposition = "claim_unwitnessed_continue"
	stopDispClaimUnwitnessedGiveUp   guardStopDisposition = "claim_unwitnessed_give_up"
	stopDispClaimWitnessed           guardStopDisposition = "claim_witnessed"
	stopDispClaimWitnessShadow       guardStopDisposition = "claim_witness_shadow"

	guardClaimUnwitnessedReason = "CLAIM_UNWITNESSED"
)

type guardWitnessedDoneFinding struct {
	Claimed   bool
	Witnessed bool
	Commit    string
	Reason    string
	Detail    string
}

type guardWitnessedDoneRunner interface {
	Run(context.Context, string, ...string) ([]byte, error)
}

type guardWitnessedDoneGit struct{ dir string }

func (g guardWitnessedDoneGit) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = g.dir
	configureDispatchHelperCommand(cmd)
	return cmd.CombinedOutput()
}

var (
	guardDoneClaimRE = regexp.MustCompile(`(?i)(?:\b(?:done|completed|finished|implemented|fixed|shipped)\b|\btests? (?:pass|passed)\b)`)
	guardCommitRefRE = regexp.MustCompile(`(?i)\b(?:commit(?:ted)?(?:\s+as|\s+is|:)?\s+)([0-9a-f]{7,40})\b`)
)

func normalizeGuardWitnessedDoneMode(raw string) (string, error) {
	if strings.TrimSpace(raw) == "" {
		return guardWitnessedDoneDefault, nil
	}
	return normalizeGuardPreCompactMode(raw)
}

func guardWitnessedDoneMaxFromEnv() int {
	v, err := strconv.Atoi(strings.TrimSpace(os.Getenv(guardWitnessedDoneMaxEnv)))
	if err != nil || v < 1 {
		return guardWitnessedDoneMax
	}
	return v
}

// inspectGuardWitnessedDone binds a narrated completion to a commit named in the
// final assistant turn, then corroborates that immutable object through the in-process
// witness resolver and the same stamp/path lint used at commit time.
func inspectGuardWitnessedDone(ctx context.Context, transcriptPath, root string, runner guardWitnessedDoneRunner) guardWitnessedDoneFinding {
	final := guardLastAssistantText(transcript.LoadFile(transcriptPath))
	if !guardDoneClaimRE.MatchString(final) {
		return guardWitnessedDoneFinding{}
	}
	finding := guardWitnessedDoneFinding{Claimed: true, Reason: guardClaimUnwitnessedReason}
	match := guardCommitRefRE.FindStringSubmatch(final)
	if len(match) < 2 {
		finding.Detail = "completion was narrated without a commit reference"
		return finding
	}
	finding.Commit = match[1]
	if runner == nil {
		runner = guardWitnessedDoneGit{dir: root}
	}
	// The resolver is the independent object-existence witness required by #3302.
	gitWitnessRun := witness.Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = root
		configureDispatchHelperCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err == nil {
			return string(out), 0, nil
		}
		if exit, ok := err.(*exec.ExitError); ok {
			return string(out), exit.ExitCode(), nil
		}
		return string(out), -1, err
	})
	if outcome := witness.NewWithRunner(gitWitnessRun, root).Resolve(ctx, &abi.ToolCall{Tool: "guard-stophook"}, "commit:"+finding.Commit); outcome != abi.WitnessConfirmed {
		finding.Detail = fmt.Sprintf("the named commit %s is not independently present in git rooted at %s", finding.Commit, root)
		return finding
	}
	subjectOut, err := runner.Run(ctx, "git", "show", "-s", "--format=%s", finding.Commit)
	if err != nil {
		finding.Detail = "could not read the named commit subject"
		return finding
	}
	pathsOut, err := runner.Run(ctx, "git", "diff-tree", "--no-commit-id", "--name-only", "-r", finding.Commit)
	if err != nil {
		finding.Detail = "could not read the named commit paths"
		return finding
	}
	paths := strings.Fields(strings.ReplaceAll(string(pathsOut), "\\", "/"))
	report := hooks.LintCommitMessage(strings.TrimSpace(string(subjectOut)), paths, root)
	if !report.OK || !report.Gradeable || !report.LeafRecognized || !report.LeafMatches || len(report.Issues) > 0 {
		finding.Detail = "the named commit lacks a bindable, lane-matching witness-gradeable stamp"
		return finding
	}
	finding.Witnessed = true
	finding.Reason = "CLAIM_WITNESSED"
	finding.Detail = "completion is bound to a witnessed stamped commit"
	return finding
}

func guardLastAssistantText(records []transcript.Record) string {
	for i := len(records) - 1; i >= 0; i-- {
		if records[i].Role() == "assistant" {
			return records[i].Text()
		}
	}
	return ""
}

func runGuardWitnessedDoneGate(stderr io.Writer, rawMode, transcriptPath, root, session string, max int) (int, guardStopDisposition, string, bool) {
	mode, err := normalizeGuardWitnessedDoneMode(rawMode)
	if err != nil || mode == guardPreCompactModeOff {
		return 0, "", "", false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	finding := inspectGuardWitnessedDone(ctx, transcriptPath, root, nil)
	if !finding.Claimed {
		return 0, "", "", false
	}
	seq := guardWitnessedDoneBlockCount(session)
	claim := stopgate.WitnessClaim{
		Claimed:   finding.Claimed,
		Witnessed: finding.Witnessed,
		Commit:    finding.Commit,
		Reason:    finding.Reason,
		Detail:    finding.Detail,
	}
	cfg := stopgate.WitnessGateConfig{
		Mode: stopgate.Mode(mode),
		Max:  max,
	}
	dec := stopgate.EvaluateWitness(cfg, claim, seq)
	if dec.Note != "" {
		fmt.Fprintln(stderr, dec.Note)
	} else if dec.OperatorMsg != "" {
		fmt.Fprintln(stderr, dec.OperatorMsg)
	} else if dec.Guidance != "" {
		fmt.Fprintln(stderr, dec.Guidance)
	}
	return dec.ExitCode, guardStopDisposition(dec.Disposition), dec.Reason, true
}

func guardWitnessedDoneKind(d guardStopDisposition) guardStopKind {
	return guardStopKind(stopgate.Disposition(d).Kind())
}
func guardWitnessedDoneBlockCount(session string) int {
	ledger := guardStopsLedgerConfigured()
	if ledger == "" || strings.TrimSpace(session) == "" {
		return 0
	}
	b, err := os.ReadFile(ledger)
	if err != nil {
		return 0
	}
	rows := jsonlledger.Parse(string(b), func(r guardStopRecord) bool { return r.Schema == guardStopRecordSchema })
	count := 0
	for _, row := range rows {
		if row.Session == session && guardStopDisposition(row.Disposition) == stopDispClaimUnwitnessedContinue {
			count++
		}
	}
	return count
}
