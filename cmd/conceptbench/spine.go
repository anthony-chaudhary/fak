// Spine (#2729, epic #2721): the minimal end-to-end slice — ONE concept
// (commit_stamp + trunk fidelity) x TWO model arms x a REAL `dos commit-audit`
// grade x one report row per model.
//
// --spine <fixture.json> replays each arm's recorded transcript (the commit the
// model produced for the task) into a fresh scratch git repo seeded to the
// fixture's known state, then grades the PRODUCED COMMIT with a real
// `dos commit-audit` call — the row's verdict/witness are the kernel referee's
// reading of the commit, never the transcript's self-report. A subject-only
// claim grades CLAIM_UNWITNESSED; a correct `(fak <leaf>)`-stamped path-scoped
// commit on main grades OK / diff-witnessed. Arms are labeled by source
// ("replay" today; "live" waits for the model-driver registry, #2731), and the
// report pins result_claim_allowed:false until a live frontier-vs-small
// comparison runs (#868 discipline).
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/hooks"
)

const (
	spineReportSchema  = "fak.conceptbench.v1"
	spineFixtureSchema = "fak.conceptbench.spine.v1"
	spineConcept       = "commit_stamp"

	// spineGraderID names the real referee behind every spine row: the dos CLI's
	// commit-audit verb, not the self-contained replay-exact-witness grader.
	spineGraderID = "dos_commit_audit/cli"

	// spineClaimReason is the honesty-gate reason a spine report can never claim a
	// leaderboard result: the arms replay recorded transcripts, so the missing
	// witness is a LIVE frontier-vs-small gateway comparison (#868 discipline).
	spineClaimReason = "spine arms replay recorded transcripts into a scratch fixture repo (source:\"replay\"); a live frontier-vs-small gateway comparison has not run, so no result claim is allowed (mirrors #868)"
)

// spineCommitSpec is the commit an arm's model produced for the task, as recorded
// in its transcript: the subject line plus the file contents it wrote. An empty
// file set must be declared honest via allow_empty (the --allow-empty "shipped"
// shape) — it is exactly the subject-only claim the referee must catch.
type spineCommitSpec struct {
	Subject    string            `json:"subject"`
	Files      map[string]string `json:"files,omitempty"`
	AllowEmpty bool              `json:"allow_empty,omitempty"`
}

type spineArm struct {
	Model  string          `json:"model"`
	Source string          `json:"source"` // "replay" | "live" (#2731)
	Commit spineCommitSpec `json:"commit"`
}

type spineFixture struct {
	Schema  string            `json:"schema"`
	Note    string            `json:"note,omitempty"`
	Concept string            `json:"concept"`
	Task    string            `json:"task"`
	Seed    map[string]string `json:"seed"`
	Arms    []spineArm        `json:"arms"`
}

// spineRow is one graded model arm: {model, concept, pass, witness_source,
// evidence} plus the referee's verdict/witness verbatim, so a weak grade can
// never be read as a strong one.
type spineRow struct {
	Model         string `json:"model"`
	Concept       string `json:"concept"`
	Pass          bool   `json:"pass"`
	Source        string `json:"source"`
	Verdict       string `json:"verdict"`
	Witness       string `json:"witness"`
	WitnessSource string `json:"witness_source"`
	StampKind     string `json:"stamp_kind"`
	StampLeaf     string `json:"stamp_leaf,omitempty"`
	Branch        string `json:"branch"`
	CommitSHA     string `json:"commit_sha"`
	Subject       string `json:"subject"`
	Evidence      string `json:"evidence"`
}

type spineReport struct {
	Schema             string     `json:"schema"`
	AppVersion         string     `json:"app_version"`
	Mode               string     `json:"mode"`
	Fixture            string     `json:"fixture"`
	Concept            string     `json:"concept"`
	Task               string     `json:"task"`
	Grader             string     `json:"grader"`
	Budget             budgetInfo `json:"budget"`
	ResultClaimAllowed bool       `json:"result_claim_allowed"`
	ResultClaimReason  string     `json:"result_claim_reason"`
	Rows               []spineRow `json:"rows"`
}

// dosAudit is the slice of a `dos commit-audit --json` element the spine reads.
type dosAudit struct {
	SHA     string `json:"sha"`
	Verdict string `json:"verdict"`
	Witness string `json:"witness"`
	Reason  string `json:"reason"`
}

// runSpine drives the two-arm spine end to end and emits the fak.conceptbench.v1
// report (one row per model) through the lineage-stamping artifact writer.
func runSpine(f flags, budget budgetInfo) int {
	fx, err := loadSpineFixture(f.spine)
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench --spine:", err)
		return 1
	}
	arms := fx.Arms
	if filter := csvSet(f.models); len(filter) > 0 {
		var kept []spineArm
		for _, a := range arms {
			if filter[a.Model] {
				kept = append(kept, a)
			}
		}
		arms = kept
	}
	if len(arms) != 2 {
		fmt.Fprintf(os.Stderr, "conceptbench --spine: the spine is a two-model contrast; fixture + --models selected %d arm(s)\n", len(arms))
		return 2
	}
	for _, a := range arms {
		if a.Source != "replay" {
			fmt.Fprintf(os.Stderr, "conceptbench --spine: arm %q has source %q — only \"replay\" is runnable until the model-driver registry (#2731) lands\n", a.Model, a.Source)
			return 2
		}
	}
	dosBin, err := exec.LookPath("dos")
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench --spine: the `dos` CLI is required — the grade is a real dos commit-audit call, never a recording:", err)
		return 2
	}

	var rows []spineRow
	for _, arm := range arms {
		row, err := runSpineArm(dosBin, fx, arm)
		if err != nil {
			fmt.Fprintf(os.Stderr, "conceptbench --spine: arm %q: %v\n", arm.Model, err)
			return 1
		}
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Model < rows[j].Model })

	rep := spineReport{
		Schema:             spineReportSchema,
		AppVersion:         appversion.Current(),
		Mode:               "replay",
		Fixture:            f.spine,
		Concept:            fx.Concept,
		Task:               fx.Task,
		Grader:             spineGraderID,
		Budget:             budget,
		ResultClaimAllowed: false,
		ResultClaimReason:  spineClaimReason,
		Rows:               rows,
	}
	return writeArtifact(f.out, rep)
}

// runSpineArm replays one arm into a fresh scratch repo and grades the produced
// commit. The pass criterion is the concept's whole contract: the referee's
// verdict is OK AND diff-witnessed (dos commit-audit), the subject carries a
// parseable ship stamp (internal/hooks), and the commit landed on main (trunk
// fidelity). The transcript's own claim text is never consulted.
func runSpineArm(dosBin string, fx spineFixture, arm spineArm) (spineRow, error) {
	dir, err := os.MkdirTemp("", "conceptbench-spine-")
	if err != nil {
		return spineRow{}, err
	}
	defer os.RemoveAll(dir)

	if err := gitRun(dir, "init", "-q", "-b", "main", "."); err != nil {
		return spineRow{}, err
	}
	seedPaths, err := writeSpineFiles(dir, fx.Seed)
	if err != nil {
		return spineRow{}, err
	}
	if err := gitRun(dir, append([]string{"add", "--"}, seedPaths...)...); err != nil {
		return spineRow{}, err
	}
	if err := gitRun(dir, "commit", "-q", "-m", "seed: conceptbench spine fixture baseline"); err != nil {
		return spineRow{}, err
	}

	armPaths, err := writeSpineFiles(dir, arm.Commit.Files)
	if err != nil {
		return spineRow{}, err
	}
	if len(armPaths) > 0 {
		if err := gitRun(dir, append([]string{"add", "--"}, armPaths...)...); err != nil {
			return spineRow{}, err
		}
	} else if !arm.Commit.AllowEmpty {
		return spineRow{}, fmt.Errorf("transcript commit writes no files and allow_empty is not set")
	}
	commitArgs := []string{"commit", "-q", "-m", arm.Commit.Subject}
	if len(armPaths) == 0 {
		commitArgs = append(commitArgs, "--allow-empty")
	}
	if err := gitRun(dir, commitArgs...); err != nil {
		return spineRow{}, err
	}
	sha, err := gitOut(dir, "rev-parse", "HEAD")
	if err != nil {
		return spineRow{}, err
	}
	branch, err := gitOut(dir, "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return spineRow{}, err
	}

	audit, err := dosCommitAudit(dosBin, dir, sha)
	if err != nil {
		return spineRow{}, err
	}
	stampKind, stampLeaf := hooks.StampOf(arm.Commit.Subject)
	onTrunk := branch == "main"
	pass := strings.EqualFold(audit.Verdict, "OK") &&
		audit.Witness == "diff-witnessed" &&
		stampLeaf != "" &&
		onTrunk

	return spineRow{
		Model:         arm.Model,
		Concept:       fx.Concept,
		Pass:          pass,
		Source:        arm.Source,
		Verdict:       audit.Verdict,
		Witness:       audit.Witness,
		WitnessSource: "dos_commit_audit",
		StampKind:     stampKind,
		StampLeaf:     stampLeaf,
		Branch:        branch,
		CommitSHA:     sha,
		Subject:       arm.Commit.Subject,
		Evidence:      fmt.Sprintf("dos_commit_audit(%s): %s | stamp=%s/%s branch=%s", audit.SHA, audit.Reason, stampKind, stampLeaf, branch),
	}, nil
}

// dosCommitAudit shells the real referee. Exit 1 means an unwitnessed claim was
// FOUND — a graded verdict the spine must report, never a harness error; only
// exit 2 (unreadable ref) or a spawn failure is an error.
func dosCommitAudit(dosBin, workspace, ref string) (dosAudit, error) {
	cmd := exec.Command(dosBin, "commit-audit", ref, "--workspace", workspace, "--json")
	out, err := cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) || ee.ExitCode() != 1 {
			return dosAudit{}, fmt.Errorf("dos commit-audit %s: %w", ref, err)
		}
	}
	var audits []dosAudit
	if err := json.Unmarshal(out, &audits); err != nil {
		return dosAudit{}, fmt.Errorf("parse dos commit-audit output: %w", err)
	}
	if len(audits) == 0 {
		return dosAudit{}, fmt.Errorf("dos commit-audit %s returned no audit rows", ref)
	}
	return audits[0], nil
}

func loadSpineFixture(path string) (spineFixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return spineFixture{}, fmt.Errorf("read fixture %s: %w", path, err)
	}
	var fx spineFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		return spineFixture{}, fmt.Errorf("parse fixture %s: %w", path, err)
	}
	if fx.Schema != spineFixtureSchema {
		return spineFixture{}, fmt.Errorf("fixture %s schema %q != %q", path, fx.Schema, spineFixtureSchema)
	}
	if fx.Concept != spineConcept {
		return spineFixture{}, fmt.Errorf("fixture %s concept %q — the spine grades only %q (the other concepts are #2733-#2737)", path, fx.Concept, spineConcept)
	}
	if len(fx.Seed) == 0 {
		return spineFixture{}, fmt.Errorf("fixture %s has no seed files (the scratch repo needs a known state)", path)
	}
	for _, a := range fx.Arms {
		if a.Model == "" || a.Source == "" || a.Commit.Subject == "" {
			return spineFixture{}, fmt.Errorf("fixture %s: every arm needs model, source, and commit.subject", path)
		}
	}
	return fx, nil
}

// writeSpineFiles writes the given repo-relative files under dir and returns
// their paths sorted, so git add order (and thus the fixture repo) is byte-stable.
func writeSpineFiles(dir string, files map[string]string) ([]string, error) {
	paths := make([]string, 0, len(files))
	for p := range files {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, p := range paths {
		abs := filepath.Join(dir, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(abs, []byte(files[p]), 0o644); err != nil {
			return nil, err
		}
	}
	return paths, nil
}

// gitRun executes git against the scratch repo with a pinned fixture identity so
// the produced commits are deterministic on any host.
func gitRun(dir string, args ...string) error {
	out, err := gitCmd(dir, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return nil
}

func gitOut(dir string, args ...string) (string, error) {
	out, err := gitCmd(dir, args...).Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return strings.TrimSpace(string(out)), nil
}

func gitCmd(dir string, args ...string) *exec.Cmd {
	full := append([]string{
		"-C", dir,
		"-c", "user.name=conceptbench",
		"-c", "user.email=conceptbench@fak.invalid",
		"-c", "commit.gpgsign=false",
	}, args...)
	return exec.Command("git", full...)
}
