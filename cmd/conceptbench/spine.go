// Spine (#2729, epic #2721): the minimal end-to-end slice — ONE concept
// (commit_stamp + trunk fidelity) x TWO model arms x a REAL `dos commit-audit`
// grade x one report row per model.
//
// --spine <fixture.json> grades each arm's PRODUCED COMMIT (the commit the model
// produced for the task) in a fresh scratch git repo seeded to the fixture's known
// state, then grades that commit with a real `dos commit-audit` call — the row's
// verdict/witness are the kernel referee's reading of the commit, never the
// transcript's self-report. A subject-only claim grades CLAIM_UNWITNESSED; a
// correct `(fak <leaf>)`-stamped path-scoped commit on main grades OK /
// diff-witnessed.
//
// Arms are labeled by source. A "replay" arm re-runs a recorded transcript. A
// "live" arm resolves through the #2731 model-driver registry (internal/conceptbench,
// modelarm.go) — that registry landed as a library, so a live arm now DISPATCHES
// through it instead of being hard-refused (#5311): an unbound arm refuses the typed
// arm_gated class (live calls need a key/GPU), a driven arm that hits an
// entitlement/usage wall is CLASSIFIED (recorded, never scored), and a clean live
// drive produces a headline row. The report's honesty gate (report.go, #2739 /
// #868) is COMPUTED from the resolved rows — result_claim_allowed is true only when
// a non-replay, referee-witnessed headline row exists, so a replay-only run still
// pins it false.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	conceptbench "github.com/anthony-chaudhary/fak/internal/conceptbench"
	"github.com/anthony-chaudhary/fak/internal/hooks"
)

const (
	spineReportSchema  = "fak.conceptbench.v1"
	spineFixtureSchema = "fak.conceptbench.spine.v1"
	spineConcept       = "commit_stamp"

	// spineGraderID names the real referee behind every scored spine row: the dos
	// CLI's commit-audit verb, not the self-contained replay-exact-witness grader.
	spineGraderID = "dos_commit_audit/cli"
)

// spineCommitSpec is the commit an arm's model produced for the task: the subject
// line plus the file contents it wrote. An empty file set must be declared honest
// via allow_empty (the --allow-empty "shipped" shape) — it is exactly the
// subject-only claim the referee must catch.
type spineCommitSpec struct {
	Subject    string            `json:"subject"`
	Files      map[string]string `json:"files,omitempty"`
	AllowEmpty bool              `json:"allow_empty,omitempty"`
}

type spineArm struct {
	Model  string          `json:"model"`
	Source string          `json:"source"` // "replay" | "live" (#2731 registry)
	Commit spineCommitSpec `json:"commit"`
}

type spineFixture struct {
	Schema  string            `json:"schema"`
	Note    string            `json:"note,omitempty"`
	Concept string            `json:"concept"`
	Task    string            `json:"task"`
	Leaf    string            `json:"leaf,omitempty"` // the lane a correct ship stamp names; derived from the seed paths when absent
	Seed    map[string]string `json:"seed"`
	Arms    []spineArm        `json:"arms"`
}

// stampLeaf resolves the lane leaf the #5380 affordance hint would echo into its
// `(fak <leaf>)` template: an explicit fixture "leaf" wins, else the seed paths'
// own directory convention. An empty answer means the hint cannot name the lane
// honestly, and AffordanceAsk then refuses to inject rather than teach a guess.
func (fx spineFixture) stampLeaf() string {
	if s := strings.TrimSpace(fx.Leaf); s != "" {
		return s
	}
	paths := make([]string, 0, len(fx.Seed))
	for p := range fx.Seed {
		paths = append(paths, p)
	}
	return conceptbench.LeafOfPaths(paths)
}

// spineRow is one graded model arm: {model, concept, pass, witness_source,
// evidence} plus the referee's verdict/witness verbatim, so a weak grade can
// never be read as a strong one. SignalClass marks a live arm that hit a
// usage/entitlement wall — recorded, never scored (report.go excludes it from
// every headline claim).
// AffordanceHint records whether THIS row's arm was driven with the #5380
// tier-gated affordance hint in its frame, so a re-run artifact states which rows
// were treated and which were the untouched control — the promotion evidence the
// issue asks for is a comparison, and a comparison needs the treatment labeled.
type spineRow struct {
	Model          string `json:"model"`
	Concept        string `json:"concept"`
	Pass           bool   `json:"pass"`
	Source         string `json:"source"`
	AffordanceHint bool   `json:"affordance_hint,omitempty"`
	Verdict        string `json:"verdict"`
	Witness        string `json:"witness"`
	WitnessSource  string `json:"witness_source"`
	SignalClass    string `json:"signal_class,omitempty"`
	StampKind      string `json:"stamp_kind"`
	StampLeaf      string `json:"stamp_leaf,omitempty"`
	Branch         string `json:"branch"`
	CommitSHA      string `json:"commit_sha"`
	Subject        string `json:"subject"`
	Evidence       string `json:"evidence"`
}

type spineReport struct {
	Schema             string                   `json:"schema"`
	AppVersion         string                   `json:"app_version"`
	Mode               string                   `json:"mode"`
	Fixture            string                   `json:"fixture"`
	Concept            string                   `json:"concept"`
	Task               string                   `json:"task"`
	Grader             string                   `json:"grader"`
	Budget             budgetInfo               `json:"budget"`
	ResultClaimAllowed bool                     `json:"result_claim_allowed"`
	ResultClaimReason  string                   `json:"result_claim_reason"`
	HonestyGate        conceptbench.HonestyGate `json:"honesty_gate"`
	Rows               []spineRow               `json:"rows"`
}

// dosAudit is the slice of a `dos commit-audit --json` element the spine reads.
type dosAudit struct {
	SHA     string `json:"sha"`
	Verdict string `json:"verdict"`
	Witness string `json:"witness"`
	Reason  string `json:"reason"`
}

// spineTransportBinder binds the live Transports onto the registry before a
// --spine run resolves any live arm. Production (bindLiveTransports) binds the
// gateway Transport when the fleet's gateway credentials are configured — else the
// gateway arm stays UNBOUND and a live arm refuses arm_gated, the walled-host
// outcome #5311's invalidating assumption anticipates. A test swaps this to bind a
// stub Transport so the resolve->drive->classify->grade pipeline runs with no
// key/GPU, or to bind nothing so the arm_gated refusal fires.
var spineTransportBinder = bindLiveTransports

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

	// Bind the live Transports, then PRE-RESOLVE every live arm through the #2731
	// registry so an arm_gated / model_unknown refusal fires typed (never a crash)
	// BEFORE we require the dos grader — a gated arm must refuse without depending
	// on the referee being installed.
	reg := conceptbench.NewRegistry()
	spineTransportBinder(reg)
	resolved := make([]conceptbench.ModelArm, len(arms))
	for i, a := range arms {
		switch a.Source {
		case "replay":
			// replay arms keep today's path — no registry resolution.
		case "live":
			ma, err := reg.Resolve(a.Model)
			if err != nil {
				var ae *conceptbench.ArmError
				if errors.As(err, &ae) {
					fmt.Fprintf(os.Stderr, "conceptbench --spine: arm %q: %s: %s\n", a.Model, ae.Class, ae.Msg)
				} else {
					fmt.Fprintf(os.Stderr, "conceptbench --spine: arm %q: %v\n", a.Model, err)
				}
				return 2
			}
			resolved[i] = ma
		default:
			fmt.Fprintf(os.Stderr, "conceptbench --spine: arm %q has unknown source %q (want \"replay\" or \"live\")\n", a.Model, a.Source)
			return 2
		}
	}

	dosBin, err := exec.LookPath("dos")
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench --spine: the `dos` CLI is required — the grade is a real dos commit-audit call, never a recording:", err)
		return 2
	}

	var rows []spineRow
	anyLive := false
	for i, arm := range arms {
		var (
			row spineRow
			err error
		)
		if arm.Source == "live" {
			anyLive = true
			// The #5380 ask is built PER ARM: the tier comes from the registry's own
			// rating of this model id, so --affordance-hint can turn the experiment on
			// but can never reach a frontier arm's frame. A replay arm is deliberately
			// not asked — its commit is recorded, so a hint in its frame would change
			// nothing while making the artifact claim a treatment that did not happen.
			ask := conceptbench.AffordanceAsk{
				Concept: conceptbench.Concept(fx.Concept),
				Leaf:    fx.stampLeaf(),
				Tier:    reg.TierOf(arm.Model),
				Enabled: f.affordanceHint,
			}
			row, err = driveAndGradeLiveArm(resolved[i], dosBin, fx, arm, ask)
		} else {
			row, err = runSpineArm(dosBin, fx, arm.Model, arm.Source, arm.Commit)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "conceptbench --spine: arm %q: %v\n", arm.Model, err)
			return 1
		}
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Model < rows[j].Model })

	// The honesty gate + result_claim are COMPUTED (never hard-coded) by report.go's
	// existing #868 gate over the resolved rows: result_claim_allowed is true only
	// when a non-replay, referee-witnessed headline row exists and no row is
	// unwitnessed. Mode reflects whether a live arm actually ran.
	gate, claimAllowed := spineHonesty(rows)
	mode := "replay"
	if anyLive {
		mode = "live"
	}

	rep := spineReport{
		Schema:             spineReportSchema,
		AppVersion:         appversion.Current(),
		Mode:               mode,
		Fixture:            f.spine,
		Concept:            fx.Concept,
		Task:               fx.Task,
		Grader:             spineGraderID,
		Budget:             budget,
		ResultClaimAllowed: claimAllowed,
		ResultClaimReason:  gate.Reason,
		HonestyGate:        gate,
		Rows:               rows,
	}
	return emitSpineReport(f.out, rep)
}

// emitSpineReport writes the spine's fak.conceptbench.v1 artifact through
// benchcli's lineage stamper, so the four lineage axes (version / utc / git_commit /
// machine) ride on the report the spine produces — the contract
// internal/benchlineagegate enforces at this emit site. --out takes
// benchcli.WriteReport (which additionally pins the artifact's own path into the
// lineage envelope); with no --out the stamped bytes are produced by
// benchcli.MarshalReport and printed, matching writeArtifact's stdout form.
func emitSpineReport(out string, rep spineReport) int {
	if out != "" {
		if err := benchcli.WriteReport(out, rep); err != nil {
			fmt.Fprintln(os.Stderr, "conceptbench:", err)
			return 1
		}
		fmt.Fprintln(os.Stderr, "wrote", out)
		return 0
	}
	blob, err := benchcli.MarshalReport(rep)
	if err != nil {
		fmt.Fprintln(os.Stderr, "conceptbench:", err)
		return 1
	}
	fmt.Println(string(blob))
	return 0
}

// spineHonesty lifts the spine rows into report.go's ReportRow shape and returns
// its COMPUTED honesty gate + result_claim decision — the same #868 machinery the
// leaderboard artifact uses. A replay row is labeled and excluded from every
// headline claim; a walled (SignalClass) live row is classified, never scored; a
// clean live drive graded by the dos referee is a headline row. The spine never
// hand-sets result_claim_allowed — it derives it here.
func spineHonesty(rows []spineRow) (conceptbench.HonestyGate, bool) {
	rr := make([]conceptbench.ReportRow, 0, len(rows))
	for _, r := range rows {
		rr = append(rr, conceptbench.ReportRow{
			Model:         r.Model,
			Concept:       conceptbench.Concept(r.Concept),
			Pass:          r.Pass,
			WitnessSource: r.WitnessSource,
			Replay:        r.Source == "replay",
			SignalClass:   r.SignalClass,
		})
	}
	rep := conceptbench.BuildReport("", rr)
	return rep.HonestyGate, rep.ResultClaimAllowed
}

// driveAndGradeLiveArm drives a resolved live arm through its bound Transport, then
// grades the commit it produced. A drive that hits an entitlement/usage wall
// (Scoreable()==false) is CLASSIFIED — recorded with its signal class, never scored
// as a concept failure. A clean drive re-uses the same real dos-commit-audit grade
// path as a replay arm; the model's returned text is the produced commit's subject
// (the diff still comes from the fixture recipe until #2741 wires the full forward
// pass), so the row is a genuine live, referee-witnessed headline candidate.
// ask carries the #5380 tier-gated affordance hint: ask.Frame returns the fixture's
// task text UNCHANGED for a frontier, unrated, or opted-out arm, and appends the
// hint only for an opted-in weaker-tier arm — so the frame the transport receives
// is the honest record of which arm was treated.
func driveAndGradeLiveArm(ma conceptbench.ModelArm, dosBin string, fx spineFixture, arm spineArm, ask conceptbench.AffordanceAsk) (spineRow, error) {
	_, hinted := ask.Hint()
	tr, err := ma.Drive(conceptbench.Task{ID: arm.Model, Prompt: ask.Frame(fx.Task)})
	if err != nil {
		return spineRow{}, err
	}
	if !tr.Scoreable() {
		return spineRow{
			Model:          arm.Model,
			Concept:        fx.Concept,
			Pass:           false,
			Source:         "live",
			AffordanceHint: hinted,
			Verdict:        "CLASSIFIED",
			Witness:        "walled",
			SignalClass:    tr.SignalClass,
			Subject:        arm.Commit.Subject,
			Evidence:       fmt.Sprintf("live arm walled (%s), recorded not scored: %s", tr.SignalClass, strings.TrimSpace(tr.ErrText)),
		}, nil
	}
	commit := arm.Commit
	if s := strings.TrimSpace(tr.Output); s != "" {
		commit.Subject = s
	}
	row, err := runSpineArm(dosBin, fx, arm.Model, "live", commit)
	if err != nil {
		return row, err
	}
	row.AffordanceHint = hinted
	return row, nil
}

// runSpineArm grades one arm's produced commit in a fresh scratch repo. The pass
// criterion is the concept's whole contract: the referee's verdict is OK AND
// diff-witnessed (dos commit-audit), the subject carries a parseable ship stamp
// (internal/hooks), and the commit landed on main (trunk fidelity). The
// transcript's own claim text is never consulted.
func runSpineArm(dosBin string, fx spineFixture, model, source string, commit spineCommitSpec) (spineRow, error) {
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

	armPaths, err := writeSpineFiles(dir, commit.Files)
	if err != nil {
		return spineRow{}, err
	}
	if len(armPaths) > 0 {
		if err := gitRun(dir, append([]string{"add", "--"}, armPaths...)...); err != nil {
			return spineRow{}, err
		}
	} else if !commit.AllowEmpty {
		return spineRow{}, fmt.Errorf("transcript commit writes no files and allow_empty is not set")
	}
	commitArgs := []string{"commit", "-q", "-m", commit.Subject}
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
	stampKind, stampLeaf := hooks.StampOf(commit.Subject)
	onTrunk := branch == "main"
	pass := strings.EqualFold(audit.Verdict, "OK") &&
		audit.Witness == "diff-witnessed" &&
		stampLeaf != "" &&
		onTrunk

	return spineRow{
		Model:         model,
		Concept:       fx.Concept,
		Pass:          pass,
		Source:        source,
		Verdict:       audit.Verdict,
		Witness:       audit.Witness,
		WitnessSource: "dos_commit_audit",
		StampKind:     stampKind,
		StampLeaf:     stampLeaf,
		Branch:        branch,
		CommitSHA:     sha,
		Subject:       commit.Subject,
		Evidence:      fmt.Sprintf("dos_commit_audit(%s): %s | stamp=%s/%s branch=%s", audit.SHA, audit.Reason, stampKind, stampLeaf, branch),
	}, nil
}

// bindLiveTransports binds the live gateway Transport when the fleet's gateway
// credentials are configured; otherwise the gateway arm stays unbound (a live arm
// then refuses arm_gated). The serve arm is bound where a local `fak serve --gguf`
// GGUF is present — this seam leaves it unbound, never silently live.
func bindLiveTransports(reg *conceptbench.Registry) {
	if base := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")); base != "" && strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")) != "" {
		reg.Bind(conceptbench.ArmGateway, anthropicTransport, false)
	}
}

// anthropicTransport is the live gateway arm's Transport: it drives a model through
// the fak Anthropic-compatible gateway (ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY) and
// returns the model's text output. A non-2xx response is returned as errText (never
// err) so the registry's sessionsignals classifier folds an entitlement/usage wall
// into a typed class instead of scoring it as a concept failure; err is reserved
// for a call that could not be made at all.
func anthropicTransport(model, prompt string) (output, errText string, err error) {
	base := strings.TrimRight(strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")), "/")
	key := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if base == "" || key == "" {
		return "", "", fmt.Errorf("gateway transport: ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY not set")
	}
	reqBody, err := json.Marshal(map[string]any{
		"model":      model,
		"max_tokens": 1024,
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
	})
	if err != nil {
		return "", "", err
	}
	req, err := http.NewRequest(http.MethodPost, base+"/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", "2023-06-01")
	resp, err := (&http.Client{Timeout: 120 * time.Second}).Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Sprintf("gateway HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body))), nil
	}
	var out struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return "", "", fmt.Errorf("gateway transport: parse response: %w", err)
	}
	var sb strings.Builder
	for _, c := range out.Content {
		if c.Type == "text" {
			sb.WriteString(c.Text)
		}
	}
	return sb.String(), "", nil
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
