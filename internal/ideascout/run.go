package ideascout

// The run orchestration: option resolution, the mandatory filed-issue index gate,
// the shared finish path both the live and fixture runs land in, and the helpers
// that assemble the run record.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

func Run(opts RunOptions) (RunResult, error) {
	workspace := opts.Workspace
	if workspace == "" {
		workspace = "."
	}
	topics, cfg, err := LoadConfig(opts.ConfigPath)
	if err != nil {
		return RunResult{}, fmt.Errorf("config error: %w", err)
	}
	if opts.MaxIssues != nil {
		cfg.MaxIssues = *opts.MaxIssues
	}
	if opts.MinScore != nil {
		cfg.MinScore = *opts.MinScore
	}
	if opts.UntriagedCap != nil {
		cfg.UntriagedCap = *opts.UntriagedCap
	}
	if opts.Milestone != nil {
		cfg.Milestone = *opts.Milestone
	}
	if opts.Project != nil {
		cfg.Project = *opts.Project
	}
	if opts.ProjectOwner != nil {
		cfg.ProjectOwner = *opts.ProjectOwner
	}
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	today := opts.Today
	if today == "" {
		today = now.Format("2006-01-02")
	}
	topicsByKey := map[string]Topic{}
	for _, t := range topics {
		topicsByKey[t.Key] = t
	}

	var errorsOut []string
	candidates := opts.Candidates
	issues := opts.Existing
	if !opts.UseFixtures {
		fetcher := opts.Fetcher
		if fetcher == nil {
			fetcher = LiveFetcher{}
		}
		candidates = GatherCandidates(fetcher, topics, cfg, &errorsOut)
		if len(candidates) == 0 && len(errorsOut) > 0 {
			return RunResult{}, fmt.Errorf("refuse: every source failed: %s", strings.Join(errorsOut, "; "))
		}
		seen, loadErr := LoadSeen(workspace)
		if loadErr != nil {
			errorsOut = append(errorsOut, "seen-cache: "+loadErr.Error())
			seen = map[string]SeenRecord{}
		}

		// ---- Rung 2, the durable one -------------------------------------------
		// The scout's OWN filing history, pulled by label so the query is targeted at
		// exactly the population being deduped. This is what makes "filed once, never
		// filed again" true without trusting the git-ignored local cache. It is
		// MANDATORY: a partial index is indistinguishable from "this source is new"
		// and re-files an already-triaged source, so it cannot be waived by any
		// option or covered by a populated seen-cache.
		scoutLimit := cfg.ScoutScanLimit
		if scoutLimit <= 0 {
			scoutLimit = DefaultConfig().ScoutScanLimit
		}
		scoutIssues, scoutErr := fetcher.FetchScoutIssues(scoutLimit)
		if scoutErr != nil {
			return RunResult{}, fmt.Errorf("refuse: cannot build the filed-issue index (gh issue list --label %s); filing now could re-file an already-triaged source (%w)", ScoutLabel, scoutErr)
		}
		if len(scoutIssues) >= scoutLimit {
			// Saturation is ambiguous — gh returns exactly `limit` both when that is
			// all there is and when it truncated. Refuse loudly rather than let the
			// guarantee rot silently the way the 800-issue window did.
			return RunResult{}, fmt.Errorf("refuse: the filed-issue index came back saturated at scout_scan_limit=%d, so it may be truncated and a previously-filed source could be re-filed; raise thresholds.scout_scan_limit (--config) above the number of issues the scout has ever filed", scoutLimit)
		}

		// ---- Rungs 3/4: the soft, windowed corpus of everything else -------------
		// Human-opened issues referencing the same URL, or carrying a near-identical
		// title. Nice to have, never the guarantee. The pre-existing refusal is kept
		// exactly as it was — nothing here is relaxed on the back of rung 2, because
		// degrading these rungs onto a bare local cache is still a worse run than no
		// run.
		issues, err = fetcher.FetchExistingIssues(cfg.IssueScanLimit)
		if err != nil {
			errorsOut = append(errorsOut, "issues: "+err.Error())
			if len(seen) == 0 {
				return RunResult{}, fmt.Errorf("refuse: cannot fetch existing issues and no seen-cache to fall back on (%w)", err)
			}
			issues = nil
		}
		// Per-lane attribution of whatever the gather recorded: a lane that failed on
		// EVERY topic that armed it is a source that is down, and the run says so in
		// its status instead of exiting 0 with six error strings nobody reads (#6506).
		health := SourceHealth(topics, cfg, errorsOut)
		return finishRun(finishInput{Workspace: workspace, Topics: topicsByKey, Config: cfg, Today: today, Now: now, Candidates: candidates, Issues: issues, ScoutIssues: scoutIssues, ScoutScanLimit: scoutLimit, Errors: errorsOut, Health: health, Live: opts.Live, Fetcher: fetcher, Seen: seen})
	}
	seen, err := LoadSeen(workspace)
	if err != nil {
		errorsOut = append(errorsOut, "seen-cache: "+err.Error())
		seen = map[string]SeenRecord{}
	}
	// Fixture replay: no network, so nothing can have been truncated. When the
	// caller did not separate the two corpora, the supplied issues stand in for
	// both — the same stamps still gate the durable rung.
	scoutIssues := opts.ScoutIssues
	if scoutIssues == nil {
		scoutIssues = issues
	}
	return finishRun(finishInput{Workspace: workspace, Topics: topicsByKey, Config: cfg, Today: today, Now: now, Candidates: candidates, Issues: issues, ScoutIssues: scoutIssues, ScoutScanLimit: cfg.ScoutScanLimit, Errors: errorsOut, Live: opts.Live, Fetcher: opts.Fetcher, Seen: seen})
}

type finishInput struct {
	Workspace      string
	Topics         map[string]Topic
	Config         Config
	Today          string
	Now            time.Time
	Candidates     []Candidate
	Issues         []ExistingIssue
	ScoutIssues    []ExistingIssue
	ScoutScanLimit int
	Errors         []string
	Health         []LaneHealth
	Live           bool
	Fetcher        Fetcher
	Seen           map[string]SeenRecord
}

func finishRun(in finishInput) (RunResult, error) {
	stamped := StampIndex(in.ScoutIssues)
	winStamped, titleSets, bodiesJoined := ExistingIssueIndex(in.Issues)
	// Union, not replacement: a filed issue whose label a human stripped is still
	// ours, and its stamp is still proof we filed that source.
	for sid := range winStamped {
		stamped[sid] = struct{}{}
	}
	toFile, skipStats, dropped := PlanIssues(in.Candidates, in.Topics, in.Seen, stamped, titleSets, bodiesJoined, in.Config, in.Today, in.Now)
	// The conversion gate reads the SAME corpus the durable dedup rung just used, so
	// it costs no extra fetch: what the scout has filed, how much of it is still
	// untriaged, and how much of it ever converted. `Planned` is still reported when
	// the gate holds — a paused run is a dry-run whose plan an operator can read
	// while draining, not a run with nothing to say.
	backlog := Backlog(in.ScoutIssues, in.Now)
	gate := GateFiling(backlog, in.Config.UntriagedCap)
	var filed []FiledIssue
	if in.Live && len(toFile) > 0 && !gate.Paused {
		if in.Fetcher == nil {
			in.Fetcher = LiveFetcher{}
		}
		if err := in.Fetcher.EnsureLabels(); err != nil {
			in.Errors = append(in.Errors, "labels: "+err.Error())
		}
		for _, issue := range toFile {
			u, err := in.Fetcher.CreateIssue(issue, in.Config.Milestone)
			if err != nil {
				in.Errors = append(in.Errors, "create["+issue.SourceID+"]: "+err.Error())
				continue
			}
			if in.Config.Project != "" {
				if err := in.Fetcher.AddToProject(u, in.Config.Project, in.Config.ProjectOwner); err != nil {
					in.Errors = append(in.Errors, "project["+issue.SourceID+"]: "+err.Error())
				}
			}
			in.Seen[issue.SourceID] = SeenRecord{FiledAt: in.Today, IssueURL: u, Score: issue.Score, Topic: issue.Topic}
			filed = append(filed, FiledIssue{Title: issue.Title, IssueURL: u})
		}
		if len(filed) > 0 {
			if err := SaveSeen(in.Workspace, in.Seen); err != nil {
				return RunResult{}, err
			}
		}
	}
	return RunResult{
		Schema:             Schema,
		Date:               in.Today,
		Mode:               mode(in.Live),
		Status:             RunStatus(gate, in.Health),
		Backlog:            backlog,
		FilingGate:         gate,
		SourceHealth:       in.Health,
		CandidatesGathered: len(in.Candidates),
		DedupIndex: DedupIndex{
			FiledIssuesScanned:  len(in.ScoutIssues),
			FiledStamps:         len(stamped),
			ScoutScanLimit:      in.ScoutScanLimit,
			ScoutIndexComplete:  true,
			WindowIssuesScanned: len(in.Issues),
			IssueScanLimit:      in.Config.IssueScanLimit,
		},
		Skipped:      skipStats,
		Dropped:      dropped,
		Planned:      publicPlans(toFile),
		Filed:        filed,
		Errors:       in.Errors,
		SourceDigest: candidateDigest(in.Candidates),
		Topics:       len(in.Topics),
		Thresholds: map[string]any{
			"max_issues":  in.Config.MaxIssues,
			"min_score":   in.Config.MinScore,
			"dup_jaccard": in.Config.DupJaccard,
		},
	}, nil
}

func mode(live bool) string {
	if live {
		return "live"
	}
	return "dry-run"
}

func publicPlans(in []IssuePlan) []IssuePlan {
	out := make([]IssuePlan, len(in))
	for i, p := range in {
		p.Body = ""
		out[i] = p
	}
	return out
}

func candidateDigest(cands []Candidate) string {
	if len(cands) == 0 {
		return ""
	}
	raw, _ := json.Marshal(cands)
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
