package dispatchstatus

import (
	"fmt"
	"strings"
)

// Schema is the status-card payload schema id, shared verbatim with the Python
// parity target tools/dispatch_status.py.
const Schema = "fleet-dispatch-status/1"

// CardInputs are the already-collected sub-tool documents BuildCard folds. Each
// arm keeps the raw JSON-object shape its collector produced (the Python card
// passes the same dicts around), so the fold is a pure function a test drives
// with synthetic documents — no subprocess, no gh, no schtasks.
//
// Arms of the Python card NOT yet folded natively (weekly cap, silent workers,
// backend health/stub rates, hook failures, run-status digests, worker/lease
// cross-check, seat inventory, Slack, --md) stay owned by the Python shim; their
// payload keys are OMITTED here rather than emitted as fake zeroes, so an absent
// fold never reads as "checked and clean" (#1406).
type CardInputs struct {
	Workspace  string
	Preflight  map[string]any // tools/dispatch_preflight.py --json
	Supervisor map[string]any // tools/dos_supervisor_status.py --json
	Watchdog   map[string]any // the schtasks always-on-task fold
	Backlog    map[string]any // tools/issue_lane_router.py --json (may carry _skipped/_error)
	Closure    map[string]any // tools/issue_closure_audit.py --json (may carry _skipped/_error)
	Throughput map[string]any // tools/dispatch_throughput.py --json (may carry _skipped/_error)
	Guard      map[string]any // GuardCoverage fold; nil omits the guard arm entirely
	Merge      map[string]any // the MERGE_HEAD wait-state fold
	Leases     map[string]any // the LeaseState summary (JSON-shaped, see SummarizeLeases)
	Fast       bool
}

// backlogFold is the router's authoritative routed/unrouted fold, extracted from
// BuildCard's backlog arm so the per-lane tally has a nameable home.
type backlogFold struct {
	laneCounts map[string]int
	openIssues int
	routed     any
	unrouted   any
	na         bool
}

// foldBacklog mirrors BuildCard's backlog section: per-lane issue counts, the
// routed/unrouted totals, and whether the whole arm is unavailable.
func foldBacklog(backlog map[string]any) backlogFold {
	lanes := mp(backlog["lanes"])
	bcounts := mp(backlog["counts"])
	laneCounts := map[string]int{}
	laneTotal := 0
	for ln, info := range lanes {
		n := laneIssueCount(info)
		laneCounts[ln] = n
		laneTotal += n
	}
	_, backlogSkipped := backlog["_skipped"]
	_, backlogErr := backlog["_error"]
	return backlogFold{
		laneCounts: laneCounts,
		openIssues: intDefault(bcounts["open"], laneTotal),
		routed:     intOrNil(bcounts["routed"]),
		unrouted:   intOrNil(bcounts["unrouted"]),
		na:         backlogSkipped || (backlogErr && len(lanes) == 0),
	}
}

// closureFold is the closure-honesty arm's counts plus its availability.
type closureFold struct {
	counts        map[string]any
	closureRate   any
	honestRate    any
	na            bool
	openWitnessed int
}

// foldClosure mirrors BuildCard's closure-honesty section.
func foldClosure(closure map[string]any) closureFold {
	counts := mp(closure["counts"])
	closureRate := closure["closure_rate"]
	_, closureSkipped := closure["_skipped"]
	_, closureErr := closure["_error"]
	return closureFold{
		counts:        counts,
		closureRate:   closureRate,
		honestRate:    closure["honest_close_rate"],
		na:            closureSkipped || (closureErr && closureRate == nil),
		openWitnessed: intDefault(counts["OPEN_WITNESSED"], 0),
	}
}

// isThroughputNA mirrors BuildCard's throughput availability check: the
// closed/hour arm reads as unavailable when it was skipped, errored, or never
// stamped a schema.
func isThroughputNA(tp map[string]any) bool {
	_, tpSkipped := tp["_skipped"]
	_, tpErr := tp["_error"]
	return tpSkipped || tpErr || strOf(tp["schema"]) == ""
}

// computeBaseVerdict mirrors BuildCard's overall-verdict switch: host safety
// and the preflight verdict decide ok/verdict/reasons before the merge, guard,
// and throughput arms get a chance to append or override.
func computeBaseVerdict(pre map[string]any, preVerdict any, capAny, liveAny any, hostSafe bool, acct map[string]any) (bool, string, []string) {
	var reasons []string
	var ok bool
	var verdict string
	switch {
	case !hostSafe:
		ok = false
		verdict = "HOST_FLAGGED"
		reasons = append(reasons, "host resource guard flagged a process — reap/inspect before growing")
	case strOf(preVerdict) == "REFUSE_INSPECT":
		ok = false
		verdict = "INSPECT"
		reasons = append(reasons, fmt.Sprintf("a safety preflight could not run: %s", pyAtom(pre["reason"])))
	case strOf(preVerdict) == "REFUSE_NO_ACCOUNT":
		ok = true
		verdict = "BLOCKED_ON_ACCOUNT"
		reasons = append(reasons, "no worker account free right now (switcher will resume when one frees)")
	case strOf(preVerdict) == "REFUSE_NO_SEAT":
		ok = true
		verdict = "BLOCKED_ON_SEAT"
		reasons = append(reasons, "no dispatch seat free right now (seat pool will resume when one frees)")
	case strOf(preVerdict) == "REFUSE_AT_CAP":
		ok = true
		verdict = "AT_CAP"
		reasons = append(reasons, fmt.Sprintf("%s/%s workers live — at the configured ceiling", pyAtom(liveAny), pyAtom(capAny)))
	default:
		ok = true
		verdict = "READY_TO_GROW"
		reasons = append(reasons, fmt.Sprintf("safe to spawn: %s/%s live, account '%s' free",
			pyAtom(liveAny), pyAtom(capAny), pyAtom(acct["tag"])))
	}
	return ok, verdict, reasons
}

// applyMergeOverride mirrors BuildCard's merge-in-progress override: a live
// merge dominates the base verdict and leads with its own next-action reason.
func applyMergeOverride(ok bool, verdict string, reasons []string, merge map[string]any) (bool, string, []string) {
	if !boolOf(merge["merge_in_progress"]) {
		return ok, verdict, reasons
	}
	next := strOf(merge["next_action"])
	if next == "" {
		next = "wait for MERGE_HEAD to clear before starting worker edits"
	}
	return false, "MERGE_IN_PROGRESS", append([]string{next}, reasons...)
}

// appendWatchdogReason mirrors BuildCard's always-on watchdog reason.
func appendWatchdogReason(reasons []string, wd map[string]any) []string {
	installed, isBool := wd["installed"].(bool)
	if !isBool {
		return reasons
	}
	if !installed {
		return append(reasons, "always-on watchdog NOT installed (register_dos_dispatch_watchdog.ps1)")
	}
	status := strOf(wd["status"])
	if status == "" {
		status = "scheduled"
	}
	return append(reasons, fmt.Sprintf("always-on watchdog installed (%s)", status))
}

// appendGuardReason mirrors BuildCard's guard-coverage reason: the witnessed
// proof the dispatch path ran THROUGH `fak guard`. Informational — it adds a
// reason but never flips ok.
func appendGuardReason(reasons []string, guard map[string]any) []string {
	if guard == nil {
		return reasons
	}
	gSessions := intDefault(guard["sessions"], 0)
	gRows := intDefault(guard["rows"], 0)
	switch {
	case gSessions > 0 && gRows > 0:
		crashes := intDefault(mp(guard["by_kind"])["CHILD_CRASH"], 0)
		mix := fmt.Sprintf("%d denied, %d quarantined",
			intDefault(guard["denied"], 0), intDefault(guard["quarantined"], 0))
		if crashes > 0 {
			suffix := ""
			if crashes != 1 {
				suffix = "es"
			}
			mix = fmt.Sprintf("%s, %d child crash%s", mix, crashes, suffix)
		}
		loopText := ""
		if label := guardLivelockLabel(guard); label != "" {
			loopText = fmt.Sprintf("; loop candidate: %s", label)
		}
		return append(reasons, fmt.Sprintf(
			"fak guard witnessed %d kernel decision(s) across %d dispatch session(s) (%s)%s",
			gRows, gSessions, mix, loopText))
	case gSessions > 0:
		return append(reasons, fmt.Sprintf(
			"fak guard ran %d dispatch session(s) but recorded 0 decisions (%d empty) — workers booted under guard but proposed no adjudicated tool call",
			gSessions, intDefault(guard["empty_sessions"], 0)))
	default:
		return reasons
	}
}

func guardLivelockLabel(guard map[string]any) string {
	candidates := listOf(guard["livelock_candidates"])
	if len(candidates) == 0 {
		return ""
	}
	row := mp(candidates[0])
	tool := strOf(row["tool"])
	if tool == "" {
		tool = strOf(row["kind"])
	}
	if tool == "" {
		tool = "?"
	}
	reason := strOf(row["reason"])
	if reason == "" {
		reason = strOf(row["verdict"])
	}
	if reason == "" {
		reason = "?"
	}
	digest := strOf(row["digest"])
	if len(digest) > 12 {
		digest = digest[:12]
	}
	if digest == "" {
		digest = "-"
	}
	return fmt.Sprintf("%s %s/%s digest=%s count=%s run=%s",
		pyAtom(row["file"]), tool, reason, digest,
		pyAtom(row["count"]), pyAtom(row["longest_run"]))
}

// appendThroughputReason mirrors BuildCard's throughput (closed/hour vs
// target) reason.
func appendThroughputReason(reasons []string, tp map[string]any, tpNA bool) []string {
	if tpNA {
		return reasons
	}
	tpVerdict := strOf(tp["verdict"])
	rate := pyAtom(tp["completed_rate_per_hour"])
	target := pyAtom(tp["target_per_hour"])
	win := pyAtom(tp["primary_window_hours"])
	if tpVerdict == "BELOW_TARGET" || tpVerdict == "AUDIT_ERROR" {
		return append(reasons, fmt.Sprintf(
			"throughput %s/h completed over the %sh analysis window — below the %s/h target", rate, win, target))
	}
	return append(reasons, fmt.Sprintf(
		"throughput %s (%s/h completed over the %sh analysis window, target %s/h)", tpVerdict, rate, win, target))
}

// appendLeasesReason mirrors BuildCard's lane-lease reason: whether the lease
// read failed, how many leases are active, and whether any block the current
// candidate issue(s).
func appendLeasesReason(reasons []string, leases map[string]any) []string {
	if readErr := strOf(leases["read_error"]); readErr != "" {
		return append(reasons, fmt.Sprintf("lease read unavailable: %s", readErr))
	}
	activeCount := intDefault(leases["active_count"], 0)
	if activeCount <= 0 {
		return reasons
	}
	blocking := intDefault(leases["blocking_count"], 0)
	if avail, isBool := leases["candidate_source_available"].(bool); isBool && !avail {
		return append(reasons, fmt.Sprintf(
			"%d active lane lease(s); candidate blocking unknown (backlog fold unavailable)", activeCount))
	}
	if blocking > 0 {
		var blockedNums []string
		for _, rowAny := range listOf(leases["active"]) {
			row := mp(rowAny)
			if !boolOf(row["blocks_candidate"]) {
				continue
			}
			for _, candAny := range listOf(row["blocking_candidates"]) {
				if issue := mp(candAny)["issue"]; issue != nil {
					blockedNums = append(blockedNums, "#"+pyAtom(issue))
				}
			}
		}
		suffix := ""
		if len(blockedNums) > 0 {
			if len(blockedNums) > 6 {
				blockedNums = blockedNums[:6]
			}
			suffix = fmt.Sprintf(" (%s)", strings.Join(blockedNums, ", "))
		}
		return append(reasons, fmt.Sprintf(
			"%d/%d active lane lease(s) block current candidate issue(s)%s", blocking, activeCount, suffix))
	}
	return append(reasons, fmt.Sprintf(
		"%d active lane lease(s), none blocking current candidates", activeCount))
}

// computeHeadroom mirrors the cap-minus-live projection shared by BuildCard's
// dispatcher.headroom field and dispatchLimiter's raw.headroom: nil unless
// both sides parse as numbers.
func computeHeadroom(capAny, liveAny any) any {
	c, cok := intOf(capAny)
	if !cok {
		return nil
	}
	l, lok := intOf(liveAny)
	if !lok {
		return nil
	}
	return c - l
}

// BuildCard is the Go port of dispatch_status.build_payload for the composite
// operator card's core arms: preflight + supervisor + watchdog + lane router +
// closure audit + throughput + lane leases + merge state, folded into the one
// verdict/reasons payload the operator reads. Healthy = host clean AND (can grow
// OR already at a healthy steady state); "no account free" / "at cap" are normal,
// not breakage.
func BuildCard(in CardInputs) map[string]any {
	pre := orEmpty(in.Preflight)
	sup := orEmpty(in.Supervisor)
	wd := orEmpty(in.Watchdog)
	backlog := orEmpty(in.Backlog)
	closure := orEmpty(in.Closure)
	tp := orEmpty(in.Throughput)
	merge := orEmpty(in.Merge)
	leases := orEmpty(in.Leases)

	// --- dispatcher liveness / capacity ---
	capAny := intOrNil(pre["cap"])
	liveAny := intOrNil(pre["live"])
	hostSafe := boolOf(mp(pre["host"])["safe"])
	acct := mp(pre["account"])
	preVerdict := pre["verdict"]

	bf := foldBacklog(backlog)
	cf := foldClosure(closure)
	tpNA := isThroughputNA(tp)

	// --- overall verdict ---
	ok, verdict, reasons := computeBaseVerdict(pre, preVerdict, capAny, liveAny, hostSafe, acct)
	// The weekly-cap override arm stays with the Python shim until its sidecar
	// fold is ported; without it READY_TO_GROW is the honest native verdict.

	ok, verdict, reasons = applyMergeOverride(ok, verdict, reasons, merge)
	reasons = appendWatchdogReason(reasons, wd)
	reasons = appendGuardReason(reasons, in.Guard)
	reasons = appendThroughputReason(reasons, tp, tpNA)
	reasons = appendLeasesReason(reasons, leases)

	limiter := dispatchLimiter(pre, backlog, closure, leases)
	headroom := computeHeadroom(capAny, liveAny)

	payload := map[string]any{
		"schema":    Schema,
		"ok":        ok,
		"verdict":   verdict,
		"reasons":   reasons,
		"workspace": in.Workspace,
		"dispatcher": map[string]any{
			"cap":               capAny,
			"live":              liveAny,
			"headroom":          headroom,
			"host_safe":         hostSafe,
			"preflight_verdict": preVerdict,
			"limiter":           limiter,
			"account": map[string]any{
				"tag": acct["tag"], "tier": acct["tier"], "model": acct["model"], "available": acct["available"],
			},
			"watchdog": wd,
		},
		"supervisor": map[string]any{
			"verdict": sup["verdict"],
			"target":  mp(sup["supervise"])["target"],
			"alive":   mp(sup["supervise"])["alive"],
			"plans":   sup["plans"],
		},
		"backlog": map[string]any{
			"na":          bf.na,
			"open_issues": nilWhen(bf.na, bf.openIssues),
			"routed":      nilWhenAny(bf.na, bf.routed),
			"by_lane":     nilWhenAny(bf.na, bf.laneCounts),
			"unrouted":    nilWhenAny(bf.na, bf.unrouted),
		},
		"closure": map[string]any{
			"na":                      cf.na,
			"closure_rate":            cf.closureRate,
			"honest_close_rate":       cf.honestRate,
			"counts":                  nilWhenAny(len(cf.counts) == 0, cf.counts),
			"open_witnessed_closable": nilWhen(cf.na, cf.openWitnessed),
		},
		"throughput": map[string]any{
			"na":                      tpNA,
			"verdict":                 nilWhenAny(tpNA, tp["verdict"]),
			"target_per_hour":         nilWhenAny(tpNA, tp["target_per_hour"]),
			"primary_window_hours":    nilWhenAny(tpNA, tp["primary_window_hours"]),
			"completed_rate_per_hour": nilWhenAny(tpNA, tp["completed_rate_per_hour"]),
			"raw_rate_per_hour":       nilWhenAny(tpNA, tp["raw_rate_per_hour"]),
			"per_window":              nilWhenAny(tpNA, mp(tp["gh"])["per_window"]),
			"loop_per_window":         nilWhenAny(tpNA, mp(tp["loop"])["per_window"]),
			"last_loop_close_age_min": nilWhenAny(tpNA, mp(tp["loop"])["last_loop_close_age_min"]),
		},
		"leases": leases,
		"git": map[string]any{
			"merge_in_progress": boolOf(merge["merge_in_progress"]),
			"merge_head":        merge["merge_head"],
			"next_action":       merge["next_action"],
		},
		"fast": in.Fast,
	}
	if in.Guard != nil {
		payload["guard"] = in.Guard
	}
	return payload
}

// dispatchLimiter mirrors dispatch_status._dispatch_limiter: the single "what is
// limiting spawn right now" projection. A GitHub rate-limit error anywhere in the
// gh-backed folds dominates; a blocking lane lease dominates the preflight's own
// capacity limiter; otherwise the preflight's limiter (or unknown) stands.
func dispatchLimiter(pre, backlog, closure, leases map[string]any) map[string]any {
	base := copyMap(mp(pre["capacity_limiter"]))
	raw := copyMap(mp(base["raw"]))
	setDefault(raw, "cap", pre["cap"])
	setDefault(raw, "live", pre["live"])
	setDefault(raw, "headroom", computeHeadroom(pre["cap"], pre["live"]))
	setDefault(raw, "max_workers", pre["max_workers"])
	setDefault(raw, "host_cap", pre["host_cap"])
	seat := mp(pre["seat"])
	setDefault(raw, "seat_total", seat["total"])
	setDefault(raw, "seat_free", seat["free"])
	setDefault(raw, "seat_leased", seat["leased"])
	raw["lane_leases_active"] = leases["active_count"]
	raw["lane_leases_blocking"] = leases["blocking_count"]

	if ghErr := githubRateLimitError(backlog, closure); ghErr != "" {
		raw["github_error"] = ghErr
		return map[string]any{"primary": "github_rate_limit", "term": "github_error", "raw": raw}
	}
	if intDefault(leases["blocking_count"], 0) > 0 {
		return map[string]any{"primary": "leases", "term": "lane_leases_blocking", "raw": raw}
	}
	if len(base) > 0 {
		base["raw"] = raw
		return base
	}
	return map[string]any{"primary": "unknown", "term": "unknown", "raw": raw}
}

// githubRateLimitError mirrors dispatch_status._github_rate_limit_error.
func githubRateLimitError(docs ...map[string]any) string {
	for _, doc := range docs {
		err := strOf(doc["_error"])
		lower := strings.ToLower(err)
		if strings.Contains(lower, "rate limit") || strings.Contains(lower, "secondary rate") {
			return err
		}
	}
	return ""
}
