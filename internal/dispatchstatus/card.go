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

	// --- backlog --- (counts is the router's authoritative routed/unrouted fold)
	lanes := mp(backlog["lanes"])
	bcounts := mp(backlog["counts"])
	laneCounts := map[string]int{}
	laneTotal := 0
	for ln, info := range lanes {
		n := laneIssueCount(info)
		laneCounts[ln] = n
		laneTotal += n
	}
	openIssues := intDefault(bcounts["open"], laneTotal)
	routed := intOrNil(bcounts["routed"])
	unrouted := intOrNil(bcounts["unrouted"])
	_, backlogSkipped := backlog["_skipped"]
	_, backlogErr := backlog["_error"]
	backlogNA := backlogSkipped || (backlogErr && len(lanes) == 0)

	// --- closure honesty ---
	counts := mp(closure["counts"])
	closureRate := closure["closure_rate"]
	honestRate := closure["honest_close_rate"]
	_, closureSkipped := closure["_skipped"]
	_, closureErr := closure["_error"]
	closureNA := closureSkipped || (closureErr && closureRate == nil)
	openWitnessed := intDefault(counts["OPEN_WITNESSED"], 0)

	// --- throughput (closed/hour vs target) ---
	_, tpSkipped := tp["_skipped"]
	_, tpErr := tp["_error"]
	tpNA := tpSkipped || tpErr || strOf(tp["schema"]) == ""

	// --- overall verdict ---
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
	// The weekly-cap override arm stays with the Python shim until its sidecar
	// fold is ported; without it READY_TO_GROW is the honest native verdict.

	if boolOf(merge["merge_in_progress"]) {
		ok = false
		verdict = "MERGE_IN_PROGRESS"
		next := strOf(merge["next_action"])
		if next == "" {
			next = "wait for MERGE_HEAD to clear before starting worker edits"
		}
		reasons = append([]string{next}, reasons...)
	}

	if installed, isBool := wd["installed"].(bool); isBool {
		if installed {
			status := strOf(wd["status"])
			if status == "" {
				status = "scheduled"
			}
			reasons = append(reasons, fmt.Sprintf("always-on watchdog installed (%s)", status))
		} else {
			reasons = append(reasons, "always-on watchdog NOT installed (register_dos_dispatch_watchdog.ps1)")
		}
	}

	// Guard coverage: the witnessed proof the dispatch path ran THROUGH `fak guard`.
	// Informational — it adds a reason but never flips ok.
	if in.Guard != nil {
		gSessions := intDefault(in.Guard["sessions"], 0)
		gRows := intDefault(in.Guard["rows"], 0)
		if gSessions > 0 && gRows > 0 {
			reasons = append(reasons, fmt.Sprintf(
				"fak guard witnessed %d kernel decision(s) across %d dispatch session(s) (%d denied, %d quarantined)",
				gRows, gSessions, intDefault(in.Guard["denied"], 0), intDefault(in.Guard["quarantined"], 0)))
		} else if gSessions > 0 {
			reasons = append(reasons, fmt.Sprintf(
				"fak guard ran %d dispatch session(s) but recorded 0 decisions (%d empty) — workers booted under guard but proposed no adjudicated tool call",
				gSessions, intDefault(in.Guard["empty_sessions"], 0)))
		}
	}

	if !tpNA {
		tpVerdict := strOf(tp["verdict"])
		rate := pyAtom(tp["completed_rate_per_hour"])
		target := pyAtom(tp["target_per_hour"])
		win := pyAtom(tp["primary_window_hours"])
		if tpVerdict == "BELOW_TARGET" || tpVerdict == "AUDIT_ERROR" {
			reasons = append(reasons, fmt.Sprintf(
				"throughput %s/h completed over the %sh analysis window — below the %s/h target", rate, win, target))
		} else {
			reasons = append(reasons, fmt.Sprintf(
				"throughput %s (%s/h completed over the %sh analysis window, target %s/h)", tpVerdict, rate, win, target))
		}
	}

	if readErr := strOf(leases["read_error"]); readErr != "" {
		reasons = append(reasons, fmt.Sprintf("lease read unavailable: %s", readErr))
	} else if activeCount := intDefault(leases["active_count"], 0); activeCount > 0 {
		blocking := intDefault(leases["blocking_count"], 0)
		if avail, isBool := leases["candidate_source_available"].(bool); isBool && !avail {
			reasons = append(reasons, fmt.Sprintf(
				"%d active lane lease(s); candidate blocking unknown (backlog fold unavailable)", activeCount))
		} else if blocking > 0 {
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
			reasons = append(reasons, fmt.Sprintf(
				"%d/%d active lane lease(s) block current candidate issue(s)%s", blocking, activeCount, suffix))
		} else {
			reasons = append(reasons, fmt.Sprintf(
				"%d active lane lease(s), none blocking current candidates", activeCount))
		}
	}

	limiter := dispatchLimiter(pre, backlog, closure, leases)

	var headroom any
	if c, cok := intOf(capAny); cok {
		if l, lok := intOf(liveAny); lok {
			headroom = c - l
		}
	}

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
			"na":          backlogNA,
			"open_issues": nilWhen(backlogNA, openIssues),
			"routed":      nilWhenAny(backlogNA, routed),
			"by_lane":     nilWhenAny(backlogNA, laneCounts),
			"unrouted":    nilWhenAny(backlogNA, unrouted),
		},
		"closure": map[string]any{
			"na":                      closureNA,
			"closure_rate":            closureRate,
			"honest_close_rate":       honestRate,
			"counts":                  nilWhenAny(len(counts) == 0, counts),
			"open_witnessed_closable": nilWhen(closureNA, openWitnessed),
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
	var headroom any
	if c, cok := intOf(pre["cap"]); cok {
		if l, lok := intOf(pre["live"]); lok {
			headroom = c - l
		}
	}
	setDefault(raw, "headroom", headroom)
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
