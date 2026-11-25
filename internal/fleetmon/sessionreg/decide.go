package sessionreg

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
)

// ResumePrompt is the synthetic prompt every planned resume carries.
const ResumePrompt = "Resume where you left off; re-establish any /goal or /loop and continue toward it."

var (
	deadDisps = map[string]bool{"DEAD_MIDTOOL": true, "DEAD_KILLED": true}
	// The server interrupted these sessions (they did not finish), so they resume
	// regardless of autonomy.
	infraResumable = map[string]bool{"STOPPED_LIMIT": true, "STOPPED_APIERR": true, "INFRA_ORG_DISABLED": true}
	// STOPPED_CONTEXT_EXHAUSTED is deliberately absent: a resume re-hits the same wall, so
	// it never earns a resume command.
	stopLike = map[string]bool{
		"DEAD_MIDTOOL": true, "DEAD_KILLED": true, "STOPPED_LIMIT": true, "STOPPED_APIERR": true,
		"INFRA_ORG_DISABLED": true, "STOPPED_QUIET": true, "INFRA_AUTH": true, "USER_CLOSED": true,
		"PARKED_WAIT": true, "NEVER_STARTED": true,
	}
	preStamped = map[string]bool{"DEFER_DUPLICATE_TASK": true, "SKIP_LIVE_DRIVER": true, "SKIP_STALE_COPY": true}
)

func (o Options) configDir(acct string) string { return pyJoin(o.Home, acct) }

func resumableDisp(r *Row) bool {
	if infraResumable[r.Disp] {
		return true
	}
	return r.Autonomous && (deadDisps[r.Disp] || r.Disp == "NEVER_STARTED")
}

// copyRank orders copies of one session id: last-turn timestamp, then transcript size, then
// file mtime. mtime is last on purpose: a dead driver appending a synthetic error banner to
// a stale prefix bumps the mtime without adding a real turn.
func copyRankGreater(a, b *Row) bool {
	if a.LastTS != b.LastTS {
		return a.LastTS > b.LastTS
	}
	if a.Records != b.Records {
		return a.Records > b.Records
	}
	return a.SeenUTC > b.SeenUTC
}

func newestBySID(rows []*Row) map[string]*Row {
	best := map[string]*Row{}
	for _, r := range rows {
		if cur, ok := best[r.Session]; !ok || copyRankGreater(r, cur) {
			best[r.Session] = r
		}
	}
	return best
}

// liveOrStaleCopyGate drops every copy of a session a live driver is advancing and lets
// only the newest copy of each remaining session reach the decision ladder.
func liveOrStaleCopyGate(rows []*Row, liveSIDs map[string]bool) {
	live := map[string]bool{}
	for s := range liveSIDs {
		live[s] = true
	}
	for _, r := range rows {
		if r.Disp == "LIVE" {
			live[r.Session] = true
		}
	}
	delete(live, "")
	newest := newestBySID(rows)
	for _, r := range rows {
		if r.Disp == "LIVE" {
			continue
		}
		if r.Session != "" && live[r.Session] {
			r.stampEarly("SKIP_LIVE_DRIVER")
		} else if newest[r.Session] != r {
			r.stampEarly("SKIP_STALE_COPY")
		}
	}
}

// descLess compares the way the Python tuple(-ord(c) for c in s) key does: a higher code
// point at the first difference wins, and a proper prefix wins over its extension.
func descLess(a, b string) bool {
	ar, br := []rune(a), []rune(b)
	for i := 0; i < len(ar) && i < len(br); i++ {
		if ar[i] != br[i] {
			return ar[i] > br[i]
		}
	}
	return len(ar) < len(br)
}

// dedupDefer resumes only one of several crashed sessions that are the same recurring
// autonomous task, deferring the rest; a LIVE/DONE sibling covers the whole task.
func dedupDefer(rows []*Row, blocked map[string]bool) {
	bySID := map[string]*Row{}
	var order []string
	for _, r := range rows {
		cur, ok := bySID[r.Session]
		if !ok {
			order = append(order, r.Session)
		}
		if !ok || r.SeenUTC > cur.SeenUTC || (r.SeenUTC == cur.SeenUTC && r.Records > cur.Records) {
			bySID[r.Session] = r
		}
	}
	groups := map[string][]*Row{}
	var sigs []string
	for _, sid := range order {
		r := bySID[sid]
		if r.TaskSig == "" {
			continue
		}
		if _, ok := groups[r.TaskSig]; !ok {
			sigs = append(sigs, r.TaskSig)
		}
		groups[r.TaskSig] = append(groups[r.TaskSig], r)
	}
	for _, sig := range sigs {
		members := groups[sig]
		var resumable []*Row
		covered := false
		for _, r := range members {
			if resumableDisp(r) {
				resumable = append(resumable, r)
			}
			if r.Disp == "LIVE" || r.Disp == "DONE" {
				covered = true
			}
		}
		if len(resumable) <= 1 {
			continue
		}
		if covered {
			for _, r := range resumable {
				r.stampEarly("DEFER_DUPLICATE_TASK")
			}
			continue
		}
		var eligible []*Row
		for _, r := range resumable {
			if !blocked[r.Session] {
				eligible = append(eligible, r)
			}
		}
		if len(eligible) == 0 {
			eligible = resumable
		}
		winner := eligible[0]
		for _, r := range eligible[1:] {
			if r.Records != winner.Records {
				if r.Records > winner.Records {
					winner = r
				}
				continue
			}
			if r.SeenUTC != winner.SeenUTC {
				if descLess(r.SeenUTC, winner.SeenUTC) {
					winner = r
				}
				continue
			}
			if r.Session < winner.Session {
				winner = r
			}
		}
		for _, r := range resumable {
			if r != winner {
				r.stampEarly("DEFER_DUPLICATE_TASK")
			}
		}
	}
}

func targetsOf(av []availability) []rehome.Target {
	out := make([]rehome.Target, len(av))
	for i, a := range av {
		out[i] = rehome.Target{
			Account: a.Account, Available: a.Available, LiveSessions: a.LiveSessions,
			ActiveSessions: a.ActiveSessions, Tag: a.Tag, ConfigDir: a.ConfigDir, VerdictSource: a.VerdictSource,
		}
	}
	return out
}

func hasPositiveEvidence(source string) bool { return source == "probe" || source == "passive" }

// admissibleTargets is the ranked re-home pool cut to seats whose health rests on positive
// evidence: a carried "available" never takes a workload until a fresh probe confirms it.
func (d *decider) admissibleTargets(exclude string) []rehome.Target {
	var out []rehome.Target
	for _, t := range rehome.RehomeTargets(d.targets, exclude, d.assigned, d.o.RehomeCap) {
		if hasPositiveEvidence(t.VerdictSource) {
			out = append(out, t)
		}
	}
	return out
}

func (d *decider) ownerAvailable(account string) bool {
	for _, a := range d.av {
		if a.Account == account {
			return a.Available && hasPositiveEvidence(a.VerdictSource)
		}
	}
	return false
}

type decider struct {
	o        Options
	av       []availability
	targets  []rehome.Target
	assigned map[string]int
	inplace  map[string]int
}

func (d *decider) rehomeTo(r *Row, t rehome.Target) {
	r.Rehomed = true
	r.ResumeAccount = t.Account
	r.ResumeConfigDir = t.ConfigDir
	if r.ResumeConfigDir == "" {
		r.ResumeConfigDir = d.o.configDir(t.Account)
	}
	d.assigned[t.Account]++
}

// resumeInplaceOrEscalate stamps AUTO_RESUME on the owner, but moves the resume to a less
// loaded proven seat when the owner already carries RehomeCap live sessions, or to another
// seat once in-place resumes have failed EscalateAfter times.
func (d *decider) resumeInplaceOrEscalate(r *Row) {
	r.Action = "AUTO_RESUME"
	if d.o.LoadRehome {
		live, overloaded := 0, false
		for _, a := range d.av {
			if a.Account == r.Account {
				live = a.LiveSessions
				overloaded = d.o.RehomeCap > 0 && live >= d.o.RehomeCap
				break
			}
		}
		if overloaded {
			if ts := d.admissibleTargets(r.Account); len(ts) > 0 {
				if ts[0].LiveSessions+d.assigned[ts[0].Account] < live {
					d.rehomeTo(r, ts[0])
					return
				}
			}
		}
	}
	if d.o.EscalateAfter <= 0 || d.inplace[r.Session] < d.o.EscalateAfter {
		return
	}
	if ts := d.admissibleTargets(r.Account); len(ts) > 0 {
		d.rehomeTo(r, ts[0])
	}
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// decide stamps every row with a deterministic action and an account-correct resume
// command. Only resumable sessions earn AUTO_RESUME; finished and user-closed sessions are
// held back explicitly.
func decide(o Options, rows []*Row, throttle *Obj, av []availability, liveSIDs map[string]bool, lf ledgerFolds) {
	liveOrStaleCopyGate(rows, liveSIDs)
	dedupDefer(rows, lf.blocked)
	d := &decider{o: o, av: av, targets: targetsOf(av), assigned: map[string]int{}, inplace: lf.inplace}
	for _, r := range rows {
		cwd := r.cwdString()
		cwdOK := cwd != "" && isDir(cwd)
		r.decided = true
		r.SourceConfigDir = o.configDir(r.Account)
		r.ResumeAccount = r.Account
		r.ResumeConfigDir = o.configDir(r.Account)
		r.Rehomed = false
		if preStamped[r.Action] {
			r.hasResumeCmd = false
			continue
		}
		_, throttled := throttle.Get(r.Account)
		switch {
		case strings.Contains(r.Project, "pytest") || !cwdOK:
			r.Action = "SKIP_EPHEMERAL"
		case r.Disp == "LIVE":
			r.Action = "SKIP_LIVE"
		case r.Disp == "DONE":
			r.Action = "SKIP_DONE"
		case r.Disp == "USER_CLOSED":
			r.Action = "SKIP_USER_CLOSED"
		case r.Disp == "PARKED_WAIT":
			r.Action = "SKIP_PARKED"
		case r.Supervised:
			r.Action = "SUPERVISED"
		case r.Disp == "STOPPED_CONTEXT_EXHAUSTED":
			r.Action = "ESCALATE_FRESH_CONTINUATION"
		case r.Disp == "STOPPED_LIMIT" || throttled:
			// A STOPPED_LIMIT banner on an owner the throttle map no longer holds, whose seat
			// reads admissible, is a stale banner copied off a once-throttled owner.
			if !throttled && r.Disp == "STOPPED_LIMIT" && d.ownerAvailable(r.Account) {
				r.Action = "AUTO_RESUME"
				break
			}
			resumable := resumableDisp(r) && (deadDisps[r.Disp] || r.Disp == "STOPPED_LIMIT" || r.Disp == "STOPPED_APIERR")
			var ts []rehome.Target
			if resumable {
				ts = d.admissibleTargets(r.Account)
			}
			if len(ts) > 0 {
				r.Action = "AUTO_RESUME"
				d.rehomeTo(r, ts[0])
			} else {
				r.Action = "DEFER_THROTTLED"
			}
		case r.Disp == "INFRA_ORG_DISABLED":
			var ts []rehome.Target
			if resumableDisp(r) {
				ts = d.admissibleTargets(r.Account)
			}
			if len(ts) > 0 {
				r.Action = "AUTO_RESUME"
				d.rehomeTo(r, ts[0])
			} else {
				r.Action = "DEFER_NO_USAGE"
			}
		case r.Disp == "INFRA_AUTH":
			// An admissible owner contradicts the login banner: the failure is session-local.
			if d.ownerAvailable(r.Account) {
				d.resumeInplaceOrEscalate(r)
			} else {
				r.Action = "BLOCKED_AUTH"
			}
		case r.Disp == "STOPPED_APIERR":
			d.resumeInplaceOrEscalate(r)
		case deadDisps[r.Disp] && r.Autonomous:
			d.resumeInplaceOrEscalate(r)
		case r.Disp == "NEVER_STARTED" && r.Autonomous:
			d.resumeInplaceOrEscalate(r)
		case deadDisps[r.Disp] || r.Disp == "STOPPED_QUIET" || r.Disp == "NEVER_STARTED":
			r.Action = "SURFACE"
		default:
			r.Action = "SKIP"
		}
		r.hasResumeCmd = stopLike[r.Disp]
		if r.hasResumeCmd {
			r.ResumeCmd = o.resumeCmd(r)
		}
	}
}

// resumeCmd is the operator-runnable, account-correct resume. A re-homed session first
// copies its transcript under the target account, since `claude --resume` only finds a
// conversation stored under the CLAUDE_CONFIG_DIR it runs in.
func (o Options) resumeCmd(r *Row) string {
	cfg := r.ResumeConfigDir
	if cfg == "" {
		cfg = o.configDir(r.Account)
	}
	prefix := ""
	if r.Rehomed {
		src := r.SourceConfigDir
		if src == "" {
			src = o.configDir(r.Account)
		}
		srcFile := pyJoin(src, "projects", r.Project, r.Session+".jsonl")
		dstDir := pyJoin(cfg, "projects", r.Project)
		prefix = "New-Item -ItemType Directory -Force -Path '" + dstDir + "' | Out-Null; " +
			"Copy-Item '" + srcFile + "' '" + pyJoin(dstDir, r.Session+".jsonl") + "' -Force; "
	}
	return prefix + "$env:CLAUDE_CONFIG_DIR='" + cfg + "'; claude --resume " + r.Session +
		" -p '" + ResumePrompt + "' --dangerously-skip-permissions"
}
