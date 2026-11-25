package sessionreg

import (
	"encoding/json"
	"os"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// prevRegistry is the sessions.json this pass replaces: its throttle/auth maps carry
// forward account blockers whose evidence has aged out of the scan window.
type prevRegistry struct {
	throttle     *Obj
	auth         *Obj
	generatedUTC any
}

func loadPrevRegistry(path string) prevRegistry {
	var p prevRegistry
	data, err := os.ReadFile(path)
	if err != nil {
		return p
	}
	v, err := decodeOrdered(data)
	if err != nil {
		return p
	}
	doc, ok := v.(*Obj)
	if !ok {
		return p
	}
	p.throttle, _ = doc.Val("throttle").(*Obj)
	p.auth, _ = doc.Val("auth").(*Obj)
	p.generatedUTC = doc.Val("generated_utc")
	return p
}

// mergeKnownThrottle carries forward cached account limits whose reset has not expired,
// dropping accounts whose newest row is a clean LIVE turn.
func mergeKnownThrottle(o Options, throttle *Obj, rows []*Row, prev prevRegistry, isWorker func(string) bool) *Obj {
	sorted := append([]*Row(nil), rows...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].AgeMin < sorted[j].AgeMin })
	newest := map[string]*Row{}
	for _, r := range sorted {
		if _, ok := newest[r.Account]; !ok {
			newest[r.Account] = r
		}
	}
	cleared := map[string]bool{}
	for acct, r := range newest {
		if r.Disp == "LIVE" && !r.ThrottleCurrent {
			cleared[acct] = true
		}
	}
	merged := NewObj()
	for _, src := range []*Obj{prev.throttle, throttle} {
		for _, acct := range src.Keys() {
			info := src.Val(acct)
			if !isWorker(acct) || cleared[acct] || !throttleActive(info, o.Now) {
				continue
			}
			if m, ok := info.(*Obj); ok {
				merged.Set(acct, m)
			} else {
				e := NewObj()
				e.Set("reset", info)
				merged.Set(acct, e)
			}
		}
	}
	return merged
}

func rowSeen(o Options, r *Row) time.Time {
	if t := parseUTC(r.SeenUTC); t != nil {
		return *t
	}
	return o.Now.Add(-time.Duration(r.AgeMin * float64(time.Minute)))
}

func authInfoSeen(info *Obj, generated any) *time.Time {
	if info == nil {
		return nil
	}
	if t := parseUTC(info.Val("seen_utc")); t != nil {
		return t
	}
	age := pyFloat(info.Val("age_min"))
	gen := parseUTC(generated)
	if age == nil || gen == nil {
		return nil
	}
	t := gen.Add(-time.Duration(*age * float64(time.Minute)))
	return &t
}

func authInfoFromRow(o Options, r *Row) *Obj {
	last := r.Last
	if last == "" {
		last = r.Reason
	}
	e := NewObj()
	e.Set("block_kind", sessionsignals.AuthBlockKind(last))
	e.Set("block_reason", sessionsignals.AuthBlockReason(last))
	e.Set("seen_utc", isoUTC(rowSeen(o, r)))
	e.Set("age_min", r.AgeMin)
	e.Set("session", r.Session)
	e.Set("project", r.Project)
	e.Set("last", runeHead(last, 200))
	return e
}

func normalizeAuthInfo(info any) *Obj {
	var row *Obj
	if m, ok := info.(*Obj); ok {
		row = m.Copy()
	} else {
		row = NewObj()
		row.Set("block_kind", "auth")
		if truthy(info) {
			row.Set("block_reason", pyStr(info))
		} else {
			row.Set("block_reason", "auth/login required")
		}
	}
	text := ""
	for i, k := range []string{"last", "reason", "block_reason"} {
		if i > 0 {
			text += " "
		}
		if v := row.Val(k); truthy(v) {
			text += pyStr(v)
		}
	}
	if pyStrip(text) != "" {
		row.Set("block_kind", sessionsignals.AuthBlockKind(text))
		row.Set("block_reason", sessionsignals.AuthBlockReason(text))
	} else {
		if _, ok := row.Get("block_kind"); !ok {
			row.Set("block_kind", "auth")
		}
		if _, ok := row.Get("block_reason"); !ok {
			row.Set("block_reason", "auth/login required")
		}
	}
	return row
}

// mergeKnownAuth carries forward account auth blockers until a newer successful turn on the
// same account clears them.
func mergeKnownAuth(o Options, rows []*Row, prev prevRegistry, isWorker func(string) bool) *Obj {
	latestSuccess := map[string]time.Time{}
	current := NewObj()
	for _, r := range rows {
		if r.Account == "" {
			continue
		}
		seen := rowSeen(o, r)
		if r.Disp == "LIVE" || r.Disp == "DONE" {
			if cur, ok := latestSuccess[r.Account]; !ok || seen.After(cur) {
				latestSuccess[r.Account] = seen
			}
		}
		if r.Disp == "INFRA_AUTH" {
			cur, _ := current.Val(r.Account).(*Obj)
			curSeen := time.Time{}
			if t := authInfoSeen(cur, nil); t != nil {
				curSeen = *t
			}
			if cur == nil || seen.After(curSeen) {
				current.Set(r.Account, authInfoFromRow(o, r))
			}
		}
	}
	merged := NewObj()
	for _, src := range []*Obj{prev.auth, current} {
		for _, acct := range src.Keys() {
			if !isWorker(acct) {
				continue
			}
			row := normalizeAuthInfo(src.Val(acct))
			seen := authInfoSeen(row, prev.generatedUTC)
			if ok, has := latestSuccess[acct]; has && seen != nil && ok.After(*seen) {
				continue
			}
			merged.Set(acct, row)
		}
	}
	return merged
}

// availability is one worker account's admission verdict as the re-home ranking and the
// registry's accounts[] block see it.
type availability struct {
	Account, Tag, ConfigDir      string
	Available, Blocked           bool
	BlockKind                    *string
	BlockReason                  string
	Throttled                    bool
	Reset, Weekly                *string
	ActiveSessions, LiveSessions int
	AuthBlockedSessions          int
	VerdictSource                string
	VerdictAgeMin                *float64
}

func (a availability) obj() *Obj {
	e := NewObj()
	e.Set("account", a.Account)
	e.Set("tag", a.Tag)
	e.Set("config_dir", a.ConfigDir)
	e.Set("available", a.Available)
	e.Set("blocked", a.Blocked)
	e.Set("block_kind", strOrNil(a.BlockKind))
	e.Set("block_reason", a.BlockReason)
	e.Set("throttled", a.Throttled)
	e.Set("reset", strOrNil(a.Reset))
	e.Set("weekly", strOrNil(a.Weekly))
	e.Set("active_sessions", a.ActiveSessions)
	e.Set("live_sessions", a.LiveSessions)
	e.Set("auth_blocked_sessions", a.AuthBlockedSessions)
	e.Set("verdict_source", a.VerdictSource)
	if a.VerdictAgeMin == nil {
		e.Set("verdict_age_min", nil)
	} else {
		e.Set("verdict_age_min", *a.VerdictAgeMin)
	}
	return e
}

func strOrNil(p *string) any {
	if p == nil {
		return nil
	}
	return *p
}

func availabilityObjs(av []availability) []any {
	out := make([]any, len(av))
	for i, a := range av {
		out[i] = a.obj()
	}
	return out
}

// sessionRecord is the subset of a row the fleetaccounts runtime fold reads.
func sessionRecord(r *Row) map[string]any {
	m := map[string]any{
		"account": r.Account, "project": r.Project, "disp": r.Disp,
		"age_min": r.AgeMin, "seen_utc": r.SeenUTC, "last": r.Last, "reason": r.Reason,
	}
	if r.Action != "" {
		m["action"] = r.Action
	}
	if r.ThrottleReset != "" {
		m["throttle_reset"] = r.ThrottleReset
	}
	if r.ThrottleWeekly != "" {
		m["throttle_weekly"] = r.ThrottleWeekly
	}
	return m
}

// accountAvailability is the per-worker "safe to offer right now" view, folded by
// fleetaccounts from exactly this pass's registry: the merged throttle/auth maps and the
// rows just classified.
func accountAvailability(o Options, roster []fleetaccounts.Account, throttle, auth *Obj, rows []*Row) []availability {
	sessions := make([]map[string]any, len(rows))
	for i, r := range rows {
		sessions[i] = sessionRecord(r)
	}
	doc := map[string]any{"generated_utc": isoUTC(o.Now), "auth": plain(auth), "sessions": sessions}
	if throttle.Len() > 0 {
		doc["throttle"] = plain(throttle)
	}
	data, _ := json.Marshal(doc)
	annotated := fleetaccounts.AnnotatePassive(roster, fleetaccounts.ParseRegistry(data), o.Now)
	var out []availability
	for _, a := range annotated {
		if a.Kind != fleetaccounts.KindWorker {
			continue
		}
		src, age := verdictFreshness(o, a.Account, rows, auth)
		out = append(out, availability{
			Account: a.Account, Tag: a.Tag, ConfigDir: pyJoin(o.Home, a.Account),
			Available: derefBool(a.Available), Blocked: derefBool(a.Blocked),
			BlockKind: a.BlockKind, BlockReason: derefStr(a.BlockReason),
			Throttled: derefBool(a.Throttled), Reset: a.Reset, Weekly: a.Weekly,
			ActiveSessions: derefInt(a.ActiveSessions), LiveSessions: derefInt(a.LiveSessions),
			AuthBlockedSessions: derefInt(a.AuthBlockedSessions),
			VerdictSource:       src, VerdictAgeMin: age,
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Available != out[j].Available {
			return out[i].Available
		}
		return out[i].Tag < out[j].Tag
	})
	return out
}

// verdictFreshness names where an account's verdict comes from: "passive" when a real row
// inside the window backs it, else "carried" from the auth map (aged from its seen_utc).
func verdictFreshness(o Options, account string, rows []*Row, auth *Obj) (string, *float64) {
	have := false
	minAge := 0.0
	for _, r := range rows {
		if r.Account != account || r.Project == "_probe" {
			continue
		}
		if !have || r.AgeMin < minAge {
			minAge, have = r.AgeMin, true
		}
	}
	if have {
		if minAge == 0 {
			minAge = 0 // the writer's float(age or 0.0) turns a rounded -0.0 into 0.0
		}
		v := pyRound(minAge, 1)
		return "passive", &v
	}
	if info, ok := auth.Val(account).(*Obj); ok {
		if seen := parseUTC(info.Val("seen_utc")); seen != nil {
			v := pyRound(minutesSince(o.Now, *seen), 1)
			return "carried", &v
		}
	}
	return "carried", nil
}

func derefBool(p *bool) bool { return p != nil && *p }

func derefStr(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
