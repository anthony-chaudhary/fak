package trajctl

// hygiene.go — issue #2570, the hostname/absolute-path hygiene rung. A persisted
// ledger row's free text (an objective statement, an evidence detail, a steer
// reason or packet, an annotation detail) can carry the machine's hostname or an
// absolute filesystem path. The nightrun ledgers learned this the hard way: a
// leaked absolute path or hostname makes a ledger un-shareable and, once noticed,
// gets the whole steering substrate disabled. ScrubRow redacts both classes of leak
// at the single persistence choke point (Append), and rowLeaks is the guard the
// hygiene test asserts on — it is RED on a row that carries a leak and GREEN once
// the row has been scrubbed.

import (
	"os"
	"regexp"
	"sort"
	"strings"
)

const (
	// redactedPath replaces a leaked absolute filesystem path.
	redactedPath = "[path]"
	// redactedHost replaces a leaked hostname.
	redactedHost = "[host]"
)

// absPathRE matches machine-revealing absolute filesystem paths: a Windows drive
// path (C:\Users\...), a UNC share (\\host\share\...), or a POSIX path rooted at a
// real top-level directory with at least one nested segment (/home/alice/...). A
// RELATIVE path (docs/nightrun/trajctl.jsonl) is intentionally NOT matched, so the
// ledger's own relative references survive scrubbing; only rooted, layout-revealing
// paths are redacted.
var absPathRE = regexp.MustCompile(
	`[A-Za-z]:\\[^\s"']*` +
		`|\\\\[A-Za-z0-9._-]+\\[^\s"']*` +
		`|/(?:home|Users|root|mnt|media|var|opt|srv|private|Volumes|usr|work|data|scratch|tmp)/[^\s"']+`,
)

// localHostnames returns the local machine's names (FQDN and short name), longest
// first so a longer name is redacted before a shorter name it contains. Empty,
// too-short, and "localhost" names are dropped so scrubbing cannot over-match.
func localHostnames() []string {
	h, err := os.Hostname()
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, 2)
	add := func(s string) {
		s = strings.TrimSpace(s)
		if len(s) < 2 || strings.EqualFold(s, "localhost") || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}
	add(h)
	if i := strings.IndexByte(h, '.'); i > 0 {
		add(h[:i])
	}
	sort.Slice(out, func(i, j int) bool { return len(out[i]) > len(out[j]) })
	return out
}

// scrubText redacts absolute paths and any local hostname from one free-text field.
// hosts is passed in (not read from the OS) so the fold is deterministic and
// testable; production ScrubRow supplies localHostnames().
func scrubText(text string, hosts []string) string {
	if text == "" {
		return text
	}
	out := absPathRE.ReplaceAllString(text, redactedPath)
	for _, h := range hosts {
		if len(h) >= 2 {
			out = strings.ReplaceAll(out, h, redactedHost)
		}
	}
	return out
}

// leakScan returns the machine-revealing substrings in text: absolute paths and any
// occurrence of a local hostname. A non-empty result means text would leak if
// persisted — the assertion the hygiene test is RED on.
func leakScan(text string, hosts []string) []string {
	leaks := absPathRE.FindAllString(text, -1)
	for _, h := range hosts {
		if len(h) >= 2 && strings.Contains(text, h) {
			leaks = append(leaks, h)
		}
	}
	return leaks
}

// scrubEvidence returns a deep copy of ev with every ref and detail scrubbed, so the
// caller's slice is never mutated.
func scrubEvidence(ev []EvidenceRef, hosts []string) []EvidenceRef {
	if len(ev) == 0 {
		return ev
	}
	out := make([]EvidenceRef, len(ev))
	copy(out, ev)
	for i := range out {
		out[i].Ref = scrubText(out[i].Ref, hosts)
		out[i].Detail = scrubText(out[i].Detail, hosts)
	}
	return out
}

// scrubRow returns row with every persisted free-text field scrubbed of absolute
// paths and local hostnames. It copies each payload before mutating it, so the
// caller's Row (and the structs it points at) are left untouched.
func scrubRow(row Row, hosts []string) Row {
	switch row.Kind {
	case KindObjective:
		if row.Objective != nil {
			o := *row.Objective
			o.Statement = scrubText(o.Statement, hosts)
			if len(o.Plan) > 0 {
				plan := make([]PlanPhase, len(o.Plan))
				copy(plan, o.Plan)
				for i := range plan {
					plan[i].Title = scrubText(plan[i].Title, hosts)
				}
				o.Plan = plan
			}
			row.Objective = &o
		}
	case KindScore:
		if row.Score != nil {
			s := *row.Score
			s.Evidence = scrubEvidence(s.Evidence, hosts)
			row.Score = &s
		}
	case KindSteer:
		if row.Steer != nil {
			d := *row.Steer
			d.Reason = scrubText(d.Reason, hosts)
			d.Packet = scrubText(d.Packet, hosts)
			d.DeliverErr = scrubText(d.DeliverErr, hosts)
			row.Steer = &d
		}
	case KindAnnotation:
		if row.Annotation != nil {
			a := *row.Annotation
			a.Detail = scrubText(a.Detail, hosts)
			row.Annotation = &a
		}
	}
	return row
}

// rowLeaks returns every machine-revealing substring across a row's persisted
// free-text fields. It is the hygiene guard: non-empty means the row would leak a
// hostname or absolute path if written to the ledger.
func rowLeaks(row Row, hosts []string) []string {
	var leaks []string
	switch row.Kind {
	case KindObjective:
		if row.Objective != nil {
			leaks = append(leaks, leakScan(row.Objective.Statement, hosts)...)
			for _, p := range row.Objective.Plan {
				leaks = append(leaks, leakScan(p.Title, hosts)...)
			}
		}
	case KindScore:
		if row.Score != nil {
			for _, e := range row.Score.Evidence {
				leaks = append(leaks, leakScan(e.Ref, hosts)...)
				leaks = append(leaks, leakScan(e.Detail, hosts)...)
			}
		}
	case KindSteer:
		if row.Steer != nil {
			leaks = append(leaks, leakScan(row.Steer.Reason, hosts)...)
			leaks = append(leaks, leakScan(row.Steer.Packet, hosts)...)
			leaks = append(leaks, leakScan(row.Steer.DeliverErr, hosts)...)
		}
	case KindAnnotation:
		if row.Annotation != nil {
			leaks = append(leaks, leakScan(row.Annotation.Detail, hosts)...)
		}
	}
	return leaks
}

// ScrubRow redacts absolute filesystem paths and the local hostname from every
// persisted free-text field of row. Append calls it at the single persistence choke
// point, so no ledger row can leak machine layout regardless of the caller.
func ScrubRow(row Row) Row { return scrubRow(row, localHostnames()) }
