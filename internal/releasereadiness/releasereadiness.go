package releasereadiness

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const Schema = "fak-release-readiness/1"

var Bands = []string{"discover", "automate", "validate", "trust"}

type Facts struct {
	ReleaseVerb, StalenessVerb, AgentsRelease, LLMSRelease, CadenceAutoCut, CadenceDefaultOn, StalenessWired, ReleaseToolsTested, PostPublishVerify, PostPublishQuarantine, LockPresent, GotchasBounded, ArtifactsSigned, GHReachable, Arm64Shipped, ArtifactsPresent bool
	LatestTag, StalenessVerdict                                                                                                                                                                                                                                       string
	CommitsBehind, GotchaCount                                                                                                                                                                                                                                        int
	DaysBehind                                                                                                                                                                                                                                                        float64
	StableTags                                                                                                                                                                                                                                                        []string
}
type Row struct {
	Key         string `json:"key"`
	Band        string `json:"band"`
	Weight      int    `json:"weight"`
	OK          bool   `json:"ok"`
	Unwitnessed bool   `json:"unwitnessed"`
	Label       string `json:"label"`
	Fix         string `json:"fix"`
}
type Staleness struct {
	LatestTag     string  `json:"latest_tag"`
	CommitsBehind int     `json:"commits_behind"`
	DaysBehind    float64 `json:"days_behind"`
	Verdict       string  `json:"verdict"`
}
type Payload struct {
	Schema      string             `json:"schema"`
	OK          bool               `json:"ok"`
	Verdict     string             `json:"verdict"`
	Finding     string             `json:"finding"`
	NextAction  string             `json:"next_action"`
	Workspace   string             `json:"workspace"`
	Score       float64            `json:"score"`
	Grade       string             `json:"grade"`
	ReleaseDebt int                `json:"release_debt"`
	SoftSignals int                `json:"soft_signals"`
	BandScores  map[string]float64 `json:"band_scores"`
	DebtByBand  map[string]int     `json:"debt_by_band"`
	Staleness   Staleness          `json:"staleness"`
	GotchaCount int                `json:"gotcha_count"`
	StableTags  []string           `json:"stable_tags"`
	Rows        []Row              `json:"rows"`
	Epic        int                `json:"epic"`
}
type kpi struct {
	key, band, label, fix string
	weight                int
	value                 func(Facts) *bool
}

func bp(v bool) *bool { return &v }

var kpis = []kpi{
	{"fak_release_verb", "discover", "`fak release` is a dispatched verb", "Add a `fak release` subcommand (#1356)", 3, func(f Facts) *bool { return bp(f.ReleaseVerb) }},
	{"agents_md_release", "discover", "AGENTS.md documents the release path", "Add a Releasing section to AGENTS.md (#1357)", 2, func(f Facts) *bool { return bp(f.AgentsRelease) }},
	{"llms_release", "discover", "llms.txt points at the release path", "Add a release-path entry to llms.txt (#1357)", 2, func(f Facts) *bool { return bp(f.LLMSRelease) }},
	{"staleness_verb", "discover", "`fak release-staleness` exists", "Keep the staleness verb dispatched", 1, func(f Facts) *bool { return bp(f.StalenessVerb) }},
	{"cadence_auto_cut", "automate", "cadence can cut on a scheduled tick", "Add guarded auto-cut to release-cadence.yml (#1355)", 3, func(f Facts) *bool { return bp(f.CadenceAutoCut) }},
	{"cadence_auto_cut_default_on", "automate", "scheduled auto-cut is DEFAULT-ON (kill switch, not arm)", "Make auto-cut default-on with a FAK_AUTO_RELEASE=0 kill switch (#1355)", 2, func(f Facts) *bool { return bp(f.CadenceDefaultOn) }},
	{"staleness_wired", "automate", "staleness signal wired into make/CI", "Wire `fak release-staleness --check` into a target/CI (#1367)", 2, func(f Facts) *bool { return bp(f.StalenessWired) }},
	{"not_very_stale", "automate", "@latest is not VERY_STALE", "Cut a release; automate the cadence (#1355)", 2, func(f Facts) *bool { return bp(f.StalenessVerdict != "VERY_STALE" && f.StalenessVerdict != "UNKNOWN") }},
	{"fresh", "automate", "@latest is FRESH vs HEAD", "Cut on green at agentic cadence (#1355)", 2, func(f Facts) *bool { return bp(f.StalenessVerdict == "FRESH") }},
	{"release_tools_tested", "validate", "release helpers carry unit tests", "Add tests for decide/cut/tag/publish", 2, func(f Facts) *bool { return bp(f.ReleaseToolsTested) }},
	{"post_publish_verify", "validate", "a published release is verified", "Add post-publish verification (#1369)", 3, func(f Facts) *bool { return bp(f.PostPublishVerify) }},
	{"post_publish_quarantine", "validate", "failed published-release verification quarantines the release", "Mark failed releases prerelease + non-latest (#1388)", 2, func(f Facts) *bool { return bp(f.PostPublishQuarantine) }},
	{"lock_present", "validate", "single-writer release lock present", "Keep the native release lock", 1, func(f Facts) *bool { return bp(f.LockPresent) }},
	{"gotchas_bounded", "validate", "documented gotcha count <= 1", "Eliminate the chicken-egg gotchas (#1368)", 2, func(f Facts) *bool { return bp(f.GotchasBounded) }},
	{"stable_exercised", "trust", "a stable/* rollback anchor exists", "Cut the first stable/* tag (#1370)", 2, func(f Facts) *bool { return bp(len(f.StableTags) > 0) }},
	{"artifacts_signed", "trust", "artifacts carry signing/provenance", "Sign artifacts with cosign/SLSA (#1372)", 2, func(f Facts) *bool { return bp(f.ArtifactsSigned) }},
	{"arm64_shipped", "trust", "linux/arm64 leg actually shipped", "Ship the arm64 asset (#1371)", 1, func(f Facts) *bool {
		if !f.GHReachable {
			return nil
		}
		return bp(f.Arm64Shipped)
	}},
	{"artifacts_present", "trust", "recent release carries archives+checksums", "Fix release-artifacts.yml", 1, func(f Facts) *bool {
		if !f.GHReachable {
			return nil
		}
		return bp(f.ArtifactsPresent)
	}},
}

func read(root, rel string) string {
	b, e := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if e != nil {
		return ""
	}
	return string(b)
}
func exists(root, rel string) bool {
	_, e := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return e == nil
}
func run(root string, timeout time.Duration, name string, args ...string) (string, bool) {
	c := exec.Command(name, args...)
	c.Dir = root
	var b strings.Builder
	c.Stdout = &b
	if e := c.Start(); e != nil {
		return "", false
	}
	done := make(chan error, 1)
	go func() { done <- c.Wait() }()
	select {
	case e := <-done:
		return b.String(), e == nil
	case <-time.After(timeout):
		_ = c.Process.Kill()
		<-done
		return "", false
	}
}
func Gather(root string, probeGH bool) Facts {
	var f Facts
	main := read(root, "cmd/fak/main.go")
	f.ReleaseVerb = regexp.MustCompile(`case\s+"release"\s*:`).MatchString(main)
	f.StalenessVerb = regexp.MustCompile(`case\s+"release-staleness"\s*:`).MatchString(main)
	a := strings.ToLower(read(root, "AGENTS.md"))
	f.AgentsRelease = strings.Contains(a, "/release") || strings.Contains(a, "cut a release") || regexp.MustCompile(`(?m)^#+\s*releas`).MatchString(a)
	l := strings.ToLower(read(root, "llms.txt"))
	f.LLMSRelease = strings.Contains(l, "skills/release") || strings.Contains(l, "release skill") || strings.Contains(l, "docs/releases") || strings.Contains(l, "cut a release")
	c := read(root, ".github/workflows/release-cadence.yml")
	cl := strings.ToLower(c)
	f.CadenceAutoCut = strings.Contains(cl, "fak_auto_release") || strings.Contains(cl, "auto-cut")
	f.CadenceDefaultOn = f.CadenceAutoCut && strings.Contains(cl, "default-on") && strings.Contains(c, `!= "0"`)
	f.StalenessWired = strings.Contains(read(root, "Makefile"), "release-staleness")
	if !f.StalenessWired {
		m, _ := filepath.Glob(filepath.Join(root, ".github", "workflows", "*.yml"))
		for _, p := range m {
			b, _ := os.ReadFile(p)
			if strings.Contains(string(b), "release-staleness") {
				f.StalenessWired = true
				break
			}
		}
	}
	staleness(root, &f)
	n := 0
	for _, s := range []string{"decide", "cut", "tag", "publish"} {
		if exists(root, "tools/release_"+s+"_test.py") || exists(root, "cmd/fak/release_"+s+"_test.go") {
			n++
		}
	}
	f.ReleaseToolsTested = n >= 3
	w := read(root, ".github/workflows/release-artifacts.yml")
	wl := strings.ToLower(w)
	f.PostPublishVerify = strings.Contains(wl, "release-verify") || strings.Contains(wl, "go install") || exists(root, "tools/release_verify.py") || strings.Contains(wl, "verify-release")
	f.PostPublishQuarantine = (strings.Contains(wl, "failure()") || strings.Contains(wl, ".outcome == 'failure'")) && strings.Contains(wl, "gh api") && strings.Contains(wl, "releases/${release_id}") && strings.Contains(wl, "prerelease=true") && strings.Contains(wl, "make_latest=false")
	f.LockPresent = exists(root, "internal/release/lock.go") || exists(root, "tools/release_lock.py")
	s := read(root, ".claude/skills/release/SKILL.md")
	f.GotchaCount = strings.Count(s, "⚠")
	f.GotchasBounded = f.GotchaCount <= 1
	if out, ok := run(root, 10*time.Second, "git", "tag", "--list", "stable/*"); ok {
		f.StableTags = strings.Fields(out)
	}
	f.ArtifactsSigned = strings.Contains(wl, "cosign") || strings.Contains(wl, "attest-build-provenance") || strings.Contains(wl, "slsa") || strings.Contains(wl, "sigstore")
	if probeGH {
		artifacts(root, &f)
	}
	return f
}
func staleness(root string, f *Facts) {
	out, ok := run(root, 10*time.Second, "git", "tag", "--merged", "HEAD", "--list", "v*", "--sort=-v:refname")
	if !ok {
		f.StalenessVerdict = "UNKNOWN"
		return
	}
	t := strings.Fields(out)
	if len(t) == 0 {
		f.StalenessVerdict = "NO_TAG"
		return
	}
	f.LatestTag = t[0]
	s, _ := run(root, 10*time.Second, "git", "rev-list", "--count", f.LatestTag+"..HEAD")
	f.CommitsBehind, _ = strconv.Atoi(strings.TrimSpace(s))
	tt, _ := run(root, 10*time.Second, "git", "log", "-1", "--format=%ct", f.LatestTag)
	ht, _ := run(root, 10*time.Second, "git", "log", "-1", "--format=%ct", "HEAD")
	ti, _ := strconv.ParseInt(strings.TrimSpace(tt), 10, 64)
	hi, _ := strconv.ParseInt(strings.TrimSpace(ht), 10, 64)
	f.DaysBehind = math.Round(float64(hi-ti)/8640) / 10
	if f.CommitsBehind >= 100 || f.DaysBehind >= 45 {
		f.StalenessVerdict = "VERY_STALE"
	} else if f.CommitsBehind >= 20 || f.DaysBehind >= 14 {
		f.StalenessVerdict = "STALE"
	} else {
		f.StalenessVerdict = "FRESH"
	}
}
func artifacts(root string, f *Facts) {
	tag, ok := run(root, 20*time.Second, "gh", "release", "list", "--limit", "1", "--json", "tagName", "--jq", ".[0].tagName")
	if !ok || strings.TrimSpace(tag) == "" {
		return
	}
	f.GHReachable = true
	n, ok := run(root, 20*time.Second, "gh", "release", "view", strings.TrimSpace(tag), "--json", "assets", "--jq", ".assets[].name")
	if !ok {
		return
	}
	arc, sum := false, false
	for _, a := range strings.Fields(n) {
		x := strings.ToLower(a)
		f.Arm64Shipped = f.Arm64Shipped || (strings.Contains(x, "arm64") && strings.Contains(x, "linux"))
		arc = arc || strings.HasSuffix(x, ".zip") || strings.HasSuffix(x, ".tar.gz")
		sum = sum || strings.HasSuffix(x, ".sha256") || strings.Contains(a, "SHA256")
	}
	f.ArtifactsPresent = arc && sum
}
func Build(root string, probeGH bool) Payload {
	f := Gather(root, probeGH)
	p := Payload{Schema: Schema, Workspace: root, BandScores: map[string]float64{}, DebtByBand: map[string]int{}, Staleness: Staleness{f.LatestTag, f.CommitsBehind, f.DaysBehind, f.StalenessVerdict}, GotchaCount: f.GotchaCount, StableTags: f.StableTags, Epic: 1354, NextAction: "fix the worst-first HARD defect under epic #1354"}
	got, max := map[string]int{}, map[string]int{}
	worst := -1
	for _, k := range kpis {
		v := k.value(f)
		r := Row{Key: k.key, Band: k.band, Weight: k.weight, Label: k.label, Fix: k.fix}
		max[k.band] += k.weight
		if v == nil {
			r.Unwitnessed = true
			p.SoftSignals++
		} else if *v {
			r.OK = true
			got[k.band] += k.weight
		} else {
			p.ReleaseDebt++
			p.DebtByBand[k.band]++
			if k.weight > worst {
				worst = k.weight
				p.NextAction = k.fix
			}
		}
		p.Rows = append(p.Rows, r)
	}
	tg, tm := 0, 0
	for _, b := range Bands {
		tg += got[b]
		tm += max[b]
		p.BandScores[b] = math.Round(1000*float64(got[b])/float64(max[b])) / 10
	}
	p.Score = math.Round(1000*float64(tg)/float64(tm)) / 10
	p.Grade = grade(p.Score)
	p.OK = p.ReleaseDebt == 0
	if p.OK {
		p.Verdict = "OK"
		p.Finding = "release at agentic speed: no HARD debt"
	} else if p.ReleaseDebt <= 4 {
		p.Verdict = "RELEASE_DEBT"
		p.Finding = fmt.Sprintf("%d HARD release defect(s) — worst-first via #1354", p.ReleaseDebt)
	} else {
		p.Verdict = "RELEASE_DEBT"
		p.Finding = fmt.Sprintf("%d HARD release defects — release is hand-driven, not agentic", p.ReleaseDebt)
	}
	if p.StableTags == nil {
		p.StableTags = []string{}
	}
	return p
}
func grade(v float64) string {
	if v >= 90 {
		return "A"
	}
	if v >= 80 {
		return "B"
	}
	if v >= 70 {
		return "C"
	}
	if v >= 60 {
		return "D"
	}
	return "F"
}
func Render(p Payload) string {
	var b strings.Builder
	fmt.Fprintf(&b, "release-readiness: %.1f/100 (grade %s) · release-debt %d\n", p.Score, p.Grade, p.ReleaseDebt)
	fmt.Fprintf(&b, "@latest %s: %d commits / %.1fd behind HEAD -> %s\n", or(p.Staleness.LatestTag, "(none)"), p.Staleness.CommitsBehind, p.Staleness.DaysBehind, p.Staleness.Verdict)
	for _, x := range Bands {
		fmt.Fprintf(&b, "  %-9s %5.1f/100  (debt %d)\n", x, p.BandScores[x], p.DebtByBand[x])
	}
	for _, r := range p.Rows {
		m := "DEBT"
		if r.OK {
			m = "ok "
		} else if r.Unwitnessed {
			m = "·· "
		}
		fmt.Fprintf(&b, "    [%s] %s\n", m, r.Label)
	}
	fmt.Fprintf(&b, "next: %s", p.NextAction)
	return b.String()
}
func Markdown(p Payload, stamp string) string {
	if stamp == "" {
		stamp = time.Now().Format("2006-01-02")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "---\ntitle: \"fak release-readiness scorecard — release-debt measuring stick\"\n---\n\n# Release-readiness scorecard — can fak release at agentic speed\n\n<!-- release-readiness-scorecard: %s · process: fak release readiness -->\n\n**Release-debt:** **%d** · score %.1f/100 (grade %s) · @latest %s is %s.\n\n", stamp, p.ReleaseDebt, p.Score, p.Grade, or(p.Staleness.LatestTag, "(none)"), p.Staleness.Verdict)
	for _, x := range Bands {
		fmt.Fprintf(&b, "## %s — %.1f/100\n\n| KPI | State | Fix if open |\n|---|---|---|\n", x, p.BandScores[x])
		for _, r := range p.Rows {
			if r.Band != x {
				continue
			}
			state, fix := "✅ met", ""
			if r.Unwitnessed {
				state, fix = "·· unwitnessed", r.Fix
			} else if !r.OK {
				state, fix = "❌ **debt**", r.Fix
			}
			fmt.Fprintf(&b, "| %s | %s | %s |\n", r.Label, state, fix)
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "**Next action:** %s\n", p.NextAction)
	return b.String()
}
func or(v, d string) string {
	if v == "" {
		return d
	}
	return v
}
