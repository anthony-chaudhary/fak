package architest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// Third-party EFFECT register gate (issue #5650, parent #3807).
//
// THE GAP THIS CLOSES. sbom_drift_test.go keeps docs/sbom/fak.spdx.json honest about exactly
// one surface: the Go modules go.mod pins. That is the surface a language SBOM sees, and it
// is a small minority of what this repository actually fetches, executes, wraps, ships, or
// delegates to. Nothing checked that the tree also shells out to `nvidia-smi`, `claude`,
// `docker` and 20-odd other host binaries; runs `actions/checkout@v4` and ten other CI
// actions at a MOVING tag; builds `FROM nvidia/cuda:12.6.2-devel-ubuntu22.04` at a moving
// tag; `pip install`s dos-kernel and `go install`s govulncheck@latest unpinned; and pipes
// https://tailscale.com/install.sh straight into `sh`. A reviewer who reads the SPDX file and
// stops has seen two modules and none of that.
//
// So this gate takes the same move sbom_drift_test.go made for modules and applies it to the
// whole effect surface: a checked-in register (docs/sbom/third-party-effects.json) that CI
// re-derives from the tree on every run. A newly introduced effect surface with no register
// row and no explicit exclusion reds HERE.
//
// WHY THIS LIVES IN architest AND NOT A NEW LEAF. Identical reasoning to sbom_drift_test.go
// and zerodep_claim_test.go, and it still holds: a repo-hygiene contract over checked-in
// artifacts, stdlib-only, off the request path, never registered into the kernel. No new
// package leaf means no architest tier row and the push gate's UNTIERED_LEAF check is
// untouched. A gate whose job is to police the third-party surface must not widen it.
//
// THE THREE FIELDS ARE INDEPENDENT ON PURPOSE. `pin`, `license_record` and `enforcement`
// answer three different questions and are routinely confused into one "is it safe?" bit:
//
//   - pin           — can the bytes change under us without a commit? (immutability)
//   - license_record— is this effect's license captured anywhere? (legal coverage)
//   - enforcement   — is there a MECHANISM that reds when it drifts? (mechanical teeth)
//
// A Go module is immutable (go.sum), license-covered (the SPDX file), AND enforced
// (sbom_drift_test.go). A CI action today is none of the three. Collapsing them would let the
// module surface's good posture launder the rest, which is exactly the false comfort a
// supply-chain artifact must never sell.
//
// SCOPE — WHAT IS AND IS NOT THE CORPUS. The corpus is the BUILD / SHIP / RUN surface, which
// is declared literally in effectCorpusGlobs below and asserted non-empty. Deliberately
// outside it:
//
//   - `_test.go` files. The register describes what SHIPS and what CI runs. Test-only
//     spawns (`exec.Command("unused")`, `ping`, `wsl.exe`) are scaffolding, not effects a
//     downstream consumer inherits.
//   - Dated histories (docs/notes/**, docs/archive/**) and prose generally. A June note that
//     mentions a tool is a record, not an effect.
//   - Non-literal subprocess targets (`exec.Command(bin, ...)`). They cannot be resolved
//     statically, so they are COUNTED as skipped rather than silently dropped — see
//     scanStats.Skipped, which the gate prints on every run.
//   - Assets fetched by an interpreter rather than curl/wget (a python downloader, a
//     huggingface client call). Those are a real residual blind spot, named here rather
//     than papered over, and are the obvious next widening of effectCorpusGlobs.

// ---------------------------------------------------------------------------
// The closed vocabularies
// ---------------------------------------------------------------------------

// effectClass is the closed source-class vocabulary. It is the DISCOVERY MECHANISM that
// found the effect, which is what a fixer needs first: the same nominal dependency reached
// through a container base and through a package runner has different pinning options and a
// different blast radius.
type effectClass string

const (
	// classGoModule is the primary language dependency graph — go.mod `require`.
	classGoModule effectClass = "go_module"
	// classSubprocessTool is a host binary the shipped Go code spawns by literal name.
	classSubprocessTool effectClass = "subprocess_tool"
	// classPackageRunner is a fetch-and-execute at build/CI time (`go install`, `pip install`).
	classPackageRunner effectClass = "package_runner"
	// classCIAction is a GitHub Actions `uses:` reference.
	classCIAction effectClass = "ci_action"
	// classContainerBase is a Dockerfile `FROM` image.
	classContainerBase effectClass = "container_base"
	// classDownloadedAsset is a curl/wget fetch of an https artifact.
	classDownloadedAsset effectClass = "downloaded_asset"
)

var effectClasses = []effectClass{
	classGoModule, classSubprocessTool, classPackageRunner,
	classCIAction, classContainerBase, classDownloadedAsset,
}

func knownClass(c effectClass) bool {
	for _, k := range effectClasses {
		if k == c {
			return true
		}
	}
	return false
}

// pinState is the closed IMMUTABILITY vocabulary. The distinction that matters is
// pinUnresolved: a moving tag is not "accepted", it is UNRESOLVED — the register is required
// to say so out loud, so nobody reads a row as a pin it is not.
type pinState string

const (
	// pinImmutable — the reference cannot change bytes without a commit here (a content
	// digest, or a version backed by a checked-in lock such as go.sum).
	pinImmutable pinState = "immutable"
	// pinUnresolved — a moving tag / `@latest` / bare package name. Resolvable to a digest,
	// but not pinned by this tree, so upstream can change the bytes under us silently.
	pinUnresolved pinState = "unresolved"
	// pinHostProvided — this repo cannot pin it at all: the host supplies the binary. Naming
	// it is the only control available, which is precisely why it belongs in the register.
	pinHostProvided pinState = "host_provided"
)

// effectStatus separates effects the tree USES now from prospective / documentation-only
// references. A prospective row is not expected to be observed; a used row is.
type effectStatus string

const (
	statusUsed        effectStatus = "used"
	statusProspective effectStatus = "prospective"
)

// ---------------------------------------------------------------------------
// The discovered effect
// ---------------------------------------------------------------------------

// effect is one distinct third-party effect, deduplicated by (class, reference) across every
// call site. Domains, not exact file paths, are the recorded consumer evidence: a 241st
// `exec.Command("git", ...)` call site is not a supply-chain event and must not red the
// trunk, but `git` first appearing in a NEW domain is a real widening.
type effect struct {
	Class     effectClass
	Reference string
	Domains   []string // sorted, deduped
}

func (e effect) key() string { return string(e.Class) + "\x00" + e.Reference }

// skipRecord is one thing the scanner SAW but could not resolve. These are counted and
// printed, never dropped: a discovery pass that silently ignores what it cannot parse is how
// this class of gate rots into a no-op.
type skipRecord struct {
	Path   string
	Class  effectClass
	Reason string
	Text   string
}

type scanStats struct {
	FilesScanned int
	ByClass      map[effectClass]int
	ByDomain     map[string]int
	Skipped      []skipRecord
}

// domainOf buckets a repo-relative path into the coarse consumer domain the register records.
func domainOf(path string) string {
	path = filepath.ToSlash(path)
	if !strings.Contains(path, "/") {
		return path // go.mod, Makefile, Dockerfile, Dockerfile.cuda
	}
	if strings.HasPrefix(path, ".github/workflows/") {
		return ".github/workflows"
	}
	return path[:strings.Index(path, "/")]
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

var (
	reShellVar  = regexp.MustCompile(`\\?\$\{[^}]*\}|\\?\$[A-Za-z_][A-Za-z0-9_]*`)
	reActionSHA = regexp.MustCompile(`^[0-9a-f]{40}$`)
	reSemver    = regexp.MustCompile(`^v\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$`)
)

// normalizeVars collapses shell/ARG interpolation to a stable `${}` placeholder so a
// version-carrying variable does not make the reference unstable across environments — and,
// just as importantly, so no environment-specific value can reach the public register.
func normalizeVars(s string) string { return reShellVar.ReplaceAllString(s, "${}") }

// ---------------------------------------------------------------------------
// Discovery — one function per mechanism, all pure (bytes in, effects out)
// ---------------------------------------------------------------------------

type collector struct {
	byKey   map[string]*effect
	domains map[string]map[string]bool
	stats   scanStats
}

func newCollector() *collector {
	return &collector{
		byKey:   map[string]*effect{},
		domains: map[string]map[string]bool{},
		stats:   scanStats{ByClass: map[effectClass]int{}, ByDomain: map[string]int{}},
	}
}

func (c *collector) add(class effectClass, ref, path string) {
	if ref == "" {
		return
	}
	e := effect{Class: class, Reference: ref}
	k := e.key()
	if _, ok := c.byKey[k]; !ok {
		c.byKey[k] = &e
		c.domains[k] = map[string]bool{}
	}
	c.domains[k][domainOf(path)] = true
}

func (c *collector) skip(path string, class effectClass, reason, text string) {
	if len(text) > 120 {
		text = text[:120] + "…"
	}
	c.stats.Skipped = append(c.stats.Skipped, skipRecord{Path: path, Class: class, Reason: reason, Text: text})
}

func (c *collector) result() ([]effect, scanStats) {
	out := make([]effect, 0, len(c.byKey))
	for k, e := range c.byKey {
		doms := make([]string, 0, len(c.domains[k]))
		for d := range c.domains[k] {
			doms = append(doms, d)
		}
		sort.Strings(doms)
		e.Domains = doms
		out = append(out, *e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return out[i].Reference < out[j].Reference
	})
	for _, e := range out {
		c.stats.ByClass[e.Class]++
		for _, d := range e.Domains {
			c.stats.ByDomain[d]++
		}
	}
	return out, c.stats
}

// --- go.mod -----------------------------------------------------------------

func (c *collector) scanGoMod(path, text string) {
	facts, err := parseGoMod([]byte(text))
	if err != nil {
		c.skip(path, classGoModule, "go.mod did not parse", err.Error())
		return
	}
	for _, r := range facts.Requires {
		c.add(classGoModule, r.Path+"@"+r.Version, path)
	}
}

// --- GitHub Actions ---------------------------------------------------------

var reUses = regexp.MustCompile(`(?m)^\s*(?:-\s*)?uses:\s*["']?([^\s"'#]+)`)

func (c *collector) scanWorkflow(path, text string) {
	for _, m := range reUses.FindAllStringSubmatch(text, -1) {
		ref := m[1]
		switch {
		case strings.HasPrefix(ref, "./"), strings.HasPrefix(ref, "../"):
			// A local composite action in this repo is first-party, not a third-party effect.
			continue
		case !strings.Contains(ref, "@"):
			c.skip(path, classCIAction, "uses: with no @ref", ref)
			continue
		}
		c.add(classCIAction, ref, path)
	}
}

// --- Package runners --------------------------------------------------------

var (
	reGoInstall  = regexp.MustCompile(`go install\s+([^\s"'#;&|]+)`)
	rePipInstall = regexp.MustCompile(`pip install\s+([^\n\r#;&|]+)`)
)

// pipPackage picks the installed distribution out of a `pip install` argument list. Flags,
// the `-r requirements.txt` form and index-url values are not packages; an interpolated
// argument cannot be resolved. Every one of those is recorded as a skip, not dropped.
func pipPackage(args string) (pkg, skipReason string) {
	fields := strings.Fields(args)
	for i := 0; i < len(fields); i++ {
		// Shell/PowerShell quoting is presentation, not identity: `"huggingface_hub[hf_xet]"`
		// and `huggingface_hub[hf_xet]` are the same distribution and must dedupe to one row.
		f := strings.Trim(fields[i], `"'`)
		switch {
		case f == "-r", f == "--requirement":
			return "", "pip install -r <file>: the package set lives in a requirements file"
		case f == "--index-url", f == "--extra-index-url", f == "-i", f == "-f", f == "--find-links":
			i++ // consume the value
		case strings.HasPrefix(f, "-"):
			// a bare flag
		case strings.ContainsAny(f, "$%"):
			return "", "interpolated package argument"
		default:
			return f, ""
		}
	}
	return "", "no package argument found"
}

func (c *collector) scanPackageRunners(path, text string) {
	for _, m := range reGoInstall.FindAllStringSubmatch(text, -1) {
		target := m[1]
		// `go install ...@latest` and `go install …@version` appear in prose and help text as
		// placeholders, not as real fetches.
		if !strings.Contains(target, "@") || !strings.Contains(target, "/") ||
			strings.Contains(target, "...") || strings.Contains(target, "…") {
			c.skip(path, classPackageRunner, "go install target is a placeholder, not a module path", target)
			continue
		}
		if strings.HasPrefix(target, "github.com/anthony-chaudhary/fak") {
			continue // first-party self-install
		}
		c.add(classPackageRunner, "go:"+normalizeVars(target), path)
	}
	for _, m := range rePipInstall.FindAllStringSubmatch(text, -1) {
		pkg, reason := pipPackage(m[1])
		if pkg == "" {
			c.skip(path, classPackageRunner, reason, strings.TrimSpace(m[1]))
			continue
		}
		c.add(classPackageRunner, "pip:"+pkg, path)
	}
}

// --- Container bases --------------------------------------------------------

var reFrom = regexp.MustCompile(`(?mi)^\s*FROM\s+(.+)$`)

func (c *collector) scanDockerfile(path, text string) {
	stages := map[string]bool{}
	for _, m := range reFrom.FindAllStringSubmatch(text, -1) {
		fields := strings.Fields(m[1])
		image := ""
		for i := 0; i < len(fields); i++ {
			if strings.HasPrefix(fields[i], "--") {
				continue
			}
			image = fields[i]
			// `FROM <image> AS <stage>` — remember the alias so a later `FROM <stage>` is not
			// misread as a third-party image.
			if i+2 < len(fields) && strings.EqualFold(fields[i+1], "AS") {
				stages[strings.ToLower(fields[i+2])] = true
			}
			break
		}
		if image == "" || stages[strings.ToLower(image)] {
			continue
		}
		c.add(classContainerBase, normalizeVars(image), path)
	}
}

// --- Downloaded assets ------------------------------------------------------

var reFetchURL = regexp.MustCompile(`\b(?:curl|wget)\b[^\n\r]*?(https://[^\s"'\\)]+)`)

func (c *collector) scanFetches(path, text string) {
	for _, line := range strings.Split(text, "\n") {
		// A heredoc escapes interpolation as `\${repo}`. Unescape before extracting, or the URL
		// silently truncates at the backslash and the register records a prefix that fetches
		// nothing — a row that looks covered while naming the wrong artifact.
		line = strings.ReplaceAll(line, `\$`, `$`)
		for _, m := range reFetchURL.FindAllStringSubmatch(line, -1) {
			url := strings.TrimRight(normalizeVars(m[1]), ".,;")
			c.add(classDownloadedAsset, url, path)
		}
	}
}

// --- Subprocess tools -------------------------------------------------------

var (
	reExecAny     = regexp.MustCompile(`exec\.Command(?:Context)?\(`)
	reExecLiteral = regexp.MustCompile(`exec\.Command(?:Context)?\(\s*(?:ctx\s*,\s*)?"([a-zA-Z0-9_.+-]+)"`)
)

func (c *collector) scanGoSource(path, text string) {
	total := len(reExecAny.FindAllString(text, -1))
	lits := reExecLiteral.FindAllStringSubmatch(text, -1)
	for _, m := range lits {
		c.add(classSubprocessTool, m[1], path)
	}
	if n := total - len(lits); n > 0 {
		c.skip(path, classSubprocessTool, "subprocess target is not a string literal", fmt.Sprintf("%d call site(s)", n))
	}
}

// ---------------------------------------------------------------------------
// The corpus + the top-level discovery entry point
// ---------------------------------------------------------------------------

// effectCorpusGlobs declares the build/ship/run surface this gate reads, as repo-relative
// slash paths. It is the SCOPE statement in executable form: widening the gate means adding
// a line here, and the gate fails closed if any bucket matches nothing.
var effectCorpusGlobs = []string{
	"go.mod",
	"Makefile",
	"Dockerfile*",
	".github/workflows/*.yml",
	"scripts/*.sh",
	"scripts/*.ps1",
	"cmd/**/*.go",
	"internal/**/*.go",
}

// discoverEffects is the generator: a tree (repo-relative slash path -> contents) in, the
// deduplicated effect set and the scan denominators out. Pure — no filesystem, no network —
// so the synthetic-repository proof below runs on an in-memory tree and the current-tree
// gate runs on the real one through exactly the same code.
func discoverEffects(tree map[string]string) ([]effect, scanStats, error) {
	c := newCollector()
	paths := make([]string, 0, len(tree))
	for p := range tree {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	for _, p := range paths {
		text := tree[p]
		c.stats.FilesScanned++
		base := filepath.Base(p)
		switch {
		case p == "go.mod":
			c.scanGoMod(p, text)
		case strings.HasPrefix(p, ".github/workflows/"):
			c.scanWorkflow(p, text)
			c.scanPackageRunners(p, text)
			c.scanFetches(p, text)
		case strings.HasPrefix(base, "Dockerfile"):
			c.scanDockerfile(p, text)
			c.scanPackageRunners(p, text)
			c.scanFetches(p, text)
		case base == "Makefile", strings.HasPrefix(p, "scripts/"):
			c.scanPackageRunners(p, text)
			c.scanFetches(p, text)
		case strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go"):
			c.scanGoSource(p, text)
		}
	}

	effects, stats := c.result()
	if len(effects) == 0 {
		return nil, stats, fmt.Errorf("discovered zero third-party effects across %d files — a repository "+
			"that fetches, executes and ships nothing is not what this tree is, so the extractors matched "+
			"nothing and this gate would be silently inert", stats.FilesScanned)
	}
	return effects, stats, nil
}

// ---------------------------------------------------------------------------
// Pin classification
// ---------------------------------------------------------------------------

// computePin derives immutability from the reference itself. This is the half of the register
// a human must not be trusted to fill in: it is exactly the field where "we'll pin it later"
// quietly becomes "we said it was pinned".
func computePin(e effect) pinState {
	switch e.Class {
	case classGoModule:
		// go.sum carries a content hash for every required module version.
		return pinImmutable
	case classSubprocessTool:
		return pinHostProvided
	case classCIAction:
		if i := strings.LastIndex(e.Reference, "@"); i >= 0 && reActionSHA.MatchString(e.Reference[i+1:]) {
			return pinImmutable
		}
		return pinUnresolved
	case classContainerBase:
		if strings.Contains(e.Reference, "@sha256:") {
			return pinImmutable
		}
		return pinUnresolved
	case classPackageRunner:
		if strings.HasPrefix(e.Reference, "go:") {
			if i := strings.LastIndex(e.Reference, "@"); i >= 0 && reSemver.MatchString(e.Reference[i+1:]) {
				return pinImmutable
			}
		}
		if strings.HasPrefix(e.Reference, "pip:") && strings.Contains(e.Reference, "==") {
			return pinImmutable
		}
		return pinUnresolved
	}
	return pinUnresolved // downloaded_asset: only a recorded digest can upgrade this
}

// ---------------------------------------------------------------------------
// The register artifact
// ---------------------------------------------------------------------------

const effectRegisterSchema = "fak-third-party-effects/1"

// registerRow keeps license coverage, pin immutability and mechanical enforcement as three
// independent fields — see the package-level note on why collapsing them is the defect.
type registerRow struct {
	Class         effectClass  `json:"class"`
	Reference     string       `json:"reference"`
	Domains       []string     `json:"consumer_domains"`
	Status        effectStatus `json:"status"`
	Pin           pinState     `json:"pin"`
	Digest        string       `json:"digest"`
	LicenseRecord string       `json:"license_record"`
	Enforcement   string       `json:"enforcement"`
	Note          string       `json:"note,omitempty"`
}

type registerExclusion struct {
	Class     effectClass `json:"class"`
	Reference string      `json:"reference"`
	Reason    string      `json:"reason"`
}

type effectRegister struct {
	Schema     string              `json:"schema"`
	Regenerate string              `json:"regenerate"`
	Rows       []registerRow       `json:"rows"`
	Exclusions []registerExclusion `json:"exclusions"`
}

func parseRegister(data []byte) (effectRegister, error) {
	var r effectRegister
	if err := json.Unmarshal(data, &r); err != nil {
		return r, fmt.Errorf("not valid JSON: %w", err)
	}
	if r.Schema != effectRegisterSchema {
		return r, fmt.Errorf("schema is %q, want %q — the register and this gate must agree on one "+
			"versioned schema", r.Schema, effectRegisterSchema)
	}
	if len(r.Rows) == 0 {
		return r, fmt.Errorf("zero rows — a register that lists nothing witnesses nothing")
	}
	return r, nil
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

type effectDriftKind string

const (
	// effectMissing is the BLIND SPOT this gate exists for: the tree pulls in a third-party
	// effect that the register does not mention and no exclusion covers.
	effectMissing effectDriftKind = "EFFECT_MISSING_FROM_REGISTER"
	// effectUnobserved is the opposite, weaker direction: a `used` row nothing produces.
	effectUnobserved effectDriftKind = "REGISTER_ROW_UNOBSERVED"
	// effectPinMisdeclared is the dangerous one: the register calls a moving reference pinned.
	effectPinMisdeclared effectDriftKind = "REGISTER_PIN_MISDECLARED"
	// effectStatusStale is a `prospective` row the tree actually uses now.
	effectStatusStale effectDriftKind = "REGISTER_STATUS_STALE"
	// effectDomainDrift is a used row whose consumer domains no longer match the tree.
	effectDomainDrift effectDriftKind = "REGISTER_DOMAINS_DISAGREE"
	// effectBadRow / effectDuplicate / effectDeadExclusion are structural register defects.
	effectBadRow        effectDriftKind = "REGISTER_ROW_MALFORMED"
	effectDuplicate     effectDriftKind = "REGISTER_DUPLICATE_ROW"
	effectDeadExclusion effectDriftKind = "REGISTER_EXCLUSION_UNOBSERVED"
)

type effectFinding struct {
	Kind      effectDriftKind
	Reference string
	Msg       string
}

type effectDriftID struct {
	Kind      effectDriftKind
	Reference string
}

func sameDomains(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	x, y := append([]string(nil), a...), append([]string(nil), b...)
	sort.Strings(x)
	sort.Strings(y)
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

// registerDrift compares the discovered effect set against the committed register. Pure, so
// the mutation table below can prove it RED — the only way a green run means agreement rather
// than a checker that found nothing to check.
func registerDrift(effects []effect, reg effectRegister) []effectFinding {
	var findings []effectFinding

	rows := map[string]registerRow{}
	for _, row := range reg.Rows {
		id := string(row.Class) + "\x00" + row.Reference
		if !knownClass(row.Class) {
			findings = append(findings, effectFinding{effectBadRow, row.Reference, fmt.Sprintf(
				"register row %q declares class %q, which is not in the closed source-class vocabulary %v. "+
					"A row outside the vocabulary is invisible to every per-class rule, including the pin check.",
				row.Reference, row.Class, effectClasses)})
			continue
		}
		if row.Status != statusUsed && row.Status != statusProspective {
			findings = append(findings, effectFinding{effectBadRow, row.Reference, fmt.Sprintf(
				"register row %q has status %q; want %q or %q. The status is what separates an effect the "+
					"tree uses today from a prospective or documentation-only reference.",
				row.Reference, row.Status, statusUsed, statusProspective)})
			continue
		}
		if row.LicenseRecord == "" || row.Enforcement == "" {
			findings = append(findings, effectFinding{effectBadRow, row.Reference, fmt.Sprintf(
				"register row %q leaves license_record or enforcement empty. These are independent fields "+
					"from `pin` and each must state its own answer (use \"none\" to record an honest gap).",
				row.Reference)})
			continue
		}
		if _, dup := rows[id]; dup {
			findings = append(findings, effectFinding{effectDuplicate, row.Reference, fmt.Sprintf(
				"register lists %s %q twice — a duplicated row makes the document self-contradicting about "+
					"that effect's pin/license/enforcement posture.", row.Class, row.Reference)})
			continue
		}
		rows[id] = row
	}

	excluded := map[string]bool{}
	for _, ex := range reg.Exclusions {
		excluded[string(ex.Class)+"\x00"+ex.Reference] = true
	}

	seen := map[string]bool{}
	for _, e := range effects {
		id := e.key()
		seen[id] = true
		if excluded[id] {
			continue
		}
		row, ok := rows[id]
		if !ok {
			findings = append(findings, effectFinding{effectMissing, e.Reference, fmt.Sprintf(
				"the tree pulls in %s %q (consumer domains: %s) and docs/sbom/third-party-effects.json has "+
					"neither a row nor an exclusion for it. This is the BLIND-SPOT direction: a third-party "+
					"effect a reviewer reading the register never learns about. Add a row (pin=%s per the "+
					"reference itself) or an explicit exclusion with a reason.",
				e.Class, e.Reference, strings.Join(e.Domains, ", "), computePin(e))})
			continue
		}
		if row.Status == statusProspective {
			findings = append(findings, effectFinding{effectStatusStale, e.Reference, fmt.Sprintf(
				"register marks %s %q as %q, but the tree uses it now (domains: %s). A prospective row that "+
					"has become real understates the live surface; set status to %q.",
				e.Class, e.Reference, statusProspective, strings.Join(e.Domains, ", "), statusUsed)})
		}
		if want := computePin(e); row.Pin != want {
			extra := ""
			if want == pinUnresolved && row.Pin == pinImmutable {
				extra = " Calling a moving reference immutable is the failure this field exists to prevent: " +
					"upstream can change these bytes with no commit here."
			}
			findings = append(findings, effectFinding{effectPinMisdeclared, e.Reference, fmt.Sprintf(
				"register records %s %q with pin %q, but the reference itself is %q.%s Either pin the "+
					"reference (a digest / an exact version) or record it honestly as %q.",
				e.Class, e.Reference, row.Pin, want, extra, want)})
		}
		if row.Digest != "" && row.Pin != pinImmutable {
			findings = append(findings, effectFinding{effectBadRow, e.Reference, fmt.Sprintf(
				"register gives %s %q a digest but a pin of %q — a recorded digest IS the immutability, so "+
					"these two fields contradict each other.", e.Class, e.Reference, row.Pin)})
		}
		if !sameDomains(row.Domains, e.Domains) {
			findings = append(findings, effectFinding{effectDomainDrift, e.Reference, fmt.Sprintf(
				"register records %s %q as consumed by %v, but the tree consumes it from %v. A new consumer "+
					"domain is a real widening of where this effect reaches; update consumer_domains.",
				e.Class, e.Reference, row.Domains, e.Domains)})
		}
	}

	for id, row := range rows {
		if row.Status == statusProspective || seen[id] {
			continue
		}
		findings = append(findings, effectFinding{effectUnobserved, row.Reference, fmt.Sprintf(
			"register lists %s %q as %q, but nothing in the corpus produces it. This is the weaker "+
				"direction — no unlisted effect ships — but a stale register still makes a false claim. "+
				"Drop the row, or set status to %q if it is a forward-looking reference.",
			row.Class, row.Reference, statusUsed, statusProspective)})
	}

	for _, ex := range reg.Exclusions {
		if !seen[string(ex.Class)+"\x00"+ex.Reference] {
			findings = append(findings, effectFinding{effectDeadExclusion, ex.Reference, fmt.Sprintf(
				"register excludes %s %q, but the corpus no longer produces it. A dead exclusion is a "+
					"standing permission nobody re-examined; drop it.", ex.Class, ex.Reference)})
		}
	}

	sort.Slice(findings, func(i, j int) bool {
		if findings[i].Kind != findings[j].Kind {
			return findings[i].Kind < findings[j].Kind
		}
		return findings[i].Reference < findings[j].Reference
	})
	return findings
}

// ---------------------------------------------------------------------------
// Loading the real tree
// ---------------------------------------------------------------------------

func repoRoot(t *testing.T) string {
	t.Helper()
	return filepath.Dir(internalDir(t))
}

// loadEffectCorpus materializes effectCorpusGlobs from disk into the in-memory tree shape
// discoverEffects consumes. It fails closed on any bucket that matched nothing, because a
// silently empty corpus bucket is a gate that stops checking a whole mechanism.
func loadEffectCorpus(root string) (map[string]string, error) {
	tree := map[string]string{}
	for _, g := range effectCorpusGlobs {
		var matches []string
		if strings.Contains(g, "**") {
			prefix := strings.SplitN(g, "/**", 2)[0]
			err := filepath.WalkDir(filepath.Join(root, prefix), func(p string, d os.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if d.IsDir() {
					return nil
				}
				if strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go") {
					matches = append(matches, p)
				}
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("walk %s: %w", g, err)
			}
		} else {
			m, err := filepath.Glob(filepath.Join(root, filepath.FromSlash(g)))
			if err != nil {
				return nil, fmt.Errorf("glob %s: %w", g, err)
			}
			matches = m
		}
		if len(matches) == 0 {
			return nil, fmt.Errorf("corpus bucket %q matched no files — this gate would stop checking a "+
				"whole discovery mechanism without saying so; the tree layout moved, so move "+
				"effectCorpusGlobs with it", g)
		}
		for _, m := range matches {
			data, err := os.ReadFile(m)
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", m, err)
			}
			rel, err := filepath.Rel(root, m)
			if err != nil {
				return nil, err
			}
			tree[filepath.ToSlash(rel)] = string(data)
		}
	}
	return tree, nil
}

func registerPath(root string) string {
	return filepath.Join(root, "docs", "sbom", "third-party-effects.json")
}

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

// TestThirdPartyEffectsRegisterCoversTree is the gate itself: every third-party effect the
// corpus produces must have a register row or an explicit exclusion, with an honestly
// declared pin. Introducing a new CI action, container base, package runner, downloaded
// asset, module or spawned host binary without recording it reds here.
func TestThirdPartyEffectsRegisterCoversTree(t *testing.T) {
	root := repoRoot(t)
	tree, err := loadEffectCorpus(root)
	if err != nil {
		t.Fatalf("load effect corpus: %v", err)
	}
	effects, stats, err := discoverEffects(tree)
	if err != nil {
		t.Fatalf("effect discovery could not run: %v", err)
	}
	data, err := os.ReadFile(registerPath(root))
	if err != nil {
		t.Fatalf("read %s: %v — the committed register is the artifact this gate keeps honest; "+
			"if it moved, move this gate with it", registerPath(root), err)
	}
	reg, err := parseRegister(data)
	if err != nil {
		t.Fatalf("parse register: %v", err)
	}

	for _, f := range registerDrift(effects, reg) {
		t.Errorf("[%s] %s", f.Kind, f.Msg)
	}

	byClass := make([]string, 0, len(effectClasses))
	for _, c := range effectClasses {
		byClass = append(byClass, fmt.Sprintf("%s=%d", c, stats.ByClass[c]))
	}
	domains := make([]string, 0, len(stats.ByDomain))
	for d := range stats.ByDomain {
		domains = append(domains, d)
	}
	sort.Strings(domains)
	domCounts := make([]string, 0, len(domains))
	for _, d := range domains {
		domCounts = append(domCounts, fmt.Sprintf("%s=%d", d, stats.ByDomain[d]))
	}
	t.Logf("third-party effects: %d distinct across %d corpus files; by class: %s",
		len(effects), stats.FilesScanned, strings.Join(byClass, " "))
	t.Logf("by consumer domain: %s", strings.Join(domCounts, " "))
	t.Logf("skipped/unparsed references: %d (see -v for the list)", len(stats.Skipped))
	for _, s := range stats.Skipped {
		t.Logf("  skipped %s in %s: %s (%s)", s.Class, s.Path, s.Reason, s.Text)
	}
}

// TestEffectDiscoveryFindsEachMechanism is the known-positive fixture set: a SYNTHETIC
// repository carrying one of each discovery mechanism — a module dependency, a subprocess
// tool, a package runner, a CI action, a container base and a downloaded asset. It is the
// issue's proof requirement in executable form, and it is what stops a future refactor from
// silently switching a whole mechanism off (the real-tree gate would still pass, because a
// mechanism that discovers nothing produces no missing-row findings).
func syntheticRepo() map[string]string {
	return map[string]string{
		"go.mod": "module example.com/synth\n\ngo 1.26\n\nrequire example.org/dep v1.2.3\n",
		".github/workflows/ci.yml": "jobs:\n  build:\n    steps:\n      - uses: actions/checkout@v4\n" +
			"      - run: go install example.org/tool/cmd/tool@latest\n",
		"Dockerfile":          "FROM golang:1.26 AS build\nRUN curl -fsSL https://example.org/asset.tgz -o /tmp/a.tgz\n",
		"internal/x/spawn.go": "package x\n\nfunc f() { _ = exec.Command(\"nvidia-smi\", \"-L\") }\n",
	}
}

func TestEffectDiscoveryFindsEachMechanism(t *testing.T) {
	effects, stats, err := discoverEffects(syntheticRepo())
	if err != nil {
		t.Fatalf("discovery over the synthetic repository failed: %v", err)
	}
	got := map[string]bool{}
	for _, e := range effects {
		got[e.key()] = true
	}
	want := []effect{
		{Class: classGoModule, Reference: "example.org/dep@v1.2.3"},
		{Class: classSubprocessTool, Reference: "nvidia-smi"},
		{Class: classPackageRunner, Reference: "go:example.org/tool/cmd/tool@latest"},
		{Class: classCIAction, Reference: "actions/checkout@v4"},
		{Class: classContainerBase, Reference: "golang:1.26"},
		{Class: classDownloadedAsset, Reference: "https://example.org/asset.tgz"},
	}
	for _, w := range want {
		if !got[w.key()] {
			t.Errorf("the %s mechanism discovered nothing for %q — a discovery mechanism that finds "+
				"nothing makes the real-tree gate pass by looking at an empty set, which is the silently "+
				"inert failure this fixture exists to catch. Discovered: %v",
				w.Class, w.Reference, effects)
		}
	}
	for _, c := range effectClasses {
		if stats.ByClass[c] == 0 {
			t.Errorf("class %q has a zero denominator over the synthetic repository", c)
		}
	}
}

// TestEffectRegisterMutationsAreCaught is the witness that the gate can actually FAIL, and it
// is the issue's central proof: REMOVING EACH REGISTER ROW INDEPENDENTLY must fail the check.
// It runs against the REAL committed register rather than a frozen inline fixture, so the
// fixtures cannot drift away from the artifact they model.
func TestEffectRegisterMutationsAreCaught(t *testing.T) {
	root := repoRoot(t)
	tree, err := loadEffectCorpus(root)
	if err != nil {
		t.Fatalf("load effect corpus: %v", err)
	}
	effects, _, err := discoverEffects(tree)
	if err != nil {
		t.Fatalf("discovery: %v", err)
	}
	data, err := os.ReadFile(registerPath(root))
	if err != nil {
		t.Fatalf("read register: %v", err)
	}
	reg, err := parseRegister(data)
	if err != nil {
		t.Fatalf("parse register: %v", err)
	}
	if base := registerDrift(effects, reg); len(base) != 0 {
		t.Fatalf("the committed register already disagrees with the tree, so a per-row mutation proves "+
			"nothing; fix the baseline first:\n%s", renderEffectFindings(base))
	}

	perClass := map[effectClass]int{}
	for i, row := range reg.Rows {
		if row.Status != statusUsed {
			continue
		}
		row := row
		t.Run(fmt.Sprintf("%s/%s", row.Class, row.Reference), func(t *testing.T) {
			mut := reg
			mut.Rows = append(append([]registerRow(nil), reg.Rows[:i]...), reg.Rows[i+1:]...)
			findings := registerDrift(effects, mut)
			want := effectDriftID{effectMissing, row.Reference}
			found := false
			for _, f := range findings {
				if (effectDriftID{f.Kind, f.Reference}) == want {
					found = true
					if !strings.Contains(f.Msg, row.Reference) {
						t.Errorf("the failure message does not name %q, so the fix is not obvious from "+
							"the failure alone: %s", row.Reference, f.Msg)
					}
				}
			}
			if !found {
				t.Fatalf("removing the register row for %s %q did NOT produce %s — that row is "+
					"unwitnessed: it could be deleted and nothing would red.\nfindings:\n%s",
					row.Class, row.Reference, effectMissing, renderEffectFindings(findings))
			}
		})
		perClass[row.Class]++
	}
	for _, c := range effectClasses {
		if perClass[c] == 0 {
			t.Errorf("no `used` register row of class %q was mutated — this class is unproven by the "+
				"mutation table, so a row of that class could be silently unenforced", c)
		}
	}
}

// TestEffectRegisterRejectsLaunderedPins pins the "mutable references are visible as
// unresolved rather than silently accepted" done-condition: relabelling a moving reference as
// immutable must red, per class that can move.
func TestEffectRegisterRejectsLaunderedPins(t *testing.T) {
	root := repoRoot(t)
	tree, err := loadEffectCorpus(root)
	if err != nil {
		t.Fatalf("load effect corpus: %v", err)
	}
	effects, _, err := discoverEffects(tree)
	if err != nil {
		t.Fatalf("discovery: %v", err)
	}
	data, err := os.ReadFile(registerPath(root))
	if err != nil {
		t.Fatalf("read register: %v", err)
	}
	reg, err := parseRegister(data)
	if err != nil {
		t.Fatalf("parse register: %v", err)
	}

	movable := []effectClass{classCIAction, classContainerBase, classPackageRunner, classDownloadedAsset}
	for _, class := range movable {
		idx := -1
		for i, row := range reg.Rows {
			if row.Class == class && row.Status == statusUsed && row.Pin == pinUnresolved {
				idx = i
				break
			}
		}
		if idx < 0 {
			t.Errorf("no `used` %s row is recorded as %q — either every reference of this class is now "+
				"genuinely pinned (good, and this case should be retired) or the register is laundering "+
				"them; check before deleting this assertion", class, pinUnresolved)
			continue
		}
		mut := reg
		mut.Rows = append([]registerRow(nil), reg.Rows...)
		mut.Rows[idx].Pin = pinImmutable
		findings := registerDrift(effects, mut)
		want := effectDriftID{effectPinMisdeclared, reg.Rows[idx].Reference}
		found := false
		for _, f := range findings {
			if (effectDriftID{f.Kind, f.Reference}) == want {
				found = true
			}
		}
		if !found {
			t.Errorf("relabelling the moving %s reference %q as %q did NOT red — a mutable reference "+
				"would then be silently accepted as a pin.\nfindings:\n%s",
				class, reg.Rows[idx].Reference, pinImmutable, renderEffectFindings(findings))
		}
	}
}

// TestEffectRegisterCarriesNoSensitiveValues pins the public-output done-condition. The
// register is a reviewer-facing artifact; a credential, an absolute local path or a
// host-specific value in it is both a leak and a claim that cannot reproduce elsewhere.
func TestEffectRegisterCarriesNoSensitiveValues(t *testing.T) {
	data, err := os.ReadFile(registerPath(repoRoot(t)))
	if err != nil {
		t.Fatalf("read register: %v", err)
	}
	text := string(data)
	banned := []struct{ pat, why string }{
		{`(?i)\b(ghp|gho|ghs|ghu)_[A-Za-z0-9]{16,}`, "a GitHub token"},
		{`(?i)\bsk-[A-Za-z0-9_-]{16,}`, "an API secret key"},
		{`\bAKIA[0-9A-Z]{12,}`, "an AWS access key id"},
		{`(?i)\bBearer\s+[A-Za-z0-9._-]{12,}`, "a bearer credential"},
		{`(?i)"[^"]*(api[_-]?key|secret|password|token)"\s*:\s*"[^"]+"`, "a credential-shaped field"},
		{`[A-Za-z]:\\\\`, "a Windows absolute path (host-specific)"},
		{`/(?:home|Users)/[A-Za-z0-9._-]+`, "a home-directory path (host-specific)"},
		{`(?m)^\s*"[^"]*":\s*"/(?:opt|var|tmp|usr)/`, "an absolute host path"},
	}
	for _, b := range banned {
		re := regexp.MustCompile(b.pat)
		if m := re.FindString(text); m != "" {
			t.Errorf("docs/sbom/third-party-effects.json contains %s (%q). The register is public "+
				"reviewer-facing output: it must carry no credentials, no local paths and no "+
				"environment-specific values. Use the ${} placeholder the generator emits for "+
				"interpolated values.", b.why, m)
		}
	}
}

// TestEffectRegisterBindsEveryFindingKind binds each declared drift kind to a case that
// actually emits it. A vocabulary constant with no emitter is the declared-but-unwired defect
// class: the register would appear to police a condition nothing can ever report. The
// current-tree gate cannot cover these, because a healthy register produces none of them.
func TestEffectRegisterBindsEveryFindingKind(t *testing.T) {
	ok := registerRow{
		Class: classCIAction, Reference: "owner/act@v1", Domains: []string{".github/workflows"},
		Status: statusUsed, Pin: pinUnresolved, LicenseRecord: "none", Enforcement: "none",
	}
	observed := []effect{{Class: classCIAction, Reference: "owner/act@v1", Domains: []string{".github/workflows"}}}
	reg := func(rows []registerRow, ex ...registerExclusion) effectRegister {
		return effectRegister{Schema: effectRegisterSchema, Rows: rows, Exclusions: ex}
	}
	withRow := func(mut func(*registerRow)) []registerRow {
		r := ok
		r.Domains = append([]string(nil), ok.Domains...)
		mut(&r)
		return []registerRow{r}
	}

	cases := []struct {
		name string
		reg  effectRegister
		eff  []effect
		want effectDriftKind
	}{
		{"unregistered effect", reg([]registerRow{ok}), append(observed,
			effect{Class: classCIAction, Reference: "owner/new@v2", Domains: []string{".github/workflows"}}),
			effectMissing},
		{"used row nothing produces", reg([]registerRow{ok}),
			[]effect{{Class: classCIAction, Reference: "owner/other@v1", Domains: []string{".github/workflows"}}},
			effectUnobserved},
		{"moving reference relabelled immutable",
			reg(withRow(func(r *registerRow) { r.Pin = pinImmutable })), observed, effectPinMisdeclared},
		{"prospective row the tree now uses",
			reg(withRow(func(r *registerRow) { r.Status = statusProspective })), observed, effectStatusStale},
		{"consumer domains drifted",
			reg(withRow(func(r *registerRow) { r.Domains = []string{"scripts"} })), observed, effectDomainDrift},
		{"class outside the vocabulary",
			reg(withRow(func(r *registerRow) { r.Class = "wishful_thinking" })), observed, effectBadRow},
		{"license_record left empty",
			reg(withRow(func(r *registerRow) { r.LicenseRecord = "" })), observed, effectBadRow},
		{"digest recorded but pin not immutable",
			reg(withRow(func(r *registerRow) { r.Digest = "sha256:dead" })), observed, effectBadRow},
		{"same effect registered twice", reg([]registerRow{ok, ok}), observed, effectDuplicate},
		{"exclusion nothing produces",
			reg([]registerRow{ok}, registerExclusion{classCIAction, "owner/gone@v1", "retired"}),
			observed, effectDeadExclusion},
	}

	seen := map[effectDriftKind]bool{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			findings := registerDrift(tc.eff, tc.reg)
			for _, f := range findings {
				seen[f.Kind] = true
				if f.Kind == tc.want {
					return
				}
			}
			t.Fatalf("want a %s finding, got:\n%s", tc.want, renderEffectFindings(findings))
		})
	}
	for _, k := range []effectDriftKind{effectMissing, effectUnobserved, effectPinMisdeclared,
		effectStatusStale, effectDomainDrift, effectBadRow, effectDuplicate, effectDeadExclusion} {
		if !seen[k] {
			t.Errorf("drift kind %q is declared but no case emits it — a vocabulary constant with no "+
				"emitter polices nothing", k)
		}
	}
}

// TestEffectDiscoveryFailsClosed pins the other half of "the gate can fail": inputs that
// cannot be honestly checked must ERROR, never read as agreement.
func TestEffectDiscoveryFailsClosed(t *testing.T) {
	if _, _, err := discoverEffects(map[string]string{"README.md": "# nothing to see"}); err == nil {
		t.Error("a corpus with no effects at all returned no error — a discovery pass that finds " +
			"nothing must fail closed, not read as `no third-party effects`")
	}
	regCases := []struct{ name, body, wantErr string }{
		{"not JSON", "{ nope", "not valid JSON"},
		{"wrong schema", `{"schema":"other/9","rows":[{"reference":"x"}]}`, "schema is"},
		{"no rows", `{"schema":"` + effectRegisterSchema + `","rows":[]}`, "zero rows"},
	}
	for _, tc := range regCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseRegister([]byte(tc.body)); err == nil ||
				!strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("parseRegister error = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// TestEffectRegisterDump is the GENERATOR side of the artifact: it prints a register that
// agrees with the current tree, with the human-owned fields (license_record, enforcement,
// status, digest, note) carried over from the committed register where a row already exists.
// It never writes — a supply-chain artifact should be reviewed, not silently regenerated.
//
//	FAK_EFFECTS_DUMP=1 go test ./internal/architest -run TestEffectRegisterDump -v
func TestEffectRegisterDump(t *testing.T) {
	if os.Getenv("FAK_EFFECTS_DUMP") == "" {
		t.Skip("set FAK_EFFECTS_DUMP=1 to print a regenerated register")
	}
	root := repoRoot(t)
	tree, err := loadEffectCorpus(root)
	if err != nil {
		t.Fatalf("load effect corpus: %v", err)
	}
	effects, _, err := discoverEffects(tree)
	if err != nil {
		t.Fatalf("discovery: %v", err)
	}
	prior := map[string]registerRow{}
	if data, err := os.ReadFile(registerPath(root)); err == nil {
		if reg, err := parseRegister(data); err == nil {
			for _, row := range reg.Rows {
				prior[string(row.Class)+"\x00"+row.Reference] = row
			}
		}
	}
	out := effectRegister{
		Schema: effectRegisterSchema,
		Regenerate: "FAK_EFFECTS_DUMP=1 go test ./internal/architest -run TestEffectRegisterDump -v " +
			"(review every field before committing; license_record and enforcement are human-owned)",
	}
	for _, e := range effects {
		row := registerRow{
			Class: e.Class, Reference: e.Reference, Domains: e.Domains,
			Status: statusUsed, Pin: computePin(e),
			LicenseRecord: "none", Enforcement: "none",
		}
		if p, ok := prior[e.key()]; ok {
			row.Status, row.Digest, row.LicenseRecord, row.Enforcement, row.Note =
				p.Status, p.Digest, p.LicenseRecord, p.Enforcement, p.Note
		}
		out.Rows = append(out.Rows, row)
	}
	blob, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("regenerated register:\n%s\n", blob)
}

func renderEffectFindings(findings []effectFinding) string {
	var b strings.Builder
	for _, f := range findings {
		fmt.Fprintf(&b, "  [%s] %s\n", f.Kind, f.Msg)
	}
	return b.String()
}
