package architest

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// SBOM/go.mod drift gate (issue #5374, child of #3279 — the air-gapped deployment kit).
//
// docs/sbom/fak.spdx.json is a HAND-CUT, checked-in SPDX 2.3 document that a regulated
// reviewer is invited to trust: the kit page tells them to stage it across the boundary as
// "the reviewer's evidence". Until this gate, nothing in the tree reds when go.mod grows and
// the SBOM does not — the kit page says so about itself, under "Invalidating assumption".
//
// That is not a hypothetical. This repo's dependency set went from zero modules to two, and
// the stale "zero external dependencies" claim survived in go.mod's own header, in the
// parent issue, and across several docs pages until a build-verified triage caught it. A
// supply-chain artifact that silently rots is WORSE than no artifact, because a reviewer
// reads it and stops looking.
//
// So the SBOM stops being prose someone re-runs three commands against by hand, and becomes
// a derived artifact CI re-derives from go.mod on every run — the same move the rest of this
// package makes against ARCHITECTURE.md.
//
// WHY THIS LIVES IN architest AND NOT A NEW LEAF. This is a repo-hygiene contract over
// checked-in artifacts, which is exactly what this package already is: stdlib-only, off the
// request path, never registered into the kernel. The direct precedent is
// workflow_refs_test.go in this same package — a "the checked-in artifact still matches the
// tree" gate, right down to the fail-closed clause that reds when the checker compared
// nothing. Putting it here also means no new package leaf, so no architest tier row is
// needed and the push gate's UNTIERED_LEAF check is untouched.
//
// WHY THE go.mod PARSER IS HAND-ROLLED. golang.org/x/mod/modfile is not a dependency of this
// module, and a gate whose job is to police the dependency surface must not expand it — the
// gate would then be its own first defect. The parser below is a small stdlib subset, the
// same trade internal/deploymanifest makes for TOML. It fails closed on syntax it does not
// model rather than skipping it, because a dependency checker that silently ignores a
// directive is how this class of gate rots into a no-op.

// driftKind is the closed vocabulary of drift this gate can report. The kind is the
// DIRECTION of the disagreement, which is the thing a fixer needs first: "the SBOM is
// blind to a module that ships" and "the SBOM names a module that does not ship" have
// opposite fixes and very different severity.
type driftKind string

const (
	// driftMissing is the supply-chain BLIND SPOT: bytes link into the binary that the
	// SBOM does not mention at all. This is the direction the gate exists for.
	driftMissing driftKind = "MODULE_MISSING_FROM_SBOM"
	// driftVersion is an affirmatively WRONG statement: the module is listed, at a
	// version that is not the version that ships. Worse than silence — a reviewer
	// checks advisories against the version the document names.
	driftVersion driftKind = "SBOM_VERSION_STALE"
	// driftExtra is the weaker, opposite direction: the SBOM over-declares.
	driftExtra driftKind = "SBOM_LISTS_UNREQUIRED_MODULE"
	// driftPurl / driftDownload / driftDirectness are the half-edited-entry classes: the
	// SBOM's own fields disagree with each other about the module go.mod pins. These only
	// fire when the entry's versionInfo already AGREES with go.mod, so they are never
	// redundant noise on top of a plain version bump.
	driftPurl       driftKind = "SBOM_PURL_DISAGREES"
	driftDownload   driftKind = "SBOM_DOWNLOAD_URL_DISAGREES"
	driftDirectness driftKind = "SBOM_DIRECTNESS_DISAGREES"
	// driftReplace is fail-closed: go.mod redirects a module and this SBOM shape has no
	// field that can say so.
	driftReplace driftKind = "GOMOD_REPLACE_UNREPRESENTED"
	// driftMainModule / driftDuplicate are structural: the document does not describe the
	// module it claims to, or describes one module twice.
	driftMainModule driftKind = "SBOM_MAIN_MODULE_ABSENT"
	driftDuplicate  driftKind = "SBOM_DUPLICATE_PACKAGE"
)

// sbomFinding is one drift. Msg is the deliverable: it names the module, the direction, and
// the value to write, so the fix needs no second command to work out.
type sbomFinding struct {
	Kind   driftKind
	Module string
	Msg    string
}

// driftID is a finding stripped to its assertable identity, for the mutation table.
type driftID struct {
	Kind   driftKind
	Module string
}

// ---------------------------------------------------------------------------
// go.mod (stdlib subset parser)
// ---------------------------------------------------------------------------

type goModRequire struct {
	Path     string
	Version  string
	Indirect bool
}

type goModFacts struct {
	Module   string
	Requires []goModRequire
	Replaces []string
	Excludes []string
}

// goModIgnoredVerbs are the top-level directives that describe the TOOLCHAIN, debug
// defaults, tool dependencies or withdrawn versions rather than the module bytes that link
// into the shipped binary. `exclude` is parsed but likewise not compared: an exclude only
// forbids a version during selection, it never puts bytes in the artifact, so an SBOM that
// is silent about it is not lying. Anything NOT in this set and not modeled below is a
// hard parse error — see parseGoMod.
var goModIgnoredVerbs = map[string]bool{
	"go":        true,
	"toolchain": true,
	"godebug":   true,
	"retract":   true,
	"tool":      true,
	"ignore":    true,
}

func splitGoModLine(raw string) (code, comment string) {
	if i := strings.Index(raw, "//"); i >= 0 {
		return strings.TrimSpace(raw[:i]), strings.TrimSpace(raw[i+2:])
	}
	return strings.TrimSpace(raw), ""
}

func (f *goModFacts) record(verb, text, comment string, line int) error {
	switch verb {
	case "require":
		fields := strings.Fields(text)
		if len(fields) != 2 {
			return fmt.Errorf("line %d: malformed require %q, want `<module path> <version>`", line, text)
		}
		f.Requires = append(f.Requires, goModRequire{
			Path:     fields[0],
			Version:  fields[1],
			Indirect: strings.Contains(comment, "indirect"),
		})
	case "replace":
		f.Replaces = append(f.Replaces, text)
	case "exclude":
		f.Excludes = append(f.Excludes, text)
	}
	return nil
}

func parseGoMod(data []byte) (goModFacts, error) {
	var f goModFacts
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	block := ""
	line := 0
	for sc.Scan() {
		line++
		code, comment := splitGoModLine(sc.Text())
		if code == "" {
			continue
		}
		if block != "" {
			if code == ")" {
				block = ""
				continue
			}
			if err := f.record(block, code, comment, line); err != nil {
				return f, err
			}
			continue
		}
		verb := strings.Fields(code)[0]
		switch {
		case verb == "module":
			fields := strings.Fields(code)
			if len(fields) != 2 {
				return f, fmt.Errorf("line %d: malformed module directive %q", line, code)
			}
			f.Module = strings.Trim(fields[1], `"`)
		case verb == "require" || verb == "replace" || verb == "exclude":
			rest := strings.TrimSpace(strings.TrimPrefix(code, verb))
			if rest == "(" {
				block = verb
				continue
			}
			if err := f.record(verb, rest, comment, line); err != nil {
				return f, err
			}
		case goModIgnoredVerbs[verb]:
			// Toolchain / retraction metadata: not part of the linked module set.
		default:
			return f, fmt.Errorf("line %d: unrecognized directive %q — this gate fails closed on "+
				"go.mod syntax it does not model, because quietly skipping a directive is how a "+
				"dependency checker rots into a no-op; teach parseGoMod what %q means", line, verb, verb)
		}
	}
	if err := sc.Err(); err != nil {
		return f, fmt.Errorf("scan: %w", err)
	}
	if block != "" {
		return f, fmt.Errorf("unterminated %s ( ... ) block", block)
	}
	if f.Module == "" {
		return f, fmt.Errorf("no module directive — this does not look like a go.mod")
	}
	return f, nil
}

// ---------------------------------------------------------------------------
// SPDX 2.3 JSON
// ---------------------------------------------------------------------------

type spdxExternalRef struct {
	ReferenceType    string `json:"referenceType"`
	ReferenceLocator string `json:"referenceLocator"`
}

type spdxPackage struct {
	Name             string            `json:"name"`
	VersionInfo      string            `json:"versionInfo"`
	DownloadLocation string            `json:"downloadLocation"`
	SourceInfo       string            `json:"sourceInfo"`
	ExternalRefs     []spdxExternalRef `json:"externalRefs"`
}

type spdxDocument struct {
	SPDXVersion string        `json:"spdxVersion"`
	Packages    []spdxPackage `json:"packages"`
}

func parseSPDX(data []byte) (spdxDocument, error) {
	var doc spdxDocument
	if err := json.Unmarshal(data, &doc); err != nil {
		return doc, fmt.Errorf("not valid JSON: %w", err)
	}
	if doc.SPDXVersion == "" {
		return doc, fmt.Errorf("no spdxVersion field — this is not an SPDX document")
	}
	if len(doc.Packages) == 0 {
		return doc, fmt.Errorf("zero packages — an SBOM that lists nothing witnesses nothing")
	}
	return doc, nil
}

func spdxPurl(p spdxPackage) (string, bool) {
	for _, ref := range p.ExternalRefs {
		if ref.ReferenceType == "purl" {
			return ref.ReferenceLocator, true
		}
	}
	return "", false
}

func requireKindWord(r goModRequire) string {
	if r.Indirect {
		return "indirect"
	}
	return "direct"
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

// sbomDrift compares a go.mod against an SPDX SBOM and returns every disagreement, plus the
// number of module comparisons it actually performed.
//
// It is deliberately pure — bytes in, findings out, no filesystem — so the mutation table
// below can prove the gate RED against a mutated pair, which is the only way to know a green
// run means agreement rather than a checker that found nothing to check.
//
// An error (as opposed to findings) is the fail-closed channel: an unreadable go.mod, a
// document that is not SPDX, an SBOM with no packages, or a pair with nothing to compare.
// Those must never read as "no drift".
func sbomDrift(goModBytes, sbomBytes []byte) (findings []sbomFinding, compared int, err error) {
	gm, err := parseGoMod(goModBytes)
	if err != nil {
		return nil, 0, fmt.Errorf("parse go.mod: %w", err)
	}
	doc, err := parseSPDX(sbomBytes)
	if err != nil {
		return nil, 0, fmt.Errorf("parse SBOM: %w", err)
	}

	// Split the document into the main module's own entry and the dependency entries. The
	// main module's versionInfo is a RELEASE version; go.mod does not carry it at all, so
	// comparing them would be a category error. Only its identity is checked.
	byName := make(map[string]spdxPackage, len(doc.Packages))
	var deps []spdxPackage
	mainSeen := false
	for _, p := range doc.Packages {
		if p.Name == gm.Module {
			mainSeen = true
			continue
		}
		if _, dup := byName[p.Name]; dup {
			findings = append(findings, sbomFinding{driftDuplicate, p.Name, fmt.Sprintf(
				"docs/sbom/fak.spdx.json contains two packages named %s — a duplicated entry makes the "+
					"document self-contradicting about which version ships; keep one entry per module.", p.Name)})
			continue
		}
		byName[p.Name] = p
		deps = append(deps, p)
	}

	compared = len(gm.Requires) + len(deps)
	if compared == 0 {
		return nil, 0, fmt.Errorf("nothing to compare: go.mod declares no requirements and the SBOM " +
			"lists no dependency packages, so this gate would pass without checking anything — which is " +
			"the silently-inert failure it exists to prevent. If the module genuinely carries no external " +
			"dependencies again, that is a deliberate change and this gate must be updated to say so")
	}

	if !mainSeen {
		findings = append(findings, sbomFinding{driftMainModule, gm.Module, fmt.Sprintf(
			"docs/sbom/fak.spdx.json has no package named %s — the document does not describe the module "+
				"it is supposed to be the bill of materials FOR (go.mod's module path). Re-cut the SBOM, or "+
				"update its main package entry if the module path moved.", gm.Module)})
	}

	required := make(map[string]goModRequire, len(gm.Requires))
	for _, r := range gm.Requires {
		required[r.Path] = r
	}

	for _, r := range gm.Requires {
		p, ok := byName[r.Path]
		if !ok {
			findings = append(findings, sbomFinding{driftMissing, r.Path, fmt.Sprintf(
				"go.mod requires %s %s (%s) but docs/sbom/fak.spdx.json lists no package named %s. This is "+
					"the BLIND-SPOT direction: bytes link into the shipped binary that a reviewer reading the "+
					"SBOM never learns about. Re-cut the SBOM per docs/air-gapped-deployment-kit.md — add %s "+
					"with versionInfo %s and purl pkg:golang/%s@%s.",
				r.Path, r.Version, requireKindWord(r), r.Path, r.Path, r.Version, r.Path, r.Version)})
			continue
		}
		if p.VersionInfo != r.Version {
			findings = append(findings, sbomFinding{driftVersion, r.Path, fmt.Sprintf(
				"docs/sbom/fak.spdx.json records %s at versionInfo %q but go.mod requires %q. The SBOM states "+
					"a version that is not the one that ships, so a reviewer checks advisories against the wrong "+
					"release. Set %s to %s in versionInfo, in the purl (pkg:golang/%s@%s), and in "+
					"downloadLocation before re-cutting.",
				r.Path, p.VersionInfo, r.Version, r.Path, r.Version, r.Path, r.Version)})
			// The derived-field checks below would all restate this one bump; skip them so the
			// report names one fix per real defect.
			continue
		}
		if locator, ok := spdxPurl(p); ok {
			want := "pkg:golang/" + r.Path + "@" + r.Version
			if locator != want {
				findings = append(findings, sbomFinding{driftPurl, r.Path, fmt.Sprintf(
					"docs/sbom/fak.spdx.json gives %s the purl %q while its own versionInfo and go.mod both say "+
						"%s — a half-edited entry. The purl is the machine-readable half a scanner consumes, so "+
						"it is the half that matters; set it to %q.", r.Path, locator, r.Version, want)})
			}
		}
		if strings.Contains(p.DownloadLocation, "proxy.golang.org") &&
			!strings.Contains(p.DownloadLocation, "/@v/"+r.Version+".zip") {
			findings = append(findings, sbomFinding{driftDownload, r.Path, fmt.Sprintf(
				"docs/sbom/fak.spdx.json points %s at downloadLocation %q while go.mod requires %s — an operator "+
					"staging this module across an air-gapped boundary would fetch the wrong zip. Point it at "+
					"https://proxy.golang.org/%s/@v/%s.zip.", r.Path, p.DownloadLocation, r.Version, r.Path, r.Version)})
		}
		if claim, stated := sourceInfoDirectness(p); stated && claim != requireKindWord(r) {
			findings = append(findings, sbomFinding{driftDirectness, r.Path, fmt.Sprintf(
				"docs/sbom/fak.spdx.json describes %s as an %s requirement, but go.mod records it as %s "+
					"(the `// indirect` marker %s). A module that became a direct requirement is a widened "+
					"attack surface a reviewer should see named as such; update the sourceInfo.",
				r.Path, claim, requireKindWord(r),
				map[bool]string{true: "is present", false: "is absent"}[r.Indirect])})
		}
	}

	for _, p := range deps {
		if _, ok := required[p.Name]; ok {
			continue
		}
		findings = append(findings, sbomFinding{driftExtra, p.Name, fmt.Sprintf(
			"docs/sbom/fak.spdx.json lists %s %s but go.mod requires no such module. This is the WEAKER "+
				"direction — no unlisted bytes ship, so it is not a blind spot — but it is still a stale "+
				"generated artifact making a false claim, and an air-gapped operator would stage a module zip "+
				"the build never asks for. Re-cut the SBOM against `go list -m all`; %s no longer belongs in it.",
			p.Name, p.VersionInfo, p.Name)})
	}

	for _, r := range gm.Replaces {
		findings = append(findings, sbomFinding{driftReplace, r, fmt.Sprintf(
			"go.mod carries `replace %s`, and this SBOM shape cannot express a replacement — every package "+
				"entry names an upstream path, version and proxy URL, so the document would assert provenance "+
				"for bytes that are not the bytes that link. Failing closed: either re-cut without the replace, "+
				"or land a replace-aware SBOM shape and teach this gate about it.", r)})
	}

	return findings, compared, nil
}

// sourceInfoDirectness reads the direct/indirect claim the SBOM entry makes in prose, if it
// makes one. A silent entry is not contradicted — the gate only argues with a claim that is
// actually there. Note the "indirect" test runs first: "indirect requirement" contains
// "direct requirement" as a substring.
func sourceInfoDirectness(p spdxPackage) (claim string, stated bool) {
	lower := strings.ToLower(p.SourceInfo)
	switch {
	case strings.Contains(lower, "indirect requirement"):
		return "indirect", true
	case strings.Contains(lower, "direct requirement"):
		return "direct", true
	}
	return "", false
}

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

func sbomPair(t *testing.T) (goModBytes, sbomBytes []byte) {
	t.Helper()
	root := filepath.Dir(internalDir(t)) // repo root = parent of internal/
	goModPath := filepath.Join(root, "go.mod")
	sbomPath := filepath.Join(root, "docs", "sbom", "fak.spdx.json")
	var err error
	if goModBytes, err = os.ReadFile(goModPath); err != nil {
		t.Fatalf("read %s: %v", goModPath, err)
	}
	if sbomBytes, err = os.ReadFile(sbomPath); err != nil {
		t.Fatalf("read %s: %v — the committed SBOM is the artifact this gate exists to keep honest; "+
			"if it moved, move this gate with it", sbomPath, err)
	}
	return goModBytes, sbomBytes
}

// TestSBOMMatchesGoMod is the gate itself: the committed SBOM must describe exactly the
// module set go.mod pins. Bumping or adding a module without re-cutting
// docs/sbom/fak.spdx.json reds here.
func TestSBOMMatchesGoMod(t *testing.T) {
	goModBytes, sbomBytes := sbomPair(t)
	findings, compared, err := sbomDrift(goModBytes, sbomBytes)
	if err != nil {
		t.Fatalf("SBOM/go.mod drift gate could not run: %v", err)
	}
	for _, f := range findings {
		t.Errorf("[%s] %s", f.Kind, f.Msg)
	}
	t.Logf("SBOM/go.mod agreement: %d module comparisons, %d findings", compared, len(findings))
}

// TestSBOMDriftGateCatchesMutations is the witness that the gate above can actually FAIL. A
// contract test that only ever runs against an agreeing pair proves nothing — this is the
// declared-but-unwired defect class, and a supply-chain gate is the worst place to have it.
//
// Each case mutates the REAL committed bytes rather than a frozen inline fixture, so the
// fixtures cannot drift away from the artifact they are supposed to model: if the SBOM is
// re-cut, these mutations are applied to the new bytes.
func TestSBOMDriftGateCatchesMutations(t *testing.T) {
	goModBytes, sbomBytes := sbomPair(t)
	identity := func(s string) string { return s }

	cases := []struct {
		name  string
		goMod func(string) string
		sbom  func(string) string
		want  []driftID
	}{
		{
			name: "go.mod grows a module and the SBOM is not re-cut",
			goMod: func(s string) string {
				return s + "\nrequire golang.org/x/text v0.30.0 // indirect\n"
			},
			sbom: identity,
			want: []driftID{{driftMissing, "golang.org/x/text"}},
		},
		{
			name: "go.mod bumps a version and the SBOM keeps the old one",
			goMod: func(s string) string {
				return strings.Replace(s, "golang.org/x/term v0.44.0", "golang.org/x/term v0.45.0", 1)
			},
			sbom: identity,
			want: []driftID{{driftVersion, "golang.org/x/term"}},
		},
		{
			name: "go.mod stops requiring a module the SBOM still lists",
			goMod: func(s string) string {
				return strings.Replace(s, "require golang.org/x/sys v0.46.0 // indirect", "", 1)
			},
			sbom: identity,
			want: []driftID{{driftExtra, "golang.org/x/sys"}},
		},
		{
			name:  "SBOM entry half-edited: versionInfo agrees, purl left behind",
			goMod: identity,
			sbom: func(s string) string {
				return strings.Replace(s, "pkg:golang/golang.org/x/term@v0.44.0",
					"pkg:golang/golang.org/x/term@v0.43.0", 1)
			},
			want: []driftID{{driftPurl, "golang.org/x/term"}},
		},
		{
			name:  "SBOM entry half-edited: proxy download URL left at the old version",
			goMod: identity,
			sbom: func(s string) string {
				return strings.Replace(s, "golang.org/x/sys/@v/v0.46.0.zip",
					"golang.org/x/sys/@v/v0.45.0.zip", 1)
			},
			want: []driftID{{driftDownload, "golang.org/x/sys"}},
		},
		{
			name: "an indirect requirement becomes direct and the SBOM still says indirect",
			goMod: func(s string) string {
				return strings.Replace(s, "require golang.org/x/sys v0.46.0 // indirect",
					"require golang.org/x/sys v0.46.0", 1)
			},
			sbom: identity,
			want: []driftID{{driftDirectness, "golang.org/x/sys"}},
		},
		{
			name: "go.mod redirects a module the SBOM cannot describe",
			goMod: func(s string) string {
				return s + "\nreplace golang.org/x/term => ./vendorlocal/term\n"
			},
			sbom: identity,
			want: []driftID{{driftReplace, "golang.org/x/term => ./vendorlocal/term"}},
		},
		{
			name:  "the SBOM stops describing the module it is a bill of materials for",
			goMod: identity,
			sbom: func(s string) string {
				return strings.Replace(s, `"name": "github.com/anthony-chaudhary/fak"`,
					`"name": "github.com/someone-else/fork"`, 1)
			},
			want: []driftID{
				{driftMainModule, "github.com/anthony-chaudhary/fak"},
				{driftExtra, "github.com/someone-else/fork"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mutGoMod := tc.goMod(string(goModBytes))
			mutSBOM := tc.sbom(string(sbomBytes))
			if mutGoMod == string(goModBytes) && mutSBOM == string(sbomBytes) {
				t.Fatal("the mutation changed nothing — the substitution missed, so this case would " +
					"vacuously assert against the unmutated pair; update the fixture to match the artifact")
			}
			findings, _, err := sbomDrift([]byte(mutGoMod), []byte(mutSBOM))
			if err != nil {
				t.Fatalf("drift gate errored on a mutated pair it should have reported on: %v", err)
			}
			got := make([]driftID, 0, len(findings))
			for _, f := range findings {
				got = append(got, driftID{f.Kind, f.Module})
			}
			if !sameDriftSet(got, tc.want) {
				t.Fatalf("findings = %v, want %v\nmessages:\n%s", got, tc.want, renderFindings(findings))
			}
			for _, f := range findings {
				if !strings.Contains(f.Msg, f.Module) {
					t.Errorf("[%s] message does not name the module %q, so the fix is not obvious "+
						"from the failure alone: %s", f.Kind, f.Module, f.Msg)
				}
			}
			t.Logf("caught:\n%s", renderFindings(findings))
		})
	}
}

// TestSBOMDriftGateFailsClosed pins the other half of "the gate can fail": inputs it cannot
// honestly compare must ERROR, never read as agreement. The last case is the one that
// matters most — an empty pair with nothing to compare is a gate that passes because it
// looked at nothing.
func TestSBOMDriftGateFailsClosed(t *testing.T) {
	const okGoMod = "module github.com/anthony-chaudhary/fak\n\ngo 1.26\n\nrequire golang.org/x/term v0.44.0\n"
	const okSBOM = `{"spdxVersion":"SPDX-2.3","packages":[
		{"name":"github.com/anthony-chaudhary/fak","versionInfo":"v0.41.0"},
		{"name":"golang.org/x/term","versionInfo":"v0.44.0"}]}`
	const mainOnlySBOM = `{"spdxVersion":"SPDX-2.3","packages":[
		{"name":"github.com/anthony-chaudhary/fak","versionInfo":"v0.41.0"}]}`

	cases := []struct {
		name, goMod, sbom, wantErr string
	}{
		{"SBOM is not JSON", okGoMod, "{ this is not json", "not valid JSON"},
		{"JSON that is not an SPDX document", okGoMod, `{"hello":"world"}`, "not an SPDX document"},
		{"SPDX document with no packages", okGoMod, `{"spdxVersion":"SPDX-2.3","packages":[]}`, "zero packages"},
		{"go.mod with no module directive", "go 1.26\n", okSBOM, "no module directive"},
		{"go.mod with a directive the parser does not model", okGoMod + "\nsomefutureverb foo\n", okSBOM, "unrecognized directive"},
		{"go.mod with a malformed require", "module m\n\nrequire golang.org/x/term\n", okSBOM, "malformed require"},
		{"go.mod with an unterminated block", "module m\n\nrequire (\n", okSBOM, "unterminated"},
		{"zero modules on both sides — nothing to compare", "module github.com/anthony-chaudhary/fak\n\ngo 1.26\n", mainOnlySBOM, "nothing to compare"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			findings, compared, err := sbomDrift([]byte(tc.goMod), []byte(tc.sbom))
			if err == nil {
				t.Fatalf("want an error containing %q, got nil (findings=%d, compared=%d) — this input "+
					"cannot be honestly compared, so passing is the silently-inert failure",
					tc.wantErr, len(findings), compared)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// TestSBOMDriftGateIgnoresNonLinkingDirectives pins the SCOPE decision: `exclude` and the
// toolchain directives never put bytes in the artifact, so an SBOM that is silent about them
// is not stale. Without this, a future `exclude` line would red the trunk for no supply-chain
// reason and the gate would get switched off.
func TestSBOMDriftGateIgnoresNonLinkingDirectives(t *testing.T) {
	goModBytes, sbomBytes := sbomPair(t)
	base, _, err := sbomDrift(goModBytes, sbomBytes)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	widened := string(goModBytes) + "\nexclude golang.org/x/term v0.20.0\n\ngodebug default=go1.26\n"
	findings, _, err := sbomDrift([]byte(widened), sbomBytes)
	if err != nil {
		t.Fatalf("exclude/godebug made the gate unrunnable: %v", err)
	}
	if len(findings) != len(base) {
		t.Errorf("exclude + godebug produced %d findings vs %d at baseline — these directives do not "+
			"change the linked module set, so they must not be reported as SBOM drift:\n%s",
			len(findings), len(base), renderFindings(findings))
	}
}

func sameDriftSet(got, want []driftID) bool {
	if len(got) != len(want) {
		return false
	}
	key := func(d driftID) string { return string(d.Kind) + "\x00" + d.Module }
	g := make([]string, 0, len(got))
	w := make([]string, 0, len(want))
	for _, d := range got {
		g = append(g, key(d))
	}
	for _, d := range want {
		w = append(w, key(d))
	}
	sort.Strings(g)
	sort.Strings(w)
	for i := range g {
		if g[i] != w[i] {
			return false
		}
	}
	return true
}

func renderFindings(findings []sbomFinding) string {
	var b strings.Builder
	for _, f := range findings {
		fmt.Fprintf(&b, "  [%s] %s\n", f.Kind, f.Msg)
	}
	return b.String()
}
