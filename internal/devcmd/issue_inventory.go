package devcmd

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/issuededup"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/modver"
	"github.com/anthony-chaudhary/fak/internal/releasestale"
)

var issueInventoryNow = time.Now

type issueInventoryInput struct {
	Schema       string                         `json:"schema,omitempty"`
	GeneratedAt  string                         `json:"generated_at,omitempty"`
	Observations []issuepolicy.ErrorObservation `json:"observations"`
}

func RunIssueInventory(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromIssues := fs.String("from-issues", "", "read a cached gh issue array from PATH or '-'")
	fromObservations := fs.String("from-observations", "", "read trusted error observations from PATH")
	asJSON := fs.Bool("json", false, "emit fak.issue-error-inventory/1 JSON")
	requireActionable := fs.Int("require-actionable", 0, "gate dispatch for one issue number")
	root := fs.String("root", ".", "local repository root for read-only provenance resolution")
	trunkRef := fs.String("trunk-ref", "", "trunk ref whose exact commit was tested (default: configured development branch)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 || *fromIssues == "" || *fromObservations == "" {
		fmt.Fprintln(stderr, "fak-dev issue inventory: --from-issues and --from-observations are required")
		return 2
	}
	issueRaw, err := readIssueDedupInput(*fromIssues)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	issues, err := issuededup.ParseBacklog(issueRaw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	obsRaw, err := os.ReadFile(*fromObservations)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	input, err := decodeIssueInventoryInput(obsRaw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	if err := validateInventoryIssueSet(issues, input.Observations); err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	generatedAt := issueInventoryNow().UTC()
	if input.GeneratedAt != "" {
		generatedAt, err = time.Parse(time.RFC3339, input.GeneratedAt)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue inventory: invalid generated_at: %v\n", err)
			return 2
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	resolvedTrunk, err := issueInventoryTrunkRef(*root, *trunkRef)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: branch roles: %v\n", err)
		return 1
	}
	proof, err := newIssueInventoryProof(ctx, *root, resolvedTrunk)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 1
	}
	var verified []issuepolicy.ErrorObservation
	var refused []issuepolicy.ErrorInventoryIssue
	for _, observation := range input.Observations {
		resolved, reason := proof.resolveObservation(ctx, observation)
		if reason == "" {
			verified = append(verified, resolved)
		} else {
			refused = append(refused, issuepolicy.ErrorInventoryIssue{
				Issue: observation.Issue, CanonicalIssue: observation.Issue,
				Disposition: issuepolicy.DispositionReproRequired, ReasonCodes: []string{reason},
				ObservedFailure: observation.Observed, ProvenFix: observation.Fix,
				CurrentTestedState: observation.Current, ScopeException: observation.ScopeException,
			})
		}
	}
	if ctx.Err() != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: provenance deadline: %v\n", ctx.Err())
		return 1
	}
	if !proof.unchanged(ctx) {
		fmt.Fprintln(stderr, "fak-dev issue inventory: repository references changed during provenance resolution")
		return 1
	}
	digest, err := issueInventoryDigest(issues, input.Observations)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 1
	}
	// Bind the input snapshot to the actual trunk, tags and resolved evidence.
	// Resolution partitions observations; keep both partitions canonical too.
	sort.Slice(verified, func(i, j int) bool { return verified[i].Issue < verified[j].Issue })
	sort.Slice(refused, func(i, j int) bool { return refused[i].Issue < refused[j].Issue })
	bound, err := json.Marshal([]any{digest, proof.trunk, proof.tags, verified, refused})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 1
	}
	sum := sha256.Sum256(bound)
	digest = "sha256:" + hex.EncodeToString(sum[:])
	rep, err := issuepolicy.BuildErrorInventory(issuepolicy.ErrorInventoryInput{GeneratedAt: generatedAt, SnapshotDigest: digest, Observations: verified})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	rep.Issues = append(rep.Issues, refused...)
	sort.Slice(rep.Issues, func(i, j int) bool { return rep.Issues[i].Issue < rep.Issues[j].Issue })
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, rep, "fak-dev issue inventory"); code != 0 {
			return code
		}
	} else {
		renderIssueInventory(stdout, rep)
	}
	if *requireActionable != 0 {
		for _, row := range rep.Issues {
			if row.Issue == *requireActionable {
				return issuepolicy.ActionabilityExit(row.Disposition)
			}
		}
		fmt.Fprintf(stderr, "fak-dev issue inventory: issue %d is not in the inventory\n", *requireActionable)
		return 2
	}
	return 0
}

func decodeIssueInventoryInput(b []byte) (issueInventoryInput, error) {
	var in issueInventoryInput
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		return in, fmt.Errorf("parse observations: %w", err)
	}
	if in.Schema != "" && in.Schema != issuepolicy.ErrorInventorySchema {
		return in, fmt.Errorf("unsupported observations schema %q", in.Schema)
	}
	if len(in.Observations) == 0 {
		return in, fmt.Errorf("observations are required")
	}
	return in, nil
}

func validateInventoryIssueSet(issues []issuededup.BacklogIssue, obs []issuepolicy.ErrorObservation) error {
	known := make(map[int]bool, len(issues))
	for _, issue := range issues {
		known[issue.Number] = true
	}
	seen := make(map[int]bool)
	for _, row := range obs {
		if row.Issue <= 0 || seen[row.Issue] {
			return fmt.Errorf("observation issue numbers must be positive and unique")
		}
		seen[row.Issue] = true
		if !known[row.Issue] {
			return fmt.Errorf("observation issue %d is absent from --from-issues", row.Issue)
		}
	}
	return nil
}

// Repository facts are read here, never accepted from issue bodies or caller
// ancestry booleans. Witness pass/fail remains a trusted receipt input: resolving
// a commit does not itself execute a reproducer or authenticate a test result.
type issueInventoryProof struct {
	root, trunk, tags string
	refs              map[string]string
	modules           map[string]modver.Report
	tagObjects        map[string]string
	ancestors         map[[2]string]bool
	run               modver.Runner
	complete          bool
}

func (p *issueInventoryProof) git(ctx context.Context, args ...string) ([]byte, error) {
	run := p.run
	if run == nil {
		run = modver.RealRunner
	}
	// Fail closed on Git versions that cannot disable lazy object fetching.
	// This applies to every read, including reads issued by modver.SnapshotAt.
	return run(ctx, p.root, append([]string{"--no-lazy-fetch", "--no-replace-objects", "--no-optional-locks"}, args...)...)
}

func newIssueInventoryProof(ctx context.Context, root, trunk string) (*issueInventoryProof, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	abs, err = filepath.EvalSymlinks(abs)
	if err != nil {
		return nil, err
	}
	p := &issueInventoryProof{root: abs, refs: map[string]string{}, modules: map[string]modver.Report{}, tagObjects: map[string]string{}, ancestors: map[[2]string]bool{}}
	top, err := p.git(ctx, "rev-parse", "--show-toplevel")
	if err != nil {
		return nil, err
	}
	actual, err := filepath.EvalSymlinks(strings.TrimSpace(string(top)))
	if err != nil || actual != abs {
		return nil, fmt.Errorf("repository root does not match Git context")
	}
	p.trunk, err = p.commit(ctx, trunk)
	if err != nil {
		return nil, fmt.Errorf("resolve tested trunk: %w", err)
	}
	shallow, err := p.git(ctx, "rev-parse", "--is-shallow-repository")
	if err != nil {
		return nil, err
	}
	p.complete = strings.TrimSpace(string(shallow)) == "false"
	tags, err := p.git(ctx, "for-each-ref", "--format=%(refname:strip=2) %(objectname)", "refs/tags")
	if err != nil {
		return nil, err
	}
	p.tags = string(tags)
	for _, line := range strings.Split(p.tags, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && releasestale.IsSemverTag(fields[0]) {
			p.tagObjects[fields[0]] = fields[1]
		}
	}
	return p, nil
}

func (p *issueInventoryProof) commit(ctx context.Context, ref string) (string, error) {
	if strings.TrimSpace(ref) == "" {
		return "", fmt.Errorf("missing commit or ref")
	}
	if sha, ok := p.refs[ref]; ok {
		return sha, nil // Rechecked independently by unchanged before publication.
	}
	out, err := p.git(ctx, "rev-parse", "--verify", "--end-of-options", ref+"^{commit}")
	if err != nil {
		return "", err
	}
	sha := strings.TrimSpace(string(out))
	p.refs[ref] = sha
	return sha, nil
}

func (p *issueInventoryProof) module(ctx context.Context, sha, module string) (string, error) {
	report, ok := p.modules[sha]
	if !ok {
		var err error
		report, err = modver.SnapshotAt(ctx, p.root, func(ctx context.Context, _ string, args ...string) ([]byte, error) { return p.git(ctx, args...) }, sha)
		if err != nil {
			return "", err
		}
		p.modules[sha] = report
	}
	for _, m := range report.Modules {
		if m.Name == module {
			return module + "@" + m.Version(), nil
		}
	}
	return "", fmt.Errorf("module absent at commit")
}

func (p *issueInventoryProof) evidence(ctx context.Context, e issuepolicy.ErrorEvidence, module string) (issuepolicy.ErrorEvidence, error) {
	if e.Module == "" {
		e.Module = module
	}
	if e.Module != module || module == "" {
		return e, fmt.Errorf("module mismatch")
	}
	ref := e.Commit
	if ref == "" {
		ref = e.Ref
	}
	sha, err := p.commit(ctx, ref)
	if err != nil {
		return e, err
	}
	if e.Ref != "" {
		resolved, err := p.commit(ctx, e.Ref)
		if err != nil || resolved != sha {
			return e, fmt.Errorf("ref and commit disagree")
		}
	}
	version, err := p.module(ctx, sha, module)
	if err != nil {
		return e, err
	}
	if e.ModuleVersion != "" && e.ModuleVersion != version {
		return e, fmt.Errorf("module version mismatch")
	}
	e.Commit, e.ModuleVersion = sha, version
	return e, nil
}

func (p *issueInventoryProof) ancestor(ctx context.Context, older, newer string) (bool, error) {
	if older == newer {
		return true, nil
	}
	key := [2]string{older, newer}
	if yes, ok := p.ancestors[key]; ok {
		return yes, nil
	}
	// Unlike merge-base's no-common-ancestor exit, an empty ancestry path is a
	// successful negative result, distinguishable from an operational failure.
	out, err := p.git(ctx, "rev-list", "-n", "1", "--ancestry-path", older+".."+newer)
	yes := strings.TrimSpace(string(out)) != ""
	if err == nil {
		p.ancestors[key] = yes
	}
	return yes, err
}

func (p *issueInventoryProof) resolveObservation(ctx context.Context, o issuepolicy.ErrorObservation) (issuepolicy.ErrorObservation, string) {
	if !p.complete {
		return o, "REPOSITORY_HISTORY_INCOMPLETE"
	}
	var err error
	module := o.Observed.Module
	o.Observed, err = p.evidence(ctx, o.Observed, module)
	if err != nil {
		return o, "OBSERVED_PROVENANCE_UNVERIFIED"
	}
	o.Current, err = p.evidence(ctx, o.Current, module)
	if err != nil {
		return o, "CURRENT_PROVENANCE_UNVERIFIED"
	}
	if o.Current.Commit != p.trunk {
		o.Current.Stale = true
	}
	if o.Fix.Commit == "" && o.Fix.Ref == "" {
		if len(o.Releases) != 0 {
			return o, "FIX_PROVENANCE_REQUIRED"
		}
		return o, ""
	}
	o.Fix, err = p.evidence(ctx, o.Fix, module)
	if err != nil {
		return o, "FIX_PROVENANCE_UNVERIFIED"
	}
	afterFailure, failureErr := p.ancestor(ctx, o.Observed.Commit, o.Fix.Commit)
	onTrunk, trunkErr := p.ancestor(ctx, o.Fix.Commit, p.trunk)
	if failureErr != nil || trunkErr != nil || !afterFailure || !onTrunk {
		return o, "FIX_ANCESTRY_UNVERIFIED"
	}
	claims := map[string]issuepolicy.ReleaseEvidence{}
	for _, claim := range o.Releases {
		if _, duplicate := claims[claim.Tag]; duplicate {
			return o, "RELEASE_PROVENANCE_UNVERIFIED"
		}
		if _, ok := releasestale.ParseSemver(claim.Tag); !ok {
			return o, "RELEASE_PROVENANCE_UNVERIFIED"
		}
		// Peel the object captured in the initial tag snapshot, never a later
		// value of the mutable tag name (including annotated tag objects).
		sha, err := p.commit(ctx, p.tagObjects[claim.Tag])
		if err != nil {
			return o, "RELEASE_PROVENANCE_UNVERIFIED"
		}
		version, err := p.module(ctx, sha, module)
		if err != nil || (claim.ModuleVersion != "" && claim.ModuleVersion != version) {
			return o, "RELEASE_PROVENANCE_UNVERIFIED"
		}
		contains, err := p.ancestor(ctx, o.Fix.Commit, sha)
		if err != nil || (claim.FixAncestor && !contains) {
			return o, "RELEASE_ANCESTRY_UNVERIFIED"
		}
		claims[claim.Tag] = claim
	}
	var tags []string
	for tag := range p.tagObjects {
		tags = append(tags, tag)
	}
	sort.Slice(tags, func(i, j int) bool {
		a, _ := releasestale.ParseSemver(tags[i])
		b, _ := releasestale.ParseSemver(tags[j])
		for k := range a {
			if a[k] != b[k] {
				return a[k] < b[k]
			}
		}
		return tags[i] < tags[j]
	})
	o.Releases = nil
	for _, tag := range tags {
		sha, err := p.commit(ctx, p.tagObjects[tag])
		if err != nil {
			return o, "RELEASE_PROVENANCE_UNVERIFIED"
		}
		contains, err := p.ancestor(ctx, o.Fix.Commit, sha)
		if err != nil {
			return o, "RELEASE_ANCESTRY_UNVERIFIED"
		}
		claim := claims[tag]
		if !contains && !(claim.PassingReceipt && claim.Witness != "") {
			continue
		}
		version, err := p.module(ctx, sha, module)
		if err != nil {
			return o, "RELEASE_PROVENANCE_UNVERIFIED"
		}
		witness := claim.Witness
		if contains {
			witness = "git:" + o.Fix.Commit + " ancestor-of " + sha
		}
		o.Releases = []issuepolicy.ReleaseEvidence{{Tag: tag, FixAncestor: contains, PassingReceipt: claim.PassingReceipt, ModuleVersion: version, Witness: witness}}
		break
	}
	return o, ""
}

func (p *issueInventoryProof) unchanged(ctx context.Context) bool {
	for ref, sha := range p.refs {
		out, err := p.git(ctx, "rev-parse", "--verify", "--end-of-options", ref+"^{commit}")
		if err != nil || strings.TrimSpace(string(out)) != sha {
			return false
		}
	}
	tags, err := p.git(ctx, "for-each-ref", "--format=%(refname:strip=2) %(objectname)", "refs/tags")
	return err == nil && string(tags) == p.tags
}

func issueInventoryDigest(issues []issuededup.BacklogIssue, obs []issuepolicy.ErrorObservation) (string, error) {
	sort.Slice(issues, func(i, j int) bool { return issues[i].Number < issues[j].Number })
	sort.Slice(obs, func(i, j int) bool { return obs[i].Issue < obs[j].Issue })
	b, err := json.Marshal(struct {
		Issues       []issuededup.BacklogIssue      `json:"issues"`
		Observations []issuepolicy.ErrorObservation `json:"observations"`
	}{issues, obs})
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func renderIssueInventory(w io.Writer, rep issuepolicy.ErrorInventory) {
	fmt.Fprintf(w, "issue error inventory %s\n", rep.SnapshotDigest)
	for _, row := range rep.Issues {
		fmt.Fprintf(w, "#%s %-20s canonical=#%s %s\n", strconv.Itoa(row.Issue), row.Disposition, strconv.Itoa(row.CanonicalIssue), row.ReasonCodes[0])
	}
}

// issueInventoryTrunkRef resolves the local repository's development role while
// preserving an explicit evidence ref and the legacy default for unconfigured repos.
func issueInventoryTrunkRef(root, explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	path := filepath.Join(root, "dos.toml")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return "refs/heads/" + branchrole.Defaults().DevelopmentBranch, nil
	} else if err != nil {
		return "", err
	}
	roles, err := branchrole.LoadFile(path)
	if err != nil {
		return "", err
	}
	return "refs/heads/" + roles.DevelopmentBranch, nil
}
