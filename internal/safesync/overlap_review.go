package safesync

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

const CleanOverlapReviewSchema = "fak.sync.clean-overlap-review.v1"

// CleanOverlapReview is a reviewable description, NOT an approval. Identity and
// both parent-to-result diffs participate in its digest. Names are audit metadata,
// not authentication; the authorized operator's explicit acceptance is the boundary.
type CleanOverlapReview struct {
	Schema       string                    `json:"schema"`
	Repository   string                    `json:"repository"`
	Worktree     string                    `json:"worktree"`
	LocalHead    string                    `json:"local_head"`
	TargetSHA    string                    `json:"target_sha"`
	MergeBases   []string                  `json:"merge_bases"`
	MergedTree   string                    `json:"merged_tree"`
	OverlapPaths []string                  `json:"overlap_paths"`
	WritePaths   []string                  `json:"write_paths"`
	LocalOnly    bool                      `json:"local_only"`
	Reviewer     string                    `json:"reviewer"`
	Evidence     string                    `json:"evidence"`
	LocalDiff    string                    `json:"local_to_result_diff"`
	TargetDiff   string                    `json:"target_to_result_diff"`
	LockWaivers  []leaseref.SnapshotWaiver `json:"lock_waivers,omitempty"`
	Digest       string                    `json:"digest"`
}

// ReviewedOverlapApproval records explicit acceptance of exactly one manifest.
// It is intentionally separate from the generated manifest.
type ReviewedOverlapApproval struct {
	Digest string `json:"digest"`
}

// ReviewedOverlapRequest is accepted only by local-only packet execution.
type ReviewedOverlapRequest struct {
	Review   CleanOverlapReview      `json:"review"`
	Approval ReviewedOverlapApproval `json:"approval"`
}

// CleanOverlapReviewOptions specifies immutable inputs and accountable review metadata.
type CleanOverlapReviewOptions struct {
	Repo        string
	TargetSHA   string
	Reviewer    string
	Evidence    string
	LockWaivers []leaseref.SnapshotWaiver
	Runner      Runner
	Now         func() time.Time
}

type reviewedOverlapAdmission struct{ review CleanOverlapReview }

func reviewDigest(review CleanOverlapReview) string {
	review.Digest = ""
	data, _ := json.Marshal(review)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func exactObjectID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil && s == strings.ToLower(s)
}

func reviewOutput(ctx context.Context, run Runner, repo string, args ...string) (string, error) {
	r := run(ctx, repo, args...)
	if r.Err != nil || r.Code != 0 {
		return "", fmt.Errorf("review inspection %v: code=%d: %v %s", args, r.Code, r.Err, r.Stderr)
	}
	return string(r.Stdout), nil
}

func reviewIdentity(ctx context.Context, run Runner, repo, arg string) (string, error) {
	s, err := reviewOutput(ctx, run, repo, "rev-parse", "--path-format=absolute", arg)
	if err != nil {
		return "", err
	}
	p, err := filepath.EvalSymlinks(strings.TrimSpace(s))
	if err != nil {
		return "", err
	}
	return filepath.Abs(p)
}

func reviewCleanState(ctx context.Context, run Runner, repo string) error {
	dirty, err := workingTreeDirtyPaths(ctx, run, repo)
	if err != nil {
		return err
	}
	if len(dirty) != 0 {
		return fmt.Errorf("review requires clean index and worktree: %v", dirty)
	}
	gd, err := worktreeGitDir(repo)
	if err != nil {
		return err
	}
	for _, name := range []string{"MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply", "sequencer"} {
		_, err := os.Stat(filepath.Join(gd, name))
		if err == nil {
			return fmt.Errorf("review refuses in-progress operation: %s", name)
		}
		if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// reviewTreePaths preserves Git's literal NUL-delimited names. The initial
// capability explicitly rejects names the existing lane/glob geometry cannot
// safely represent, instead of normalizing a distinct name into an owned path.
func reviewTreePaths(ctx context.Context, run Runner, repo, from, to string) ([]string, error) {
	raw, err := reviewOutput(ctx, run, repo, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--name-only", "-z", from, to, "--")
	if err != nil {
		return nil, err
	}
	if raw == "" {
		return nil, nil
	}
	data := []byte(raw)
	if data[len(data)-1] != 0 {
		return nil, fmt.Errorf("unterminated native review path output")
	}
	fields := bytes.Split(data[:len(data)-1], []byte{0})
	paths := make([]string, 0, len(fields))
	for _, field := range fields {
		p := string(field)
		if p == "" || strings.ContainsAny(p, "\\*?[]") || strings.IndexFunc(p, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsControl(r) }) >= 0 {
			return nil, fmt.Errorf("reviewed overlap does not support literal path %q", p)
		}
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths, nil
}

func reviewGeometryOverlaps(p string, geometry []string) bool {
	// TreesOverlap treats an empty side as unknown/universal ownership. A missing
	// optional taxonomy entry is not such a claim when explicit geometry exists.
	return len(geometry) != 0 && (dispatchorder.TreesOverlap([]string{p}, geometry) || matchesAnyGlob(p, geometry))
}

// PrepareCleanOverlapReview inspects a clean divergent merge without updating
// refs, index or worktree. Git merge-tree may create unreachable scratch objects.
func PrepareCleanOverlapReview(ctx context.Context, opts CleanOverlapReviewOptions) (*CleanOverlapReview, error) {
	if opts.Repo == "" {
		opts.Repo = "."
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	if !exactObjectID(opts.TargetSHA) || strings.TrimSpace(opts.Reviewer) == "" || strings.TrimSpace(opts.Evidence) == "" {
		return nil, fmt.Errorf("review requires exact target commit, reviewer and evidence reference")
	}
	if err := leaseref.ValidateSnapshotWaivers(opts.LockWaivers); err != nil {
		return nil, fmt.Errorf("invalid review lock waivers: %w", err)
	}
	run, repo := opts.Runner, opts.Repo
	if err := reviewCleanState(ctx, run, repo); err != nil {
		return nil, err
	}
	head, err := rev(ctx, run, repo, "HEAD^{commit}")
	if err != nil {
		return nil, err
	}
	target, err := rev(ctx, run, repo, opts.TargetSHA+"^{commit}")
	if err != nil || target != opts.TargetSHA {
		return nil, fmt.Errorf("review target is not the exact commit: %v", err)
	}
	common, err := reviewIdentity(ctx, run, repo, "--git-common-dir")
	if err != nil {
		return nil, err
	}
	worktree, err := reviewIdentity(ctx, run, repo, "--show-toplevel")
	if err != nil {
		return nil, err
	}
	bases, err := reviewOutput(ctx, run, repo, "merge-base", "--all", head, target)
	if err != nil {
		return nil, err
	}
	baseSet := strings.Fields(bases)
	// Bounded initial capability: ambiguous/criss-cross merge bases remain refused.
	if len(baseSet) != 1 || baseSet[0] == head || baseSet[0] == target {
		return nil, fmt.Errorf("review requires divergent commits with one merge base")
	}
	local, err := reviewTreePaths(ctx, run, repo, baseSet[0], head)
	if err != nil {
		return nil, err
	}
	remote, err := reviewTreePaths(ctx, run, repo, baseSet[0], target)
	if err != nil {
		return nil, err
	}
	localPaths := map[string]bool{}
	for _, p := range local {
		localPaths[p] = true
	}
	overlap := map[string]bool{}
	for _, p := range remote {
		if localPaths[p] {
			overlap[p] = true
		}
	}
	paths := make([]string, 0, len(overlap))
	for p := range overlap {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, fmt.Errorf("review requires nonempty committed overlap")
	}
	tree, err := reviewOutput(ctx, run, repo, "merge-tree", "--write-tree", "--no-messages", head, target)
	if err != nil {
		return nil, fmt.Errorf("review requires a clean native merge preview: %w", err)
	}
	tree = strings.TrimSpace(tree)
	if !exactObjectID(tree) {
		return nil, fmt.Errorf("invalid merge preview tree")
	}
	writeSet := map[string]bool{}
	for _, paths := range [][]string{local, remote} {
		for _, p := range paths {
			writeSet[p] = true
		}
	}
	// A merge commit can introduce paths absent from ordinary log --name-only.
	// Include both complete parent/result tree differences, never history text.
	for _, parent := range []string{head, target} {
		changed, err := reviewTreePaths(ctx, run, repo, parent, tree)
		if err != nil {
			return nil, err
		}
		for _, p := range changed {
			writeSet[p] = true
		}
	}
	writes := make([]string, 0, len(writeSet))
	for p := range writeSet {
		writes = append(writes, p)
	}
	sort.Strings(writes)
	localDiff, err := reviewOutput(ctx, run, repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", head, tree, "--")
	if err != nil {
		return nil, err
	}
	targetDiff, err := reviewOutput(ctx, run, repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", target, tree, "--")
	if err != nil {
		return nil, err
	}
	actual, err := rev(ctx, run, repo, "HEAD^{commit}")
	if err != nil || actual != head {
		return nil, fmt.Errorf("HEAD moved while preparing review")
	}
	if err := reviewCleanState(ctx, run, repo); err != nil {
		return nil, err
	}
	review := &CleanOverlapReview{Schema: CleanOverlapReviewSchema, Repository: common, Worktree: worktree,
		LocalHead: head, TargetSHA: target, MergeBases: baseSet, MergedTree: tree, OverlapPaths: paths, WritePaths: writes, LocalOnly: true,
		Reviewer: opts.Reviewer, Evidence: opts.Evidence, LocalDiff: localDiff, TargetDiff: targetDiff,
		LockWaivers: append([]leaseref.SnapshotWaiver(nil), opts.LockWaivers...)}
	review.Digest = reviewDigest(*review)
	return review, nil
}

// checkReviewWriter verifies the actual token synchronously, not only the
// heartbeat's eventually observed Lost signal. Independent path claims are never
// exempted by this token or by a caller-supplied session/holder string.
func checkReviewWriter(repo string, lease *WriterLease) error {
	if lease == nil || lease.Lost() {
		return fmt.Errorf("review requires held writer lease")
	}
	gd, err := worktreeGitDir(repo)
	if err != nil {
		return err
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	if filepath.Clean(lease.path) != filepath.Join(gd, writerLeaseFile) {
		return fmt.Errorf("writer lease belongs to another worktree")
	}
	actual, err := readLease(lease.path)
	if err != nil {
		return err
	}
	if actual != lease.info || leaseStale(actual, time.Now(), lease.ttl) {
		return ErrWriterLeaseLost
	}
	return nil
}

func checkReviewPathOwners(ctx context.Context, opts CleanOverlapReviewOptions, review CleanOverlapReview) error {
	run := opts.Runner
	if run == nil {
		run = RealRunner
	}
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	lr := func(c context.Context, dir string, args ...string) (string, int, error) {
		r := run(c, dir, args...)
		return string(r.Stdout), r.Code, r.Err
	}
	store := leaseref.NewWithRunner(lr, opts.Repo)
	var records []leaseref.Record
	var err error
	if len(review.LockWaivers) == 0 {
		records, _, err = store.StrictLiveSnapshot(ctx, now())
	} else {
		// This helper is reached only after explicit digest acceptance and native
		// review recomputation. No other sync mode receives these exceptions.
		records, _, err = store.StrictLiveSnapshotWithWaivers(ctx, now(), review.LockWaivers)
	}
	if err != nil {
		return fmt.Errorf("inspect independent path leases: %w", err)
	}
	tax, err := reviewTaxonomy(opts.Repo)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("inspect path taxonomy: %w", err)
	}
	// The freshly recomputed manifest includes complete base/parent/result
	// tree differences. Its path bytes have not passed through log, quoting, or
	// whitespace normalization.
	for _, rec := range records {
		for _, pattern := range rec.TreeGlobs {
			if err := reviewGlob(pattern); err != nil {
				return fmt.Errorf("cannot resolve malformed live path claim %s", rec.ID)
			}
		}
		if len(rec.TreeGlobs) == 0 && len(tax.Trees[rec.ID]) == 0 {
			return fmt.Errorf("cannot resolve live lane claim %s", rec.ID)
		}
		// Delegate exclusivity and hierarchical lane semantics to the native
		// admission engine, without setting SelfID from caller-owned text.
		lane := rec.ID
		if _, known := tax.Trees[lane]; !known && !tax.Exclusive[lane] {
			lane = ""
		}
		geometry := rec.TreeGlobs
		if len(geometry) == 0 {
			geometry = tax.Trees[rec.ID]
		}
		live := []regionadmit.Lease{{ID: rec.ID, Holder: rec.Holder, Lane: lane, Tree: geometry}}
		for _, p := range review.WritePaths {
			if decision := regionadmit.Decide(regionadmit.Request{Lane: regionadmit.LaneOf([]string{p}, tax), Tree: []string{p}}, live, tax); !decision.Admit {
				return fmt.Errorf("independent lease admission refused: %s", decision.Detail)
			}
		}
		for _, p := range review.WritePaths {
			if resolveLaneForPath(p, tax) == rec.ID || reviewGeometryOverlaps(p, tax.Trees[rec.ID]) || reviewGeometryOverlaps(p, rec.TreeGlobs) {
				return fmt.Errorf("independent live path claim %s held by %s covers %s", rec.ID, rec.Holder, p)
			}
		}
	}
	return nil
}

// ValidateCleanOverlapReview returns a private execution admission, never a
// dispatchable packet. Inspection is repeated under the actual native exclusion.
func ValidateCleanOverlapReview(ctx context.Context, opts CleanOverlapReviewOptions, request ReviewedOverlapRequest, lease *WriterLease) (*reviewedOverlapAdmission, error) {
	review := request.Review
	if review.Schema != CleanOverlapReviewSchema || !review.LocalOnly || review.Digest == "" || request.Approval.Digest != review.Digest || reviewDigest(review) != review.Digest {
		return nil, fmt.Errorf("missing, malformed or mismatched explicit review acceptance")
	}
	if err := checkReviewWriter(opts.Repo, lease); err != nil {
		return nil, err
	}
	opts.TargetSHA, opts.Reviewer, opts.Evidence = review.TargetSHA, review.Reviewer, review.Evidence
	opts.LockWaivers = append([]leaseref.SnapshotWaiver(nil), review.LockWaivers...)
	fresh, err := PrepareCleanOverlapReview(ctx, opts)
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(*fresh, review) {
		return nil, fmt.Errorf("review binding is stale or differs from native source state")
	}
	if err := checkReviewPathOwners(ctx, opts, review); err != nil {
		return nil, err
	}
	if err := checkReviewWriter(opts.Repo, lease); err != nil {
		return nil, err
	}
	return &reviewedOverlapAdmission{review: *fresh}, nil
}

func (e *PacketExecutor) checkReviewedResult(ctx context.Context, admission *reviewedOverlapAdmission, head string, lease *WriterLease) error {
	if err := checkReviewWriter(e.opts.Repo, lease); err != nil {
		return err
	}
	review := admission.review
	tree, err := rev(ctx, e.opts.Runner, e.opts.Repo, head+"^{tree}")
	if err != nil || tree != review.MergedTree {
		return fmt.Errorf("result tree differs from reviewed tree: %s (%v)", tree, err)
	}
	parents, err := reviewOutput(ctx, e.opts.Runner, e.opts.Repo, "rev-list", "--parents", "-n", "1", head)
	if err != nil || strings.TrimSpace(parents) != head+" "+review.LocalHead+" "+review.TargetSHA {
		return fmt.Errorf("result does not have exact ordered reviewed parents: %v", err)
	}
	return nil
}

// reviewTaxonomy accepts the repository's double-quoted string-array lane
// subset, not arbitrary TOML. Unsupported or malformed lane syntax refuses;
// the general-purpose tolerant taxonomy scanner is unsuitable for admission.
func reviewTaxonomy(repo string) (regionadmit.Taxonomy, error) {
	tax := regionadmit.Taxonomy{Exclusive: map[string]bool{}, Trees: map[string][]string{}}
	raw, err := os.ReadFile(filepath.Join(repo, "dos.toml"))
	if err != nil {
		return tax, err
	}
	section, key, value := "", "", ""
	seen := map[string]bool{}
	tables := map[string]bool{}
	for _, rawLine := range strings.Split(string(raw), "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Headers are checked before array continuation: an incomplete array cannot
		// swallow a later table and silently erase exclusive lane declarations.
		if key == "" && strings.HasPrefix(line, "[") {
			header := strings.TrimSpace(strings.SplitN(line, "#", 2)[0])
			if header == "[lanes]" || header == "[lanes.trees]" {
				if tables[header] {
					return tax, fmt.Errorf("duplicate review taxonomy table %s", header)
				}
				tables[header] = true
				section = strings.TrimSuffix(strings.TrimPrefix(header, "["), "]")
			} else {
				if strings.Contains(header, "lanes") {
					return tax, fmt.Errorf("unsupported review taxonomy table %q", header)
				}
				section = ""
			}
			continue
		}
		if section == "" {
			lhs, _, assignment := strings.Cut(line, "=")
			if assignment && strings.Contains(lhs, "lanes") {
				return tax, fmt.Errorf("unsupported dotted review taxonomy assignment %q", lhs)
			}
			continue
		}
		// Strip comments only outside JSON-compatible double-quoted strings.
		quoted, escaped := false, false
		end := len(line)
		for i := 0; i < len(line); i++ {
			ch := line[i]
			if escaped {
				escaped = false
				continue
			}
			if quoted && ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				quoted = !quoted
			}
			if !quoted && ch == '#' {
				end = i
				break
			}
		}
		if quoted || escaped {
			return tax, fmt.Errorf("unsupported multiline taxonomy string")
		}
		line = strings.TrimSpace(line[:end])
		if line == "" {
			continue
		}
		if key == "" {
			lhs, rhs, ok := strings.Cut(line, "=")
			if !ok {
				return tax, fmt.Errorf("malformed review taxonomy assignment %q", line)
			}
			key = strings.TrimSpace(lhs)
			if strings.HasPrefix(key, "\"") {
				var decoded string
				if json.Unmarshal([]byte(key), &decoded) != nil {
					return tax, fmt.Errorf("malformed taxonomy key")
				}
				key = decoded
			}
			if key == "" || strings.IndexFunc(key, func(r rune) bool {
				return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-')
			}) >= 0 {
				return tax, fmt.Errorf("unsupported taxonomy key %q", key)
			}
			if section == "lanes" && key != "exclusive" && key != "concurrent" && key != "autopick" {
				return tax, fmt.Errorf("unsupported lanes key %q", key)
			}
			if seen[section+"."+key] {
				return tax, fmt.Errorf("duplicate taxonomy key %s.%s", section, key)
			}
			seen[section+"."+key] = true
			value = strings.TrimSpace(rhs)
			if !strings.HasPrefix(value, "[") {
				return tax, fmt.Errorf("taxonomy value is not a string array")
			}
		} else {
			value += "\n" + line
		}
		if !strings.HasSuffix(value, "]") {
			continue
		}
		// TOML permits one trailing comma; remove only the array's final comma.
		inner := strings.TrimSpace(strings.TrimSuffix(value, "]"))
		inner = strings.TrimSpace(strings.TrimSuffix(inner, ","))
		var values []string
		if err := json.Unmarshal([]byte(inner+"]"), &values); err != nil {
			return tax, fmt.Errorf("malformed review taxonomy array %s.%s: %w", section, key, err)
		}
		for _, v := range values {
			if strings.TrimSpace(v) == "" {
				return tax, fmt.Errorf("empty review taxonomy value")
			}
			if section == "lanes.trees" {
				if err := reviewGlob(v); err != nil {
					return tax, err
				}
			}
		}
		if section == "lanes.trees" {
			tax.Trees[key] = values
		}
		if section == "lanes" && key == "exclusive" {
			for _, v := range values {
				tax.Exclusive[v] = true
			}
		}
		key, value = "", ""
	}
	if key != "" {
		return tax, fmt.Errorf("unterminated review taxonomy array")
	}
	return tax, nil
}

func reviewGlob(pattern string) error {
	// Native geometry supports global wildcards and literal subtree suffixes,
	// not a general recursive-glob language. Refuse unsupported recursive forms
	// rather than allowing a live claim to disappear between path segments.
	if strings.Contains(pattern, "**") && pattern != "**" && pattern != "**/*" {
		prefix := strings.TrimSuffix(pattern, "/**")
		if prefix == pattern || strings.ContainsAny(prefix, "*?[\\") {
			return fmt.Errorf("unsupported recursive review path geometry %q", pattern)
		}
	}
	if _, err := filepath.Match(pattern, ""); err != nil || strings.TrimSpace(pattern) == "" {
		return fmt.Errorf("malformed review path geometry %q", pattern)
	}
	return nil
}
