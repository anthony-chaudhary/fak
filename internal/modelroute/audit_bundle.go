package modelroute

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	IssueAuditBundleSchema         = "fak-issue-audit-bundle/v1"
	IssueAuditReviewEnvelopeSchema = "fak-issue-audit-review-envelope/v1"
	IssueAuditTrustedRole          = "TRUSTED_INSTRUCTION"
	IssueAuditUntrustedRole        = "UNTRUSTED_DATA"

	defaultAuditBundleMaxBytes        = 512 << 10
	defaultAuditBundleMaxBlobBytes    = 24 << 10
	defaultAuditBundleMaxContentBytes = 96 << 10
	defaultAuditBundleMaxComments     = 32
	defaultAuditBundleMaxPaths        = 128
	defaultAuditBundleMaxRefs         = 32
	maxAuditBundleScalarBytes         = 512
	maxAuditBundleClosingCommits      = 16
)

type IssueAuditComment struct {
	ID        string `json:"id"`
	URL       string `json:"url,omitempty"`
	Author    string `json:"author,omitempty"`
	CreatedAt string `json:"created_at,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty"`
	Body      string `json:"body"`
}

// IssueAuditClosingCommit is the independently gathered git object evidence
// for one closing commit. PatchSHA256 may be supplied by the fetch edge; when
// present the pure builder refuses if it differs from the exact Patch bytes.
type IssueAuditClosingCommit struct {
	SHA                string   `json:"sha"`
	FirstParentSHA     string   `json:"first_parent_sha"`
	TreeOID            string   `json:"tree_oid"`
	FirstParentTreeOID string   `json:"first_parent_tree_oid"`
	Patch              string   `json:"patch"`
	PatchSHA256        string   `json:"patch_sha256,omitempty"`
	ChangedPaths       []string `json:"changed_paths"`
}

type IssueAuditBundleOmission struct {
	Kind   string `json:"kind"`
	Ref    string `json:"ref"`
	Reason string `json:"reason"`
}

type IssueAuditBundleLimits struct {
	MaxBundleBytes  int `json:"max_bundle_bytes"`
	MaxBlobBytes    int `json:"max_blob_bytes"`
	MaxContentBytes int `json:"max_content_bytes"`
	MaxComments     int `json:"max_comments"`
	MaxChangedPaths int `json:"max_changed_paths"`
	MaxRefsPerKind  int `json:"max_refs_per_kind"`
}

type IssueAuditBundleOptions struct {
	Limits IssueAuditBundleLimits
}

type IssueAuditBundle struct {
	Schema            string                             `json:"schema"`
	Channel           string                             `json:"channel"`
	Complete          bool                               `json:"complete"`
	IncompleteReasons []IssueAuditBundleIncompleteReason `json:"incomplete_reasons,omitempty"`
	Issue             IssueAuditBundleIssue              `json:"issue"`
	Contract          IssueAuditBundleContract           `json:"contract"`
	Closure           IssueAuditBundleClosure            `json:"closure"`
	Evidence          IssueAuditBundleEvidence           `json:"evidence"`
	Blobs             []IssueAuditBundleBlob             `json:"blobs"`
	Manifest          IssueAuditBundleManifest           `json:"manifest"`
	BundleDigest      string                             `json:"bundle_digest"`
}

type IssueAuditBundleIssue struct {
	Number      int                       `json:"number"`
	URL         string                    `json:"url"`
	State       string                    `json:"state"`
	ClosedAt    string                    `json:"closed_at,omitempty"`
	TitleBlobID string                    `json:"title_blob_id"`
	BodyBlobID  string                    `json:"body_blob_id"`
	Comments    []IssueAuditBundleComment `json:"comments,omitempty"`
}

type IssueAuditBundleComment struct {
	ID         string `json:"id"`
	URL        string `json:"url,omitempty"`
	Author     string `json:"author,omitempty"`
	CreatedAt  string `json:"created_at,omitempty"`
	UpdatedAt  string `json:"updated_at,omitempty"`
	BodyBlobID string `json:"body_blob_id"`
}

type IssueAuditBundleContract struct {
	DoneConditionBlobID  string `json:"done_condition_blob_id,omitempty"`
	WitnessBlobID        string `json:"witness_blob_id,omitempty"`
	AcceptanceGateBlobID string `json:"acceptance_gate_blob_id,omitempty"`
}

type IssueAuditBundleClosure struct {
	PrimaryCommit string                          `json:"primary_commit"`
	Commits       []IssueAuditBundleClosingCommit `json:"commits"`
}

type IssueAuditBundleClosingCommit struct {
	SHA                string   `json:"sha"`
	FirstParentSHA     string   `json:"first_parent_sha"`
	TreeOID            string   `json:"tree_oid"`
	FirstParentTreeOID string   `json:"first_parent_tree_oid"`
	PatchBlobID        string   `json:"patch_blob_id"`
	PatchSHA256        string   `json:"patch_sha256"`
	ChangedPaths       []string `json:"changed_paths"`
}

type IssueAuditBundleEvidence struct {
	Tests         []EvidenceRef `json:"tests,omitempty"`
	CI            []EvidenceRef `json:"ci,omitempty"`
	DOS           []EvidenceRef `json:"dos,omitempty"`
	Artifacts     []EvidenceRef `json:"artifacts,omitempty"`
	PriorFindings []EvidenceRef `json:"prior_findings,omitempty"`
	Other         []EvidenceRef `json:"other,omitempty"`
}

type IssueAuditBundleBlob struct {
	ID            string `json:"id"`
	Kind          string `json:"kind"`
	Channel       string `json:"channel"`
	Encoding      string `json:"encoding"`
	SourceSHA256  string `json:"source_sha256"`
	ContentSHA256 string `json:"content_sha256"`
	OriginalBytes int    `json:"original_bytes"`
	IncludedBytes int    `json:"included_bytes"`
	Content       string `json:"content"`
}

type IssueAuditBundleManifest struct {
	Limits        IssueAuditBundleLimits       `json:"limits"`
	InputBytes    int                          `json:"input_bytes"`
	IncludedBytes int                          `json:"included_bytes"`
	Redactions    []IssueAuditBundleRedaction  `json:"redactions,omitempty"`
	Truncations   []IssueAuditBundleTruncation `json:"truncations,omitempty"`
	Omissions     []IssueAuditBundleOmission   `json:"omissions,omitempty"`
}

type IssueAuditBundleRedaction struct {
	BlobID string `json:"blob_id"`
	Rule   string `json:"rule"`
	Count  int    `json:"count"`
}

type IssueAuditBundleTruncation struct {
	Kind          string `json:"kind"`
	Ref           string `json:"ref"`
	OriginalBytes int    `json:"original_bytes"`
	IncludedBytes int    `json:"included_bytes"`
	SourceSHA256  string `json:"source_sha256"`
}

type IssueAuditBundleIncompleteReason string

const (
	IssueAuditIncompleteNotClosed       IssueAuditBundleIncompleteReason = "ISSUE_NOT_CLOSED"
	IssueAuditIncompleteClosedAtMissing IssueAuditBundleIncompleteReason = "CLOSED_AT_MISSING"
	IssueAuditIncompleteCommitMissing   IssueAuditBundleIncompleteReason = "CLOSING_COMMIT_MISSING"
	IssueAuditIncompleteParentMissing   IssueAuditBundleIncompleteReason = "FIRST_PARENT_MISSING"
	IssueAuditIncompleteTreeMissing     IssueAuditBundleIncompleteReason = "CLOSING_TREE_MISSING"
	IssueAuditIncompleteParentTree      IssueAuditBundleIncompleteReason = "FIRST_PARENT_TREE_MISSING"
	IssueAuditIncompletePatchMissing    IssueAuditBundleIncompleteReason = "CLOSING_PATCH_MISSING"
)

type IssueAuditBundleFailureReason string

const (
	IssueAuditBundleInvalidInput   IssueAuditBundleFailureReason = "BUNDLE_INPUT_INVALID"
	IssueAuditBundleIncomplete     IssueAuditBundleFailureReason = "BUNDLE_CLOSURE_INCOMPLETE"
	IssueAuditBundleDigestMismatch IssueAuditBundleFailureReason = "BUNDLE_DIGEST_MISMATCH"
	IssueAuditBundleLimitExceeded  IssueAuditBundleFailureReason = "BUNDLE_LIMIT_EXCEEDED"
)

type IssueAuditBundleError struct {
	Reason IssueAuditBundleFailureReason
	Detail string
	Bundle IssueAuditBundle
}

func (e *IssueAuditBundleError) Error() string {
	return fmt.Sprintf("modelroute: %s: %s", e.Reason, e.Detail)
}

func (e *IssueAuditBundleError) Is(target error) bool {
	_, ok := target.(*IssueAuditBundleError)
	return ok
}

func IsIssueAuditBundleFailure(err error) bool {
	var target *IssueAuditBundleError
	return errors.As(err, &target)
}

type IssueAuditReviewChannel struct {
	Role      string `json:"role"`
	MediaType string `json:"media_type"`
	SHA256    string `json:"sha256"`
	Content   string `json:"content"`
}

type IssueAuditReviewEnvelope struct {
	Schema             string                  `json:"schema"`
	TrustedInstruction IssueAuditReviewChannel `json:"trusted_instruction"`
	UntrustedEvidence  IssueAuditReviewChannel `json:"untrusted_evidence"`
}

type auditBundleBuilder struct {
	limits           IssueAuditBundleLimits
	contentRemaining int
	manifest         IssueAuditBundleManifest
	blobs            []IssueAuditBundleBlob
}

var auditBundleRedactors = []struct {
	name string
	re   *regexp.Regexp
	repl string
}{
	{"authorization-bearer", regexp.MustCompile(`(?i)(authorization\s*:\s*bearer\s+)[^\s"']+`), `${1}<REDACTED>`},
	{"github-token", regexp.MustCompile(`\bgh[pousr]_[A-Za-z0-9_]{16,}\b`), `<REDACTED_GITHUB_TOKEN>`},
	{"api-key", regexp.MustCompile(`\bsk-[A-Za-z0-9_-]{16,}\b`), `<REDACTED_API_KEY>`},
	{"secret-assignment", regexp.MustCompile(`(?i)\b(api[_-]?key|access[_-]?token|password|secret)(\s*[:=]\s*)([^\s,;]+)`), `${1}${2}<REDACTED>`},
	{"private-companion-path", regexp.MustCompile(`(?i)(?:[A-Za-z]:)?[^\s"']*fak-private[^\s"']*`), `<REDACTED_PRIVATE_PATH>`},
}

func defaultIssueAuditBundleLimits() IssueAuditBundleLimits {
	return IssueAuditBundleLimits{
		MaxBundleBytes: defaultAuditBundleMaxBytes, MaxBlobBytes: defaultAuditBundleMaxBlobBytes,
		MaxContentBytes: defaultAuditBundleMaxContentBytes, MaxComments: defaultAuditBundleMaxComments,
		MaxChangedPaths: defaultAuditBundleMaxPaths, MaxRefsPerKind: defaultAuditBundleMaxRefs,
	}
}

func normalizeIssueAuditBundleLimits(in IssueAuditBundleLimits) (IssueAuditBundleLimits, error) {
	out := defaultIssueAuditBundleLimits()
	for src, dst := range map[*int]*int{
		&in.MaxBundleBytes: &out.MaxBundleBytes, &in.MaxBlobBytes: &out.MaxBlobBytes,
		&in.MaxContentBytes: &out.MaxContentBytes, &in.MaxComments: &out.MaxComments,
		&in.MaxChangedPaths: &out.MaxChangedPaths, &in.MaxRefsPerKind: &out.MaxRefsPerKind,
	} {
		if *src < 0 {
			return IssueAuditBundleLimits{}, fmt.Errorf("audit bundle limits cannot be negative")
		}
		if *src > 0 {
			*dst = *src
		}
	}
	if out.MaxBundleBytes > defaultAuditBundleMaxBytes || out.MaxBlobBytes > defaultAuditBundleMaxBlobBytes || out.MaxContentBytes > defaultAuditBundleMaxContentBytes || out.MaxComments > defaultAuditBundleMaxComments || out.MaxChangedPaths > defaultAuditBundleMaxPaths || out.MaxRefsPerKind > defaultAuditBundleMaxRefs {
		return IssueAuditBundleLimits{}, fmt.Errorf("audit bundle limits cannot exceed hard v1 bounds")
	}
	if out.MaxBlobBytes > out.MaxContentBytes {
		out.MaxBlobBytes = out.MaxContentBytes
	}
	return out, nil
}

// BuildIssueAuditBundle constructs the complete credential-free evidence
// object. It never interprets evidence text as instructions and returns the
// partially built, explicitly incomplete bundle with every fail-closed error.
func BuildIssueAuditBundle(evidence IssueAuditEvidence, options IssueAuditBundleOptions) (IssueAuditBundle, error) {
	limits, err := normalizeIssueAuditBundleLimits(options.Limits)
	if err != nil {
		return IssueAuditBundle{}, &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: err.Error()}
	}
	builder := &auditBundleBuilder{limits: limits, contentRemaining: limits.MaxContentBytes}
	builder.manifest.Limits = limits
	bundle := IssueAuditBundle{
		Schema: IssueAuditBundleSchema, Channel: IssueAuditUntrustedRole,
		Issue: IssueAuditBundleIssue{Number: evidence.IssueNumber, State: strings.ToUpper(strings.TrimSpace(evidence.State))},
	}
	bundle.Issue.URL = builder.scalar("issue", "url", evidence.IssueURL)
	bundle.Issue.ClosedAt = builder.scalar("issue", "closed_at", evidence.ClosedAt)
	if evidence.IssueNumber <= 0 || bundle.Issue.URL == "" {
		return bundle, &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "issue number and URL are required", Bundle: bundle}
	}

	commits := append([]IssueAuditClosingCommit(nil), evidence.ClosingCommits...)
	if len(commits) > maxAuditBundleClosingCommits {
		return bundle, &IssueAuditBundleError{Reason: IssueAuditBundleLimitExceeded, Detail: fmt.Sprintf("closing commit count %d exceeds %d", len(commits), maxAuditBundleClosingCommits), Bundle: bundle}
	}
	sort.Slice(commits, func(i, j int) bool { return commits[i].SHA < commits[j].SHA })
	bundle.Closure.PrimaryCommit = strings.TrimSpace(evidence.CommitSHA)
	for i, commit := range commits {
		if commit.SHA == bundle.Closure.PrimaryCommit && evidence.Diff != "" && evidence.Diff != commit.Patch {
			return bundle, &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: "legacy primary diff does not match exact closing patch", Bundle: bundle}
		}
		built, buildErr := builder.closingCommit(i, commit)
		if buildErr != nil {
			bundle.Blobs, bundle.Manifest = builder.finish()
			bundle.BundleDigest = bundle.recomputeDigest()
			buildErr.Bundle = bundle
			return bundle, buildErr
		}
		bundle.Closure.Commits = append(bundle.Closure.Commits, built)
	}

	bundle.Issue.TitleBlobID = builder.blob("issue-title", "issue-title", evidence.Title).ID
	bundle.Issue.BodyBlobID = builder.blob("issue-body", "issue-body", evidence.Body).ID
	for _, section := range []struct {
		name string
		dst  *string
	}{
		{"done condition", &bundle.Contract.DoneConditionBlobID},
		{"witness", &bundle.Contract.WitnessBlobID},
		{"acceptance gate", &bundle.Contract.AcceptanceGateBlobID},
	} {
		content := extractIssueAuditContractSection(evidence.Body, section.name)
		if content == "" {
			builder.omit("contract", section.name, "section not present")
			continue
		}
		*section.dst = builder.blob("contract-"+strings.ReplaceAll(section.name, " ", "-"), "issue-contract", content).ID
	}

	comments := append([]IssueAuditComment(nil), evidence.Comments...)
	sort.SliceStable(comments, func(i, j int) bool {
		if comments[i].CreatedAt != comments[j].CreatedAt {
			return comments[i].CreatedAt < comments[j].CreatedAt
		}
		return comments[i].ID < comments[j].ID
	})
	if len(comments) > limits.MaxComments {
		builder.omit("comment", fmt.Sprintf("count:%d", len(comments)-limits.MaxComments), "comment count limit")
		comments = comments[:limits.MaxComments]
	}
	for i, comment := range comments {
		id := builder.scalar("comment", fmt.Sprintf("%d:id", i), comment.ID)
		if id == "" {
			id = fmt.Sprintf("comment-%04d", i+1)
		}
		blob := builder.blob(fmt.Sprintf("comment-%04d-%s", i+1, safeAuditBundleID(id)), "issue-comment", comment.Body)
		bundle.Issue.Comments = append(bundle.Issue.Comments, IssueAuditBundleComment{
			ID: id, URL: builder.scalar("comment", id+":url", comment.URL), Author: builder.commentAuthor(id, comment.Author),
			CreatedAt: builder.scalar("comment", id+":created_at", comment.CreatedAt), UpdatedAt: builder.scalar("comment", id+":updated_at", comment.UpdatedAt), BodyBlobID: blob.ID,
		})
	}

	bundle.Evidence.Tests = builder.refs("tests", evidence.Tests)
	bundle.Evidence.CI = builder.refs("ci", evidence.CI)
	bundle.Evidence.DOS = builder.refs("dos", evidence.DOS)
	bundle.Evidence.Artifacts = builder.refs("artifacts", evidence.Artifacts)
	bundle.Evidence.PriorFindings = builder.refs("prior-findings", evidence.PriorFindings)
	bundle.Evidence.Other = builder.refs("other", evidence.Evidence)
	omissions := evidence.Omissions
	if len(omissions) > limits.MaxRefsPerKind {
		builder.omit("input-omission", fmt.Sprintf("count:%d", len(omissions)-limits.MaxRefsPerKind), "omission manifest count limit")
		omissions = omissions[:limits.MaxRefsPerKind]
	}
	for _, omission := range omissions {
		builder.omit(omission.Kind, omission.Ref, omission.Reason)
	}

	bundle.IncompleteReasons = auditBundleIncompleteReasons(bundle)
	bundle.Complete = len(bundle.IncompleteReasons) == 0
	bundle.Blobs, bundle.Manifest = builder.finish()
	bundle.BundleDigest = bundle.recomputeDigest()
	encoded, marshalErr := json.Marshal(bundle)
	if marshalErr != nil {
		return bundle, &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: marshalErr.Error(), Bundle: bundle}
	}
	if len(encoded) > limits.MaxBundleBytes {
		return bundle, &IssueAuditBundleError{Reason: IssueAuditBundleLimitExceeded, Detail: fmt.Sprintf("bundle is %d bytes, limit %d", len(encoded), limits.MaxBundleBytes), Bundle: bundle}
	}
	if !bundle.Complete {
		return bundle, &IssueAuditBundleError{Reason: IssueAuditBundleIncomplete, Detail: fmt.Sprintf("missing closure evidence: %s", joinAuditIncompleteReasons(bundle.IncompleteReasons)), Bundle: bundle}
	}
	if err := bundle.Verify(); err != nil {
		return bundle, err
	}
	return bundle, nil
}

func (b *auditBundleBuilder) closingCommit(index int, commit IssueAuditClosingCommit) (IssueAuditBundleClosingCommit, *IssueAuditBundleError) {
	sha := strings.TrimSpace(commit.SHA)
	patchDigest := hashString(commit.Patch)
	if expected := strings.TrimSpace(commit.PatchSHA256); expected != "" && expected != patchDigest {
		return IssueAuditBundleClosingCommit{}, &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: fmt.Sprintf("closing commit %s patch digest stamped %s, recomputed %s", sha, expected, patchDigest)}
	}
	patchBlob := b.blob(fmt.Sprintf("closing-patch-%04d", index+1), "first-parent-patch", commit.Patch)
	paths := b.paths(commit.ChangedPaths, sha)
	return IssueAuditBundleClosingCommit{
		SHA: sha, FirstParentSHA: strings.TrimSpace(commit.FirstParentSHA), TreeOID: strings.TrimSpace(commit.TreeOID),
		FirstParentTreeOID: strings.TrimSpace(commit.FirstParentTreeOID), PatchBlobID: patchBlob.ID, PatchSHA256: patchDigest, ChangedPaths: paths,
	}, nil
}

func (b *auditBundleBuilder) blob(id, kind, raw string) IssueAuditBundleBlob {
	id = safeAuditBundleID(id)
	b.manifest.InputBytes += len(raw)
	redacted := raw
	for _, rule := range auditBundleRedactors {
		matches := rule.re.FindAllStringIndex(redacted, -1)
		if len(matches) == 0 {
			continue
		}
		redacted = rule.re.ReplaceAllString(redacted, rule.repl)
		b.manifest.Redactions = append(b.manifest.Redactions, IssueAuditBundleRedaction{BlobID: id, Rule: rule.name, Count: len(matches)})
	}
	limit := b.limits.MaxBlobBytes
	if b.contentRemaining < limit {
		limit = b.contentRemaining
	}
	included := truncateAuditBundleUTF8(redacted, limit)
	if len(included) < len(redacted) {
		b.manifest.Truncations = append(b.manifest.Truncations, IssueAuditBundleTruncation{
			Kind: kind, Ref: id, OriginalBytes: len(raw), IncludedBytes: len(included), SourceSHA256: hashString(raw),
		})
	}
	b.contentRemaining -= len(included)
	b.manifest.IncludedBytes += len(included)
	blob := IssueAuditBundleBlob{
		ID: id, Kind: kind, Channel: IssueAuditUntrustedRole, Encoding: "utf-8",
		SourceSHA256: hashString(raw), ContentSHA256: hashString(included), OriginalBytes: len(raw), IncludedBytes: len(included), Content: included,
	}
	b.blobs = append(b.blobs, blob)
	return blob
}

func (b *auditBundleBuilder) scalar(kind, ref, raw string) string {
	redacted := raw
	for _, rule := range auditBundleRedactors {
		matches := rule.re.FindAllStringIndex(redacted, -1)
		if len(matches) == 0 {
			continue
		}
		redacted = rule.re.ReplaceAllString(redacted, rule.repl)
		b.manifest.Redactions = append(b.manifest.Redactions, IssueAuditBundleRedaction{BlobID: kind + ":" + safeAuditBundleID(ref), Rule: rule.name, Count: len(matches)})
	}
	included := truncateAuditBundleUTF8(strings.TrimSpace(redacted), maxAuditBundleScalarBytes)
	if len(included) < len(strings.TrimSpace(redacted)) {
		b.manifest.Truncations = append(b.manifest.Truncations, IssueAuditBundleTruncation{Kind: kind, Ref: safeAuditBundleID(ref), OriginalBytes: len(raw), IncludedBytes: len(included), SourceSHA256: hashString(raw)})
	}
	return included
}

func (b *auditBundleBuilder) commentAuthor(id, raw string) string {
	if strings.TrimSpace(raw) == "" {
		return ""
	}
	b.manifest.Redactions = append(b.manifest.Redactions, IssueAuditBundleRedaction{BlobID: "comment:" + safeAuditBundleID(id) + ":author", Rule: "github-comment-author", Count: 1})
	return "<REDACTED_GITHUB_USER>"
}

func (b *auditBundleBuilder) paths(paths []string, commit string) []string {
	seen := map[string]bool{}
	var normalized []string
	invalidCount := 0
	for _, path := range paths {
		path = strings.ReplaceAll(strings.TrimSpace(path), "\\", "/")
		if path == "" || strings.Contains(strings.ToLower(path), "fak-private") || strings.HasPrefix(path, "../") || strings.Contains(path, ":/") {
			invalidCount++
			continue
		}
		if seen[path] {
			continue
		}
		seen[path] = true
		normalized = append(normalized, path)
	}
	if invalidCount > 0 {
		b.omit("changed-path", fmt.Sprintf("count:%d", invalidCount), "private, empty, or out-of-tree paths refused")
	}
	sort.Strings(normalized)
	if len(normalized) > b.limits.MaxChangedPaths {
		b.omit("changed-path", fmt.Sprintf("count:%d", len(normalized)-b.limits.MaxChangedPaths), "changed path count limit for "+commit)
		normalized = normalized[:b.limits.MaxChangedPaths]
	}
	out := make([]string, 0, len(normalized))
	for _, path := range normalized {
		if path = b.scalar("changed-path", hashString(path), path); path != "" {
			out = append(out, path)
		}
	}
	return out
}

func (b *auditBundleBuilder) refs(kind string, refs []EvidenceRef) []EvidenceRef {
	rawSeen := map[string]bool{}
	unique := make([]EvidenceRef, 0, len(refs))
	for _, ref := range refs {
		key := ref.Kind + "\x00" + ref.Ref + "\x00" + ref.SHA256
		if !rawSeen[key] {
			rawSeen[key] = true
			unique = append(unique, ref)
		}
	}
	sort.Slice(unique, func(i, j int) bool {
		if unique[i].Kind != unique[j].Kind {
			return unique[i].Kind < unique[j].Kind
		}
		if unique[i].Ref != unique[j].Ref {
			return unique[i].Ref < unique[j].Ref
		}
		return unique[i].SHA256 < unique[j].SHA256
	})
	if len(unique) > b.limits.MaxRefsPerKind {
		b.omit(kind, fmt.Sprintf("count:%d", len(unique)-b.limits.MaxRefsPerKind), "evidence ref count limit")
		unique = unique[:b.limits.MaxRefsPerKind]
	}
	seen := map[string]bool{}
	var out []EvidenceRef
	for _, ref := range unique {
		row := EvidenceRef{Kind: b.scalar("evidence-kind", kind, ref.Kind), Ref: b.scalar("evidence-ref", kind, ref.Ref), SHA256: strings.TrimSpace(ref.SHA256)}
		if row.Kind == "" || row.Ref == "" {
			b.omit(kind, hashString(ref.Ref), "empty evidence kind or ref")
			continue
		}
		if row.SHA256 != "" && !validSHA256Digest(row.SHA256) {
			b.omit(kind, hashString(ref.Ref), "invalid evidence digest")
			continue
		}
		key := row.Kind + "\x00" + row.Ref + "\x00" + row.SHA256
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		if out[i].Ref != out[j].Ref {
			return out[i].Ref < out[j].Ref
		}
		return out[i].SHA256 < out[j].SHA256
	})
	return out
}

func (b *auditBundleBuilder) omit(kind, ref, reason string) {
	b.manifest.Omissions = append(b.manifest.Omissions, IssueAuditBundleOmission{
		Kind: truncateAuditBundleUTF8(strings.TrimSpace(kind), 64), Ref: truncateAuditBundleUTF8(strings.TrimSpace(ref), 128), Reason: truncateAuditBundleUTF8(strings.TrimSpace(reason), 256),
	})
}

func (b *auditBundleBuilder) finish() ([]IssueAuditBundleBlob, IssueAuditBundleManifest) {
	sort.Slice(b.blobs, func(i, j int) bool { return b.blobs[i].ID < b.blobs[j].ID })
	sort.Slice(b.manifest.Redactions, func(i, j int) bool {
		if b.manifest.Redactions[i].BlobID != b.manifest.Redactions[j].BlobID {
			return b.manifest.Redactions[i].BlobID < b.manifest.Redactions[j].BlobID
		}
		return b.manifest.Redactions[i].Rule < b.manifest.Redactions[j].Rule
	})
	sort.Slice(b.manifest.Truncations, func(i, j int) bool {
		if b.manifest.Truncations[i].Kind != b.manifest.Truncations[j].Kind {
			return b.manifest.Truncations[i].Kind < b.manifest.Truncations[j].Kind
		}
		return b.manifest.Truncations[i].Ref < b.manifest.Truncations[j].Ref
	})
	sort.Slice(b.manifest.Omissions, func(i, j int) bool {
		if b.manifest.Omissions[i].Kind != b.manifest.Omissions[j].Kind {
			return b.manifest.Omissions[i].Kind < b.manifest.Omissions[j].Kind
		}
		if b.manifest.Omissions[i].Ref != b.manifest.Omissions[j].Ref {
			return b.manifest.Omissions[i].Ref < b.manifest.Omissions[j].Ref
		}
		return b.manifest.Omissions[i].Reason < b.manifest.Omissions[j].Reason
	})
	return append([]IssueAuditBundleBlob(nil), b.blobs...), b.manifest
}

func auditBundleIncompleteReasons(bundle IssueAuditBundle) []IssueAuditBundleIncompleteReason {
	var reasons []IssueAuditBundleIncompleteReason
	if bundle.Issue.State != "CLOSED" {
		reasons = append(reasons, IssueAuditIncompleteNotClosed)
	}
	if bundle.Issue.ClosedAt == "" {
		reasons = append(reasons, IssueAuditIncompleteClosedAtMissing)
	}
	if bundle.Closure.PrimaryCommit == "" || len(bundle.Closure.Commits) == 0 {
		reasons = append(reasons, IssueAuditIncompleteCommitMissing)
		return reasons
	}
	primaryFound := false
	for _, commit := range bundle.Closure.Commits {
		if commit.SHA == bundle.Closure.PrimaryCommit {
			primaryFound = true
		}
		if commit.FirstParentSHA == "" {
			reasons = append(reasons, IssueAuditIncompleteParentMissing)
		}
		if commit.TreeOID == "" {
			reasons = append(reasons, IssueAuditIncompleteTreeMissing)
		}
		if commit.FirstParentTreeOID == "" {
			reasons = append(reasons, IssueAuditIncompleteParentTree)
		}
		if commit.PatchBlobID == "" || commit.PatchSHA256 == hashString("") {
			reasons = append(reasons, IssueAuditIncompletePatchMissing)
		}
	}
	if !primaryFound {
		reasons = append(reasons, IssueAuditIncompleteCommitMissing)
	}
	return uniqueAuditIncompleteReasons(reasons)
}

func uniqueAuditIncompleteReasons(in []IssueAuditBundleIncompleteReason) []IssueAuditBundleIncompleteReason {
	seen := map[IssueAuditBundleIncompleteReason]bool{}
	var out []IssueAuditBundleIncompleteReason
	for _, reason := range in {
		if !seen[reason] {
			seen[reason] = true
			out = append(out, reason)
		}
	}
	return out
}

func joinAuditIncompleteReasons(in []IssueAuditBundleIncompleteReason) string {
	parts := make([]string, len(in))
	for i, reason := range in {
		parts[i] = string(reason)
	}
	return strings.Join(parts, ",")
}

func (bundle IssueAuditBundle) recomputeDigest() string {
	bundle.BundleDigest = ""
	return hashJSON(bundle)
}

func (bundle IssueAuditBundle) Verify() error {
	if bundle.Schema != IssueAuditBundleSchema || bundle.Channel != IssueAuditUntrustedRole {
		return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "bundle schema or channel is invalid", Bundle: bundle}
	}
	if bundle.Complete != (len(bundle.IncompleteReasons) == 0) {
		return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "bundle completeness contradicts incomplete reasons", Bundle: bundle}
	}
	if want := auditBundleIncompleteReasons(bundle); !sameAuditIncompleteReasons(bundle.IncompleteReasons, want) {
		return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "bundle closure completeness does not match its evidence", Bundle: bundle}
	}
	limits, err := normalizeIssueAuditBundleLimits(bundle.Manifest.Limits)
	if err != nil || limits != bundle.Manifest.Limits {
		return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "bundle limits are invalid or non-canonical", Bundle: bundle}
	}
	blobs := map[string]IssueAuditBundleBlob{}
	includedBytes := 0
	for _, blob := range bundle.Blobs {
		if blob.ID == "" || blobs[blob.ID].ID != "" || blob.Channel != IssueAuditUntrustedRole || blob.Encoding != "utf-8" {
			return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "bundle blob identity, channel, or encoding is invalid", Bundle: bundle}
		}
		if !validSHA256Digest(blob.SourceSHA256) || blob.ContentSHA256 != hashString(blob.Content) || blob.IncludedBytes != len(blob.Content) || blob.OriginalBytes < 0 || blob.IncludedBytes < 0 {
			return &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: "bundle blob digest or byte count mismatch for " + blob.ID, Bundle: bundle}
		}
		if !utf8.ValidString(blob.Content) || blob.IncludedBytes > limits.MaxBlobBytes {
			return &IssueAuditBundleError{Reason: IssueAuditBundleLimitExceeded, Detail: "bundle blob exceeds its content bound for " + blob.ID, Bundle: bundle}
		}
		includedBytes += blob.IncludedBytes
		blobs[blob.ID] = blob
	}
	if includedBytes != bundle.Manifest.IncludedBytes || includedBytes > limits.MaxContentBytes || bundle.Manifest.InputBytes < 0 {
		return &IssueAuditBundleError{Reason: IssueAuditBundleLimitExceeded, Detail: "bundle aggregate content bytes do not match the bounded manifest", Bundle: bundle}
	}
	commitSeen := map[string]bool{}
	for _, commit := range bundle.Closure.Commits {
		blob, ok := blobs[commit.PatchBlobID]
		if commit.SHA == "" || commitSeen[commit.SHA] || !ok || blob.Kind != "first-parent-patch" || commit.PatchSHA256 != blob.SourceSHA256 {
			return &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: "closing patch digest does not bind its blob for " + commit.SHA, Bundle: bundle}
		}
		commitSeen[commit.SHA] = true
		if len(commit.ChangedPaths) > limits.MaxChangedPaths || !sort.StringsAreSorted(commit.ChangedPaths) {
			return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "closing changed paths are unbounded or unstable for " + commit.SHA, Bundle: bundle}
		}
		for _, path := range commit.ChangedPaths {
			if path == "" || strings.Contains(strings.ToLower(path), "fak-private") || strings.HasPrefix(path, "../") || strings.Contains(path, ":/") {
				return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "closing changed path is private or out of tree", Bundle: bundle}
			}
		}
	}
	for _, ref := range []string{bundle.Issue.TitleBlobID, bundle.Issue.BodyBlobID, bundle.Contract.DoneConditionBlobID, bundle.Contract.WitnessBlobID, bundle.Contract.AcceptanceGateBlobID} {
		if ref != "" {
			if _, ok := blobs[ref]; !ok {
				return &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: "bundle references missing blob " + ref, Bundle: bundle}
			}
		}
	}
	for _, comment := range bundle.Issue.Comments {
		if _, ok := blobs[comment.BodyBlobID]; !ok {
			return &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: "comment references missing blob " + comment.BodyBlobID, Bundle: bundle}
		}
	}
	if bundle.BundleDigest != bundle.recomputeDigest() {
		return &IssueAuditBundleError{Reason: IssueAuditBundleDigestMismatch, Detail: "bundle digest mismatch", Bundle: bundle}
	}
	for _, refs := range [][]EvidenceRef{bundle.Evidence.Tests, bundle.Evidence.CI, bundle.Evidence.DOS, bundle.Evidence.Artifacts, bundle.Evidence.PriorFindings, bundle.Evidence.Other} {
		if len(refs) > limits.MaxRefsPerKind {
			return &IssueAuditBundleError{Reason: IssueAuditBundleLimitExceeded, Detail: "bundle evidence refs exceed their bound", Bundle: bundle}
		}
		for _, ref := range refs {
			if ref.Kind == "" || ref.Ref == "" || (ref.SHA256 != "" && !validSHA256Digest(ref.SHA256)) {
				return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: "bundle evidence ref is invalid", Bundle: bundle}
			}
		}
	}
	encoded, err := json.Marshal(bundle)
	if err != nil {
		return &IssueAuditBundleError{Reason: IssueAuditBundleInvalidInput, Detail: err.Error(), Bundle: bundle}
	}
	if len(encoded) > bundle.Manifest.Limits.MaxBundleBytes {
		return &IssueAuditBundleError{Reason: IssueAuditBundleLimitExceeded, Detail: "verified bundle exceeds stamped byte limit", Bundle: bundle}
	}
	return nil
}

func MarshalIssueAuditBundle(bundle IssueAuditBundle) ([]byte, error) {
	if err := bundle.Verify(); err != nil {
		return nil, err
	}
	return json.Marshal(bundle)
}

func IssueAuditContentDigest(content string) string {
	return hashString(content)
}

func ParseIssueAuditBundle(raw []byte) (IssueAuditBundle, error) {
	var bundle IssueAuditBundle
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&bundle); err != nil {
		return IssueAuditBundle{}, fmt.Errorf("modelroute: parse audit bundle: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return IssueAuditBundle{}, fmt.Errorf("modelroute: parse audit bundle: multiple JSON values")
		}
		return IssueAuditBundle{}, fmt.Errorf("modelroute: parse audit bundle trailing data: %w", err)
	}
	if err := bundle.Verify(); err != nil {
		return IssueAuditBundle{}, err
	}
	return bundle, nil
}

func auditBundlePrimaryCommit(bundle IssueAuditBundle) (IssueAuditBundleClosingCommit, bool) {
	for _, commit := range bundle.Closure.Commits {
		if commit.SHA == bundle.Closure.PrimaryCommit {
			return commit, true
		}
	}
	return IssueAuditBundleClosingCommit{}, false
}

func auditBundleBlobContent(bundle IssueAuditBundle, id string) string {
	for _, blob := range bundle.Blobs {
		if blob.ID == id {
			return blob.Content
		}
	}
	return ""
}

func NewIssueAuditReviewRequest(issue int, subjectDigest string, bundle IssueAuditBundle) (IssueAuditReviewRequest, error) {
	if err := bundle.Verify(); err != nil {
		return IssueAuditReviewRequest{}, err
	}
	if subjectDigest != bundle.BundleDigest {
		return IssueAuditReviewRequest{}, fmt.Errorf("modelroute: issue audit subject digest must equal the exact bundle digest")
	}
	if issue != bundle.Issue.Number {
		return IssueAuditReviewRequest{}, fmt.Errorf("modelroute: audit bundle issue %d, want %d", bundle.Issue.Number, issue)
	}
	bundleJSON, err := json.Marshal(bundle)
	if err != nil {
		return IssueAuditReviewRequest{}, err
	}
	trusted := IssueAuditReviewChannel{Role: IssueAuditTrustedRole, MediaType: "text/plain", Content: CrossAuditSystemPrompt}
	trusted.SHA256 = hashString(trusted.Content)
	untrusted := IssueAuditReviewChannel{Role: IssueAuditUntrustedRole, MediaType: "application/json", Content: string(bundleJSON)}
	untrusted.SHA256 = hashString(untrusted.Content)
	envelope := IssueAuditReviewEnvelope{Schema: IssueAuditReviewEnvelopeSchema, TrustedInstruction: trusted, UntrustedEvidence: untrusted}
	promptJSON, err := json.Marshal(envelope)
	if err != nil {
		return IssueAuditReviewRequest{}, err
	}
	req := IssueAuditReviewRequest{
		IssueNumber: issue, SubjectDigest: subjectDigest, BundleDigest: bundle.BundleDigest,
		PromptVersion: CrossAuditPromptVersion, TrustedInstruction: trusted, UntrustedEvidence: untrusted,
		Prompt: string(promptJSON),
	}
	req.PromptDigest = hashString(req.Prompt)
	if err := req.Verify(); err != nil {
		return IssueAuditReviewRequest{}, err
	}
	return req, nil
}

func (req IssueAuditReviewRequest) Verify() error {
	if req.IssueNumber <= 0 || !validSHA256Digest(req.SubjectDigest) || !validSHA256Digest(req.BundleDigest) || req.PromptVersion != CrossAuditPromptVersion {
		return fmt.Errorf("modelroute: issue audit review request binding is incomplete")
	}
	if req.TrustedInstruction.Role != IssueAuditTrustedRole || req.TrustedInstruction.MediaType != "text/plain" || req.TrustedInstruction.Content != CrossAuditSystemPrompt || req.TrustedInstruction.SHA256 != hashString(req.TrustedInstruction.Content) {
		return fmt.Errorf("modelroute: trusted issue audit instruction channel is invalid")
	}
	if req.UntrustedEvidence.Role != IssueAuditUntrustedRole || req.UntrustedEvidence.MediaType != "application/json" || req.UntrustedEvidence.SHA256 != hashString(req.UntrustedEvidence.Content) {
		return fmt.Errorf("modelroute: untrusted issue audit evidence channel is invalid")
	}
	bundle, err := ParseIssueAuditBundle([]byte(req.UntrustedEvidence.Content))
	if err != nil {
		return fmt.Errorf("modelroute: issue audit bundle channel is invalid: %w", err)
	}
	if bundle.BundleDigest != req.BundleDigest {
		return fmt.Errorf("modelroute: issue audit bundle channel digest mismatch")
	}
	if req.SubjectDigest != bundle.BundleDigest {
		return fmt.Errorf("modelroute: issue audit subject digest does not bind the bundle")
	}
	if req.IssueNumber != bundle.Issue.Number {
		return fmt.Errorf("modelroute: issue audit request number does not bind the bundle")
	}
	wantPrompt, err := json.Marshal(IssueAuditReviewEnvelope{Schema: IssueAuditReviewEnvelopeSchema, TrustedInstruction: req.TrustedInstruction, UntrustedEvidence: req.UntrustedEvidence})
	if err != nil {
		return err
	}
	if req.Prompt != string(wantPrompt) || req.PromptDigest != hashString(req.Prompt) {
		return fmt.Errorf("modelroute: issue audit review envelope digest mismatch")
	}
	return nil
}

func sameAuditIncompleteReasons(a, b []IssueAuditBundleIncompleteReason) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func extractIssueAuditContractSection(body, name string) string {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	want := strings.ToLower(strings.TrimSpace(name))
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		heading := strings.ToLower(strings.TrimSpace(strings.TrimLeft(trimmed, "#")))
		if strings.HasPrefix(trimmed, "#") && heading == want {
			var section []string
			for _, next := range lines[i+1:] {
				if strings.HasPrefix(strings.TrimSpace(next), "#") {
					break
				}
				section = append(section, next)
			}
			return strings.TrimSpace(strings.Join(section, "\n"))
		}
		prefix := want + ":"
		if strings.HasPrefix(strings.ToLower(trimmed), prefix) {
			return strings.TrimSpace(trimmed[len(prefix):])
		}
	}
	return ""
}

func truncateAuditBundleUTF8(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	b := []byte(value[:limit])
	for len(b) > 0 && !utf8.Valid(b) {
		b = b[:len(b)-1]
	}
	return string(b)
}

func safeAuditBundleID(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
		if b.Len() >= 96 {
			break
		}
	}
	value = strings.Trim(b.String(), "-")
	if value == "" {
		return "unnamed"
	}
	return value
}
