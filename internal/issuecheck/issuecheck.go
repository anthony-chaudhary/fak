package issuecheck

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
)

const (
	// ProjectionSchema identifies the canonical issue material covered by a review.
	ProjectionSchema = "fak.issuecheck.issue-projection.v1"
	// BindingSchema identifies the compact projection witness carried by a review.
	BindingSchema = "fak.issuecheck.issue-binding.v1"
	// ReviewSchema identifies the closed agent-output contract.
	ReviewSchema = "fak.issuecheck.review.v1"
	// CommentMarker is the stable first-line key for the one managed comment.
	CommentMarker = "<!-- fak-issuecheck: top-five -->"
	// CommentPayloadPrefix identifies the hidden base64url Review used for
	// strict mutation readback. It is separate from CommentMarker so only the
	// stable first-line marker participates in managed-comment discovery.
	CommentPayloadPrefix = "<!-- fak-issuecheck-review: "

	// RequiredReviewRows is the closed Top-5 response cardinality.
	RequiredReviewRows = 5
	// MaxReviewJSONBytes bounds the canonical hidden review payload.
	MaxReviewJSONBytes = 32 * 1024
	// MaxReviewFieldBytes bounds each issue-specific conclusion/action/gap field.
	MaxReviewFieldBytes = 1024
	// MaxEvidenceRefsPerRow bounds evidence fan-out for one selected check.
	MaxEvidenceRefsPerRow = 8
	// MaxEvidenceRefBytes bounds each evidence locator.
	MaxEvidenceRefBytes = 512
	// MaxReviewerVersionBytes bounds self-reported reviewer metadata.
	MaxReviewerVersionBytes = 128
	// MaxRenderedCommentBytes stays conservatively below GitHub's comment-body limit.
	MaxRenderedCommentBytes = 60 * 1024
)

// Closed evidence status tokens accepted by ValidateReview.
const (
	// EvidenceSupported indicates that the assessment is fully backed by references.
	EvidenceSupported = "supported"
	// EvidencePartial indicates that some evidence exists but an explicit gap remains.
	EvidencePartial = "partial"
	// EvidenceGap indicates that no supporting evidence exists and names the missing proof.
	EvidenceGap = "gap"
)

// Managed comment plan action tokens returned by ChooseCommentAction.
const (
	// ActionCreate instructs the caller to create a new managed comment.
	ActionCreate = "create"
	// ActionUpdate instructs the caller to update an existing managed comment.
	ActionUpdate = "update"
	// ActionNoop indicates that the existing managed comment is already up to date.
	ActionNoop = "noop"
	// ActionRefuse indicates that comment modification is refused due to ambiguity.
	ActionRefuse = "refuse"
)

var (
	digestRE          = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	reviewerVersionRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/@+:-]*$`)
)

// Issue is the GitHub issue material whose semantic meaning a review covers.
// UpdatedAt and comments are intentionally absent: commenting must not stale the
// review it just wrote.
type Issue struct {
	Number int      `json:"number"`
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	Labels []string `json:"labels,omitempty"`
}

// Projection is the canonical, stable issue representation hashed by IssueDigest.
type Projection struct {
	Schema string   `json:"schema"`
	Number int      `json:"number"`
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	Labels []string `json:"labels"`
}

// IssueBinding is the compact canonical projection carried inside a review. The
// title is retained because it is rendered visibly; body and labels only need
// component digests to authenticate the old projection after issue drift.
type IssueBinding struct {
	Schema       string `json:"schema"`
	Number       int    `json:"number"`
	Title        string `json:"title"`
	BodyDigest   string `json:"body_digest"`
	LabelsDigest string `json:"labels_digest"`
}

// Evidence states what supports an assessment. Partial evidence must say both
// what exists and what remains missing; a gap must name the missing evidence.
type Evidence struct {
	Status string   `json:"status"`
	Refs   []string `json:"refs,omitempty"`
	Gap    string   `json:"gap,omitempty"`
}

// ReviewRow is one selected check and the issue-specific conclusion it caused.
type ReviewRow struct {
	ID         string   `json:"id"`
	Relevance  string   `json:"relevance"`
	Assessment string   `json:"assessment"`
	Evidence   Evidence `json:"evidence"`
	Action     string   `json:"action"`
}

// Review is the closed agent response accepted by the trusted GitHub edge.
// ReviewerVersion is bounded self-reported diagnostic metadata, not authenticated
// provenance. A caller that has a trusted expected version must compare it itself.
type Review struct {
	Schema          string       `json:"schema"`
	IssueNumber     int          `json:"issue_number"`
	IssueDigest     string       `json:"issue_digest"`
	IssueBinding    IssueBinding `json:"issue_binding"`
	CatalogVersion  string       `json:"catalog_version"`
	ReviewerVersion string       `json:"reviewer_version"`
	Rows            []ReviewRow  `json:"rows"`
}

// ExistingComment is the read-only material the pure planner needs. The adapter
// is responsible for supplying only comments it is authorized to manage.
type ExistingComment struct {
	ID   int64  `json:"id"`
	Body string `json:"body"`
}

// CommentPlan tells a trusted edge whether to create, update, skip, or refuse.
// Refusal is reserved for ambiguous duplicate managed comments; malformed review
// data returns an error before a plan exists.
type CommentPlan struct {
	Action      string  `json:"action"`
	CommentID   int64   `json:"comment_id,omitempty"`
	Body        string  `json:"body,omitempty"`
	Reason      string  `json:"reason,omitempty"`
	MatchingIDs []int64 `json:"matching_ids,omitempty"`
}

// CommentVerification is the pure readback verdict for the one managed comment.
// Invalid and missing comments are ordinary negative verdicts; decode or issue
// projection failures additionally return an error with the precise cause.
type CommentVerification struct {
	Valid           bool    `json:"valid"`
	CommentID       int64   `json:"comment_id,omitempty"`
	IssueDigest     string  `json:"issue_digest,omitempty"`
	CatalogVersion  string  `json:"catalog_version,omitempty"`
	ReviewerVersion string  `json:"reviewer_version,omitempty"`
	Reason          string  `json:"reason,omitempty"`
	MatchingIDs     []int64 `json:"matching_ids,omitempty"`
}

// ManagedCommentState classifies the first-line production marker and its payload.
type ManagedCommentState string

// Managed comment classification states returned by InspectManagedComment.
const (
	// ManagedCommentUnmanaged indicates no managed marker was present.
	ManagedCommentUnmanaged ManagedCommentState = "unmanaged"
	// ManagedCommentCurrent indicates the comment matches the current issue digest and schema.
	ManagedCommentCurrent ManagedCommentState = "current"
	// ManagedCommentStale indicates the review was valid for a prior projection of the issue.
	ManagedCommentStale ManagedCommentState = "stale"
	// ManagedCommentMalformed indicates the comment payload is corrupt or tampered.
	ManagedCommentMalformed ManagedCommentState = "malformed"
)

// ManagedCommentInspection lets a trusted edge distinguish a canonically rendered
// prior review from a marker whose payload or visible body cannot be trusted.
type ManagedCommentInspection struct {
	State  ManagedCommentState `json:"state"`
	Review Review              `json:"review,omitempty"`
	Reason string              `json:"reason,omitempty"`
}

// CanonicalProjection normalizes semantically irrelevant transport differences:
// line endings, trailing line whitespace, label order/case, and duplicate labels.
func CanonicalProjection(issue Issue) Projection {
	return Projection{
		Schema: ProjectionSchema,
		Number: issue.Number,
		Title:  canonicalText(issue.Title),
		Body:   canonicalText(issue.Body),
		Labels: canonicalLabels(issue.Labels),
	}
}

// CanonicalIssueBinding returns the compact, deterministic witness stored in a
// Review. It retains exactly the old material needed to authenticate rendering
// while keeping an issue body out of the size-bounded GitHub comment payload.
func CanonicalIssueBinding(issue Issue) IssueBinding {
	projection := CanonicalProjection(issue)
	return IssueBinding{
		Schema:       BindingSchema,
		Number:       projection.Number,
		Title:        projection.Title,
		BodyDigest:   digestBytes([]byte(projection.Body)),
		LabelsDigest: digestJSON(projection.Labels),
	}
}

// IssueDigest returns the content binding stamped into a review and comment.
func IssueDigest(issue Issue) (string, error) {
	projection := CanonicalProjection(issue)
	if projection.Number <= 0 {
		return "", fmt.Errorf("issue number must be positive")
	}
	if projection.Title == "" {
		return "", fmt.Errorf("issue title must not be empty")
	}
	return digestIssueBinding(CanonicalIssueBinding(issue)), nil
}

func digestIssueBinding(binding IssueBinding) string { return digestJSON(binding) }

func digestJSON(value any) string {
	b, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("issuecheck: marshal deterministic digest value: %v", err))
	}
	return digestBytes(b)
}

func digestBytes(value []byte) string {
	sum := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// Invariant: issue check reviews are fail-closed and deterministic; any drift,
// schema divergence, stale issue digest, or unauthenticated comment payload is
// rejected immediately.
//
// ValidateReview verifies the closed response and binds it to the current issue.
func ValidateReview(issue Issue, review Review) error {
	wantDigest, err := IssueDigest(issue)
	if err != nil {
		return err
	}
	if err := validateReviewShape(issue.Number, review); err != nil {
		return err
	}
	if review.IssueBinding != CanonicalIssueBinding(issue) || review.IssueDigest != wantDigest {
		return fmt.Errorf("review issue_digest is stale: got %s, want %s", review.IssueDigest, wantDigest)
	}
	return nil
}

func validateReviewShape(issueNumber int, review Review) error {
	switch {
	case review.Schema != ReviewSchema:
		return fmt.Errorf("review schema %q does not match %q", review.Schema, ReviewSchema)
	case review.IssueNumber != issueNumber:
		return fmt.Errorf("review issue_number %d does not match issue #%d", review.IssueNumber, issueNumber)
	case !digestRE.MatchString(review.IssueDigest):
		return fmt.Errorf("review issue_digest must be sha256:<64 lowercase hex>")
	case review.IssueBinding.Schema != BindingSchema:
		return fmt.Errorf("review issue_binding schema %q does not match %q", review.IssueBinding.Schema, BindingSchema)
	case review.IssueBinding.Number != review.IssueNumber:
		return fmt.Errorf("review issue_binding number %d does not match issue_number %d", review.IssueBinding.Number, review.IssueNumber)
	case review.IssueBinding.Title == "" || review.IssueBinding.Title != canonicalText(review.IssueBinding.Title):
		return fmt.Errorf("review issue_binding title must be nonempty canonical text")
	case !digestRE.MatchString(review.IssueBinding.BodyDigest):
		return fmt.Errorf("review issue_binding body_digest must be sha256:<64 lowercase hex>")
	case !digestRE.MatchString(review.IssueBinding.LabelsDigest):
		return fmt.Errorf("review issue_binding labels_digest must be sha256:<64 lowercase hex>")
	case review.IssueDigest != digestIssueBinding(review.IssueBinding):
		return fmt.Errorf("review issue_digest does not match issue_binding")
	case review.CatalogVersion != CatalogVersion:
		return fmt.Errorf("review catalog_version %q does not match %q", review.CatalogVersion, CatalogVersion)
	case len(review.ReviewerVersion) > MaxReviewerVersionBytes || review.ReviewerVersion != strings.TrimSpace(review.ReviewerVersion) || !reviewerVersionRE.MatchString(review.ReviewerVersion):
		return fmt.Errorf("review reviewer_version must be a nonempty version token")
	case len(review.Rows) != RequiredReviewRows:
		return fmt.Errorf("review must contain exactly five rows, got %d", len(review.Rows))
	}

	seen := make(map[string]struct{}, RequiredReviewRows)
	for i, row := range review.Rows {
		check, ok := Lookup(row.ID)
		if !ok {
			return fmt.Errorf("row %d has unknown catalog id %q", i+1, row.ID)
		}
		if _, duplicate := seen[row.ID]; duplicate {
			return fmt.Errorf("row %d repeats catalog id %q", i+1, row.ID)
		}
		seen[row.ID] = struct{}{}
		if err := validateReviewRow(i+1, row, check); err != nil {
			return err
		}
	}
	payload, err := json.Marshal(review)
	if err != nil {
		return fmt.Errorf("marshal bounded review: %w", err)
	}
	if len(payload) > MaxReviewJSONBytes {
		return fmt.Errorf("review JSON is %d bytes, exceeds limit %d", len(payload), MaxReviewJSONBytes)
	}
	return nil
}

func validateReviewRow(position int, row ReviewRow, check Check) error {
	relevance := strings.TrimSpace(row.Relevance)
	assessment := strings.TrimSpace(row.Assessment)
	action := strings.TrimSpace(row.Action)
	switch {
	case row.ID != strings.TrimSpace(row.ID):
		return fmt.Errorf("row %d id must use the exact catalog token", position)
	case relevance == "":
		return fmt.Errorf("row %d (%s) relevance must not be empty", position, row.ID)
	case genericCatalogText(relevance, check):
		return fmt.Errorf("row %d (%s) relevance must be issue-specific, not copied catalog text", position, row.ID)
	case assessment == "":
		return fmt.Errorf("row %d (%s) assessment must not be empty", position, row.ID)
	case genericCatalogText(assessment, check):
		return fmt.Errorf("row %d (%s) assessment must state a conclusion, not copy catalog text", position, row.ID)
	case action == "":
		return fmt.Errorf("row %d (%s) action must not be empty", position, row.ID)
	case len(row.Relevance) > MaxReviewFieldBytes:
		return fmt.Errorf("row %d (%s) relevance exceeds %d bytes", position, row.ID, MaxReviewFieldBytes)
	case len(row.Assessment) > MaxReviewFieldBytes:
		return fmt.Errorf("row %d (%s) assessment exceeds %d bytes", position, row.ID, MaxReviewFieldBytes)
	case len(row.Action) > MaxReviewFieldBytes:
		return fmt.Errorf("row %d (%s) action exceeds %d bytes", position, row.ID, MaxReviewFieldBytes)
	case len(row.Evidence.Gap) > MaxReviewFieldBytes:
		return fmt.Errorf("row %d (%s) evidence gap exceeds %d bytes", position, row.ID, MaxReviewFieldBytes)
	case len(row.Evidence.Refs) > MaxEvidenceRefsPerRow:
		return fmt.Errorf("row %d (%s) has %d evidence refs, exceeds limit %d", position, row.ID, len(row.Evidence.Refs), MaxEvidenceRefsPerRow)
	}

	for _, ref := range row.Evidence.Refs {
		if strings.TrimSpace(ref) == "" {
			return fmt.Errorf("row %d (%s) evidence refs must not contain empty entries", position, row.ID)
		}
		if len(ref) > MaxEvidenceRefBytes {
			return fmt.Errorf("row %d (%s) evidence ref exceeds %d bytes", position, row.ID, MaxEvidenceRefBytes)
		}
	}
	refs := nonemptyStrings(row.Evidence.Refs)
	gap := strings.TrimSpace(row.Evidence.Gap)
	status := strings.TrimSpace(row.Evidence.Status)
	if status != row.Evidence.Status {
		return fmt.Errorf("row %d (%s) evidence status must use the exact closed token", position, row.ID)
	}
	switch status {
	case EvidenceSupported:
		if len(refs) == 0 {
			return fmt.Errorf("row %d (%s) supported evidence requires at least one ref", position, row.ID)
		}
		if gap != "" {
			return fmt.Errorf("row %d (%s) supported evidence cannot also declare a gap; use partial", position, row.ID)
		}
	case EvidencePartial:
		if len(refs) == 0 || gap == "" {
			return fmt.Errorf("row %d (%s) partial evidence requires refs and an explicit gap", position, row.ID)
		}
	case EvidenceGap:
		if gap == "" {
			return fmt.Errorf("row %d (%s) evidence gap must name what is missing", position, row.ID)
		}
		if len(refs) != 0 {
			return fmt.Errorf("row %d (%s) gap evidence cannot include refs; use partial", position, row.ID)
		}
	default:
		return fmt.Errorf("row %d (%s) evidence status must be supported, partial, or gap", position, row.ID)
	}
	return nil
}

func genericCatalogText(value string, check Check) bool {
	value = strings.ToLower(canonicalText(value))
	return value == strings.ToLower(canonicalText(check.Name)) ||
		value == strings.ToLower(canonicalText(check.Question)) ||
		value == strings.ToLower(canonicalText(check.When))
}

// FormatReviewComment formats the one durable, marker-keyed review comment. All agent
// and issue prose is escaped as text; it can never manufacture another marker or
// an HTML instruction in the rendered comment.
func FormatReviewComment(issue Issue, review Review) (string, error) {
	if err := ValidateReview(issue, review); err != nil {
		return "", err
	}
	return renderValidatedComment(review)
}

func renderValidatedComment(review Review) (string, error) {
	payload, err := json.Marshal(review)
	if err != nil {
		return "", fmt.Errorf("marshal validated review payload: %w", err)
	}
	var b strings.Builder
	b.WriteString(CommentMarker + "\n")
	b.WriteString(CommentPayloadPrefix)
	b.WriteString(base64.RawURLEncoding.EncodeToString(payload))
	b.WriteString(" -->\n")
	b.WriteString("## Top-5 Thought Check\n\n")
	fmt.Fprintf(&b, "Issue: `#%d` — %s\n\n", review.IssueNumber, markdownText(review.IssueBinding.Title))
	fmt.Fprintf(&b, "- Issue digest: `%s`\n", review.IssueDigest)
	fmt.Fprintf(&b, "- Catalog: `%s`\n", CatalogVersion)
	fmt.Fprintf(&b, "- Reviewer: `%s`\n", review.ReviewerVersion)
	b.WriteString("- Selection: exactly five issue-specific checks, ordered by reviewer relevance\n\n")

	for i, row := range review.Rows {
		check, _ := Lookup(row.ID)
		fmt.Fprintf(&b, "### %d. `%s` — %s\n\n", i+1, check.ID, markdownText(check.Name))
		fmt.Fprintf(&b, "- **Why relevant:** %s\n", markdownText(row.Relevance))
		fmt.Fprintf(&b, "- **Assessment:** %s\n", markdownText(row.Assessment))
		fmt.Fprintf(&b, "- **Evidence status:** `%s`\n", strings.TrimSpace(row.Evidence.Status))
		for _, ref := range nonemptyStrings(row.Evidence.Refs) {
			fmt.Fprintf(&b, "  - Evidence: %s\n", markdownText(ref))
		}
		if gap := strings.TrimSpace(row.Evidence.Gap); gap != "" {
			fmt.Fprintf(&b, "  - Evidence gap: %s\n", markdownText(gap))
		}
		fmt.Fprintf(&b, "- **Required action:** %s\n\n", markdownText(row.Action))
	}
	b.WriteString("This comment records bounded conclusions and evidence, not private chain-of-thought.\n")
	body := b.String()
	if len(body) > MaxRenderedCommentBytes {
		return "", fmt.Errorf("rendered managed comment is %d bytes, exceeds limit %d", len(body), MaxRenderedCommentBytes)
	}
	return body, nil
}

// IsManagedComment reports whether body starts with the exact marker on its
// first line. Marker-like prose embedded later in untrusted text does not match.
func IsManagedComment(body string) bool {
	return hasMarker(body)
}

// InspectManagedComment classifies a production-marker comment without treating
// expected issue drift as payload corruption. Stale is returned only when every
// invariant except the current issue digest still validates and the visible body
// remains the canonical rendering of the stored compact issue binding.
func InspectManagedComment(issue Issue, body string) ManagedCommentInspection {
	if !hasMarker(body) {
		return ManagedCommentInspection{State: ManagedCommentUnmanaged, Reason: "managed thought-check marker is missing"}
	}
	review, err := decodeManagedReview(body)
	if err != nil {
		return ManagedCommentInspection{State: ManagedCommentMalformed, Reason: err.Error()}
	}
	wantDigest, err := IssueDigest(issue)
	if err != nil {
		return ManagedCommentInspection{State: ManagedCommentMalformed, Reason: err.Error()}
	}
	if err := validateReviewShape(issue.Number, review); err != nil {
		return ManagedCommentInspection{State: ManagedCommentMalformed, Reason: err.Error()}
	}
	wantBody, err := renderValidatedComment(review)
	if err != nil {
		return ManagedCommentInspection{State: ManagedCommentMalformed, Reason: err.Error()}
	}
	if normalizeLineEndings(body) != wantBody {
		return ManagedCommentInspection{State: ManagedCommentMalformed, Reason: "managed comment body does not match its validated review payload"}
	}
	if review.IssueBinding != CanonicalIssueBinding(issue) || review.IssueDigest != wantDigest {
		return ManagedCommentInspection{
			State:  ManagedCommentStale,
			Review: review,
			Reason: fmt.Sprintf("review issue_digest is stale: got %s, want %s", review.IssueDigest, wantDigest),
		}
	}
	return ManagedCommentInspection{State: ManagedCommentCurrent, Review: review}
}

// ParseReviewComment decodes, validates, and byte-verifies a managed comment.
// Re-render equality means altered visible conclusions cannot ride on valid
// hidden metadata, and altered metadata cannot bless an old visible review.
func ParseReviewComment(issue Issue, body string) (Review, error) {
	inspection := InspectManagedComment(issue, body)
	if inspection.State == ManagedCommentCurrent {
		return inspection.Review, nil
	}
	return Review{}, fmt.Errorf("%s", inspection.Reason)
}

func decodeManagedReview(body string) (Review, error) {
	if len(body) > MaxRenderedCommentBytes {
		return Review{}, fmt.Errorf("managed comment is %d bytes, exceeds limit %d", len(body), MaxRenderedCommentBytes)
	}
	var encoded []string
	for _, line := range strings.Split(normalizeLineEndings(body), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, CommentPayloadPrefix) && strings.HasSuffix(line, " -->") {
			encoded = append(encoded, strings.TrimSuffix(strings.TrimPrefix(line, CommentPayloadPrefix), " -->"))
		}
	}
	if len(encoded) != 1 {
		return Review{}, fmt.Errorf("managed comment must contain exactly one review payload, got %d", len(encoded))
	}
	if base64.RawURLEncoding.DecodedLen(len(encoded[0])) > MaxReviewJSONBytes {
		return Review{}, fmt.Errorf("managed review payload exceeds limit %d bytes", MaxReviewJSONBytes)
	}
	payload, err := base64.RawURLEncoding.DecodeString(encoded[0])
	if err != nil {
		return Review{}, fmt.Errorf("decode managed review payload: %w", err)
	}
	var review Review
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&review); err != nil {
		return Review{}, fmt.Errorf("decode managed review JSON: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return Review{}, fmt.Errorf("decode managed review JSON: trailing data")
	}
	return review, nil
}

// VerifyComment verifies that exactly one current, internally consistent managed
// comment exists. Duplicate markers refuse without selecting a winner.
func VerifyComment(issue Issue, existing []ExistingComment) (CommentVerification, error) {
	matches := managedComments(existing)
	ids := commentIDs(matches)
	switch len(matches) {
	case 0:
		return CommentVerification{Reason: "managed thought-check comment is missing"}, nil
	case 1:
		if matches[0].ID <= 0 {
			return CommentVerification{Reason: "managed comment has an invalid non-positive id", MatchingIDs: ids}, nil
		}
		review, err := ParseReviewComment(issue, matches[0].Body)
		if err != nil {
			return CommentVerification{CommentID: matches[0].ID, Reason: err.Error(), MatchingIDs: ids}, err
		}
		return CommentVerification{
			Valid: true, CommentID: matches[0].ID, IssueDigest: review.IssueDigest,
			CatalogVersion: review.CatalogVersion, ReviewerVersion: review.ReviewerVersion,
			MatchingIDs: ids,
		}, nil
	default:
		return CommentVerification{Reason: "multiple managed thought-check comments found", MatchingIDs: ids}, nil
	}
}

// ChooseCommentAction compares the desired body with existing managed comments.
// More than one marker is ambiguous and always refuses rather than choosing a
// comment that could discard another review.
func ChooseCommentAction(issue Issue, review Review, existing []ExistingComment) (CommentPlan, error) {
	desired, err := FormatReviewComment(issue, review)
	if err != nil {
		return CommentPlan{}, err
	}
	matches := managedComments(existing)
	ids := commentIDs(matches)
	switch len(matches) {
	case 0:
		return CommentPlan{Action: ActionCreate, Body: desired}, nil
	case 1:
		if matches[0].ID <= 0 {
			return CommentPlan{Action: ActionRefuse, Reason: "managed comment has an invalid non-positive id", MatchingIDs: ids}, nil
		}
		if normalizeLineEndings(matches[0].Body) == desired {
			return CommentPlan{Action: ActionNoop, CommentID: matches[0].ID, MatchingIDs: ids}, nil
		}
		return CommentPlan{Action: ActionUpdate, CommentID: matches[0].ID, Body: desired, MatchingIDs: ids}, nil
	default:
		return CommentPlan{Action: ActionRefuse, Reason: "multiple managed thought-check comments found", MatchingIDs: ids}, nil
	}
}

func managedComments(existing []ExistingComment) []ExistingComment {
	matches := make([]ExistingComment, 0, 1)
	for _, comment := range existing {
		if hasMarker(comment.Body) {
			matches = append(matches, comment)
		}
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].ID < matches[j].ID })
	return matches
}

func commentIDs(comments []ExistingComment) []int64 {
	ids := make([]int64, len(comments))
	for i, comment := range comments {
		ids[i] = comment.ID
	}
	return ids
}

func canonicalText(value string) string {
	value = normalizeLineEndings(value)
	lines := strings.Split(value, "\n")
	for i := range lines {
		lines[i] = strings.TrimRight(lines[i], " \t")
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func canonicalLabels(labels []string) []string {
	set := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		label = strings.ToLower(strings.TrimSpace(label))
		if label != "" {
			set[label] = struct{}{}
		}
	}
	out := make([]string, 0, len(set))
	for label := range set {
		out = append(out, label)
	}
	sort.Strings(out)
	return out
}

func nonemptyStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			out = append(out, value)
		}
	}
	return out
}

func normalizeLineEndings(value string) string {
	value = strings.ReplaceAll(value, "\r\n", "\n")
	return strings.ReplaceAll(value, "\r", "\n")
}

func hasMarker(body string) bool {
	first, _, _ := strings.Cut(normalizeLineEndings(body), "\n")
	return first == CommentMarker
}

func markdownText(value string) string {
	value = canonicalText(value)
	value = strings.ReplaceAll(value, "&", "&amp;")
	value = strings.ReplaceAll(value, "<", "&lt;")
	value = strings.ReplaceAll(value, ">", "&gt;")
	for _, replacement := range []struct{ old, new string }{
		{"\\", `\\`}, {"`", "\\`"}, {"*", "\\*"}, {"_", "\\_"},
		{"[", "\\["}, {"]", "\\]"}, {"(", "\\("}, {")", "\\)"},
		{"#", "\\#"}, {"!", "\\!"}, {"|", "\\|"},
	} {
		value = strings.ReplaceAll(value, replacement.old, replacement.new)
	}
	return strings.ReplaceAll(value, "\n", "<br>")
}
