// Package fixproof exposes the public, content-bound parent-red/candidate-green
// proof mechanism without exposing fak's internal witness ABI.
package fixproof

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
	"time"
)

const (
	RequestSchema = "fak.fixproof.request/v1"
	ResultSchema  = "fak.fixproof.result/v1"
	Contract      = "go-test-parent-red-candidate-green/v1"

	MaxTimeoutMillis = int64((24 * time.Hour) / time.Millisecond)
	MaxDetailBytes   = 1024
)

var (
	ErrInvalidRequest        = errors.New("fixproof: invalid request")
	ErrIdentityMismatch      = errors.New("fixproof: identity mismatch")
	ErrUnsupportedProofShape = errors.New("fixproof: unsupported proof shape")
	ErrCheckoutChanged       = errors.New("fixproof: checkout changed")
)

// Verdict is the canonical witness outcome. Confirmed carries exactly the
// authority of internal/witness: it does not report numeric event counts or
// claim that every selected test failed at the parent.
type Verdict string

const (
	VerdictConfirmed Verdict = "confirmed"
	VerdictRefuted   Verdict = "refuted"
	VerdictAbstained Verdict = "abstained"
)

// Overlay binds one changed Go test file to its candidate Git blob.
type Overlay struct {
	Path string `json:"path"`
	Blob string `json:"blob"`
}

// ExecutionContext is asserted identity supplied by the caller and bound into
// Request.Digest. Execute validates its shape but does not claim to observe
// these values. The retained proof runner must measure and compare them before
// calling Execute.
type ExecutionContext struct {
	GoExecutableDigest string `json:"go_executable_digest"`
	ToolchainDigest    string `json:"toolchain_digest"`
	TestEnvDigest      string `json:"test_env_digest"`
	WorkspaceDigest    string `json:"workspace_digest"`
	VerifierDigest     string `json:"verifier_digest"`
}

// Request is the immutable identity of one proof. It deliberately accepts no
// command or argv: Execute uses the canonical witness implementation only.
type Request struct {
	Schema          string           `json:"schema"`
	Contract        string           `json:"contract"`
	ParentCommit    string           `json:"parent_commit"`
	ParentTree      string           `json:"parent_tree"`
	CandidateCommit string           `json:"candidate_commit"`
	CandidateTree   string           `json:"candidate_tree"`
	CompanionCommit string           `json:"companion_commit"`
	CompanionTree   string           `json:"companion_tree"`
	Paths           []string         `json:"paths"`
	TestOverlays    []Overlay        `json:"test_overlays"`
	PackageDir      string           `json:"package_dir"`
	Selectors       []string         `json:"selectors"`
	BuildTags       []string         `json:"build_tags,omitempty"`
	TimeoutMillis   int64            `json:"timeout_millis"`
	Context         ExecutionContext `json:"context"`
}

// Roots are runtime locators, never proof identity. Execute requires detached,
// clean sibling independent clones with distinct Git common namespaces and
// excludes these paths from Request.Digest.
type Roots struct {
	RepositoryDir string
	CompanionDir  string
}

// Result is a bounded mapping of the canonical witness outcome, bound to the
// complete request. Detail contains no test output or source bytes.
type Result struct {
	Schema        string  `json:"schema"`
	Contract      string  `json:"contract"`
	RequestDigest string  `json:"request_digest"`
	Verdict       Verdict `json:"verdict"`
	Detail        string  `json:"detail"`
}

var buildTagRE = regexp.MustCompile(`^[A-Za-z0-9_]+$`)

func (r Request) Validate() error {
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidRequest, fmt.Sprintf(format, args...))
	}
	if r.Schema != RequestSchema || r.Contract != Contract {
		return invalid("closed schema/contract mismatch")
	}
	objects := []struct{ name, value string }{
		{"parent commit", r.ParentCommit}, {"parent tree", r.ParentTree},
		{"candidate commit", r.CandidateCommit}, {"candidate tree", r.CandidateTree},
		{"companion commit", r.CompanionCommit}, {"companion tree", r.CompanionTree},
	}
	for _, object := range objects {
		if !lowerObjectID(object.value) {
			return invalid("%s is not a lowercase Git object id", object.name)
		}
	}
	if r.ParentCommit == r.CandidateCommit {
		return invalid("parent and candidate commits are identical")
	}
	if err := canonicalPaths(r.Paths, false); err != nil {
		return invalid("paths: %v", err)
	}
	if !validRelativePath(r.PackageDir, true) {
		return invalid("invalid package_dir %q", r.PackageDir)
	}
	pathSet := make(map[string]struct{}, len(r.Paths))
	for _, p := range r.Paths {
		pathSet[p] = struct{}{}
	}
	lastOverlay := ""
	for i, overlay := range r.TestOverlays {
		if !validRelativePath(overlay.Path, false) || !strings.HasSuffix(overlay.Path, "_test.go") {
			return invalid("overlay %d has invalid Go test path %q", i, overlay.Path)
		}
		if i > 0 && overlay.Path <= lastOverlay {
			return invalid("test overlays are not sorted and unique")
		}
		if _, ok := pathSet[overlay.Path]; !ok {
			return invalid("overlay %q is not in paths", overlay.Path)
		}
		if path.Dir(overlay.Path) != r.PackageDir {
			return invalid("overlay %q is outside package_dir %q", overlay.Path, r.PackageDir)
		}
		if !lowerObjectID(overlay.Blob) {
			return invalid("overlay %q blob is not a lowercase Git object id", overlay.Path)
		}
		lastOverlay = overlay.Path
	}
	if len(r.Selectors) == 0 || !sort.StringsAreSorted(r.Selectors) {
		return invalid("selectors must be nonempty, sorted and unique")
	}
	for i, selector := range r.Selectors {
		if selector == "" || selector != strings.TrimSpace(selector) || (i > 0 && selector == r.Selectors[i-1]) {
			return invalid("invalid or duplicate selector %q", selector)
		}
		re, err := regexp.Compile(selector)
		if err != nil || re.MatchString("") {
			return invalid("selector must be valid and must not match empty: %q", selector)
		}
	}
	if !sort.StringsAreSorted(r.BuildTags) {
		return invalid("build_tags must be sorted and unique")
	}
	for i, tag := range r.BuildTags {
		if !buildTagRE.MatchString(tag) || (i > 0 && tag == r.BuildTags[i-1]) {
			return invalid("invalid or duplicate build tag %q", tag)
		}
	}
	if r.TimeoutMillis <= 0 || r.TimeoutMillis > MaxTimeoutMillis {
		return invalid("timeout_millis must be within 1..%d", MaxTimeoutMillis)
	}
	digests := []struct{ name, value string }{
		{"go executable", r.Context.GoExecutableDigest},
		{"toolchain", r.Context.ToolchainDigest},
		{"test environment", r.Context.TestEnvDigest},
		{"workspace", r.Context.WorkspaceDigest},
		{"verifier", r.Context.VerifierDigest},
	}
	for _, digest := range digests {
		if !lowerSHA256(digest.value) {
			return invalid("%s digest is not lowercase sha256", digest.name)
		}
	}
	return nil
}

// Digest returns the SHA-256 of the canonical request JSON. Runtime Roots are
// intentionally absent; content identity must survive checkout relocation.
func (r Request) Digest() (string, error) {
	if err := r.Validate(); err != nil {
		return "", err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func (r Result) Validate(request Request) error {
	if err := request.Validate(); err != nil {
		return err
	}
	digest, err := request.Digest()
	if err != nil {
		return err
	}
	if r.Schema != ResultSchema || r.Contract != Contract || r.RequestDigest != digest {
		return fmt.Errorf("%w: result binding mismatch", ErrInvalidRequest)
	}
	switch r.Verdict {
	case VerdictConfirmed, VerdictRefuted, VerdictAbstained:
	default:
		return fmt.Errorf("%w: unknown result verdict %q", ErrInvalidRequest, r.Verdict)
	}
	if r.Detail == "" || len(r.Detail) > MaxDetailBytes {
		return fmt.Errorf("%w: result detail must contain 1..%d bytes", ErrInvalidRequest, MaxDetailBytes)
	}
	return nil
}

func canonicalPaths(paths []string, allowEmpty bool) error {
	if len(paths) == 0 && !allowEmpty {
		return errors.New("empty")
	}
	if !sort.StringsAreSorted(paths) {
		return errors.New("not sorted")
	}
	for i, p := range paths {
		if !validRelativePath(p, false) {
			return fmt.Errorf("invalid path %q", p)
		}
		if i > 0 && p == paths[i-1] {
			return fmt.Errorf("duplicate path %q", p)
		}
	}
	return nil
}

func validRelativePath(p string, allowDot bool) bool {
	if p == "." {
		return allowDot
	}
	if p == "" || p != strings.TrimSpace(p) || strings.ContainsAny(p, "\\\x00\r\n") || path.IsAbs(p) {
		return false
	}
	if len(p) >= 2 && ((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) && p[1] == ':' {
		return false
	}
	return path.Clean(p) == p && p != ".." && !strings.HasPrefix(p, "../")
}

func lowerObjectID(s string) bool {
	return (len(s) == 40 || len(s) == 64) && lowerHex(s)
}

func lowerSHA256(s string) bool { return len(s) == 64 && lowerHex(s) }

func lowerHex(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range []byte(s) {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
