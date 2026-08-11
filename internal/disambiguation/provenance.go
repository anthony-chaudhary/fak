package disambiguation

import (
	"errors"
	"fmt"
	"path"
	"regexp"
	"strings"
	"time"
)

// Public source kinds are intentionally closed: each kind denotes material that
// can be independently read from the public fak repository checkout.
const (
	SourceKindDocument       = "document"
	SourceKindGoSource       = "go-source"
	SourceKindTest           = "test"
	SourceKindGeneratedIndex = "generated-index"
	SourceKindGitHubMetadata = "github-metadata"

	ErrProvenanceSourceKind       = "DISAMBIGUATION_PROVENANCE_SOURCE_KIND_UNVERIFIABLE"
	ErrProvenanceLocatorAbsolute  = "DISAMBIGUATION_PROVENANCE_LOCATOR_ABSOLUTE"
	ErrProvenanceLocatorEscape    = "DISAMBIGUATION_PROVENANCE_LOCATOR_ESCAPE"
	ErrProvenanceLocatorInvalid   = "DISAMBIGUATION_PROVENANCE_LOCATOR_INVALID"
	ErrProvenanceRevisionInvalid  = "DISAMBIGUATION_PROVENANCE_REVISION_INVALID"
	ErrProvenanceCheckedAtInvalid = "DISAMBIGUATION_PROVENANCE_CHECKED_AT_INVALID"
	ErrProvenanceProbeInvalid     = "DISAMBIGUATION_PROVENANCE_PROBE_IDENTITY_INVALID"
)

var (
	windowsAbsoluteLocator = regexp.MustCompile(`^[A-Za-z]:[\\/]`)
	provenanceRevision     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}$`)
	probeIdentity          = regexp.MustCompile(`^[a-z0-9][a-z0-9._/-]{0,127}$`)
)

// ValidationError is the stable machine-readable provenance admission error.
// Code and Field are contract fields; Message is deterministic human context.
type ValidationError struct {
	Code    string `json:"code"`
	Field   string `json:"field"`
	Message string `json:"message"`
}

func (e *ValidationError) Error() string { return e.Code + ": " + e.Field + ": " + e.Message }

// ValidationCode extracts a stable code without parsing an error string.
func ValidationCode(err error) string {
	var target *ValidationError
	if errors.As(err, &target) {
		return target.Code
	}
	return ""
}

func provenanceError(code, field, message string) error {
	return &ValidationError{Code: code, Field: field, Message: message}
}

func validateSourceProvenance(i int, source SourceWitness) error {
	prefix := fmt.Sprintf("sources[%d]", i)
	if !isPublicSourceKind(source.Kind) {
		return provenanceError(ErrProvenanceSourceKind, prefix+".kind", fmt.Sprintf("%q is not a verifiable public source kind", source.Kind))
	}
	if err := validateRepositoryLocator(prefix+".locator", source.Locator); err != nil {
		return err
	}
	if !provenanceRevision.MatchString(source.Revision) {
		return provenanceError(ErrProvenanceRevisionInvalid, prefix+".revision", "must be a 1..128 character public revision identity")
	}
	checkedAt, err := time.Parse(time.RFC3339, source.CheckedAt)
	if err != nil || checkedAt.Format(time.RFC3339) != source.CheckedAt || checkedAt.Location() != time.UTC {
		return provenanceError(ErrProvenanceCheckedAtInvalid, prefix+".checked_at", "must be canonical UTC RFC3339")
	}
	if !probeIdentity.MatchString(source.Probe) {
		return provenanceError(ErrProvenanceProbeInvalid, prefix+".probe", "must be a stable lowercase public probe identity")
	}
	return nil
}

func isPublicSourceKind(kind string) bool {
	switch kind {
	case SourceKindDocument, SourceKindGoSource, SourceKindTest, SourceKindGeneratedIndex, SourceKindGitHubMetadata:
		return true
	default:
		return false
	}
}

func validateRepositoryLocator(field, locator string) error {
	if strings.HasPrefix(locator, "/") || strings.HasPrefix(locator, `\`) || windowsAbsoluteLocator.MatchString(locator) {
		return provenanceError(ErrProvenanceLocatorAbsolute, field, "must be repository-relative")
	}
	if locator == "" || strings.TrimSpace(locator) != locator || strings.Contains(locator, `\`) || strings.ContainsRune(locator, '\x00') {
		return provenanceError(ErrProvenanceLocatorInvalid, field, "must use a non-empty clean slash-separated repository path")
	}
	filePart := locator
	if at := strings.IndexByte(filePart, '#'); at >= 0 {
		if at == 0 || at == len(filePart)-1 || strings.Contains(filePart[at+1:], "#") {
			return provenanceError(ErrProvenanceLocatorInvalid, field, "fragment must follow a repository path")
		}
		filePart = filePart[:at]
	}
	clean := path.Clean(filePart)
	if clean == ".." || strings.HasPrefix(clean, "../") {
		return provenanceError(ErrProvenanceLocatorEscape, field, "must not escape the repository root")
	}
	if clean == "." || clean != filePart || strings.Contains(filePart, "//") {
		return provenanceError(ErrProvenanceLocatorInvalid, field, "must be a normalized repository-relative path")
	}
	return nil
}
