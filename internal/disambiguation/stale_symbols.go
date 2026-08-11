package disambiguation

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Public reference kinds identify repository-visible contracts that can become
// stale independently of the file carrying them.
const (
	ReferenceKindGoSymbol   = "go-symbol"
	ReferenceKindCLIVerb    = "cli-verb"
	ReferenceKindReasonCode = "reason-code"
	ReferenceKindDocAnchor  = "doc-anchor"

	FreshnessReasonPublicSymbolMissing = "PUBLIC_SYMBOL_MISSING"
	FreshnessReasonCLIVerbMissing      = "CLI_VERB_MISSING"
	FreshnessReasonReasonCodeMissing   = "REASON_CODE_MISSING"
	FreshnessReasonDocAnchorMissing    = "DOCUMENT_ANCHOR_MISSING"
)

var (
	goIdentifier = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	cliVerb      = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*(?: [a-z0-9][a-z0-9-]*)*$`)
	reasonCode   = regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)
	docAnchor    = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)
)

// PublicReference names a contract inside a public repository source.
type PublicReference struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
}

// ProbePublicReferences re-evaluates an entry's cited public references from a
// repository checkout. It only reads repository-visible files and does not
// write or register entries. Invalid evidence outranks unavailable probes;
// unavailable probes outrank missing references, preserving #6280 precedence.
func ProbePublicReferences(repoRoot string, entry Entry) Entry {
	if err := entry.Validate(); err != nil {
		entry.Freshness = probeFreshness(entry, FreshnessInvalid, FreshnessReasonEvidenceMalformed)
		return entry
	}
	if strings.TrimSpace(repoRoot) == "" {
		entry.Freshness = probeFreshness(entry, FreshnessUnknown, FreshnessReasonProbeUnavailable)
		return entry
	}

	verdict := FreshnessFresh
	reason := FreshnessReasonSourceCurrent
	for _, source := range entry.Sources {
		sourceVerdict, sourceReason := probePublicSource(repoRoot, source)
		if freshnessRank(sourceVerdict) > freshnessRank(verdict) {
			verdict, reason = sourceVerdict, sourceReason
		}
	}
	entry.Freshness = probeFreshness(entry, verdict, reason)
	return entry
}

func probeFreshness(entry Entry, verdict FreshnessVerdict, reason string) Freshness {
	return Freshness{Verdict: verdict, ReasonCode: reason, CheckedAt: entry.Freshness.CheckedAt, Probe: entry.Freshness.Probe}
}

func freshnessRank(verdict FreshnessVerdict) int {
	switch verdict {
	case FreshnessInvalid:
		return 3
	case FreshnessUnknown:
		return 2
	case FreshnessStale:
		return 1
	default:
		return 0
	}
}

func probePublicSource(root string, source SourceWitness) (FreshnessVerdict, string) {
	filePart := strings.SplitN(source.Locator, "#", 2)[0]
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(filePart)))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return FreshnessStale, missingReason(source.Reference)
		}
		return FreshnessUnknown, FreshnessReasonProbeUnavailable
	}
	if source.Reference == nil {
		return FreshnessFresh, FreshnessReasonSourceCurrent
	}
	found, valid := referenceExists(filePart, data, *source.Reference)
	if !valid {
		return FreshnessInvalid, FreshnessReasonEvidenceMalformed
	}
	if !found {
		return FreshnessStale, missingReason(source.Reference)
	}
	return FreshnessFresh, FreshnessReasonSourceCurrent
}

func referenceExists(locator string, data []byte, reference PublicReference) (bool, bool) {
	switch reference.Kind {
	case ReferenceKindGoSymbol:
		file, err := parser.ParseFile(token.NewFileSet(), locator, data, 0)
		if err != nil {
			return false, false
		}
		return hasGoSymbol(file, reference.Name), true
	case ReferenceKindReasonCode:
		file, err := parser.ParseFile(token.NewFileSet(), locator, data, 0)
		if err != nil {
			return false, false
		}
		return hasReasonCode(file, reference.Name), true
	case ReferenceKindCLIVerb:
		file, err := parser.ParseFile(token.NewFileSet(), locator, data, 0)
		if err != nil {
			return false, false
		}
		return hasStringLiteral(file, reference.Name), true
	case ReferenceKindDocAnchor:
		return hasMarkdownAnchor(string(data), reference.Name), true
	default:
		return false, false
	}
}

func hasGoSymbol(file *ast.File, name string) bool {
	for _, decl := range file.Decls {
		switch node := decl.(type) {
		case *ast.FuncDecl:
			if node.Name.Name == name {
				return true
			}
		case *ast.GenDecl:
			for _, spec := range node.Specs {
				switch item := spec.(type) {
				case *ast.TypeSpec:
					if item.Name.Name == name {
						return true
					}
				case *ast.ValueSpec:
					for _, ident := range item.Names {
						if ident.Name == name {
							return true
						}
					}
				}
			}
		}
	}
	return false
}

func hasReasonCode(file *ast.File, code string) bool {
	found := false
	ast.Inspect(file, func(node ast.Node) bool {
		lit, ok := node.(*ast.BasicLit)
		if ok && lit.Kind == token.STRING && strings.Trim(lit.Value, "`\"") == code {
			found = true
		}
		return !found
	})
	return found
}

func hasStringLiteral(file *ast.File, value string) bool { return hasReasonCode(file, value) }

func hasMarkdownAnchor(document, wanted string) bool {
	for _, line := range strings.Split(strings.ReplaceAll(document, "\r\n", "\n"), "\n") {
		heading := strings.TrimSpace(line)
		if !strings.HasPrefix(heading, "#") {
			continue
		}
		heading = strings.TrimSpace(strings.TrimLeft(heading, "#"))
		if markdownAnchor(heading) == wanted {
			return true
		}
	}
	return false
}

func markdownAnchor(heading string) string {
	heading = strings.ToLower(heading)
	var b strings.Builder
	dash := false
	for _, r := range heading {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			dash = false
			continue
		}
		if (r == ' ' || r == '-') && b.Len() > 0 && !dash {
			b.WriteByte('-')
			dash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

func missingReason(reference *PublicReference) string {
	if reference == nil {
		return FreshnessReasonPublicSymbolMissing
	}
	switch reference.Kind {
	case ReferenceKindCLIVerb:
		return FreshnessReasonCLIVerbMissing
	case ReferenceKindReasonCode:
		return FreshnessReasonReasonCodeMissing
	case ReferenceKindDocAnchor:
		return FreshnessReasonDocAnchorMissing
	default:
		return FreshnessReasonPublicSymbolMissing
	}
}

func validatePublicReference(field string, reference *PublicReference) error {
	if reference == nil {
		return nil
	}
	valid := false
	switch reference.Kind {
	case ReferenceKindGoSymbol:
		valid = goIdentifier.MatchString(reference.Name) && token.IsExported(reference.Name)
	case ReferenceKindCLIVerb:
		valid = cliVerb.MatchString(reference.Name)
	case ReferenceKindReasonCode:
		valid = reasonCode.MatchString(reference.Name)
	case ReferenceKindDocAnchor:
		valid = docAnchor.MatchString(reference.Name)
	}
	if !valid {
		return provenanceError(ErrProvenanceReferenceInvalid, field, fmt.Sprintf("invalid %q public reference %q", reference.Kind, reference.Name))
	}
	return nil
}
