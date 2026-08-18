package disambiguation

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

const OwnershipSourceSelfTestSchemaVersion = "fak-disambiguation-ownership-source-self-test/1"

type OwnershipMismatchKind string

const (
	OwnershipPathUnknown   OwnershipMismatchKind = "path-unknown"
	OwnershipLeafMismatch  OwnershipMismatchKind = "leaf-mismatch"
	OwnershipLaneMismatch  OwnershipMismatchKind = "lane-mismatch"
	OwnershipStampMismatch OwnershipMismatchKind = "stamp-mismatch"
)

type OwnershipMismatchError struct {
	Kind   OwnershipMismatchKind
	Detail string
}

func (e *OwnershipMismatchError) Error() string { return string(e.Kind) + ": " + e.Detail }

type OwnershipBinding struct {
	Path        string `json:"path"`
	ModuleAtRev string `json:"module_at_rev"`
	Leaf        string `json:"leaf"`
	Lane        string `json:"lane"`
	Stamp       string `json:"stamp"`
}

type OwnershipFixture struct {
	Path   string
	Module string
	Rev    int
	SHA    string
	Leaf   string
	Lane   string
	Stamp  string
}

var stampPattern = regexp.MustCompile(`^\(fak ([a-z0-9][a-z0-9-]*)\)$`)

func ResolveOwnershipFixture(f OwnershipFixture) (OwnershipBinding, error) {
	if strings.TrimSpace(f.Path) == "" || strings.TrimSpace(f.Module) == "" || f.Rev < 1 || strings.TrimSpace(f.SHA) == "" {
		return OwnershipBinding{}, &OwnershipMismatchError{Kind: OwnershipPathUnknown, Detail: f.Path}
	}
	normalized := filepath.ToSlash(filepath.Clean(f.Path))
	if normalized != f.Module && !strings.HasPrefix(normalized, strings.TrimSuffix(f.Module, "/")+"/") {
		return OwnershipBinding{}, &OwnershipMismatchError{Kind: OwnershipLeafMismatch, Detail: normalized + " outside " + f.Module}
	}
	if strings.TrimSpace(f.Leaf) == "" {
		return OwnershipBinding{}, &OwnershipMismatchError{Kind: OwnershipLeafMismatch, Detail: "empty leaf"}
	}
	if strings.TrimSpace(f.Lane) == "" {
		return OwnershipBinding{}, &OwnershipMismatchError{Kind: OwnershipLaneMismatch, Detail: "empty lane"}
	}
	match := stampPattern.FindStringSubmatch(f.Stamp)
	if len(match) != 2 || match[1] != f.Leaf {
		return OwnershipBinding{}, &OwnershipMismatchError{Kind: OwnershipStampMismatch, Detail: f.Stamp + " does not bind " + f.Leaf}
	}
	return OwnershipBinding{Path: normalized, ModuleAtRev: fmt.Sprintf("%s@r%d+g%s", f.Module, f.Rev, f.SHA), Leaf: f.Leaf, Lane: f.Lane, Stamp: f.Stamp}, nil
}

type OwnershipSourceSelfTestReport struct {
	Schema             string           `json:"schema"`
	Binding            OwnershipBinding `json:"binding"`
	LeafMismatchTyped  bool             `json:"leaf_mismatch_typed"`
	LaneMismatchTyped  bool             `json:"lane_mismatch_typed"`
	StampMismatchTyped bool             `json:"stamp_mismatch_typed"`
}

func RunOwnershipSourceSelfTest() (OwnershipSourceSelfTestReport, error) {
	fixture := OwnershipFixture{Path: "internal/disambiguation/query.go", Module: "internal/disambiguation", Rev: 31, SHA: "76d63130cd", Leaf: "disambiguation", Lane: "disambiguation", Stamp: "(fak disambiguation)"}
	binding, err := ResolveOwnershipFixture(fixture)
	if err != nil {
		return OwnershipSourceSelfTestReport{}, err
	}
	report := OwnershipSourceSelfTestReport{Schema: OwnershipSourceSelfTestSchemaVersion, Binding: binding}
	wrongLeaf := fixture
	wrongLeaf.Module = "internal/gateway"
	wrongLane := fixture
	wrongLane.Lane = ""
	wrongStamp := fixture
	wrongStamp.Stamp = "(fak gateway)"
	var mismatch *OwnershipMismatchError
	report.LeafMismatchTyped = errors.As(resolveOwnershipError(wrongLeaf), &mismatch) && mismatch.Kind == OwnershipLeafMismatch
	mismatch = nil
	report.LaneMismatchTyped = errors.As(resolveOwnershipError(wrongLane), &mismatch) && mismatch.Kind == OwnershipLaneMismatch
	mismatch = nil
	report.StampMismatchTyped = errors.As(resolveOwnershipError(wrongStamp), &mismatch) && mismatch.Kind == OwnershipStampMismatch
	if !report.LeafMismatchTyped || !report.LaneMismatchTyped || !report.StampMismatchTyped {
		return report, errors.New("ownership mismatch typing failed")
	}
	return report, nil
}

func resolveOwnershipError(f OwnershipFixture) error {
	_, err := ResolveOwnershipFixture(f)
	return err
}
