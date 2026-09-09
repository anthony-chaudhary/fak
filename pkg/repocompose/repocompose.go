// Package repocompose plans a caller-declared set of local repository roots.
//
// It deliberately performs only bounded filesystem probes. It does not discover
// siblings, create directories, clone repositories, or activate anything from a
// selected root.
package repocompose

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ReceiptSchema is the stable schema identifier for composition receipts.
const ReceiptSchema = "fak.repo-compose-receipt/v1"

const maxMarkerBytes = 1 << 20

// Mode controls how a candidate participates in composition.
type Mode string

const (
	// ModeAuto selects a present, valid root and tolerates an implicitly
	// supplied root being absent.
	ModeAuto Mode = "auto"
	// ModeRequired requires the root to be present and valid.
	ModeRequired Mode = "required"
	// ModeOff records the candidate as disabled without probing it.
	ModeOff Mode = "off"
)

// State is the filesystem-independent outcome vocabulary in a receipt.
type State string

const (
	StateSelected State = "selected"
	StateAbsent   State = "absent"
	StateDisabled State = "disabled"
	StateInvalid  State = "invalid"
)

// Identity describes a bounded marker used to establish that a present root is
// the repository the caller intended. Path must be relative to the candidate
// root. Contains, when non-empty, must occur within the first 1 MiB of the marker.
type Identity struct {
	Path     string `json:"path"`
	Contains string `json:"contains,omitempty"`
}

// Candidate is one caller-declared repository member. Explicit distinguishes an
// operator-requested auto candidate from an inferred default: absence of the
// former is an error, while absence of the latter is expected.
type Candidate struct {
	Name     string   `json:"name,omitempty"`
	Root     string   `json:"root"`
	Mode     Mode     `json:"mode,omitempty"`
	Explicit bool     `json:"explicit,omitempty"`
	Source   string   `json:"source,omitempty"`
	Identity Identity `json:"identity"`
}

// Entry records the outcome of one candidate in caller order.
type Entry struct {
	Name          string `json:"name"`
	Root          string `json:"root"`
	CanonicalRoot string `json:"canonical_root,omitempty"`
	Mode          Mode   `json:"mode"`
	Explicit      bool   `json:"explicit,omitempty"`
	State         State  `json:"state"`
	Source        string `json:"source"`
	Reason        string `json:"reason"`
}

// Receipt is deterministic: it contains no clock or environment-derived search
// results, and Entries preserve caller order.
type Receipt struct {
	Schema   string   `json:"schema"`
	Selected []string `json:"selected,omitempty"`
	Entries  []Entry  `json:"entries"`
}

// Plan contains canonical, deduplicated roots in caller precedence order and
// the evidence explaining every candidate disposition.
type Plan struct {
	Roots   []string `json:"roots,omitempty"`
	Receipt Receipt  `json:"receipt"`
}

// MissingError reports a candidate whose policy makes absence actionable.
type MissingError struct {
	Name     string
	Root     string
	Mode     Mode
	Explicit bool
}

func (e *MissingError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("repository candidate %q is missing at %q (mode=%s explicit=%t)", e.Name, e.Root, e.Mode, e.Explicit)
}

// InvalidError reports an invalid candidate declaration, inaccessible root, or
// identity-marker failure. Reason is a stable receipt reason.
type InvalidError struct {
	Name   string
	Root   string
	Reason string
	Err    error
}

func (e *InvalidError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("repository candidate %q at %q is invalid (%s): %v", e.Name, e.Root, e.Reason, e.Err)
}

func (e *InvalidError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// Compose evaluates primary followed by overlays. An omitted primary mode
// defaults to required; omitted overlay modes default to auto. On failure the
// returned plan still contains the complete deterministic receipt.
func Compose(primary Candidate, overlays ...Candidate) (Plan, error) {
	candidates := make([]Candidate, 0, 1+len(overlays))
	candidates = append(candidates, primary)
	candidates = append(candidates, overlays...)

	plan := Plan{Receipt: Receipt{Schema: ReceiptSchema, Entries: make([]Entry, 0, len(candidates))}}
	selected := make([]selectedRoot, 0, len(candidates))
	var firstErr error

	for i, candidate := range candidates {
		mode := candidate.Mode
		if mode == "" {
			if i == 0 {
				mode = ModeRequired
			} else {
				mode = ModeAuto
			}
		}
		name := strings.TrimSpace(candidate.Name)
		if name == "" {
			if i == 0 {
				name = "primary"
			} else {
				name = fmt.Sprintf("overlay-%d", i)
			}
		}
		source := strings.TrimSpace(candidate.Source)
		if source == "" {
			source = "caller"
		}

		entry := Entry{
			Name:     name,
			Root:     candidate.Root,
			Mode:     mode,
			Explicit: candidate.Explicit,
			Source:   source,
		}
		if i == 0 && mode != ModeRequired {
			entry.State = StateInvalid
			entry.Reason = "primary_mode_not_required"
			firstErr = preserveFirst(firstErr, invalid(name, candidate.Root, entry.Reason, fmt.Errorf("primary mode must be %q, got %q", ModeRequired, mode)))
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}
		if mode == ModeOff {
			entry.State = StateDisabled
			entry.Reason = "mode_off"
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}
		if mode != ModeAuto && mode != ModeRequired {
			firstErr = preserveFirst(firstErr, invalid(name, candidate.Root, "invalid_mode", fmt.Errorf("unsupported mode %q", mode)))
			entry.State = StateInvalid
			entry.Reason = "invalid_mode"
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}

		canonical, err := canonicalPath(candidate.Root)
		if err != nil {
			firstErr = preserveFirst(firstErr, invalid(name, candidate.Root, "invalid_root", err))
			entry.State = StateInvalid
			entry.Reason = "invalid_root"
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}
		entry.CanonicalRoot = canonical

		info, err := os.Stat(canonical)
		if errors.Is(err, os.ErrNotExist) {
			entry.State = StateAbsent
			entry.Reason = "root_not_found"
			if mode == ModeRequired || candidate.Explicit {
				firstErr = preserveFirst(firstErr, &MissingError{Name: name, Root: canonical, Mode: mode, Explicit: candidate.Explicit})
			}
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}
		if err != nil {
			firstErr = preserveFirst(firstErr, invalid(name, canonical, "root_unreadable", err))
			entry.State = StateInvalid
			entry.Reason = "root_unreadable"
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}
		if !info.IsDir() {
			err = errors.New("root is not a directory")
			firstErr = preserveFirst(firstErr, invalid(name, canonical, "root_not_directory", err))
			entry.State = StateInvalid
			entry.Reason = "root_not_directory"
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}

		reason, err := verifyIdentity(canonical, candidate.Identity)
		if err != nil {
			firstErr = preserveFirst(firstErr, invalid(name, canonical, reason, err))
			entry.State = StateInvalid
			entry.Reason = reason
			plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
			continue
		}

		entry.State = StateSelected
		if sameSelectedRoot(canonical, info, selected) {
			entry.Reason = "duplicate_root"
		} else {
			entry.Reason = "identity_verified"
			selected = append(selected, selectedRoot{path: canonical, info: info})
			plan.Roots = append(plan.Roots, canonical)
			plan.Receipt.Selected = append(plan.Receipt.Selected, canonical)
		}
		plan.Receipt.Entries = append(plan.Receipt.Entries, entry)
	}

	return plan, firstErr
}

func canonicalPath(root string) (string, error) {
	if strings.TrimSpace(root) == "" {
		return "", errors.New("root is empty")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("canonicalize root: %w", err)
	}
	abs = filepath.Clean(abs)
	_, err = os.Lstat(abs)
	if errors.Is(err, os.ErrNotExist) {
		return abs, nil
	}
	if err != nil {
		return "", fmt.Errorf("inspect root declaration: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err == nil {
		return filepath.Clean(resolved), nil
	}
	return "", fmt.Errorf("resolve root symlinks: %w", err)
}

func verifyIdentity(root string, identity Identity) (string, error) {
	marker := filepath.Clean(identity.Path)
	if strings.TrimSpace(identity.Path) == "" || marker == "." || filepath.IsAbs(marker) || escapesRoot(marker) {
		return "invalid_identity_marker", errors.New("identity marker must be a non-empty relative path within the root")
	}
	markerPath := filepath.Join(root, marker)
	resolved, err := filepath.EvalSymlinks(markerPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "identity_marker_missing", fmt.Errorf("identity marker %q is missing", identity.Path)
		}
		return "identity_marker_unreadable", fmt.Errorf("resolve identity marker %q: %w", identity.Path, err)
	}
	if !withinRoot(root, resolved) {
		return "invalid_identity_marker", errors.New("identity marker resolves outside candidate root")
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "identity_marker_unreadable", fmt.Errorf("stat identity marker %q: %w", identity.Path, err)
	}
	if !info.Mode().IsRegular() {
		return "invalid_identity_marker", fmt.Errorf("identity marker %q is not a regular file", identity.Path)
	}
	if identity.Contains == "" {
		return "", nil
	}
	file, err := os.Open(resolved)
	if err != nil {
		return "identity_marker_unreadable", fmt.Errorf("open identity marker %q: %w", identity.Path, err)
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxMarkerBytes))
	if err != nil {
		return "identity_marker_unreadable", fmt.Errorf("read identity marker %q: %w", identity.Path, err)
	}
	if !bytes.Contains(data, []byte(identity.Contains)) {
		return "identity_mismatch", fmt.Errorf("identity marker %q does not contain expected identity", identity.Path)
	}
	return "", nil
}

func escapesRoot(path string) bool {
	return path == ".." || strings.HasPrefix(path, ".."+string(filepath.Separator))
}

func withinRoot(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

type selectedRoot struct {
	path string
	info os.FileInfo
}

func sameSelectedRoot(path string, info os.FileInfo, selected []selectedRoot) bool {
	for _, existing := range selected {
		if os.SameFile(info, existing.info) {
			return true
		}
		// Exact cleaned-path equality is a safe fallback for filesystems whose
		// identity metadata is unavailable. Do not case-fold: Windows can host
		// case-sensitive directories.
		if filepath.Clean(path) == filepath.Clean(existing.path) {
			return true
		}
	}
	return false
}

func invalid(name, root, reason string, err error) *InvalidError {
	return &InvalidError{Name: name, Root: root, Reason: reason, Err: err}
}

func preserveFirst(current, next error) error {
	if current != nil {
		return current
	}
	return next
}
