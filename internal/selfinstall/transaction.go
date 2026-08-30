package selfinstall

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Copy declares one source-to-target activation in a transaction.
type Copy struct {
	Source string
	Target string
}

// TransactionResult is the outcome of RunTransaction.
type TransactionResult interface {
	transactionResult()
}

// Updated means every target was activated.
type Updated struct {
	Attempted      int
	Changed        int
	Err            error
	RollbackErrors []error
}

// RolledBack means activation failed and every changed target was restored.
type RolledBack struct {
	Attempted      int
	Changed        int
	Err            error
	RollbackErrors []error
}

// RollbackFailed means activation failed and at least one changed target could not be restored.
type RollbackFailed struct {
	Attempted      int
	Changed        int
	Err            error
	RollbackErrors []error
}

func (Updated) transactionResult()        {}
func (RolledBack) transactionResult()     {}
func (RollbackFailed) transactionResult() {}

type preparedCopy struct {
	copy      Copy
	candidate string
	snapshot  string
}

// RunTransaction stages every candidate and snapshots every target before activating
// targets in lexical path order. The swapper must consume its source only on success.
func RunTransaction(copies []Copy, swap Swapper) TransactionResult {
	return runTransaction(copies, "", swap)
}

// RunLaunchTransaction is RunTransaction plus the stable-launch replacement
// boundary for launchTarget. The state is published only after every candidate
// and snapshot is complete, immediately before the first activation, and stays
// active through rollback.
func RunLaunchTransaction(copies []Copy, launchTarget string, swap Swapper) TransactionResult {
	return runTransaction(copies, launchTarget, swap)
}

func runTransaction(copies []Copy, launchTarget string, swap Swapper) TransactionResult {
	ordered, err := validateCopies(copies)
	if err != nil {
		return RolledBack{Err: err}
	}
	if strings.TrimSpace(launchTarget) != "" {
		launchTarget, err = filepath.Abs(filepath.Clean(launchTarget))
		if err != nil {
			return RolledBack{Err: fmt.Errorf("launch target: %w", err)}
		}
		found := false
		for _, item := range ordered {
			if sameTransactionPath(item.Target, launchTarget) {
				found = true
				break
			}
		}
		if !found {
			return RolledBack{Err: fmt.Errorf("launch target %q is not in the transaction", launchTarget)}
		}
	}

	prepared := make([]preparedCopy, 0, len(ordered))
	defer func() {
		for _, item := range prepared {
			_ = os.Remove(item.candidate)
			_ = os.Remove(item.snapshot)
		}
	}()

	for _, item := range ordered {
		equal, err := ArtifactsEqual(item.Source, item.Target)
		if err != nil {
			return RolledBack{Err: fmt.Errorf("compare %q: %w", item.Target, err)}
		}
		if equal {
			continue
		}
		candidate, err := stageCopy(item.Source, item.Target, "stage")
		if err != nil {
			return RolledBack{Err: fmt.Errorf("stage %q: %w", item.Target, err)}
		}
		prepared = append(prepared, preparedCopy{copy: item, candidate: candidate})

		snapshot, err := stageCopy(item.Target, item.Target, "snapshot")
		if err != nil {
			return RolledBack{Err: fmt.Errorf("snapshot %q: %w", item.Target, err)}
		}
		prepared[len(prepared)-1].snapshot = snapshot
	}

	if launchTarget != "" && len(prepared) > 0 {
		finish, err := BeginLaunchTransaction(launchTarget)
		if err != nil {
			return RolledBack{Err: fmt.Errorf("publish launch transaction: %w", err)}
		}
		defer finish()
	}

	changed := 0
	for i := range prepared {
		attempted := i + 1
		if err := swap(prepared[i].candidate, prepared[i].copy.Target); err != nil {
			activationErr := fmt.Errorf("activate %q: %w", prepared[i].copy.Target, err)
			rollbackErrors := rollback(prepared[:changed], swap)
			if len(rollbackErrors) != 0 {
				return RollbackFailed{
					Attempted:      attempted,
					Changed:        changed,
					Err:            activationErr,
					RollbackErrors: rollbackErrors,
				}
			}
			return RolledBack{Attempted: attempted, Changed: changed, Err: activationErr}
		}
		changed++
	}

	return Updated{Attempted: len(ordered), Changed: changed}
}

func sameTransactionPath(a, b string) bool {
	if filepath.Separator == '\\' {
		return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
	}
	return filepath.Clean(a) == filepath.Clean(b)
}

func validateCopies(copies []Copy) ([]Copy, error) {
	if len(copies) == 0 {
		return nil, errors.New("at least one copy is required")
	}

	ordered := append([]Copy(nil), copies...)
	seen := make(map[string]struct{}, len(ordered))
	for i := range ordered {
		if strings.TrimSpace(ordered[i].Source) == "" || strings.TrimSpace(ordered[i].Target) == "" {
			return nil, fmt.Errorf("copy %d: source and target are required", i)
		}

		target, err := filepath.Abs(ordered[i].Target)
		if err != nil {
			return nil, fmt.Errorf("copy %d target: %w", i, err)
		}
		target = filepath.Clean(target)
		key := target
		if filepath.Separator == '\\' {
			key = strings.ToLower(key)
		}
		if _, ok := seen[key]; ok {
			return nil, fmt.Errorf("duplicate target %q", ordered[i].Target)
		}
		seen[key] = struct{}{}
		ordered[i].Target = target

		source, err := filepath.Abs(ordered[i].Source)
		if err != nil {
			return nil, fmt.Errorf("copy %d source: %w", i, err)
		}
		ordered[i].Source = filepath.Clean(source)
	}

	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Target < ordered[j].Target })
	for _, item := range ordered {
		if _, err := os.Stat(item.Source); err != nil {
			return nil, fmt.Errorf("source %q: %w", item.Source, err)
		}
		if _, err := os.Stat(item.Target); err != nil {
			return nil, fmt.Errorf("target %q: %w", item.Target, err)
		}
	}
	return ordered, nil
}

func stageCopy(source, target, kind string) (path string, err error) {
	in, err := os.Open(source)
	if err != nil {
		return "", err
	}
	defer in.Close()

	info, err := in.Stat()
	if err != nil {
		return "", err
	}
	out, err := os.CreateTemp(filepath.Dir(target), "."+filepath.Base(target)+".selfinstall-"+kind+"-*")
	if err != nil {
		return "", err
	}
	path = out.Name()
	defer func() {
		closeErr := out.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			_ = os.Remove(path)
			path = ""
		}
	}()

	if err = out.Chmod(info.Mode().Perm()); err != nil {
		return path, err
	}
	if _, err = io.Copy(out, in); err != nil {
		return path, err
	}
	if err = out.Sync(); err != nil {
		return path, err
	}
	return path, nil
}

func rollback(changed []preparedCopy, swap Swapper) []error {
	var rollbackErrors []error
	for i := len(changed) - 1; i >= 0; i-- {
		item := changed[i]
		candidate, err := stageCopy(item.snapshot, item.copy.Target, "rollback")
		if err != nil {
			rollbackErrors = append(rollbackErrors, fmt.Errorf("stage rollback %q: %w", item.copy.Target, err))
			continue
		}
		if err := swap(candidate, item.copy.Target); err != nil {
			rollbackErrors = append(rollbackErrors, fmt.Errorf("rollback %q: %w", item.copy.Target, err))
		}
		_ = os.Remove(candidate)
	}
	return rollbackErrors
}
