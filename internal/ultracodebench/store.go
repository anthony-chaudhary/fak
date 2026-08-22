package ultracodebench

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

func ActivationReceiptPath(root, runID, childID string) (string, error) {
	if !activationToken(runID) || !activationToken(childID) {
		return "", fmt.Errorf("activation run_id and child_id must be opaque tokens")
	}
	return filepath.Join(root, "fak-ultracode-activations", runID, childID+".json"), nil
}

func WriteActivation(root string, receipt ActivationReceipt) error {
	if err := receipt.Validate(); err != nil {
		return err
	}
	path, err := ActivationReceiptPath(root, receipt.RunID, receipt.ChildID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".activation-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func ReadActivations(root, runID string) ([]ActivationReceipt, error) {
	if !activationToken(runID) {
		return nil, fmt.Errorf("activation run_id must be an opaque token")
	}
	paths, err := filepath.Glob(filepath.Join(root, "fak-ultracode-activations", runID, "*.json"))
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	out := make([]ActivationReceipt, 0, len(paths))
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		r, err := DecodeActivation(raw)
		if err != nil {
			return nil, fmt.Errorf("read activation %s: %w", filepath.Base(path), err)
		}
		if r.RunID != runID {
			return nil, fmt.Errorf("activation %s belongs to run %s", filepath.Base(path), r.RunID)
		}
		out = append(out, r)
	}
	return out, nil
}
