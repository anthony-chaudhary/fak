package selfinstall

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

// BeginLaunchTransaction publishes a complete prior executable before the
// replacement window becomes visible to stable launchers.
func BeginLaunchTransaction(target string) (func(), error) {
	target, err := filepath.Abs(target)
	if err != nil {
		return nil, err
	}
	prior := target + ".self-update-prior"
	// stageCopy creates a complete unique file; rename it to the deterministic
	// name only after the copy has been flushed.
	staged, err := stageCopy(target, target, "launch-prior")
	if err != nil {
		return nil, fmt.Errorf("stage launch prior: %w", err)
	}
	_ = os.Remove(prior)
	if err := OSSwap(staged, prior); err != nil {
		_ = os.Remove(staged)
		return nil, err
	}
	b, _ := json.Marshal(struct {
		Target string `json:"target"`
		Prior  string `json:"prior"`
	}{target, prior})
	state := launchshim.UpdateStatePath(target)
	tmp, err := os.CreateTemp(filepath.Dir(state), ".self-update-state-*")
	if err != nil {
		_ = os.Remove(prior)
		return nil, err
	}
	tmpName := tmp.Name()
	if _, err = tmp.Write(b); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err == nil {
		err = OSSwap(tmpName, state)
	}
	if err != nil {
		_ = os.Remove(tmpName)
		_ = os.Remove(prior)
		return nil, err
	}
	// Keep one prior copy after completion. A launcher may have read the state and
	// not reached exec yet; deleting it here would reintroduce the exact race this
	// protocol closes. The next transaction replaces this single bounded copy.
	return func() { _ = os.Remove(state) }, nil
}
