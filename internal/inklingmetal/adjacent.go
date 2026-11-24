package inklingmetal

import (
	"errors"
	"os"
	"path/filepath"
)

func adjacentDraft(path string) (string, error) {
	dir := filepath.Join(filepath.Dir(path), "Inkling-2-Mini-DFlash2")
	if _, err := os.Stat(dir); errors.Is(err, os.ErrNotExist) {
		return "", nil
	} else if err != nil {
		return "", err
	}
	return dir, nil
}
