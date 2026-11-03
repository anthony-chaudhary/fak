package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// loadExplicitServeTokenizer is the existing --tokenizer file/directory route.
// Its error is fatal at resolveServeTokenizer; no embedded/mock fallback occurs.
// Both legacy BPE and the explicitly versioned llama-SPM JSON use LoadJSON.
func loadExplicitServeTokenizer(path string) (*tokenizer.Tokenizer, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "tokenizer.json")
	}
	return tokenizer.LoadJSON(path)
}

// loadAutomaticServeDirectoryTokenizer preserves legacy BPE directory loading.
// The new native profile requires an explicit --tokenizer selection, including
// when that selection names this same directory. Recognized profile failures
// must be fatal at resolveServeTokenizer, not a silent MockPlanner fallback.
func loadAutomaticServeDirectoryTokenizer(path string) (*tokenizer.Tokenizer, error) {
	tok, err := tokenizer.LoadJSON(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	if tok.IsSPM() {
		return nil, fmt.Errorf("%w: native profile requires explicit --tokenizer", tokenizer.ErrNativeSPMProfile)
	}
	return tok, nil
}
