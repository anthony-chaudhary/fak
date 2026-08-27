package studyforge

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Read loads and validates a corpus or partial resume checkpoint.
func Read(path string) (Corpus, error) {
	return readCorpus(path, validateCheckpoint)
}

// ReadResume loads either a current checkpoint or the one exact legacy shape
// that Capture can upgrade on resume. It does not mutate or rewrite the file.
func ReadResume(path string) (Corpus, error) {
	return readCorpus(path, validateResumeCheckpoint)
}

func readCorpus(path string, validate func(Corpus) error) (Corpus, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return Corpus{}, e
	}
	var c Corpus
	if e = json.Unmarshal(b, &c); e != nil {
		return Corpus{}, fmt.Errorf("decode corpus: %w", e)
	}
	if e = validate(c); e != nil {
		return Corpus{}, e
	}
	return c, nil
}

// Write atomically persists a deterministic indented corpus after validation.
func Write(path string, c Corpus) error {
	return writeCorpus(path, c, os.Rename)
}

func writeCorpus(path string, c Corpus, rename func(string, string) error) error {
	sortCorpus(&c)
	refreshChecksums(&c)
	if e := validateCheckpoint(c); e != nil {
		return e
	}
	b, e := json.MarshalIndent(c, "", "  ")
	if e != nil {
		return e
	}
	b = append(b, '\n')
	if e = os.MkdirAll(filepath.Dir(path), 0755); e != nil {
		return e
	}
	tmp := path + ".tmp"
	if e = os.WriteFile(tmp, b, 0600); e != nil {
		return e
	}
	return rename(tmp, path)
}
