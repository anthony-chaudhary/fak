package studyforge

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Read loads and validates a corpus or partial resume checkpoint.
// Transparently handles both raw JSON and gzip-compressed JSON.
func Read(path string) (Corpus, error) {
	return readCorpus(path, validateCheckpoint)
}

// ReadResume loads either a current checkpoint or a narrowly recognized
// historical shape that Capture can upgrade. It does not mutate the file.
// Transparently handles both raw JSON and gzip-compressed JSON.
func ReadResume(path string) (Corpus, error) {
	return readCorpus(path, validateResumeCheckpoint)
}

func readCorpus(path string, validate func(Corpus) error) (Corpus, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return Corpus{}, e
	}
	isGzExt := strings.HasSuffix(path, ".gz")
	isGzMagic := len(b) >= 2 && b[0] == 0x1f && b[1] == 0x8b
	if isGzExt && !isGzMagic {
		return Corpus{}, fmt.Errorf("decode corpus: file %q has .gz extension but is not gzip-compressed", path)
	}
	if isGzMagic {
		zr, err := gzip.NewReader(bytes.NewReader(b))
		if err != nil {
			return Corpus{}, fmt.Errorf("decode corpus: %w", err)
		}
		decompressed, err := io.ReadAll(zr)
		if err != nil {
			_ = zr.Close()
			return Corpus{}, fmt.Errorf("decode corpus: %w", err)
		}
		if err := zr.Close(); err != nil {
			return Corpus{}, fmt.Errorf("decode corpus: %w", err)
		}
		b = decompressed
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
// When path ends with .gz, it persists deterministic byte-identical gzip-compressed JSON.
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
	if strings.HasSuffix(path, ".gz") {
		var buf bytes.Buffer
		gw, err := gzip.NewWriterLevel(&buf, gzip.BestCompression)
		if err != nil {
			gw = gzip.NewWriter(&buf)
		}
		// Deterministic gzip header: no timestamp, no OS, no filename
		gw.Header.Name = ""
		gw.Header.Comment = ""
		gw.Header.ModTime = time.Time{}
		gw.Header.OS = 255
		if _, err := gw.Write(b); err != nil {
			return err
		}
		if err := gw.Close(); err != nil {
			return err
		}
		b = buf.Bytes()
	}
	if e = os.MkdirAll(filepath.Dir(path), 0755); e != nil {
		return e
	}
	tmp := path + ".tmp"
	if e = os.WriteFile(tmp, b, 0600); e != nil {
		return e
	}
	return rename(tmp, path)
}
