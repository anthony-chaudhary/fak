package workerworktree

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type PreservingIntent struct {
	Message       string
	Paths         []string
	AdmittedPaths []string
}

// ValidatePreservingPaths rejects paths that the normal fence would normalize
// to empty or broader authority. Preservation admission accepts concrete files.
func ValidatePreservingPaths(paths []string) error {
	if len(paths) == 0 {
		return fmt.Errorf("concrete admitted source paths required")
	}
	for _, p := range paths {
		if strings.TrimSpace(p) == "" || p == "." || filepath.IsAbs(p) || strings.Contains(p, "\\") || strings.ContainsAny(p, "*?[\x00\r\n") || p == ".." || strings.HasPrefix(p, "../") || filepath.ToSlash(filepath.Clean(p)) != p || normalizeFencePath(p) != p {
			return fmt.Errorf("admitted paths must be concrete nonempty repository-relative paths: %q", p)
		}
	}
	return nil
}

func validatePreservingIntent(intents []PreservingIntent) error {
	if len(intents) > 1 {
		return fmt.Errorf("one immutable intent allowed")
	}
	if len(intents) == 0 {
		return nil
	}
	in := intents[0]
	if err := ValidatePreservingPaths(in.AdmittedPaths); err != nil {
		return err
	}
	if (strings.TrimSpace(in.Message) == "") != (len(in.Paths) == 0) {
		return fmt.Errorf("intent message and paths must be supplied together")
	}
	if len(in.Paths) == 0 {
		return nil
	}
	if err := ValidatePreservingPaths(in.Paths); err != nil {
		return err
	}
	admitted := map[string]bool{}
	for _, p := range in.AdmittedPaths {
		admitted[p] = true
	}
	for _, p := range in.Paths {
		if !admitted[p] {
			return fmt.Errorf("intent path is outside admitted paths: %s", p)
		}
	}
	_, err := intendedIssueNumber(strings.TrimSpace(in.Message))
	return err
}

func preservingSidecars(target string) []string {
	var paths []string
	for _, p := range []string{OwnerStampPath(target), poolMemberPath(target), poolLegacyMarker(filepath.Dir(target), filepath.Base(target)), intentPath(target), messagePath(target)} {
		paths = append(paths, p, p+".tmp")
	}
	return paths
}

func checkPreservingAbsent(path string, lstat func(string) (os.FileInfo, error)) error {
	if _, err := lstat(path); !os.IsNotExist(err) {
		return fmt.Errorf("existing or unreadable preservation metadata: %s", path)
	}
	return nil
}

func preservingMetadataAbsent(target string) error {
	for _, p := range append(preservingSidecars(target), filepath.Join(target, WorkerLeaseFileName), filepath.Join(target, WorkerLeaseFileName+".tmp")) {
		if err := checkPreservingAbsent(p, os.Lstat); err != nil {
			return err
		}
	}
	return nil
}

// Create final metadata exclusively: no temp rename, replacement, migration,
// remove-on-rename fallback or failure cleanup. A short/failed write is retained.
func writePreservingJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return writePreservingFile(path, append(data, '\n'))
}

func writePreservingFile(path string, data []byte) error {
	return writePreservingFileWith(path, data, func(p string) (io.WriteCloser, error) {
		return os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	})
}

func writePreservingFileWith(path string, data []byte, create func(string) (io.WriteCloser, error)) error {
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		return fmt.Errorf("existing or unreadable metadata: %s", path)
	}
	if _, err := os.Lstat(path + ".tmp"); !os.IsNotExist(err) {
		return fmt.Errorf("existing or unreadable metadata temp: %s", path+".tmp")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	file, err := create(path)
	if err != nil {
		return err
	}
	n, writeErr := file.Write(data)
	if writeErr == nil && n != len(data) {
		writeErr = io.ErrShortWrite
	}
	if writeErr == nil {
		if durable, ok := file.(interface{ Sync() error }); ok {
			writeErr = durable.Sync()
		}
	}
	closeErr := file.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

func writePreservingIntent(target, base string, intents []PreservingIntent) error {
	if err := validatePreservingIntent(intents); err != nil {
		return err
	}
	if len(intents) == 0 || strings.TrimSpace(intents[0].Message) == "" {
		return nil
	}
	in := intents[0]
	issue, err := intendedIssueNumber(strings.TrimSpace(in.Message))
	if err != nil {
		return err
	}
	intent := Intent{Schema: inventorySchema, Path: target, BaseSHA: base, Message: strings.TrimSpace(in.Message), IssueNumber: issue, Paths: canonicalIntentPaths(in.Paths)}
	if err := writePreservingFile(messagePath(target), []byte(intent.Message+"\n")); err != nil {
		return err
	}
	if err := writePreservingJSON(intentPath(target), intent); err != nil {
		return err
	}
	data, err := os.ReadFile(intentPath(target))
	if err != nil {
		return err
	}
	var stored Intent
	if err := json.Unmarshal(data, &stored); err != nil {
		return err
	}
	if stored.Schema != intent.Schema || stored.Path != target || stored.BaseSHA != base || stored.Message != intent.Message || !equalStrings(stored.Paths, intent.Paths) {
		return fmt.Errorf("immutable intent readback mismatch")
	}
	return nil
}

// CanonicalPreservingRoots defines one physical absolute identity for all stat,
// lock, registration and Git operations. Relative worker roots are repo-relative.
func CanonicalPreservingRoots(root, workers string) (string, string, error) {
	absolute, err := filepath.Abs(root)
	if err != nil {
		return "", "", err
	}
	root, err = filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", "", err
	}
	if workers == "" {
		workers = DefaultRoot()
	} else if !filepath.IsAbs(workers) {
		workers = filepath.Join(root, workers)
	}
	workers, err = filepath.Abs(workers)
	if err != nil {
		return "", "", err
	}
	probe := filepath.Clean(workers)
	var missing []string
	for {
		_, err := os.Lstat(probe)
		if err == nil {
			physical, err := filepath.EvalSymlinks(probe)
			if err != nil {
				return "", "", err
			}
			for i := len(missing) - 1; i >= 0; i-- {
				physical = filepath.Join(physical, missing[i])
			}
			return root, physical, nil
		}
		if !os.IsNotExist(err) {
			return "", "", err
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return "", "", fmt.Errorf("worker root identity unavailable")
		}
		missing = append(missing, filepath.Base(probe))
		probe = parent
	}
}
