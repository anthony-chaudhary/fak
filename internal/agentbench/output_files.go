package agentbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

func prepareOutput(out string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	for _, name := range []string{"manifest.json", "inputs", "events.jsonl", "summary.json", "receipt.json"} {
		_, err := os.Lstat(filepath.Join(out, name))
		if err == nil {
			return fmt.Errorf("artifact %s already exists", name)
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func writeExclusiveBytes(path string, content []byte) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, writeErr := file.Write(content)
	syncErr := file.Sync()
	closeErr := file.Close()
	if writeErr != nil {
		return writeErr
	}
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

func writeExclusiveJSON(path string, value any) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	encodeErr := encoder.Encode(value)
	syncErr := file.Sync()
	closeErr := file.Close()
	if encodeErr != nil {
		return encodeErr
	}
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

func writeAtomicJSON(path string, value any) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, "."+filepath.Base(path)+"-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	keep := false
	defer func() {
		if !keep {
			_ = os.Remove(tempPath)
		}
	}()
	encoder := json.NewEncoder(temp)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return err
	}
	keep = true
	return nil
}

func writeReceiptAtomic(path, out, runID string, artifacts []string) error {
	hashes := make(map[string]string, len(artifacts))
	names := make([]string, 0, len(artifacts))
	for _, artifact := range artifacts {
		name, err := filepath.Rel(out, artifact)
		if err != nil {
			return err
		}
		name = filepath.ToSlash(name)
		digest, err := hashFile(artifact)
		if err != nil {
			return err
		}
		hashes[name] = digest
		names = append(names, name)
	}
	sort.Strings(names)
	runHasher := sha256.New()
	for _, name := range names {
		_, _ = io.WriteString(runHasher, name)
		_, _ = runHasher.Write([]byte{0})
		_, _ = io.WriteString(runHasher, hashes[name])
		_, _ = runHasher.Write([]byte{'\n'})
	}
	value := receipt{
		Schema:         "fak.agentbench.receipt.v1",
		RunID:          runID,
		Algorithm:      "sha256",
		ArtifactSHA256: hashes,
		RunSHA256:      hex.EncodeToString(runHasher.Sum(nil)),
	}
	return writeAtomicJSON(path, value)
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	hasher := sha256.New()
	_, copyErr := io.Copy(hasher, file)
	closeErr := file.Close()
	if copyErr != nil {
		return "", copyErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func printValue(writer io.Writer, value any, asJSON bool) error {
	if asJSON {
		return json.NewEncoder(writer).Encode(value)
	}
	switch value := value.(type) {
	case replayPlan:
		_, err := fmt.Fprintf(writer, "agentbench quick plan: %d scored replay requests plus %d task (max %d turns); C1 concurrency=1, C2 concurrency=2; request ceiling=%d; dispatch=%t\n", value.ScoredRequests, value.TaskFixtures, value.MaxTaskTurns, value.RequestCeiling, value.Dispatch)
		return err
	case *runSummary:
		_, err := fmt.Fprintf(writer, "agentbench quick replay: status=%s terminal=%d/%d service=%s cache=%s task=%s overall=%s\n", value.Status, value.TerminalRequests, value.PlannedRequests, value.ServiceVerdict, value.CacheVerdict, value.TaskVerdict, value.OverallVerdict)
		return err
	default:
		return json.NewEncoder(writer).Encode(value)
	}
}

func boolPointer(value bool) *bool { return &value }

func intPointer(value int) *int { return &value }
