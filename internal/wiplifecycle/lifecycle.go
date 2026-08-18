package wiplifecycle

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

const Schema = "fak-wip-lifecycle/1"

type Capture struct {
	Known      bool   `json:"known"`
	Artifact   string `json:"artifact,omitempty"`
	ObservedAt string `json:"observed_at,omitempty"`
	Error      string `json:"error,omitempty"`
}

type Receipt struct {
	Schema      string  `json:"schema"`
	OperationID string  `json:"operation_id"`
	Kind        string  `json:"kind"`
	Repository  string  `json:"repository"`
	StartedAt   string  `json:"started_at"`
	FinishedAt  string  `json:"finished_at,omitempty"`
	Before      Capture `json:"before"`
	After       Capture `json:"after"`
	ReceiptPath string  `json:"receipt_path"`
}

func NewID(now time.Time) (string, error) {
	var suffix [6]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		return "", err
	}
	return now.UTC().Format("20060102T150405.000000000Z") + "-" + hex.EncodeToString(suffix[:]), nil
}

func Begin(root, kind, id string, now time.Time) (Receipt, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return Receipt{}, err
	}
	kind = strings.TrimSpace(kind)
	if kind == "" {
		return Receipt{}, errors.New("lifecycle kind is required")
	}
	if id == "" {
		id, err = NewID(now)
		if err != nil {
			return Receipt{}, err
		}
	}
	if !validID(id) {
		return Receipt{}, fmt.Errorf("invalid lifecycle operation id %q", id)
	}
	dir, err := operationDir(root, id)
	if err != nil {
		return Receipt{}, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Receipt{}, err
	}
	receipt := Receipt{
		Schema: Schema, OperationID: id, Kind: kind, Repository: filepath.ToSlash(root),
		StartedAt: now.UTC().Format(time.RFC3339Nano), ReceiptPath: filepath.ToSlash(filepath.Join(dir, "receipt.json")),
	}
	receipt.Before = capture(root, filepath.Join(dir, "before.json"), now)
	if err := writeReceipt(receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}

func Finish(root, id string, now time.Time) (Receipt, error) {
	dir, err := operationDir(root, id)
	if err != nil {
		return Receipt{}, err
	}
	receipt, err := Read(filepath.Join(dir, "receipt.json"))
	if err != nil {
		return Receipt{}, err
	}
	receipt.After = capture(root, filepath.Join(dir, "after.json"), now)
	receipt.FinishedAt = now.UTC().Format(time.RFC3339Nano)
	if err := writeReceipt(receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}

func Read(path string) (Receipt, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Receipt{}, err
	}
	var receipt Receipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		return Receipt{}, err
	}
	if receipt.Schema != Schema {
		return Receipt{}, fmt.Errorf("unsupported lifecycle receipt schema %q", receipt.Schema)
	}
	return receipt, nil
}

func capture(root, path string, now time.Time) Capture {
	rep := wipinventory.Collect(root, now, wipinventory.GitRunner{})
	b, err := rep.JSON()
	if err == nil {
		err = atomicWrite(path, append(b, '\n'))
	}
	capture := Capture{Known: err == nil && len(rep.Errors) == 0, Artifact: filepath.ToSlash(path), ObservedAt: rep.ObservedAt.Format(time.RFC3339Nano)}
	if err != nil {
		capture.Error = err.Error()
	} else if len(rep.Errors) > 0 {
		capture.Error = strings.Join(rep.Errors, "; ")
	}
	return capture
}

func writeReceipt(receipt Receipt) error {
	b, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	return atomicWrite(filepath.FromSlash(receipt.ReceiptPath), append(b, '\n'))
}

func atomicWrite(path string, body []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".capture-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(body); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if removeErr := os.Remove(path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
		return removeErr
	}
	return os.Rename(name, path)
}

func operationDir(root, id string) (string, error) {
	if !validID(id) {
		return "", fmt.Errorf("invalid lifecycle operation id %q", id)
	}
	gitPath, err := wipinventory.GitRunner{}.Run(root, "rev-parse", "--git-path", "fak-wip-lifecycle")
	if err != nil {
		return "", err
	}
	store := strings.TrimSpace(string(gitPath))
	if !filepath.IsAbs(store) {
		store = filepath.Join(root, store)
	}
	return filepath.Join(store, id), nil
}

func validID(id string) bool {
	return id != "" && id == filepath.Base(id) && id != "." && id != ".." && !strings.ContainsAny(id, `/\\`)
}
