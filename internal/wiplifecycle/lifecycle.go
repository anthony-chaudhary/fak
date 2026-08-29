package wiplifecycle

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
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
	return BeginWithRunner(root, kind, id, now, wipinventory.GitRunner{})
}

// BeginWithRunner captures the before inventory through the caller's git
// runner. Bounded operational callers use this seam to keep lifecycle evidence
// inside the same wall-clock contract as the mutation it brackets.
func BeginWithRunner(root, kind, id string, now time.Time, runner wipinventory.Runner) (Receipt, error) {
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
	dir, err := operationDir(root, id, runner)
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
	receipt.Before = capture(root, filepath.Join(dir, "before.json"), now, runner)
	if err := writeReceipt(receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}

func Finish(root, id string, now time.Time) (Receipt, error) {
	return FinishWithRunner(root, id, now, wipinventory.GitRunner{})
}

// FinishWithRunner captures the after inventory through the same runner used
// for BeginWithRunner.
func FinishWithRunner(root, id string, now time.Time, runner wipinventory.Runner) (Receipt, error) {
	dir, err := operationDir(root, id, runner)
	if err != nil {
		return Receipt{}, err
	}
	receipt, err := Read(filepath.Join(dir, "receipt.json"))
	if err != nil {
		return Receipt{}, err
	}
	receipt.After = capture(root, filepath.Join(dir, "after.json"), now, runner)
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

// List returns every durable lifecycle receipt in newest-first order. Receipts live
// under the Git directory, so this remains available after the refs or worktrees whose
// transition they witnessed have disappeared.
func List(root string) ([]Receipt, error) {
	store, err := storeDir(root, wipinventory.GitRunner{})
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(store)
	if errors.Is(err, os.ErrNotExist) {
		return []Receipt{}, nil
	}
	if err != nil {
		return nil, err
	}
	receipts := make([]Receipt, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !validID(entry.Name()) {
			continue
		}
		receipt, err := Read(filepath.Join(store, entry.Name(), "receipt.json"))
		if err != nil {
			return nil, fmt.Errorf("read lifecycle receipt %s: %w", entry.Name(), err)
		}
		receipts = append(receipts, receipt)
	}
	sort.Slice(receipts, func(i, j int) bool {
		left := receipts[i].FinishedAt
		if left == "" {
			left = receipts[i].StartedAt
		}
		right := receipts[j].FinishedAt
		if right == "" {
			right = receipts[j].StartedAt
		}
		if left == right {
			return receipts[i].OperationID < receipts[j].OperationID
		}
		return left > right
	})
	return receipts, nil
}

func capture(root, path string, now time.Time, runner wipinventory.Runner) Capture {
	rep := wipinventory.Collect(root, now, runner)
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

func operationDir(root, id string, runner wipinventory.Runner) (string, error) {
	if !validID(id) {
		return "", fmt.Errorf("invalid lifecycle operation id %q", id)
	}
	store, err := storeDir(root, runner)
	if err != nil {
		return "", err
	}
	return filepath.Join(store, id), nil
}

func storeDir(root string, runner wipinventory.Runner) (string, error) {
	gitPath, err := runner.Run(root, "rev-parse", "--git-path", "fak-wip-lifecycle")
	if err != nil {
		return "", err
	}
	store := strings.TrimSpace(string(gitPath))
	if !filepath.IsAbs(store) {
		store = filepath.Join(root, store)
	}
	return store, nil
}

func validID(id string) bool {
	return id != "" && id == filepath.Base(id) && id != "." && id != ".." && !strings.ContainsAny(id, `/\\`)
}
