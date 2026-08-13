package armbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// CorpusSchema tags the corpus file the runner consumes. It is a separate
// artifact from the manifest on purpose: the manifest pins the corpus by HASH,
// so the tasks themselves can be produced by the upstream fixture importer
// (#6677) and swapped without editing the experiment description — and any swap
// that is not reflected in the pinned hash shows up as a changed identity.
const CorpusSchema = "fak.armbench.corpus/1"

// CorpusFile is the on-disk corpus.
type CorpusFile struct {
	Schema string `json:"schema"`
	ID     string `json:"id"`
	Tasks  []Task `json:"tasks"`
}

// UnmarshalCorpus parses a corpus file, refusing an unknown schema tag or an
// unknown field rather than silently measuring a shape it does not understand.
func UnmarshalCorpus(b []byte) (*CorpusFile, error) {
	var c CorpusFile
	if err := decodeStrict(b, &c); err != nil {
		return nil, refuse(ReasonManifestInvalid, "decode corpus: %v", err)
	}
	if c.Schema != CorpusSchema {
		return nil, refuse(ReasonManifestInvalid, "corpus schema %q is not %q", c.Schema, CorpusSchema)
	}
	if strings.TrimSpace(c.ID) == "" {
		return nil, refuse(ReasonManifestInvalid, "corpus id is empty")
	}
	if err := validateTasks(c.Tasks); err != nil {
		return nil, err
	}
	return &c, nil
}

// HashTasks returns the manifest corpus hash: sha256 over encoding/json's
// canonical encoding of the ORDERED task slice. Struct fields have a fixed
// order and no maps, so another Go consumer can reproduce it byte-for-byte.
func HashTasks(tasks []Task) string {
	b, _ := json.Marshal(tasks) // Task contains only JSON-marshallable scalars.
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// ValidateCorpus binds a loaded corpus to the manifest instead of trusting the
// manifest's hash label. A same-sized but edited task set is incomparable.
func ValidateCorpus(m *Manifest, c *CorpusFile) error {
	if m == nil || c == nil {
		return refuse(ReasonManifestInvalid, "manifest and corpus are required")
	}
	if err := validateTasks(c.Tasks); err != nil {
		return err
	}
	if c.ID != m.Corpus.ID {
		return refuse(ReasonIncomparableManifest, "corpus id %q does not match manifest corpus.id %q", c.ID, m.Corpus.ID)
	}
	return validateTasksAgainstManifest(m, c.Tasks)
}

func validateTasksAgainstManifest(m *Manifest, tasks []Task) error {
	if err := validateTasks(tasks); err != nil {
		return err
	}
	if len(tasks) != m.Corpus.TaskCount {
		return refuse(ReasonIncomparableManifest, "corpus supplied %d task(s) but manifest.corpus.task_count pins %d", len(tasks), m.Corpus.TaskCount)
	}
	if got := HashTasks(tasks); got != m.Corpus.Hash {
		return refuse(ReasonIncomparableManifest, "corpus content hashes to %s but manifest.corpus.hash pins %s", got, m.Corpus.Hash)
	}
	return nil
}

func validateTasks(tasks []Task) error {
	if len(tasks) == 0 {
		return refuse(ReasonManifestInvalid, "corpus is empty — there is nothing to measure")
	}
	seen := map[string]bool{}
	for i, t := range tasks {
		if strings.TrimSpace(t.ID) == "" {
			return refuse(ReasonManifestInvalid, "corpus tasks[%d].id is empty", i)
		}
		if seen[t.ID] {
			return refuse(ReasonManifestInvalid, "corpus tasks[%d].id %q is declared twice — duplicate task ids collide in the resume key", i, t.ID)
		}
		seen[t.ID] = true
	}
	return nil
}

func decodeStrict(b []byte, dst any) error {
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing JSON value")
		}
		return fmt.Errorf("trailing data: %w", err)
	}
	return nil
}

// MarshalCorpus renders a corpus file as strict, stable JSON.
func MarshalCorpus(c *CorpusFile) ([]byte, error) {
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// DemoCorpusFile is the demo corpus in its on-disk shape, so `--emit-demo`
// writes a pair an operator can run unedited.
func DemoCorpusFile() *CorpusFile {
	return &CorpusFile{Schema: CorpusSchema, ID: DemoManifest().Corpus.ID, Tasks: DemoCorpus()}
}

// MarshalManifest renders a manifest as strict, stable JSON.
func MarshalManifest(m *Manifest) ([]byte, error) {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
