package snapshot

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const QualityCorpusRegistrySchema = "fak.quality-corpus-registry.v1"

type QualityCorpus struct {
	ID                 string   `json:"id"`
	Revision           string   `json:"revision"`
	Provenance         string   `json:"provenance"`
	FailureClass       string   `json:"failure_class"`
	Split              string   `json:"split"`
	ContaminationNotes string   `json:"contamination_notes"`
	Owner              string   `json:"owner"`
	Tier               string   `json:"tier"`
	CostSeconds        float64  `json:"cost_seconds"`
	Cases              []string `json:"cases"`
	Digest             string   `json:"digest"`
}
type QualityCorpusRegistry struct {
	Schema  string          `json:"schema"`
	Corpora []QualityCorpus `json:"corpora"`
}
type CorpusReplay struct {
	Schema          string `json:"schema"`
	CorpusID        string `json:"corpus_id"`
	Revision        string `json:"revision"`
	FirstDivergence string `json:"first_divergence"`
	Scrubbed        bool   `json:"scrubbed"`
}

func CorpusDigest(c QualityCorpus) string {
	c.Digest = ""
	c.Cases = append([]string(nil), c.Cases...)
	sort.Strings(c.Cases)
	b, _ := json.Marshal(c)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
func (r QualityCorpusRegistry) Validate() error {
	if r.Schema != QualityCorpusRegistrySchema {
		return fmt.Errorf("schema: want %s", QualityCorpusRegistrySchema)
	}
	seen := map[string]bool{}
	for i, c := range r.Corpora {
		p := fmt.Sprintf("corpora[%d]", i)
		if strings.TrimSpace(c.ID) == "" || strings.TrimSpace(c.Revision) == "" || strings.TrimSpace(c.Provenance) == "" || strings.TrimSpace(c.FailureClass) == "" || strings.TrimSpace(c.Split) == "" || strings.TrimSpace(c.Owner) == "" || strings.TrimSpace(c.Tier) == "" || c.CostSeconds <= 0 || len(c.Cases) == 0 {
			return fmt.Errorf("%s: incomplete replay-critical metadata", p)
		}
		if seen[c.ID+"@"+c.Revision] {
			return fmt.Errorf("%s: duplicate", p)
		}
		seen[c.ID+"@"+c.Revision] = true
		if c.Digest != CorpusDigest(c) {
			return fmt.Errorf("%s.digest: content mutation or stale digest", p)
		}
	}
	return nil
}
func (r QualityCorpusRegistry) Lookup(id, rev string) (QualityCorpus, CorpusReplay, error) {
	if err := r.Validate(); err != nil {
		return QualityCorpus{}, CorpusReplay{Schema: QualityCorpusRegistrySchema, CorpusID: id, Revision: rev, FirstDivergence: err.Error(), Scrubbed: true}, err
	}
	for _, c := range r.Corpora {
		if c.ID == id && c.Revision == rev {
			return c, CorpusReplay{}, nil
		}
	}
	e := fmt.Errorf("unregistered corpus %s@%s", id, rev)
	return QualityCorpus{}, CorpusReplay{Schema: QualityCorpusRegistrySchema, CorpusID: id, Revision: rev, FirstDivergence: e.Error(), Scrubbed: true}, e
}
