package disambiguation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const IndexVersionSchema = "fak-disambiguation-index-version/1"

type IndexVersion struct {
	Schema         string `json:"schema"`
	IndexSchema    string `json:"index_schema"`
	SourceRevision string `json:"source_revision"`
	EntryCount     int    `json:"entry_count"`
	ContentSHA256  string `json:"content_sha256"`
}

func CurrentIndexVersion() (IndexVersion, error) { return VersionIndex(publicEntries) }

func VersionIndex(entries []Entry) (IndexVersion, error) {
	generated, err := GenerateIndex(entries)
	if err != nil {
		return IndexVersion{}, err
	}
	var header struct {
		Schema         string `json:"schema"`
		SourceRevision string `json:"source_revision"`
		EntryCount     int    `json:"entry_count"`
	}
	if err := json.Unmarshal(generated, &header); err != nil {
		return IndexVersion{}, fmt.Errorf("read generated index header: %w", err)
	}
	digest := sha256.Sum256(generated)
	return IndexVersion{Schema: IndexVersionSchema, IndexSchema: header.Schema, SourceRevision: header.SourceRevision, EntryCount: header.EntryCount, ContentSHA256: hex.EncodeToString(digest[:])}, nil
}
