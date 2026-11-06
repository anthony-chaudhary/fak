package pricebook

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"time"
)

// VersionStamp returns the hex-encoded SHA-256 digest of the book's
// deterministic serialization: the built table with models sorted by ID and
// each rate in its canonical null-or-number form. Two books that price
// identically stamp identically regardless of model order; any change to
// any card changes the stamp. The stamp is the value a usage record's
// PriceBookVersion field points at.
//
// A nil book stamps as the empty string: no book, no version.
func (b *PriceBook) VersionStamp() string {
	if b == nil {
		return ""
	}
	models := make([]canonicalCard, 0, len(b.Models))
	for _, m := range b.Models {
		models = append(models, canonicalCard{
			ID:             m.ID,
			Version:        m.Version,
			SourceRevision: m.SourceRevision,
			SourceTime:     m.SourceTime.UTC().Format(time.RFC3339Nano),
			Currency:       m.Currency,
			Units:          m.Units,
			FreshInputRate: ratePtr(m.FreshInputRate),
			CacheReadRate:  ratePtr(m.CacheReadRate),
			CacheWriteRate: ratePtr(m.CacheWriteRate),
			OutputRate:     ratePtr(m.OutputRate),
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	buf, _ := json.Marshal(canonicalBook{SchemaVersion: b.SchemaVersion, Models: models})
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:])
}

// ratePtr projects a Rate onto its canonical wire value: nil for unpriced,
// the value otherwise.
func ratePtr(r Rate) *float64 {
	if r.IsUnpriced() {
		return nil
	}
	v := r.Value
	return &v
}

// canonicalBook and canonicalCard are the deterministic serialization shape
// VersionStamp digests: fixed field order, no omitempty, models sorted by
// ID, rates as null-or-number. They are internal to the stamp and are not
// the operator-file format.
type canonicalBook struct {
	SchemaVersion int             `json:"schema_version"`
	Models        []canonicalCard `json:"models"`
}

type canonicalCard struct {
	ID             string   `json:"id"`
	Version        int      `json:"version"`
	SourceRevision string   `json:"source_revision"`
	SourceTime     string   `json:"source_time"`
	Currency       string   `json:"currency"`
	Units          Units    `json:"units"`
	FreshInputRate *float64 `json:"fresh_input_rate"`
	CacheReadRate  *float64 `json:"cache_read_rate"`
	CacheWriteRate *float64 `json:"cache_write_rate"`
	OutputRate     *float64 `json:"output_rate"`
}
