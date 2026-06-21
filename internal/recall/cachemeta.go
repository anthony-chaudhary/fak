package recall

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// CacheEntry lowers a persisted page-table row into the common cache metadata
// contract without exposing or paging in the page bytes.
func (p Page) CacheEntry(sessionID string) cachemeta.Entry {
	return cachemeta.FromContextPage(cachemeta.ContextPage{
		SessionID:   sessionID,
		Step:        p.Step,
		Role:        p.Role,
		Descriptor:  p.Descriptor,
		Digest:      p.Digest,
		Len:         p.Len,
		Taint:       abi.TaintLabel(p.Taint),
		Quarantined: p.Quarantined,
		QID:         p.QID,
		Reason:      p.Reason,
		Witness:     p.Witness,
		TrustEpoch:  p.TrustEpoch,
	})
}
