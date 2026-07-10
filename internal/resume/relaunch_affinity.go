package resume

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// RelaunchCacheAffinity derives the opaque cache-affinity key used by every
// process launched for the same Claude transcript. The transcript UUID is the
// durable identity available on both sides of an OS-process relaunch, unlike a
// gateway trace ID, which may change when the child reconnects.
//
// An empty UUID has no stable lineage and therefore produces no affinity key.
func RelaunchCacheAffinity(transcriptUUID string) string {
	uuid := strings.TrimSpace(transcriptUUID)
	if uuid == "" {
		return ""
	}
	h := sha256.Sum256([]byte("fak-resume-cache-affinity\x00" + uuid))
	return hex.EncodeToString(h[:])[:32]
}
