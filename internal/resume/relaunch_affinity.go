package resume

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

const relaunchCacheAffinityDomain = "fak-resume-cache-affinity\x00"

// RelaunchCacheAffinityEnv is the environment variable the watchdog sets on every
// relaunched `claude --resume` child to carry the transcript's derived cache route
// across the OS relaunch (#4140). An env var (not an argv flag) because neither the
// claude CLI nor the `fak guard` front takes a cache-route flag, and the child env is
// already the watchdog's carrier for per-child launch state (CLAUDE_CONFIG_DIR). The
// value is RelaunchCacheAffinityKey(session) — 32 hex chars, never a credential — so
// it survives the launch broker's secret-floor env sanitizer by construction.
const RelaunchCacheAffinityEnv = "FAK_RESUME_CACHE_AFFINITY"

// RelaunchCacheAffinityKey derives the stable cache route shared by every OS-level
// relaunch of one Claude transcript. A blank UUID cannot name a transcript and
// therefore yields no route.
func RelaunchCacheAffinityKey(transcriptUUID string) string {
	uuid := strings.TrimSpace(transcriptUUID)
	if uuid == "" {
		return ""
	}
	h := sha256.Sum256([]byte(relaunchCacheAffinityDomain + uuid))
	return hex.EncodeToString(h[:])[:32]
}

// RelaunchCacheAffinity preserves the original leaf spelling for callers that
// landed alongside the issue implementation. New launch plumbing may use the
// explicit Key suffix; both names derive the same route.
func RelaunchCacheAffinity(transcriptUUID string) string {
	return RelaunchCacheAffinityKey(transcriptUUID)
}

// RelaunchAffinityRow is the append-only transcript-keyed cache route record.
// Keeping the derived key in the row lets operators audit what route a relaunch
// used while the fold gives launch plumbing a last-row-wins lookup.
type RelaunchAffinityRow struct {
	TS          string `json:"ts,omitempty"`
	Session     string `json:"session"`
	AffinityKey string `json:"affinity_key"`
}

// NewRelaunchAffinityRow derives the append-only ledger row for one OS relaunch of a
// transcript. Pure by construction, mirroring NewRelaunchResetRow: there is no clock in
// the core, so TS (audit-only, ignored by the fold) is left "" for the shell to stamp at
// write time. A blank session yields a row the fold skips (blank key), never a panic.
func NewRelaunchAffinityRow(session string) RelaunchAffinityRow {
	uuid := strings.TrimSpace(session)
	return RelaunchAffinityRow{Session: uuid, AffinityKey: RelaunchCacheAffinityKey(uuid)}
}

// FoldRelaunchAffinity returns the latest valid affinity key per transcript UUID.
func FoldRelaunchAffinity(rows []RelaunchAffinityRow) map[string]string {
	out := make(map[string]string)
	for _, row := range rows {
		session := strings.TrimSpace(row.Session)
		key := strings.TrimSpace(row.AffinityKey)
		if session == "" || key == "" {
			continue
		}
		out[session] = key
	}
	return out
}
