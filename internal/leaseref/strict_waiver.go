package leaseref

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"sort"
	"strings"
	"time"
)

// SnapshotWaiver is explicit risk acceptance for one opaque, non-record blob.
// It does not classify any namespace as metadata or authorize deletion. Callers
// must bind this complete value to their repository and explicit review approval.
type SnapshotWaiver struct {
	Ref         string `json:"ref"`
	ExpectedOID string `json:"expected_oid"`
	Reason      string `json:"reason"`
	Evidence    string `json:"evidence"`
}

func waiverObjectID(oid string) bool {
	if len(oid) != 40 && len(oid) != 64 {
		return false
	}
	_, err := hex.DecodeString(oid)
	return err == nil && oid == strings.ToLower(oid)
}

// waiverRefSnapshot refuses failed enumeration; an absent namespace is a
// successful Git invocation with empty output, never a nonzero exit status.
func (s *Store) waiverRefSnapshot(ctx context.Context) (map[string]string, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname) %(objectname)", refPrefix)
	if err != nil || code != 0 {
		return nil, fmt.Errorf("leaseref: strict waiver enumeration exited %d: %v", code, err)
	}
	refs := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		fields := strings.Split(line, " ")
		if len(fields) != 2 || !strings.HasPrefix(fields[0], refPrefix) || !waiverObjectID(fields[1]) {
			return nil, fmt.Errorf("leaseref: malformed strict waiver enumeration")
		}
		ref := fields[0]
		if !isLeaseRef(ref) {
			continue
		}
		if _, exists := refs[ref]; exists {
			return nil, fmt.Errorf("leaseref: duplicate enumerated ref %s", ref)
		}
		refs[ref] = fields[1]
	}
	return refs, nil
}

// waiverFullRef validates an exact full Git ref, not a Record ID. Legacy
// opaque refs may contain multiple components and '#' while ordinary Record
// IDs remain subject to validID. Namespace classification is checked separately.
func waiverFullRef(ref string) bool {
	if !strings.HasPrefix(ref, refPrefix) || strings.HasSuffix(ref, ".") || strings.Contains(ref, "..") || strings.Contains(ref, "@{") || strings.ContainsAny(ref, " ~^:?*[\\") {
		return false
	}
	for _, c := range []byte(ref) {
		if c < 0x20 || c == 0x7f {
			return false
		}
	}
	for _, part := range strings.Split(ref, "/") {
		if part == "" || strings.HasPrefix(part, ".") || strings.HasSuffix(part, ".lock") {
			return false
		}
	}
	return true
}

// ValidateSnapshotWaivers checks exact syntax and duplicate bindings, not approval.
func ValidateSnapshotWaivers(waivers []SnapshotWaiver) error {
	approved := map[string]SnapshotWaiver{}
	for _, w := range waivers {
		if !waiverFullRef(w.Ref) || !isLeaseRef(w.Ref) || !waiverObjectID(w.ExpectedOID) || strings.TrimSpace(w.Reason) == "" || strings.TrimSpace(w.Evidence) == "" {
			return fmt.Errorf("leaseref: invalid exact snapshot waiver for %q", w.Ref)
		}
		if _, exists := approved[w.Ref]; exists {
			return fmt.Errorf("leaseref: duplicate snapshot waiver for %s", w.Ref)
		}
		approved[w.Ref] = w
	}
	return nil
}

// StrictLiveSnapshotWithWaivers reads immutable blob OIDs and checks the ref
// mapping again before returning. Every waiver must match and be used; known
// records, including invalid records, cannot be exempted. All other records
// undergo the same ID and expiry checks as StrictLiveSnapshot. This observes
// changes during the read, but does not lock ref writers after it returns.
func (s *Store) StrictLiveSnapshotWithWaivers(ctx context.Context, now time.Time, waivers []SnapshotWaiver) (live []Record, expired []string, err error) {
	if s == nil {
		return nil, nil, fmt.Errorf("leaseref: nil store")
	}
	if err := ValidateSnapshotWaivers(waivers); err != nil {
		return nil, nil, err
	}
	approved := map[string]SnapshotWaiver{}
	for _, w := range waivers {
		approved[w.Ref] = w
	}
	refs, err := s.waiverRefSnapshot(ctx)
	if err != nil {
		return nil, nil, err
	}
	for ref, w := range approved {
		if refs[ref] != w.ExpectedOID {
			return nil, nil, fmt.Errorf("leaseref: missing or changed waived ref %s (expected %s, got %s)", ref, w.ExpectedOID, refs[ref])
		}
	}
	names := make([]string, 0, len(refs))
	for ref := range refs {
		names = append(names, ref)
	}
	sort.Strings(names)
	records := []Record{}
	for _, ref := range names {
		oid := refs[ref]
		kind, code, readErr := s.run(ctx, s.dir, "--no-replace-objects", "cat-file", "-t", oid)
		if readErr != nil || code != 0 || strings.TrimSpace(kind) != "blob" {
			return nil, nil, &CorruptRefError{Ref: ref, Err: fmt.Errorf("expected readable blob %s (type %q, code %d): %v", oid, strings.TrimSpace(kind), code, readErr)}
		}
		data, code, readErr := s.run(ctx, s.dir, "--no-replace-objects", "cat-file", "blob", oid)
		if readErr != nil || code != 0 {
			return nil, nil, &CorruptRefError{Ref: ref, Err: fmt.Errorf("read pinned blob %s exited %d: %v", oid, code, readErr)}
		}
		var rec Record
		decodeErr := json.Unmarshal([]byte(data), &rec)
		if _, waived := approved[ref]; waived {
			// Objects may contain partial or future lease records. Never exempt them,
			// even when their fields or JSON are malformed. Null also decodes without
			// an error and must not become an opaque-record bypass.
			if !json.Valid([]byte(data)) || decodeErr == nil || strings.HasPrefix(strings.TrimSpace(data), "{") {
				return nil, nil, fmt.Errorf("leaseref: waiver cannot exempt a lease-shaped record at %s", ref)
			}
			continue
		}
		if decodeErr != nil {
			return nil, nil, &CorruptRefError{Ref: ref, Err: decodeErr}
		}
		expectedID := strings.TrimPrefix(ref, refPrefix)
		if rec.ID == "" {
			rec.ID = expectedID
		} // Preserve readRef's existing compatibility.
		if !validID(rec.ID) || rec.ID != expectedID {
			return nil, nil, &CorruptRefError{Ref: ref, Err: fmt.Errorf("invalid or mismatched lease id %q", rec.ID)}
		}
		records = append(records, rec)
	}
	after, err := s.waiverRefSnapshot(ctx)
	if err != nil {
		return nil, nil, err
	}
	if !maps.Equal(refs, after) {
		return nil, nil, fmt.Errorf("leaseref: lease refs changed during strict waiver snapshot")
	}
	live, expired = liveExpire(records, now, func(r Record) string { return r.ID })
	return live, expired, nil
}
