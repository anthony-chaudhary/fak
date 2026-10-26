package remotemaint

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const maxRequestBytes = 16 << 10

func ParseRequest(raw []byte) (Request, error) {
	if len(raw) == 0 || len(raw) > maxRequestBytes {
		return Request{}, fmt.Errorf("%w: invalid JSON size", ErrInvalidRequest)
	}
	var req Request
	if err := decodeStrict(raw, &req); err != nil {
		return Request{}, fmt.Errorf("%w: malformed JSON", ErrInvalidRequest)
	}
	if err := validateRequest(req); err != nil {
		return Request{}, err
	}
	return req, nil
}

func decodeStrict(raw []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return errorsForTrailing(err)
	}
	return nil
}

func errorsForTrailing(err error) error {
	if err == nil {
		return fmt.Errorf("multiple JSON values")
	}
	return err
}

func validateRequest(req Request) error {
	if !validName(req.ProfileID, 128) || !isDigest(req.OperationDigest) {
		return fmt.Errorf("%w: invalid profile or digest", ErrInvalidRequest)
	}
	return nil
}

func validateGrant(g Grant, now time.Time) error {
	if g.Operation == OperationMaintainLeaseAcquire || g.Operation == OperationMaintainLeaseRelease {
		return validateReservationGrant(g, now)
	}
	if g.Reservation != nil {
		return fmt.Errorf("%w: unexpected reservation scope", ErrScope)
	}
	if !validName(g.ProfileID, 128) || !isDigest(g.OperationDigest) || !isDigest(g.ContextID) ||
		!validName(g.Target, 512) || !validHostKey(g.HostKey) || (g.Operation != OperationSystemdProfileInstallReload && g.Operation != OperationSystemdProfileReadback) ||
		!isDigest(g.PayloadSHA256) || !validName(g.Holder, 256) || !validName(g.BootID, 256) ||
		!isDigest(g.PriorStateSHA256) {
		return fmt.Errorf("%w: invalid grant identity", ErrScope)
	}
	if !canonicalStrings(g.Paths) || !canonicalPathPrefixes(g.PathPrefixes) || !canonicalStrings(g.Units) ||
		(g.Operation == OperationSystemdProfileInstallReload && g.PathPrefixes == nil) ||
		(g.Operation == OperationSystemdProfileReadback && len(g.PathPrefixes) != 0) ||
		len(g.Paths) == 0 || len(g.Units) == 0 {
		return fmt.Errorf("%w: non-canonical scope", ErrScope)
	}
	for _, p := range g.Paths {
		if !validAbsolutePOSIX(p) {
			return fmt.Errorf("%w: invalid exact path", ErrScope)
		}
	}
	for _, p := range g.PathPrefixes {
		if !validPathPrefix(p) {
			return fmt.Errorf("%w: invalid path prefix", ErrScope)
		}
	}
	for _, unit := range g.Units {
		if !validName(unit, 256) || strings.Contains(unit, "/") {
			return fmt.Errorf("%w: invalid unit", ErrScope)
		}
	}
	if g.ExpiresAt.IsZero() || !g.ExpiresAt.Equal(g.ExpiresAt.UTC()) || !g.ExpiresAt.After(now) {
		return fmt.Errorf("%w: grant", ErrExpired)
	}
	if len(g.LeasePins) == 0 {
		return fmt.Errorf("%w: missing lease pins", ErrScope)
	}
	lastNode := ""
	for _, pin := range g.LeasePins {
		if !validName(pin.NodeID, 256) || !validName(pin.ID, 256) || pin.Epoch == 0 || pin.NodeID <= lastNode {
			return fmt.Errorf("%w: invalid lease pin", ErrScope)
		}
		if pin.ExpiresAt.IsZero() || !pin.ExpiresAt.Equal(pin.ExpiresAt.UTC()) || !pin.ExpiresAt.After(now) || pin.ExpiresAt.Before(g.ExpiresAt) {
			return fmt.Errorf("%w: lease pin", ErrExpired)
		}
		lastNode = pin.NodeID
	}
	return nil
}

func validateReceipt(r Receipt, g Grant, now time.Time) error {
	if err := validateReceiptIdentity(r, g); err != nil {
		return err
	}
	if r.Outcome != OutcomeApplied && r.Outcome != OutcomeUnchanged && r.Outcome != OutcomeRolledBack {
		return fmt.Errorf("%w: invalid success outcome", ErrAuthority)
	}
	if r.BackupDisposition != "" && !validBackupDisposition(r.BackupDisposition) {
		return fmt.Errorf("%w: invalid backup disposition", ErrAuthority)
	}
	if !r.Verified || !validReceiptTimes(r, g, now) {
		return fmt.Errorf("%w: unverified receipt", ErrAuthority)
	}
	if g.Operation == OperationSystemdProfileReadback {
		if err := validateReadback(r, g); err != nil {
			return err
		}
	}
	if g.Reservation != nil {
		if r.Outcome != OutcomeApplied || len(r.ReservationLeases) != len(g.Reservation.NodeIDs) {
			return fmt.Errorf("%w: reservation receipt scope", ErrAuthority)
		}
		if g.Operation == OperationMaintainLeaseRelease {
			if len(g.LeasePins) != len(r.ReservationLeases) {
				return fmt.Errorf("%w: release receipt scope", ErrAuthority)
			}
			for i, pin := range r.ReservationLeases {
				if pin != g.LeasePins[i] {
					return fmt.Errorf("%w: release receipt pin changed", ErrChanged)
				}
			}
			return nil
		}
		if uint64(len(g.Reservation.NodeIDs)) > ^uint64(0)-g.Reservation.ExpectedStoreEpoch {
			return fmt.Errorf("%w: reservation receipt epoch overflow", ErrAuthority)
		}
		seenIDs := make(map[string]bool, len(r.ReservationLeases))
		for i, pin := range r.ReservationLeases {
			if pin.NodeID != g.Reservation.NodeIDs[i] || !validName(pin.ID, 256) || seenIDs[pin.ID] ||
				pin.Epoch != g.Reservation.ExpectedStoreEpoch+uint64(i)+1 ||
				!pin.ExpiresAt.Equal(pin.ExpiresAt.UTC()) || !pin.ExpiresAt.After(r.EndedAt) ||
				pin.ExpiresAt.After(r.EndedAt.Add(time.Duration(g.Reservation.TTLSeconds)*time.Second)) {
				return fmt.Errorf("%w: reservation receipt lease", ErrAuthority)
			}
			seenIDs[pin.ID] = true
		}
	} else if len(r.ReservationLeases) != 0 {
		return fmt.Errorf("%w: unexpected reservation receipt leases", ErrAuthority)
	}
	return nil
}

func validateFailureReceipt(r Receipt, g Grant, now time.Time) error {
	if err := validateReceiptIdentity(r, g); err != nil {
		return err
	}
	if r.Outcome != OutcomeUnknownEffect && r.Outcome != OutcomeRollbackFailed {
		return fmt.Errorf("%w: invalid failure outcome", ErrAuthority)
	}
	if r.Verified || (r.BackupDisposition != BackupUnknown && r.BackupDisposition != BackupRetained && r.BackupDisposition != BackupRestored) {
		return fmt.Errorf("%w: invalid failure evidence", ErrAuthority)
	}
	if !validReceiptTimes(r, g, now) {
		return fmt.Errorf("%w: invalid failure receipt time", ErrAuthority)
	}
	return nil
}

func validateReceiptIdentity(r Receipt, g Grant) error {
	if r.Schema != ReceiptSchema || r.GrantDigest != GrantDigest(g) || r.ProfileID != g.ProfileID ||
		r.OperationDigest != g.OperationDigest || r.ContextID != g.ContextID || r.Target != g.Target ||
		r.HostKey != g.HostKey || r.Operation != g.Operation || r.BootID != g.BootID {
		return fmt.Errorf("%w: receipt identity mismatch", ErrChanged)
	}
	if (g.Reservation == nil || !r.Verified) && len(r.ReservationLeases) != 0 {
		return fmt.Errorf("%w: unexpected reservation leases", ErrChanged)
	}
	if r.Readback != nil && (g.Operation != OperationSystemdProfileReadback || !r.Verified) {
		return fmt.Errorf("%w: unexpected readback observations", ErrChanged)
	}
	return nil
}

func validateReadback(r Receipt, g Grant) error {
	b := r.Readback
	if r.Outcome != OutcomeUnchanged || b == nil || !validName(b.Hostname, 253) || strings.ContainsAny(b.Hostname, " /\\@:#?") ||
		len(b.Files) != len(g.Paths) || len(b.Files) > 64 || len(b.UnitGraph) == 0 || len(b.UnitGraph) > 64<<10 || strings.ContainsRune(b.UnitGraph, 0) ||
		(r.BackupDisposition != "" && r.BackupDisposition != BackupNotRequired) {
		return fmt.Errorf("%w: invalid readback observations", ErrAuthority)
	}
	for i, f := range b.Files {
		if f.Path != g.Paths[i] || !isDigest(f.SHA256) || len(f.Mode) != 4 {
			return fmt.Errorf("%w: readback file scope", ErrAuthority)
		}
		for _, digit := range f.Mode {
			if digit < '0' || digit > '7' {
				return fmt.Errorf("%w: readback file mode", ErrAuthority)
			}
		}
	}
	if ReadbackDigest(*b) != g.PriorStateSHA256 {
		return fmt.Errorf("%w: changed readback observations", ErrChanged)
	}
	return nil
}

// ReadbackDigest binds the canonical typed observations, including every graph
// byte, to the trusted grant. Files must already be in exact grant path order.
func ReadbackDigest(readback SystemdProfileReadback) string {
	raw, err := json.Marshal(readback)
	if err != nil {
		panic("remotemaint: readback contains only JSON-marshalable fields")
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func validReceiptTimes(r Receipt, g Grant, now time.Time) bool {
	return !r.StartedAt.IsZero() && !r.EndedAt.IsZero() && r.StartedAt.Equal(r.StartedAt.UTC()) &&
		r.EndedAt.Equal(r.EndedAt.UTC()) && !r.EndedAt.Before(r.StartedAt) &&
		!r.StartedAt.After(g.ExpiresAt) && !r.EndedAt.After(now.Add(time.Minute))
}

func validBackupDisposition(d BackupDisposition) bool {
	return d == BackupUnknown || d == BackupRetained || d == BackupRestored || d == BackupNotRequired
}

func GrantDigest(g Grant) string {
	g = cloneGrant(g)
	g.Paths = sortedClone(g.Paths)
	g.PathPrefixes = sortedPathPrefixes(g.PathPrefixes)
	g.Units = sortedClone(g.Units)
	sort.Slice(g.LeasePins, func(i, j int) bool { return g.LeasePins[i].NodeID < g.LeasePins[j].NodeID })
	g.ExpiresAt = g.ExpiresAt.UTC()
	for i := range g.LeasePins {
		g.LeasePins[i].ExpiresAt = g.LeasePins[i].ExpiresAt.UTC()
	}
	raw, err := json.Marshal(g)
	if err != nil {
		panic("remotemaint: Grant contains only JSON-marshalable fields")
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func cloneGrant(g Grant) Grant {
	g.Paths = append([]string(nil), g.Paths...)
	g.PathPrefixes = append([]PathPrefix(nil), g.PathPrefixes...)
	g.Units = append([]string(nil), g.Units...)
	g.LeasePins = append([]LeasePin(nil), g.LeasePins...)
	if g.Reservation != nil {
		r := *g.Reservation
		r.NodeIDs = append([]string(nil), r.NodeIDs...)
		g.Reservation = &r
	}
	return g
}

func validateReservationGrant(g Grant, now time.Time) error {
	r := g.Reservation
	if r == nil || !validName(g.ProfileID, 128) || !isDigest(g.OperationDigest) || !isDigest(g.ContextID) || !validName(g.Holder, 256) ||
		g.Target != "" || g.HostKey != "" || g.BootID != "" || g.PayloadSHA256 != "" || g.PriorStateSHA256 != "" ||
		len(g.Units) != 0 || len(g.PathPrefixes) != 0 {
		return fmt.Errorf("%w: invalid reservation identity", ErrScope)
	}
	if !filepath.IsAbs(r.WorkspaceRoot) || filepath.Clean(r.WorkspaceRoot) != r.WorkspaceRoot ||
		r.Kind != "maintain" || !validName(r.CatalogGeneration, 256) ||
		len(r.NodeIDs) == 0 || !canonicalStrings(r.NodeIDs) || len(g.Paths) == 0 || !canonicalStrings(g.Paths) {
		return fmt.Errorf("%w: invalid reservation scope", ErrScope)
	}
	switch g.Operation {
	case OperationMaintainLeaseAcquire:
		if len(g.LeasePins) != 0 || r.TTLSeconds <= 0 || r.TTLSeconds > 86400 || uint64(len(r.NodeIDs)) > ^uint64(0)-r.ExpectedStoreEpoch {
			return fmt.Errorf("%w: invalid acquire scope or epoch overflow", ErrScope)
		}
	case OperationMaintainLeaseRelease:
		if r.TTLSeconds != 0 || len(g.LeasePins) != len(r.NodeIDs) {
			return fmt.Errorf("%w: invalid release scope", ErrScope)
		}
	default:
		return fmt.Errorf("%w: unsupported reservation operation", ErrScope)
	}
	for _, node := range r.NodeIDs {
		if !validName(node, 256) {
			return fmt.Errorf("%w: invalid reservation node", ErrScope)
		}
	}
	for _, p := range g.Paths {
		if !filepath.IsAbs(p) || filepath.Clean(p) != p {
			return fmt.Errorf("%w: invalid reservation path", ErrScope)
		}
	}
	if g.ExpiresAt.IsZero() || !g.ExpiresAt.Equal(g.ExpiresAt.UTC()) || !g.ExpiresAt.After(now) {
		return fmt.Errorf("%w: reservation grant", ErrExpired)
	}
	if g.Operation == OperationMaintainLeaseRelease {
		seenIDs := make(map[string]bool, len(g.LeasePins))
		for i, pin := range g.LeasePins {
			if pin.NodeID != r.NodeIDs[i] || !validName(pin.ID, 256) || seenIDs[pin.ID] || pin.Epoch == 0 || pin.Epoch > r.ExpectedStoreEpoch {
				return fmt.Errorf("%w: invalid release pin", ErrScope)
			}
			// Release removes exact existing rows; their expiry is identity,
			// not current authority. The fresh grant above still bounds cleanup.
			if pin.ExpiresAt.IsZero() || pin.ExpiresAt.Location() != time.UTC {
				return fmt.Errorf("%w: release pin", ErrExpired)
			}
			seenIDs[pin.ID] = true
		}
	}
	return nil
}

func normalizeGrant(g Grant) Grant {
	g = cloneGrant(g)
	g.Paths = sortedClone(g.Paths)
	g.PathPrefixes = sortedPathPrefixes(g.PathPrefixes)
	g.Units = sortedClone(g.Units)
	sort.Slice(g.LeasePins, func(i, j int) bool { return g.LeasePins[i].NodeID < g.LeasePins[j].NodeID })
	g.ExpiresAt = g.ExpiresAt.UTC()
	for i := range g.LeasePins {
		g.LeasePins[i].ExpiresAt = g.LeasePins[i].ExpiresAt.UTC()
	}
	return g
}

func sortedClone(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func canonicalStrings(v []string) bool {
	for i, s := range v {
		if s == "" || (i > 0 && v[i-1] >= s) {
			return false
		}
	}
	return true
}

func validName(s string, max int) bool {
	if s == "" || len(s) > max || strings.TrimSpace(s) != s {
		return false
	}
	for _, r := range s {
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}

func isDigest(s string) bool {
	if len(s) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil && strings.ToLower(s) == s
}

func validHostKey(s string) bool {
	if !strings.HasPrefix(s, "SHA256:") {
		return false
	}
	b64 := strings.TrimPrefix(s, "SHA256:")
	if len(b64) != 43 || strings.Contains(b64, "=") {
		return false
	}
	_, err := base64.RawStdEncoding.DecodeString(b64)
	return err == nil
}

func validAbsolutePOSIX(p string) bool {
	return strings.HasPrefix(p, "/") && p != "/" && path.Clean(p) == p && !strings.Contains(p, "\\")
}

func validPathPrefix(p PathPrefix) bool {
	if !validAbsolutePOSIX(p.Directory) || p.FilenamePrefix == "" || len(p.FilenamePrefix) > 255 || strings.TrimSpace(p.FilenamePrefix) != p.FilenamePrefix {
		return false
	}
	if strings.ContainsAny(p.FilenamePrefix, "/\\*?[]{}") || strings.Contains(p.FilenamePrefix, "..") {
		return false
	}
	for _, r := range p.FilenamePrefix {
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}

func sortedPathPrefixes(in []PathPrefix) []PathPrefix {
	out := append([]PathPrefix(nil), in...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Directory != out[j].Directory {
			return out[i].Directory < out[j].Directory
		}
		return out[i].FilenamePrefix < out[j].FilenamePrefix
	})
	return out
}

func canonicalPathPrefixes(v []PathPrefix) bool {
	for i, prefix := range v {
		if !validPathPrefix(prefix) {
			return false
		}
		if i > 0 {
			previous := v[i-1]
			if previous.Directory > prefix.Directory ||
				(previous.Directory == prefix.Directory && previous.FilenamePrefix >= prefix.FilenamePrefix) {
				return false
			}
		}
	}
	return true
}
