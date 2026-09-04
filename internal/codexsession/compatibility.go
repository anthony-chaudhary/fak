package codexsession

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// CompatibilityReceipt binds a tested Codex binary/protocol pair to the exact
// scrubbed schema fixture used by conformance tests.
type CompatibilityReceipt struct {
	BinaryVersion   string
	ProtocolVersion string
	SchemaDigest    string
	Fixture         string
}

// CompatibilityEnvelope captures the startup capability configuration presented by
// a Codex app-server, containing its binary version, protocol version, schema bytes,
// and registered authority methods.
type CompatibilityEnvelope struct {
	BinaryVersion    string
	ProtocolVersion  string
	Schema           []byte
	AuthorityMethods []string
}

var testedFixtureDigests = map[string]string{
	"minimum-supported.json":      "f6c0daacc1453e5ce750461336ffe09ffbb87b37e09656f32c64d46956be4d02",
	"current-supported.json":      "fce8bb3a8f012bfc0b2f3403e11d24c23234ec3e3a1bef42c4e81c19f612a32d",
	"incompatible-authority.json": "d4252b92b3ad0b7832d23e6922b5f2a0e3e0741540f29192c986c5a961e851fd",
}

type schemaFixture struct {
	Fixture          string   `json:"fixture"`
	BinaryVersion    string   `json:"binaryVersion"`
	ProtocolVersion  string   `json:"protocolVersion"`
	AuthorityMethods []string `json:"authorityMethods"`
}

// CompatibilityError is returned before Codex starts when authority-bearing
// protocol drift falls outside the tested envelope.
type CompatibilityError struct {
	BinaryVersion   string
	ProtocolVersion string
	SchemaDigest    string
	Reason          string
}

// Error formats the compatibility rejection diagnostic detailing mismatched version, protocol, or digest.
func (e *CompatibilityError) Error() string {
	return fmt.Sprintf("codexsession: unsupported Codex compatibility envelope binary=%q protocol=%q schema_sha256=%s: %s; refresh conformance fixtures or roll back Codex", e.BinaryVersion, e.ProtocolVersion, e.SchemaDigest, e.Reason)
}

// LoadCompatibilityReceipt reads, parses, and validates a versioned conformance fixture,
// ensuring its scrubbed schema digest matches tested fixture manifests.
// Precondition: fixture file path must exist, be readable, and contain valid JSON schema bindings.
// Postcondition: returns validated CompatibilityReceipt whose digest matches known tested fixture manifests.
func LoadCompatibilityReceipt(path string) (CompatibilityReceipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return CompatibilityReceipt{}, fmt.Errorf("read compatibility fixture: %w", err)
	}
	var fixture schemaFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return CompatibilityReceipt{}, fmt.Errorf("decode compatibility fixture: %w", err)
	}
	if fixture.Fixture == "" || fixture.BinaryVersion == "" || fixture.ProtocolVersion == "" || len(fixture.AuthorityMethods) == 0 {
		return CompatibilityReceipt{}, fmt.Errorf("compatibility fixture %q is missing a required binding", path)
	}
	digest := digest(data)
	wantDigest, tested := testedFixtureDigests[filepath.Base(path)]
	if !tested || digest != wantDigest {
		return CompatibilityReceipt{}, fmt.Errorf("compatibility fixture %q has untested schema digest %s; refresh the tested matrix", path, digest)
	}
	return CompatibilityReceipt{
		BinaryVersion:   fixture.BinaryVersion,
		ProtocolVersion: fixture.ProtocolVersion,
		SchemaDigest:    digest,
		Fixture:         fixture.Fixture,
	}, nil
}

// CheckCompatibility fails closed when the presented schema or authority
// surface does not exactly match a tested receipt. Callers must invoke this
// before starting the app-server process.
// Precondition: got envelope, tested receipt, and expected authority method slices must be populated.
// Postcondition: returns nil if schema digest and authority methods match tested receipt, error otherwise.
func CheckCompatibility(got CompatibilityEnvelope, tested CompatibilityReceipt, expectedAuthority []string) error {
	gotDigest := digest(got.Schema)
	if got.BinaryVersion != tested.BinaryVersion || got.ProtocolVersion != tested.ProtocolVersion || gotDigest != tested.SchemaDigest {
		return &CompatibilityError{BinaryVersion: got.BinaryVersion, ProtocolVersion: got.ProtocolVersion, SchemaDigest: gotDigest, Reason: "version/protocol/schema digest does not match the tested receipt"}
	}
	gotMethods := append([]string(nil), got.AuthorityMethods...)
	wantMethods := append([]string(nil), expectedAuthority...)
	sort.Strings(gotMethods)
	sort.Strings(wantMethods)
	if strings.Join(gotMethods, "\x00") != strings.Join(wantMethods, "\x00") {
		return &CompatibilityError{BinaryVersion: got.BinaryVersion, ProtocolVersion: got.ProtocolVersion, SchemaDigest: gotDigest, Reason: fmt.Sprintf("authority methods changed: got %v want %v", gotMethods, wantMethods)}
	}
	return nil
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
