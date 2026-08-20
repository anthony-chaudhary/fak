package harnessprofile

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
)

var adapterVersionPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)

func validAdapterVersion(version string) bool {
	return adapterVersionPattern.MatchString(version)
}

// SemanticDigest identifies every runtime-relevant field of a harness adapter while
// excluding AdapterVersion itself. A version therefore names one immutable descriptor
// shape instead of changing the evidence whenever only its label changes.
func SemanticDigest(profile HarnessProfile) (string, error) {
	type semantics struct {
		Name           string             `json:"name"`
		Names          []string           `json:"names"`
		Wire           Wire               `json:"wire"`
		DefaultBaseURL string             `json:"default_base_url,omitempty"`
		Repoint        []RepointMechanism `json:"repoint"`
		Credential     CredentialSource   `json:"credential,omitempty"`
		ConfigHomeGlob string             `json:"config_home_glob,omitempty"`
		Identity       IdentityKind       `json:"identity,omitempty"`
	}
	raw, err := json.Marshal(semantics{
		Name:           profile.Name,
		Names:          append([]string(nil), profile.Names...),
		Wire:           profile.Wire,
		DefaultBaseURL: profile.DefaultBaseURL,
		Repoint:        append([]RepointMechanism(nil), profile.Repoint...),
		Credential:     profile.Credential,
		ConfigHomeGlob: profile.ConfigHomeGlob,
		Identity:       profile.Identity,
	})
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
