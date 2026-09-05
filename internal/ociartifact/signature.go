package ociartifact

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ErrBundleSignatureInvalid is the closed typed token for signature validation failures.
const ErrBundleSignatureInvalid = "BUNDLE_SIGNATURE_INVALID"

// SimpleSigningPayload implements the Cosign simplesigning JSON schema.
type SimpleSigningPayload struct {
	Critical CriticalPayload `json:"critical"`
	Optional map[string]any  `json:"optional,omitempty"`
}

// CriticalPayload carries critical binding metadata for the simple signing verification.
type CriticalPayload struct {
	Identity CriticalIdentity `json:"identity"`
	Image    CriticalImage    `json:"image"`
	Type     string           `json:"type"`
}

// CriticalIdentity designates reference identity for the signed artifact.
type CriticalIdentity struct {
	DockerReference string `json:"docker-reference"`
}

// CriticalImage designates the target manifest digest.
type CriticalImage struct {
	DockerManifestDigest string `json:"docker-manifest-digest"`
}

// SignatureRecord represents a Cosign signature payload record stored inside or alongside a bundle.
type SignatureRecord struct {
	MediaType string `json:"mediaType"`
	Payload   string `json:"payload"`
	Signature string `json:"signature"`
	KeyID     string `json:"key_id,omitempty"`
}

// GenerateEd25519KeyPair generates a new Ed25519 private/public key pair encoded as PKCS#8 / PKIX PEM strings.
func GenerateEd25519KeyPair() (privPEM, pubPEM string, err error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return "", "", err
	}
	privBytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return "", "", err
	}
	privBlock := &pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: privBytes,
	}
	pubBytes, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return "", "", err
	}
	pubBlock := &pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: pubBytes,
	}
	return string(pem.EncodeToMemory(privBlock)), string(pem.EncodeToMemory(pubBlock)), nil
}

// SignArtifact signs an OCI artifact or .fakpack bundle's manifest digest using standard Go crypto (Ed25519 or ECDSA).
// It embeds a Cosign simplesigning payload inside the archive as signature.json (or writes a sidecar if not an archive).
func SignArtifact(bundlePath, privateKeyPEMOrHex string) error {
	privKey, err := parsePrivateKey(privateKeyPEMOrHex)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "key", fmt.Sprintf("failed to parse private key: %v", err))
	}

	fi, err := os.Stat(bundlePath)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "bundle", err.Error())
	}

	if fi.IsDir() {
		manifestPath := filepath.Join(bundlePath, "manifest.json")
		manifestBytes, err := os.ReadFile(manifestPath)
		if err != nil {
			return fail(ErrBundleSignatureInvalid, "sign", "manifest", fmt.Sprintf("manifest.json missing: %v", err))
		}
		sigRec, err := createSignatureRecord(privKey, manifestBytes)
		if err != nil {
			return err
		}
		sigBytes, err := json.MarshalIndent(sigRec, "", "  ")
		if err != nil {
			return fail(ErrBundleSignatureInvalid, "sign", "signature", err.Error())
		}
		sigPath := filepath.Join(bundlePath, "signature.json")
		if err := os.WriteFile(sigPath, sigBytes, 0644); err != nil {
			return fail(ErrBundleSignatureInvalid, "sign", "write", err.Error())
		}
		return nil
	}

	// For archive file (.fakpack or .tar.gz)
	bundleBytes, err := os.ReadFile(bundlePath)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "read", err.Error())
	}

	entries, manifestBytes, err := readTarArchive(bundleBytes)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "archive", fmt.Sprintf("failed to parse archive: %v", err))
	}
	if manifestBytes == nil {
		return fail(ErrBundleSignatureInvalid, "sign", "manifest", "manifest.json not found in archive")
	}

	sigRec, err := createSignatureRecord(privKey, manifestBytes)
	if err != nil {
		return err
	}
	sigBytes, err := json.MarshalIndent(sigRec, "", "  ")
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "signature", err.Error())
	}

	// Update or append signature.json entry
	found := false
	for i := range entries {
		if entries[i].name == "signature.json" {
			entries[i].data = sigBytes
			found = true
			break
		}
	}
	if !found {
		entries = append(entries, archiveEntry{
			name: "signature.json",
			data: sigBytes,
			mode: 0644,
		})
	}

	newBundleBytes, err := writeTarArchive(entries)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "write", err.Error())
	}

	if err := os.WriteFile(bundlePath, newBundleBytes, 0644); err != nil {
		return fail(ErrBundleSignatureInvalid, "sign", "write", err.Error())
	}

	return nil
}

// VerifyArtifact validates an OCI artifact bundle against a verification key.
// It enforces path and key existence checks (EMPTY_PATH, NOT_FOUND, EMPTY_KEY) before Cosign signature verification.
func VerifyArtifact(bundlePath, publicKeyPEMOrHex string) error {
	if bundlePath == "" {
		return fail("EMPTY_PATH", "verify-artifact", "path", "artifact path cannot be empty")
	}
	if _, err := os.Stat(bundlePath); err != nil {
		return fail("NOT_FOUND", "verify-artifact", "path", err.Error())
	}
	if publicKeyPEMOrHex == "" {
		return fail("EMPTY_KEY", "verify-artifact", "verifyKey", "verification key cannot be empty")
	}
	return VerifyArtifactCosign(bundlePath, publicKeyPEMOrHex)
}

// VerifyArtifactCosign verifies the Cosign signature of an OCI artifact or .fakpack bundle against the provided public key.
// Returns an error with typed token BUNDLE_SIGNATURE_INVALID if signature is missing, invalid, or key doesn't match.
func VerifyArtifactCosign(bundlePath, publicKeyPEMOrHex string) error {
	pubKey, err := parsePublicKey(publicKeyPEMOrHex)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "key", fmt.Sprintf("failed to parse public key: %v", err))
	}

	fi, err := os.Stat(bundlePath)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "bundle", err.Error())
	}

	var manifestBytes []byte
	var sigBytes []byte

	if fi.IsDir() {
		manifestBytes, err = os.ReadFile(filepath.Join(bundlePath, "manifest.json"))
		if err != nil {
			return fail(ErrBundleSignatureInvalid, "verify", "manifest", "manifest.json missing from directory")
		}
		sigBytes, err = os.ReadFile(filepath.Join(bundlePath, "signature.json"))
		if err != nil {
			// check sidecar
			sigBytes, err = os.ReadFile(bundlePath + ".sig")
			if err != nil {
				return fail(ErrBundleSignatureInvalid, "verify", "signature", "signature.json missing from directory")
			}
		}
	} else {
		bundleBytes, err := os.ReadFile(bundlePath)
		if err != nil {
			return fail(ErrBundleSignatureInvalid, "verify", "read", err.Error())
		}
		entries, mb, err := readTarArchive(bundleBytes)
		if err != nil {
			return fail(ErrBundleSignatureInvalid, "verify", "archive", fmt.Sprintf("failed to parse archive: %v", err))
		}
		manifestBytes = mb
		if manifestBytes == nil {
			return fail(ErrBundleSignatureInvalid, "verify", "manifest", "manifest.json missing from bundle")
		}
		for _, e := range entries {
			if e.name == "signature.json" {
				sigBytes = e.data
				break
			}
		}
		if sigBytes == nil {
			// Check sidecar <bundlePath>.sig
			sigBytes, _ = os.ReadFile(bundlePath + ".sig")
		}
	}

	if len(sigBytes) == 0 {
		return fail(ErrBundleSignatureInvalid, "verify", "signature", "signature is missing from bundle")
	}

	var sigRec SignatureRecord
	if err := json.Unmarshal(sigBytes, &sigRec); err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "signature", fmt.Sprintf("invalid signature JSON: %v", err))
	}

	if sigRec.MediaType != "" && sigRec.MediaType != SignatureMediaType {
		return fail(ErrBundleSignatureInvalid, "verify", "mediaType", fmt.Sprintf("unexpected signature media type: %s", sigRec.MediaType))
	}

	payloadBytes, err := base64.StdEncoding.DecodeString(sigRec.Payload)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "payload", "invalid base64 in signature payload")
	}

	sig, err := base64.StdEncoding.DecodeString(sigRec.Signature)
	if err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "signature", "invalid base64 signature")
	}

	var payload SimpleSigningPayload
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "payload", fmt.Sprintf("malformed simple signing payload: %v", err))
	}

	manifestDigest := Digest(manifestBytes)
	if payload.Critical.Image.DockerManifestDigest != manifestDigest {
		return fail(ErrBundleSignatureInvalid, "verify", "digest", fmt.Sprintf("manifest digest mismatch: expected %s, got %s", manifestDigest, payload.Critical.Image.DockerManifestDigest))
	}

	if err := verifySignatureBytes(pubKey, payloadBytes, sig); err != nil {
		return fail(ErrBundleSignatureInvalid, "verify", "crypto", err.Error())
	}

	return nil
}

func createSignatureRecord(privKey any, manifestBytes []byte) (*SignatureRecord, error) {
	manifestDigest := Digest(manifestBytes)
	payload := SimpleSigningPayload{
		Critical: CriticalPayload{
			Identity: CriticalIdentity{
				DockerReference: "fak/bundle",
			},
			Image: CriticalImage{
				DockerManifestDigest: manifestDigest,
			},
			Type: "cosign artifact signature",
		},
		Optional: map[string]any{
			"creator":   "fak",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		},
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fail(ErrBundleSignatureInvalid, "sign", "payload", err.Error())
	}

	sigBytes, err := signPayloadBytes(privKey, payloadBytes)
	if err != nil {
		return nil, fail(ErrBundleSignatureInvalid, "sign", "crypto", err.Error())
	}

	return &SignatureRecord{
		MediaType: SignatureMediaType,
		Payload:   base64.StdEncoding.EncodeToString(payloadBytes),
		Signature: base64.StdEncoding.EncodeToString(sigBytes),
	}, nil
}

func parsePrivateKey(input string) (any, error) {
	trimmed := strings.TrimSpace(input)
	if data, err := os.ReadFile(trimmed); err == nil {
		trimmed = strings.TrimSpace(string(data))
	}
	if strings.Contains(trimmed, "-----BEGIN") {
		block, _ := pem.Decode([]byte(trimmed))
		if block == nil {
			return nil, errors.New("failed to decode PEM block")
		}
		key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
		if err == nil {
			return key, nil
		}
		ecKey, err := x509.ParseECPrivateKey(block.Bytes)
		if err == nil {
			return ecKey, nil
		}
		return nil, fmt.Errorf("unsupported PEM private key: %w", err)
	}
	hexBytes, err := hex.DecodeString(trimmed)
	if err == nil {
		if len(hexBytes) == 32 {
			return ed25519.NewKeyFromSeed(hexBytes), nil
		}
		if len(hexBytes) == 64 {
			return ed25519.PrivateKey(hexBytes), nil
		}
	}
	return nil, errors.New("unrecognized private key format (expected PEM or hex)")
}

func parsePublicKey(input string) (any, error) {
	trimmed := strings.TrimSpace(input)
	if data, err := os.ReadFile(trimmed); err == nil {
		trimmed = strings.TrimSpace(string(data))
	}
	if strings.Contains(trimmed, "-----BEGIN") {
		block, _ := pem.Decode([]byte(trimmed))
		if block == nil {
			return nil, errors.New("failed to decode PEM block")
		}
		pub, err := x509.ParsePKIXPublicKey(block.Bytes)
		if err == nil {
			return pub, nil
		}
		return nil, fmt.Errorf("unsupported PEM public key: %w", err)
	}
	hexBytes, err := hex.DecodeString(trimmed)
	if err == nil {
		if len(hexBytes) == 32 {
			return ed25519.PublicKey(hexBytes), nil
		}
	}
	return nil, errors.New("unrecognized public key format (expected PEM or hex)")
}

func signPayloadBytes(privKey any, payload []byte) ([]byte, error) {
	switch k := privKey.(type) {
	case ed25519.PrivateKey:
		return ed25519.Sign(k, payload), nil
	case *ecdsa.PrivateKey:
		h := sha256.Sum256(payload)
		return ecdsa.SignASN1(rand.Reader, k, h[:])
	default:
		return nil, fmt.Errorf("unsupported private key type: %T", privKey)
	}
}

func verifySignatureBytes(pubKey any, payload, sig []byte) error {
	switch k := pubKey.(type) {
	case ed25519.PublicKey:
		if !ed25519.Verify(k, payload, sig) {
			return errors.New("ed25519 signature verification failed")
		}
		return nil
	case *ecdsa.PublicKey:
		h := sha256.Sum256(payload)
		if !ecdsa.VerifyASN1(k, h[:], sig) {
			return errors.New("ecdsa signature verification failed")
		}
		return nil
	default:
		return fmt.Errorf("unsupported public key type: %T", pubKey)
	}
}

type archiveEntry struct {
	name string
	data []byte
	mode int64
}

func readTarArchive(b []byte) ([]archiveEntry, []byte, error) {
	var r io.Reader = bytes.NewReader(b)
	if len(b) >= 2 && b[0] == 0x1f && b[1] == 0x8b {
		gr, err := gzip.NewReader(r)
		if err != nil {
			return nil, nil, err
		}
		defer gr.Close()
		r = gr
	}

	tr := tar.NewReader(r)
	var entries []archiveEntry
	var manifestBytes []byte

	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if hdr.Typeflag == tar.TypeDir {
			continue
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			return nil, nil, err
		}
		cleanName := filepath.ToSlash(hdr.Name)
		entries = append(entries, archiveEntry{
			name: cleanName,
			data: data,
			mode: hdr.Mode,
		})
		if cleanName == "manifest.json" {
			manifestBytes = data
		}
	}
	return entries, manifestBytes, nil
}

func writeTarArchive(entries []archiveEntry) ([]byte, error) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)

	for _, e := range entries {
		hdr := &tar.Header{
			Name:     e.name,
			Mode:     e.mode,
			Size:     int64(len(e.data)),
			ModTime:  time.Now(),
			Typeflag: tar.TypeReg,
		}
		if hdr.Mode == 0 {
			hdr.Mode = 0644
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return nil, err
		}
		if _, err := tw.Write(e.data); err != nil {
			return nil, err
		}
	}

	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
