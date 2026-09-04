package harnessdiscover

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessselect"
)

// Schema identifies the v1alpha1 harness discovery manifest format.
const Schema = "fak.harness-discovery/v1alpha1"

// Registry declares trusted cryptographic signers, revoked content digests,
// and discoverable harness source locations.
type Registry struct {
	Schema          string            `json:"schema"`
	TrustedSigners  map[string]string `json:"trusted_signers,omitempty"`
	RevokedDigests  []string          `json:"revoked_digests,omitempty"`
	Sources         []Source          `json:"sources,omitempty"`
	DiscoverRepo    bool              `json:"discover_repo,omitempty"`
	RepoDeclaration string            `json:"repo_declaration,omitempty"`
}

// Source specifies a scoped harness configuration source, its trust level,
// cryptographic signature, and principal admission requirements.
type Source struct {
	ID            string   `json:"id"`
	Scope         string   `json:"scope"`
	Owner         string   `json:"owner"`
	Root          string   `json:"root"`
	Path          string   `json:"path"`
	Trust         string   `json:"trust"`
	Signer        string   `json:"signer,omitempty"`
	Signature     string   `json:"signature,omitempty"`
	RefreshPolicy string   `json:"refresh_policy"`
	Principals    []string `json:"principals,omitempty"`
}

// Candidate represents a validated, scoped harness manifest discovered from a source,
// carrying provenance details including its content digest and trust tier.
type Candidate struct {
	ID            string `json:"id"`
	Scope         string `json:"scope"`
	Owner         string `json:"owner"`
	Source        string `json:"source"`
	Digest        string `json:"digest"`
	Trust         string `json:"trust"`
	Signer        string `json:"signer,omitempty"`
	RefreshPolicy string `json:"refresh_policy"`
	Manifest      string `json:"manifest_schema"`
}

// Result holds the set of discovered candidates ordered deterministically by scope
// and identity, alongside the merged harness selection manifest.
type Result struct {
	Schema     string                 `json:"schema"`
	Candidates []Candidate            `json:"candidates"`
	Manifest   harnessselect.Manifest `json:"manifest"`
}

// Options configures the discovery search paths and caller principal identity.
type Options struct {
	RegistryPath string
	StartPath    string
	Principal    string
}

var scopes = map[string]int{"company": 10, "team": 20, "person": 30, "repo": 40, "project": 50}
var refreshPolicies = map[string]bool{"immutable": true, "session": true, "manual": true}

// Discover resolves, validates, and orders scoped harness declarations from the
// specified registry, verifying cryptographic signatures and content digests.
func Discover(opts Options) (Result, error) {
	registryPath, err := filepath.Abs(opts.RegistryPath)
	if err != nil || strings.TrimSpace(opts.RegistryPath) == "" {
		return Result{}, fmt.Errorf("registry path is required")
	}
	raw, err := os.ReadFile(registryPath)
	if err != nil {
		return Result{}, fmt.Errorf("read registry: %w", err)
	}
	var reg Registry
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&reg); err != nil {
		return Result{}, fmt.Errorf("parse registry: %w", err)
	}
	if reg.Schema != Schema {
		return Result{}, fmt.Errorf("registry schema must be %q", Schema)
	}
	if reg.RepoDeclaration == "" {
		reg.RepoDeclaration = filepath.Join(".fak", "harness.json")
	}
	if filepath.IsAbs(reg.RepoDeclaration) || escapes(reg.RepoDeclaration) {
		return Result{}, fmt.Errorf("repo_declaration must be a confined relative path")
	}

	base := filepath.Dir(registryPath)
	sources := append([]Source(nil), reg.Sources...)
	if reg.DiscoverRepo {
		repo, ok, err := findRepoSource(opts.StartPath, reg.RepoDeclaration)
		if err != nil {
			return Result{}, err
		}
		if ok {
			sources = append(sources, repo)
		}
	}
	return discover(reg, base, sources, opts.Principal)
}

func discover(reg Registry, registryDir string, sources []Source, principal string) (Result, error) {
	seen := map[string]bool{}
	revoked := set(reg.RevokedDigests)
	candidates := make([]Candidate, 0, len(sources))
	layers := make([]harnessselect.Layer, 0)
	for _, src := range sources {
		identity := src.Scope + "/" + src.ID
		if strings.TrimSpace(src.ID) == "" || strings.TrimSpace(src.Owner) == "" {
			return Result{}, fmt.Errorf("source id and owner are required")
		}
		if _, ok := scopes[src.Scope]; !ok {
			return Result{}, fmt.Errorf("source %q has unsupported discovery scope %q", src.ID, src.Scope)
		}
		if seen[identity] {
			return Result{}, fmt.Errorf("duplicate source identity %q", identity)
		}
		seen[identity] = true
		if src.Trust != "managed" && src.Trust != "local" {
			return Result{}, fmt.Errorf("source %q trust must be managed or local", src.ID)
		}
		if !refreshPolicies[src.RefreshPolicy] {
			return Result{}, fmt.Errorf("source %q has unknown refresh_policy %q", src.ID, src.RefreshPolicy)
		}
		if src.Scope == "team" || src.Scope == "person" {
			if strings.TrimSpace(principal) == "" {
				return Result{}, fmt.Errorf("source %q requires an authenticated principal", src.ID)
			}
			if !containsFold(src.Principals, principal) {
				return Result{}, fmt.Errorf("principal %q is not admitted by %s source %q", principal, src.Scope, src.ID)
			}
		}
		path, err := confinedPath(registryDir, src.Root, src.Path)
		if err != nil {
			return Result{}, fmt.Errorf("source %q: %w", src.ID, err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return Result{}, fmt.Errorf("source %q unreadable: %w", src.ID, err)
		}
		sum := sha256.Sum256(raw)
		digest := "sha256:" + hex.EncodeToString(sum[:])
		if revoked[strings.ToLower(digest)] {
			return Result{}, fmt.Errorf("source %q content digest %s is revoked", src.ID, digest)
		}
		if src.Trust == "managed" {
			if err := verifyManaged(reg.TrustedSigners, src, raw); err != nil {
				return Result{}, err
			}
		}
		manifest, err := harnessselect.Parse(raw)
		if err != nil {
			return Result{}, fmt.Errorf("source %q: %w", src.ID, err)
		}
		for _, layer := range manifest.Layers {
			if layer.Scope != src.Scope {
				return Result{}, fmt.Errorf("source %q scope %q cannot declare %q layer %q", src.ID, src.Scope, layer.Scope, layer.ID)
			}
			layers = append(layers, layer)
		}
		candidates = append(candidates, Candidate{ID: src.ID, Scope: src.Scope, Owner: src.Owner, Source: filepath.ToSlash(path), Digest: digest, Trust: src.Trust, Signer: src.Signer, RefreshPolicy: src.RefreshPolicy, Manifest: manifest.Schema})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if scopes[candidates[i].Scope] != scopes[candidates[j].Scope] {
			return scopes[candidates[i].Scope] < scopes[candidates[j].Scope]
		}
		return candidates[i].ID < candidates[j].ID
	})
	sort.Slice(layers, func(i, j int) bool {
		if scopes[layers[i].Scope] != scopes[layers[j].Scope] {
			return scopes[layers[i].Scope] < scopes[layers[j].Scope]
		}
		return layers[i].ID < layers[j].ID
	})
	return Result{Schema: Schema, Candidates: candidates, Manifest: harnessselect.Manifest{Schema: harnessselect.Schema, Layers: layers}}, nil
}

func verifyManaged(keys map[string]string, src Source, raw []byte) error {
	if src.Signer == "" || src.Signature == "" {
		return fmt.Errorf("managed source %q must declare signer and signature", src.ID)
	}
	keyText, ok := keys[src.Signer]
	if !ok {
		return fmt.Errorf("managed source %q signer %q is not trusted", src.ID, src.Signer)
	}
	key, err := base64.StdEncoding.DecodeString(keyText)
	if err != nil || len(key) != ed25519.PublicKeySize {
		return fmt.Errorf("trusted signer %q has invalid Ed25519 public key", src.Signer)
	}
	sig, err := base64.StdEncoding.DecodeString(src.Signature)
	if err != nil || !ed25519.Verify(ed25519.PublicKey(key), raw, sig) {
		return fmt.Errorf("managed source %q signature verification failed", src.ID)
	}
	return nil
}

func confinedPath(registryDir, root, rel string) (string, error) {
	if rel == "" || filepath.IsAbs(rel) || escapes(rel) {
		return "", fmt.Errorf("path must be a confined relative path")
	}
	if root == "" {
		root = "."
	}
	if !filepath.IsAbs(root) {
		root = filepath.Join(registryDir, root)
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", fmt.Errorf("resolve root: %w", err)
	}
	candidate := filepath.Join(resolvedRoot, rel)
	resolved, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	inside, err := filepath.Rel(resolvedRoot, resolved)
	if err != nil || escapes(inside) {
		return "", fmt.Errorf("resolved path escapes declared root")
	}
	return resolved, nil
}

func findRepoSource(start, declaration string) (Source, bool, error) {
	if strings.TrimSpace(start) == "" {
		return Source{}, false, fmt.Errorf("start path is required when discover_repo is enabled")
	}
	start, err := filepath.Abs(start)
	if err != nil {
		return Source{}, false, err
	}
	if info, err := os.Stat(start); err == nil && !info.IsDir() {
		start = filepath.Dir(start)
	}
	start, err = filepath.EvalSymlinks(start)
	if err != nil {
		return Source{}, false, fmt.Errorf("resolve start path: %w", err)
	}
	for dir := start; ; dir = filepath.Dir(dir) {
		candidate := filepath.Join(dir, declaration)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return Source{ID: filepath.Base(dir), Scope: "repo", Owner: "repository", Root: dir, Path: declaration, Trust: "local", RefreshPolicy: "session"}, true, nil
		} else if err != nil && !os.IsNotExist(err) {
			return Source{}, false, fmt.Errorf("inspect repo declaration: %w", err)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return Source{}, false, nil
}

func escapes(path string) bool {
	clean := filepath.Clean(path)
	return clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator))
}

func set(values []string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, value := range values {
		out[strings.ToLower(strings.TrimSpace(value))] = true
	}
	return out
}

func containsFold(values []string, want string) bool {
	for _, value := range values {
		if strings.EqualFold(strings.TrimSpace(value), strings.TrimSpace(want)) {
			return true
		}
	}
	return false
}
