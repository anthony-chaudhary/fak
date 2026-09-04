package market

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-extension-descriptor/1"

type SeamKind string

const (
	SeamABI              SeamKind = "abi-engine"
	SeamCompute          SeamKind = "compute-backend"
	SeamTUIPane          SeamKind = "tui-pane"
	SeamQuality          SeamKind = "quality-check"
	SeamTrajectoryScorer SeamKind = "trajectory-scorer"
)

type TrustClass string

const (
	TrustData      TrustClass = "data"
	TrustUntrusted TrustClass = "untrusted"
	TrustCompiled  TrustClass = "trusted-compiled"
)

type ErrorBehavior string

const (
	ErrorClosed  ErrorBehavior = "closed"
	ErrorOpen    ErrorBehavior = "open"
	ErrorIsolate ErrorBehavior = "isolate"
)

type Compatibility struct {
	ABI string `json:"abi"`
	Min int    `json:"min"`
	Max int    `json:"max"`
}

type Witness struct {
	Required     bool     `json:"required"`
	Command      []string `json:"command,omitempty"`
	ResultSHA256 string   `json:"result_sha256,omitempty"`
}

type Descriptor struct {
	Schema         string        `json:"schema"`
	ID             string        `json:"id"`
	Seam           SeamKind      `json:"seam"`
	Module         string        `json:"module"`
	Compatibility  Compatibility `json:"compatibility"`
	Artifact       string        `json:"artifact"`
	ArtifactSHA256 string        `json:"artifact_sha256"`
	Trust          TrustClass    `json:"trust"`
	OnError        ErrorBehavior `json:"on_error"`
	Capabilities   []string      `json:"capabilities,omitempty"`
	Witness        Witness       `json:"witness"`
}

type Catalog struct {
	Schema     string       `json:"schema"`
	Extensions []Descriptor `json:"extensions"`
}

// Adapter identifies an existing extension family without importing or
// initializing it. It translates registry-specific values into inert metadata.
type Adapter struct {
	Seam SeamKind
	ABI  string
}

var (
	// ABIAdapter describes internal/abi Engine registrants.
	ABIAdapter = Adapter{SeamABI, "fak-abi"}
	// ComputeAdapter describes internal/compute Backend values.
	ComputeAdapter = Adapter{SeamCompute, "fak-compute"}
	// TUIPaneAdapter describes internal/tuiplugin Pane values.
	TUIPaneAdapter = Adapter{SeamTUIPane, "fak-tui-pane"}
	// QualityAdapter describes internal/quality Check values.
	QualityAdapter = Adapter{SeamQuality, "fak-quality"}
	// TrajectoryScorerAdapter describes internal/trajctl Scorer values.
	TrajectoryScorerAdapter = Adapter{SeamTrajectoryScorer, "fak-trajectory-scorer"}
)

// Descriptor stamps seam-specific identity onto inert metadata. It does not import,
// initialize, or execute the described extension.
func (a Adapter) Descriptor(d Descriptor) Descriptor {
	d.Schema = Schema
	d.Seam = a.Seam
	d.Compatibility.ABI = a.ABI
	return d
}

type VerifyOptions struct {
	ABIVersions map[string]int
	Root        string
	Timeout     time.Duration
}

type Report struct {
	Schema      string   `json:"schema"`
	Valid       bool     `json:"valid"`
	Descriptors int      `json:"descriptors"`
	Verified    []string `json:"verified"`
}

var idRE = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$`)
var moduleRE = regexp.MustCompile(`^[^@[:space:]]+@r[0-9]+\+g[0-9a-f]{7,40}$`)
var digestRE = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Parse enumerates and structurally validates inert descriptor JSON. It never
// loads an artifact or runs registration/witness code.
//
// Invariant: market state transitions are fail-closed and deterministic across all parsers.
// Guard: reject duplicate extension identifiers, malformed schemas, and trailing payload bytes.
func Parse(data []byte, versions map[string]int) (Catalog, error) {
	var c Catalog
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return Catalog{}, fmt.Errorf("descriptor JSON: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return Catalog{}, errors.New("descriptor JSON: trailing value")
		}
		return Catalog{}, fmt.Errorf("descriptor JSON trailer: %w", err)
	}
	if c.Schema != Schema {
		return Catalog{}, fmt.Errorf("catalog schema %q, want %q", c.Schema, Schema)
	}
	seen := map[string]bool{}
	for i := range c.Extensions {
		d := &c.Extensions[i]
		if err := validateDescriptor(*d, versions); err != nil {
			return Catalog{}, fmt.Errorf("extensions[%d]: %w", i, err)
		}
		if seen[d.ID] {
			return Catalog{}, fmt.Errorf("extensions[%d]: duplicate identity %q", i, d.ID)
		}
		seen[d.ID] = true
		sort.Strings(d.Capabilities)
	}
	sort.Slice(c.Extensions, func(i, j int) bool { return c.Extensions[i].ID < c.Extensions[j].ID })
	return c, nil
}

func validateDescriptor(d Descriptor, versions map[string]int) error {
	if d.Schema != Schema {
		return fmt.Errorf("schema %q, want %q", d.Schema, Schema)
	}
	if !idRE.MatchString(d.ID) {
		return fmt.Errorf("invalid namespaced id %q", d.ID)
	}
	if !knownSeam(d.Seam) {
		return fmt.Errorf("unknown seam kind %q", d.Seam)
	}
	if !moduleRE.MatchString(d.Module) {
		return fmt.Errorf("invalid module revision %q", d.Module)
	}
	if d.Compatibility.ABI == "" || d.Compatibility.Min < 1 || d.Compatibility.Max < d.Compatibility.Min {
		return errors.New("invalid ABI compatibility range")
	}
	v, ok := versions[d.Compatibility.ABI]
	if !ok || v < d.Compatibility.Min || v > d.Compatibility.Max {
		return fmt.Errorf("incompatible ABI %q version %d; requires %d..%d", d.Compatibility.ABI, v, d.Compatibility.Min, d.Compatibility.Max)
	}
	if d.Artifact == "" || !digestRE.MatchString(d.ArtifactSHA256) {
		return errors.New("artifact and lowercase sha256 are required")
	}
	if d.Trust != TrustData && d.Trust != TrustUntrusted && d.Trust != TrustCompiled {
		return fmt.Errorf("unknown trust class %q", d.Trust)
	}
	if d.OnError != ErrorClosed && d.OnError != ErrorOpen && d.OnError != ErrorIsolate {
		return fmt.Errorf("unknown error behavior %q", d.OnError)
	}
	seen := map[string]bool{}
	for _, cap := range d.Capabilities {
		if cap == "" || seen[cap] {
			return fmt.Errorf("invalid or duplicate capability %q", cap)
		}
		seen[cap] = true
	}
	if d.Witness.Required && (len(d.Witness.Command) == 0 || !digestRE.MatchString(d.Witness.ResultSHA256)) {
		return errors.New("required witness recipe and result digest are missing")
	}
	return nil
}

func knownSeam(s SeamKind) bool {
	return s == SeamABI || s == SeamCompute || s == SeamTUIPane || s == SeamQuality || s == SeamTrajectoryScorer
}

// Verify re-reads local artifacts and runs each required local witness. Remote
// marketplace metadata is never accepted as proof.
func Verify(ctx context.Context, c Catalog, o VerifyOptions) (Report, error) {
	root := o.Root
	if root == "" {
		root = "."
	}
	timeout := o.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	r := Report{Schema: "fak-extension-conformance/1", Descriptors: len(c.Extensions)}
	for _, d := range c.Extensions {
		artifact, err := rootedPath(root, d.Artifact)
		if err != nil {
			return r, fmt.Errorf("%s artifact: %w", d.ID, err)
		}
		b, err := os.ReadFile(artifact)
		if err != nil {
			return r, fmt.Errorf("%s artifact: %w", d.ID, err)
		}
		if SHA256(b) != d.ArtifactSHA256 {
			return r, fmt.Errorf("%s artifact digest mismatch", d.ID)
		}
		if d.Witness.Required {
			bounded, cancel := context.WithTimeout(ctx, timeout)
			cmd := exec.CommandContext(bounded, d.Witness.Command[0], d.Witness.Command[1:]...)
			cmd.Dir = root
			cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "SYSTEMROOT=" + os.Getenv("SYSTEMROOT")}
			var stdout bytes.Buffer
			cmd.Stdout = &limitedWriter{w: &stdout, remaining: 1 << 20}
			err := cmd.Run()
			out := stdout.Bytes()
			cancel()
			if err != nil {
				return r, fmt.Errorf("%s witness: %w", d.ID, err)
			}
			if SHA256(out) != d.Witness.ResultSHA256 {
				return r, fmt.Errorf("%s witness result digest mismatch", d.ID)
			}
		}
		r.Verified = append(r.Verified, d.ID)
	}
	r.Valid = true
	return r, nil
}

type limitedWriter struct {
	w         *bytes.Buffer
	remaining int64
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, errors.New("witness output exceeds 1 MiB")
	}
	n, err := w.w.Write(p)
	w.remaining -= int64(n)
	return n, err
}

func rootedPath(root, name string) (string, error) {
	if filepath.IsAbs(name) {
		return "", errors.New("absolute artifact path refused")
	}
	clean := filepath.Clean(name)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(os.PathSeparator)) {
		return "", errors.New("artifact escapes catalog root")
	}
	return filepath.Join(root, clean), nil
}

func SHA256(b []byte) string            { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func Marshal(c Catalog) ([]byte, error) { return json.MarshalIndent(c, "", "  ") }
func Module(name string, rev int, sha string) string {
	return fmt.Sprintf("%s@r%d+g%s", strings.TrimSpace(name), rev, sha)
}
