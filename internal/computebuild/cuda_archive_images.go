package computebuild

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

var (
	ErrCUDAImageAuditToolMissing = errors.New("computebuild: selected CUDA toolkit has no cuobjdump")
	ErrCUDAImageAuditInvalid     = errors.New("computebuild: CUDA archive image audit failed")
)

// CUDAArchiveImageAudit proves image coverage of one archive, independently of GPU execution.
type CUDAArchiveImageAudit struct {
	Schema        string   `json:"schema"`
	ArchivePath   string   `json:"archive_path"`
	ArchiveSHA256 string   `json:"archive_sha256"`
	SASS          []string `json:"sass"`
	PTX           []string `json:"ptx"`
	ELFImageCount int      `json:"elf_image_count"`
	PTXImageCount int      `json:"ptx_image_count"`
}

const cudaImageListingLimit = 1 << 20

var cudaImageListing = regexp.MustCompile(`^(ELF|PTX) file[ \t]+[1-9][0-9]*:[ \t]+(.+)\.(sm_[0-9]+[af]?)\.(cubin|ptx)$`)

func cudaArchiveImageRequirements(repoRoot, arch string) (sass, ptx []string, err error) {
	args, _, err := LoadCUDAArchMatrix(repoRoot, arch)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: architecture matrix: %v", ErrCUDAImageAuditInvalid, err)
	}
	for i := 1; i < len(args); i += 2 {
		_, code, ok := strings.Cut(args[i], ",code=")
		if !ok {
			return nil, nil, fmt.Errorf("%w: invalid gencode declaration", ErrCUDAImageAuditInvalid)
		}
		if strings.HasPrefix(code, "sm_") {
			sass = append(sass, code)
		} else if strings.HasPrefix(code, "compute_") {
			ptx = append(ptx, code)
		} else {
			return nil, nil, fmt.Errorf("%w: invalid image architecture", ErrCUDAImageAuditInvalid)
		}
	}
	sortCUDAImageArchitectures(sass)
	sortCUDAImageArchitectures(ptx)
	return sass, ptx, nil
}

// cudaListedSASSArch is the base name some cuobjdump releases (12.9) list for a
// family-specific sm_NNNf cubin, whose ELF arch flags equal sm_NNN's; sm_NNNa and
// plain targets have no alias.
func cudaListedSASSArch(code string) string {
	if base, suffix, ok := SplitCUDAArchSuffix(code); ok && suffix == "f" {
		return base
	}
	return code
}

// cudaImageSetMatches compares an observed image listing with the required one,
// accepting the base-name listing of sm_NNNf images. That alias cannot tell an
// sm_NNNf cubin from a plain sm_NNN one, so a caller that relies on it must confirm
// the family images with cudaFatbinELFArches.
func cudaImageSetMatches(kind string, observed, want []string) bool {
	if strings.Join(observed, ",") == strings.Join(want, ",") {
		return true
	}
	if kind != "ELF" {
		return false
	}
	seen := make(map[string]bool, len(want))
	var listed []string
	for _, arch := range want {
		if arch = cudaListedSASSArch(arch); !seen[arch] {
			seen[arch] = true
			listed = append(listed, arch)
		}
	}
	sortCUDAImageArchitectures(listed)
	return strings.Join(observed, ",") == strings.Join(listed, ",")
}

// cudaNeedsFamilyMarker reports whether observed matches want only through the
// sm_NNNf base-name alias.
func cudaNeedsFamilyMarker(observed, want []string) bool {
	return strings.Join(observed, ",") != strings.Join(want, ",") && cudaImageSetMatches("ELF", observed, want)
}

// cudaFatbinELFArches collects the arch line of every "Fatbin elf code:" header in a
// cuobjdump --dump-elf-symbols stream. The header keeps the family suffix that the
// --list-elf cubin names drop, so it is the positive witness for sm_NNNf images.
// Only short header lines are kept, so a symbol dump of any size stays bounded.
type cudaFatbinELFArches struct {
	partial []byte
	inELF   bool
	set     map[string]bool
	invalid string
}

var cudaFatbinELFArch = regexp.MustCompile(`^sm_[0-9]+[af]?$`)

func (w *cudaFatbinELFArches) Write(p []byte) (int, error) {
	n := len(p)
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			w.keep(p)
			break
		}
		w.keep(p[:i])
		w.line()
		p = p[i+1:]
	}
	return n, nil
}

func (w *cudaFatbinELFArches) keep(p []byte) {
	if room := 256 - len(w.partial); room > 0 {
		w.partial = append(w.partial, p[:min(room, len(p))]...)
	}
}

func (w *cudaFatbinELFArches) line() {
	t := strings.TrimSpace(string(w.partial))
	w.partial = w.partial[:0]
	switch {
	case t == "Fatbin elf code:":
		w.inELF = true
	case strings.HasPrefix(t, "Fatbin "):
		w.inELF = false
	case w.inELF && strings.HasPrefix(t, "arch = "):
		arch := strings.TrimPrefix(t, "arch = ")
		if !cudaFatbinELFArch.MatchString(arch) && w.invalid == "" {
			w.invalid = arch
		}
		if w.set == nil {
			w.set = make(map[string]bool)
		}
		w.set[arch] = true
		w.inELF = false
	}
}

// arches returns the sorted header architectures, refusing a stream with no ELF
// header or a malformed one.
func (w *cudaFatbinELFArches) arches() ([]string, error) {
	if len(w.partial) > 0 {
		w.line()
	}
	if w.invalid != "" {
		return nil, fmt.Errorf("malformed fatbin ELF arch %q", w.invalid)
	}
	if len(w.set) == 0 {
		return nil, errors.New("no fatbin ELF headers")
	}
	arches := make([]string, 0, len(w.set))
	for arch := range w.set {
		arches = append(arches, arch)
	}
	sortCUDAImageArchitectures(arches)
	return arches, nil
}

// cudaFamilyELFArches runs cuobjdump --dump-elf-symbols on archive and returns its
// fatbin ELF header architectures.
func cudaFamilyELFArches(ctx context.Context, tool, archive string, tc *Toolchain) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, tool, "--dump-elf-symbols", archive)
	cmd.WaitDelay = 2 * time.Second
	if tc != nil && len(tc.VsDevEnv) > 0 {
		cmd.Env = MergeEnviron(os.Environ(), tc.VsDevEnv)
	}
	headers := &cudaFatbinELFArches{}
	diagnostic := &cudaImageOutput{limit: 4096}
	cmd.Stdout, cmd.Stderr = headers, diagnostic
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%w: cuobjdump --dump-elf-symbols: %w (%s)", ErrCUDAImageAuditInvalid, err, strings.TrimSpace(string(diagnostic.data)))
	}
	arches, err := headers.arches()
	if err != nil {
		return nil, fmt.Errorf("%w: cuobjdump --dump-elf-symbols: %v", ErrCUDAImageAuditInvalid, err)
	}
	return arches, nil
}

func sortCUDAImageArchitectures(arches []string) {
	sort.Slice(arches, func(i, j int) bool {
		a, aSuffix, aOK := cudaImageArchitectureKey(arches[i])
		b, bSuffix, bOK := cudaImageArchitectureKey(arches[j])
		if aOK != bOK {
			// Ordering must not add validation to the requested architecture matrix.
			return aOK
		}
		if aOK {
			if len(a) != len(b) {
				return len(a) < len(b)
			}
			if a != b {
				return a < b
			}
			if aSuffix != bSuffix {
				return aSuffix < bSuffix
			}
		}
		return arches[i] < arches[j]
	})
}

func cudaImageArchitectureKey(arch string) (digits, suffix string, ok bool) {
	prefix, digits, found := strings.Cut(arch, "_")
	if !found || (prefix != "sm" && prefix != "compute") || digits == "" {
		return "", "", false
	}
	if last := digits[len(digits)-1]; last == 'a' || last == 'f' {
		digits, suffix = digits[:len(digits)-1], digits[len(digits)-1:]
	}
	if digits == "" {
		return "", "", false
	}
	for i := range digits {
		if digits[i] < '0' || digits[i] > '9' {
			return "", "", false
		}
	}
	// Compare digit lengths, then digits, without overflowing or changing token identity.
	return strings.TrimLeft(digits, "0"), suffix, true
}

func cudaImageAuditTool(tc *Toolchain) (string, error) {
	name := "cuobjdump"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	var directory string
	if tc != nil && tc.NVCC != "" {
		compiler, err := exec.LookPath(tc.NVCC)
		if err != nil {
			return "", fmt.Errorf("%w: compiler: %v", ErrCUDAImageAuditToolMissing, err)
		}
		directory = filepath.Dir(compiler)
	} else if tc != nil && tc.CUDAHome != "" {
		directory = filepath.Join(tc.CUDAHome, "bin")
	}
	tool := filepath.Join(directory, name)
	if directory == "" {
		return "", fmt.Errorf("%w: no nvcc or CUDAHome selected", ErrCUDAImageAuditToolMissing)
	}
	if !FileExists(tool) {
		return "", fmt.Errorf("%w: expected %s beside the selected nvcc", ErrCUDAImageAuditToolMissing, tool)
	}
	return tool, nil
}

// CUDAPreflightCuobjdumpMissing is the closed refusal token for a selected
// CUDA toolkit that cannot audit the archive it would build.
const CUDAPreflightCuobjdumpMissing = "CUDA_PREFLIGHT_CUOBJDUMP_MISSING"

// CUDAPreflightRefusal is a CUDA build refused before any kernel compiles.
// Token is closed (CUDAPreflightCuobjdumpMissing); Err keeps the sentinel.
type CUDAPreflightRefusal struct {
	Token string
	Err   error
}

func (r *CUDAPreflightRefusal) Error() string { return r.Token + ": " + r.Err.Error() }

func (r *CUDAPreflightRefusal) Unwrap() error { return r.Err }

// RequireCUDAImageAuditTool refuses before compiling when cuobjdump is missing beside
// the selected nvcc. Without it, the archive image audit fails only after
// every kernel has compiled.
func RequireCUDAImageAuditTool(tc *Toolchain) error {
	if _, err := cudaImageAuditTool(tc); err != nil {
		return &CUDAPreflightRefusal{Token: CUDAPreflightCuobjdumpMissing, Err: err}
	}
	return nil
}

type cudaImageOutput struct {
	data     []byte
	limit    int
	exceeded bool
}

func (b *cudaImageOutput) Write(p []byte) (int, error) {
	n := len(p)
	remaining := b.limit - len(b.data)
	if len(p) > remaining {
		p = p[:remaining]
		b.exceeded = true
	}
	b.data = append(b.data, p...)
	return n, nil
}

func cudaImageList(ctx context.Context, tool, flag, archive string, tc *Toolchain) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, tool, flag, archive)
	cmd.WaitDelay = 2 * time.Second
	if tc != nil && len(tc.VsDevEnv) > 0 {
		cmd.Env = MergeEnviron(os.Environ(), tc.VsDevEnv)
	}
	out := &cudaImageOutput{limit: cudaImageListingLimit}
	diagnostic := &cudaImageOutput{limit: 4096}
	cmd.Stdout, cmd.Stderr = out, diagnostic
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%w: cuobjdump %s: %w (%s)", ErrCUDAImageAuditInvalid, flag, err, strings.TrimSpace(string(diagnostic.data)))
	}
	if out.exceeded || diagnostic.exceeded {
		return nil, fmt.Errorf("%w: cuobjdump %s output exceeds limit", ErrCUDAImageAuditInvalid, flag)
	}
	return out.data, nil
}

func parseCUDAImageList(data []byte, kind string) ([]string, int, error) {
	set := make(map[string]bool)
	count := 0
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "File archived in ") || strings.HasPrefix(line, "cuobjdump info ") || (strings.HasPrefix(line, "member ") && strings.HasSuffix(line, ":")) {
			continue
		}
		parts := cudaImageListing.FindStringSubmatch(line)
		if parts == nil || parts[1] != kind || (kind == "ELF" && parts[4] != "cubin") || (kind == "PTX" && parts[4] != "ptx") {
			return nil, 0, fmt.Errorf("%w: malformed %s listing", ErrCUDAImageAuditInvalid, kind)
		}
		arch := parts[3]
		if kind == "PTX" {
			arch = "compute_" + strings.TrimPrefix(arch, "sm_")
		}
		set[arch] = true
		count++
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, fmt.Errorf("%w: reading %s listing: %v", ErrCUDAImageAuditInvalid, kind, err)
	}
	arches := make([]string, 0, len(set))
	for arch := range set {
		arches = append(arches, arch)
	}
	sortCUDAImageArchitectures(arches)
	return arches, count, nil
}

func cudaArchiveSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("%w: open archive: %v", ErrCUDAImageAuditInvalid, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("%w: hash archive: %v", ErrCUDAImageAuditInvalid, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// AuditCUDAArchiveImages checks actual SASS and PTX images against the selected build matrix.
func AuditCUDAArchiveImages(ctx context.Context, tc *Toolchain, repoRoot, arch, archive string) (*CUDAArchiveImageAudit, error) {
	wantSASS, wantPTX, err := cudaArchiveImageRequirements(repoRoot, arch)
	if err != nil {
		return nil, err
	}
	if plan, ok := ctx.Value(cudaArchSpecificKey{}).(*cudaArchSpecificOptions); ok {
		wantSASS = append(wantSASS, plan.arches...)
		sortCUDAImageArchitectures(wantSASS)
	}
	if _, ok := ctx.Value(cudaDeepGEMMKey{}).(*cudaDeepGEMMOptions); ok {
		for _, arch := range []string{"sm_100f", "sm_103f"} {
			found := false
			for _, existing := range wantSASS {
				if arch == existing {
					found = true
					break
				}
			}
			if !found {
				wantSASS = append(wantSASS, arch)
			}
		}
		sortCUDAImageArchitectures(wantSASS)
	}
	return auditCUDAImages(ctx, tc, archive, wantSASS, wantPTX)
}

func auditCUDAImages(ctx context.Context, tc *Toolchain, archive string, wantSASS, wantPTX []string) (*CUDAArchiveImageAudit, error) {
	tool, err := cudaImageAuditTool(tc)
	if err != nil {
		return nil, err
	}
	before, err := cudaArchiveSHA256(archive)
	if err != nil {
		return nil, err
	}
	audit := &CUDAArchiveImageAudit{Schema: "fak.cuda-archive-images/1", ArchivePath: archive, ArchiveSHA256: before}
	for _, spec := range []struct {
		kind, flag string
		want       []string
	}{{"ELF", "--list-elf", wantSASS}, {"PTX", "--list-ptx", wantPTX}} {
		data, err := cudaImageList(ctx, tool, spec.flag, archive, tc)
		if err != nil {
			return nil, err
		}
		arches, count, err := parseCUDAImageList(data, spec.kind)
		if err != nil {
			return nil, err
		}
		if !cudaImageSetMatches(spec.kind, arches, spec.want) {
			return nil, fmt.Errorf("%w: %s images: observed %v, require %v", ErrCUDAImageAuditInvalid, spec.kind, arches, spec.want)
		}
		if spec.kind == "ELF" && cudaNeedsFamilyMarker(arches, spec.want) {
			listed := arches
			if arches, err = cudaFamilyELFArches(ctx, tool, archive, tc); err != nil {
				return nil, err
			}
			if strings.Join(arches, ",") != strings.Join(spec.want, ",") {
				return nil, fmt.Errorf("%w: ELF images listed %v carry fatbin arches %v, require %v", ErrCUDAImageAuditInvalid, listed, arches, spec.want)
			}
		}
		if spec.kind == "ELF" {
			audit.SASS, audit.ELFImageCount = arches, count
		} else {
			audit.PTX, audit.PTXImageCount = arches, count
		}
	}
	after, err := cudaArchiveSHA256(archive)
	if err != nil {
		return nil, err
	}
	if before != after {
		return nil, fmt.Errorf("%w: archive changed during inspection", ErrCUDAImageAuditInvalid)
	}
	return audit, nil
}

func archiveCUDAWithImageAudit(ctx context.Context, tc *Toolchain, repoRoot, pkgDir, arch string, objs []string, stdout io.Writer) (*BuildArtifact, error) {
	artifact, err := ArchiveCUDALib(ctx, tc, pkgDir, objs, stdout)
	if err != nil {
		return nil, err
	}
	// The archive is already replaced on disk; stamp it before the audit so a refused audit
	// cannot leave the previous digest keying a stale link.
	digest, changed, err := WriteCUDAArchiveStamp(pkgDir, artifact.Path)
	if err != nil {
		return nil, err
	}
	if stdout != nil {
		fmt.Fprintf(stdout, "[cuda] archive stamp %s digest %.12s (changed=%t)\n", CUDAArchiveStampFile, digest, changed)
	}
	artifact.CUDAImages, err = AuditCUDAArchiveImages(ctx, tc, repoRoot, arch, artifact.Path)
	if err != nil {
		return nil, err
	}
	artifact.SHA256 = artifact.CUDAImages.ArchiveSHA256
	if stdout != nil {
		fmt.Fprintf(stdout, "[cuda] archive images verified: SASS %v, PTX %v\n", artifact.CUDAImages.SASS, artifact.CUDAImages.PTX)
	}
	return artifact, nil
}
