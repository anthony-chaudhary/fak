package recall

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const (
	DreamActionDuplicateAlias     = "duplicate_alias"
	DreamActionPruneUnreferenced  = "prune_unreferenced_cas"
	DreamActionRepairDescriptor   = "repair_sealed_descriptor"
	DreamActionSealRefutedWitness = "seal_refuted_witness"
	DreamActionSealTightenedGate  = "seal_tightened_rescreen"
	DreamActionRefreshDescriptor  = "refresh_descriptor"
)

// DreamOptions controls the offline cleanup pass. With OutputDir empty, Dream is a
// dry-run report. With OutputDir set, it writes a cleaned copy of the core image.
type DreamOptions struct {
	OutputDir string
}

// DreamAction records one deterministic cleanup or consolidation decision.
type DreamAction struct {
	Kind   string `json:"kind"`
	Step   int    `json:"step,omitempty"`
	Digest string `json:"digest,omitempty"`
	Bytes  int64  `json:"bytes,omitempty"`
	Reason string `json:"reason,omitempty"`
	Detail string `json:"detail,omitempty"`
}

// DreamReport is the "sleep pass" ledger: what the pass found, what it changed,
// and the before/after accounting. It carries no page bytes.
type DreamReport struct {
	InputDir  string `json:"input_dir"`
	OutputDir string `json:"output_dir,omitempty"`
	DryRun    bool   `json:"dry_run"`

	Before Stats `json:"before"`
	After  Stats `json:"after"`

	Actions []DreamAction `json:"actions"`

	DuplicateAliases    int   `json:"duplicate_aliases"`
	PrunedBlobs         int   `json:"pruned_blobs"`
	ReclaimedBytes      int64 `json:"reclaimed_bytes"`
	DescriptorRepairs   int   `json:"descriptor_repairs"`
	DescriptorRefreshes int   `json:"descriptor_refreshes"`
	RevokedSeals        int   `json:"revoked_seals"`
	TightenedSeals      int   `json:"tightened_seals"`

	Invariant string `json:"invariant"`
}

// Dream runs the deterministic "human dream" cleanup pass over a finished session
// core image. It never summarizes with a model and never trusts the old manifest
// blindly: benign pages are re-screened, witness-refuted pages are pre-sealed, safe
// sealed descriptors are regenerated, duplicate page aliases are surfaced, and
// unreferenced CAS blobs are pruned from the output image.
func Dream(ctx context.Context, dir string, opt DreamOptions) (DreamReport, error) {
	s, err := Load(dir)
	if err != nil {
		return DreamReport{}, err
	}

	pages := s.Pages()
	cleared := copyBoolMap(s.Manifest.Cleared)
	cas := copyCAS(s.cas)
	before := statsOf(s.Manifest, cas)
	report := DreamReport{
		InputDir:  dir,
		DryRun:    opt.OutputDir == "",
		Before:    before,
		Invariant: "the output image is still loaded through recall.Load; every later page-in still runs the witness gate plus fresh content re-screen",
	}
	if opt.OutputDir != "" {
		report.OutputDir = opt.OutputDir
		if samePath(dir, opt.OutputDir) {
			return DreamReport{}, fmt.Errorf("recall: dream output dir must differ from input dir %q", dir)
		}
	}

	nextQ := maxQID(pages)
	seen := map[string]int{}
	referenced := map[string]bool{}

	for i := range pages {
		p := &pages[i]
		referenced[p.Digest] = true
		if first, ok := seen[p.Digest]; ok {
			report.DuplicateAliases++
			report.Actions = append(report.Actions, DreamAction{
				Kind: DreamActionDuplicateAlias, Step: p.Step, Digest: short(p.Digest), Bytes: p.Len,
				Detail: fmt.Sprintf("same content address as step %d; page table keeps the alias, CAS stores bytes once", first),
			})
		} else {
			seen[p.Digest] = p.Step
		}

		body, ok := cas[p.Digest]
		if !ok {
			return DreamReport{}, fmt.Errorf("recall: page %d bytes (%s) absent from CAS", p.Step, short(p.Digest))
		}

		if p.Quarantined {
			want := sealedDescriptor(*p, "")
			if p.Descriptor != want {
				p.Descriptor = want
				report.DescriptorRepairs++
				report.Actions = append(report.Actions, DreamAction{
					Kind: DreamActionRepairDescriptor, Step: p.Step, Digest: short(p.Digest), Bytes: p.Len,
					Reason: p.Reason, Detail: "sealed descriptor regenerated from metadata only",
				})
			}
			continue
		}

		if p.Witness != "" && vdso.Default.Revoked(p.Witness) {
			nextQ++
			qid := ensureQID(p.QID, nextQ)
			sealPage(p, qid, abi.ReasonName(abi.ReasonTrustViolation), "refuted witness")
			delete(cleared, qid)
			report.RevokedSeals++
			report.Actions = append(report.Actions, DreamAction{
				Kind: DreamActionSealRefutedWitness, Step: p.Step, Digest: short(p.Digest), Bytes: p.Len,
				Reason: p.Reason, Detail: fmt.Sprintf("witness %q is refuted; page removed from resident recall candidates", p.Witness),
			})
			continue
		}

		if v := s.reScreen(ctx, p.Role, body); v.Kind == abi.VerdictQuarantine {
			nextQ++
			qid := ensureQID(p.QID, nextQ)
			sealPage(p, qid, abi.ReasonName(v.Reason), "tightened re-screen")
			delete(cleared, qid)
			report.TightenedSeals++
			report.Actions = append(report.Actions, DreamAction{
				Kind: DreamActionSealTightenedGate, Step: p.Step, Digest: short(p.Digest), Bytes: p.Len,
				Reason: p.Reason, Detail: "page was resident in the old image but fails the current content gate",
			})
			continue
		}

		want := descriptorOf(p.Role, body)
		if shouldRefreshDescriptor(p.Descriptor, want) {
			p.Descriptor = want
			report.DescriptorRefreshes++
			report.Actions = append(report.Actions, DreamAction{
				Kind: DreamActionRefreshDescriptor, Step: p.Step, Digest: short(p.Digest), Bytes: p.Len,
				Detail: "benign descriptor refreshed from the current exact bytes",
			})
		}
	}

	prunedCAS := map[string][]byte{}
	for d, b := range cas {
		if referenced[d] {
			prunedCAS[d] = b
			continue
		}
		report.PrunedBlobs++
		report.ReclaimedBytes += int64(len(b))
		report.Actions = append(report.Actions, DreamAction{
			Kind: DreamActionPruneUnreferenced, Digest: short(d), Bytes: int64(len(b)),
			Detail: "blob is not referenced by the page table",
		})
	}

	out := s.Manifest
	out.Pages = pages
	out.Cleared = cleared
	report.After = statsOf(out, prunedCAS)

	if opt.OutputDir != "" {
		if err := writeImage(opt.OutputDir, out, prunedCAS); err != nil {
			return DreamReport{}, err
		}
	}
	return report, nil
}

func writeImage(dir string, m Manifest, cas map[string][]byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	mb, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), mb, 0o644); err != nil {
		return err
	}
	cb, err := json.MarshalIndent(cas, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "cas.json"), cb, 0o644)
}

func statsOf(m Manifest, cas map[string][]byte) Stats {
	st := Stats{Version: m.Version, SessionID: m.SessionID, Pages: len(m.Pages)}
	tombstoned := map[int]bool{}
	for _, ch := range m.ContextChanges {
		if ch.Applied && ch.Action == ContextActionTombstone {
			tombstoned[ch.Step] = true
		}
	}
	for _, p := range m.Pages {
		if p.Quarantined {
			st.Quarantined++
		} else {
			st.Benign++
		}
		if tombstoned[p.Step] {
			st.Tombstoned++
		}
	}
	for _, c := range m.Cleared {
		if c {
			st.Cleared++
		}
	}
	for _, b := range cas {
		st.CASBytes += int64(len(b))
	}
	return st
}

func copyBoolMap(in map[string]bool) map[string]bool {
	out := map[string]bool{}
	for k, v := range in {
		out[k] = v
	}
	return out
}

func copyCAS(in map[string][]byte) map[string][]byte {
	out := make(map[string][]byte, len(in))
	for k, v := range in {
		out[k] = append([]byte(nil), v...)
	}
	return out
}

func samePath(a, b string) bool {
	aa, errA := filepath.Abs(a)
	bb, errB := filepath.Abs(b)
	if errA == nil && errB == nil {
		return filepath.Clean(aa) == filepath.Clean(bb)
	}
	return filepath.Clean(a) == filepath.Clean(b)
}

func maxQID(pages []Page) int {
	max := 0
	for _, p := range pages {
		if !strings.HasPrefix(p.QID, "q") {
			continue
		}
		n, err := strconv.Atoi(strings.TrimPrefix(p.QID, "q"))
		if err == nil && n > max {
			max = n
		}
	}
	return max
}

func ensureQID(existing string, next int) string {
	if existing != "" {
		return existing
	}
	return fmt.Sprintf("q%d", next)
}

func sealPage(p *Page, qid, reason, detail string) {
	if reason == "" || reason == abi.ReasonName(abi.ReasonNone) {
		reason = abi.ReasonName(abi.ReasonTrustViolation)
	}
	p.Quarantined = true
	p.QID = qid
	p.Reason = reason
	p.Taint = uint8(abi.TaintQuarantined)
	// Provenance gates the LEARNING (#540): a sealed page cannot RETAIN positive
	// utility. Zeroing here covers both seal paths — witness-refuted and tightened
	// re-screen — so any utility a page accrued while it looked clean is revoked
	// the moment its witness is, never resurrected via the phase-2 re-rank.
	p.Utility = 0
	p.Descriptor = sealedDescriptor(*p, detail)
}

func sealedDescriptor(p Page, detail string) string {
	reason := p.Reason
	if reason == "" {
		reason = abi.ReasonName(abi.ReasonTrustViolation)
	}
	if detail == "" {
		return fmt.Sprintf("%s: [sealed: %s, %d bytes]", p.Role, reason, p.Len)
	}
	return fmt.Sprintf("%s: [sealed: %s, %s, %d bytes]", p.Role, reason, detail, p.Len)
}

func shouldRefreshDescriptor(got, want string) bool {
	g := strings.TrimSpace(got)
	if g == "" || strings.EqualFold(g, "oversize") {
		return true
	}
	return strings.HasSuffix(strings.ToLower(g), ": oversize") && want != ""
}
