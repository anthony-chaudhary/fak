// Package logvault captures fak's durable logs — the guard decision journals,
// the harness session stores, the dispatch/dos/loop ledgers — into one central
// vault directory, incrementally and tamper-evidently.
//
// The vault layout is by-source first, original relative paths beneath, so a
// restore is a copy back:
//
//	<vault>/vault-manifest.jsonl                 — this package's own hash-chained ledger
//	<vault>/by-source/<source-id>/<rel-path>     — current mirror of each captured file
//	<vault>/by-source/<source-id>/.history/...   — superseded versions of rewritten files
//
// Every capture appends chained rows to the manifest; the manifest replay is the
// incremental state (no separate state file), and Verify re-derives both the
// chain and the mirror hashes. The chain construction mirrors internal/journal
// (sha256 over 0x1f-delimited fields in declaration order); the row schema is
// logvault's own because journal.Row's pre-image is frozen to guard semantics.
package logvault

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ManifestRow is one durable capture record. Field order up to Note is the
// hash-chain pre-image order — do not reorder without bumping the chain.
type ManifestRow struct {
	Seq        uint64 `json:"seq"`          // monotonic 1-based order anchor
	TSUnixNano int64  `json:"ts_unix_nano"` // wall-clock time anchor
	Op         string `json:"op"`           // capture-full | capture-append | capture-rewrite | skip-error
	Source     string `json:"source"`       // registry source id
	RelPath    string `json:"rel_path"`     // forward-slash path relative to the source root
	Bytes      int64  `json:"bytes"`        // bytes written to the vault by this op
	SizeAfter  int64  `json:"size_after"`   // source file size at capture time
	MTimeUnix  int64  `json:"mtime_unix"`   // source file mtime at capture time (seconds)
	SHA256     string `json:"sha256"`       // full-content hash of the source at capture ("" on skip-error)
	Note       string `json:"note,omitempty"`
	PrevHash   string `json:"prev_hash"` // hash of the previous row ("" at genesis)
	Hash       string `json:"hash"`      // manifestChainHash(PrevHash, this row)
}

// manifestChainHash links a row to its predecessor: sha256 over the previous
// row's hash and this row's content fields (Seq..Note, declaration order),
// 0x1f-delimited so no concatenation collision is possible.
func manifestChainHash(prev string, r ManifestRow) string {
	h := sha256.New()
	io.WriteString(h, prev)
	fmt.Fprintf(h, "\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%d\x1f%d\x1f%d\x1f%s\x1f%s",
		r.Seq, r.TSUnixNano, r.Op, r.Source, r.RelPath,
		r.Bytes, r.SizeAfter, r.MTimeUnix, r.SHA256, r.Note)
	return hex.EncodeToString(h.Sum(nil))
}

// Manifest is the vault's append-only chained ledger.
type Manifest struct {
	f        *os.File
	bw       *bufio.Writer
	path     string
	seq      uint64
	lastHash string
}

// ManifestName is the manifest's file name inside the vault root.
const ManifestName = "vault-manifest.jsonl"

// OpenManifest opens (creating if absent) the vault manifest in append mode,
// recovering the chain head from existing rows so a new capture CONTINUES the
// chain. A torn final line is tolerated (skipped), matching the house posture
// that a damaged tail never bricks startup — Verify reports it instead.
func OpenManifest(vaultDir string) (*Manifest, error) {
	if err := os.MkdirAll(vaultDir, 0o755); err != nil {
		return nil, fmt.Errorf("logvault: mkdir vault: %w", err)
	}
	path := filepath.Join(vaultDir, ManifestName)
	rows, _ := ReadManifestRows(path) // tolerant: missing or torn file is a valid empty/partial history
	m := &Manifest{path: path}
	if n := len(rows); n > 0 {
		m.seq = rows[n-1].Seq
		m.lastHash = rows[n-1].Hash
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("logvault: open manifest: %w", err)
	}
	m.f = f
	m.bw = bufio.NewWriter(f)
	return m, nil
}

// Append stamps the order anchor + chain hash and commits the row.
func (m *Manifest) Append(row ManifestRow) (ManifestRow, error) {
	m.seq++
	row.Seq = m.seq
	row.PrevHash = m.lastHash
	row.Hash = manifestChainHash(row.PrevHash, row)
	b, err := json.Marshal(row)
	if err != nil {
		return row, err
	}
	if _, err := m.bw.Write(b); err != nil {
		return row, err
	}
	if err := m.bw.WriteByte('\n'); err != nil {
		return row, err
	}
	if err := m.bw.Flush(); err != nil {
		return row, err
	}
	m.lastHash = row.Hash
	return row, nil
}

// Close flushes and closes the underlying file.
func (m *Manifest) Close() error {
	if m.bw != nil {
		if err := m.bw.Flush(); err != nil {
			return err
		}
	}
	if m.f != nil {
		return m.f.Close()
	}
	return nil
}

// ReadManifestRows reads every well-formed row; a missing file yields an empty
// history and a torn/corrupt final line is skipped (the tolerant consumer read).
func ReadManifestRows(path string) ([]ManifestRow, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var rows []ManifestRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r ManifestRow
		if err := json.Unmarshal(line, &r); err != nil {
			continue // torn or foreign line: skip, Verify names it
		}
		rows = append(rows, r)
	}
	return rows, sc.Err()
}

// VerifyManifest re-derives the chain and returns the row count, or an error
// naming the first broken link.
func VerifyManifest(path string) (int, error) {
	rows, err := ReadManifestRows(path)
	if err != nil {
		return 0, err
	}
	prev := ""
	for i, r := range rows {
		if r.Seq != uint64(i+1) {
			return i, fmt.Errorf("logvault: manifest row %d: seq %d, want %d", i+1, r.Seq, i+1)
		}
		if r.PrevHash != prev {
			return i, fmt.Errorf("logvault: manifest row %d: prev_hash mismatch", r.Seq)
		}
		if got := manifestChainHash(prev, r); got != r.Hash {
			return i, fmt.Errorf("logvault: manifest row %d: hash mismatch (row tampered or truncated)", r.Seq)
		}
		prev = r.Hash
	}
	return len(rows), nil
}

// fileState is the replayed latest-known capture state of one source file.
type fileState struct {
	SizeAfter int64
	MTimeUnix int64
	SHA256    string
}

// replayStates folds the manifest into the latest capture state per
// (source, rel-path). Skip-error rows do not advance state.
func replayStates(rows []ManifestRow) map[string]fileState {
	states := make(map[string]fileState, len(rows))
	for _, r := range rows {
		if r.SHA256 == "" {
			continue
		}
		states[r.Source+"\x00"+r.RelPath] = fileState{
			SizeAfter: r.SizeAfter,
			MTimeUnix: r.MTimeUnix,
			SHA256:    r.SHA256,
		}
	}
	return states
}
