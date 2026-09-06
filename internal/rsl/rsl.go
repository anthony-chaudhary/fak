// Package rsl implements an append-only, hash-chained reference state log
// recording observed git ref transitions to detect non-fast-forward rewrites
// and history tampering offline.
package rsl

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Row is one observed ref transition in the reference state log.
// Field order in chainHash is fixed for backwards compatibility.
type Row struct {
	Seq      uint64 `json:"seq"`
	Ref      string `json:"ref"`
	OldSHA   string `json:"old_sha"`
	NewSHA   string `json:"new_sha"`
	Signer   string `json:"signer,omitempty"`
	Sig      string `json:"sig,omitempty"`
	PrevHash string `json:"prev_hash"`
	Hash     string `json:"hash"`
}

// Signer provides optional cryptographic attribution for ref transitions.
type Signer interface {
	Identity() string
	Sign(hash string) (string, error)
}

// chainHash computes the SHA-256 link over the previous hash and row fields.
func chainHash(prev string, r Row) string {
	h := sha256.New()
	io.WriteString(h, prev)
	fmt.Fprintf(h, "\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s", r.Seq, r.Ref, r.OldSHA, r.NewSHA, r.Signer)
	return hex.EncodeToString(h.Sum(nil))
}

// Append records an observed ref transition to the log at path.
func Append(path string, row Row) (Row, error) {
	return appendRow(path, row, nil)
}

// AppendSigned attributes and signs a ref transition before appending.
func AppendSigned(path string, row Row, s Signer) (Row, error) {
	return appendRow(path, row, s)
}

func appendRow(path string, row Row, s Signer) (Row, error) {
	seq, last, err := recoverHead(path)
	if err != nil {
		return Row{}, err
	}
	if s != nil {
		row.Signer = s.Identity()
	}
	row.Seq = seq + 1
	row.PrevHash = last
	row.Hash = chainHash(row.PrevHash, row)
	if s != nil {
		sig, serr := s.Sign(row.Hash)
		if serr != nil {
			return Row{}, fmt.Errorf("rsl: sign row seq %d: %w", row.Seq, serr)
		}
		row.Sig = sig
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return Row{}, fmt.Errorf("rsl: create dir %s: %w", dir, err)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return Row{}, fmt.Errorf("rsl: open %s: %w", path, err)
	}
	defer f.Close()
	b, err := json.Marshal(row)
	if err != nil {
		return Row{}, err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		return Row{}, fmt.Errorf("rsl: append %s: %w", path, err)
	}
	return row, nil
}

// recoverHead reads the last committed sequence number and hash from path.
func recoverHead(path string) (seq uint64, lastHash string, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", nil
		}
		return 0, "", fmt.Errorf("rsl: stat %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	lineNum := 0
	var (
		tornErr  error
		tornLine int
	)
	for sc.Scan() {
		lineNum++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		if tornErr != nil {
			return 0, "", fmt.Errorf("rsl: recover %s: corrupted row at line %d: %w", path, tornLine, tornErr)
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			tornErr = err
			tornLine = lineNum
			continue
		}
		seq = r.Seq
		lastHash = r.Hash
	}
	if err := sc.Err(); err != nil {
		return 0, "", fmt.Errorf("rsl: scan %s: %w", path, err)
	}
	return seq, lastHash, nil
}

// ReadRows reads all committed rows from an RSL file in order.
func ReadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("rsl: read %s: %w", path, err)
	}
	defer f.Close()
	var out []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	lineNum := 0
	var (
		tornErr  error
		tornLine int
	)
	for sc.Scan() {
		lineNum++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		if tornErr != nil {
			return nil, fmt.Errorf("rsl: read %s: corrupted row at line %d: %w", path, tornLine, tornErr)
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			tornErr = err
			tornLine = lineNum
			continue
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("rsl: scan %s: %w", path, err)
	}
	return out, nil
}

// VerifyFile reads an RSL file and verifies its chain and fast-forward invariants.
func VerifyFile(path string) (int, error) {
	rows, err := ReadRows(path)
	if err != nil {
		return 0, err
	}
	return Verify(rows)
}

// Verify checks that rows form a valid hash chain with monotonic sequence
// numbers and that each ref advances in a fast-forward progression.
// It returns the count of verified rows and the first error encountered.
func Verify(rows []Row) (int, error) {
	var (
		prev    string
		wantSeq uint64
	)
	heads := map[string]string{}
	seen := map[string]map[string]bool{}
	for i, row := range rows {
		wantSeq++
		if row.Seq != wantSeq {
			return i, fmt.Errorf("rsl: sequence gap: seq=%d want %d", row.Seq, wantSeq)
		}
		if row.PrevHash != prev {
			return i, fmt.Errorf("rsl: broken chain at seq %d: prev_hash=%q want %q", row.Seq, row.PrevHash, prev)
		}
		if got := chainHash(row.PrevHash, row); got != row.Hash {
			return i, fmt.Errorf("rsl: tampered row at seq %d: hash=%q recomputed %q", row.Seq, row.Hash, got)
		}

		refSeen := seen[row.Ref]
		if refSeen == nil {
			refSeen = map[string]bool{}
			if row.OldSHA != "" {
				refSeen[row.OldSHA] = true
			}
			seen[row.Ref] = refSeen
		} else {
			if head := heads[row.Ref]; row.OldSHA != head {
				return i, fmt.Errorf("rsl: non-fast-forward gap on ref %s at seq %d: old_sha=%q does not continue recorded head %q (trunk rewritten/force-pushed)", row.Ref, row.Seq, row.OldSHA, head)
			}
			if refSeen[row.NewSHA] {
				return i, fmt.Errorf("rsl: non-fast-forward rewind on ref %s at seq %d: new_sha=%q revisits a prior target (trunk rewound/force-pushed)", row.Ref, row.Seq, row.NewSHA)
			}
		}
		refSeen[row.NewSHA] = true
		heads[row.Ref] = row.NewSHA
		prev = row.Hash
	}
	return len(rows), nil
}
