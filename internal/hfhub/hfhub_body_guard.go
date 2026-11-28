package hfhub

import (
	"errors"
	"io"
	"os"
)

// ErrCredentialBody reports that a Hub response body or retained partial file
// contains the client's credential (raw or percent-encoded).
var ErrCredentialBody = errors.New("hfhub: response or partial body contains credential")

// bodyGuard buffers ambiguous trailing bytes across streaming chunks so a
// split raw or percent-encoded credential cannot be written to disk before
// later bytes decide. Porting TensorFold pull_auth.zig Auth.Body (Apache-2.0).
type bodyGuard struct {
	secret  string
	pending []byte
}

func newBodyGuard(token string) *bodyGuard {
	return &bodyGuard{secret: extractSecret(token)}
}

func (b *bodyGuard) buffered() int {
	return len(b.pending)
}

// feed appends chunk and returns the prefix proven safe to write to disk.
// The caller must call commit(len(safe)) after persisting safe.
func (b *bodyGuard) feed(chunk []byte) ([]byte, error) {
	b.pending = append(b.pending, chunk...)
	if b.secret == "" {
		return b.pending, nil
	}
	m := newCredentialMatcher(b.pending)
	for i := 0; i < len(b.pending); i++ {
		if b.pending[i] != '%' && b.pending[i] != b.secret[0] {
			continue
		}
		found := m.matchSecret(i, b.secret)
		if found.consumed > 0 {
			return nil, ErrCredentialBody
		}
		if found.hold && found.prefixLen > 0 {
			return b.pending[:i], nil
		}
	}
	return b.pending, nil
}

func (b *bodyGuard) commit(sent int) {
	if sent <= 0 {
		return
	}
	if sent >= len(b.pending) {
		b.pending = b.pending[:0]
		return
	}
	copy(b.pending, b.pending[sent:])
	b.pending = b.pending[:len(b.pending)-sent]
}

// finish checks any remaining buffered tail at EOF: an incomplete encoding is
// data, while a complete credential returns ErrCredentialBody.
func (b *bodyGuard) finish() ([]byte, error) {
	if b.secret == "" || len(b.pending) == 0 {
		return b.pending, nil
	}
	m := newCredentialMatcher(b.pending)
	for i := 0; i < len(b.pending); i++ {
		if b.pending[i] != '%' && b.pending[i] != b.secret[0] {
			continue
		}
		if m.matchSecret(i, b.secret).consumed > 0 {
			return nil, ErrCredentialBody
		}
	}
	return b.pending, nil
}

// scanFileForCredential streams path in 64 KiB chunks through bodyGuard,
// returning true if the file contains a complete raw or percent-encoded secret.
func scanFileForCredential(token, path string) (bool, error) {
	secret := extractSecret(token)
	if secret == "" {
		return false, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()

	bg := newBodyGuard(secret)
	buf := make([]byte, 64<<10)
	for {
		n, readErr := f.Read(buf)
		if n > 0 {
			safe, err := bg.feed(buf[:n])
			if err != nil {
				if errors.Is(err, ErrCredentialBody) {
					return true, nil
				}
				return false, err
			}
			bg.commit(len(safe))
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return false, readErr
		}
	}
	if _, err := bg.finish(); err != nil {
		if errors.Is(err, ErrCredentialBody) {
			return true, nil
		}
		return false, err
	}
	return false, nil
}

// scanResumePartials checks partPath (and any sibling .incomplete file) for
// leaked credentials before resuming. If found, it deletes the offending file
// and returns ErrCredentialBody.
func scanResumePartials(token, partPath string) error {
	if extractSecret(token) == "" {
		return nil
	}
	candidates := []string{partPath}
	if stringsBase, ok := cutPartSuffix(partPath); ok {
		candidates = append(candidates, stringsBase+".incomplete")
	}
	for _, p := range candidates {
		fi, err := os.Stat(p)
		if err != nil || fi.Size() == 0 {
			continue
		}
		found, err := scanFileForCredential(token, p)
		if err != nil {
			_ = os.Remove(p)
			return err
		}
		if found {
			_ = os.Remove(p)
			return ErrCredentialBody
		}
	}
	return nil
}

func cutPartSuffix(p string) (string, bool) {
	if len(p) > len(partSuffix) && p[len(p)-len(partSuffix):] == partSuffix {
		return p[:len(p)-len(partSuffix)], true
	}
	return "", false
}

// streamGuardedBody streams body into f while feeding hasher and screening
// every byte (including any resumed prefix) through bodyGuard before writing.
func streamGuardedBody(f *os.File, hasher io.Writer, body io.Reader, token string, offset int64, resumed bool) (int64, error) {
	bg := newBodyGuard(token)
	if resumed && offset > 0 {
		prefixBuf := make([]byte, 64<<10)
		var at int64
		for at < offset {
			want := len(prefixBuf)
			if rem := offset - at; rem < int64(want) {
				want = int(rem)
			}
			nr, err := f.ReadAt(prefixBuf[:want], at)
			if nr > 0 {
				if _, herr := hasher.Write(prefixBuf[:nr]); herr != nil {
					return 0, herr
				}
				safe, berr := bg.feed(prefixBuf[:nr])
				if berr != nil {
					return 0, berr
				}
				bg.commit(len(safe))
				at += int64(nr)
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				return 0, err
			}
		}
	}

	writeOffset := offset - int64(bg.buffered())
	var totalRead int64
	chunk := make([]byte, 32<<10)
	for {
		nr, rerr := body.Read(chunk)
		if nr > 0 {
			totalRead += int64(nr)
			if _, herr := hasher.Write(chunk[:nr]); herr != nil {
				return totalRead, herr
			}
			safe, berr := bg.feed(chunk[:nr])
			if berr != nil {
				return totalRead, berr
			}
			if len(safe) > 0 {
				if _, werr := f.WriteAt(safe, writeOffset); werr != nil {
					return totalRead, werr
				}
				writeOffset += int64(len(safe))
				bg.commit(len(safe))
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return totalRead, rerr
		}
	}

	tail, err := bg.finish()
	if err != nil {
		return totalRead, err
	}
	if len(tail) > 0 {
		if _, werr := f.WriteAt(tail, writeOffset); werr != nil {
			return totalRead, werr
		}
		writeOffset += int64(len(tail))
		bg.commit(len(tail))
	}
	if err := f.Truncate(offset + totalRead); err != nil {
		return totalRead, err
	}
	return totalRead, nil
}
