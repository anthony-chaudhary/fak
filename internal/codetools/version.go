package codetools

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
)

const fileVersionDomain = "fak-fs-version/1"

type fileObservation struct {
	Version   string
	Content   []byte
	Info      os.FileInfo
	Bytes     int64
	Truncated bool
}

// observeFile binds bytes to the identity of the opened file handle. captureLimit bounds
// retained memory; the digest still covers the complete file so a change beyond a Read
// window invalidates a later mutation.
func observeFile(ctx context.Context, path string, captureLimit int64) (fileObservation, *Refusal) {
	RecordSubprocessAvoided()
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fileObservation{}, refuse(CodeNotFound, "file no longer exists")
		}
		return fileObservation{}, refuse(CodeIO, err.Error())
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return fileObservation{}, refuse(CodeIO, err.Error())
	}
	if info.IsDir() {
		return fileObservation{}, refuse(CodeIsDir, "target is a directory")
	}
	if !info.Mode().IsRegular() {
		return fileObservation{}, refuse(CodeIO, "target is not a regular file")
	}
	identity, err := fileIdentity(f, info)
	if err != nil {
		return fileObservation{}, refuse(CodeIO, err.Error())
	}

	h := sha256.New()
	_, _ = io.WriteString(h, fileVersionDomain)
	_, _ = h.Write([]byte{0})
	_, _ = io.WriteString(h, identity)
	_, _ = h.Write([]byte{0})
	var content []byte
	if captureLimit > 0 && info.Size() > 0 {
		capacity := info.Size()
		if capacity > captureLimit {
			capacity = captureLimit
		}
		if maxInt := int64(^uint(0) >> 1); capacity > maxInt {
			capacity = maxInt
		}
		content = make([]byte, 0, int(capacity))
	}
	buf := AcquireBuffer(ArenaClass64K)
	defer ReleaseBuffer(buf)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return fileObservation{}, refuse(CodeCanceled, err.Error())
		}
		n, readErr := f.Read(buf)
		if n > 0 {
			chunk := buf[:n]
			_, _ = h.Write(chunk)
			if int64(len(content)) < captureLimit {
				keep := int64(n)
				if remaining := captureLimit - int64(len(content)); keep > remaining {
					keep = remaining
				}
				content = append(content, chunk[:int(keep)]...)
			}
			total += int64(n)
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fileObservation{}, refuse(CodeIO, readErr.Error())
		}
	}
	return fileObservation{
		Version: "fv1:" + hex.EncodeToString(h.Sum(nil)), Content: content, Info: info,
		Bytes: total, Truncated: total > captureLimit,
	}, nil
}
