package computebuild

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

// vulkanReceiptReadLimit is a verifier resource admission policy, not a schema
// maximum. Receipts contain provenance and digests, not shader/module payloads.
// Otherwise-valid receipts with unusually large logs or metadata also refuse
// above this budget; their bytes are never silently truncated and verified.
const vulkanReceiptReadLimit int64 = 16 << 20

func readStableVulkanReceipt(path, label string) ([]byte, error) {
	clean, before, err := strictAbsoluteRegularFile(path, label)
	if err != nil {
		return nil, err
	}
	if before.Size() > vulkanReceiptReadLimit {
		return nil, fmt.Errorf("%s exceeds receipt read limit of %d bytes", label, vulkanReceiptReadLimit)
	}
	f, err := os.Open(clean)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", label, err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return nil, fmt.Errorf("%s changed before read", label)
	}
	if opened.Size() > vulkanReceiptReadLimit {
		return nil, fmt.Errorf("%s exceeds receipt read limit of %d bytes", label, vulkanReceiptReadLimit)
	}
	first, err := readVulkanReceiptTwice(f, label, vulkanReceiptReadLimit)
	if err != nil {
		return nil, err
	}
	afterHandle, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("restat %s: %w", label, err)
	}
	afterPath, err := os.Lstat(clean)
	if err != nil || afterPath.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, afterPath) || !sameFileMetadata(opened, afterHandle) {
		return nil, fmt.Errorf("%s changed during read", label)
	}
	return first, nil
}

// Each pass has its own limit, including when the file grows after stat or
// between reads. The extra byte distinguishes an exact-limit file from excess.
func readVulkanReceiptTwice(f io.ReadSeeker, label string, limit int64) ([]byte, error) {
	read := func(operation string) ([]byte, error) {
		data, err := io.ReadAll(io.LimitReader(f, limit+1))
		if err != nil {
			return nil, fmt.Errorf("%s %s: %w", operation, label, err)
		}
		if int64(len(data)) > limit {
			return nil, fmt.Errorf("%s exceeds receipt read limit of %d bytes", label, limit)
		}
		return data, nil
	}
	first, err := read("read")
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("rewind %s: %w", label, err)
	}
	second, err := read("reread")
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(first, second) {
		return nil, fmt.Errorf("%s changed during read", label)
	}
	return first, nil
}
