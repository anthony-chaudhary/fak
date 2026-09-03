package studyforge

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func captureSampleCorpus(t *testing.T) Corpus {
	t.Helper()
	fixture := newCheckpointFixture(t)
	var sample Corpus
	_, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(c Corpus) error {
			sample = c
			return errFixtureInterrupted
		},
	})
	if !errors.Is(err, errFixtureInterrupted) {
		t.Fatalf("capture sample error = %v", err)
	}
	return sample
}

func TestStoreGzipRoundTrip(t *testing.T) {
	sample := captureSampleCorpus(t)
	dir := t.TempDir()
	rawPath := filepath.Join(dir, "corpus.json")
	gzPath := filepath.Join(dir, "corpus.json.gz")

	if err := Write(rawPath, sample); err != nil {
		t.Fatalf("write raw: %v", err)
	}
	if err := Write(gzPath, sample); err != nil {
		t.Fatalf("write gz: %v", err)
	}

	fromRaw, err := Read(rawPath)
	if err != nil {
		t.Fatalf("read raw: %v", err)
	}
	fromGz, err := Read(gzPath)
	if err != nil {
		t.Fatalf("read gz: %v", err)
	}

	rawBytes, _ := os.ReadFile(rawPath)
	gzBytes, _ := os.ReadFile(gzPath)
	if len(gzBytes) >= len(rawBytes) {
		t.Errorf("gzip did not compress: raw=%d gz=%d", len(rawBytes), len(gzBytes))
	}

	if fromRaw.Receipt.Repository != fromGz.Receipt.Repository ||
		fromRaw.Receipt.Revision != fromGz.Receipt.Revision ||
		len(fromRaw.Records) != len(fromGz.Records) {
		t.Fatalf("corpus from raw and gz differ: raw=%+v gz=%+v", fromRaw.Receipt, fromGz.Receipt)
	}
}

func TestStoreGzipDeterministic(t *testing.T) {
	sample := captureSampleCorpus(t)
	dir := t.TempDir()
	path1 := filepath.Join(dir, "run1.json.gz")
	path2 := filepath.Join(dir, "run2.json.gz")

	if err := Write(path1, sample); err != nil {
		t.Fatal(err)
	}
	if err := Write(path2, sample); err != nil {
		t.Fatal(err)
	}

	b1, err := os.ReadFile(path1)
	if err != nil {
		t.Fatal(err)
	}
	b2, err := os.ReadFile(path2)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(b1, b2) {
		t.Fatalf("two writes of same corpus to .gz are not byte-identical: %d vs %d bytes", len(b1), len(b2))
	}
}

func TestStoreGzipHeaderMetadataScrubbed(t *testing.T) {
	sample := captureSampleCorpus(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "corpus.json.gz")
	if err := Write(path, sample); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 10 {
		t.Fatal("gzip stream too short")
	}
	// Gzip magic
	if b[0] != 0x1f || b[1] != 0x8b {
		t.Fatalf("invalid gzip magic: %x %x", b[0], b[1])
	}
	// ModTime bytes (bytes 4..7) must be zero
	if b[4] != 0 || b[5] != 0 || b[6] != 0 || b[7] != 0 {
		t.Fatalf("gzip header contains non-zero timestamp: %v", b[4:8])
	}
	// OS byte (byte 9) must be 255 (unknown OS)
	if b[9] != 255 {
		t.Fatalf("gzip header OS byte is %d, want 255", b[9])
	}
}

func TestStoreGzipCorruptAndTruncatedFailClosed(t *testing.T) {
	dir := t.TempDir()

	// 1. Truncated gzip header
	truncPath := filepath.Join(dir, "trunc.json.gz")
	if err := os.WriteFile(truncPath, []byte{0x1f, 0x8b, 0x08}, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(truncPath); err == nil {
		t.Fatal("read truncated gzip succeeded; want error")
	}

	// 2. Corrupt gzip payload
	corruptPath := filepath.Join(dir, "corrupt.json.gz")
	if err := os.WriteFile(corruptPath, []byte{0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x01, 0x02, 0x03}, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(corruptPath); err == nil {
		t.Fatal("read corrupt gzip payload succeeded; want error")
	}

	// 3. Extension mismatch: .gz extension on raw uncompressed text
	mismatchPath := filepath.Join(dir, "plain.json.gz")
	if err := os.WriteFile(mismatchPath, []byte(`{"schema":"fak-study-corpus/1"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(mismatchPath); err == nil {
		t.Fatal("read uncompressed file with .gz extension succeeded; want error")
	}
}

func TestStoreReadResumeGzip(t *testing.T) {
	sample := captureSampleCorpus(t)
	dir := t.TempDir()
	gzPath := filepath.Join(dir, "checkpoint.json.gz")
	if err := Write(gzPath, sample); err != nil {
		t.Fatal(err)
	}
	resumed, err := ReadResume(gzPath)
	if err != nil {
		t.Fatalf("ReadResume failed on gzip checkpoint: %v", err)
	}
	if resumed.Receipt.Repository != sample.Receipt.Repository {
		t.Fatalf("resumed repo=%q, want %q", resumed.Receipt.Repository, sample.Receipt.Repository)
	}
}
