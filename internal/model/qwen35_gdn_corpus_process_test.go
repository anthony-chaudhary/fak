package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// TestQwen35GDNCorpusProcess is the external CPU entry point. Run it once with
// FAK_QWEN35_GDN_CORPUS_MODE=produce and again, as a separate process, with
// FAK_QWEN35_GDN_CORPUS_MODE=verify. The verifier never invokes the producer.
func TestQwen35GDNCorpusProcess(t *testing.T) {
	mode := os.Getenv(qwen35GDNCorpusModeEnv)
	if mode == "" {
		t.Skipf("external entry point; set %s=produce|verify and %s=<corpus-dir>", qwen35GDNCorpusModeEnv, qwen35GDNCorpusPathEnv)
	}
	dir := os.Getenv(qwen35GDNCorpusPathEnv)
	switch mode {
	case "produce":
		digest, err := writeQwen35GDNCorpus(dir)
		if err != nil {
			t.Fatalf("produce %s: %v", qwen35GDNCorpusFormat, err)
		}
		t.Logf("mode=produce format=%s manifest_sha256=%s path=%s", qwen35GDNCorpusFormat, digest, dir)
	case "verify":
		corpus, err := loadQwen35GDNCorpus(dir)
		if err != nil {
			t.Fatalf("verify %s: %v", qwen35GDNCorpusFormat, err)
		}
		t.Logf("mode=verify format=%s manifest_sha256=%s tensors=%d steps=%d path=%s",
			corpus.Metadata.Format, corpus.ManifestSHA256, len(corpus.Tensors), len(corpus.Metadata.Steps), dir)
	default:
		t.Fatalf("%s=%q, want produce or verify", qwen35GDNCorpusModeEnv, mode)
	}
}

func TestQwen35GDNCorpusDeterministicRoundTrip(t *testing.T) {
	dirA := filepath.Join(t.TempDir(), "a")
	dirB := filepath.Join(t.TempDir(), "b")
	digestA, err := writeQwen35GDNCorpus(dirA)
	if err != nil {
		t.Fatal(err)
	}
	digestB, err := writeQwen35GDNCorpus(dirB)
	if err != nil {
		t.Fatal(err)
	}
	if digestA != digestB {
		t.Fatalf("manifest digest is nondeterministic: %s != %s", digestA, digestB)
	}
	snapshotA := snapshotQwen35GDNCorpus(t, dirA)
	snapshotB := snapshotQwen35GDNCorpus(t, dirB)
	if len(snapshotA) != len(snapshotB) {
		t.Fatalf("file count is nondeterministic: %d != %d", len(snapshotA), len(snapshotB))
	}
	for name, a := range snapshotA {
		b, ok := snapshotB[name]
		if !ok || !bytes.Equal(a, b) {
			t.Fatalf("corpus file %q is not byte-for-byte deterministic", name)
		}
	}
	corpus, err := loadQwen35GDNCorpus(dirA)
	if err != nil {
		t.Fatal(err)
	}
	if corpus.Metadata.Format != qwen35GDNCorpusFormat || corpus.ManifestSHA256 != digestA {
		t.Fatalf("round trip identity: format=%q digest=%q", corpus.Metadata.Format, corpus.ManifestSHA256)
	}
	if len(corpus.Metadata.Steps) != qwen35GDNCorpusStepCount {
		t.Fatalf("round trip steps=%d, want %d", len(corpus.Metadata.Steps), qwen35GDNCorpusStepCount)
	}
}

func TestQwen35GDNCorpusSeparateProducerVerifierProcesses(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "external-corpus")
	producerLog := runQwen35GDNCorpusProcess(t, "produce", dir)
	verifierLog := runQwen35GDNCorpusProcess(t, "verify", dir)
	for label, log := range map[string]string{"producer": producerLog, "verifier": verifierLog} {
		if !strings.Contains(log, "manifest_sha256=") || !strings.Contains(log, qwen35GDNCorpusFormat) {
			t.Fatalf("%s process did not emit corpus identity:\n%s", label, log)
		}
	}
}

func TestQwen35GDNCorpusRequiredSelectionFailsClosed(t *testing.T) {
	if _, skip, err := selectQwen35GDNCorpus(true, ""); err == nil || skip || !strings.Contains(err.Error(), qwen35GDNCorpusPathEnv) {
		t.Fatalf("required missing corpus: skip=%v err=%v", skip, err)
	}
	if path, skip, err := selectQwen35GDNCorpus(false, ""); err != nil || !skip || path != "" {
		t.Fatalf("optional missing corpus: path=%q skip=%v err=%v", path, skip, err)
	}
	if path, skip, err := selectQwen35GDNCorpus(true, " /tmp/oracle "); err != nil || skip || path != "/tmp/oracle" {
		t.Fatalf("explicit corpus: path=%q skip=%v err=%v", path, skip, err)
	}
}

func TestQwen35GDNCorpusRejectsCorruption(t *testing.T) {
	if _, err := loadQwen35GDNCorpus(filepath.Join(t.TempDir(), "absent")); err == nil || !strings.Contains(err.Error(), "read corpus directory") {
		t.Fatalf("absent corpus did not fail closed: %v", err)
	}
	inputFile := qwen35GDNTensorFile(qwen35GDNStepTensor(0, "input"))
	tests := []struct {
		name   string
		want   string
		mutate func(*testing.T, string)
	}{
		{
			name: "malformed_metadata", want: "parse corpus metadata",
			mutate: func(t *testing.T, dir string) {
				writeQwen35GDNTestFile(t, filepath.Join(dir, qwen35GDNCorpusMetadataFile), []byte("{broken\n"))
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "hash_mismatch", want: "hash mismatch",
			mutate: func(t *testing.T, dir string) {
				path := filepath.Join(dir, inputFile)
				b := readQwen35GDNTestFile(t, path)
				b[0] ^= 0xff
				writeQwen35GDNTestFile(t, path, b)
			},
		},
		{
			name: "schema_mismatch", want: "corpus schema",
			mutate: func(t *testing.T, dir string) {
				rewriteQwen35GDNTestMetadata(t, dir, func(meta *qwen35GDNCorpusMetadata) { meta.Format = "fak.qwen35-gdn-corpus.v0" })
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "producer_identity_mismatch", want: "producer identity",
			mutate: func(t *testing.T, dir string) {
				rewriteQwen35GDNTestMetadata(t, dir, func(meta *qwen35GDNCorpusMetadata) { meta.Producer.Module = "untrusted/module" })
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "malformed_manifest", want: "manifest line",
			mutate: func(t *testing.T, dir string) {
				writeQwen35GDNTestFile(t, filepath.Join(dir, qwen35GDNCorpusManifestFile), []byte("not-a-digest  corpus.json\n"))
			},
		},
		{
			name: "unsorted_manifest", want: "not strictly sorted",
			mutate: func(t *testing.T, dir string) {
				path := filepath.Join(dir, qwen35GDNCorpusManifestFile)
				lines := strings.Split(strings.TrimSuffix(string(readQwen35GDNTestFile(t, path)), "\n"), "\n")
				for left, right := 0, len(lines)-1; left < right; left, right = left+1, right-1 {
					lines[left], lines[right] = lines[right], lines[left]
				}
				writeQwen35GDNTestFile(t, path, []byte(strings.Join(lines, "\n")+"\n"))
			},
		},
		{
			name: "missing_entry", want: "directory entries mismatch",
			mutate: func(t *testing.T, dir string) { removeQwen35GDNTestFile(t, filepath.Join(dir, inputFile)) },
		},
		{
			name: "extra_entry", want: "directory entries mismatch",
			mutate: func(t *testing.T, dir string) {
				writeQwen35GDNTestFile(t, filepath.Join(dir, "unexpected.bin"), []byte("extra"))
			},
		},
		{
			name: "extra_hashed_entry", want: "manifest entries do not match schema",
			mutate: func(t *testing.T, dir string) {
				writeQwen35GDNTestFile(t, filepath.Join(dir, "unexpected.bin"), []byte("extra"))
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "malformed_tensor_length", want: "byte length",
			mutate: func(t *testing.T, dir string) {
				path := filepath.Join(dir, inputFile)
				b := readQwen35GDNTestFile(t, path)
				writeQwen35GDNTestFile(t, path, b[:len(b)-1])
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "nonfinite_tensor", want: "non-finite",
			mutate: func(t *testing.T, dir string) {
				path := filepath.Join(dir, inputFile)
				b := readQwen35GDNTestFile(t, path)
				binary.LittleEndian.PutUint32(b, math.Float32bits(float32(math.NaN())))
				writeQwen35GDNTestFile(t, path, b)
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "nonzero_requirement", want: "violates nonzero requirement",
			mutate: func(t *testing.T, dir string) {
				path := filepath.Join(dir, inputFile)
				b := readQwen35GDNTestFile(t, path)
				clear(b)
				writeQwen35GDNTestFile(t, path, b)
				rewriteQwen35GDNTestMetadata(t, dir, func(meta *qwen35GDNCorpusMetadata) {
					for i := range meta.Tensors {
						if meta.Tensors[i].Name == qwen35GDNStepTensor(0, "input") {
							meta.Tensors[i].Norm = 0
						}
					}
				})
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "shape_mismatch", want: "contract mismatch",
			mutate: func(t *testing.T, dir string) {
				rewriteQwen35GDNTestMetadata(t, dir, func(meta *qwen35GDNCorpusMetadata) {
					for i := range meta.Tensors {
						if meta.Tensors[i].Name == qwen35GDNStepTensor(0, "input") {
							meta.Tensors[i].Shape = []int{meta.Geometry.HiddenSize / 2, 2}
						}
					}
				})
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "norm_mismatch", want: "metadata=",
			mutate: func(t *testing.T, dir string) {
				rewriteQwen35GDNTestMetadata(t, dir, func(meta *qwen35GDNCorpusMetadata) {
					for i := range meta.Tensors {
						if meta.Tensors[i].Name == qwen35GDNStepTensor(0, "input") {
							meta.Tensors[i].Norm++
						}
					}
				})
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
		{
			name: "max_error_mismatch", want: "max_abs_error",
			mutate: func(t *testing.T, dir string) {
				rewriteQwen35GDNTestMetadata(t, dir, func(meta *qwen35GDNCorpusMetadata) { meta.Steps[0].Output.MaxAbsError = 1 })
				rehashQwen35GDNTestCorpus(t, dir)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "corpus")
			if _, err := writeQwen35GDNCorpus(dir); err != nil {
				t.Fatal(err)
			}
			test.mutate(t, dir)
			if _, err := loadQwen35GDNCorpus(dir); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("corruption did not fail closed with %q: %v", test.want, err)
			}
		})
	}
}

func runQwen35GDNCorpusProcess(t *testing.T, mode, dir string) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=^TestQwen35GDNCorpusProcess$", "-test.v", "-test.count=1", "-test.parallel=1")
	cmd.Env = qwen35GDNProcessEnv(mode, dir)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Run(); err != nil {
		t.Fatalf("%s process: %v\n%s", mode, err, output.String())
	}
	return output.String()
}

func qwen35GDNProcessEnv(mode, dir string) []string {
	blocked := map[string]bool{
		qwen35GDNCorpusModeEnv: true, qwen35GDNCorpusPathEnv: true,
		"GOMEMLIMIT": true, "GOMAXPROCS": true,
	}
	env := make([]string, 0, len(os.Environ())+4)
	for _, item := range os.Environ() {
		key, _, _ := strings.Cut(item, "=")
		if !blocked[key] {
			env = append(env, item)
		}
	}
	return append(env,
		qwen35GDNCorpusModeEnv+"="+mode,
		qwen35GDNCorpusPathEnv+"="+dir,
		"GOMEMLIMIT=512MiB",
		"GOMAXPROCS=1",
	)
}

func snapshotQwen35GDNCorpus(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	snapshot := make(map[string][]byte, len(entries))
	for _, entry := range entries {
		snapshot[entry.Name()] = readQwen35GDNTestFile(t, filepath.Join(dir, entry.Name()))
	}
	return snapshot
}

func rewriteQwen35GDNTestMetadata(t *testing.T, dir string, mutate func(*qwen35GDNCorpusMetadata)) {
	t.Helper()
	path := filepath.Join(dir, qwen35GDNCorpusMetadataFile)
	var metadata qwen35GDNCorpusMetadata
	if err := json.Unmarshal(readQwen35GDNTestFile(t, path), &metadata); err != nil {
		t.Fatal(err)
	}
	mutate(&metadata)
	b, err := marshalQwen35GDNMetadata(metadata)
	if err != nil {
		t.Fatal(err)
	}
	writeQwen35GDNTestFile(t, path, b)
}

func rehashQwen35GDNTestCorpus(t *testing.T, dir string) {
	t.Helper()
	if _, err := writeQwen35GDNManifest(dir); err != nil {
		t.Fatal(err)
	}
}

func readQwen35GDNTestFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func writeQwen35GDNTestFile(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func removeQwen35GDNTestFile(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}
