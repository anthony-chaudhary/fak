package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/qwen4exp"
)

const modelTraceStubUsage = "usage: fak model trace-stub --gguf <path> --token-id <id> [--token-id <id> ...] [--out <new-file>]\nDiagnostic only: 48 identity layers, sampled raw output projection, no generated tokens or parity claim.\n"

type traceStubTokenIDs []int

func (v *traceStubTokenIDs) String() string { return fmt.Sprint([]int(*v)) }
func (v *traceStubTokenIDs) Set(raw string) error {
	id, err := strconv.Atoi(raw)
	if err != nil || id < 0 || id >= 248320 || len(*v) >= 4 {
		return fmt.Errorf("require one to four token ids in [0,248320)")
	}
	*v = append(*v, id)
	return nil
}

type traceStubSource interface {
	Config() (model.Config, error)
	Tensor(string) (ggufload.TensorInfo, bool)
	TensorRowF32(string, int) ([]float32, ggufload.TensorInfo, error)
	Close() error
}

type traceStubGGUF struct{ *ggufload.WeightSource }

func (s traceStubGGUF) Config() (model.Config, error) { return s.File.Config() }

func runModelTraceStub(stdout, stderr io.Writer, args []string) int {
	return runModelTraceStubWithSource(stdout, stderr, args, func(path string) (traceStubSource, error) {
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			return nil, err
		}
		return traceStubGGUF{ws}, nil
	})
}

func runModelTraceStubWithSource(stdout, stderr io.Writer, args []string, open func(string) (traceStubSource, error)) int {
	fs := flag.NewFlagSet("model trace-stub", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, modelTraceStubUsage) }
	gguf := fs.String("gguf", "", "GGUF shard path; reads only selected embedding/output rows")
	out := fs.String("out", "", "new receipt file; default stdout")
	var ids traceStubTokenIDs
	fs.Var(&ids, "token-id", "already-tokenized input id; repeat up to four times")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *gguf == "" || len(ids) == 0 || fs.NArg() != 0 {
		fs.Usage()
		return 2
	}
	source, err := open(pathutil.ExpandTilde(*gguf))
	if err != nil {
		fmt.Fprintf(stderr, "fak model trace-stub: %v\n", err)
		return 1
	}
	closed := false
	defer func() {
		if !closed {
			_ = source.Close()
		}
	}()
	cfg, err := source.Config()
	if err != nil {
		fmt.Fprintf(stderr, "fak model trace-stub: %v\n", err)
		return 1
	}
	for _, name := range []string{"token_embd.weight", "output.weight"} {
		info, ok := source.Tensor(name)
		if !ok || len(info.Dims) != 2 || info.Dims[0] != 2560 || info.Dims[1] != 248320 {
			fmt.Fprintf(stderr, "fak model trace-stub: %s requires [2560,248320] geometry\n", name)
			return 1
		}
	}
	input, err := json.Marshal(struct {
		Config model.Config `json:"config"`
		Tokens []int        `json:"tokens"`
	}{cfg, ids})
	if err != nil {
		fmt.Fprintf(stderr, "fak model trace-stub: input identity: %v\n", err)
		return 1
	}
	digest := sha256.Sum256(input)
	seams := qwen4exp.StubSeams{
		InputDigest: hex.EncodeToString(digest[:]),
		Embed: func(id int) ([]float32, string, error) {
			row, _, err := source.TensorRowF32("token_embd.weight", id)
			return row, traceStubRowDigest(row), err
		},
		ProjectSamples: func(hidden []float32, indices []int) ([]float32, string, error) {
			values := make([]float32, len(indices))
			hash := sha256.New()
			for i, index := range indices {
				row, _, err := source.TensorRowF32("output.weight", index)
				if err != nil {
					return nil, "", err
				}
				if len(row) != len(hidden) {
					return nil, "", fmt.Errorf("output row width %d differs from hidden width %d", len(row), len(hidden))
				}
				_, _ = io.WriteString(hash, strconv.Itoa(index)+":"+traceStubRowDigest(row)+"\n")
				for j, weight := range row {
					values[i] += hidden[j] * weight
				}
			}
			return values, hex.EncodeToString(hash.Sum(nil)), nil
		},
	}
	receipt, err := model.RunQwen4ExpDiagnosticStub(cfg, ids, seams)
	if err != nil {
		fmt.Fprintf(stderr, "fak model trace-stub: %v\n", err)
		return 1
	}
	if err := source.Close(); err != nil {
		closed = true
		fmt.Fprintf(stderr, "fak model trace-stub: close checkpoint: %v\n", err)
		return 1
	}
	closed = true
	data, err := json.Marshal(receipt)
	if err != nil {
		fmt.Fprintf(stderr, "fak model trace-stub: encode receipt: %v\n", err)
		return 1
	}
	data = append(data, '\n')
	if *out != "" {
		file, err := os.OpenFile(pathutil.ExpandTilde(*out), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			fmt.Fprintf(stderr, "fak model trace-stub: %v\n", err)
			return 1
		}
		created, statErr := file.Stat()
		writeErr := writeTraceStubBytes(file, data)
		closeErr := file.Close()
		if statErr != nil || writeErr != nil || closeErr != nil {
			if current, err := os.Lstat(file.Name()); err == nil && created != nil && os.SameFile(created, current) {
				_ = os.Remove(file.Name())
			}
			fmt.Fprintf(stderr, "fak model trace-stub: write receipt: stat=%v write=%v close=%v\n", statErr, writeErr, closeErr)
			return 1
		}
		return 0
	}
	if err := writeTraceStubBytes(stdout, data); err != nil {
		fmt.Fprintf(stderr, "fak model trace-stub: write receipt: %v\n", err)
		return 1
	}
	return 0
}

func writeTraceStubBytes(w io.Writer, data []byte) error {
	n, err := w.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	return err
}

func traceStubRowDigest(row []float32) string {
	hash := sha256.New()
	var word [4]byte
	for _, value := range row {
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
		_, _ = hash.Write(word[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}
