package computetrace

// llamacpp_dump.go -- convert a llama.cpp tensor-dump directory into a bounded
// activation-sample reference artifact.
//
// The reference runtime for DeepSeek V4.1 Flash bring-up is a llama.cpp fork
// that writes named intermediate tensors to disk. ConvertLlamaCppDump reads that
// dump and samples it with the exact rule fak's own V4.1 producer uses
// (ActivationSampleIndices, 8 points per token row), so the result is directly
// comparable with ActivationErrorProfile.
//
// Dump layout (<dir>):
//
//   - index.jsonl  one JSON object per line:
//     {"seq","name","op","type","ne":[4],"dumped","file","sum","sumsq","maxabs","nonfinite"}
//   - <file>       raw little-endian f32 tensor data, relative to <dir>
//   - tokens.json  {"prompt","tokens":[...]}
//
// Names are "<base>-<layer>" for per-layer tensors and a bare "<name>" for
// tensors outside the layer stack. Which names map to which fak stage is a
// per-GGUF-arch LlamaCppStageMap (see llamacpp_stagemap.go); unmapped names are
// ignored. ConvertLlamaCppDump uses the deepseek41 map.
//
// ggml stores ne0 fastest, so a [ne0=H, ne1=hc, ne2=tokens] tensor is [t][hc][i]
// in memory -- the same order as fak's streams[t][s][i] flattened. Every
// remaining dimension counts tokens. A name that repeats (a later ubatch)
// continues its token numbering where the previous occurrence ended.
//
// The converted events carry a placeholder input digest derived from the dump's
// tokens and NO weight digest, so the strict comparators refuse them as-is. Run
// AdoptActivationDigests against the fak trace (which proves the token ids
// match) before profiling.

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// LlamaCppActivationSamples is the per-token sample count the converter takes,
// matching fak's V4.1 activation producer.
const LlamaCppActivationSamples = 8

// ErrLlamaCppDump wraps every malformed-dump refusal.
var ErrLlamaCppDump = errors.New("computetrace: malformed llama.cpp dump")

type llamaCppRow struct {
	stage string
	layer int
}

type llamaCppIndexEntry struct {
	Seq    int     `json:"seq"`
	Name   string  `json:"name"`
	Op     string  `json:"op"`
	Type   string  `json:"type"`
	Ne     []int64 `json:"ne"`
	Dumped bool    `json:"dumped"`
	File   string  `json:"file"`
}

type llamaCppTokens struct {
	Prompt string `json:"prompt"`
	Tokens []int  `json:"tokens"`
}

func dumpErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrLlamaCppDump}, args...)...)
}

// ConvertLlamaCppDump reads a llama.cpp dump directory with the deepseek41
// stage map and returns a sampled reference Artifact whose InputTokens are the
// dump's tokens.
func ConvertLlamaCppDump(dir string) (Artifact, error) {
	return ConvertLlamaCppDumpArch(dir, LlamaCppArchDeepSeek41)
}

// ConvertLlamaCppDumpArch is ConvertLlamaCppDump with the stage map registered
// for arch. An unregistered arch fails with ErrUnknownStageMap.
func ConvertLlamaCppDumpArch(dir, arch string) (Artifact, error) {
	stageMap, ok := LlamaCppStageMapFor(arch)
	if !ok {
		return Artifact{}, fmt.Errorf("%w: %q (known: %s)", ErrUnknownStageMap, arch, strings.Join(LlamaCppStageMapArches(), ", "))
	}
	var toks llamaCppTokens
	raw, err := os.ReadFile(filepath.Join(dir, "tokens.json"))
	if err != nil {
		return Artifact{}, err
	}
	if err := json.Unmarshal(raw, &toks); err != nil {
		return Artifact{}, dumpErr("tokens.json: %v", err)
	}
	if len(toks.Tokens) == 0 {
		return Artifact{}, dumpErr("tokens.json lists no tokens")
	}
	entries, err := readLlamaCppIndex(filepath.Join(dir, "index.jsonl"))
	if err != nil {
		return Artifact{}, err
	}

	var tokenText strings.Builder
	for i, id := range toks.Tokens {
		if i != 0 {
			tokenText.WriteByte(',')
		}
		tokenText.WriteString(strconv.Itoa(id))
	}
	inputDigest := Digest("llamacpp-tokens", tokenText.String())

	dumped := map[string]bool{}
	for _, e := range entries {
		if e.Dumped {
			dumped[e.Name] = true
		}
	}
	nextToken := map[llamaCppRow]int{}
	rowSource := map[llamaCppRow]string{}
	art := Artifact{Schema: Schema, InputTokens: append([]int(nil), toks.Tokens...)}
	maxLayer := -1
	for _, e := range entries {
		spec, layer, ok := stageMap.lookup(e.Name)
		if !ok || !e.Dumped {
			continue
		}
		if spec.SupersededBy != "" && dumped[spec.SupersededBy+"-"+strconv.Itoa(layer)] {
			continue
		}
		base := e.Name
		if b, _, split := splitLlamaCppName(e.Name); split && layer != -1 && layer != llamaCppPostLayer {
			base = b
		}
		key := llamaCppRow{stage: spec.Stage, layer: layer}
		if prev, seen := rowSource[key]; seen && prev != base {
			return Artifact{}, dumpErr("%s: stage %s layer %d already fed by %s", e.Name, spec.Stage, layer, prev)
		}
		rowSource[key] = base
		if layer != llamaCppPostLayer {
			maxLayer = max(maxLayer, layer)
		}
		if e.Type != "f32" {
			return Artifact{}, dumpErr("%s: tensor type %q, want f32", e.Name, e.Type)
		}
		if len(e.Ne) != 4 {
			return Artifact{}, dumpErr("%s: ne has %d dims, want 4", e.Name, len(e.Ne))
		}
		width, tokens, byteSize := int64(1), int64(1), int64(4)
		maxBytes := int64(^uint(0) >> 1)
		emptyTail := false
		for d, n := range e.Ne {
			if spec.RowRank > 0 && d >= spec.RowRank && n != 1 {
				return Artifact{}, dumpErr("%s: ne[%d]=%d, want singleton after row rank %d", e.Name, d, n, spec.RowRank)
			}
			if n == 0 && d >= spec.WidthDims && spec.TailAligned {
				emptyTail = true
				continue
			}
			if n <= 0 {
				return Artifact{}, dumpErr("%s: ne[%d]=%d", e.Name, d, n)
			}
			// A representable byte size also bounds width, tokens and sample offsets.
			if n > maxBytes/byteSize {
				return Artifact{}, dumpErr("%s: f32 byte size exceeds int for ne=%v", e.Name, e.Ne)
			}
			byteSize *= n
			if d < spec.WidthDims {
				width *= n
			} else {
				tokens *= n
			}
		}
		if emptyTail {
			continue
		}
		if !filepath.IsLocal(e.File) {
			return Artifact{}, dumpErr("%s: file %q is not local to the dump", e.Name, e.File)
		}
		data, err := os.Open(filepath.Join(dir, e.File))
		if err != nil {
			return Artifact{}, err
		}
		info, err := data.Stat()
		if err != nil {
			data.Close()
			return Artifact{}, err
		}
		if info.Size() != byteSize {
			data.Close()
			return Artifact{}, dumpErr("%s: %d bytes, want %d for ne=%v", e.Name, info.Size(), byteSize, e.Ne)
		}
		w := int(width)
		idx := ActivationSampleIndices(w, LlamaCppActivationSamples)
		var excluded []int
		if spec.RoPETail > 0 {
			for _, j := range idx {
				if j >= w-spec.RoPETail {
					excluded = append(excluded, j)
				}
			}
		}
		for t := 0; t < int(tokens); t++ {
			vals := make([]float32, len(idx))
			var raw [4]byte
			for i, j := range idx {
				off := int64(4) * (int64(t)*width + int64(j))
				if _, err := data.ReadAt(raw[:], off); err != nil {
					data.Close()
					return Artifact{}, dumpErr("%s: sample offset %d: %v", e.Name, off, err)
				}
				vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[:]))
			}
			art.Events = append(art.Events, Event{
				Sequence:              int64(len(art.Events) + 1),
				Operation:             "llamacpp.dump",
				Phase:                 spec.Stage,
				Backend:               "llamacpp",
				Device:                "reference",
				Layer:                 layer,
				Token:                 nextToken[key] + t,
				InputDigest:           inputDigest,
				Shapes:                [][]int{{w}},
				SampleIndices:         append([]int(nil), idx...),
				SampleValues:          vals,
				ExcludedSampleIndices: append([]int(nil), excluded...),
				ProvenanceDigest:      Digest("llamacpp-dump", e.Name, strconv.Itoa(e.Seq)),
				Status:                "ok",
			})
			art.RetainedSampleValues += uint64(len(vals))
		}
		if err := data.Close(); err != nil {
			return Artifact{}, err
		}
		nextToken[key] += int(tokens)
	}
	stageMap.finish(&art, maxLayer, nextToken)
	return art, nil
}

func readLlamaCppIndex(path string) ([]llamaCppIndexEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []llamaCppIndexEntry
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var e llamaCppIndexEntry
		if err := json.Unmarshal([]byte(text), &e); err != nil {
			return nil, dumpErr("index.jsonl line %d: %v", line, err)
		}
		out = append(out, e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out, nil
}

// splitLlamaCppName splits "<base>-<layer>" at the last '-'.
func splitLlamaCppName(name string) (string, int, bool) {
	cut := strings.LastIndexByte(name, '-')
	if cut <= 0 {
		return "", 0, false
	}
	layer, err := strconv.Atoi(name[cut+1:])
	if err != nil || layer < 0 {
		return "", 0, false
	}
	return name[:cut], layer, true
}
