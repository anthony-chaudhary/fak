// fak-test:runtime fast est=1s
package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type ggufKV struct {
	key string
	typ uint32
	u32 uint32
	f32 float32
	str string
}

func u32KV(k string, v uint32) ggufKV  { return ggufKV{key: k, typ: 4, u32: v} }
func f32KV(k string, v float32) ggufKV { return ggufKV{key: k, typ: 6, f32: v} }
func strKV(k, v string) ggufKV         { return ggufKV{key: k, typ: 8, str: v} }

func writeHeaderOnlyGGUF(t *testing.T, kvs []ggufKV) string {
	t.Helper()
	var b bytes.Buffer
	le := binary.LittleEndian
	putStr := func(s string) {
		_ = binary.Write(&b, le, uint64(len(s)))
		b.WriteString(s)
	}
	b.WriteString("GGUF")
	_ = binary.Write(&b, le, uint32(3))
	_ = binary.Write(&b, le, uint64(0))
	_ = binary.Write(&b, le, uint64(len(kvs)))
	for _, kv := range kvs {
		putStr(kv.key)
		_ = binary.Write(&b, le, kv.typ)
		switch kv.typ {
		case 4:
			_ = binary.Write(&b, le, kv.u32)
		case 6:
			_ = binary.Write(&b, le, math.Float32bits(kv.f32))
		case 8:
			putStr(kv.str)
		}
	}
	path := filepath.Join(t.TempDir(), "shard.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func glmHeader(keyLenMLA, valueLenMLA uint32) []ggufKV {
	return []ggufKV{
		strKV("general.architecture", "glm-dsa"),
		u32KV("glm-dsa.embedding_length", 64),
		u32KV("glm-dsa.block_count", 2),
		u32KV("glm-dsa.attention.head_count", 4),
		u32KV("glm-dsa.feed_forward_length", 128),
		f32KV("glm-dsa.attention.layer_norm_rms_epsilon", 1e-5),
		u32KV("glm-dsa.attention.qk_rope_head_dim", 64),
		u32KV("glm-dsa.attention.key_length_mla", keyLenMLA),
		u32KV("glm-dsa.attention.value_length_mla", valueLenMLA),
		u32KV("glm-dsa.attention.kv_lora_rank", 512),
		u32KV("glm-dsa.attention.q_lora_rank", 1536),
	}
}

func TestRunNoArgsExitsUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(nil, &out, &errb); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if out.Len() != 0 || errb.Len() == 0 {
		t.Fatalf("usage must go to stderr only: stdout=%q stderr=%q", out.String(), errb.String())
	}
}

func TestRunMissingShardExitsOne(t *testing.T) {
	var out bytes.Buffer
	if code := run([]string{filepath.Join(t.TempDir(), "absent.gguf")}, &out, &bytes.Buffer{}); code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if out.Len() != 0 {
		t.Fatalf("stdout on open failure = %q, want empty", out.String())
	}
}

func TestRunMissingArchitectureExitsOne(t *testing.T) {
	path := writeHeaderOnlyGGUF(t, []ggufKV{u32KV("x.block_count", 1)})
	if code := run([]string{path}, &bytes.Buffer{}, &bytes.Buffer{}); code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
}

func TestRunFixedGLMDimsReportDimsOK(t *testing.T) {
	path := writeHeaderOnlyGGUF(t, glmHeader(256, 256))
	var out, errb bytes.Buffer
	if code := run([]string{path}, &out, &errb); code != 0 {
		t.Fatalf("exit = %d, want 0; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"ModelType=glm_moe_dsa\n",
		"QKNopeHeadDim=192 VHeadDim=256 QKRopeHeadDim=64\n",
		"KVLoraRank=512 QLoraRank=1536\n",
		"DIMS_OK qkNope=192 vHead=256\n",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("stdout missing %q:\n%s", want, got)
		}
	}
}

func TestRunLatentSizedDimsExitThree(t *testing.T) {
	path := writeHeaderOnlyGGUF(t, glmHeader(576, 512))
	var out bytes.Buffer
	if code := run([]string{path}, &out, &bytes.Buffer{}); code != 3 {
		t.Fatalf("exit = %d, want 3; stdout=%q", code, out.String())
	}
	if !strings.Contains(out.String(), "DIMS_WRONG ") || strings.Contains(out.String(), "DIMS_OK") {
		t.Fatalf("want DIMS_WRONG verdict only:\n%s", out.String())
	}
}
