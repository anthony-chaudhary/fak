package toolshape

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// turn builds a Turn the way a producer-enriched recorder would: tool, verdict,
// cost, and OPEN labels (arg names/types only — never values).
func turn(tool, verdict string, bytes int64, tokens int, labels map[string]string) trajectory.Turn {
	return trajectory.Turn{
		TraceID:       "trace-1",
		Tool:          tool,
		Verdict:       verdict,
		Bytes:         bytes,
		TokenEstimate: tokens,
		Labels:        labels,
	}
}

// The issue's witness table: each canonical tool call lands in its closed
// ArgClass, sizes bucket on the log scale, and a denied turn reads as an error.
func TestFingerprintTable(t *testing.T) {
	cases := []struct {
		name string
		in   trajectory.Turn
		want ToolShape
	}{
		{
			name: "read turn is path-classed",
			in: turn("Read", "ALLOW", 512, 128, map[string]string{
				LabelArgKeys:  "file_path,limit,offset",
				LabelArgTypes: "file_path:string,limit:number,offset:number",
			}),
			want: ToolShape{
				Tool: "Read", Verdict: "ALLOW",
				ArgKeys:  []string{"file_path", "limit", "offset"},
				ArgCount: 3, ArgClass: ArgClassPath,
				OutBytesBucket: Bucket1K, OutTokensBucket: Bucket1K,
			},
		},
		{
			name: "grep turn is pattern-classed even with a path arg",
			in: turn("Grep", "ALLOW", 90, 20, map[string]string{
				LabelArgKeys: "pattern,path,glob",
			}),
			want: ToolShape{
				Tool: "Grep", Verdict: "ALLOW",
				ArgKeys:  []string{"glob", "path", "pattern"},
				ArgCount: 3, ArgClass: ArgClassPattern,
				OutBytesBucket: Bucket100, OutTokensBucket: Bucket100,
			},
		},
		{
			name: "bash turn is command-classed",
			in: turn("Bash", "ALLOW", 0, 0, map[string]string{
				LabelArgKeys: "command,description,timeout",
			}),
			want: ToolShape{
				Tool: "Bash", Verdict: "ALLOW",
				ArgKeys:  []string{"command", "description", "timeout"},
				ArgCount: 3, ArgClass: ArgClassCommand,
				OutBytesBucket: BucketZero, OutTokensBucket: BucketZero,
				Empty: true,
			},
		},
		{
			name: "write turn (payload + path target) is mixed",
			in: turn("Write", "ALLOW", 30000, 7500, map[string]string{
				LabelArgKeys: "file_path,content",
			}),
			want: ToolShape{
				Tool: "Write", Verdict: "ALLOW",
				ArgKeys:  []string{"content", "file_path"},
				ArgCount: 2, ArgClass: ArgClassMixed,
				OutBytesBucket: BucketOver, OutTokensBucket: Bucket10K,
			},
		},
		{
			name: "8k-byte result buckets 1k-10k",
			in:   turn("Read", "ALLOW", 8192, 0, nil),
			want: ToolShape{
				Tool: "Read", Verdict: "ALLOW",
				ArgClass:       ArgClassUnknown,
				OutBytesBucket: Bucket10K, OutTokensBucket: BucketZero,
			},
		},
		{
			name: "denied turn is an error",
			in:   turn("Bash", "DENY", 0, 0, nil),
			want: ToolShape{
				Tool: "Bash", Verdict: "DENY",
				ArgClass:       ArgClassUnknown,
				OutBytesBucket: BucketZero, OutTokensBucket: BucketZero,
				Empty: true, IsError: true,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Fingerprint(tc.in)
			got.ArgKeySig = "" // signature stability asserted separately
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Fingerprint = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// Totality: a zero Turn and a Turn with empty labels both fingerprint cleanly —
// unknown class, no keys, no signature, no panic.
func TestFingerprintTotalOnAbsentLabels(t *testing.T) {
	for _, in := range []trajectory.Turn{
		{},
		turn("Mystery", "", 0, 0, map[string]string{}),
		turn("Mystery", "", 0, 0, map[string]string{"unrelated": "label"}),
	} {
		got := Fingerprint(in) // must not panic
		if got.ArgClass != ArgClassUnknown {
			t.Fatalf("absent labels: ArgClass = %q, want %q", got.ArgClass, ArgClassUnknown)
		}
		if len(got.ArgKeys) != 0 || got.ArgCount != 0 || got.ArgKeySig != "" {
			t.Fatalf("absent labels must yield no keys/signature, got %+v", got)
		}
		if !got.Empty {
			t.Fatalf("a zero-cost, digest-less turn must read Empty, got %+v", got)
		}
	}
}

// Determinism: same Turn in, same ToolShape out.
func TestFingerprintDeterministic(t *testing.T) {
	in := turn("Grep", "ALLOW", 8192, 2048, map[string]string{
		LabelArgKeys:  "pattern,path",
		LabelArgTypes: "pattern:string,path:string",
	})
	a, b := Fingerprint(in), Fingerprint(in)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Fingerprint not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

// ArgKeySig is a key-set+type signature: stable across turns with the same
// keys/types (regardless of stamp order), different when a key is added or a
// type changes — and never a function of any raw value.
func TestArgKeySigStability(t *testing.T) {
	base := turn("Read", "ALLOW", 10, 5, map[string]string{
		LabelArgKeys:  "file_path,limit",
		LabelArgTypes: "file_path:string,limit:number",
	})
	sameShape := turn("Read", "DENY", 999999, 0, map[string]string{
		LabelArgKeys:  "limit, file_path", // different order + spacing, same key-set
		LabelArgTypes: "limit:number,file_path:string",
	})
	keyAdded := turn("Read", "ALLOW", 10, 5, map[string]string{
		LabelArgKeys:  "file_path,limit,offset",
		LabelArgTypes: "file_path:string,limit:number,offset:number",
	})
	typeChanged := turn("Read", "ALLOW", 10, 5, map[string]string{
		LabelArgKeys:  "file_path,limit",
		LabelArgTypes: "file_path:string,limit:string",
	})

	sig := Fingerprint(base).ArgKeySig
	if sig == "" {
		t.Fatal("a keyed turn must carry an ArgKeySig")
	}
	if got := Fingerprint(sameShape).ArgKeySig; got != sig {
		t.Fatalf("same key-set/types must share a signature: %q vs %q", got, sig)
	}
	if got := Fingerprint(keyAdded).ArgKeySig; got == sig {
		t.Fatal("adding a key must change the signature")
	}
	if got := Fingerprint(typeChanged).ArgKeySig; got == sig {
		t.Fatal("changing a type must change the signature")
	}
}

// Producer-stamped output flags fold through: truncation, error, and a
// digest-bearing zero-cost result is NOT empty.
func TestOutputFlags(t *testing.T) {
	tr := Fingerprint(turn("Read", "ALLOW", 50, 10, map[string]string{LabelTruncated: "true"}))
	if !tr.Truncated {
		t.Fatalf("truncated label must fold through, got %+v", tr)
	}
	er := Fingerprint(turn("Bash", "ALLOW", 50, 10, map[string]string{LabelError: "1"}))
	if !er.IsError {
		t.Fatalf("error label must fold through, got %+v", er)
	}
	withDigest := trajectory.Turn{Tool: "Read", ResultDigest: "sha256:abc"}
	if got := Fingerprint(withDigest); got.Empty {
		t.Fatalf("a digest-bearing result is not empty, got %+v", got)
	}
	q := Fingerprint(turn("Bash", "QUARANTINE", 0, 0, nil))
	if !q.IsError {
		t.Fatalf("QUARANTINE must read as an error, got %+v", q)
	}
}

// The bucket fold covers the whole closed vocabulary, boundaries included.
func TestBucketBoundaries(t *testing.T) {
	cases := []struct {
		n    int64
		want string
	}{
		{-5, BucketZero}, {0, BucketZero},
		{1, Bucket100}, {100, Bucket100},
		{101, Bucket1K}, {1000, Bucket1K},
		{1001, Bucket10K}, {8192, Bucket10K}, {10000, Bucket10K},
		{10001, BucketOver},
	}
	for _, tc := range cases {
		if got := bucket(tc.n); got != tc.want {
			t.Fatalf("bucket(%d) = %q, want %q", tc.n, got, tc.want)
		}
	}
}
