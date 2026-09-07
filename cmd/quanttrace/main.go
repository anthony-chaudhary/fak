// Quanttrace explains an offline Q4_K representation flow and writes its receipt.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/bench"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("quanttrace", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "raw row-major Q4_K payload (default: deterministic 2x256 demo)")
	rows := fs.Int("rows", 2, "weight rows")
	cols := fs.Int("columns", 256, "weight columns, multiple of 256")
	width := fs.Int("width", 2, "interleave width 1..256")
	indexes := fs.String("samples", "0,31,32,255", "up to 32 comma-separated logical row-major indexes")
	out := fs.String("out", "", "write a JSON receipt file")
	asJSON := fs.Bool("json", false, "print JSON instead of explanation")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	fail := func(err error) int { fmt.Fprintln(stderr, err); return 1 }
	if fs.NArg() != 0 {
		return fail(fmt.Errorf("quanttrace: unexpected positional arguments"))
	}
	if *input != "" && *out != "" {
		inInfo, inErr := os.Stat(*input)
		outInfo, outErr := os.Stat(*out)
		if inErr != nil {
			return fail(inErr)
		}
		if outErr != nil && !os.IsNotExist(outErr) {
			return fail(outErr)
		}
		if outErr == nil && os.SameFile(inInfo, outInfo) {
			return fail(fmt.Errorf("quanttrace: output must not overwrite input weights"))
		}
	}
	var raw []byte
	if *input == "" {
		raw = bench.QuantTraceDemo()
	} else {
		f, err := os.Open(*input)
		if err != nil {
			return fail(err)
		}
		raw, err = io.ReadAll(io.LimitReader(f, bench.QuantTraceMaxBytes+1))
		closeErr := f.Close()
		if err != nil {
			return fail(err)
		}
		if closeErr != nil {
			return fail(closeErr)
		}
		if len(raw) > bench.QuantTraceMaxBytes {
			return fail(fmt.Errorf("quanttrace: input exceeds %d bytes", bench.QuantTraceMaxBytes))
		}
	}
	// Bound dimensions before allocating the deterministic activation vector.
	if *cols <= 0 || *cols%256 != 0 || *cols/256 > bench.QuantTraceMaxBytes/144 {
		return fail(fmt.Errorf("quanttrace: invalid columns"))
	}
	var samples []int
	if *indexes != "" {
		parts := strings.Split(*indexes, ",")
		if len(parts) > 32 {
			return fail(fmt.Errorf("quanttrace: at most 32 samples"))
		}
		for _, s := range parts {
			i, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil {
				return fail(err)
			}
			samples = append(samples, i)
		}
	}
	x := make([]float32, *cols)
	for i := range x {
		x[i] = float32(i%17-8) / 8
	}
	r, err := bench.TraceQ4K(raw, *rows, *cols, *width, x, samples)
	if err != nil {
		return fail(err)
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fail(err)
	}
	data = append(data, '\n')
	if *out != "" {
		if err := os.WriteFile(*out, data, 0600); err != nil {
			return fail(err)
		}
	}
	if *asJSON {
		_, err = stdout.Write(data)
	} else {
		_, err = fmt.Fprintf(stdout, "Q4_K %dx%d, %.1f bits/value including metadata\nUnpack: %d packed bytes -> %d one-byte codes; scales/minima interpret codes.\nRepack: same %d bytes, row-major -> %d-wide interleaved; exact inverse: %t.\nExpand: allocate %d float32 bytes in this offline diagnostic; w = (d * scale) * code - dmin * min.\nContract: W[%d,%d] x x[%d] -> y[%d] (%d bytes); layout outputs bit-identical: %t, max error: %g.\nActivation: x[i]=(i%%17-8)/8; hash included in receipt.\nExpansion decodes quantized values; original pre-quantization floats and their error are unknown.\nThese are software diagnostic allocations, not device traffic or production residency.\n", r.Rows, r.Columns, r.EffectiveBitsPerValue, r.PackedBytes, r.UnpackedCodeBytes, r.PackedBytes, r.Width, r.ExactRoundTrip, r.ExpandedFloatBytes, r.Rows, r.Columns, r.Columns, r.Rows, r.OutputBytes, r.ExactContraction, r.MaxContractionError)
		if err == nil {
			_, err = fmt.Fprintf(stdout, "Output y[0:%d]: %v\n", len(r.OutputPrefix), r.OutputPrefix)
		}
		if err == nil {
			for _, s := range r.Samples {
				_, err = fmt.Fprintf(stdout, "sample %d: byte %d -> repacked byte %d, shift %d, code %d -> value %g (d=%g scale=%d dmin=%g min=%d)\n", s.Index, s.SourceByte, s.RepackedByte, s.NibbleShift, s.Code, s.Value, s.D, s.ScaleCode, s.DMin, s.MinCode)
				if err != nil {
					break
				}
			}
		}
	}
	if err != nil {
		return fail(err)
	}
	if !r.ExactRoundTrip || !r.ExactContraction {
		return fail(fmt.Errorf("quanttrace: transformation witness failed"))
	}
	return 0
}
