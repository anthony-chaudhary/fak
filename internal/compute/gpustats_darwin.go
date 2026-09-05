//go:build darwin

package compute

import (
	"context"
	"encoding/xml"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// gpustats_darwin.go — Apple Silicon GPU telemetry via IORegistry (#11319).
//
// On macOS (Apple Silicon), nvidia-smi does not exist. Unified memory means
// total VRAM corresponds to system memory (`hw.memsize`), while Metal-allocated
// VRAM and GPU utilization are tracked by IOAccelerator in the IORegistry.
//
// We probe `ioreg -a -r -d 1 -c IOAccelerator` to extract `PerformanceStatistics`:
//   - "Alloc system memory": bytes allocated to Metal/GPU.
//   - "Device Utilization %" or "GPU Activity(%)": GPU activity percentage (0..100).

type ioregRunner func(ctx context.Context, args ...string) ([]byte, error)

func execIOReg(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "/usr/sbin/ioreg", args...)
	return cmd.Output()
}

// AppleSiliconGPUStats reads Apple Silicon GPU memory and utilization via IORegistry.
// Fail-soft: any error (no IOAccelerator, timeout, unparseable output) returns (nil, false).
func AppleSiliconGPUStats() (stats []GPUStat, present bool) {
	return appleSiliconGPUStats(execIOReg)
}

func appleSiliconGPUStats(run ioregRunner) ([]GPUStat, bool) {
	if run == nil {
		return nil, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	type runResult struct {
		out []byte
		err error
	}
	resCh := make(chan runResult, 1)
	go func() {
		out, err := run(ctx, "-a", "-r", "-d", "1", "-c", "IOAccelerator")
		resCh <- runResult{out: out, err: err}
	}()

	var out []byte
	select {
	case <-ctx.Done():
		return nil, false
	case res := <-resCh:
		if res.err != nil || len(res.out) == 0 {
			return nil, false
		}
		out = res.out
	}

	totalMem, _, memKnown := hostSystemMemory()
	var vramTotal uint64
	if memKnown && totalMem > 0 {
		vramTotal = uint64(totalMem)
	}

	stats, ok := parseIORegAcceleratorXML(out, vramTotal)
	if !ok || len(stats) == 0 {
		return nil, false
	}
	return stats, true
}

// parseIORegAcceleratorXML parses the XML plist from `ioreg -a -r -d 1 -c IOAccelerator`.
// It searches for dictionaries immediately inside `PerformanceStatistics` and extracts
// "Alloc system memory" and "Device Utilization %" (or "GPU Activity(%)").
func parseIORegAcceleratorXML(data []byte, vramTotal uint64) ([]GPUStat, bool) {
	decoder := xml.NewDecoder(strings.NewReader(string(data)))

	var stats []GPUStat
	inPerfStats := false
	perfDepth := 0
	depth := 0
	var currentKey string
	var textBuf strings.Builder

	var allocMem uint64
	var allocFound bool
	var utilPct float64
	var utilFound bool

	for {
		tok, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				break
			}
			break
		}

		switch el := tok.(type) {
		case xml.StartElement:
			depth++
			textBuf.Reset()

		case xml.CharData:
			textBuf.Write(el)

		case xml.EndElement:
			text := strings.TrimSpace(textBuf.String())
			switch el.Name.Local {
			case "key":
				if text == "PerformanceStatistics" {
					inPerfStats = true
					perfDepth = depth
					allocMem = 0
					allocFound = false
					utilPct = 0
					utilFound = false
				} else if inPerfStats && depth == perfDepth+1 {
					currentKey = text
				}
			case "integer", "real":
				if inPerfStats && depth == perfDepth+1 && currentKey != "" {
					switch currentKey {
					case "Alloc system memory":
						if v, err := strconv.ParseUint(text, 10, 64); err == nil {
							allocMem = v
							allocFound = true
						}
					case "Device Utilization %", "GPU Activity(%)":
						if v, err := strconv.ParseFloat(text, 64); err == nil {
							utilPct = v
							utilFound = true
						}
					}
					currentKey = ""
				}
			case "dict":
				if inPerfStats && depth == perfDepth {
					if allocFound || utilFound {
						stats = append(stats, GPUStat{
							Index:          len(stats),
							VRAMUsedBytes:  allocMem,
							VRAMTotalBytes: vramTotal,
							UtilizationPct: utilPct,
						})
					}
					inPerfStats = false
					perfDepth = 0
					currentKey = ""
				}
			}
			textBuf.Reset()
			depth--
		}
	}

	if len(stats) == 0 {
		return nil, false
	}
	return stats, true
}
