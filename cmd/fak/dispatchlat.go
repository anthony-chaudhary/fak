package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
	"os"
	"sort"
	"strings"
)

func cmdDispatchLat(argv []string) {
	fs := flag.NewFlagSet("dispatchlat", flag.ExitOnError)
	jsonOut := fs.Bool("json", false, "emit JSON")
	_ = fs.Parse(argv)
	paths := fs.Args()
	if len(paths) == 0 {
		paths = []string{filepathFromRoot(".fak", "loops.jsonl")}
	}
	rows := []map[string]int64{}
	for _, p := range paths {
		f, e := os.Open(p)
		if e != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			var ev loopmgr.Event
			if json.Unmarshal(sc.Bytes(), &ev) != nil {
				continue
			}
			row := map[string]int64{}
			for k, v := range ev.Metrics {
				if strings.HasSuffix(k, "_ms") {
					phase := strings.TrimSuffix(k, "_ms")
					if phase == "tick_total" {
						phase = "total"
					}
					row[phase] = v
				}
			}
			if len(row) > 0 {
				rows = append(rows, row)
			}
		}
		f.Close()
	}
	out := turntaxmeter.FoldDispatchLatency(rows)
	if *jsonOut {
		b, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(b))
		return
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Phase < out[j].Phase })
	fmt.Printf("%-20s %7s %8s %8s %8s\n", "PHASE", "N", "P50MS", "P90MS", "P99MS")
	for _, r := range out {
		fmt.Printf("%-20s %7d %8d %8d %8d\n", r.Phase, r.Samples, r.P50MS, r.P90MS, r.P99MS)
	}
}
func filepathFromRoot(parts ...string) string { return strings.Join(parts, string(os.PathSeparator)) }
