package dispatchconservation

import (
	"bufio"
	"os"
	"strings"
)

const DefaultTailLines = 10000

// ReadTailLines bounds memory and parse work to the newest max non-empty rows.
// A ring buffer preserves append order without reading the whole file into RAM.
func ReadTailLines(path string, max int) []string {
	if max <= 0 {
		max = DefaultTailLines
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	ring := make([]string, max)
	count := 0
	sc := bufio.NewScanner(f)
	buf := make([]byte, 64<<10)
	sc.Buffer(buf, 2<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		ring[count%max] = line
		count++
	}
	n := count
	if n > max {
		n = max
	}
	out := make([]string, 0, n)
	start := count - n
	for i := 0; i < n; i++ {
		out = append(out, ring[(start+i)%max])
	}
	return out
}
