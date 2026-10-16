package agent

import "strings"

type stopSuffixGuard struct {
	emit    StreamSink
	stops   []string
	held    string
	stopped bool
}

func newStopSuffixGuard(emit StreamSink, stops []string) *stopSuffixGuard {
	filtered := make([]string, 0, len(stops))
	for _, stop := range stops {
		if stop != "" {
			filtered = append(filtered, stop)
		}
	}
	return &stopSuffixGuard{emit: emit, stops: filtered}
}

func (g *stopSuffixGuard) feed(piece string) error {
	if piece == "" || g.stopped {
		return nil
	}
	buf := g.held + piece
	g.held = ""
	if trimmed, hit := checkStop(buf, g.stops); hit {
		g.stopped = true
		return g.emitText(trimmed)
	}
	keep := 0
	for _, stop := range g.stops {
		max := len(stop) - 1
		if max > len(buf) {
			max = len(buf)
		}
		for n := max; n > keep; n-- {
			if strings.HasSuffix(buf, stop[:n]) {
				keep = n
				break
			}
		}
	}
	if keep > 0 {
		g.held = buf[len(buf)-keep:]
		buf = buf[:len(buf)-keep]
	}
	return g.emitText(buf)
}

func (g *stopSuffixGuard) flush() error {
	if g.stopped {
		return nil
	}
	held := g.held
	g.held = ""
	return g.emitText(held)
}

func (g *stopSuffixGuard) emitText(text string) error {
	if text == "" {
		return nil
	}
	return g.emit(text)
}
