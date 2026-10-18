package agent

import (
	"errors"
	"strings"
)

// dflash2StopPreview is private, owned, and side-effect-free. Its scanner can
// replace the live scanner only after the corresponding token prefix commits.
type dflash2StopPreview struct {
	tokens  []int
	pieces  []string
	scanner *incrementalStopScanner
	trim    int
	stopped bool
}

func cloneDFlash2StopScanner(s *incrementalStopScanner) *incrementalStopScanner {
	if s == nil {
		return nil
	}
	out := *s
	out.tail = make([]byte, len(s.tail), cap(s.tail))
	out.hist = make([]byte, len(s.hist), cap(s.hist))
	copy(out.tail, s.tail)
	copy(out.hist, s.hist)
	out.stops = append([]string(nil), s.stops...)
	return &out
}

// previewDFlash2Stop decodes only the prefix that can be committed. Token-ID
// stops exclude their token; a string stop includes its triggering token.
// The existing stream projector still withholds cross-token stop suffixes.
func previewDFlash2Stop(tokens []int, stops map[int]bool, scanner *incrementalStopScanner, decode func([]int) (string, error)) (dflash2StopPreview, error) {
	if len(tokens) < 1 || len(tokens) > 8 || scanner == nil || decode == nil {
		return dflash2StopPreview{}, errors.New("dflash2: bounded tokens, scanner and decoder required")
	}
	if cap(scanner.tail) > 64<<10 || len(scanner.stops) > 1024 {
		return dflash2StopPreview{}, errors.New("dflash2: stop preview budget exceeded")
	}
	out := dflash2StopPreview{scanner: cloneDFlash2StopScanner(scanner)}
	for _, token := range tokens {
		if stops[token] {
			out.stopped = true
			break
		}
		piece, err := decode([]int{token})
		if err != nil {
			return dflash2StopPreview{}, err
		}
		if len(piece) > 64<<10 {
			return dflash2StopPreview{}, errors.New("dflash2: decoded token piece budget exceeded")
		}
		out.tokens = append(out.tokens, token)
		out.pieces = append(out.pieces, piece)
		// Record the exact longest suffix before appendPiece trims its own
		// bounded history. This never materializes the whole transcript.
		tail := string(out.scanner.tail) + piece
		for _, stop := range out.scanner.stops {
			if stop != "" && len(stop) > out.trim && strings.HasSuffix(tail, stop) {
				out.trim = len(stop)
			}
		}
		hit := out.scanner.appendPiece(piece)
		if hit != (out.trim != 0) {
			return dflash2StopPreview{}, errors.New("dflash2: inconsistent stop preview")
		}
		if hit {
			out.stopped = true
			break
		}
	}
	return out, nil
}
