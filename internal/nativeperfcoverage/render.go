package nativeperfcoverage

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"strings"
)

// RenderMode selects the synthetic visual state demonstrated by an aggregate
// witness. It does not convert fixture data into a live-performance claim.
type RenderMode string

const (
	// RenderPopulated renders the dashboard preview matrix in the populated fixture state.
	RenderPopulated RenderMode = "POPULATED"
	// RenderUnavailable renders the dashboard preview matrix in the explicit unavailable state.
	RenderUnavailable RenderMode = "UNAVAILABLE"
)

var (
	background = color.RGBA{R: 16, G: 21, B: 30, A: 255}
	card       = color.RGBA{R: 31, G: 41, B: 55, A: 255}
	green      = color.RGBA{R: 22, G: 163, B: 74, A: 255}
	amber      = color.RGBA{R: 217, G: 119, B: 6, A: 255}
	blue       = color.RGBA{R: 37, G: 99, B: 235, A: 255}
	white      = color.RGBA{R: 241, G: 245, B: 249, A: 255}
	muted      = color.RGBA{R: 148, G: 163, B: 184, A: 255}
)

// RenderAggregatePNG renders every dashboard and panel with only standard
// library image primitives. The fixed geometry and bitmap font make the PNG
// byte-for-byte deterministic for a given matrix and mode.
func RenderAggregatePNG(w io.Writer, matrix Matrix, mode RenderMode) error {
	if mode != RenderPopulated && mode != RenderUnavailable {
		return fmt.Errorf("unsupported render mode %q", mode)
	}
	const width, columns, cardHeight = 1600, 4, 64
	height := 96
	for _, dashboard := range matrix.Dashboards {
		rows := (len(dashboard.Panels) + columns - 1) / columns
		height += 58 + rows*(cardHeight+12)
		if len(dashboard.Queries) > 0 {
			height += 36
		}
	}
	height += 28
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(canvas, canvas.Bounds(), &image.Uniform{C: background}, image.Point{}, draw.Src)
	drawText(canvas, 28, 24, 3, white, "FAK NATIVE PERFORMANCE DASHBOARD COVERAGE")
	drawText(canvas, 28, 58, 2, muted, "SYNTHETIC VISUAL CONTRACT - NOT LIVE PERFORMANCE EVIDENCE - "+string(mode))

	y := 96
	for _, dashboard := range matrix.Dashboards {
		drawText(canvas, 28, y, 2, white, strings.ToUpper(dashboard.UID)+"  "+strings.ToUpper(dashboard.Title))
		y += 34
		for index, panel := range dashboard.Panels {
			column := index % columns
			row := index / columns
			x := 28 + column*388
			py := y + row*(cardHeight+12)
			draw.Draw(canvas, image.Rect(x, py, x+370, py+cardHeight), &image.Uniform{C: card}, image.Point{}, draw.Src)
			stateColor, state := blue, "STATIC"
			if len(panel.Queries) > 0 {
				if mode == RenderPopulated {
					stateColor, state = green, "POPULATED"
				} else {
					stateColor, state = amber, "UNAVAILABLE"
				}
			}
			draw.Draw(canvas, image.Rect(x, py, x+8, py+cardHeight), &image.Uniform{C: stateColor}, image.Point{}, draw.Src)
			drawText(canvas, x+18, py+10, 2, white, fmt.Sprintf("P%02d %s", panel.ID, truncateASCII(panel.Title, 28)))
			drawText(canvas, x+18, py+38, 1, stateColor, state+fmt.Sprintf("  TARGETS=%d", len(panel.Queries)))
		}
		rows := (len(dashboard.Panels) + columns - 1) / columns
		y += rows * (cardHeight + 12)
		if len(dashboard.Queries) > 0 {
			var annotations, variables int
			for _, query := range dashboard.Queries {
				if query.Kind == Annotation {
					annotations++
				} else if query.Kind == Variable {
					variables++
				}
			}
			drawText(canvas, 32, y+3, 1, muted, fmt.Sprintf("AUXILIARY QUERIES  ANNOTATIONS=%d  VARIABLES=%d", annotations, variables))
			y += 36
		}
		y += 24
	}
	return png.Encode(w, canvas)
}

func truncateASCII(text string, maximum int) string {
	text = strings.ToUpper(text)
	var out []rune
	for _, r := range text {
		if r < 32 || r > 126 {
			r = ' '
		}
		out = append(out, r)
		if len(out) == maximum {
			break
		}
	}
	return string(out)
}

func drawText(dst draw.Image, x, y, scale int, ink color.Color, text string) {
	cursor := x
	for _, raw := range strings.ToUpper(text) {
		glyph, ok := font5x7[raw]
		if !ok {
			glyph = font5x7['?']
		}
		for row, bits := range glyph {
			for column := 0; column < 5; column++ {
				if bits&(1<<uint(4-column)) == 0 {
					continue
				}
				rect := image.Rect(cursor+column*scale, y+row*scale, cursor+(column+1)*scale, y+(row+1)*scale)
				draw.Draw(dst, rect, &image.Uniform{C: ink}, image.Point{}, draw.Src)
			}
		}
		cursor += 6 * scale
	}
}

var font5x7 = map[rune][7]byte{
	' ': {},
	'?': {14, 17, 1, 2, 4, 0, 4},
	'A': {14, 17, 17, 31, 17, 17, 17}, 'B': {30, 17, 17, 30, 17, 17, 30},
	'C': {14, 17, 16, 16, 16, 17, 14}, 'D': {30, 17, 17, 17, 17, 17, 30},
	'E': {31, 16, 16, 30, 16, 16, 31}, 'F': {31, 16, 16, 30, 16, 16, 16},
	'G': {14, 17, 16, 23, 17, 17, 15}, 'H': {17, 17, 17, 31, 17, 17, 17},
	'I': {31, 4, 4, 4, 4, 4, 31}, 'J': {1, 1, 1, 1, 17, 17, 14},
	'K': {17, 18, 20, 24, 20, 18, 17}, 'L': {16, 16, 16, 16, 16, 16, 31},
	'M': {17, 27, 21, 21, 17, 17, 17}, 'N': {17, 25, 21, 19, 17, 17, 17},
	'O': {14, 17, 17, 17, 17, 17, 14}, 'P': {30, 17, 17, 30, 16, 16, 16},
	'Q': {14, 17, 17, 17, 21, 18, 13}, 'R': {30, 17, 17, 30, 20, 18, 17},
	'S': {15, 16, 16, 14, 1, 1, 30}, 'T': {31, 4, 4, 4, 4, 4, 4},
	'U': {17, 17, 17, 17, 17, 17, 14}, 'V': {17, 17, 17, 17, 17, 10, 4},
	'W': {17, 17, 17, 21, 21, 21, 10}, 'X': {17, 17, 10, 4, 10, 17, 17},
	'Y': {17, 17, 10, 4, 4, 4, 4}, 'Z': {31, 1, 2, 4, 8, 16, 31},
	'0': {14, 17, 19, 21, 25, 17, 14}, '1': {4, 12, 4, 4, 4, 4, 14},
	'2': {14, 17, 1, 2, 4, 8, 31}, '3': {30, 1, 1, 14, 1, 1, 30},
	'4': {2, 6, 10, 18, 31, 2, 2}, '5': {31, 16, 16, 30, 1, 1, 30},
	'6': {14, 16, 16, 30, 17, 17, 14}, '7': {31, 1, 2, 4, 8, 8, 8},
	'8': {14, 17, 17, 14, 17, 17, 14}, '9': {14, 17, 17, 15, 1, 1, 14},
	'-': {0, 0, 0, 31, 0, 0, 0}, '_': {0, 0, 0, 0, 0, 0, 31},
	'.': {0, 0, 0, 0, 0, 12, 12}, ':': {0, 12, 12, 0, 12, 12, 0},
	'/': {1, 2, 2, 4, 8, 8, 16}, '=': {0, 0, 31, 0, 31, 0, 0},
}
