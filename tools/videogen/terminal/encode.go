// encode.go — the MP4, and the check that it is the video the timeline asked
// for.
//
// WHY THE ENCODE LIVES IN HERE. Until #750 the last step of the regen was an
// ffmpeg command line typed out of SHARE-README.txt prose. Two things are wrong
// with that. Law 2: the raw command belongs INSIDE the tool, not in the hands
// of whoever regenerates the artifact next. And nothing checked the result —
// which matters because this exact seam has already shipped a wrong duration
// twice. The concat demuxer gives the FINAL entry the duration of the entry
// BEFORE it, which silently shipped an 83.3 s cut as 79.3 s; the "obvious" fix
// shipped the same playlist as 95.3 s. hires.go quantises the playlist to one
// entry per frame so that rule is harmless, but a defused bug with no probe on
// it is a bug waiting for the next edit. checkDuration is that probe: the
// renderer knows exactly how long the cut is meant to be, so it asks ffprobe
// how long the file actually is and fails on the difference.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// nixInstall is the no-sudo route to ffmpeg on the box this was written for.
// A missing encoder should tell you how to get one, not stack-trace.
const nixInstall = `nix --extra-experimental-features 'nix-command flakes' profile install nixpkgs#ffmpeg`

func needTool(name string) (string, error) {
	if override := os.Getenv("VIDEOGEN_" + strings.ToUpper(name)); override != "" {
		info, err := os.Stat(override)
		if err != nil || info.IsDir() {
			return "", fmt.Errorf("VIDEOGEN_%s=%q is not an executable file", strings.ToUpper(name), override)
		}
		return override, nil
	}
	p, err := exec.LookPath(name)
	if err != nil {
		return "", fmt.Errorf("%s not on PATH — install it with:\n    %s", name, nixInstall)
	}
	return p, nil
}

// encodeMP4 turns the PNG sequence into the shipped video and then verifies it.
//
// The flags are pinned here rather than documented: -preset veryslow is not
// belt-and-braces, it is what makes flat-colour terminal glyphs SMALLER as well
// as sharper (measured: veryslow/crf18 beat slow/crf26 on both). The chapter
// metadata comes in as a second input and is mapped over the video stream, so
// scrubbing to a named step is a click rather than a hunt.
//
// CRF 22, measured on this content 2026-07-28 at the paced 291.5 s length:
// crf18 -> 54 MiB, crf20 -> 50, crf22 -> 46. The knob barely moves it, because
// the bits go into ~478 distinct full-screen scrolls of antialiased text, not
// into the holds. 22 was picked over 18 for one reason that is not quality:
// 46 MiB stays under GitHub's 50 MiB blob warning. The frame at 135 s — the
// negative arm's refusal, the densest store paths in the cut — was extracted
// from both encodes and compared; every hash character is identical to the eye.
// Do not raise it further without repeating that check: the sha256 strings
// being legible at full screen is the whole reason this renders at 1960x1192.
func encodeMP4(playlist, chapters, out string, wantSecs, tolerance float64) error {
	ffmpeg, err := needTool("ffmpeg")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return fmt.Errorf("create MP4 output directory: %w", err)
	}
	args := []string{
		"-y", "-hide_banner", "-loglevel", "error",
		"-f", "concat", "-safe", "0", "-i", playlist,
		"-i", chapters,
		"-map", "0:v", "-map_metadata", "1",
		"-vf", "fps=15,format=yuv420p",
		"-c:v", "libx264", "-preset", "veryslow", "-crf", "22", "-tune", "animation",
		"-movflags", "+faststart",
		out,
	}
	fmt.Fprintf(os.Stderr, "video: encoding %s (this is the slow part)\n", out)
	cmd := exec.Command(ffmpeg, args...)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg: %w", err)
	}
	st, err := os.Stat(out)
	if err != nil {
		return err
	}
	got, chaps, err := probe(out)
	if err != nil {
		return err
	}
	fmt.Printf("wrote %s: %.1fs, %d chapters, %.1f MiB\n", out, got, chaps, float64(st.Size())/(1<<20))
	if d := math.Abs(got - wantSecs); d > tolerance {
		return fmt.Errorf("encoded duration %.2fs but the timeline says %.2fs (off by %.2fs, tolerance %.2fs) — "+
			"this is the ffconcat last-entry trap; see hires.go", got, wantSecs, d, tolerance)
	}
	return nil
}

// probe reads back the duration and chapter count of the file that actually
// shipped. ⛔ Read the DELIVERED file, never the intermediate: the intermediate
// is what every one of these bugs looked correct in.
func probe(path string) (secs float64, chapters int, err error) {
	ffprobe, err := needTool("ffprobe")
	if err != nil {
		return 0, 0, err
	}
	out, err := exec.Command(ffprobe, "-v", "error",
		"-show_entries", "format=duration", "-show_chapters",
		"-of", "json", path).Output()
	if err != nil {
		return 0, 0, fmt.Errorf("ffprobe: %w", err)
	}
	var v struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
		Chapters []struct {
			Tags map[string]string `json:"tags"`
		} `json:"chapters"`
	}
	if err := json.Unmarshal(out, &v); err != nil {
		return 0, 0, fmt.Errorf("ffprobe json: %w", err)
	}
	secs, err = strconv.ParseFloat(v.Format.Duration, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("ffprobe duration %q: %w", v.Format.Duration, err)
	}
	return secs, len(v.Chapters), nil
}

// ── the pacing assertions ────────────────────────────────────────────────────

// checkPacing turns the #750 DoD into failures. Every rule here is one a bad
// edit can actually trip: setting a hold to zero, deleting the step pattern so
// nothing is a chapter any more, or letting the cut balloon past what anyone
// will sit through. It returns the failures rather than exiting so the caller
// can print all of them at once.
func checkPacing(cfg config, tl *timeline) []string {
	var f []string
	v := cfg.Verify

	if tl.Frames == 0 {
		return []string{"the render produced no frames at all"}
	}
	// ⛔ A probe that finds nothing because it LOOKED at nothing passes. If the
	// capture's grammar stops matching — a renamed banner, a changed prompt —
	// every per-class rule below goes vacuously true, so require the classes to
	// be non-empty first.
	for _, c := range []string{"step", "cmd", "rc"} {
		if tl.Counts[c] == 0 {
			f = append(f, fmt.Sprintf("no %q units were classified — the capture's structure patterns matched nothing, "+
				"so every pacing rule below is vacuous", c))
		}
	}
	if len(f) > 0 {
		return f
	}

	if d, ok := tl.MinDwell["cmd"]; ok && d < v.MinCmdSecs {
		f = append(f, fmt.Sprintf("a `$ ` command line is held only %.2fs (floor %.2fs) — "+
			"its output floods before the command can be read", d, v.MinCmdSecs))
	}
	if d, ok := tl.MinDwell["rc"]; ok && d < v.MinRCSecs {
		f = append(f, fmt.Sprintf("a return code is held only %.2fs (floor %.2fs)", d, v.MinRCSecs))
	}
	if d, ok := tl.MinDwell["emph"]; ok && d < v.MinEmphasisSecs {
		f = append(f, fmt.Sprintf("an emphasis line is held only %.2fs (floor %.2fs)", d, v.MinEmphasisSecs))
	}
	if tl.Counts["card"] > 0 {
		if cfg.Pacing.CardReveal < v.MinCardRevealSecs {
			f = append(f, fmt.Sprintf("card lines reveal in %.2fs (floor %.2fs) — the opening lands as a flash",
				cfg.Pacing.CardReveal, v.MinCardRevealSecs))
		}
		if cfg.Pacing.CardConceptHold < v.MinConceptHoldSecs {
			f = append(f, fmt.Sprintf("card concepts hold for %.2fs (floor %.2fs) — focus moves before the idea can be read",
				cfg.Pacing.CardConceptHold, v.MinConceptHoldSecs))
		}
		if len(tl.Segments) == 0 || tl.Segments[0].Kind != "card" {
			f = append(f, "the render has card frames but no opening card segment — opening pacing is unmeasurable")
		} else if tl.Segments[0].Secs < v.MinOpeningSecs {
			f = append(f, fmt.Sprintf("the opening is %.1fs (floor %.1fs) — it does not establish the story smoothly",
				tl.Segments[0].Secs, v.MinOpeningSecs))
		}
	}
	for _, s := range tl.Steps {
		if s.Secs < v.MinStepSecs {
			f = append(f, fmt.Sprintf("step %.1fs < %.1fs floor: %s", s.Secs, v.MinStepSecs, s.Title))
		}
	}
	if v.MinBars > 0 && tl.Bars < v.MinBars {
		f = append(f, fmt.Sprintf("%d bar marks drawn, want at least %d — the cut's comparisons "+
			"have gone back to being figures the viewer has to subtract", tl.Bars, v.MinBars))
	}
	if len(tl.Chapters) < v.MinChapters {
		f = append(f, fmt.Sprintf("%d chapters, want at least %d — a viewer cannot skip to a section",
			len(tl.Chapters), v.MinChapters))
	}
	for i, c := range tl.Chapters {
		if c.End <= c.Start {
			f = append(f, fmt.Sprintf("chapter %d (%s) is empty or inverted: %.2f..%.2f", i, c.Title, c.Start, c.End))
		}
		if i > 0 && c.Start < tl.Chapters[i-1].End-1e-6 {
			f = append(f, fmt.Sprintf("chapter %d (%s) starts before chapter %d ends", i, c.Title, i-1))
		}
	}
	f = append(f, checkFrameLog(tl)...)

	if v.TotalSecsMin > 0 && tl.TotalSec < v.TotalSecsMin {
		f = append(f, fmt.Sprintf("the cut is %.1fs, under the %.1fs floor — pacing was gutted", tl.TotalSec, v.TotalSecsMin))
	}
	if v.TotalSecsMax > 0 && tl.TotalSec > v.TotalSecsMax {
		f = append(f, fmt.Sprintf("the cut is %.1fs, over the %.1fs ceiling — nobody will watch it", tl.TotalSec, v.TotalSecsMax))
	}
	return f
}

// checkFrameLog asserts the per-frame audit trail is one — complete, ordered,
// and adding up to the cut the encoder is checked against.
//
// WHY THIS IS NOT JUST A SCHEMA CHECK. Every other assertion in checkPacing is
// an aggregate: MinDwell says A floor was missed, never WHICH frame missed it,
// and a reader who cannot find the frame cannot confirm the fix. The log is what
// makes those failures locatable, so the log's own completeness is load-bearing
// — and a log that silently covers 95% would make the audit read clean while
// hiding the regression. Hence: rows == frames, exactly.
//
// ⛔ The strongest rule here is the LAST one. Σ dwell must equal totalSecs, and
// totalSecs is what encodeMP4 hands ffprobe as the intended duration. So the log
// is tied to the shipped file, not to itself: any future edit that advances the
// clock without logging a frame — the exact shape of the two historical duration
// bugs — fails here, in under a second, before anything is rasterised.
func checkFrameLog(tl *timeline) []string {
	var f []string
	if n := len(tl.Log); n != tl.Frames {
		return []string{fmt.Sprintf("the frame log has %d rows for %d emitted frames — %d frames left no "+
			"record, so timeline.json is a sample and not an audit trail (#750 B1)", n, tl.Frames, tl.Frames-n)}
	}
	if len(tl.Log) == 0 {
		return nil
	}
	// The four classes a frame can only ENTER by matching a pattern — a step
	// banner, a `$ ` echo, an `[rc=]`, an emphasis rule. Their trigger line is
	// therefore never legitimately empty, and an empty one means the log lost it.
	//
	// ⛔ card, prose, out and tail are deliberately NOT here, and not as a
	// weakening: a card's spacer line and a prose group of only blank lines are
	// real frames whose source text genuinely is empty (measured on the 07-27
	// captures: 29 of 110 card frames, 26 of 74 prose, 2 of 130 out, and the
	// closing tail hold). Requiring a line there would require inventing one.
	needLine := map[string]bool{"step": true, "cmd": true, "rc": true, "emph": true}
	blank, firstBlank := 0, ""
	prev := -1.0
	sum := 0.0
	for i, fr := range tl.Log {
		sum += fr.Secs
		if fr.Secs <= 0 {
			f = append(f, fmt.Sprintf("frame %d (%s) is emitted with a dwell of %.3fs — a frame nobody can see",
				fr.N, fr.Class, fr.Secs))
			break
		}
		if fr.Start < prev-1e-9 {
			f = append(f, fmt.Sprintf("frame %d starts at %.3fs, before its predecessor at %.3fs — "+
				"the log is not in screen order", fr.N, fr.Start, prev))
			break
		}
		prev = fr.Start
		if needLine[fr.Class] && fr.Line == "" {
			blank++
			if firstBlank == "" {
				firstBlank = fmt.Sprintf("frame %d, index %d", fr.N, i)
			}
		}
	}
	if blank > 0 {
		f = append(f, fmt.Sprintf("%d frames of a class that exists to be read carry no trigger line "+
			"(first: %s) — B1 wants the line that decided every frame", blank, firstBlank))
	}
	if d := math.Abs(sum - tl.TotalSec); d > 1e-6 {
		f = append(f, fmt.Sprintf("the frame log sums to %.6fs but the cut is %.6fs (off by %.6fs) — "+
			"the log does not account for the whole video, and it is totalSecs that ffprobe is checked against",
			sum, tl.TotalSec, d))
	}
	return f
}

// report prints the pacing budget and writes timeline.json. The share table is
// the number #750 was filed about: what fraction of the runtime is cards versus
// the evidence they introduce.
func report(cfg config, tl *timeline, path string) {
	if err := tl.writeJSON(path); err != nil {
		fmt.Fprintln(os.Stderr, "video: timeline:", err)
	}
	names := []string{"card", "step", "cmd", "out", "rc", "emph", "prose", "tail"}
	label := map[string]string{
		"card": "title cards", "step": "step banners", "cmd": "command echoes",
		"out": "streamed output", "rc": "return codes", "emph": "emphasis lines",
		"prose": "narration", "tail": "closing hold",
	}
	var b strings.Builder
	fmt.Fprintf(&b, "\npacing budget — %.1fs total, %d frames, %d units skipped\n", tl.TotalSec, tl.Frames, tl.Skipped)
	for _, n := range names {
		if tl.Counts[n] == 0 {
			continue
		}
		fmt.Fprintf(&b, "  %-16s %5.1fs  %4.1f%%  over %3d units\n",
			label[n], tl.Secs[n], 100*tl.Secs[n]/tl.TotalSec, tl.Counts[n])
	}
	fmt.Fprintf(&b, "  %-16s %5d\n", "chapters", len(tl.Chapters))
	fmt.Fprintf(&b, "  %-16s %5d\n", "bar marks", tl.Bars)
	if len(tl.Steps) > 0 {
		lo, hi := tl.Steps[0], tl.Steps[0]
		for _, s := range tl.Steps {
			if s.Secs < lo.Secs {
				lo = s
			}
			if s.Secs > hi.Secs {
				hi = s
			}
		}
		fmt.Fprintf(&b, "  %d proof steps: shortest %.1fs (%s)\n", len(tl.Steps), lo.Secs, trunc(lo.Title, 58))
		fmt.Fprintf(&b, "  %*s longest  %.1fs (%s)\n", len(fmt.Sprint(len(tl.Steps))), "", hi.Secs, trunc(hi.Title, 58))
	}
	traced := 0
	for _, fr := range tl.Log {
		if fr.Line != "" {
			traced++
		}
	}
	if len(tl.Log) > 0 {
		fmt.Fprintf(&b, "  %-16s %5d rows = %d frames, %d name their trigger line (%.0f%%)\n",
			"frame log", len(tl.Log), tl.Frames, traced, 100*float64(traced)/float64(len(tl.Log)))
	}
	fmt.Fprintf(&b, "  timeline -> %s\n", path)
	fmt.Print(b.String())
}

func trunc(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}
