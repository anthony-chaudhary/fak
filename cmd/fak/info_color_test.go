package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// TestGuardInfoColorEnabled pins the gate: color only on a real TTY, and never when NO_COLOR is set
// (empty NO_COLOR counts as unset, the community-standard reading).
func TestGuardInfoColorEnabled(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	if !guardInfoColorEnabled(true) {
		t.Fatalf("TTY with NO_COLOR unset must enable color")
	}
	if guardInfoColorEnabled(false) {
		t.Fatalf("non-TTY must never enable color")
	}
	t.Setenv("NO_COLOR", "1")
	if guardInfoColorEnabled(true) {
		t.Fatalf("NO_COLOR set must disable color even on a TTY")
	}
}

// TestColorizeGuardInfoBlockPassThrough proves the byte-clean contract: color=false returns the
// block verbatim, so a piped/redirected pane keeps plain text.
func TestColorizeGuardInfoBlockPassThrough(t *testing.T) {
	block := "── trends ──\n cache  saving money\n safety nothing blocked"
	if got := colorizeGuardInfoBlock(block, false); got != block {
		t.Fatalf("color=false must return the block verbatim, got %q", got)
	}
	if got := colorizeGuardInfoBlock("", true); got != "" {
		t.Fatalf("empty block must stay empty, got %q", got)
	}
}

// TestColorizeGuardInfoBlockRoles proves each structural role gets its expected SGR and that every
// colored row is closed with a reset (so a hue can never bleed into the next row).
func TestColorizeGuardInfoBlockRoles(t *testing.T) {
	cases := []struct {
		name string
		row  string
		want string
	}{
		{"section rule", "── trends ─────", tuiSGRCyanBold},
		{"incident", " incident upstream auth/401 x1", tuiSGRRedBold},
		{"safety clean", " safety nothing blocked", tuiSGRGreen},
		{"safety active", " safety blocked 1, fixed 2", tuiSGRYellowBold},
		{"cache saving", " cache  saving money — reused 88% of text", tuiSGRGreen},
		{"cache not yet", " cache  not saving yet — reused 0% of text", ""},
		{"why note", " why    blocked: policy_block x2", tuiSGRDim},
		{"ablation framing", " ablation  turn a mechanism off → tokens you'd lose", tuiSGRDim},
		{"ablation provider", "   provider prompt-cache  ██░░  +65.1k tok", tuiSGRCyanBold},
		{"ablation fak shed", "   fak compaction shed    ██░░  +52.3k tok", tuiSGRGreen},
		{"ablation fak kv", "   fak KV-prefix reuse    ██░░  +25.0k tok", tuiSGRGreen},
		{"ablation fak vdso", "   fak vDSO memo          ·  3 engine calls avoided (not tokens)", tuiSGRGreen},
		{"identity row", "fak v1 · ↑3h · replies 5 · busy 1", ""},
		{"trends row", " save  ▁▂▃  +12,345 tok", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := colorizeGuardInfoBlock(tc.row, true)
			if tc.want == "" {
				if got != tc.row {
					t.Fatalf("row must stay plain, got %q", got)
				}
				return
			}
			if !strings.HasPrefix(got, tc.want) {
				t.Fatalf("row must open with %q, got %q", tc.want, got)
			}
			if !strings.HasSuffix(got, tuiSGRReset) {
				t.Fatalf("colored row must end with a reset, got %q", got)
			}
		})
	}
}

// TestColorizeGuardInfoBlockPreservesDisplay proves color adds ZERO display cells: stripping the
// SGR escapes from the colored block yields the original monochrome block byte-for-byte, so the
// width-capping renderGuardInfoVisualBlock already did stays exact and nothing can wrap the pane.
func TestColorizeGuardInfoBlockPreservesDisplay(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < 6; i++ {
		tr.push(provenVisualVars())
	}
	for _, width := range []int{24, 40, 80, 120} {
		plain := renderGuardInfoVisualBlock(provenVisualVars(), tr, width, 0 /*roomy*/)
		colored := colorizeGuardInfoBlock(plain, true)
		if colored == plain {
			t.Fatalf("w=%d: a proven+active block must gain at least one colored row", width)
		}
		if stripped := stripSGR(colored); stripped != plain {
			t.Fatalf("w=%d: stripping color must recover the plain block\n plain=%q\n strip=%q", width, plain, stripped)
		}
		for _, r := range strings.Split(colored, "\n") {
			if dw := dispWidthTUI(stripSGR(r)); dw > width {
				t.Fatalf("w=%d: colored row exceeds width after strip: %d > %d: %q", width, dw, width, r)
			}
		}
	}
}

// TestRunInfoOverlayVisualColorGated proves color is wired end-to-end through the overlay AND gated:
// a visual-mode TTY emits SGR color escapes in its frames, and the same run under NO_COLOR emits
// none (only the cursor-control escapes the in-place redraw needs).
func TestRunInfoOverlayVisualColorGated(t *testing.T) {
	t.Run("color on a TTY", func(t *testing.T) {
		t.Setenv("NO_COLOR", "")
		c := healthyThenGoneClient(t, 2)
		var stdout, stderr bytes.Buffer
		code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false, true /*tty*/, 80, 8, "visual", "auto")
		if code != 0 {
			t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
		}
		if !sgrRe.MatchString(stdout.String()) {
			t.Fatalf("visual TTY frame must carry SGR color:\n%q", stdout.String())
		}
	})
	t.Run("no color under NO_COLOR", func(t *testing.T) {
		t.Setenv("NO_COLOR", "1")
		c := healthyThenGoneClient(t, 2)
		var stdout, stderr bytes.Buffer
		code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false, true /*tty*/, 80, 8, "visual", "auto")
		if code != 0 {
			t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
		}
		if sgrRe.MatchString(stdout.String()) {
			t.Fatalf("NO_COLOR must suppress every SGR color escape:\n%q", stdout.String())
		}
	})
}

func TestGuardInfoOverlayColorMode(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	// never -> off even on a TTY
	if got := resolveGuardInfoColorMode("never", true); got {
		t.Fatalf("never should be off, got true")
	}
	// always -> on even without a TTY
	if got := resolveGuardInfoColorMode("always", false); !got {
		t.Fatalf("always should be on, got false")
	}
	// auto -> follows TTY
	if got := resolveGuardInfoColorMode("auto", false); got {
		t.Fatalf("auto non-TTY should be off, got true")
	}
	// NO_COLOR wins over always
	t.Setenv("NO_COLOR", "1")
	if got := resolveGuardInfoColorMode("always", true); got {
		t.Fatalf("NO_COLOR should beat always, got true")
	}
}
