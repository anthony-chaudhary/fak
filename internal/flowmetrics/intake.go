package flowmetrics

import (
	"fmt"
	"io"
	"math"
	"time"
)

// RenderArrivalServiceReadout writes the human-readable arrival vs service
// readout for a Report (#6199, parent epic #6194):
//  1. The per-day series over the report window (at least 14 days), showing
//     arrivals (opened), started, closes (closed), and keeping backlog and
//     in_flight as separate, distinctly labelled quantities.
//  2. Trailing arrival rate and close rate per day side by side over three
//     windows (7d, 30d, 60d) with the net (arrivals - closes) and ratio.
//  3. The measured service rate as a declared intake cap (arrivals per day equal
//     to trailing close rate), together with current overshoot.
func RenderArrivalServiceReadout(w io.Writer, rep Report, now time.Time) {
	if len(rep.Curve) == 0 && len(rep.ArrivalWindows) == 0 {
		fmt.Fprintf(w, "arrival vs service: no data in window\n")
		return
	}

	if len(rep.Curve) > 0 {
		fmt.Fprintf(w, "arrival vs service (daily series — backlog vs in-flight):\n")
		fmt.Fprintf(w, "  %-10s  %17s  %7s  %15s  %7s  %21s\n",
			"date", "arrivals (opened)", "started", "closes (closed)", "backlog", "in_flight (in-flight)")
		for _, r := range rep.Curve {
			fmt.Fprintf(w, "  %-10s  %17d  %7d  %15d  %7d  %21d\n",
				r.Date, r.Opened, r.Started, r.Closed, r.Backlog, r.InFlight)
		}
	}

	windows := arrivalWindowsFromReport(rep)
	if len(windows) > 0 {
		fmt.Fprintf(w, "\nrates over trailing windows:\n")
		fmt.Fprintf(w, "  %-8s  %16s  %14s  %6s  %8s  %6s  %7s\n",
			"window", "arrival rate/day", "close rate/day", "net", "arrivals", "closes", "ratio")
		for _, win := range windows {
			ratioStr := "-"
			if win.Ratio != nil {
				ratioStr = fmt.Sprintf("%.2f", *win.Ratio)
			}
			fmt.Fprintf(w, "  %-8s  %16.1f  %14.1f  %+6d  %8d  %6d  %7s\n",
				fmt.Sprintf("%.0fd", win.WindowDays),
				win.ArrivalRate,
				win.ServiceRate,
				win.Opened-win.Closed,
				win.Opened,
				win.Closed,
				ratioStr,
			)
		}

		// Use 30d window for declared intake cap if present, otherwise the last window.
		var targetWin ArrivalServiceWindow
		found := false
		for _, win := range windows {
			if int(win.WindowDays) == 30 {
				targetWin = win
				found = true
				break
			}
		}
		if !found && len(windows) > 0 {
			targetWin = windows[len(windows)-1]
		}

		capRate := targetWin.ServiceRate
		arrRate := targetWin.ArrivalRate
		overshoot := arrRate - capRate
		if math.Abs(overshoot) < 1e-9 {
			overshoot = 0
		}

		fmt.Fprintf(w, "\ndeclared intake cap: %.1f arrivals/day (arrivals-per-day equal to trailing close rate)\n", capRate)
		fmt.Fprintf(w, "current overshoot: %+0.1f arrivals/day (arrivals %.1f/day - close_rate %.1f/day)\n",
			overshoot, arrRate, capRate)
	}
}

// RenderIntakeReport is an alias for RenderArrivalServiceReadout.
func RenderIntakeReport(w io.Writer, rep Report, now time.Time) {
	RenderArrivalServiceReadout(w, rep, now)
}

func arrivalWindowsFromReport(rep Report) []ArrivalServiceWindow {
	if len(rep.ArrivalWindows) > 0 {
		return rep.ArrivalWindows
	}
	if len(rep.Curve) == 0 {
		return nil
	}
	windows := []int{7, 30, 60}
	var out []ArrivalServiceWindow
	n := len(rep.Curve)
	for _, w := range windows {
		startIdx := n - w
		if startIdx < 0 {
			startIdx = 0
		}
		days := float64(w)
		var opened, closed int
		for i := startIdx; i < n; i++ {
			opened += rep.Curve[i].Opened
			closed += rep.Curve[i].Closed
		}
		win := ArrivalServiceWindow{
			Opened:      opened,
			Closed:      closed,
			ArrivalRate: float64(opened) / days,
			ServiceRate: float64(closed) / days,
			WindowDays:  days,
		}
		if closed > 0 {
			ratio := float64(opened) / float64(closed)
			win.Ratio = &ratio
		}
		out = append(out, win)
	}
	return out
}
