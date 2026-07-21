// Package stallpage turns stallscan's reboot high-water decision into a
// durable, deduped operator page.
//
// It composes the pure stallscan decision with choicetriage: a reboot drops all
// live sessions, so the resulting choice explicitly names operator approval and
// must earn HUMAN_RESIDUAL. The publisher serializes concurrent monitor
// processes and emits at most one page per (axis, process) in a bounded window;
// it never kills a process or reboots the host.
//
// Tier: composer (3) - see internal/architest. This package may import only
// packages whose tier is <= 3; an upward import fails the architest gate.
package stallpage
