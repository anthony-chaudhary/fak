// Package sysproc provides cross-platform subprocess attribute configuration
// and command constructors with Windows console window suppression for headless
// and background automation.
//
// On Windows, child console processes spawned by windowless parents allocate a
// transient visible console window (hosted by conhost.exe / OpenConsole.exe)
// unless suppressed with CREATE_NO_WINDOW (0x08000000) and HideWindow = true,
// or completely detached via DETACHED_PROCESS (0x00000008). sysproc exposes
// standardized constructors and configuration helpers to safely manage process
// creation flags and attributes across Windows and POSIX environments.
package sysproc
