// Package sysproc provides cross-platform subprocess attribute configuration
// and command constructors, with Windows console window suppression for
// headless and background automation.
//
// On Windows, a child console process spawned by a windowless parent allocates
// a transient visible console (hosted by conhost.exe / OpenConsole.exe) unless
// suppressed with CREATE_NO_WINDOW plus HideWindow, or fully detached via
// DETACHED_PROCESS.
package sysproc
