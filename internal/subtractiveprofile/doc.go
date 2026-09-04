// Package subtractiveprofile provides deterministic capability subtraction
// and sticky removal resolution for agent profiles.
//
// Invariant: subtractive profile resolution is fail-closed and sticky.
// Guard: once a capability is marked removed, subsequent profile layers,
// alias remappings, or replacement directives cannot resurrect it on any surface.
// Guard: all dependency prerequisites must resolve against active capabilities;
// references to removed or absent capabilities fail closed with an explicit error.
package subtractiveprofile
