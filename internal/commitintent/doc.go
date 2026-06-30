// Package commitintent defines the durable, pure queue record that sits before
// an effectful fak commit drain.
//
// The package deliberately does not run git, inspect the live index, take locks,
// or push. It validates and orders submit records so a later single-writer drain
// can consume one deterministic stream.
package commitintent
