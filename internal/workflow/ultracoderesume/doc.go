// Package ultracoderesume persists the small identity and receipt spine needed
// to resume an interrupted Ultracode graph without trusting controller memory.
// It maps orchestration plans into the generic workflow journal, rechecks every
// completed node through an independent witness, and reruns only nodes whose
// prior attempt is safely known to be incomplete or dependency-invalidated.
package ultracoderesume
