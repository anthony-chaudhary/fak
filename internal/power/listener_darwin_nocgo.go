//go:build darwin && !cgo

package power

func newDarwinIOKitListener(b *PowerBroadcaster) SleepListener {
	return nil
}
