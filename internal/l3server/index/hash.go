package index

// Wyhash implementation - fast non-cryptographic hash
// Based on wyhash v4 by Wang Yi

const (
	wyp0 = 0xa0761d6478bd642f
	wyp1 = 0xe7037ed1a0b428db
	wyp2 = 0x8ebc6af09c88c6e3
	wyp3 = 0x589965cc75374cc3
	wyp4 = 0x1d8e4e27c47d124f
)

func wyMix(a, b uint64) uint64 {
	hi, lo := mul64(a, b)
	return hi ^ lo
}

func mul64(a, b uint64) (hi, lo uint64) {
	// Full 128-bit multiplication
	a0 := a & 0xFFFFFFFF
	a1 := a >> 32
	b0 := b & 0xFFFFFFFF
	b1 := b >> 32
	w0 := a0 * b0
	t := a1*b0 + (w0 >> 32)
	w1 := t & 0xFFFFFFFF
	w2 := t >> 32
	w1 += a0 * b1
	hi = a1*b1 + w2 + (w1 >> 32)
	lo = a * b
	return
}

func wyRead8(b []byte) uint64 {
	return uint64(b[0])<<56 | uint64(b[1])<<48 | uint64(b[2])<<40 | uint64(b[3])<<32 |
		uint64(b[4])<<24 | uint64(b[5])<<16 | uint64(b[6])<<8 | uint64(b[7])
}

func wyRead4(b []byte) uint64 {
	return uint64(b[0])<<24 | uint64(b[1])<<16 | uint64(b[2])<<8 | uint64(b[3])
}

func wyRead3(b []byte, k int) uint64 {
	return (uint64(b[0]) << 16) | (uint64(b[k>>1]) << 8) | uint64(b[k-1])
}

// Hash computes a wyhash of the given key with seed.
func Hash(key []byte, seed uint64) uint64 {
	length := len(key)
	seed ^= wyp0
	var a, b uint64

	switch {
	case length <= 0:
		return seed
	case length <= 3:
		a = wyRead3(key, length)
		b = 0
	case length <= 8:
		a = wyRead4(key[:4])
		b = wyRead4(key[length-4:])
	case length <= 16:
		a = wyRead8(key[:8])
		b = wyRead8(key[length-8:])
	case length <= 32:
		a = wyRead8(key[:8]) ^ wyRead8(key[8:16])
		b = wyRead8(key[length-16:length-8]) ^ wyRead8(key[length-8:])
	case length <= 64:
		a = wyRead8(key[:8]) ^ wyRead8(key[8:16])
		b = wyRead8(key[16:24]) ^ wyRead8(key[24:32])
		a ^= wyRead8(key[length-32:length-24]) ^ wyRead8(key[length-24:length-16])
		b ^= wyRead8(key[length-16:length-8]) ^ wyRead8(key[length-8:])
	default:
		seed1 := seed
		seed2 := seed
		i := 0
		for length-i > 48 {
			seed = wyMix(wyRead8(key[i:])^wyp1, wyRead8(key[i+8:])^seed)
			seed1 = wyMix(wyRead8(key[i+16:])^wyp2, wyRead8(key[i+24:])^seed1)
			seed2 = wyMix(wyRead8(key[i+32:])^wyp3, wyRead8(key[i+40:])^seed2)
			i += 48
		}
		seed ^= seed1 ^ seed2
		for length-i > 16 {
			seed = wyMix(wyRead8(key[i:])^wyp1, wyRead8(key[i+8:])^seed)
			i += 16
		}
		a = wyRead8(key[length-16 : length-8])
		b = wyRead8(key[length-8:])
	}

	return wyMix(wyp1^uint64(length), wyMix(a^wyp1, b^seed))
}

// KeyHash returns a 64-bit hash for the given key.
func KeyHash(key []byte) uint64 {
	return Hash(key, 0)
}

// H1 returns the upper 57 bits used for slot indexing.
func H1(h uint64) uint64 {
	return h >> 7
}

// H2 returns the lower 7 bits used as the ctrl fingerprint (0x00-0x7F).
func H2(h uint64) uint8 {
	return uint8(h & 0x7F)
}
