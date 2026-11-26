package model

// kimik31_kda_chunked.go — the chunk-level reformulation of the K3.1 KDA
// prefill recurrence (ticket #14371 spine). The token-by-token oracle is
// ForwardKimiK31KDAPrefillSeq / kimiK31KDAHeadStep (kimik31_kda.go); this file
// reorders the SAME recurrence into chunks so the recurrent state matrix S is
// touched once per chunk (the inter-chunk carry) instead of once per token,
// with per-token contributions expressed through within-chunk cumulative
// decay factors — the standard linear-attention chunk trick (FLA's chunked
// Gated-DeltaNet; MoonshotAI/FlashKDA is the CUTLASS SM90+ port of the same
// form, see the cuda-kda-chunked-prefill SOTA row).

import (
	"fmt"
	"math"
)

// CHUNK REFORMULATION (per head h, key dim i, value channel c; D_r is the
// diagonal decay of chunk token r with alpha_{r,c} = D_r[c] in (0,1]):
//
//	scan step:  S_r = S_{r-1} D_r + k_r (beta_r (v_r - D_r S_{r-1}^T k_r))^T,
//	            readout o_r = S_r^T q_r.
//
// Unrolling a chunk of C tokens (chunk-relative r = 0..C-1, S_in the state
// entering the chunk) gives, with the cumulative decay
// cum[r][c] = Π_{u<=r} alpha_{u,c} (A_r) and w[c] = A_{C-1}:
//
//	delta_r = beta_r (v_r - A_r S_in^T k_r
//	                     - Σ_{s<r} (A_r/A_s) delta_s (k_s·k_r))
//	o_r     = A_r S_in^T q_r + Σ_{s<r} (A_r/A_s) delta_s (k_s·q_r)
//	            + delta_r (k_r·q_r)
//	S_out   = S_in W + Σ_r k_r delta_r^T T_r,  T_r = Π_{u=r+1}^{C-1} D_u
//
// The decay ratios A_r/A_s and the trailing factors T_r are NEVER evaluated
// by division: every ratio is multiplied out token-by-token as a running
// product of the stored alphas, so a decay that underflows to zero (possible
// in the softplus gate form) stays a benign zero factor instead of a 0/0.
// Everything else — per-token projections, the causal conv (whose history
// spans chunks and stays sequential), both decay gate forms, the FLA L2 norm
// + 1/sqrt(d) query scale, the gated RMSNorm readout, and o_proj — is the
// oracle's own math through the oracle's own helpers; only the S recurrence
// is reordered.
//
// CHUNK INVARIANT: the state carried across chunks is exactly the state the
// scan would hold after the same tokens — S_out equals the unrolled scan's S
// in exact arithmetic, because the scan applies each D_r, predict, delta, and
// rank-1 update in the same order the unrolled identity distributes them.
// In float32 the factorization re-associates the sums (the decay moves from
// the state entries into scalar factors), so size>=2 chunks match the scan
// only within the measured tolerance stated in kimik31_kda_chunked_test.go;
// a chunk that happens to hold a single token (chunkSize==1, or an odd
// sequence tail) delegates verbatim to ForwardKimiK31KDAStep, which makes
// chunkSize==1 bit-exact and keeps odd tails at the scan's own rounding.
func ForwardKimiK31KDAChunkedPrefill(st *GLM5NextKDALayerState, p *KimiK31KDAParams, xSeq []float32, T int, chunkSize int) []float32 {
	if st == nil || p == nil {
		panic("model: kimik31 kda chunked prefill: nil state or params")
	}
	if chunkSize < 1 {
		panic(fmt.Sprintf("model: kimik31 kda chunked prefill: chunkSize %d must be >= 1", chunkSize))
	}
	if len(xSeq) != T*p.HiddenSize {
		panic(fmt.Sprintf("model: kimik31 kda chunked prefill: xSeq length %d != T*HiddenSize %d", len(xSeq), T*p.HiddenSize))
	}
	if st.NumHeads != p.NumHeads || st.HeadDim != p.HeadDim || st.ConvWindow != p.ConvKernel {
		panic(fmt.Sprintf("model: kimik31 kda chunked prefill: state geometry (%d,%d,%d) != params (%d,%d,%d)",
			st.NumHeads, st.HeadDim, st.ConvWindow, p.NumHeads, p.HeadDim, p.ConvKernel))
	}

	hid := p.HiddenSize
	out := make([]float32, T*hid)
	if T == 0 {
		return out
	}

	// Degenerate chunk size: the token scan itself, verbatim.
	if chunkSize == 1 {
		for t := 0; t < T; t++ {
			copy(out[t*hid:(t+1)*hid], ForwardKimiK31KDAStep(st, p, xSeq[t*hid:(t+1)*hid]))
		}
		return out
	}

	d := p.HeadDim
	nh := p.NumHeads
	dInner := nh * d

	// Chunk scratch, allocated once and reused per chunk and per head. The
	// [cl*d] buffers are head-local; the [cl*dInner] rows are per token.
	c := chunkSize
	alpha := make([]float32, c*d) // [r*d+cc] decay per (token r, value channel cc)
	cum := make([]float32, c*d)   // [r*d+cc] cumulative decay A_r
	uIn := make([]float32, c*d)   // [r*d+cc] S_in^T k_r
	pIn := make([]float32, c*d)   // [r*d+cc] S_in^T q_r
	delta := make([]float32, c*d) // [r*d+cc] chunk-local delta terms
	kn := make([]float32, c*d)    // [r*d+i]  L2-normalized k_r (head-local)
	qn := make([]float32, c*d)    // [r*d+i]  L2-normalized + 1/sqrt(d)-scaled q_r
	vv := make([]float32, c*d)    // [r*d+cc] conv'd v_r (head-local)
	qcAll := make([]float32, c*dInner)
	kcAll := make([]float32, c*dInner)
	vcAll := make([]float32, c*dInner)
	kdot := make([]float32, c*c)  // [s*cl+r] k_s·k_r
	kqdot := make([]float32, c*c) // [s*cl+r] k_s·q_r
	betaRaw := make([]float32, c*nh)
	fRawAll := make([]float32, c*dInner)
	gateRawAll := make([]float32, c*dInner)
	outRows := make([]float32, c*dInner)
	acc := make([]float32, d)
	oo := make([]float32, d)
	run := make([]float32, d)
	trail := make([]float32, d)
	w := make([]float32, d)
	dtilde := make([]float32, d)
	qScale := float32(1.0 / math.Sqrt(float64(d)))

	for start := 0; start < T; start += c {
		cl := c
		if start+cl > T {
			cl = T - start
		}

		// Pass A — per-token projections and conv: the oracle step's lines
		// 1-3 verbatim (same helpers, same order), stored for the chunk
		// solve. The conv history lives in st and stays sequential across
		// chunks; nothing here depends on the chunk boundary.
		for r := 0; r < cl; r++ {
			x := xSeq[(start+r)*hid : (start+r+1)*hid]
			qRaw := kdaMatVecMul(p.Wq, x, dInner, hid)
			kRaw := kdaMatVecMul(p.Wk, x, dInner, hid)
			vRaw := kdaMatVecMul(p.Wv, x, dInner, hid)
			fa := kdaMatVecMul(p.WFA, x, d, hid)
			fRaw := kdaMatVecMul(p.WFB, fa, dInner, d)
			for i := range fRaw {
				fRaw[i] += p.DTBias[i]
			}
			copy(fRawAll[r*dInner:(r+1)*dInner], fRaw)
			copy(betaRaw[r*nh:(r+1)*nh], kdaMatVecMul(p.WBeta, x, nh, hid))
			copy(gateRawAll[r*dInner:(r+1)*dInner], kdaMatVecMul(p.WGate, x, dInner, hid))
			copy(qcAll[r*dInner:(r+1)*dInner], p.ConvQ.Step(qRaw, st.ConvQ))
			copy(kcAll[r*dInner:(r+1)*dInner], p.ConvK.Step(kRaw, st.ConvK))
			copy(vcAll[r*dInner:(r+1)*dInner], p.ConvV.Step(vRaw, st.ConvV))
		}

		// Per-head chunk solve: normalized q/k, cumulative decays, the S_in
		// interaction, the intra-chunk delta recurrence, the readouts, and
		// the inter-chunk state carry — the reformulated math above.
		for h := 0; h < nh; h++ {
			base := h * d

			// Oracle step 4 verbatim, per token: L2-normalize q/k per head,
			// then the 1/sqrt(d) query scale.
			for r := 0; r < cl; r++ {
				l2normInto(qn[r*d:(r+1)*d], qcAll[r*dInner+base:r*dInner+base+d], p.RMSEps)
				l2normInto(kn[r*d:(r+1)*d], kcAll[r*dInner+base:r*dInner+base+d], p.RMSEps)
				for i := 0; i < d; i++ {
					qn[r*d+i] *= qScale
				}
				copy(vv[r*d:(r+1)*d], vcAll[r*dInner+base:r*dInner+base+d])
			}

			// Decay gates -> alphas (oracle step 3, both forms verbatim).
			for r := 0; r < cl; r++ {
				fRow := fRawAll[r*dInner+base : r*dInner+base+d]
				aRow := alpha[r*d : (r+1)*d]
				if p.GateLowerBound != 0 {
					lb := p.GateLowerBound
					expAlog := -p.AFolded[h]
					for cc := 0; cc < d; cc++ {
						aRow[cc] = float32(math.Exp(float64(lb * sigmoidf(expAlog*fRow[cc]))))
					}
				} else {
					aFold := p.AFolded[h]
					for cc := 0; cc < d; cc++ {
						aRow[cc] = float32(math.Exp(float64(aFold * softplus(fRow[cc]))))
					}
				}
			}

			// Within-chunk cumulative decay A_r per value channel.
			copy(cum[0:d], alpha[0:d])
			for r := 1; r < cl; r++ {
				aRow := alpha[r*d : (r+1)*d]
				cPrev := cum[(r-1)*d : r*d]
				cRow := cum[r*d : (r+1)*d]
				for cc := 0; cc < d; cc++ {
					cRow[cc] = cPrev[cc] * aRow[cc]
				}
			}

			S := st.HeadMatrix(h)

			// S_in interaction: uIn[r] = S_in^T k_r, pIn[r] = S_in^T q_r.
			// The sums mirror the oracle's kv/readout sums (ascending i,
			// zero keys/queries skipped) so the S_in term is as close to the
			// scan's own association as the factorization allows.
			for r := 0; r < cl; r++ {
				kRow := kn[r*d : (r+1)*d]
				qRow := qn[r*d : (r+1)*d]
				uRow := uIn[r*d : (r+1)*d]
				pRow := pIn[r*d : (r+1)*d]
				for cc := 0; cc < d; cc++ {
					uRow[cc] = 0
					pRow[cc] = 0
				}
				for i := 0; i < d; i++ {
					ki := kRow[i]
					qi := qRow[i]
					if ki == 0 && qi == 0 {
						continue
					}
					row := i * d
					if ki != 0 {
						for cc := 0; cc < d; cc++ {
							uRow[cc] += S[row+cc] * ki
						}
					}
					if qi != 0 {
						for cc := 0; cc < d; cc++ {
							pRow[cc] += S[row+cc] * qi
						}
					}
				}
			}

			// Intra-chunk attention-like terms: k_s·k_r and k_s·q_r.
			for s := 0; s < cl; s++ {
				kS := kn[s*d : (s+1)*d]
				for r := s; r < cl; r++ {
					kR := kn[r*d : (r+1)*d]
					qR := qn[r*d : (r+1)*d]
					var kd, kq float32
					for i := 0; i < d; i++ {
						kd += kS[i] * kR[i]
						kq += kS[i] * qR[i]
					}
					kdot[s*cl+r] = kd
					kqdot[s*cl+r] = kq
				}
			}

			// Delta solve + readout accumulation per token (the fused
			// lower-triangular pass; run[c] is A_r/A_s built by product).
			for r := 0; r < cl; r++ {
				cRow := cum[r*d : (r+1)*d]
				uRow := uIn[r*d : (r+1)*d]
				pRow := pIn[r*d : (r+1)*d]
				for cc := 0; cc < d; cc++ {
					acc[cc] = cRow[cc] * uRow[cc]
					oo[cc] = cRow[cc] * pRow[cc]
					run[cc] = 1
				}
				for s := r - 1; s >= 0; s-- {
					aRow := alpha[(s+1)*d : (s+2)*d]
					kd := kdot[s*cl+r]
					kq := kqdot[s*cl+r]
					for cc := 0; cc < d; cc++ {
						run[cc] *= aRow[cc]
						dec := run[cc] * delta[s*d+cc]
						acc[cc] += dec * kd
						oo[cc] += dec * kq
					}
				}
				b := sigmoidf(betaRaw[r*nh+h])
				vRow := vv[r*d : (r+1)*d]
				dRow := delta[r*d : (r+1)*d]
				for cc := 0; cc < d; cc++ {
					dRow[cc] = b * (vRow[cc] - acc[cc])
				}
				kqr := kqdot[r*cl+r]
				oRow := outRows[r*dInner+base : r*dInner+base+d]
				for cc := 0; cc < d; cc++ {
					oRow[cc] = oo[cc] + dRow[cc]*kqr
				}
			}

			// Inter-chunk carry, once per chunk: S_out = S_in W + Σ_r k_r δ̃_r
			// with the trailing decay T_r folded into δ̃_r (product, no
			// division).
			copy(w, cum[(cl-1)*d:cl*d])
			for i := 0; i < d; i++ {
				row := i * d
				for cc := 0; cc < d; cc++ {
					S[row+cc] *= w[cc]
				}
			}
			for cc := 0; cc < d; cc++ {
				trail[cc] = 1
			}
			for r := cl - 1; r >= 0; r-- {
				dRow := delta[r*d : (r+1)*d]
				for cc := 0; cc < d; cc++ {
					dtilde[cc] = dRow[cc] * trail[cc]
				}
				kRow := kn[r*d : (r+1)*d]
				for i := 0; i < d; i++ {
					ki := kRow[i]
					if ki == 0 {
						continue
					}
					row := i * d
					for cc := 0; cc < d; cc++ {
						S[row+cc] += ki * dtilde[cc]
					}
				}
				aRow := alpha[r*d : (r+1)*d]
				for cc := 0; cc < d; cc++ {
					trail[cc] *= aRow[cc]
				}
			}
		}

		// Gated RMSNorm readout + o_proj per token (oracle step 6 verbatim:
		// the RMS statistic is PER HEAD over the head's d readout channels).
		for r := 0; r < cl; r++ {
			oRow := outRows[r*dInner : (r+1)*dInner]
			gRow := gateRawAll[r*dInner : (r+1)*dInner]
			for h := 0; h < nh; h++ {
				base := h * d
				oHead := oRow[base : base+d]
				var ss float32
				for _, v := range oHead {
					ss += v * v
				}
				inv := float32(1.0 / math.Sqrt(float64(ss)/float64(d)+float64(p.RMSEps)))
				for cc := 0; cc < d; cc++ {
					oHead[cc] = oHead[cc] * inv * p.ONorm[cc] * sigmoidf(gRow[base+cc])
				}
			}
			copy(out[(start+r)*hid:(start+r+1)*hid], kdaMatVecMul(p.WO, oRow, hid, dInner))
		}
	}

	return out
}
