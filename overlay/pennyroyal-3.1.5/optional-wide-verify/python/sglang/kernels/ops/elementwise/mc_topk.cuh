// Multi-CTA exact top-k (k = 512) for QSA decode logits ("lever C").
//
// Same contract as sglang fast_topk(score, lengths, k, row_starts=None):
//   row b selects the k largest of score[b, 0:lengths[b]); indices relative to
//   the row start; slots beyond lengths[b] are -1; output order unspecified.
//
// One kernel, grid = (S, rows), BLOCK threads, cooperative launch (all CTAs
// co-resident; S is capped by occupancy). Per row:
//   * nact = min(S, ceil(L / chunk_target)) CTAs are active (derived from L on
//     device, identical in every CTA of the row). Each converts its chunk to
//     order-preserving uint32 keys, cached in registers (IPT per thread; a
//     longer chunk re-reads the tail from memory on every pass).
//   * nact == 1: exact radix select entirely in shared memory (digits 12/12/8).
//   * nact  > 1: round 0 = 12-bit (key bits 31..20) histogram, smem per CTA,
//     flushed into a per-row global histogram; per-row barrier. The last
//     arriver finds the threshold digit d (largest with suffix >= k_rem) and
//     publishes (d, k_rem, mode) together with the barrier generation in one
//     64-bit word (release store; waiters poll with acquire loads). Modes:
//       DIRECT   - threshold bin holds exactly k_rem keys, or last digit round
//                  (all remaining keys equal): every CTA writes its selected
//                  keys straight to the output.
//       COMPACT  - threshold bin count <= CAP (= BLOCK*IPT): CTAs write keys
//                  above the bin to the output and the bin's (key, idx) pairs
//                  to a per-row compact buffer; the last CTA to exit finishes
//                  the selection on the compact buffer locally (registers +
//                  smem radix on the remaining bits).
//       CONTINUE - bin too large (heavy ties / very narrow values): another
//                  distributed round on the next digit (12 bits, then 8).
//     Every path is exact; there is no lossy overflow case.
//   * The last CTA to exit resets the per-row counters; histogram bins are
//     zeroed by the CTA that read them. The workspace is all-zero between
//     calls except the monotonic generation word.
//   * L <= k: indices 0..L-1 then -1 (CTA 0).
// Barrier timeout fallback. Spin waits are bounded (kSpinCap polls). Every
// active CTA of a multi-CTA row ends by adding to the per-row exit word kDone:
// +1 after a normal exit, +(kBail + 1) after leaving a barrier wait on timeout.
// Waiters also give up within 1024 polls once kDone >= kBail (someone bailed).
// The CTA whose add brings the count (low 16 bits) to nact is the last CTA of
// the row alive in this call; every other CTA's workspace/output writes
// happen-before its acq_rel add. If any CTA bailed (kDone >= kBail), it resets
// the row's whole workspace state, bumps the sticky timeout counter
// ws[:, kErrWord], and recomputes the row exactly with the single-CTA path over
// [0, L) (overwriting all k output slots). So a timeout costs time but never
// yields a wrong or out-of-range result, and the next call starts clean.
// DBG=true builds a test-only variant with fault injection (Params::dbg).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {
namespace mc_topk_detail {

constexpr int kNB = 4096;                       // 12-bit digit bins
constexpr int kH0 = 0, kH1 = 4096, kH2 = 8192;  // per-round global histograms (u32 words)
constexpr int kState = 8192 + 256;              // 16 state words
// kErr: sticky count of recovered barrier timeouts (never cleared by the kernel).
// kDone: exit word (count in bits 0..15, kBail per bailed CTA above).
constexpr int kArr = 0, kGen = 2 /* u64, words 2..3 */, kSel = 4, kTie = 5, kErr = 6, kDone = 7;
constexpr int kCompact = kState + 16;  // keys[kCapMax], then idx[kCapMax]
constexpr int kCapMax = 8192;
constexpr int kRowWords = kCompact + 2 * kCapMax;
constexpr int kErrWord = kState + kErr;
constexpr uint32_t kSpinCap = 1u << 22;
constexpr uint32_t kBail = 1u << 16;
constexpr uint32_t kDirect = 0, kCompactMode = 1, kContinue = 2;
constexpr int kTraceSlots = 10;

struct Params {
  const float* __restrict__ input;
  const int32_t* __restrict__ lengths;
  int32_t* __restrict__ indices;
  uint32_t* __restrict__ ws;
  int64_t stride;
  int32_t cols;
  int32_t chunk_target;
  unsigned long long* trace;  // optional [rows, gridDim.x, kTraceSlots] globaltimer stamps (debug)
  // DBG builds only (test fault injection), int64[16] in device memory, read every call:
  //  [0] row to perturb (-1 = off)  [1] CTA to delay  [2] where: 0 = before key load,
  //  1 = before the round-0 arrive, 2 = before the round-1 arrive  [3] delay ns
  //  (0 = until some CTA of the row bailed, max 20 s)  [4] spin cap override (0 = kSpinCap)
  //  [8] CTA the override applies to (-1 = all CTAs of the row)
  //  outputs: [5] max ns spent in a timed-out wait  [6] polls of the last timed-out wait
  //           [7] number of CTA timeouts
  long long* dbg;
};

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t key_of(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ unsigned long long ld_acquire64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_acq_rel(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__ void round_digit(int r, int& shift, uint32_t& dmask) {
  shift = r == 0 ? 20 : (r == 1 ? 8 : 0);
  dmask = r < 2 ? uint32_t(kNB - 1) : 255u;
}

// Histogram add with a fast path for warps whose keys mostly share one bin.
__device__ __forceinline__ void hist_add(uint32_t* h, bool active, uint32_t bin) {
  const int lane = threadIdx.x & 31;
  const uint32_t b0 = __shfl_sync(0xffffffffu, bin, 0);
  const bool a0 = __shfl_sync(0xffffffffu, active, 0);
  const unsigned m = __ballot_sync(0xffffffffu, active && a0 && bin == b0);
  if (lane == 0 && m) atomicAdd(h + b0, static_cast<uint32_t>(__popc(m)));
  if (active && !((m >> lane) & 1u)) atomicAdd(h + bin, 1u);
}

// Block-wide threshold search over hist[0:NB): largest digit d with
// suffix(d) >= k. Writes s_res = {d, k - suffix(d+1), mode}: DIRECT if the bin
// holds exactly the remaining k (or last round), else COMPACT if its count
// <= cap, else CONTINUE. Non-zero bins are zeroed if `zero`.
// Layout: warp w owns bins [w*ROWS*32, (w+1)*ROWS*32), lane l of row i holds
// bin base + i*32 + l (conflict-free smem / coalesced global accesses).
template <int NB, int BLOCK, bool kGlobal>
__device__ __forceinline__ void find_threshold(
    uint32_t* hist, uint32_t k, bool last_round, uint32_t cap, uint32_t* s_scan, uint32_t* s_res, bool zero) {
  constexpr int NW = BLOCK / 32;
  constexpr int ROWS = NB >= BLOCK ? NB / BLOCK : 1;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int base = warp * ROWS * 32;
  uint32_t v[ROWS];
  uint32_t lsum = 0;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int b = base + i * 32 + lane;
    uint32_t x = 0;
    if (b < NB) x = kGlobal ? __ldcg(hist + b) : hist[b];
    v[i] = x;
    lsum += x;
  }
  uint32_t wsum = lsum;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, o);
  if (lane == 0) s_scan[warp] = wsum;
  __syncthreads();
  uint32_t above = 0;
#pragma unroll
  for (int w = 0; w < NW; ++w)
    if (w > warp) above += s_scan[w];
  if (above < k && k <= above + wsum) {  // warp-uniform: the crossing warp
    uint32_t acc = above, acc_star = 0;
    int istar = -1;
#pragma unroll
    for (int i = ROWS - 1; i >= 0; --i) {
      uint32_t rs = v[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) rs += __shfl_xor_sync(0xffffffffu, rs, o);
      if (istar < 0 && acc + rs >= k) {
        istar = i;
        acc_star = acc;
      }
      acc += rs;
    }
    uint32_t xv = 0;
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
      if (i == istar) xv = v[i];
    uint32_t incl = xv;  // inclusive suffix over lanes (higher lane = higher bin)
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_down_sync(0xffffffffu, incl, o);
      if (lane + o < 32) incl += y;
    }
    const uint32_t excl = incl - xv;
    if (acc_star + excl < k && k <= acc_star + incl) {
      const uint32_t krem = k - (acc_star + excl);
      s_res[0] = base + istar * 32 + lane;
      s_res[1] = krem;
      s_res[2] = (xv == krem || last_round) ? kDirect : (xv <= cap ? kCompactMode : kContinue);
    }
  }
  if (zero) {
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      const int b = base + i * 32 + lane;
      if (b < NB && v[i] != 0) hist[b] = 0;
    }
  }
}

// Local (single-CTA) exact selection rounds over the elements visited by
// `for_each`, starting at digit round r. Updates lo/amask/krem in place.
template <int BLOCK, typename ForEach>
__device__ __forceinline__ void local_select(
    ForEach&& for_each, uint32_t* s_hist, uint32_t* s_scan, uint32_t* s_res, uint32_t& lo, uint32_t& amask,
    uint32_t& krem, int r) {
#pragma unroll 1
  for (; r < 3; ++r) {
    int shift;
    uint32_t dmask;
    round_digit(r, shift, dmask);
    for_each([&](uint32_t key, int, bool valid) {
      hist_add(s_hist, valid && (key & amask) == lo, (key >> shift) & dmask);
    });
    __syncthreads();
    if (r < 2)
      find_threshold<kNB, BLOCK, false>(s_hist, krem, false, 0u, s_scan, s_res, true);
    else
      find_threshold<256, BLOCK, false>(s_hist, krem, true, 0u, s_scan, s_res, true);
    __syncthreads();
    lo |= s_res[0] << shift;
    krem = s_res[1];
    amask |= dmask << shift;
    const uint32_t mode = s_res[2];
    __syncthreads();  // s_res / s_scan reused next round
    if (mode == kDirect) break;
  }
}

// Two-pass block collection (one counter reservation per CTA, no per-warp
// atomic round trips). Keys above the active range {key : (key & amask) == lo}
// are always selected -> out[cnt[0]++]. Keys inside it ("ties"/candidates):
//   kCompactOut = false: fill the remaining krem slots, out[n_above + cnt[1]++]
//                        (extra ties beyond krem are dropped: any is valid);
//   kCompactOut = true : appended to the compact buffer (ckeys, cidx)[cnt[1]++].
// `cnt` may be shared or global memory. s_w needs 4 * (BLOCK / 32) words.
template <int kTopK, int BLOCK, bool kCompactOut, typename ForEach>
__device__ __forceinline__ void collect(
    ForEach&& for_each, uint32_t lo, uint32_t amask, uint32_t krem, uint32_t* cnt, int32_t* out, int32_t* ckeys,
    int32_t* cidx, uint32_t* s_w) {
  constexpr int NW = BLOCK / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t hi_incl = lo | ~amask;
  const uint32_t n_above = kTopK - krem;
  const unsigned lt_mask = (1u << lane) - 1u;
  uint32_t nsel = 0, ntie = 0;
  for_each([&](uint32_t key, int, bool valid) {
    const bool is_sel = valid && key > hi_incl;
    const bool is_tie = valid && !is_sel && (key & amask) == lo;
    nsel += __popc(__ballot_sync(0xffffffffu, is_sel));
    ntie += __popc(__ballot_sync(0xffffffffu, is_tie));
  });
  if (lane == 0) {
    s_w[warp] = nsel;
    s_w[NW + warp] = ntie;
  }
  __syncthreads();
  if (tid == 0) {
    uint32_t ts = 0, tt = 0;
    for (int w = 0; w < NW; ++w) {
      const uint32_t a = s_w[w], b = s_w[NW + w];
      s_w[2 * NW + w] = ts;
      s_w[3 * NW + w] = tt;
      ts += a;
      tt += b;
    }
    uint32_t bs = 0, bt = 0;
    if (ts) bs = atomicAdd(cnt, ts);
    if (tt) bt = atomicAdd(cnt + 1, tt);
    for (int w = 0; w < NW; ++w) {
      s_w[2 * NW + w] += bs;
      s_w[3 * NW + w] += bt;
    }
  }
  __syncthreads();
  uint32_t osel = s_w[2 * NW + warp], otie = s_w[3 * NW + warp];
  for_each([&](uint32_t key, int e, bool valid) {
    const bool is_sel = valid && key > hi_incl;
    const bool is_tie = valid && !is_sel && (key & amask) == lo;
    const unsigned ms = __ballot_sync(0xffffffffu, is_sel);
    const unsigned mt = __ballot_sync(0xffffffffu, is_tie);
    if (is_sel) out[osel + __popc(ms & lt_mask)] = e;
    if (is_tie) {
      const uint32_t pos = otie + __popc(mt & lt_mask);
      if constexpr (kCompactOut) {
        ckeys[pos] = static_cast<int32_t>(key);
        cidx[pos] = e;
      } else {
        if (pos < krem) out[n_above + pos] = e;
      }
    }
    osel += __popc(ms);
    otie += __popc(mt);
  });
}

template <int kTopK, int BLOCK, int IPT, bool DBG>
__global__ __launch_bounds__(BLOCK, (BLOCK >= 512 ? 2 : 1)) void mc_topk_kernel(const Params __grid_constant__ p) {
  constexpr uint32_t CAP = BLOCK * IPT;
  static_assert(CAP <= kCapMax, "compact capacity");
  __shared__ uint32_t s_hist[kNB];
  __shared__ uint32_t s_scan[BLOCK / 32];
  __shared__ uint32_t s_res[4];
  __shared__ uint32_t s_cnt[2];
  __shared__ uint32_t s_w[4 * (BLOCK / 32)];
  __shared__ unsigned long long s_word;
  __shared__ int s_flag;

  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  const int cta = blockIdx.x;
  int L = p.lengths[row];
  L = L < 0 ? 0 : (L > p.cols ? p.cols : L);
  int32_t* __restrict__ out = p.indices + static_cast<int64_t>(row) * kTopK;
  auto trace = [&](int slot) {
    if (p.trace != nullptr && tid == 0)
      p.trace[(static_cast<int64_t>(row) * gridDim.x + cta) * kTraceSlots + slot] = globaltimer();
  };

  if (L <= kTopK) {
    if (cta == 0)
      for (int i = tid; i < kTopK; i += BLOCK) out[i] = i < L ? i : -1;
    return;
  }
  int nact = (L + p.chunk_target - 1) / p.chunk_target;
  if (nact > static_cast<int>(gridDim.x)) nact = gridDim.x;
  if (cta >= nact) return;
  trace(0);
  const bool single = nact == 1;
  const int chunk = (L + nact - 1) / nact;
  int start = cta * chunk;
  int end = min(L, start + chunk);
  const float* __restrict__ x = p.input + static_cast<int64_t>(row) * p.stride;
  uint32_t* wrow = p.ws + static_cast<int64_t>(row) * kRowWords;
  uint32_t* st = wrow + kState;
  unsigned long long* gen_word = reinterpret_cast<unsigned long long*>(st + kGen);

  // ---- test-only fault injection (compiled out unless DBG) ----
  [[maybe_unused]] bool dbg_hit = false;
  uint32_t spin_cap = kSpinCap;
  if constexpr (DBG) {
    if (p.dbg != nullptr && p.dbg[0] == row) {
      dbg_hit = true;
      if (p.dbg[4] > 0 && (p.dbg[8] < 0 || p.dbg[8] == cta)) spin_cap = static_cast<uint32_t>(p.dbg[4]);
    }
  }
  [[maybe_unused]] auto dbg_wait = [&](int where) {
    if constexpr (DBG) {
      if (dbg_hit && !single && p.dbg[1] == cta && p.dbg[2] == where) {
        if (tid == 0) {
          const unsigned long long t0 = globaltimer();
          const long long delay = p.dbg[3];
          for (;;) {
            const unsigned long long t = globaltimer() - t0;
            if (delay > 0 ? t >= static_cast<unsigned long long>(delay)
                          : (ld_relaxed(st + kDone) >= kBail || t > 20000000000ull))
              break;
            __nanosleep(2000);
          }
        }
        __syncthreads();
      }
    }
  };
  dbg_wait(0);

  for (int b = tid; b < kNB; b += BLOCK) s_hist[b] = 0;
  if (tid < 2) s_cnt[tid] = 0;
  // Generation before this call's first barrier (cannot advance until this CTA arrives).
  uint32_t g = single ? 0u : static_cast<uint32_t>(ld_relaxed64(gen_word));

  uint32_t keys[IPT];
  auto load_keys = [&]() {
#pragma unroll
    for (int j = 0; j < IPT; ++j) {
      const int e = start + j * BLOCK + tid;
      keys[j] = e < end ? key_of(__ldg(x + e)) : 0u;
    }
  };
  load_keys();
  __syncthreads();
  trace(1);

  auto for_each = [&](auto&& fn) {
#pragma unroll
    for (int j = 0; j < IPT; ++j) {
      const int e = start + j * BLOCK + tid;
      fn(keys[j], e, e < end);
    }
    // Tail beyond the register cache (only when the occupancy cap forces long
    // chunks, or in the timeout recovery): batched loads so several are in flight per thread.
    constexpr int U = 4;
#pragma unroll 1
    for (int base = start + IPT * BLOCK; base < end; base += U * BLOCK) {
      uint32_t kk[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int e = base + u * BLOCK + tid;
        kk[u] = e < end ? key_of(__ldg(x + e)) : 0u;
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int e = base + u * BLOCK + tid;
        fn(kk[u], e, e < end);
      }
    }
  };

  uint32_t lo = 0, amask = 0, krem = kTopK;
  if (single) {  // single-CTA exact selection over [start, end)
    local_select<BLOCK>(for_each, s_hist, s_scan, s_res, lo, amask, krem, 0);
    collect<kTopK, BLOCK, false>(for_each, lo, amask, krem, s_cnt, out, nullptr, nullptr, s_w);
    trace(8);
    return;
  }
  {
    // ---- distributed rounds ----
    uint32_t mode = kContinue;
    bool bailed = false;
    int r = 0;
#pragma unroll 1
    for (; r < 3 && mode == kContinue; ++r) {
      int shift;
      uint32_t dmask;
      round_digit(r, shift, dmask);
      const int nb = r < 2 ? kNB : 256;
      uint32_t* ghist = wrow + (r == 0 ? kH0 : (r == 1 ? kH1 : kH2));
      for_each([&](uint32_t key, int, bool valid) {
        hist_add(s_hist, valid && (key & amask) == lo, (key >> shift) & dmask);
      });
      __syncthreads();
      for (int b = tid; b < nb; b += BLOCK) {
        const uint32_t v = s_hist[b];
        if (v) {
          atomicAdd(ghist + b, v);
          s_hist[b] = 0;
        }
      }
      __syncthreads();
      trace(2 + 2 * r);
      dbg_wait(1 + r);
      if (tid == 0) {
        const uint32_t a = atom_add_acq_rel(st + kArr, 1u);
        s_flag = (a == static_cast<uint32_t>(nact - 1)) ? 1 : 0;
      }
      __syncthreads();
      uint32_t d, mode_new, krem_new;
      if (s_flag == 1) {
        if (r < 2)
          find_threshold<kNB, BLOCK, true>(ghist, krem, false, CAP, s_scan, s_res, false);
        else
          find_threshold<256, BLOCK, true>(ghist, krem, true, CAP, s_scan, s_res, false);
        __syncthreads();
        d = s_res[0];
        krem_new = s_res[1];
        mode_new = s_res[2];
        if (tid == 0) {
          st_relaxed(st + kArr, 0u);
          const unsigned long long info = d | (krem_new << 12) | (mode_new << 22);
          st_release64(gen_word, (info << 32) | static_cast<unsigned long long>(g + 1));
        }
        // Zero the bins just read (after the release; this round's buffer is not reused in this call).
        for (int b = tid; b < nb; b += BLOCK) ghist[b] = 0;
      } else {
        if (tid == 0) {
          uint32_t it = 0;
          unsigned long long w;
          int err = 0;
          [[maybe_unused]] unsigned long long t_w0 = 0;
          if constexpr (DBG) t_w0 = globaltimer();
          while (static_cast<uint32_t>(w = ld_acquire64(gen_word)) == g) {
            if (++it > spin_cap) {  // timeout: this CTA bails
              err = 1;
              if constexpr (DBG) {
                if (p.dbg != nullptr) {
                  atomicMax(reinterpret_cast<unsigned long long*>(p.dbg + 5), globaltimer() - t_w0);
                  p.dbg[6] = it;
                  atomicAdd(reinterpret_cast<unsigned long long*>(p.dbg + 7), 1ull);
                }
              }
              break;
            }
            if ((it & 1023u) == 0 && ld_relaxed(st + kDone) >= kBail) {  // another CTA of the row bailed
              err = 1;
              break;
            }
          }
          s_word = w;
          s_flag = err ? -1 : 0;
        }
        __syncthreads();
        if (s_flag < 0) {  // CTA-uniform; s_flag is next written after the exit barrier below
          bailed = true;
          break;
        }
        const uint32_t info = static_cast<uint32_t>(s_word >> 32);
        d = info & 0xFFFu;
        krem_new = (info >> 12) & 0x3FFu;
        mode_new = (info >> 22) & 0x3u;
      }
      __syncthreads();  // s_res / s_word / s_flag reused
      ++g;
      lo |= d << shift;
      krem = krem_new;
      amask |= dmask << shift;
      mode = mode_new;
      trace(3 + 2 * r);
    }

    int32_t* ckeys = reinterpret_cast<int32_t*>(wrow + kCompact);
    int32_t* cidx = ckeys + kCapMax;
    if (!bailed) {
      if (mode == kDirect)
        collect<kTopK, BLOCK, false>(for_each, lo, amask, krem, st + kSel, out, nullptr, nullptr, s_w);
      else  // kCompactMode
        collect<kTopK, BLOCK, true>(for_each, lo, amask, krem, st + kSel, out, ckeys, cidx, s_w);
    }

    // ---- exit: the last CTA of the row finishes (COMPACT) or recovers (timeout) and resets ----
    __syncthreads();
    trace(8);
    if (tid == 0) {
      const uint32_t inc = bailed ? kBail + 1u : 1u;
      const uint32_t a = atom_add_acq_rel(st + kDone, inc) + inc;
      s_flag = ((a & (kBail - 1u)) == static_cast<uint32_t>(nact)) ? (a >= kBail ? 2 : 1) : 0;
    }
    __syncthreads();
    const int last = s_flag;
    if (last == 0) return;
    if (last == 1) {
      if (mode == kCompactMode) {
        const int ncand = static_cast<int>(__ldcg(st + kTie));
        uint32_t ck[IPT];
        int ci[IPT];
#pragma unroll
        for (int j = 0; j < IPT; ++j) {
          const int i = j * BLOCK + tid;
          ck[j] = i < ncand ? static_cast<uint32_t>(__ldcg(ckeys + i)) : 0u;
          ci[j] = i < ncand ? __ldcg(cidx + i) : -1;
        }
        auto for_each_cand = [&](auto&& fn) {
#pragma unroll
          for (int j = 0; j < IPT; ++j) fn(ck[j], ci[j], j * BLOCK + tid < ncand);
        };
        if (tid == 0) {
          s_cnt[0] = kTopK - krem;  // keys above the compact range were already written
          s_cnt[1] = 0;
        }
        __syncthreads();
        local_select<BLOCK>(for_each_cand, s_hist, s_scan, s_res, lo, amask, krem, r);
        collect<kTopK, BLOCK, false>(for_each_cand, lo, amask, krem, s_cnt, out, nullptr, nullptr, s_w);
      }
      if (tid == 0) {
        st[kSel] = 0;
        st[kTie] = 0;
        st[kDone] = 0;
      }
      trace(9);
      return;
    }
    // last == 2: some CTA of this row timed out. Every other CTA of the row has finished
    // (its writes happen-before our acquire), so reset the row's whole workspace state and
    // recompute the row exactly on this CTA (single-CTA path over [0, L)).
    for (int b = tid; b < kState; b += BLOCK) wrow[b] = 0;  // histograms H0, H1, H2
    for (int b = tid; b < kNB; b += BLOCK) s_hist[b] = 0;
    if (tid == 0) {
      st[kArr] = 0;
      st[kSel] = 0;
      st[kTie] = 0;
      st[kDone] = 0;
      atomicAdd(st + kErr, 1u);  // sticky diagnostic: recovered timeouts on this row
      s_cnt[0] = 0;
      s_cnt[1] = 0;
    }
    start = 0;
    end = L;
    load_keys();
    lo = 0;
    amask = 0;
    krem = kTopK;
    __syncthreads();
    local_select<BLOCK>(for_each, s_hist, s_scan, s_res, lo, amask, krem, 0);
    collect<kTopK, BLOCK, false>(for_each, lo, amask, krem, s_cnt, out, nullptr, nullptr, s_w);
    trace(9);
  }
}

namespace {
int g_coop_cap = 0;
}  // namespace

}  // namespace mc_topk_detail

template <int kTopK, int BLOCK, int IPT, bool DBG = false>
struct McTopKKernel {
  static constexpr auto kernel = mc_topk_detail::mc_topk_kernel<kTopK, BLOCK, IPT, DBG>;

  // workspace: int32 [R >= B, kRowWords], zero-initialised once by the caller.
  // trace: int64 tensor; used only if numel >= rows * S * kTraceSlots (debug).
  // dbg: int64 tensor; used only in DBG builds and if numel >= 16 (fault injection, see Params).
  // chunk_target <= 0 -> BLOCK * IPT. max_ctas_per_row <= 0 -> auto.
  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView workspace,
      const tvm::ffi::TensorView trace,
      const tvm::ffi::TensorView dbg,
      int64_t chunk_target,
      int64_t max_ctas_per_row,
      int64_t cooperative) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto Lc = SymbolicSize{"cols"};
    auto S = SymbolicSize{"input_stride"};
    auto R = SymbolicSize{"ws_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, Lc}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(score);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({R, mc_topk_detail::kRowWords}).with_dtype<int32_t>().with_device(device).verify(workspace);
    const int64_t rows = B.unwrap();
    RuntimeCheck(R.unwrap() >= rows, "mc_topk: workspace has too few rows");
    if (rows == 0) return;

    // Max co-resident CTAs. Internal linkage on purpose: a function-local static in this
    // class template would be a vague-linkage symbol shared by every JIT .so that
    // instantiates the same template (different kernels -> wrong cap).
    int& s_cap = mc_topk_detail::g_coop_cap;
    if (s_cap == 0) {
      int dev = 0, sms = 0, occ = 0;
      RuntimeDeviceCheck(cudaGetDevice(&dev));
      RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
      RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, BLOCK, 0));
      s_cap = sms * occ;
    }
    const int64_t ct = chunk_target > 0 ? chunk_target : int64_t(BLOCK) * IPT;
    const int64_t cols = Lc.unwrap();
    int64_t per_row = (cols + ct - 1) / ct;
    int64_t cap_row = s_cap / rows;
    if (max_ctas_per_row > 0 && cap_row > max_ctas_per_row) cap_row = max_ctas_per_row;
    if (per_row > cap_row) per_row = cap_row;
    if (per_row < 1) per_row = 1;

    auto params = mc_topk_detail::Params{
        .input = static_cast<const float*>(score.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .ws = static_cast<uint32_t*>(workspace.data_ptr()),
        .stride = S.unwrap(),
        .cols = static_cast<int32_t>(cols),
        .chunk_target = static_cast<int32_t>(ct),
        .trace = nullptr,
        .dbg = nullptr,
    };
    {
      int64_t n = 1;
      for (int i = 0; i < trace.ndim(); ++i) n *= trace.size(i);
      if (n >= rows * per_row * mc_topk_detail::kTraceSlots)
        params.trace = static_cast<unsigned long long*>(trace.data_ptr());
    }
    if constexpr (DBG) {
      int64_t n = 1;
      for (int i = 0; i < dbg.ndim(); ++i) n *= dbg.size(i);
      if (n >= 16) params.dbg = static_cast<long long*>(dbg.data_ptr());
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(per_row), static_cast<unsigned>(rows));
    cfg.blockDim = dim3(BLOCK);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = LaunchKernel::resolve_device(device.unwrap());
    cudaLaunchAttribute attr[1];
    if (cooperative && per_row > 1) {
      attr[0].id = cudaLaunchAttributeCooperative;
      attr[0].val.cooperative = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
    }
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&cfg, kernel, params));
  }
};

}  // namespace sglang
