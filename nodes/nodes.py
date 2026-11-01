"""MMH3 Ultimate Upscale - one node for the full latent re-enhancement loop.

Pipeline (auto, no graph wiring):
    input AV latent
      -> temporal split (outer loop)
      ->   latent upscale  (H3 3D upscaler, per chunk, video only)
      ->   spatial split   (inner loop)
      ->   per-tile sampling with preview
      ->   spatial stitch
      -> temporal stitch
      -> output AV latent

Helpers (frame/token mapping, re-anchoring, spatial tiling, seam blending,
stitching) are self-contained copies of the logic used by the
Comfyui-MiniMax-H3-LatentSplit project so this plugin has no dependency on it.

Frame/token mapping mirrors comfy.ldm.minimax.model:
  * video latent token k covers FRAME_PER_TOKEN[k % 5] = (1, 4, 4, 4, 4) pixel
    frames (periodic grid, 17 frames per 5 tokens)
  * audio latent frames run at FRAME_RESCALE = 5/3 per pixel frame (40 vs 24 Hz)

The H3 3D upscaler inference code (model classes, loading, normalization stats)
lives in h3_latent_upscaler.py, copied from the
Comfyui_Minimax_h3_latent_Upscaler plugin so it works with the
minimax_h3_latent_upscaler_3d checkpoints directly. The LTX2.5 block lives in
ltx.py.
"""

import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.controlnet
import comfy.ldm.common_dit
import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import latent_preview
from comfy_api.latest import io

from .h3_latent_upscaler import _compute_upscale_target, _scan_models, upscale_video, unload_upscale_model

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

H3_UPSCALE_PARAM = io.Custom("H3_UPSCALE_PARAM")
H3_TEMPORAL_PARAM = io.Custom("H3_TEMPORAL_PARAM")
H3_SPATIAL_PARAM = io.Custom("H3_SPATIAL_PARAM")
H3_FUN_CONTROL_PARAM = io.Custom("H3_FUN_CONTROL_PARAM")
H3_INPAINT_PARAM = io.Custom("H3_INPAINT_PARAM")

# Spatial compression factor of the Minimax H3 3D VAE (16x).
VAE_DOWNSAMPLE = 16

# ---------------------------------------------------------------------------
# frame <-> token helpers (copied from Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def frames_for_tokens(n):
    """Pixel frames covered by the first `n` video latent tokens."""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f):
    """Smallest token count whose cumulative frames reach at least `f`."""
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def audio_range(f0, f1):
    """Audio latent token range [a0, a1) for the pixel-frame span [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def compute_segments(tv, chunk_length, overlap):
    """Per-chunk (video_token_start, frame_start, video_token_end, frame_end).

    Same rules as the Split node: every boundary is snapped to a keyframe token
    (index % 5 == 0), the realized overlap is a whole number of 17-frame grid
    steps, and the last chunk always ends on the exact total frame count.
    """
    frame_count = frames_for_tokens(tv)
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_length <= overlap:
        raise ValueError("overlap must be smaller than chunk_length")

    hop = chunk_length - overlap
    bounds = []
    prev_end_k = 0
    i = 0
    while True:
        s = i * hop
        e = min(s + chunk_length, frame_count)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = snap_frame_boundary(s, tv, phase=5)
            if k0 > prev_end_k:
                k0, f0 = prev_end_k, frames_for_tokens(prev_end_k)
        if e >= frame_count:
            k1, f1 = tv, frame_count
        else:
            k1, f1 = snap_frame_boundary(e, tv, phase=5)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_for_tokens(k1)
            if k1 >= tv:
                k1, f1 = tv, frame_count
        bounds.append((k0, f0, k1, f1))
        if k1 >= tv:
            break
        prev_end_k = k1
        i += 1
    return bounds, frame_count


def snap_frame_boundary(f, max_tokens, phase=None):
    """Nearest video-token boundary to pixel frame f (optionally on a phase grid)."""
    step = phase if phase is not None else 1
    best_k, best_f, best_d = 0, 0, f
    for k in range(0, max_tokens + 1, step):
        acc = frames_for_tokens(k)
        d = abs(acc - f)
        if d < best_d:
            best_k, best_f, best_d = k, acc, d
    return best_k, best_f


def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)


# ---------------------------------------------------------------------------
# spatial tiling helpers (copied from Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def _grid_1d(size, tile, ol, min_tile):
    """Tile origins/dims for one axis plus per-seam overlaps.

    If the leftover edge tile would be smaller than min_tile, the last origin is
    pulled left until the edge reaches min_tile; the extra overlap that creates
    is reported per-seam so stitching blends over its full width."""
    if size <= tile:
        return [0], [size], [0]
    sh = tile - ol
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    rows = [i * sh for i in range(n)]
    trows = [min(tile, size - r) for r in rows]
    if min_tile > 0 and n >= 2:
        edge = size - rows[-1]
        if edge < min_tile:
            new_last = size - min_tile
            if rows[-2] < new_last < rows[-2] + trows[-2]:
                rows[-1] = new_last
                trows[-1] = size - new_last
    ovl = [0] * n
    for i in range(1, n):
        ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
    return rows, trows, ovl


def compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_th=0, min_tw=0):
    """Tile a latent of size (h, w) with tiles (th, tw) and overlap (ol_h, ol_w).

    Returns (row_offsets, col_offsets, true_row_dims, true_col_dims,
    row_overlaps, col_overlaps) in latent units. Horizontal and vertical
    overlaps are independent. min_th/min_tw (0 = disabled) force the leftover
    edge tile to at least that size when possible, growing the seam overlap."""
    if th <= 0 or tw <= 0:
        raise ValueError("tile dimensions must be positive")
    if ol_h >= th or ol_w >= tw:
        raise ValueError("overlap must be smaller than the tile size")
    if min_th < 0 or min_tw < 0:
        raise ValueError("minimum tile size must be non-negative")
    if min_th > th or min_tw > tw:
        raise ValueError("minimum tile size must not exceed the tile size")
    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl


def spatial_fade_mask(tile_h, tile_w, ol_h, ol_w, done_top, done_left, fade_h=0, fade_w=0):
    """Per-tile video noise mask [tile_h, tile_w]: 1 = re-sample freely, 0 = frozen.

    Every tile is sampled at its true extent (no padding), so the mask only
    freezes the overlap strips shared with an already-processed neighbor
    (done_top / done_left). Each overlap strip splits into a FROZEN segment on
    the seam side (mask = 0, keeps the neighbour's content) and a FADE segment
    on the interior side (mask rises 0 -> 1 toward the tile interior).
    fade_width/fade_height is the FADE segment length; the frozen segment takes
    the rest of the overlap strip (ol - fade). 0 (default) = whole strip
    frozen. The two axes use independent fade widths."""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)
    if done_left and ol_w > 0:
        if fade_w == 0:
            mask[:, :ol_w] = 0.0
        else:
            f = min(fade_w, ol_w)
            frozen_w = ol_w - f
            w = torch.linspace(0.0, 1.0, f)
            mask[:, :frozen_w] = 0.0
            mask[:, frozen_w:ol_w] = torch.minimum(mask[:, frozen_w:ol_w], w[None, :])
    if done_top and ol_h > 0:
        if fade_h == 0:
            mask[:ol_h, :] = 0.0
        else:
            f = min(fade_h, ol_h)
            frozen_h = ol_h - f
            w = torch.linspace(0.0, 1.0, f)
            mask[:frozen_h, :] = 0.0
            mask[frozen_h:ol_h, :] = torch.minimum(mask[frozen_h:ol_h, :], w[:, None])
    return mask


def _fade_band(band, fade, axis):
    """Fill one overlap band `band` (2D view) with a FROZEN+FADE structure.

    `band` is a slice of the tile mask covering one overlap strip; `axis` is the
    tile axis the overlap runs along (0 = vertical band across rows, 1 =
    horizontal band across columns). `fade` is the fade segment length (<= the
    band length on that axis); the frozen segment takes the rest. The frozen
    (mask=0) segment sits on the seam side, the ramp 0->1 toward the interior.
    Values are written with `minimum` so a cell frozen by another axis's band
    (the corner where the horizontal and vertical overlap strips cross) stays
    frozen instead of being raised by this band's ramp."""
    n = band.shape[axis]
    f = min(int(fade), n)
    if f == 0:
        band[:] = 0.0
        return
    w = torch.linspace(0.0, 1.0, f, dtype=band.dtype, device=band.device)
    frozen = n - f
    if axis == 1:
        w = w[None, :]
        band[:, :frozen] = torch.minimum(band[:, :frozen], torch.zeros(frozen, dtype=band.dtype, device=band.device))
        band[:, frozen:] = torch.minimum(band[:, frozen:], w)
    else:
        w = w[:, None]
        band[:frozen, :] = torch.minimum(band[:frozen, :], torch.zeros((frozen, 1), dtype=band.dtype, device=band.device))
        band[frozen:, :] = torch.minimum(band[frozen:, :], w)


def make_fade_mask(tile_h, tile_w, ol_h, ol_w, done_top, done_left,
                   fade_h=0, fade_w=0):
    """Tile mask with a FADE width that can vary per sampling step.

    Same frozen-seam layout as `spatial_fade_mask`, rebuilt fresh for each fade
    width. Only the overlap bands shared with an already-processed neighbour
    (done_top / done_left) are masked. The bands combine with `minimum`, so the
    crossing corner between the two overlap strips keeps whichever axis is more
    conservative (more frozen) at each cell."""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)
    if done_left and ol_w > 0:
        _fade_band(mask[:, :ol_w], fade_w, 1)
    if done_top and ol_h > 0:
        _fade_band(mask[:ol_h, :], fade_h, 0)
    return mask


def _dynamic_fade_closure(sp, fw, fh, tr, tc, tr_s, tc_s, ovh, ovw, done_top, done_left, video_flat, mn=0.0):
    """Return a per-step denoise_mask_function that narrows/widens the fade, or
    None when the fade schedule is off or there is nothing to vary.

    The incoming denoise_mask is ComfyUI's packed per-tile latent mask (video
    tokens first, then audio and any other chunks). Only the first `video_flat`
    elements belong to the video tile; the rest (audio) must be left untouched.
    `video_flat` is the flattened size of the video tile latent. `mn` is the
    masked_area_noise in 0..1: it raises the per-step mask toward 1 letting
    noise into the masked band."""
    schedule = sp.get("dynamic_fade", "off")
    if schedule == "off":
        return None
    fmin_w = int(sp.get("dynamic_fade_min", 0)) // 16
    fmin_h = int(sp.get("dynamic_fade_min", 0)) // 16
    if fw <= fmin_w and fh <= fmin_h:
        return None
    fw_start, fh_start = max(fw, 0), max(fh, 0)
    fmin_w, fmin_h = min(fmin_w, fw_start), min(fmin_h, fh_start)
    # Dynamic fade always uses the frozen-seam mask0 geometry. Only mask0 grows
    # its ramp monotonically (freeing columns without ever re-freezing a column
    # the sampler has already generated), so a moving fade width stays stable.
    # The mask1 layout (ramp against the seam, no frozen run) cannot track a
    # changing fade width without a free->kept flip, which shows up as a seam.
    done_top = done_top and ovh > 0
    done_left = done_left and ovw > 0
    s_tok = tr_s * tc_s
    n_frames = video_flat // s_tok

    def fade_at(p):
        # p in 0..1 across the tile's sampling (0 = first step, 1 = last)
        if schedule == "widening":
            return fmin_w + (fw_start - fmin_w) * p, fmin_h + (fh_start - fmin_h) * p
        return fw_start - (fw_start - fmin_w) * p, fh_start - (fh_start - fmin_h) * p

    # Build the full per-step mask schedule once (from the sigma schedule a tile
    # will actually sample with) and hand out masks by step index, instead of
    # re-deriving sigma -> progress on every call. `dynamic_fade = off` already
    # returns None above, so the schedule only builds when a fade varies.
    cache = {}

    def step_fn(sigma, denoise_mask, **kwargs):
        sigmas = kwargs.get("extra_options", {}).get("sigmas")
        n_sigmas = int(sigmas.numel()) if sigmas is not None else 0
        masks = cache.get(n_sigmas)
        if masks is None:
            step_count = max(n_sigmas - 1, 1)
            masks = []
            for i in range(n_sigmas - 1):
                p = i / (step_count - 1) if step_count > 1 else 0.0
                cw, ch = fade_at(p)
                m = make_fade_mask(tr_s, tc_s, ovh, ovw, done_top, done_left,
                                   fade_h=round(ch), fade_w=round(cw))
                m[tr:tr_s, :] = 0.0
                m[:, tc:tc_s] = 0.0
                if mn > 0:
                    m = m + mn * (1.0 - m)
                masks.append(m)
            cache[n_sigmas] = masks
        idx = 0
        if sigmas is not None:
            # index of this call in the decreasing sigma schedule
            idx = int((sigmas > sigma + 1e-6).sum())
        m = masks[min(idx, len(masks) - 1)]
        flat = denoise_mask.clone()
        flat.reshape(denoise_mask.shape[0], -1)[:, :video_flat] = \
            m.reshape(1, -1).repeat(denoise_mask.shape[0], n_frames)
        return flat

    return step_fn


def blend_weights(t, overlap_blend, overlap_mode):
    """Weight given to the NEW tile's content across an overlap band.

    t runs 0..1 from the done-seam toward the tile interior. overlap_mode 'later'
    hands the band to the new tile; 'earlier' to the accumulated content.
    overlap_blend selects the transition shape."""
    if overlap_blend == "overwrite":
        return torch.ones_like(t) if overlap_mode == "later" else torch.zeros_like(t)
    if overlap_blend == "midpoint":
        step = (t >= 0.5).to(t.dtype)
    elif overlap_blend == "smoothstep":
        step = t * t * (3.0 - 2.0 * t)
    else:
        step = t
    if overlap_mode == "earlier":
        return step
    return 1.0 - step


def bright_match_tile(tile, ref, clamp=0.05):
    """Per-frame, per-channel median brightness (DC) match of a sampled tile to
    a same-sized reference region.

    Adapted from the median-DC seam matching in the
    Comfyui_Minimax_h3_latent_Upscaler plugin (dc_correct / grade_pin): there the
    offset is a single per-channel value over the whole chunk, here it is split
    per (frame, channel) so a tile's brightness baseline tracks the source at the
    same (tile, frame) instead of drifting per tile / per frame. Median (not
    mean) is robust to local highlights; clamp bounds the correction. Chained
    into one isolated call so it is trivial to drop. Experimental."""
    d = (tile - ref).float().reshape(tile.shape[0], tile.shape[1], tile.shape[2], -1)
    dc = d.median(dim=-1).values.clamp(-clamp, clamp)
    return tile - dc.to(tile.dtype).view(tile.shape[0], tile.shape[1], tile.shape[2], 1, 1)


def _solve_equal_tiles(total_px, count, base_overlap_px, granularity):
    """Solve (tile_px, overlap_px) so `count` tiles of EXACTLY equal size cover
    total_px, edge tiles included: count*tile - (count-1)*overlap == total_px.

    The solved overlap is a multiple of `granularity` (one latent token:
    16px for MiniMax H3, 32px for LTX). `base_overlap_px` is the user's desired
    overlap; the search starts from the smallest tile that honours it, so the
    solved overlap lands as close to it as the divisibility constraints allow.
    Every tile is aligned to the model's spatial patch grid (two latent tokens),
    so a tile size that is not a multiple of that grid is pushed up by growing
    the overlap rather than left as a seam-prone odd tile. Returns (tile_px,
    overlap_px)."""
    g = int(granularity)
    pg = 2 * g
    if count <= 1:
        return -(-int(total_px) // pg) * pg, 0
    start = -(-((int(total_px) + (count - 1) * int(base_overlap_px)) // count) // pg) * pg
    # s - overlap == (total - s) / (count - 1), so s is bounded from above too:
    # every tile needs at least one token of non-overlapped content.
    upper = int(total_px) - g * (count - 1)
    if start <= upper:
        for s in range(start, upper + 1, pg):
            num = count * s - int(total_px)
            if num % (count - 1) == 0:
                o = num // (count - 1)
                if o % g == 0 and 0 <= o <= s - g:
                    return s, o
    # Exact equal-tile solve not reachable: fall back to the smallest
    # patch-aligned tile and derive the nearest valid overlap for it.
    s = start if start <= upper else int(total_px)
    o = int(round((count * s - int(total_px)) / (count - 1) / g) * g)
    o = max(0, min(o, s - g))
    return s, o


def _solve_equal_tiles_ld(total_px, count, base_overlap_px, granularity):
    """Solve (tile_px, overlap_px) so `count` tiles of EXACTLY equal size cover
    total_px, edge tiles included: count*tile - (count-1)*overlap == total_px.

    The solved overlap is a multiple of `granularity` (one latent token:
    16px for MiniMax H3, 32px for LTX). `base_overlap_px` is the user's desired
    overlap; the search starts from the smallest tile that honours it, so the
    solved overlap lands as close to it as the divisibility constraints allow.
    Returns (tile_px, overlap_px); raises ValueError if no solution exists."""
    g = int(granularity)
    if count <= 1:
        return -(-int(total_px) // g) * g, 0
    start = -(-((int(total_px) + (count - 1) * int(base_overlap_px)) // count) // g) * g
    # s - overlap == (total - s) / (count - 1), so s is bounded from above too:
    # every tile needs at least one token of non-overlapped content.
    upper = int(total_px) - g * (count - 1)
    if start > upper:
        raise ValueError(
            f"Cannot split {total_px}px into {count} equal tiles with "
            f"{base_overlap_px}px overlap: tiles would have no unique content. "
            f"Reduce the overlap or the tile count.")
    for s in range(start, upper + 1, g):
        num = count * s - int(total_px)
        if num % (count - 1) == 0:
            o = num // (count - 1)
            if o % g == 0 and 0 <= o <= s - g:
                return s, o
    raise ValueError(
        f"No equal-tile solution for {total_px}px / {count} tiles / overlap "
        f"{base_overlap_px}px on a {g}px grid; adjust the overlap slightly.")


TOKEN_CEILING = 74898

def tokens_per_forward(tile_w_px, tile_h_px, video_tokens):
    """Sequence length of ONE forward over a tile of this pixel size.

    H3 patchifies a 2x2 block of 16px latent tokens, so a forward sees
    (w/32) x (h/32) spatial positions for every video latent token in the
    chunk. A 102-frame chunk is 30 video tokens, so a 2048x864 strip is
    64 * 27 * 30 = 51,840.
    """
    return (int(tile_w_px) // 32) * (int(tile_h_px) // 32) * int(video_tokens)


def solve_grid_for_budget(frame_w_px, frame_h_px, video_tokens,
                          ol_w_px, ol_h_px, budget, min_tile_px, max_side=9):
    """Fewest equal tiles whose per-tile forward fits `budget` tokens.

    Fewest wins because every extra tile is another forward on every step.
    Ties are broken towards splitting the LONGER frame axis, so a tall frame
    comes out as horizontal strips rather than a checkerboard: 3x1 has two
    seam lines, 2x2 has two plus a crossing where four tiles meet.

    Returns (grid_rows, grid_cols, tile_w_px, tile_h_px, ol_w_px, ol_h_px),
    or None if even a 9x9 grid cannot get under the budget.
    """
    for total in range(1, max_side * max_side + 1):
        best = None
        for gr in range(1, max_side + 1):
            if total % gr:
                continue
            gc = total // gr
            if gc > max_side:
                continue
            try:
                twpx, owpx = _solve_equal_tiles_ld(frame_w_px, gc, ol_w_px, 32)
                thpx, ohpx = _solve_equal_tiles_ld(frame_h_px, gr, ol_h_px, 32)
            except ValueError:
                continue
            if twpx < min_tile_px or thpx < min_tile_px:
                continue
            if tokens_per_forward(twpx, thpx, video_tokens) > budget:
                continue
            long_split = gr if frame_h_px >= frame_w_px else gc
            rank = (-long_split, abs(gr - gc))
            if best is None or rank < best[0]:
                best = (rank, gr, gc, twpx, thpx, owpx, ohpx)
        if best is not None:
            return best[1:]
    return None


def resolve_grid_on_frame(sp, w, h, t, ol_w, ol_h, min_tile):
    """Re-solve the tile grid against the frame ACTUALLY in hand.

    The params node solves its equal-tile grid from upscale_width/height,
    because it runs before any latent exists and has nothing else to go on.
    That turns those two widgets into a promise about a frame this node has
    not seen, and a stale promise fails silently: a grid solved for 544x544
    and applied to 832x1248 came out 7 rows x 2 cols when ONE column was
    asked for, so a vertical seam appeared in a layout chosen to have none.

    Here the actual chunk is in hand, so the promise is not needed.

      rows_cols - the SHAPE is the setting: ask for 3 rows and 1 column and
                  that is what you get at 1MP, at 5MP, at anything.
      auto      - the SEQUENCE LENGTH is the setting: the shape is solved
                  from it, so raising the megapixel dial adds tiles by
                  itself instead of walking into the token ceiling.

    All four values in and out are LATENT tokens (16px), which is what the
    callers work in.
    """
    mode = str(sp.get("tile_size_mode") or "")
    tw, th = None, None

    if mode == "auto":
        budget = int(sp.get("token_budget") or TOKEN_CEILING)
        got = solve_grid_for_budget(w * 16, h * 16, t, ol_w * 16, ol_h * 16,
                                    budget, min_tile * 16)
        if got is None:
            raise ValueError(
                "auto tiling: no grid up to 9x9 fits %d tokens on a %dx%dpx "
                "frame of %d video tokens (min_tile_size %dpx). Raise "
                "token_budget, lower min_tile_size, or shorten the temporal "
                "chunk." % (budget, w * 16, h * 16, t, min_tile * 16))
        _gr, _gc, _twpx, _thpx, _owpx, _ohpx = got
        seq = tokens_per_forward(_twpx, _thpx, t)
        whole = tokens_per_forward(w * 16, h * 16, t)
        print("[MMH3] auto tiling on the real frame %dx%dpx (%d video tokens): "
              "%dx%d grid, tiles %dx%dpx, overlap w=%d h=%d | %d tokens per "
              "forward vs %d whole-frame, budget %d"
              % (w * 16, h * 16, t, _gr, _gc, _twpx, _thpx, _owpx, _ohpx,
                 seq, whole, budget))
        tw, th = _twpx // 16, _thpx // 16
        ol_w, ol_h = _owpx // 16, _ohpx // 16

    elif mode == "rows_cols":
        _gr = max(1, int(sp.get("grid_rows") or 1))
        _gc = max(1, int(sp.get("grid_cols") or 1))
        _twpx, _owpx = _solve_equal_tiles_ld(w * 16, _gc, ol_w * 16, 32)
        _thpx, _ohpx = _solve_equal_tiles_ld(h * 16, _gr, ol_h * 16, 32)
        if _twpx < min_tile * 16 or _thpx < min_tile * 16:
            raise ValueError(
                "rows_cols: a %dx%d grid over the real %dx%dpx frame solves to "
                "%dx%dpx tiles, below min_tile_size (%dpx). Ask for fewer "
                "rows/cols, raise the megapixel target, or lower min_tile_size."
                % (_gr, _gc, w * 16, h * 16, _twpx, _thpx, min_tile * 16))
        print("[MMH3] rows_cols solved on the real frame %dx%dpx: %dx%d grid, "
              "tiles %dx%dpx, overlap w=%d h=%d | %d tokens per forward"
              % (w * 16, h * 16, _gr, _gc, _twpx, _thpx, _owpx, _ohpx,
                 tokens_per_forward(_twpx, _thpx, t)))
        tw, th = _twpx // 16, _thpx // 16
        ol_w, ol_h = _owpx // 16, _ohpx // 16

    if tw is None:
        return None, None, ol_w, ol_h
    return tw, th, ol_w, ol_h


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    """Spatially crop every keyframe's video latent to a tile of the source frame.

    Keyframes whose latent already matches the source spatial size are cropped to
    the tile's latent region. If a keyframe is at a DIFFERENT spatial scale (e.g.
    the source was latent-upscaled before tiling, or a different VAE/resolution
    produced the conditioning), it is resized to the source spatial size first so
    the cropped keyframe exactly matches the tile's row count - otherwise the
    model's cond/video row broadcast (`all_video_rows[~img_update] = cond_video_rows`)
    fails with a shape mismatch. Audio keyframes are untouched (audio is not spatial)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    kh, kw = lt.shape[3], lt.shape[4]
                    if kh == src_h and kw == src_w:
                        # The model pads the target latent to the 2x2 spatial patch
                        # grid but its cond path does not, so the cropped keyframe
                        # must be padded to the same even grid or patchify fails.
                        crop = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                        nkf["latent"] = comfy.ldm.common_dit.pad_to_patch_size(crop, (1, 2, 2))
                    else:
                        # Keyframe latent is at a different spatial scale than the
                        # tile source. Resize it to (src_h, src_w) so the crop
                        # produces a keyframe that matches the tile dimensions.
                        B, C, T, H, W = lt.shape
                        lt_r = torch.nn.functional.interpolate(
                            lt.to(torch.float32).reshape(B * T, C, H, W),
                            size=(src_h, src_w), mode="bilinear", align_corners=False,
                        ).reshape(B, C, T, src_h, src_w)
                        nkf["latent"] = comfy.ldm.common_dit.pad_to_patch_size(
                            lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous(), (1, 2, 2))
                    cropped.append(nkf)
                else:
                    cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


def trim_keyframe(kf, f0, f1):
    """Copy a keyframe cut to the portion fully inside pixel frames [f0, f1)."""
    idx = kf["resolved_frame_index"]
    latent = kf.get("latent")
    audio_latent = kf.get("audio_latent")
    has_v = latent is not None
    has_a = audio_latent is not None

    if not has_v and not has_a:
        if idx < f0 or idx >= f1:
            return None
        return {"resolved_frame_index": idx - f0}

    out = {}
    if has_v:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if has_a:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    if "latent" not in out and "audio_latent" not in out:
        return None
    return out


def reanchor_conditioning(cond, f0, f1, spatial=None):
    """Cut/re-anchor minimax_keyframes to the pixel-frame segment [f0, f1).

    When `spatial` (latent_h, latent_w) is given, keyframe video latents whose
    spatial size differs are resized to it (bilinear)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [trim_keyframe(kf, f0, f1) for kf in kfs]
            trimmed = [kf for kf in trimmed if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = F.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear", align_corners=False
                            ).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def anchor_conditioning(cond, prev_video, f0, strength):
    """Replace the frame-0 keyframe with the previous chunk's re-sampled frame.

    Mirrors the 'Anchor MiniMax H3 Latent' node: keyframes are frozen rows in
    the H3 packed sequence, so pinning frame 0 to the content the previous chunk
    ended with removes the detail mismatch at the seam. `strength` becomes
    minimax_visual_cond_noise_aug (0.999 = model default)."""
    t = tokens_for_frames(f0)
    if t >= prev_video.shape[2]:
        raise ValueError("previous result does not extend to the current segment's start frame")
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t:t + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def normalize_minimax_refs(cond):
    """Make minimax_refs blocks SELF-CONSISTENT for the H3 packed layout.

    The model counts frozen rows from two paths that must agree exactly:
      * PackedLayout reserves ref rows from each block's METADATA
        (latent_h/latent_w/latent_t) and does NOT check whether the block
        actually carries a "latent";
      * cond_video_latents delivers rows from blocks where "latent" EXISTS,
        sized by the latent's real shape.
    If an upstream node/version writes metadata that disagrees with the latent
    (or emits a visual block without a latent), layout reserves one or more
    phantom frames and sampling crashes with
    'all_video_rows[~img_update] = cond_video_rows: shape mismatch'.
    Here we drop visual blocks without a latent and rewrite the metadata from
    the real latent shape, so both paths can never diverge."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        refs = nd.get("minimax_refs")
        if refs:
            fixed = []
            for blk in refs:
                nblk = dict(blk)
                lt = nblk.get("latent")
                if lt is None:
                    # visual block without a latent would reserve phantom rows
                    if nblk.get("kind") in ("image", "video", "video_audio"):
                        continue
                    fixed.append(nblk)
                    continue
                nblk["latent_h"] = int(lt.shape[3])
                nblk["latent_w"] = int(lt.shape[4])
                if nblk.get("kind") in ("video", "video_audio"):
                    nblk["latent_t"] = int(lt.shape[2])
                fixed.append(nblk)
            nd["minimax_refs"] = fixed
        out.append([tensor, nd])
    return out


def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    w = w.view(shape)
    return a + (b - a) * w


# Model-free (no upscale model) spatial resize and the model/model-free dispatch.
# The H3 3D model-based upscaler lives in h3_latent_upscaler.py.
def upscale_video_interp(video, param):
    """Model-free upscale of one chunk's video latent via interpolation (audio
    untouched) - mirrors ComfyUI's 'Upscale Latent' node. Returns (upscaled_video,
    new_h, new_w); the video latent [B,24,T,H,W] is resized in HxW only."""
    method = param["method"]
    width = int(param["width"])
    height = int(param["height"])

    _, c, t, h_in, w_in = video.shape
    h_out, w_out, _ = _compute_upscale_target(width, height, h_in, w_in)
    if h_out == h_in and w_out == w_in:
        return video, h_in, w_in

    video_bt = video.permute(0, 2, 1, 3, 4).reshape(-1, c, h_in, w_in)
    up = torch.nn.functional.interpolate(video_bt, size=(h_out, w_out), mode=method)
    up = up.reshape(video.shape[0], t, c, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
    return up, h_out, w_out


def upscale_latent(video, param):
    """Dispatch a chunk's video upscale: H3 3D model (param has 'model_name') or
    model-free interpolation (param has 'method'). Audio is never touched."""
    if "model_name" in param:
        return upscale_video(video, param)
    return upscale_video_interp(video, param)


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------

def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider


def sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg):
    """Sample one piece (full chunk or tile). Mirrors SamplerCustomAdvanced,
    including the x0 preview callback. Returns nested samples (video+audio)."""
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_image
    noise_mask = latent.get("noise_mask")

    guider = build_guider(model, cond, negative, cfg)
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise.generate_noise(latent), latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=noise.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples


def _resize_images(frames, height, width):
    """Bilinear-resize pixel frames [T, C, H, W] to [T, C, height, width]."""
    B, C = frames.shape[0], frames.shape[1]
    return F.interpolate(
        frames.reshape(B, C, frames.shape[2], frames.shape[3]),
        size=(height, width), mode="bilinear", align_corners=False)


def _chunk_fun_control(param, f0, f1, fc_buffer):
    """Resolve the control video for one chunk [f0, f1) honouring the upscale mode.

    Returns ([T', C, H, W], mode). For 'per_chunk'/'all' the returned frames are
    already upscaled to the target size (either for the chunk or the whole
    video); for 'per_tile' they stay low-res so each tile crops+upscales only
    its own piece."""
    lo = param["control_video"]
    up_h, up_w = param["upscale_height"], param["upscale_width"]
    mode = param["control_upscale_mode"]
    if mode == "all":
        return fc_buffer[f0:f1], mode
    chunk_lo = lo[f0:f1]
    if mode == "per_chunk":
        return _resize_images(chunk_lo, up_h, up_w), mode
    return chunk_lo, mode  # per_tile: low-res kept


def _tile_fun_control(param, chunk_ctrl, mode, r0, c0, tr, tc):
    """Crop a chunk control representation to a tile's spatial window.

    For 'per_tile' the chunk frames are low-res: map the tile's upscale-pixel
    window back to low-res, crop it, and upscale the small crop to the tile's
    upscale footprint (no full upscale frame is ever materialized). For
    'per_chunk'/'all' the chunk frames are already at the upscale size, so the
    upscale-pixel window is cropped directly."""
    up_h, up_w = param["upscale_height"], param["upscale_width"]
    if mode == "per_tile":
        lo_h, lo_w = chunk_ctrl.shape[2], chunk_ctrl.shape[3]
        pr1 = min((r0 + tr) * VAE_DOWNSAMPLE * lo_h // up_h, lo_h)
        pc1 = min((c0 + tc) * VAE_DOWNSAMPLE * lo_w // up_w, lo_w)
        crop = chunk_ctrl[:, :, r0 * VAE_DOWNSAMPLE * lo_h // up_h:pr1,
                          c0 * VAE_DOWNSAMPLE * lo_w // up_w:pc1]
        return _resize_images(crop, tr * VAE_DOWNSAMPLE, tc * VAE_DOWNSAMPLE)
    pr1 = min((r0 + tr) * VAE_DOWNSAMPLE, up_h)
    pc1 = min((c0 + tc) * VAE_DOWNSAMPLE, up_w)
    return chunk_ctrl[:, :, r0 * VAE_DOWNSAMPLE:pr1, c0 * VAE_DOWNSAMPLE:pc1]


def resolve_control_range(d, sigmas, sampling):
    """Resolve the active denoising (start, end) percent range from a start/end
    set. In 'percent' mode the pair is used as-is. In 'step' mode the steps map to
    the actual `sigmas` schedule as a half-open [start_step, end_step) range, so
    start=0/end=1 gates only the first denoising step. Both percent boundaries are
    back-solved from sigma midpoints between adjacent scheduled steps (with virtual
    bounds at the schedule ends), so the inclusive controlnet comparison lands
    strictly between gated and ungated steps regardless of grid quantization."""
    if d["start_end_set"] == "percent":
        return (d["start_percent"], d["end_percent"])
    n = max(int(sigmas.shape[-1]) - 1, 1)
    start_step = max(int(d["start_step"]), 0)
    end_step = max(int(d["end_step"]), start_step + 1)
    c0 = min(start_step, n)
    c1 = min(end_step, n)

    if c0 > 0:
        top = (float(sigmas[c0]) + float(sigmas[c0 - 1])) / 2.0
    else:
        top = (float(sigmas[0]) + 1.0) / 2.0
    last = max(c1 - 1, 0)
    if c1 < n:
        bottom = (float(sigmas[last]) + float(sigmas[c1])) / 2.0
    else:
        bottom = float(sigmas[last]) / 2.0

    alpha = float(getattr(sampling, "shift", 1.0))
    if alpha == 1.0:
        start_p, end_p = 1.0 - top, 1.0 - bottom
    else:
        start_p = 1.0 - top / (alpha - (alpha - 1.0) * top)
        end_p = 1.0 - bottom / (alpha - (alpha - 1.0) * bottom)
    return (min(start_p, end_p), max(start_p, end_p))


def inject_fun_control(cond, control_net, vae, control_video, strength, sigmas, d, sampling):
    """Drive the conditioning's control field with a fresh MiniMax H3 Fun
    ControlNet copy carrying the given pixel control video."""
    c_net = control_net.copy()
    start_p, end_p = resolve_control_range(d, sigmas, sampling)
    c_net.set_cond_hint(control_video, strength, (start_p, end_p), vae=vae)
    return apply_control(cond, c_net)


def apply_control(cond, c_net):
    """Set the conditioning's control field to a ready controlnet copy."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd["control"] = c_net
        nd["control_apply_to_uncond"] = True
        out.append([tensor, nd])
    return out


def _build_fun_net(fun_control, r0, c0, tr, tc, sigmas, sampling):
    """Build one fresh Fun ControlNet copy carrying the tile's cropped control
    video (the hint cache is keyed on size only, so each tile needs its own)."""
    tile_hint = _tile_fun_control(fun_control, fun_control["chunk_ctrl"],
                                  fun_control["mode"], r0, c0, tr, tc)
    c_net = fun_control["control_net"].copy()
    start_p, end_p = resolve_control_range(fun_control, sigmas, sampling)
    c_net.set_cond_hint(tile_hint, fun_control["strength"], (start_p, end_p), vae=fun_control["vae"])
    return c_net


# ---------------------------------------------------------------------------
# stitching helpers
# ---------------------------------------------------------------------------

def temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0, f0):
    """Stitch one re-sampled chunk into the accumulated latent (cross-fade).
    Mirrors 'Append MiniMax H3 Latents'. Returns (result_v, result_a)."""
    if acc_v is None:
        return chunk_v, chunk_a

    gi = k0
    agi = round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    result_v = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                           device=acc_v.device, dtype=acc_v.dtype)
    result_a = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    result_v[:, :, :acc_v.shape[2]] = acc_v
    result_a[:, :, :, :acc_a.shape[-1]] = acc_a

    v = chunk_v
    a = chunk_a
    ov = (acc_v.shape[2] - gi) if index > 0 else 0
    if ov > 0:
        ov = min(ov, v.shape[2])
        tail = result_v[:, :, gi:gi + ov].clone()
        result_v[:, :, gi:gi + ov] = _crossfade(tail, v[:, :, :ov], dim=2)
        v = v[:, :, ov:]
    write_v = gi + max(ov, 0)
    if v.shape[2] > 0:
        result_v[:, :, write_v:write_v + v.shape[2]] = v

    ova = (acc_a.shape[-1] - agi) if index > 0 else 0
    if ova > 0:
        ova = min(ova, a.shape[-1])
        tail = result_a[:, :, :, agi:agi + ova].clone()
        result_a[:, :, :, agi:agi + ova] = _crossfade(tail, a[:, :, :, :ova], dim=3)
        a = a[:, :, :, ova:]
    write_a = agi + max(ova, 0)
    if a.shape[-1] > 0:
        result_a[:, :, :, write_a:write_a + a.shape[-1]] = a

    return result_v, result_a


class _PreparedNoise:
    """Hands back a noise tensor that was made earlier, instead of a new one.

    WHY THIS EXISTS - THE SEAM.

    Every tile used to call noise.generate_noise(its own latent), which is
    comfy.sample.prepare_noise(shape, seed). The seed is the same for all of
    them, but torch.randn fills a tensor ROW-MAJOR from the generator's
    stream, so the value that lands at a given spatial position depends on the
    tensor's WIDTH. Two tiles that overlap have different widths and different
    offsets, so for one picture position inside the shared band, tile A reads
    stream index y*W_A + x while tile B reads y*W_B + (x - offset). Those are
    unrelated numbers.

    The consequence is not subtle: the two tiles denoise the SAME region from
    DIFFERENT noise, converge on two different pictures, and the stitcher then
    cross-fades between them. That is a seam by construction. Overlap width,
    fade length and blend curve all act downstream of it, which is why tuning
    them can soften the edge but never remove it.

    The fix is to make the noise a property of the CHUNK rather than of the
    tile: draw one field at the chunk's full size, then hand each tile the
    slice that sits under it. Both tiles then see identical noise where they
    overlap and, given the same conditioning and the same schedule, converge on
    very nearly the same picture there - so the blend is joining two views of
    one image rather than two different images.

    This is the same reason a tiled VAE decode has no seams: it is
    deterministic, so neighbouring tiles cannot disagree.
    """

    def __init__(self, seed, noise):
        self.seed = seed
        self._noise = noise

    def generate_noise(self, input_latent):
        return self._noise


def _feather_mask(tr, tc, ovh, ovw, top, left, bottom, right, device, dtype):
    """A tile's weight in the accumulator: 1 in the middle, ramping to 0 on any
    edge that has a neighbour. Straight out of comfy.utils.tiled_scale_multidim,
    which is what a tiled VAE decode uses - the same bookkeeping, applied to a
    model prediction instead of a finished tile.
    """
    # COSINE, NOT LINEAR. A linear ramp meets 1.0 with a kink - the weight is
    # continuous but its slope is not, and that discontinuity is visible as a
    # faint line at the inner edge of the band. A raised cosine leaves at zero
    # slope and arrives at zero slope. TenStrip's LTX tiled sampler uses the
    # same window and its changelog records the same hunt, which is decent
    # evidence it is worth the two extra lines.
    def ramp(n, rising):
        i = torch.arange(n, device=device, dtype=dtype)
        w = 0.5 * (1.0 - torch.cos(math.pi * i / n))
        return w if rising else torch.flip(w, (0,))

    m = torch.ones((tr, tc), device=device, dtype=dtype)
    if left and ovw > 0:
        m[:, :ovw] *= ramp(min(ovw, tc), True)[None, :]
    if right and ovw > 0:
        m[:, -ovw:] *= ramp(min(ovw, tc), False)[None, :]
    if top and ovh > 0:
        m[:ovh, :] *= ramp(min(ovh, tr), True)[:, None]
    if bottom and ovh > 0:
        m[-ovh:, :] *= ramp(min(ovh, tr), False)[:, None]
    return m


class _ZeroNoise:
    """No noise. Every step after the first starts from the latent as it is."""

    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        t = input_latent["samples"]
        if hasattr(t, "tensors"):
            return comfy.nested_tensor.NestedTensor(
                tuple(torch.zeros_like(x) for x in t.tensors))
        return torch.zeros_like(t)


def spatial_process_joint(chunk_v, chunk_a, cond, sp, model, noise, sampler,
                          sigmas, negative, cfg):
    """Tiles that are stepped TOGETHER instead of finished one at a time.

    WHY THIS EXISTS.

    A tiled VAE decode has no seams, and it is worth being precise about why:
    its per-tile function is a single deterministic pass, so two tiles that
    overlap compute the SAME pixels twice and the feathered accumulator simply
    reconstructs them. The mask is bookkeeping, not repair.

    spatial_process() borrows that accumulator but hands it a whole denoising
    TRAJECTORY per tile. A trajectory is not deterministic across tiles: the
    same latent patch sits at the edge of one tile and in the middle of
    another, so the transformer sees different neighbours and returns different
    content. The accumulator is then averaging two different pictures, and a
    blend between two different pictures is a seam. That is why widening the
    overlap, lengthening the fade or changing the curve only ever softened it.

    So this keeps comfy's accumulator exactly and changes what is accumulated:
    ONE forward per tile per step, merged into ONE latent, stepped once. A
    single forward is near enough deterministic given the same input, which is
    the property the decode has and a trajectory does not. Tiles are
    re-synchronised after every step, so they never get the chance to diverge.

    Cost is the same as before - tiles x steps forwards, one tile of
    activations at a time. What changes is that the tiles are solving one
    problem together rather than N problems separately.
    """
    tw = int(sp["tile_width"]) // 16
    th = int(sp["tile_height"]) // 16
    ol_w = int(sp["spatial_w_overlap"]) // 16
    ol_h = int(sp["spatial_h_overlap"]) // 16
    min_tile = int(sp["min_tile_size"]) // 16
    if tw <= 0 or th <= 0:
        raise ValueError("tile_width/tile_height must be multiples of 32 pixels")
    if ol_w >= tw or ol_h >= th:
        raise ValueError("overlap must be smaller than the tile")

    _, c, t, h, w = chunk_v.shape

    # Re-solve the grid against the frame in hand; see
    # resolve_grid_on_frame() for why the params node cannot.
    _tw, _th, ol_w, ol_h = resolve_grid_on_frame(
        sp, w, h, t, ol_w, ol_h, min_tile)
    if _tw is not None:
        tw, th = _tw, _th
    rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(
        h, w, th, tw, ol_h, ol_w, min_tile, min_tile)
    odd = ([x * 16 for x in tcols if x % 2] + [x * 16 for x in trows if x % 2])
    if odd:
        raise ValueError(
            "Tile grid produced an extent that is not a multiple of 32px: %s. "
            "H3 patchifies 2x2 latent tokens." %
            ", ".join("%dpx" % v for v in sorted(set(odd))))

    nrows, ncols = len(rows), len(cols)
    ta = chunk_a.shape[-1]
    dev, dt = chunk_v.device, chunk_v.dtype
    seed = getattr(noise, "seed", 0)

    # ONE noise field for the chunk, sliced per tile - so the region two tiles
    # share starts from identical noise as well as identical latent.
    try:
        base = comfy.sample.prepare_noise(
            comfy.nested_tensor.NestedTensor((chunk_v, chunk_a)), seed)
        nv_full, na_full = base.tensors[0], base.tensors[1]
    except Exception as e:
        raise RuntimeError("could not draw the chunk noise field: %s" % e)

    # cond and mask per tile, built once rather than per step
    tiles = []
    for i in range(nrows):
        for j in range(ncols):
            r0, c0, tr, tc = rows[i], cols[j], trows[i], tcols[j]
            m = _feather_mask(tr, tc, row_ovl[i] or ol_h, col_ovl[j] or ol_w,
                              top=(i > 0), left=(j > 0),
                              bottom=(i < nrows - 1), right=(j < ncols - 1),
                              device=dev, dtype=torch.float32)
            tiles.append({
                "r0": r0, "c0": c0, "tr": tr, "tc": tc,
                "cond": crop_keyframes_to_tile(cond, h, w, r0, c0, tr, tc),
                "mask": m[None, None, None],
            })
    print("[MMH3] joint tiling: %dx%d = %d tiles, %d step(s), stepped together"
          % (nrows, ncols, len(tiles), max(1, len(sigmas) - 1)))

    x = chunk_v
    zero = _ZeroNoise(seed)
    audio_out = chunk_a
    # audio is never tiled and never re-sampled: it is frozen in every piece,
    # exactly as the sequential path froze it.
    ma = torch.zeros((1, 32, 2, ta), device=chunk_a.device, dtype=chunk_a.dtype)

    for step in range(len(sigmas) - 1):
        two = sigmas[step:step + 2]
        acc = torch.zeros_like(x, dtype=torch.float32)
        wsum = torch.zeros((1, 1, 1, x.shape[3], x.shape[4]),
                           device=dev, dtype=torch.float32)
        for td in tiles:
            r0, c0, tr, tc = td["r0"], td["c0"], td["tr"], td["tc"]
            xt = x[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
            mv = torch.ones((1, 1, 1, tr, tc), device=dev, dtype=torch.float32)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((xt, audio_out)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
            }
            if step == 0:
                nv = nv_full[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                n = _PreparedNoise(seed, comfy.nested_tensor.NestedTensor(
                    (nv, na_full)))
            else:
                n = zero
            out = sample_piece(piece, td["cond"], model, n, sampler, two,
                               negative, cfg)
            ot = out.tensors[0].to(device=dev, dtype=torch.float32)
            acc[:, :, :, r0:r0 + tr, c0:c0 + tc] += ot * td["mask"]
            wsum[:, :, :, r0:r0 + tr, c0:c0 + tc] += td["mask"][:, :, 0]
        x = (acc / wsum.clamp(min=1e-8)).to(dtype=dt)

    info = {"rows": rows, "cols": cols, "tile_rows": trows, "tile_cols": tcols,
            "row_overlaps": row_ovl, "col_overlaps": col_ovl,
            "orig_h": h, "orig_w": w, "joint": True,
            "steps": max(1, len(sigmas) - 1), "tiles": len(tiles)}
    return x, info


def spatial_process(chunk_v, chunk_a, cond, sp, model, noise, sampler, sigmas, negative, cfg,
                    fun_control=None, inpaint=None):
    """Inner loop: spatial split -> per-tile sampling -> spatial stitch.
    Mirrors the spatial split/extract/append trio. Audio is carried unchanged
    (frozen in every tile, never re-sampled). Returns (reassembled_video, info)."""
    sampling = getattr(model, "model_sampling", None)
    tw = int(sp["tile_width"]) // 16
    th = int(sp["tile_height"]) // 16
    ol_w = int(sp["spatial_w_overlap"]) // 16
    ol_h = int(sp["spatial_h_overlap"]) // 16
    fw = int(sp["fade_width"]) // 16
    fh = int(sp["fade_height"]) // 16
    min_tile = int(sp["min_tile_size"]) // 16
    overlap_mode = sp["overlap_mode"]
    overlap_blend = sp["overlap_blend"]
    mn = float(sp.get("masked_area_noise", 0.0))
    bright = bool(sp.get("brightness_match", False))

    if tw <= 0 or th <= 0:
        raise ValueError("tile_width/tile_height must be multiples of 32 pixels")
    if ol_w >= tw or ol_h >= th:
        raise ValueError("spatial_w_overlap/spatial_h_overlap must be smaller than the tile size")
    if min_tile > th or min_tile > tw:
        raise ValueError("min_tile_size must not exceed the tile size")

    _, c, t, h, w = chunk_v.shape
    # Grid modes must use the actual chunk, which may differ from the parameter preview.
    _tw, _th, ol_w, ol_h = resolve_grid_on_frame(sp, w, h, t, ol_w, ol_h, min_tile)
    if _tw is not None:
        tw, th = _tw, _th
    rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_tile, min_tile)
    nrows, ncols = len(rows), len(cols)
    ta = chunk_a.shape[-1]

    acc_v = chunk_v.clone()
    tile_info = {
        "rows": rows, "cols": cols, "tile_h": th, "tile_w": tw,
        "overlap_h": ol_h, "overlap_w": ol_w,
        "row_overlaps": row_ovl, "col_overlaps": col_ovl, "min_tile": min_tile,
        "tile_rows": trows, "tile_cols": tcols, "n_cols": ncols,
        "orig_h": h, "orig_w": w, "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
    }

    seed = getattr(noise, "seed", 0)
    tiled_noise = None
    try:
        base = comfy.sample.prepare_noise(comfy.nested_tensor.NestedTensor((chunk_v, chunk_a)), seed)
        tiled_noise = (base.tensors[0], base.tensors[1])
    except Exception as exc:
        print("[MMH3] Shared tile noise unavailable; using per-tile noise: %s" % exc)

    for i in range(nrows):
        for j in range(ncols):
            r0, c0 = rows[i], cols[j]
            tr, tc = trows[i], tcols[j]
            ovh = row_ovl[i]
            ovw = col_ovl[j]
            # The model patches the latent on a 2x2 grid, so every tile must be
            # sampled at even spatial dims or its internal pad shifts the patch
            # grid off the neighbour's and leaves a seam. Odd edge tiles (only the
            # last row/col, when the source dim is odd) are padded up to even; the
            # added strip is frozen and never written back.
            tr_s, tc_s = tr + (tr % 2), tc + (tc % 2)

            tile = torch.zeros((1, c, t, tr_s, tc_s), device=chunk_v.device, dtype=chunk_v.dtype)
            tile[:, :, :, :tr, :tc] = chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc]
            # pre-fill done-overlap strips from the accumulated re-sampled result
            # (only the real extent; the even-pad strip is frozen and left as-is)
            if j > 0 and ovw > 0:
                tile[:, :, :, :tr, :ovw] = acc_v[:, :, :, r0:r0 + tr, c0:c0 + ovw]
            if i > 0 and ovh > 0:
                tile[:, :, :, :ovh, :tc] = acc_v[:, :, :, r0:r0 + ovh, c0:c0 + tc]

            m = spatial_fade_mask(tr_s, tc_s, ovh, ovw,
                                  done_top=(i > 0), done_left=(j > 0),
                                  fade_h=fh, fade_w=fw)
            m[tr:tr_s, :] = 0.0
            m[:, tc:tc_s] = 0.0
            mv = (m + mn * (1.0 - m))[None, None, None]
            ma = torch.zeros((1, 32, 2, ta), device=chunk_a.device, dtype=chunk_a.dtype)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
            }

            cond_tile = crop_keyframes_to_tile(cond, h, w, r0, c0, tr, tc)
            c_net = None
            if fun_control is not None:
                # crop the per-chunk control to this tile's spatial window (per the
                # upscale mode), then drive a fresh controlnet copy - the control
                # hint cache is keyed on size only, so tiles must not share one object
                c_net = _build_fun_net(fun_control, r0, c0, tr, tc, sigmas, sampling)
            if inpaint is not None and (i > 0 or j > 0):
                # Latent-space inpaint: the controlnet's inpaint channel is
                # masked_latent == encode(source * visibility), where source is the
                # already-sampled neighbour content on the kept seams. The VAE
                # encoder is linear, so encode(source * visibility) == source_latent *
                # visibility_latent; we can pin those strips directly from acc_v,
                # skipping the pixel decode/re-encode round-trip entirely. The spatial
                # fade mask doubles as the inpaint mask (1 = keep neighbour, 0 =
                # regenerate) and is chained under the fun control so both guides stack.
                vis = (1.0 - (m > 0.5).to(torch.float32))[None, None, None]  # [1,1,1,tr_s,tc_s]
                vis = vis.expand(1, 1, t, tr_s, tc_s).to(device=acc_v.device)
                masked = torch.zeros((1, 24, t, tr_s, tc_s), device=vis.device, dtype=vis.dtype)
                masked[:, :, :, :tr, :tc] = acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc] * vis[:, :, :, :tr, :tc]
                base = torch.zeros(1, 24, t, tr_s, tc_s, device=vis.device, dtype=vis.dtype)
                inp_net = inpaint["control_net"].copy()
                inp_net.cond_hint = torch.cat([base, vis, masked], dim=1)  # [1,49,t,tr_s,tc_s]
                inp_net.strength = inpaint["strength"]
                inp_net.timestep_percent_range = resolve_control_range(inpaint, sigmas, sampling)
                if c_net is not None:
                    inp_net.set_previous_controlnet(c_net)
                c_net = inp_net
            if c_net is not None:
                cond_tile = apply_control(cond_tile, c_net)

            dynamic = _dynamic_fade_closure(
                sp, fw, fh, tr, tc, tr_s, tc_s, ovh, ovw,
                done_top=(i > 0), done_left=(j > 0),
                video_flat=math.prod(tile.shape[1:]), mn=mn)
            if dynamic is not None:
                model.set_model_denoise_mask_function(dynamic)
            tile_noise = noise
            if tiled_noise is not None:
                nv = tiled_noise[0][:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                if tr_s != tr or tc_s != tc:
                    nv = F.pad(nv, (0, tc_s - tc, 0, tr_s - tr))
                tile_noise = _PreparedNoise(seed, comfy.nested_tensor.NestedTensor((nv, tiled_noise[1])))
            try:
                out = sample_piece(piece, cond_tile, model, tile_noise, sampler, sigmas, negative, cfg)
            finally:
                if dynamic is not None:
                    model.model_options.pop("denoise_mask_function", None)

            tile_v = out.tensors[0][:, :, :, :tr, :tc]  # drop the frozen even-pad strip

            if bright:
                tile_v = bright_match_tile(tile_v, chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc])

            region = acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
            if j > 0 and ovw > 0:
                tt = torch.linspace(0.0, 1.0, ovw, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts[None, None, None, None, :])
                                            + tile_v[:, :, :, :, :ovw] * wts[None, None, None, None, :])
            if i > 0 and ovh > 0:
                tt = torch.linspace(0.0, 1.0, ovh, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts[None, None, None, :, None])
                                            + tile_v[:, :, :, :ovh, :] * wts[None, None, None, :, None])
            band = torch.zeros((1, 1, 1, tr, tc), device=region.device, dtype=torch.bool)
            if j > 0 and ovw > 0:
                band[:, :, :, :, :ovw] = True
            if i > 0 and ovh > 0:
                band[:, :, :, :ovh, :] = True
            region = torch.where(band, region, tile_v)
            acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

    return acc_v, tile_info


# ---------------------------------------------------------------------------
# parameter nodes
# ---------------------------------------------------------------------------

class MMH3LatentUpscaleWithModelParams(io.ComfyNode):
    """Bundle the H3 3D model-based latent upscale settings consumed by the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentUpscaleWithModelParams",
            display_name="MMH3 Latent Upscale with Model Params",
            category="model/latent/minimax",
            description=(
                "Bundle the H3 3D latent upscale settings for the 'MMH3 Ultimate "
                "Upscale' node. Uses the minimax_h3_latent_upscaler_3d checkpoints "
                "from the latent_upscale_models folder (not the standard LatentUpscale "
                "loader - the H3 weights do not match its supported architectures). "
                "Only .safetensors checkpoints are accepted; pickle formats such as "
                ".pth are refused because they can execute arbitrary code when loaded."
            ),
            search_aliases=["h3 upscale params", "upscale param", "h3 upscale"],
            inputs=[
                io.Combo.Input("model_name", options=_scan_models(),
                               tooltip="The H3 latent upscale model file in the latent_upscale_models folder (e.g. minimax_h3_latent_upscaler_3d_*.safetensors). Only .safetensors is accepted - .pth/.pt/.ckpt pickle checkpoints are refused for security reasons. Loading a non-H3 upscale model may error."),
                io.Int.Input("width", default=1280, min=64, max=4096, step=32,
                             tooltip="Target overall pixel width of the upscaled frame (snapped to a multiple of 32, the H3 upscaler's required grid). Must match the conditioning's generation size."),
                io.Int.Input("height", default=704, min=64, max=4096, step=32,
                             tooltip="Target overall pixel height of the upscaled frame (snapped to a multiple of 32, the H3 upscaler's required grid). Must match the conditioning's generation size."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp16", "fp32", "bf16"], default="fp16"),
                io.Boolean.Input("offload_model", default=True,
                                 tooltip="Whether to offload the H3 upscale model to CPU (freeing GPU VRAM) right after upscaling. Default on keeps the low-VRAM behaviour; turn it off to keep the model resident on the device so the next chunk skips the re-upload and runs faster."),
            ],
            outputs=[
                H3_UPSCALE_PARAM.Output("latent_upscale_param",
                                        tooltip="Upscale settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, model_name, width, height, device, precision, offload_model) -> io.NodeOutput:
        width = int(round(width / 32.0)) * 32
        height = int(round(height / 32.0)) * 32
        param = {
            "model_name": model_name,
            "width": width,
            "height": height,
            "device": device,
            "precision": precision,
            "offload_model": offload_model,
        }
        return io.NodeOutput(param)


class MMH3LatentUpscaleParams(io.ComfyNode):
    """Bundle model-free latent upscale settings (interpolation) consumed by the
    Ultimate Upscale node. The video latent is resized spatially, audio passes
    through. Mirrors ComfyUI's 'Upscale Latent' node but keeps the H3 nested
    (video+audio) structure intact."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentUpscaleParams",
            display_name="MMH3 Latent Upscale Params",
            category="model/latent/minimax",
            description=(
                "Bundle model-free latent upscale settings for the 'MMH3 Ultimate "
                "Upscale' node. The chunk's video latent is resized spatially by "
                "interpolation (audio untouched) - no H3 upscale model is loaded. "
                "Target size must match the conditioning's generation size. Reference: "
                "ComfyUI 'Upscale Latent'."
            ),
            search_aliases=["h3 upscale params", "upscale param", "h3 latent upscale", "model-free upscale"],
            inputs=[
                io.Combo.Input("method", options=["nearest-exact", "bilinear", "area", "bicubic"],
                               default="bilinear",
                               tooltip="Interpolation used to resize the video latent's spatial HxW (same as Upscale Latent)."),
                io.Int.Input("width", default=1280, min=64, max=4096, step=32,
                                tooltip="Target overall pixel width of the upscaled frame (snapped to a multiple of 32). Must match the conditioning's generation size."),
                io.Int.Input("height", default=704, min=64, max=4096, step=32,
                                tooltip="Target overall pixel height of the upscaled frame (snapped to a multiple of 32). Must match the conditioning's generation size."),
            ],
            outputs=[
                H3_UPSCALE_PARAM.Output("latent_upscale_param",
                                        tooltip="Model-free upscale settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, method, width, height) -> io.NodeOutput:
        width = int(round(width / 32.0)) * 32
        height = int(round(height / 32.0)) * 32
        param = {
            "method": method,
            "width": width,
            "height": height,
        }
        return io.NodeOutput(param)


class MMH3TemporalSplitParams(io.ComfyNode):
    """Bundle the temporal split settings consumed by the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalSplitParams",
            display_name="MMH3 Temporal Split Params",
            category="model/latent/minimax",
            description=(
                "Bundle the temporal split settings for the 'MMH3 Ultimate Upscale' "
                "node: how the input latent is cut into overlapping time chunks "
                "(outer loop) and how seams are anchored."
            ),
            search_aliases=["h3 temporal params", "temporal split param", "time split"],
            inputs=[
                io.Int.Input("chunk_length", default=136, min=17, max=100000, step=17,
                             tooltip="Target pixel frames per chunk (at 24 fps). MUST be a multiple of 17 (one keyframe grid step). 136 = ~5.7s, 153 = ~6.4s."),
                io.Int.Input("temporal_overlap", default=17, min=0, max=100000, step=17,
                             tooltip="Pixel frames of overlap between consecutive chunks. MUST be a multiple of 17; recommended 17. Must be smaller than chunk_length."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="How much of the previous chunk's re-sampled boundary the frozen frame-0 anchor keeps: 1.0 = exact content, 0.999 = model default, 0.0 = no anchoring."),
            ],
            outputs=[
                H3_TEMPORAL_PARAM.Output("temporal_split_param",
                                         tooltip="Temporal split settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, chunk_length, temporal_overlap, anchor_strength) -> io.NodeOutput:
        if chunk_length % 17 != 0:
            raise ValueError(f"chunk_length must be a multiple of 17 (the model's keyframe grid step); got {chunk_length}")
        if temporal_overlap % 17 != 0:
            raise ValueError(f"temporal_overlap must be a multiple of 17 (the model's keyframe grid step); got {temporal_overlap}")
        if temporal_overlap >= chunk_length:
            raise ValueError("temporal_overlap must be smaller than chunk_length")
        param = {
            "chunk_length": chunk_length,
            "temporal_overlap": temporal_overlap,
            "anchor_strength": anchor_strength,
        }
        return io.NodeOutput(param)


class MMH3SpatialSplitParams(io.ComfyNode):
    """Bundle the spatial split settings consumed by the Ultimate Upscale node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SpatialSplitParams",
            display_name="MMH3 Spatial Split Params",
            category="model/latent/minimax",
            description=(
                "Bundle the spatial tile settings for the 'MMH3 Ultimate Upscale' "
                "node: tile size, per-axis overlap and fade, and seam stitching "
                "rules (inner loop). Two tile sizing modes: enter explicit pixel "
                "sizes, or enter a row/column count and let the node solve an "
                "equal-size tile grid (all tiles identical, edge tiles included) - "
                "the resolved tile size is exposed as tile_width/tile_height outputs."
            ),
            search_aliases=["h3 spatial params", "spatial split param", "tile param"],
            inputs=[
                io.Int.Input("upscale_width", default=1024, min=32, max=100000, step=32,
                             tooltip="NOT USED. The tile grid is solved against the frame actually produced by the upscale, at sampling time, in every mode - so this widget cannot go stale and does not need to match anything. Kept only so saved workflows keep their widget positions."),
                io.Int.Input("upscale_height", default=1024, min=32, max=100000, step=32,
                             tooltip="NOT USED. See upscale_width. Kept only so saved workflows keep their widget positions."),
                io.Combo.Input("tile_size_mode", options=["specific_size", "rows_cols", "auto"], default="specific_size",
                               tooltip="How the tile size is determined. 'specific_size' (default): use tile_width/tile_height below. 'rows_cols': split the frame into grid_rows x grid_cols EQUAL-SIZE tiles (edge tiles included) - the per-axis overlap is auto-solved so every tile ends up exactly the same size; errors out if the solved tiles would be smaller than min_tile_size. 'auto': solve the grid from token_budget on the REAL frame - the fewest equal tiles whose per-tile forward fits the budget, preferring strips on the long axis. In 'auto' the megapixel dial moves freely and the tile count follows it; grid_rows/grid_cols and tile_width/tile_height are ignored. Both 'rows_cols' and 'auto' resolve at sampling time against the frame actually in hand, so upscale_width/height below are not used."),
                io.Int.Input("tile_width", default=512, min=32, max=100000, step=32,
                             tooltip="[specific_size mode] Tile width in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32."),
                io.Int.Input("tile_height", default=512, min=32, max=100000, step=32,
                             tooltip="[specific_size mode] Tile height in PIXELS at the (upscaled) chunk resolution. Must be a multiple of 32."),
                io.Int.Input("grid_rows", default=2, min=1, max=9, step=1,
                             tooltip="[rows_cols mode] Number of tile ROWS along the height axis (1-9)."),
                io.Int.Input("grid_cols", default=2, min=1, max=9, step=1,
                             tooltip="[rows_cols mode] Number of tile COLUMNS along the width axis (1-9)."),
                io.Int.Input("spatial_w_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Horizontal overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile width. In rows_cols mode this is the DESIRED overlap; the node auto-solves the actual value (multiple of 16px, the H3 latent token) so all tiles stay equal."),
                io.Int.Input("spatial_h_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Vertical overlap in PIXELS between neighbouring tiles. Must be a multiple of 32 and smaller than the tile height. In rows_cols mode this is the DESIRED overlap; the node auto-solves the actual value (multiple of 16px, the H3 latent token) so all tiles stay equal."),
                io.Int.Input("fade_width", default=32, min=0, max=100000, step=32,
                             tooltip="Width in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. The overlap band splits into a FROZEN segment (seam side, mask=0, keeps the neighbour's content) + this FADE segment (interior side). fade_width sets the fade length; the frozen segment takes the rest (overlap - fade). Default 32. Set to 0 to freeze the entire overlap strip. Clamped to the solved overlap in rows_cols mode."),
                io.Int.Input("fade_height", default=32, min=0, max=100000, step=32,
                             tooltip="Height in PIXELS of the FADE segment (mask 0->1) at the interior edge of the overlap band. The overlap band splits into a FROZEN segment (seam side, mask=0, keeps the neighbour's content) + this FADE segment (interior side). fade_height sets the fade length; the frozen segment takes the rest (overlap - fade). Default 32. Set to 0 to freeze the entire overlap strip. Clamped to the solved overlap in rows_cols mode."),
                io.Int.Input("min_tile_size", default=256, min=0, max=100000, step=32,
                             tooltip="Minimum PIXEL size of edge tiles. If a leftover edge tile would be smaller, the last tile is pulled back until it reaches at least this size; the seam overlap then grows and is blended over its full width. 256 (default) keeps small leftover tiles as-is. Must not exceed the tile size. In rows_cols mode an error is raised if the solved tile size falls below this."),
                io.Combo.Input("overlap_mode", options=["earlier", "later"], default="earlier",
                               tooltip="Who wins each shared overlap band when stitching. 'earlier' (default): the already-stitched content wins. 'later': the re-sampled tile wins. Does NOT affect the noise mask."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear",
                               tooltip="How the overlap band transitions when stitching: linear cross-fade (default), smoothstep (eased), overwrite (whole band from the overlap_mode side), midpoint (hard switch at the band's middle)."),
                io.Float.Input("masked_area_noise", default=0.0, min=0.0, max=1.0, step=0.01, round=0.01,
                               tooltip="How much noise is allowed into the masked (frozen/fade) overlap band during sampling. 0 (default): current behaviour, the masked area gets no noise injection and stays fixed. 1.0: the mask has no effect and every tile is sampled freely. Small values like 0.01 let a little noise through to probe how much the scene moves under the mask."),
                io.Boolean.Input("brightness_match", default=False,
                                 tooltip="Per-frame, per-channel median brightness match: after sampling, shift each tile's brightness baseline to the source region at the same (tile, frame). Reduces tile-to-tile and frame-to-frame luminance drift. Experimental - may be removed."),
                io.Combo.Input("dynamic_fade", options=["off", "narrowing", "widening"], default="off",
                               tooltip="Temporal fade schedule over each tile's sampling. 'off' (default): the FADE segment keeps its fixed width for the whole tile (current behaviour). 'narrowing': the fade width starts at fade_width/fade_height and shrinks linearly to dynamic_fade_min by the end of the tile. 'widening': it starts at dynamic_fade_min and grows linearly to fade_width/fade_height. Only takes effect when the tile is sampled in more than one step; if a per-axis fade is not larger than dynamic_fade_min it behaves like 'off'."),
                io.Int.Input("dynamic_fade_min", default=32, min=0, max=100000, step=32,
                             tooltip="Minimum width (PIXELS) the FADE segment can reach when dynamic_fade is 'narrowing' or 'widening'. Only effective when a per-axis fade is larger than this; otherwise the fade stays at its static width. Default 32."),
                io.Boolean.Input("joint_steps", default=True,
                                 tooltip="Step every tile TOGETHER through each sampling step and merge them into one latent, instead of finishing each tile alone and blending the results. On: merge tiles at every step to reduce divergence. Off: sample tiles sequentially. ControlNet, inpainting, and sequential fade/brightness controls use sequential sampling."),
                io.Int.Input("token_budget", default=70000, min=4096, max=1000000, step=1024,
                             tooltip="[auto mode] Largest sequence length allowed in ONE tile forward, in tokens. A tile of W x H pixels over a chunk of N video latent tokens is (W/32) * (H/32) * N. The INT8-Fast kernels fault with an illegal memory access above ~74,898, and attention cost is ~quadratic below it, so this is both the crash line and the speed dial. 70000 leaves a margin under the ceiling; lower it for faster steps and more tiles, raise it for fewer, larger tiles. Ignored in the other two modes."),
            ],
            outputs=[
                H3_SPATIAL_PARAM.Output("spatial_split_param",
                                        tooltip="Spatial split settings consumed by 'MMH3 Ultimate Upscale'."),
                io.Int.Output("tile_width",
                              tooltip="Resolved tile width in PIXELS: the validated input in specific_size mode, or the equal-tile solution computed from upscale_width/grid_cols in rows_cols mode."),
                io.Int.Output("tile_height",
                              tooltip="Resolved tile height in PIXELS: the validated input in specific_size mode, or the equal-tile solution computed from upscale_height/grid_rows in rows_cols mode."),
            ],
        )

    @classmethod
    def execute(cls, upscale_width, upscale_height, tile_size_mode, tile_width,
                tile_height, grid_rows, grid_cols,
                spatial_w_overlap, spatial_h_overlap,
                fade_width, fade_height, min_tile_size, overlap_mode,
                overlap_blend, masked_area_noise=0.0, brightness_match=False,
                dynamic_fade="off", dynamic_fade_min=32, joint_steps=True, token_budget=70000) -> io.NodeOutput:
        if tile_size_mode == "auto":
            # Nothing to solve yet: the grid comes from the frame the upscale
            # actually produces, which does not exist until sampling. Carry the
            # budget through and hand back placeholder tiles that satisfy the
            # downstream sanity checks (overlap < tile) and nothing more.
            for name, v in (("spatial_w_overlap", spatial_w_overlap),
                            ("spatial_h_overlap", spatial_h_overlap),
                            ("min_tile_size", min_tile_size)):
                if v % 32 != 0:
                    raise ValueError(f"'{name}' must be a multiple of 32 pixels (the model's 2x2 latent patch grid); got {v}.")
            ph_w = max(spatial_w_overlap + 32, min_tile_size, 512)
            ph_h = max(spatial_h_overlap + 32, min_tile_size, 512)
            param = {
                "tile_width": ph_w, "tile_height": ph_h,
                "spatial_w_overlap": spatial_w_overlap,
                "spatial_h_overlap": spatial_h_overlap,
                "fade_width": min(fade_width, spatial_w_overlap),
                "fade_height": min(fade_height, spatial_h_overlap),
                "min_tile_size": min_tile_size,
                "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
                "tile_size_mode": tile_size_mode, "joint_steps": joint_steps,
                "token_budget": int(token_budget),
                "masked_area_noise": masked_area_noise, "brightness_match": brightness_match,
                "dynamic_fade": dynamic_fade, "dynamic_fade_min": dynamic_fade_min,
            }
            print(f"[MMH3 Spatial Split Params] auto mode: grid solved at "
                  f"sampling time for <= {int(token_budget)} tokens per tile "
                  f"forward (desired overlap h={spatial_h_overlap} "
                  f"w={spatial_w_overlap}, min tile {min_tile_size}px). "
                  f"tile_width/tile_height outputs are placeholders.")
            return io.NodeOutput(param, ph_w, ph_h)

        if tile_size_mode == "rows_cols":
            # The SHAPE is the setting; the tile size that realises it depends on
            # the frame, which does not exist yet. Solving it here from
            # upscale_width/height made those two widgets a promise about an
            # unseen frame, and a stale promise fails: 3x1 asked for, 7x2
            # delivered, and later a hard stop when the real frame was smaller
            # than the promised tile. So carry the shape and resolve it in
            # spatial_process*(), where the actual chunk is in hand.
            for name, v in (("spatial_w_overlap", spatial_w_overlap),
                            ("spatial_h_overlap", spatial_h_overlap),
                            ("min_tile_size", min_tile_size)):
                if v % 32 != 0:
                    raise ValueError(f"'{name}' must be a multiple of 32 pixels (the model's 2x2 latent patch grid); got {v}.")
            if grid_rows < 1 or grid_cols < 1:
                raise ValueError("grid_rows and grid_cols must be at least 1")
            ph_w = max(spatial_w_overlap + 32, min_tile_size, 512)
            ph_h = max(spatial_h_overlap + 32, min_tile_size, 512)
            param = {
                "tile_width": ph_w, "tile_height": ph_h,
                "spatial_w_overlap": spatial_w_overlap,
                "spatial_h_overlap": spatial_h_overlap,
                "fade_width": min(fade_width, spatial_w_overlap),
                "fade_height": min(fade_height, spatial_h_overlap),
                "min_tile_size": min_tile_size,
                "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
                "tile_size_mode": tile_size_mode, "joint_steps": joint_steps,
                "grid_rows": grid_rows, "grid_cols": grid_cols,
                "token_budget": int(token_budget),
                "masked_area_noise": masked_area_noise, "brightness_match": brightness_match,
                "dynamic_fade": dynamic_fade, "dynamic_fade_min": dynamic_fade_min,
            }
            print(f"[MMH3 Spatial Split Params] rows_cols mode: {grid_rows}x{grid_cols} "
                  f"grid, tiles solved at sampling time against the real frame "
                  f"(desired overlap h={spatial_h_overlap} w={spatial_w_overlap}, "
                  f"min tile {min_tile_size}px). upscale_width/upscale_height are "
                  f"not used; tile_width/tile_height outputs are placeholders.")
            return io.NodeOutput(param, ph_w, ph_h)

        for name, v in (("tile_width", tile_width), ("tile_height", tile_height),
                        ("spatial_w_overlap", spatial_w_overlap), ("spatial_h_overlap", spatial_h_overlap),
                        ("fade_width", fade_width), ("fade_height", fade_height),
                        ("min_tile_size", min_tile_size)):
            if v % 32 != 0:
                raise ValueError(f"'{name}' must be a multiple of 32 pixels (the model's 2x2 latent patch grid); got {v}.")
        if spatial_w_overlap >= tile_width:
            raise ValueError("spatial_w_overlap must be smaller than tile_width")
        if spatial_h_overlap >= tile_height:
            raise ValueError("spatial_h_overlap must be smaller than tile_height")
        if fade_width > spatial_w_overlap:
            raise ValueError("fade_width must not exceed spatial_w_overlap")
        if fade_height > spatial_h_overlap:
            raise ValueError("fade_height must not exceed spatial_h_overlap")
        if min_tile_size > tile_width or min_tile_size > tile_height:
            raise ValueError("min_tile_size must not exceed the tile size")
        param = {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "spatial_w_overlap": spatial_w_overlap,
            "spatial_h_overlap": spatial_h_overlap,
            "fade_width": fade_width,
            "fade_height": fade_height,
            "min_tile_size": min_tile_size,
            "overlap_mode": overlap_mode,
            "overlap_blend": overlap_blend,
            "tile_size_mode": tile_size_mode,
            "joint_steps": joint_steps,
            "token_budget": int(token_budget),
                "masked_area_noise": masked_area_noise, "brightness_match": brightness_match,
                "dynamic_fade": dynamic_fade, "dynamic_fade_min": dynamic_fade_min,
        }
        return io.NodeOutput(param, tile_width, tile_height)


class MMH3FunControlnetParams(io.ComfyNode):
    """Bundle a MiniMax H3 Fun ControlNet plus its control video and strength.

    The `control_video` is the low-resolution video after the ControlNet-Union
    preprocessor, upscaled to the target upscale dimensions (full frame). The
    'MMH3 Ultimate Upscale' node auto-crops it to each chunk's time range and
    each tile's spatial window, so it guides every piece toward the same content
    and keeps high-denoise tiling from drifting away from the source."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3FunControlnetParams",
            display_name="MMH3 Fun Controlnet Params",
            category="model/latent/minimax",
            description=(
                "Bundle Fun ControlNet settings for the 'MMH3 Ultimate Upscale' "
                "node. Provide the ControlNet-Union checkpoint, a VAE (required "
                "to encode the control video into the model's latent space), the "
                "control video at its ORIGINAL low resolution, and the target "
                "upscale size. The main node crops and upscales the control video "
                "per chunk/tile automatically, so each tile is guided toward its "
                "own region of the source video. 'control_upscale_mode' controls "
                "how the low-res control frames are upscaled in batches to trade "
                "peak memory against up-front work."
            ),
            search_aliases=["h3 fun controlnet", "fun controlnet param", "h3 controlnet"],
            inputs=[
                io.ControlNet.Input("control_net", tooltip="The MiniMax H3 Fun ControlNet (load with 'ControlNet Loader'/'DiffControlNet Loader')."),
                io.Vae.Input("vae", tooltip="VAE used to encode the control video into latent space. Required by the MiniMax H3 Fun ControlNet."),
                io.Image.Input("control_video", tooltip="ControlNet-Union control video at its ORIGINAL LOW resolution (the preprocessed low-res video, before upscaling), as a sequence of frames [frames, H, W, C]."),
                io.Int.Input("upscale_width", default=1280, min=32, max=100000, step=32,
                             tooltip="Target upscaled pixel width of the control video. Must match the upscaled generation size (and the width used by the upscale params)."),
                io.Int.Input("upscale_height", default=704, min=32, max=100000, step=32,
                             tooltip="Target upscaled pixel height of the control video. Must match the upscaled generation size (and the height used by the upscale params)."),
                io.Combo.Input("control_upscale_mode", options=["per_chunk", "per_tile", "all"], default="per_chunk",
                               tooltip="How the low-res control frames are upscaled in batches. 'per_chunk' (default): upscale only the current temporal chunk's control frames before processing that chunk. 'per_tile': upscale only the current tile's control crop before sampling that tile - lowest peak memory. 'all': upscale the whole control video up front, then crop during tiling - highest peak memory (the original behaviour)."),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01,
                               tooltip="How strongly the control video guides each piece's generation."),
                io.Combo.Input("start_end_set", options=["percent", "step"], default="percent",
                               tooltip="How the control's active denoising range is expressed. 'percent' (default): fraction of the denoising run (0.0 = very start, 1.0 = very end). 'step': count of sampling steps from the start, expressed as a half-open [start_step, end_step) range. start_step 0 means the control applies from the very first step, and end_step is exclusive, so end_step 1 means it applies for the first step only and later steps generate freely."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True,
                               tooltip="[percent mode] Denoising step fraction at which the control starts taking effect. Ignored in step mode."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True,
                               tooltip="[percent mode] Denoising step fraction at which the control stops taking effect. Ignored in step mode."),
                io.Int.Input("start_step", default=0, min=0, max=100000, step=1, advanced=True,
                             tooltip="[step mode] Sampling step at which the control starts taking effect (0 = the very first step). Ignored in percent mode."),
                io.Int.Input("end_step", default=100000, min=0, max=100000, step=1, advanced=True,
                             tooltip="[step mode] Exclusive upper step bound: the control applies for steps from start_step up to but NOT including end_step. So start 0 / end 1 gates only the first step, and end 2 gates the first two steps; later steps generate freely. Clamped to the piece's total step count. Ignored in percent mode."),
            ],
            outputs=[
                H3_FUN_CONTROL_PARAM.Output("fun_control_param",
                                            tooltip="Fun ControlNet settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, control_net, vae, control_video, upscale_width, upscale_height,
                control_upscale_mode, strength, start_end_set, start_percent, end_percent,
                start_step, end_step) -> io.NodeOutput:
        if not isinstance(control_net, comfy.controlnet.MiniMaxH3ControlNet):
            raise ValueError("MMH3FunControlnetParams needs a MiniMax H3 Fun ControlNet")
        upscale_width = int(round(upscale_width / 32.0)) * 32
        upscale_height = int(round(upscale_height / 32.0)) * 32
        param = {
            "control_net": control_net,
            "vae": vae,
            "control_video": control_video.movedim(-1, 1).contiguous(),  # [T, C, H, W] at low res
            "upscale_width": upscale_width,
            "upscale_height": upscale_height,
            "control_upscale_mode": control_upscale_mode,
            "strength": strength,
            "start_end_set": start_end_set,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "start_step": start_step,
            "end_step": end_step,
        }
        return io.NodeOutput(param)


class MMH3SpatialInpaintParams(io.ComfyNode):
    """Bundle a MiniMax H3 Fun ControlNet configured for spatial-seam inpaint.

    When consumed by 'MMH3 Ultimate Upscale' together with a spatial split, the
    overlap strips of every non-first tile are pinned to the already-sampled
    neighbour content in latent space: the accumulated latent's kept strips feed
    the controlnet's masked_latent channel and the spatial fade mask doubles as
    the visibility channel (1 = keep the neighbour, 0 = regenerate). This removes
    the visible stitch seams left by high-denoise tiling without a pixel
    decode/re-encode round-trip. The inpaint controlnet is chained under the Fun
    ControlNet so the two guides stack."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SpatialInpaintParams",
            display_name="MMH3 Fun Controlnet Inpaint",
            category="model/latent/minimax",
            description=(
                "Bundle Fun ControlNet inpaint settings for the 'MMH3 Ultimate "
                "Upscale' node's spatial split. On every non-first tile the "
                "overlap strips are pinned to the already-sampled neighbour "
                "content directly in latent space: the accumulated latent's "
                "kept strips feed the controlnet's masked_latent channel and the "
                "spatial fade mask doubles as the visibility channel, so no pixel "
                "decode/re-encode round-trip is needed. Eliminates the visible "
                "seam between tiles. Requires the spatial split to be active; "
                "stackable with the Fun ControlNet params."
            ),
            search_aliases=["h3 spatial inpaint", "inpaint param", "h3 seam inpaint"],
            inputs=[
                io.ControlNet.Input("control_net", tooltip="The MiniMax H3 Fun ControlNet used to apply the inpaint guidance."),
                io.Float.Input("inpaint_strength", default=1.0, min=0.0, max=10.0, step=0.01,
                               tooltip="How strongly the inpaint strips pin the tile to the neighbour content."),
                io.Combo.Input("start_end_set", options=["percent", "step"], default="percent",
                               tooltip="How the inpaint guidance's active denoising range is expressed. 'percent' (default): fraction of the denoising run (0.0 = very start, 1.0 = very end). 'step': count of sampling steps from the start, expressed as a half-open [start_step, end_step) range. start_step 0 means the guidance applies from the very first step and end_step is exclusive, so end_step 1 means it applies for the first step only and later steps re-generate the seams freely."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="[percent mode] Denoising fraction at which the inpaint guidance starts applying (0.0 = the start of denoising). Ignored in step mode."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="[percent mode] Denoising fraction at which the inpaint guidance stops applying (1.0 = the end of denoising). Ignored in step mode."),
                io.Int.Input("start_step", default=0, min=0, max=100000, step=1, advanced=True,
                             tooltip="[step mode] Sampling step at which the inpaint guidance starts applying (0 = the very first step). Ignored in percent mode."),
                io.Int.Input("end_step", default=100000, min=0, max=100000, step=1, advanced=True,
                             tooltip="[step mode] Exclusive upper step bound: the guidance applies for steps from start_step up to but NOT including end_step. So start 0 / end 1 gates only the first step, and end 2 gates the first two steps; later steps re-generate the seams freely. Clamped to the piece's total step count. Ignored in percent mode."),
            ],
            outputs=[
                H3_INPAINT_PARAM.Output("inpaint_param",
                                        tooltip="Spatial seam inpaint settings consumed by 'MMH3 Ultimate Upscale'."),
            ],
        )

    @classmethod
    def execute(cls, control_net, inpaint_strength, start_end_set, start_percent, end_percent,
                start_step, end_step) -> io.NodeOutput:
        if not isinstance(control_net, comfy.controlnet.MiniMaxH3ControlNet):
            raise ValueError("MMH3SpatialInpaintParams needs a MiniMax H3 Fun ControlNet")
        return io.NodeOutput({
            "control_net": control_net,
            "strength": inpaint_strength,
            "start_end_set": start_end_set,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "start_step": start_step,
            "end_step": end_step,
        })


# ---------------------------------------------------------------------------
# main node
# ---------------------------------------------------------------------------

class MMH3UltimateUpscale(io.ComfyNode):
    """One node for the full latent re-enhancement pipeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3UltimateUpscale",
            display_name="MMH3 Ultimate Upscale",
            category="model/latent/minimax",
            description=(
                "Re-sample an already-denoised MiniMax H3 AV latent through the full "
                "auto pipeline in one node: temporal split (outer loop) -> latent "
                "upscale (per chunk) -> spatial split (inner loop) -> per-tile "
                "sampling with preview -> spatial stitch -> temporal stitch. Each "
                "chunk/tile is sampled with a fresh guider built from the per-piece "
                "conditioning (re-anchored and cropped keyframes), keeping peak VRAM "
                "to one tile. 'latent_upscale_param', 'temporal_split_param' and "
                "'spatial_split_param' are optional - leave any unconnected to skip "
                "that stage (no upscale / single chunk / full-chunk sampling)."
            ),
            search_aliases=["h3 ultimate upscale", "ultimate upscale", "h3 auto upscale", "h3 enhance"],
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model used to re-sample every chunk/tile (guider is built internally)."),
                io.Conditioning.Input("conditioning",
                                      tooltip="Conditioning used to generate this latent. Per chunk it is re-anchored in time; per tile its keyframes are spatially cropped; the frame-0 keyframe is pinned to the previous chunk's re-sampled frame."),
                io.Latent.Input("latent", tooltip="Denoised MiniMax H3 AV latent to enhance."),
                io.Noise.Input("noise", tooltip="Noise source; one noise tensor is generated per piece."),
                io.Sampler.Input("sampler", tooltip="Sampler used for every chunk/tile."),
                io.Sigmas.Input("sigmas", tooltip="Sigma schedule used for every chunk/tile."),
                io.Conditioning.Input("negative", optional=True,
                                      tooltip="Negative conditioning. When connected, a CFGGuider is used with the 'cfg' value; otherwise a basic guider (positive only)."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="CFG scale used when 'negative' is connected."),
                H3_UPSCALE_PARAM.Input("latent_upscale_param", optional=True,
                                       tooltip="Output of 'MMH3 Latent Upscale with Model Params' (H3 3D upscaler) OR 'MMH3 Latent Upscale Params' (model-free interpolation). Leave unconnected to skip upscaling."),
                H3_TEMPORAL_PARAM.Input("temporal_split_param", optional=True,
                                        tooltip="Output of 'MMH3 Temporal Split Params'. Leave unconnected to process the latent as a single chunk."),
                H3_SPATIAL_PARAM.Input("spatial_split_param", optional=True,
                                       tooltip="Output of 'MMH3 Spatial Split Params'. Leave unconnected to sample each chunk whole (no tiling)."),
                H3_FUN_CONTROL_PARAM.Input("fun_control_param", optional=True,
                                           tooltip="Output of 'MMH3 Fun Controlnet Params'. When connected, every chunk/tile is sampled with a Fun ControlNet that guides the re-generation toward the provided control video (auto-cropped to each chunk's time range and tile's spatial window, and upscaled from low-res per the selected mode). Leave unconnected to disable."),
                H3_INPAINT_PARAM.Input("inpaint_param", optional=True,
                                       tooltip="Output of 'MMH3 Spatial Inpaint Params'. When connected with a spatial split, every non-first tile's overlap strips are pinned to the already-sampled neighbour content (VAE-decoded source + spatial fade mask), removing visible tile seams. Leave unconnected to disable."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Upscaled, re-sampled, stitched MiniMax H3 AV latent."),
                io.Dict.Output("segments_info",
                               tooltip="DEBUG ONLY. Per-chunk metadata: frame start/count, video/audio token ranges, upscale applied."),
                io.Dict.Output("tiles_info",
                               tooltip="DEBUG ONLY. Per-chunk spatial grid metadata: offsets, tile extents, overlaps, stitching mode."),
            ],
        )

    @classmethod
    def execute(cls, latent, conditioning, model, noise, sampler, sigmas,
                negative=None, cfg=1.0,
                temporal_split_param=None, spatial_split_param=None,
                latent_upscale_param=None, fun_control_param=None,
                inpaint_param=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("MMH3UltimateUpscale expects a MiniMax H3 AV latent (nested video [B,24,T,H,W] + audio [B,32,2,T])")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("MMH3UltimateUpscale expects a single-video latent (batch 1)")

        # keep the H3 packed layout and cond row counts in lockstep (refs metadata
        # rewritten from the real latents; phantom latent-less visual refs dropped)
        conditioning = normalize_minimax_refs(conditioning)

        # fail early if the upscale target is smaller than the spatial tile size;
        # tiles can never cover a chunk smaller than one tile, which would only
        # surface as a confusing error during the sampling/stitching phase.
        if (latent_upscale_param is not None and spatial_split_param is not None
                and spatial_split_param.get("tile_size_mode") not in ("rows_cols", "auto")):
            up_w = int(latent_upscale_param["width"])
            up_h = int(latent_upscale_param["height"])
            tile_w = int(spatial_split_param["tile_width"])
            tile_h = int(spatial_split_param["tile_height"])
            if up_w < tile_w:
                raise ValueError(
                    f"Upscale width ({up_w}) must be >= tile_width ({tile_w})"
                )
            if up_h < tile_h:
                raise ValueError(
                    f"Upscale height ({up_h}) must be >= tile_height ({tile_h})"
                )

        tv = video.shape[2]
        ta = audio.shape[-1]

        if temporal_split_param is not None:
            chunk_length = int(temporal_split_param["chunk_length"])
            overlap = int(temporal_split_param["temporal_overlap"])
            bounds, frame_count = compute_segments(tv, chunk_length, overlap)
            anchor_strength = temporal_split_param["anchor_strength"]
        else:
            frame_count = frames_for_tokens(tv)
            bounds = [(0, 0, tv, frame_count)]
            anchor_strength = 0.999

        acc_v = None
        acc_a = None
        segments_debug = []
        tiles_debug = []

        # 'all' upscale mode: materialize the whole upscaled control video once so
        # every chunk crops from it; the other modes keep the low-res control and
        # upscale only per chunk / per tile to limit peak memory.
        fc_buffer = None
        if fun_control_param is not None and fun_control_param["control_upscale_mode"] == "all":
            fc_buffer = _resize_images(fun_control_param["control_video"],
                                       fun_control_param["upscale_height"],
                                       fun_control_param["upscale_width"])

        # High-VRAM fast path: when offload_model is OFF the H3 upscaler stays
        # resident on the device for the whole pass (no per-chunk unload/reload),
        # and the diffusion model is not offloaded during upscale either. The
        # upscaler is then unloaded once at the end. Only the model-based upscaler
        # node carries 'offload_model'; model-free interpolation has no model.
        keep_upscaler_resident = (latent_upscale_param is not None
                                  and "model_name" in latent_upscale_param
                                  and not latent_upscale_param.get("offload_model", True))

        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            a0, a1 = audio_range(f0, f1)
            a1 = min(a1, ta)
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            # 1. upscale this chunk's video (audio untouched). While the 3D upscaler
            #    is on the GPU the diffusion model isn't needed, so offload it first
            #    to avoid H3 + upscaler resident simultaneously; the next sample
            #    reloads H3 automatically.
            upscaled = False
            if latent_upscale_param is not None:
                use_model = "model_name" in latent_upscale_param
                if (use_model and not keep_upscaler_resident
                        and str(latent_upscale_param["device"]) == "cuda" and hasattr(model, "clone_base_uuid")):
                    # the 3D upscaler is on the GPU during upscale; offload the
                    # diffusion model so they don't reside simultaneously
                    comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
                    comfy.model_management.soft_empty_cache()
                chunk_v, _, _ = upscale_latent(chunk_v, latent_upscale_param)
                upscaled = True

            # 2. time re-anchor; keyframe video latents are always resized to the
            #    (possibly upscaled) chunk size - the H3 packed layout requires
            #    keyframes on the sampled target's spatial grid, and in the intended
            #    workflow the conditioning is generated at the upscaled size already
            cond_i = reanchor_conditioning(conditioning, f0, f1, (chunk_v.shape[3], chunk_v.shape[4]))

            # 3. pin frame-0 keyframe to the previous chunk's re-sampled frame
            if i > 0 and acc_v is not None:
                cond_i = anchor_conditioning(cond_i, acc_v, f0, anchor_strength)

            # 4. inner loop: spatial split -> sample -> stitch
            fun_control = None
            if fun_control_param is not None:
                chunk_ctrl, fc_mode = _chunk_fun_control(fun_control_param, f0, f1, fc_buffer)
                fun_control = dict(fun_control_param)
                fun_control["chunk_ctrl"] = chunk_ctrl
                fun_control["mode"] = fc_mode
            if spatial_split_param is not None:
                # These upstream controls operate on sequential tiles and cannot be
                # represented by the LD joint sampler; retain their behavior.
                sequential_controls = (fun_control is not None or inpaint_param is not None
                    or spatial_split_param.get("masked_area_noise", 0.0) != 0.0
                    or spatial_split_param.get("brightness_match", False)
                    or spatial_split_param.get("dynamic_fade", "off") != "off")
                joint = spatial_split_param.get("joint_steps", True) and not sequential_controls
                if spatial_split_param.get("joint_steps", True) and sequential_controls:
                    print("[MMH3] Using sequential tiles for ControlNet, inpaint, or sequential fade/brightness controls")
                if joint:
                    chunk_out_v, tile_info = spatial_process_joint(
                        chunk_v, chunk_a, cond_i, spatial_split_param,
                        model, noise, sampler, sigmas, negative, cfg)
                else:
                    chunk_out_v, tile_info = spatial_process(
                        chunk_v, chunk_a, cond_i, spatial_split_param,
                        model, noise, sampler, sigmas, negative, cfg,
                        fun_control=fun_control, inpaint=inpaint_param)
                tile_info = dict(tile_info)
                tile_info["chunk"] = i
                tiles_debug.append(tile_info)
            else:
                if fun_control is not None:
                    # whole-chunk path: the control video covers the full frame, so
                    # just upscale the low-res chunk to the target size (per_tile) or
                    # use the already-upscaled chunk, then drive a fresh controlnet copy
                    if fun_control["mode"] == "per_tile":
                        chunk_hint = _resize_images(fun_control["chunk_ctrl"],
                                                    fun_control["upscale_height"],
                                                    fun_control["upscale_width"])
                    else:
                        chunk_hint = fun_control["chunk_ctrl"]
                    cond_i = inject_fun_control(
                        cond_i, fun_control["control_net"], fun_control["vae"], chunk_hint,
                        fun_control["strength"], sigmas, fun_control,
                        getattr(model, "model_sampling", None))
                piece = {"samples": comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))}
                out = sample_piece(piece, cond_i, model, noise, sampler, sigmas, negative, cfg)
                chunk_out_v = out.tensors[0]

            # 5. temporal stitch
            acc_v, acc_a = temporal_append(acc_v, acc_a, chunk_out_v, chunk_a, i, k0, f0)

            segments_debug.append({
                "chunk": i,
                "frame_start": f0,
                "frame_count": f1 - f0,
                "video_tokens": [k0, k1],
                "audio_tokens": list(audio_range(f0, f1)),
                "upscaled": upscaled,
                "spatial_h": chunk_v.shape[3],
                "spatial_w": chunk_v.shape[4],
            })

        # all chunks sampled & stitched: the diffusion model is no longer needed,
        # unload it so the caller (e.g. VAE decode of the large latent) gets the VRAM
        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()

        # If the upscaler was kept resident for the full pass, unload it once now
        if keep_upscaler_resident and latent_upscale_param is not None:
            unload_upscale_model(
                latent_upscale_param["model_name"],
                latent_upscale_param["device"],
                latent_upscale_param["precision"],
            )

        out = {"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))}
        return io.NodeOutput(out, segments_debug, tiles_debug)
