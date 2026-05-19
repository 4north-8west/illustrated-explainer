/**
 * Global pacing defaults for the preset catalog.
 *
 * Every preset multiplies its base timing through `scaled()`. Adjusting
 * BASE_SCALE here uniformly speeds up or slows down the whole library
 * without touching individual presets.
 *
 *   BASE_SCALE = 1.0   → original anime.js-typical "snappy" pace
 *   BASE_SCALE = 1.6   → presentation pace (current default — readable for audiences)
 *   BASE_SCALE = 2.0   → deliberate, almost theatrical
 *   BASE_SCALE = 0.7   → fast UI feedback
 *
 * This is a *default* multiplier. Per-call `opts.duration` / `opts.stagger`
 * still override the baseline absolutely. The gallery's global sliders also
 * override per-replay.
 */
export const BASE_SCALE = 1.6;

export function scaled(ms) {
  return Math.round(ms * BASE_SCALE);
}
