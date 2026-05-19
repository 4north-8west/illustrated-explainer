// Transition + inline-emphasis presets.
import { animate, cubicBezier, createTimeline } from '../anime.esm.min.js';
import { registerPreset } from './index.js';
import { scaled } from '../defaults.js';

// --- inline emphasis -------------------------------------------------------

registerPreset('highlight-mark', (el, opts = {}) => {
  // Wrap content in an inner span so the highlighter background applies to
  // the text directly, not to a container whose own `background:` shorthand
  // (cards, demo cells, etc.) would override the linear-gradient.
  const inner = document.createElement('span');
  inner.classList.add('anime-highlight');
  inner.style.display = 'inline';
  inner.innerHTML = el.innerHTML;
  el.innerHTML = '';
  el.appendChild(inner);
  return animate(inner, {
    backgroundSize: ['0% 100%', '100% 100%'],
    duration: opts.duration ?? scaled(700),
    ease: cubicBezier(0.65, 0, 0.35, 1),
  });
}, { kind: 'transition', intensity: 'low' });

registerPreset('underline-draw', (el, opts = {}) => {
  el.classList.add('anime-underline');
  // Inject a real underline span we can animate (the ::after pseudo isn't queryable in tests).
  const u = document.createElement('span');
  u.style.position = 'absolute';
  u.style.left = '0';
  u.style.right = '0';
  u.style.bottom = '-2px';
  u.style.height = '2px';
  u.style.background = 'var(--anime-gold, #DAA900)';
  u.style.transformOrigin = 'left center';
  u.style.transform = 'scaleX(0)';
  el.appendChild(u);
  return animate(u, {
    scaleX: [0, 1],
    duration: opts.duration ?? scaled(600),
    ease: cubicBezier(0.65, 0, 0.35, 1),
  });
}, { kind: 'transition', intensity: 'low' });

registerPreset('box-reveal', (el, opts = {}) => {
  el.style.position = el.style.position || 'relative';
  const box = document.createElement('span');
  box.style.position = 'absolute';
  box.style.inset = '-4px';
  box.style.border = '2px solid var(--anime-gold, #DAA900)';
  box.style.pointerEvents = 'none';
  box.style.clipPath = 'inset(0 100% 0 0)';
  el.appendChild(box);
  return animate(box, {
    clipPath: ['inset(0 100% 0 0)', 'inset(0 0 0 0)'],
    duration: opts.duration ?? scaled(700),
    ease: cubicBezier(0.65, 0, 0.35, 1),
  });
}, { kind: 'transition', intensity: 'low' });

registerPreset('weight-shift', (el, opts = {}) => {
  return animate(el, {
    fontWeight: [opts.from ?? 400, opts.to ?? 700],
    duration: opts.duration ?? scaled(700),
    ease: cubicBezier(0.16, 1, 0.3, 1),
  });
}, { kind: 'transition', intensity: 'low' });

// --- state-change ----------------------------------------------------------

registerPreset('morph-text', (el, opts = {}) => {
  const from = el.textContent;
  const to = opts.to ?? from;
  const len = Math.max(from.length, to.length);
  const obj = { p: 0 };
  return animate(obj, {
    p: 1,
    duration: opts.duration ?? scaled(900),
    ease: cubicBezier(0.65, 0, 0.35, 1),
    onUpdate: () => {
      const settled = Math.floor(obj.p * len);
      let out = '';
      for (let i = 0; i < len; i++) {
        if (i < settled) out += to[i] ?? '';
        else if (i < from.length) out += from[i];
        else out += '';
      }
      el.textContent = out;
      if (obj.p >= 1) el.textContent = to;
    },
    onComplete: () => { el.textContent = to; },
  });
}, { kind: 'transition', intensity: 'medium' });

registerPreset('flip-swap', (el, opts = {}) => {
  const target = opts.to ?? el.textContent;
  el.style.display = 'inline-block';
  el.style.transformStyle = 'preserve-3d';
  const half = (opts.duration ?? scaled(600)) / 2;
  const tl = createTimeline({
    defaults: {
      duration: half,
      ease: cubicBezier(0.65, 0, 0.35, 1),
    },
    // Top-level onComplete fires when seek() lands on the end (test path)
    // and after the second segment finishes (real playback). Either way,
    // text is final-state by the time the timeline ends.
    onComplete: () => { el.textContent = target; },
  });
  tl.add(el, {
    rotateX: [0, -90],
    opacity: [1, 0],
    // Per-segment onComplete fires at the rotation midpoint during real
    // playback so the text swaps while the element is edge-on (invisible).
    onComplete: () => { el.textContent = target; },
  });
  tl.add(el, {
    rotateX: [90, 0],
    opacity: [0, 1],
  });
  return tl;
}, { kind: 'transition', intensity: 'medium' });
