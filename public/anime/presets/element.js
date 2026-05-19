import { animate, stagger, cubicBezier } from '../anime.esm.min.js';
import { registerPreset } from './index.js';
import { scaled } from '../defaults.js';

const EASE = cubicBezier(0.65, 0, 0.35, 1);

registerPreset('draw-svg', (el, opts = {}) => {
  // Expects an inline <svg> child with <text> elements that have stroke set in CSS,
  // or <path> elements. We compute path length with getTotalLength() directly.
  const paths = el.querySelectorAll('svg text, svg path');
  const targets = [];
  paths.forEach(p => {
    let len = 200;
    try {
      if (p.getTotalLength) len = p.getTotalLength();
    } catch { /* unsupported in test env — fine */ }
    p.style.strokeDasharray = String(len);
    p.style.strokeDashoffset = String(len);
    targets.push(p);
  });
  return animate(targets, {
    strokeDashoffset: [(_t, i) => {
      const t = targets[i];
      try { return t.getTotalLength ? t.getTotalLength() : 200; } catch { return 200; }
    }, 0],
    fillOpacity: [0, 1],
    duration: opts.duration ?? scaled(1400),
    ease: EASE,
    delay: stagger(opts.stagger ?? scaled(100)),
  });
}, { kind: 'element', intensity: 'medium' });

registerPreset('gradient-sweep', (el, opts = {}) => {
  // Wrap content in an inner span and apply the gradient to that, so the
  // outer element's background (e.g. card styling, demo container) doesn't
  // override the text-clip gradient via CSS shorthand specificity.
  const inner = document.createElement('span');
  inner.classList.add('anime-gradient-text');
  inner.style.display = 'inline-block';
  inner.innerHTML = el.innerHTML;
  el.innerHTML = '';
  el.appendChild(inner);
  return animate(inner, {
    backgroundPositionX: ['100%', '-100%'],
    duration: opts.duration ?? scaled(1600),
    ease: EASE,
  });
}, { kind: 'element', intensity: 'medium' });

registerPreset('shimmer', (el, opts = {}) => {
  // One-shot diagonal light pass via an injected overlay span.
  const overlay = document.createElement('span');
  overlay.style.position = 'absolute';
  overlay.style.inset = '0';
  overlay.style.background = 'linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.5) 50%, transparent 70%)';
  overlay.style.transform = 'translateX(-100%)';
  overlay.style.pointerEvents = 'none';
  el.style.position = el.style.position || 'relative';
  el.appendChild(overlay);
  return animate(overlay, {
    translateX: ['-100%', '100%'],
    duration: opts.duration ?? scaled(1100),
    ease: EASE,
    onComplete: () => overlay.remove(),
  });
}, { kind: 'element', intensity: 'low' });

registerPreset('text-mask-reveal', (el, opts = {}) => {
  // Wrap the content in an inner positioned span so the mask overlay sits
  // tight on the text, not on whatever container the preset is invoked on.
  const inner = document.createElement('span');
  inner.style.position = 'relative';
  inner.style.display = 'inline-block';
  inner.style.overflow = 'hidden';
  inner.innerHTML = el.innerHTML;
  el.innerHTML = '';
  el.appendChild(inner);
  const mask = document.createElement('span');
  mask.style.position = 'absolute';
  mask.style.inset = '0';
  mask.style.background = 'var(--anime-navy, #002856)';
  mask.style.transformOrigin = 'right center';
  inner.appendChild(mask);
  return animate(mask, {
    scaleX: [1, 0],
    duration: opts.duration ?? scaled(700),
    ease: EASE,
    onComplete: () => mask.remove(),
  });
}, { kind: 'element', intensity: 'low' });

registerPreset('chromatic-split', (el, opts = {}) => {
  return animate(el, {
    textShadow: [
      { to: '0 0 0 transparent' },
      { to: '-3px 0 0 #f33, 3px 0 0 #3cf' },
      { to: '0 0 0 transparent' },
    ],
    duration: opts.duration ?? scaled(700),
    ease: cubicBezier(0.16, 1, 0.3, 1),
  });
}, { kind: 'element', intensity: 'high' });

registerPreset('fade-up', (el, opts = {}) => {
  return animate(el, {
    opacity: [0, 1],
    translateY: [12, 0],
    duration: opts.duration ?? scaled(600),
    ease: cubicBezier(0.16, 1, 0.3, 1),
  });
}, { kind: 'element', intensity: 'low' });
