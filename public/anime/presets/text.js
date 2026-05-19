import { animate, stagger, cubicBezier, eases } from '../anime.esm.min.js';
import { registerPreset } from './index.js';
import { scaled } from '../defaults.js';

const EASE = cubicBezier(0.16, 1, 0.3, 1);

// --- helpers ---------------------------------------------------------------

function splitWords(el) {
  const text = el.textContent;
  const words = text.split(/(\s+)/); // keep whitespace
  el.textContent = '';
  const spans = [];
  for (const w of words) {
    if (/^\s+$/.test(w)) {
      el.appendChild(document.createTextNode(w));
    } else if (w.length) {
      const s = document.createElement('span');
      s.className = 'anime-word';
      s.textContent = w;
      el.appendChild(s);
      spans.push(s);
    }
  }
  return spans;
}

function splitChars(el) {
  const text = el.textContent;
  el.textContent = '';
  const spans = [];
  for (const ch of text) {
    const s = document.createElement('span');
    s.className = 'anime-char';
    s.textContent = ch;
    if (ch === ' ') s.style.whiteSpace = 'pre';
    el.appendChild(s);
    spans.push(s);
  }
  return spans;
}

function splitLines(el) {
  const html = el.innerHTML;
  const parts = html.split(/<br\s*\/?>/i);
  el.innerHTML = '';
  const lines = [];
  for (const p of parts) {
    const span = document.createElement('span');
    span.className = 'anime-line';
    span.innerHTML = p;
    el.appendChild(span);
    lines.push(span);
  }
  return lines;
}

// --- presets ---------------------------------------------------------------

registerPreset('stagger-words', (el, opts = {}) => {
  const spans = splitWords(el);
  return animate(spans, {
    opacity: [0, 1],
    translateY: [12, 0],
    duration: opts.duration ?? scaled(600),
    delay: stagger(opts.stagger ?? scaled(60), { start: opts.delay ?? 0 }),
    ease: EASE,
  });
}, { kind: 'text', intensity: 'low' });

registerPreset('stagger-chars', (el, opts = {}) => {
  const spans = splitChars(el);
  return animate(spans, {
    opacity: [0, 1],
    translateY: [8, 0],
    duration: opts.duration ?? scaled(500),
    delay: stagger(opts.stagger ?? scaled(22), { start: opts.delay ?? 0 }),
    ease: EASE,
  });
}, { kind: 'text', intensity: 'low' });

registerPreset('wave-chars', (el, opts = {}) => {
  const spans = splitChars(el);
  return animate(spans, {
    translateY: [
      { to: -10, duration: scaled(300) },
      { to: 0, duration: scaled(300) },
    ],
    opacity: [0, 1],
    delay: stagger(opts.stagger ?? scaled(40), { start: opts.delay ?? 0 }),
    ease: EASE,
  });
}, { kind: 'text', intensity: 'medium' });

registerPreset('cascade-lines', (el, opts = {}) => {
  const lines = splitLines(el);
  // Wrap each line's content in an inner span we can translate without affecting overflow:hidden parent
  for (const line of lines) {
    const inner = document.createElement('span');
    inner.style.display = 'inline-block';
    inner.innerHTML = line.innerHTML;
    line.innerHTML = '';
    line.appendChild(inner);
  }
  return animate(Array.from(el.querySelectorAll('span.anime-line > span')), {
    translateY: ['100%', 0],
    opacity: [0, 1],
    duration: opts.duration ?? scaled(600),
    delay: stagger(opts.stagger ?? scaled(80), { start: opts.delay ?? 0 }),
    ease: EASE,
  });
}, { kind: 'text', intensity: 'low' });

registerPreset('random-chars', (el, opts = {}) => {
  const spans = splitChars(el);
  const order = spans.map((_, i) => i).sort(() => Math.random() - 0.5);
  return animate(spans, {
    opacity: [0, 1],
    scale: [0.8, 1],
    duration: opts.duration ?? scaled(500),
    delay: (_t, i) => (opts.delay ?? 0) + order[i] * (opts.stagger ?? scaled(30)),
    ease: EASE,
  });
}, { kind: 'text', intensity: 'medium' });

// --- text-as-content presets ----------------------------------------------

const SCRAMBLE_GLYPHS = '!<>-_\\/[]{}—=+*^?#';

registerPreset('typewriter', (el, opts = {}) => {
  const target = el.textContent;
  el.dataset.targetText = target;
  el.textContent = '';
  const caret = document.createElement('span');
  caret.textContent = '​'; // ZWSP placeholder
  el.appendChild(caret);
  const obj = { i: 0 };
  return animate(obj, {
    i: target.length,
    round: 1,
    duration: opts.duration ?? scaled(target.length * 60),
    ease: 'linear',
    onUpdate: () => {
      el.textContent = target.slice(0, obj.i) + '​';
    },
  });
}, { kind: 'text', intensity: 'low' });

registerPreset('count-up', (el, opts = {}) => {
  const from = opts.from ?? 0;
  const to = opts.to ?? (parseInt(el.textContent, 10) || 0);
  const decimals = Number.isFinite(opts.decimals) ? opts.decimals : 0;
  const pad = Number.isFinite(opts.pad) ? opts.pad : 0;
  const thousands = opts.thousands === true || opts.thousands === 'true' || opts.thousands === 1;
  const prefix = opts.prefix ?? '';
  const suffix = opts.suffix ?? '';
  const factor = Math.pow(10, decimals);
  const obj = { v: from * factor };
  function format(raw) {
    let n = (raw / factor).toFixed(decimals);
    if (thousands) {
      const parts = n.split('.');
      parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      n = parts.join('.');
    }
    if (pad > 0) {
      const [intPart, fracPart] = n.split('.');
      const padded = intPart.padStart(pad, '0');
      n = fracPart !== undefined ? `${padded}.${fracPart}` : padded;
    }
    return `${prefix}${n}${suffix}`;
  }
  return animate(obj, {
    v: to * factor,
    round: 1,
    duration: opts.duration ?? scaled(2400),
    ease: cubicBezier(0.16, 1, 0.3, 1),
    onUpdate: () => { el.textContent = format(obj.v); },
    onComplete: () => { el.textContent = format(to * factor); },
  });
}, { kind: 'text', intensity: 'low' });

registerPreset('scramble', (el, opts = {}) => {
  const target = el.textContent;
  const duration = opts.duration ?? Math.max(scaled(600), scaled(target.length * 30));
  const obj = { p: 0 };
  return animate(obj, {
    p: 1,
    duration,
    ease: eases.inOutQuad,
    onUpdate: () => {
      let out = '';
      const settled = Math.floor(obj.p * target.length);
      for (let i = 0; i < target.length; i++) {
        if (i < settled) out += target[i];
        else if (target[i] === ' ') out += ' ';
        else out += SCRAMBLE_GLYPHS[Math.floor(Math.random() * SCRAMBLE_GLYPHS.length)];
      }
      el.textContent = out;
    },
  });
}, { kind: 'text', intensity: 'medium' });

registerPreset('decode', (el, opts = {}) => {
  const target = el.textContent;
  const duration = opts.duration ?? Math.max(scaled(800), scaled(target.length * 40));
  const obj = { p: 0 };
  return animate(obj, {
    p: 1,
    duration,
    ease: 'linear',
    onUpdate: () => {
      let out = '';
      const settled = Math.floor(obj.p * target.length);
      for (let i = 0; i < target.length; i++) {
        if (i < settled) out += target[i];
        else if (target[i] === ' ') out += ' ';
        else out += String.fromCharCode(33 + Math.floor(Math.random() * 94));
      }
      el.textContent = out;
    },
  });
}, { kind: 'text', intensity: 'medium' });

registerPreset('rolling-numbers', (el, opts = {}) => {
  const target = el.textContent;
  el.textContent = '';
  el.style.display = 'inline-flex';
  const slots = [];
  for (const ch of target) {
    if (/\d/.test(ch)) {
      const slot = document.createElement('span');
      slot.style.display = 'inline-block';
      slot.style.height = '1em';
      slot.style.overflow = 'hidden';
      slot.style.lineHeight = '1em';
      const reel = document.createElement('span');
      reel.style.display = 'block';
      for (let n = 0; n <= 9; n++) {
        const d = document.createElement('span');
        d.style.display = 'block';
        d.style.height = '1em';
        d.textContent = String(n);
        reel.appendChild(d);
      }
      slot.appendChild(reel);
      el.appendChild(slot);
      slots.push({ reel, target: parseInt(ch, 10) });
    } else {
      const txt = document.createElement('span');
      txt.textContent = ch;
      el.appendChild(txt);
    }
  }
  return animate(slots.map(s => s.reel), {
    translateY: (_t, i) => `-${slots[i].target}em`,
    duration: opts.duration ?? scaled(1200),
    delay: stagger(opts.stagger ?? scaled(80)),
    ease: cubicBezier(0.16, 1, 0.3, 1),
  });
}, { kind: 'text', intensity: 'medium' });

registerPreset('split-flap', (el, opts = {}) => {
  const target = el.textContent;
  el.textContent = '';
  el.style.display = 'inline-flex';
  el.style.gap = '0.05em';
  const cards = [];
  for (const ch of target) {
    const card = document.createElement('span');
    card.style.display = 'inline-block';
    card.style.transformStyle = 'preserve-3d';
    card.style.perspective = '400px';
    card.textContent = ch;
    card.style.opacity = '0';
    el.appendChild(card);
    cards.push(card);
  }
  return animate(cards, {
    rotateX: [-90, 0],
    opacity: [0, 1],
    duration: opts.duration ?? scaled(500),
    delay: stagger(opts.stagger ?? scaled(60)),
    ease: cubicBezier(0.16, 1, 0.3, 1),
  });
}, { kind: 'text', intensity: 'medium' });
