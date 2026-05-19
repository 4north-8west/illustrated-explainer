/**
 * Reveal.js plugin for anime-helpers.
 *
 * Wires elements tagged with [data-anime="<preset>"] to slide enter/leave events.
 * Animations run when the slide becomes active and reset when it leaves so backward
 * navigation re-plays the entrance.
 *
 * Markup convention:
 *   - On a heading/paragraph: `<!-- .element: data-anime="stagger-words" -->`
 *     (Reveal markdown injects the attributes onto the previous element)
 *   - Or as raw HTML: `<h1 data-anime="stagger-words">...</h1>`
 *   - data-anime-duration / data-anime-stagger / data-anime-delay / data-anime-from /
 *     data-anime-to are passed through as numeric or string options.
 *
 * Usage in Reveal.js initialize:
 *   import RevealAnime from './anime/reveal-anime.js';
 *   Reveal.initialize({ plugins: [ RevealMarkdown, RevealAnime ] });
 */

import { animateText, animateElement, listPresets, motionSafe } from './anime-helpers.js';

const PARSED_KEYS = ['delay', 'duration', 'stagger', 'from', 'to'];

function readOpts(el) {
  const opts = {};
  for (const key of PARSED_KEYS) {
    const dsKey = `anime${key.charAt(0).toUpperCase() + key.slice(1)}`;
    const v = el.dataset[dsKey];
    if (v === undefined) continue;
    const n = Number(v);
    opts[key] = Number.isFinite(n) && /^[\d.\-]+$/.test(v) ? n : v;
  }
  return opts;
}

// Cache the original innerHTML (before any preset mutated it) so we can reset on slide leave.
const originalHTML = new WeakMap();

function captureOriginal(slide) {
  const els = slide.querySelectorAll('[data-anime]');
  for (const el of els) {
    if (!originalHTML.has(el)) {
      originalHTML.set(el, el.innerHTML);
    }
  }
}

function resetSlide(slide) {
  const els = slide.querySelectorAll('[data-anime]');
  for (const el of els) {
    const html = originalHTML.get(el);
    if (html !== undefined) {
      el.innerHTML = html;
      el.removeAttribute('style');
    }
  }
}

function playSlide(slide) {
  const els = slide.querySelectorAll('[data-anime]');
  for (const el of els) {
    const preset = el.dataset.anime;
    if (!preset) continue;
    const meta = listPresets().find(p => p.name === preset);
    if (!meta) {
      console.warn(`reveal-anime: unknown preset "${preset}" on`, el);
      continue;
    }
    const opts = readOpts(el);
    if (meta.kind === 'text') animateText(el, preset, opts);
    else animateElement(el, preset, opts);
  }
}

const RevealAnime = {
  id: 'anime',
  init(deck) {
    // First pass: capture originals so we can reset on slide leave.
    deck.getSlides().forEach(captureOriginal);

    // Play on initial slide once Reveal is ready.
    deck.on('ready', (event) => {
      playSlide(event.currentSlide);
    });

    // On every slide change: reset the prior slide, play the new one.
    deck.on('slidechanged', (event) => {
      if (event.previousSlide) resetSlide(event.previousSlide);
      playSlide(event.currentSlide);
    });

    return Promise.resolve();
  },
  // Exposed for the user / tests
  __internals: { playSlide, resetSlide, readOpts },
};

export default RevealAnime;
