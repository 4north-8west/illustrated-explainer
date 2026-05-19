import { animateText } from './anime-helpers.js';

// Reads every data-anime-* attribute as an option. Numeric strings become
// numbers; everything else stays as strings. The preset names the attribute,
// so adding a new option (e.g. data-anime-prefix) requires no helper change.
function readOpts(el) {
  const opts = {};
  for (const key in el.dataset) {
    if (!key.startsWith('anime') || key === 'anime' || key === 'animeBound') continue;
    const optKey = key.slice(5);
    const camel = optKey.charAt(0).toLowerCase() + optKey.slice(1);
    const v = el.dataset[key];
    if (v === '') { opts[camel] = ''; continue; }
    const n = Number(v);
    opts[camel] = Number.isFinite(n) && /^-?[\d.]+$/.test(v) ? n : v;
  }
  return opts;
}

export function bindElement(el) {
  const preset = el.dataset.anime;
  if (!preset || el.dataset.animeBound === '1') return false;
  el.dataset.animeBound = '1';
  const opts = readOpts(el);
  const obs = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        animateText(el, preset, opts);
        obs.unobserve(el);
      }
    }
  }, { threshold: 0.2 });
  obs.observe(el);
  return true;
}

// Held module-scope so we set up the observer at most once per document.
let mutationObserver = null;

export function autoBind(root = document) {
  const els = root.querySelectorAll('[data-anime]');
  let n = 0;
  for (const el of els) if (bindElement(el)) n++;

  // For SPAs and any page that inserts [data-anime] elements dynamically:
  // observe the document body and bind elements as they appear. Set up only
  // once, only when binding from the document root — sub-tree autoBind calls
  // skip this.
  if (root === document && !mutationObserver && typeof MutationObserver !== 'undefined') {
    mutationObserver = new MutationObserver((records) => {
      for (const rec of records) {
        for (const node of rec.addedNodes) {
          if (node.nodeType !== 1) continue; // element nodes only
          if (node.matches && node.matches('[data-anime]')) bindElement(node);
          if (node.querySelectorAll) {
            node.querySelectorAll('[data-anime]').forEach(bindElement);
          }
        }
      }
    });
    mutationObserver.observe(document.body, { childList: true, subtree: true });
  }

  return n;
}
