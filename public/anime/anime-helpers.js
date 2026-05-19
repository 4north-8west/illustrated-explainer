import { motionSafe } from './motion-safe.js';
import { presetRegistry, registerPreset, getPreset, listPresets } from './presets/index.js';
import './presets/text.js';
import './presets/element.js';
import './presets/transition.js';

function runPreset(el, presetName, opts = {}) {
  const preset = getPreset(presetName);
  if (!preset) {
    throw new Error(`unknown preset: ${presetName}`);
  }
  if (!motionSafe()) {
    return null;
  }
  return preset.fn(el, opts);
}

export function animateText(el, presetName, opts) {
  return runPreset(el, presetName, opts);
}

export function animateElement(el, presetName, opts) {
  return runPreset(el, presetName, opts);
}

export { registerPreset, getPreset, listPresets, motionSafe };
export { autoBind, bindElement } from './intersection.js';
