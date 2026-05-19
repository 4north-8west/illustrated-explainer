// Preset registry. Each preset entry: { name, kind, fn, intensity }
// kind: 'text' | 'element' | 'transition'
// intensity: 'low' | 'medium' | 'high'  (UI hints only; high = use sparingly)
export const presetRegistry = new Map();

export function registerPreset(name, fn, { kind = 'element', intensity = 'low' } = {}) {
  presetRegistry.set(name, { name, kind, fn, intensity });
}

export function getPreset(name) {
  return presetRegistry.get(name);
}

export function listPresets() {
  return [...presetRegistry.values()];
}
