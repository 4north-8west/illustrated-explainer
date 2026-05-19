export function motionSafe() {
  const override = document.documentElement.getAttribute('data-motion');
  if (override === 'off') return false;
  if (override === 'on') return true;
  const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
  return !mq.matches;
}
