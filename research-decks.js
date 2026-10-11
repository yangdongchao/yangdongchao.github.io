/* No autoplay. Each presentation owns its position and keyboard navigation. */
document.querySelectorAll('[data-deck]').forEach(deck => {
  const slides = [...deck.querySelectorAll('.research-slide')];
  const tabs = [...deck.querySelectorAll('[data-page]')];
  const previous = deck.querySelector('[data-step="-1"]');
  const next = deck.querySelector('[data-step="1"]');
  let current = 0;
  function show(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      slide.hidden = i !== current;
      slide.inert = i !== current;
      slide.setAttribute('aria-hidden', String(i !== current));
      if (i !== current) slide.querySelectorAll('audio, video').forEach(media => media.pause());
    });
    tabs.forEach((tab, i) => {
      if (i === current) tab.setAttribute('aria-current', 'step');
      else tab.removeAttribute('aria-current');
    });
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    deck.querySelector('.deck-counter').textContent = `${current + 1} / ${slides.length}`;
  }
  tabs.forEach(tab => tab.addEventListener('click', () => show(Number(tab.dataset.page))));
  [previous, next].forEach(button => button.addEventListener('click', () => show(current + Number(button.dataset.step))));
  deck.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.target.closest('audio, video, input, textarea, select, [contenteditable]')) return;
    const positions = {ArrowLeft: current - 1, ArrowRight: current + 1, Home: 0, End: slides.length - 1};
    if (!(event.key in positions)) return;
    event.preventDefault();
    show(positions[event.key]);
    if (event.target.tagName === 'BUTTON') tabs[current].focus();
  });
  show(0);
  deck.classList.add('is-ready');
  deck.querySelector('.deck-controls').hidden = false;
});
document.querySelectorAll('.research-deck audio').forEach(media => media.addEventListener('play', () => {
  document.querySelectorAll('audio, video').forEach(other => { if (other !== media) other.pause(); });
}));
// Collapsing an explanation must stop any audio it contains.
document.querySelectorAll('.story-details').forEach(details => {
  details.addEventListener('toggle', () => {
    if (!details.open) details.querySelectorAll('audio, video').forEach(media => media.pause());
  });
});
function revealResearchAnchor() {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  if (!id) return;
  const target = document.getElementById(id);
  const details = target?.closest('.story-details');
  if (details) details.open = true;
  const slide = target?.closest('.research-slide');
  if (slide) {
    const deck = slide.closest('[data-deck]');
    const slides = [...deck.querySelectorAll('.research-slide')];
    deck.querySelectorAll('[data-page]')[slides.indexOf(slide)]?.click();
  }
  // Revealing a hidden slide or loading styles can move its anchor.
  if (target) requestAnimationFrame(() => target.scrollIntoView({block: 'start', behavior: 'instant'}));
}
revealResearchAnchor();
window.addEventListener('hashchange', revealResearchAnchor);
window.addEventListener('load', revealResearchAnchor);
