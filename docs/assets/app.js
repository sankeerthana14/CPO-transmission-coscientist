'use strict';
(() => {
  const toggle = document.querySelector('.menu-toggle');
  const nav = document.querySelector('.site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
    nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
      nav.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    }));
  }
  const filters = document.querySelectorAll('.filter-button');
  const cards = document.querySelectorAll('.figure-card');
  filters.forEach(button => button.addEventListener('click', () => {
    const type = button.dataset.filter;
    filters.forEach(b => {
      const selected = b === button;
      b.classList.toggle('active', selected);
      b.setAttribute('aria-pressed', String(selected));
    });
    cards.forEach(card => { card.hidden = type !== 'all' && card.dataset.category !== type; });
  }));
  const dialog = document.getElementById('figure-dialog');
  const dialogImage = document.getElementById('lightbox-image');
  const dialogText = document.getElementById('lightbox-caption');
  const dialogLink = document.getElementById('lightbox-open');
  const close = document.querySelector('.lightbox-close');
  let lastTrigger = null;
  document.querySelectorAll('.image-open').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const url = trigger.getAttribute('data-full');
      if (!url) return;
      if (!dialog || typeof dialog.showModal !== 'function') { window.open(url, '_blank', 'noopener'); return; }
      lastTrigger = trigger;
      dialogImage.setAttribute('src', url);
      const img = trigger.querySelector('img');
      dialogImage.setAttribute('alt', img?.getAttribute('alt') || 'Enlarged research figure');
      dialogText.textContent = trigger.getAttribute('data-caption') || 'Research figure';
      dialogLink.setAttribute('href', url);
      dialog.showModal();
      document.body.classList.add('modal-open');
    });
  });
  function dismiss() {
    if (dialog && dialog.open) dialog.close();
  }
  close?.addEventListener('click', dismiss);
  dialog?.addEventListener('click', event => { if (event.target === dialog) dismiss(); });
  dialog?.addEventListener('close', () => {
    document.body.classList.remove('modal-open');
    dialogImage.removeAttribute('src');
    lastTrigger?.focus();
  });
  const citation = 'Satini, S., Chen, C. K., Tan, D. T. E., Venkatesh, M., & Tan, C. W. (2026). AI-Assisted Multimodal Source Prioritisation for Hospital-Acquired Carbapenemase-Producing Organism Transmission. i-AIM 2026, research poster 7926.';
  const copyButton = document.getElementById('copy-citation');
  copyButton?.addEventListener('click', async () => {
    const status = document.getElementById('copy-feedback');
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(citation);
        status.textContent = 'Poster citation copied.';
      } else {
        const input = document.createElement('textarea');
        input.value = citation;
        input.style.position = 'fixed'; input.style.left = '-9999px';
        document.body.appendChild(input); input.select();
        const success = document.execCommand('copy');
        input.remove();
        status.textContent = success ? 'Poster citation copied.' : citation;
      }
    } catch (_) { status.textContent = citation; }
  });
  if ('IntersectionObserver' in window) {
    const links = [...document.querySelectorAll('.site-nav a[href^="#"]')];
    const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const observer = new IntersectionObserver(entries => {
      const item = entries.filter(e => e.isIntersecting).sort((a,b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!item) return;
      links.forEach(link => link.classList.toggle('active', link.getAttribute('href') === '#' + item.target.id));
    }, {rootMargin:'-90px 0px -63% 0px', threshold:0});
    sections.forEach(section => observer.observe(section));
  }
})();
