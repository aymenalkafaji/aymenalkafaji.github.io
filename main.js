const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* === Mobile Nav Toggle === */
document.addEventListener('DOMContentLoaded', () => {
  const navToggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');
  if (!navToggle || !navLinks) return;

  function closeNav() {
    navToggle.setAttribute('aria-expanded', 'false');
    navLinks.classList.remove('is-open');
  }

  navToggle.addEventListener('click', () => {
    const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!isOpen));
    navLinks.classList.toggle('is-open', !isOpen);
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', closeNav);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeNav();
  });
});

/* === Active Nav Link on Scroll === */
document.addEventListener('DOMContentLoaded', () => {
  const sections = Array.from(document.querySelectorAll('section[id]'));
  const navAnchors = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  if (!sections.length || !navAnchors.length) return;

  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navAnchors.forEach((a) => {
        a.classList.toggle('is-active', a.getAttribute('href') === `#${entry.target.id}`);
      });
    });
  }, { rootMargin: '-40% 0px -50% 0px' });

  sections.forEach((section) => sectionObserver.observe(section));
});

/* === Scroll Reveal === */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;

    const el = entry.target;
    const delay = Number(el.dataset.revealDelay) || 0;

    setTimeout(() => el.classList.add('is-visible'), delay);
    revealObserver.unobserve(el);

    if (el.classList.contains('stat-strip')) {
      el.querySelectorAll('.stat-number').forEach(animateCount);
    }
  });
}, { threshold: 0.15 });

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.reveal').forEach((el) => revealObserver.observe(el));
});

/* === Count-Up Stats === */
function animateCount(el) {
  const target = Number(el.dataset.target) || 0;
  const suffix = el.dataset.suffix || '';

  if (prefersReducedMotion) {
    el.textContent = target.toLocaleString() + suffix;
    return;
  }

  const duration = 1200;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(eased * target).toLocaleString() + suffix;

    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      el.textContent = target.toLocaleString() + suffix;
    }
  }

  requestAnimationFrame(tick);
}
