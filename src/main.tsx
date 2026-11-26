import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/tokens.css';
import './styles/global.css';
import './styles/components.css';
import { App } from './App';

// Conditional progressive-enhancement polyfills per modern-web-guidance
if (!('commandForElement' in HTMLButtonElement.prototype)) {
  import('invokers-polyfill');
}

if (!Object.hasOwn(HTMLElement.prototype, 'popover')) {
  import('@oddbird/popover-polyfill');
}

if (!('closedBy' in HTMLDialogElement.prototype)) {
  requestAnimationFrame(() => {
    const outside = (e: PointerEvent, dialog: HTMLDialogElement) => {
      const r = dialog.getBoundingClientRect();
      return (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      );
    };
    for (const dialog of document.querySelectorAll<HTMLDialogElement>('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dialog.addEventListener('pointerdown', (e) => {
        pressedOutside = outside(e, dialog);
      });
      dialog.addEventListener('pointerup', (e) => {
        if (pressedOutside && outside(e, dialog)) dialog.close();
      });
    }
  });
}

// Fallback for hiding the compact header on scroll-down where scroll-state queries aren't supported
if (!CSS.supports('container-type', 'scroll-state')) {
  let lastY = window.scrollY;
  window.addEventListener(
    'scroll',
    () => {
      const bar = document.querySelector<HTMLElement>('.app-header');
      if (!bar) return;
      const y = Math.max(
        0,
        Math.min(window.scrollY, document.documentElement.scrollHeight - window.innerHeight),
      );
      if (y === lastY) return;
      bar.classList.toggle('is-hidden', y > lastY && y > bar.offsetHeight);
      lastY = y;
    },
    { passive: true },
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
