// DHANT service worker — cache-first for the app shell so it works offline.
const CACHE = 'dhant-v1';
const ASSETS = [
  './',
  './index.html',
  './manifest.webmanifest',
  './jss-logo.jpg',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './instructions-photos/frontal.jpg',
  './instructions-photos/upper.jpg',
  './instructions-photos/lower.jpg',
  './instructions-photos/left.jpg',
  './instructions-photos/right.jpg',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  // Never cache the AI API requests — always go to network.
  const url = e.request.url;
  if (url.includes('api.anthropic.com') || url.includes('api.openai.com') || url.includes('api.groq.com')) return;

  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((cached) => cached || fetch(e.request).then((resp) => {
      // Cache successful same-origin responses
      if (resp && resp.status === 200 && new URL(url).origin === self.location.origin) {
        const clone = resp.clone();
        caches.open(CACHE).then((c) => c.put(e.request, clone));
      }
      return resp;
    }).catch(() => caches.match('./index.html')))
  );
});
