const CACHE_NAME = "fusionshield-ai-pwa-v3.6";
const STATIC_ASSETS = [
  "/",
  "/manifest.json",
  "/index.html",
  "/styles.css?v=3.5",
  "/app.js?v=3.5",
  "/logo.png?v=3.5",
  "/icon-192.png",
  "/icon-512.png",
  "/icon-maskable-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith("fusionshield-ai-pwa-") && key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Always bypass cache for POST requests and API routes (/api/predict, /api/report, /api/status)
  if (request.method !== "GET" || request.url.includes("/api/")) {
    return;
  }

  // Network-first with cache fallback for HTML/CSS/JS so updates appear immediately when online,
  // while still working as an installable offline-capable PWA for APK builders (PWABuilder / TWA)
  event.respondWith(
    fetch(request)
      .then((networkResponse) => {
        if (
          networkResponse &&
          networkResponse.status === 200 &&
          networkResponse.type === "basic"
        ) {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(request, responseClone);
          });
        }
        return networkResponse;
      })
      .catch(() => caches.match(request).then((cached) => cached || caches.match("/")))
  );
});

