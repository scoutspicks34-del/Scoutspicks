/* Picks Fresh: service worker. La página se pide primero a internet (los datos cambian varias veces al día);
   si no hay conexión o tarda más de 4 s, se usa la última copia guardada. Íconos y manifiesto salen de la copia. */
const V = "pf-v1";
const SHELL = ["./", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png", "icons/favicon-32.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(V).then(c => Promise.allSettled(SHELL.map(u => c.add(u)))).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const r = e.request;
  if (r.method !== "GET" || new URL(r.url).origin !== location.origin) return;
  const isPage = r.mode === "navigate" || (r.headers.get("accept") || "").includes("text/html");
  if (isPage) {
    e.respondWith((async () => {
      const cache = await caches.open(V);
      try {
        const res = await Promise.race([fetch(r), new Promise((_, no) => setTimeout(() => no(new Error("lento")), 4000))]);
        if (res && res.ok) cache.put("./", res.clone());
        return res;
      } catch (err) {
        return (await cache.match("./")) || (await cache.match(r)) || new Response("Sin conexión y todavía no hay copia guardada.", { status: 503, headers: { "Content-Type": "text/plain; charset=utf-8" } });
      }
    })());
    return;
  }
  e.respondWith(caches.match(r).then(hit => hit || fetch(r).then(res => {
    if (res && res.ok) caches.open(V).then(c => c.put(r, res.clone()));
    return res;
  })));
});
