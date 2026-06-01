/* ============================================================
   stats.js — small statistics toolkit for the interactive pages.
   Pure client-side; no dependencies.
   ============================================================ */
(function (global) {
  "use strict";
  const S = {};

  /* ---- seeded RNG (mulberry32) so demos are reproducible ---- */
  S.makeRNG = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  /* ---- special functions ---- */
  // Abramowitz & Stegun 7.1.26 error-function approximation
  S.erf = function (x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return sign * y;
  };
  S.logGamma = function (z) {
    const g = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let x = z, tmp = z + 5.5;
    tmp -= (z + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) { x += 1; ser += g[j] / x; }
    return -tmp + Math.log(2.5066282746310005 * ser / z);
  };
  S.logChoose = function (n, k) {
    if (k < 0 || k > n) return -Infinity;
    return S.logGamma(n + 1) - S.logGamma(k + 1) - S.logGamma(n - k + 1);
  };
  S.logFact = function (n) { return S.logGamma(n + 1); };

  /* ---- densities / mass functions ---- */
  S.normalPdf = (x, mu, sigma) =>
    Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
  S.normalCdf = (x, mu, sigma) =>
    0.5 * (1 + S.erf((x - mu) / (sigma * Math.SQRT2)));
  S.binomialPmf = (k, n, p) => {
    if (k < 0 || k > n) return 0;
    if (p <= 0) return k === 0 ? 1 : 0;
    if (p >= 1) return k === n ? 1 : 0;
    return Math.exp(S.logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
  };
  S.poissonPmf = (k, mu) => k < 0 ? 0 : Math.exp(k * Math.log(mu) - mu - S.logFact(k));
  S.geometricPmf = (k, p) => k < 1 ? 0 : Math.pow(1 - p, k - 1) * p;
  S.exponentialPdf = (x, lambda) => x < 0 ? 0 : lambda * Math.exp(-lambda * x);

  /* ---- random sampling ---- */
  S.sampleNormal = (rng, mu, sigma) => {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  S.sampleUniform = (rng, a, b) => a + (b - a) * rng();
  S.sampleExponential = (rng, lambda) => -Math.log(1 - rng()) / lambda;
  S.sampleBernoulli = (rng, p) => (rng() < p ? 1 : 0);
  // a strongly skewed distribution for CLT demos (sum of squares of exp)
  S.samplers = {
    uniform: (rng) => S.sampleUniform(rng, 0, 1),
    exponential: (rng) => S.sampleExponential(rng, 1),
    bernoulli: (rng) => S.sampleBernoulli(rng, 0.35),
    skewed: (rng) => 1 - Math.pow(1 - rng(), 1 / 3),  // density 3(1-x)^2 on [0,1]
    bimodal: (rng) => (rng() < 0.5 ? S.sampleNormal(rng, 0.25, 0.07)
      : S.sampleNormal(rng, 0.78, 0.07)),
  };
  S.samplerMoments = {
    // mean and variance of a single draw, for standardizing CLT plots
    uniform: { m: 0.5, v: 1 / 12 },
    exponential: { m: 1, v: 1 },
    bernoulli: { m: 0.35, v: 0.35 * 0.65 },
    skewed: { m: 0.25, v: 3 / 80 },             // density 3(1-x)^2: mean 1/4, var 3/80
    bimodal: { m: 0.515, v: 0.0707 },
  };

  /* ---- summary statistics ---- */
  S.linspace = (a, b, n) => {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  };
  S.mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  S.variance = (a) => {
    const m = S.mean(a);
    return a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1);
  };
  S.std = (a) => Math.sqrt(S.variance(a));
  // empirical quantile via linear interpolation (i/(n+1) convention)
  S.quantile = (sorted, p) => {
    const n = sorted.length;
    if (n === 0) return NaN;
    const h = p * (n + 1);
    if (h <= 1) return sorted[0];
    if (h >= n) return sorted[n - 1];
    const lo = Math.floor(h) - 1, frac = h - Math.floor(h);
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  };
  S.fiveNumber = (data) => {
    const s = [...data].sort((a, b) => a - b);
    return {
      min: s[0], q1: S.quantile(s, 0.25), med: S.quantile(s, 0.5),
      q3: S.quantile(s, 0.75), max: s[s.length - 1], sorted: s,
    };
  };

  /* ---- gaussian kernel density estimate ---- */
  S.kde = (data, xs, bandwidth) => {
    const n = data.length, h = bandwidth;
    return xs.map((x) => {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const u = (x - data[i]) / h;
        s += Math.exp(-0.5 * u * u);
      }
      return s / (n * h * Math.sqrt(2 * Math.PI));
    });
  };

  /* ---- simple linear regression (with & without intercept) ---- */
  S.linreg = (x, y) => {
    const n = x.length;
    const sx = x.reduce((a, b) => a + b, 0), sy = y.reduce((a, b) => a + b, 0);
    const sxy = x.reduce((a, b, i) => a + b * y[i], 0);
    const sxx = x.reduce((a, b) => a + b * b, 0);
    const beta = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const alpha = sy / n - beta * sx / n;
    return { alpha, beta };
  };
  S.sse = (x, y, alpha, beta) =>
    x.reduce((s, xi, i) => s + (y[i] - alpha - beta * xi) ** 2, 0);

  global.Stats = S;
})(window);
