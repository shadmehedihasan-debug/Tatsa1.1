/*
 * SORT (Simple Online and Realtime Tracking) in plain JavaScript.
 *
 *  - One constant-velocity Kalman filter per box coordinate (cx, cy, w, h)
 *  - IoU cost matrix + Hungarian algorithm for detection-to-track matching
 *  - Tracks are confirmed after `minHits` and dropped after `maxAge` missed frames
 *
 * Usage:
 *   const tracker = new Sort({ maxAge: 30, minHits: 3, iouThreshold: 0.3 });
 *   const tracks = tracker.update([{ bbox: [x, y, w, h], cls: 'person', score: 0.9 }]);
 *   // tracks -> [{ id, cls, score, bbox: [x1, y1, x2, y2], trail: [[cx, cy], ...] }]
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Sort = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- 1D constant-velocity Kalman filter: state [position, velocity] ---------- */
  class Kalman1D {
    constructor(z, { q0 = 1, q1 = 0.1, r = 25 } = {}) {
      this.x = z;
      this.v = 0;
      this.p00 = 10;
      this.p01 = 0;
      this.p10 = 0;
      this.p11 = 1000; // velocity is unknown at first
      this.q0 = q0;
      this.q1 = q1;
      this.r = r;
    }
    predict() {
      this.x += this.v;
      const p00 = this.p00 + this.p01 + this.p10 + this.p11 + this.q0;
      const p01 = this.p01 + this.p11;
      const p10 = this.p10 + this.p11;
      const p11 = this.p11 + this.q1;
      this.p00 = p00;
      this.p01 = p01;
      this.p10 = p10;
      this.p11 = p11;
    }
    update(z) {
      const s = this.p00 + this.r;
      const k0 = this.p00 / s;
      const k1 = this.p10 / s;
      const y = z - this.x;
      this.x += k0 * y;
      this.v += k1 * y;
      const p00 = (1 - k0) * this.p00;
      const p01 = (1 - k0) * this.p01;
      const p10 = this.p10 - k1 * this.p00;
      const p11 = this.p11 - k1 * this.p01;
      this.p00 = p00;
      this.p01 = p01;
      this.p10 = p10;
      this.p11 = p11;
    }
  }

  /* ---------- Geometry ---------- */
  // Boxes are [x1, y1, x2, y2]
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    const union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
  }

  /* ---------- Hungarian algorithm (minimum-cost assignment) ---------- */
  // cost: n x m matrix. Returns an array of length n: assigned column index or -1.
  function hungarian(cost) {
    const n = cost.length;
    const m = n ? cost[0].length : 0;
    if (!n || !m) return new Array(n).fill(-1);

    const transposed = n > m;
    const a = transposed
      ? Array.from({ length: m }, (_, j) => cost.map((row) => row[j]))
      : cost;
    const N = a.length;
    const M = a[0].length; // N <= M
    const INF = 1e9;

    const u = new Array(N + 1).fill(0);
    const v = new Array(M + 1).fill(0);
    const p = new Array(M + 1).fill(0);
    const way = new Array(M + 1).fill(0);

    for (let i = 1; i <= N; i++) {
      p[0] = i;
      let j0 = 0;
      const minv = new Array(M + 1).fill(INF);
      const used = new Array(M + 1).fill(false);
      do {
        used[j0] = true;
        const i0 = p[j0];
        let delta = INF;
        let j1 = 0;
        for (let j = 1; j <= M; j++) {
          if (used[j]) continue;
          const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
        for (let j = 0; j <= M; j++) {
          if (used[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else {
            minv[j] -= delta;
          }
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do {
        const j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0);
    }

    const result = new Array(n).fill(-1);
    for (let j = 1; j <= M; j++) {
      if (!p[j]) continue;
      if (transposed) result[j - 1] = p[j] - 1;
      else result[p[j] - 1] = j - 1;
    }
    return result;
  }

  /* ---------- Single tracked object ---------- */
  const TRAIL_LENGTH = 40;

  class Track {
    constructor(id, det) {
      const [x, y, w, h] = det.bbox;
      this.id = id;
      this.kf = [
        new Kalman1D(x + w / 2),
        new Kalman1D(y + h / 2),
        new Kalman1D(w, { q0: 1, q1: 0.05 }),
        new Kalman1D(h, { q0: 1, q1: 0.05 }),
      ];
      this.votes = { [det.cls]: det.score };
      this.cls = det.cls;
      this.score = det.score;
      this.hits = 1;
      this.hitStreak = 1;
      this.age = 0;
      this.timeSinceUpdate = 0;
      this.trail = [[x + w / 2, y + h / 2]];
    }
    predict() {
      for (const k of this.kf) k.predict();
      this.age++;
      if (this.timeSinceUpdate > 0) this.hitStreak = 0;
      this.timeSinceUpdate++;
    }
    update(det) {
      const [x, y, w, h] = det.bbox;
      this.kf[0].update(x + w / 2);
      this.kf[1].update(y + h / 2);
      this.kf[2].update(w);
      this.kf[3].update(h);
      // Label voting: the class with the best (slowly decaying) total score wins, so labels stop flickering
      for (const k in this.votes) this.votes[k] *= 0.95;
      this.votes[det.cls] = (this.votes[det.cls] || 0) + det.score;
      this.cls = Object.keys(this.votes).reduce((p, q) => (this.votes[p] >= this.votes[q] ? p : q));
      this.score = this.score * 0.7 + det.score * 0.3;
      this.hits++;
      this.hitStreak++;
      this.timeSinceUpdate = 0;
      this.trail.push([this.kf[0].x, this.kf[1].x]);
      if (this.trail.length > TRAIL_LENGTH) this.trail.shift();
    }
    box() {
      const cx = this.kf[0].x;
      const cy = this.kf[1].x;
      const w = Math.max(1, this.kf[2].x);
      const h = Math.max(1, this.kf[3].x);
      return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    }
  }

  /* ---------- Tracker ---------- */
  class Sort {
    constructor({ maxAge = 30, minHits = 3, iouThreshold = 0.3, highThresh = 0.5, lowIou = 0.5 } = {}) {
      this.maxAge = maxAge;
      this.minHits = minHits;
      this.iouThreshold = iouThreshold;
      this.highThresh = highThresh; // detections at or above this score may start tracks
      this.lowIou = lowIou;
      this.reset();
    }

    reset() {
      this.tracks = [];
      this.frameCount = 0;
      this.nextId = 1;
    }

    get idsAssigned() {
      return this.nextId - 1;
    }

    // dets: [{ bbox: [x, y, w, h], cls, score }]
    // ByteTrack-style: confident detections match first, weak ones only rescue tracks that were missed.
    update(dets) {
      this.frameCount++;
      for (const t of this.tracks) t.predict();
      const predicted = this.tracks.map((t) => t.box());
      const boxes = dets.map((d) => [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]]);

      // IoU + Hungarian matching of some tracks to some detections (a different label is allowed, at a penalty)
      const match = (tIdx, dIdx, thr) => {
        const pairs = [];
        const usedT = new Set();
        const usedD = new Set();
        if (tIdx.length && dIdx.length) {
          const cost = tIdx.map((i) =>
            dIdx.map((j) => 1 - iou(predicted[i], boxes[j]) * (this.tracks[i].cls === dets[j].cls ? 1 : 0.8))
          );
          hungarian(cost).forEach((c, r) => {
            if (c >= 0 && 1 - cost[r][c] >= thr) {
              pairs.push([tIdx[r], dIdx[c]]);
              usedT.add(tIdx[r]);
              usedD.add(dIdx[c]);
            }
          });
        }
        return { pairs, tLeft: tIdx.filter((i) => !usedT.has(i)), dLeft: dIdx.filter((j) => !usedD.has(j)) };
      };

      const high = [];
      const low = [];
      dets.forEach((d, j) => (d.score >= this.highThresh ? high : low).push(j));

      const s1 = match(this.tracks.map((_, i) => i), high, this.iouThreshold); // 1. confident detections
      const s2 = match(s1.tLeft, low, this.lowIou); // 2. weak detections rescue missed tracks
      for (const [i, j] of [...s1.pairs, ...s2.pairs]) this.tracks[i].update(dets[j]);

      // 3. Only confident, unmatched detections start new tracks
      s1.dLeft.forEach((j) => this.tracks.push(new Track(this.nextId++, dets[j])));

      // 4. Drop tracks that have been lost for too long
      this.tracks = this.tracks.filter((t) => t.timeSinceUpdate <= this.maxAge);

      // 5. Report confirmed tracks that were seen this frame
      return this.tracks
        .filter((t) => t.timeSinceUpdate < 1 && (t.hitStreak >= this.minHits || this.frameCount <= this.minHits))
        .map((t) => ({ id: t.id, cls: t.cls, score: t.score, bbox: t.box(), trail: t.trail }));
    }
  }

  Sort.iou = iou;
  Sort.hungarian = hungarian;
  return Sort;
});
