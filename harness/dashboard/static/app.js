const $ = (id) => document.getElementById(id);
const usd = (x) => "$" + x.toFixed(4);
const pct = (x) => (x > 0 ? "+" : "") + (x * 100).toFixed(1) + "%";

async function refresh() {
  const d = await (await fetch("/api/ab")).json();
  const small = Math.min(d.groups.policies.n, d.groups.control.n) < d.min_per_group;
  $("note").textContent = [d.simulated && "Simulated", small && `Needs ${d.min_per_group}+ per group`].filter(Boolean).join(" · ");
  $("change").textContent = d.change == null ? "-" : pct(d.change);
  $("ci").textContent = d.ci ? `per conversation · 95% CI ${pct(d.ci[0])} to ${pct(d.ci[1])}` : "";
  for (const key of ["policies", "control"]) {
    const g = d.groups[key];
    $(`${key}-mean`).textContent = g.n ? usd(g.mean) : "-";
    $(`${key}-meta`).textContent = `${g.n} conversations · ${(g.cache_hit * 100).toFixed(0)}% cached`;
  }
  chart([["a", d.groups.policies.running_mean], ["b", d.groups.control.running_mean]]);
}

function chart(series) {
  const W = 880, H = 260, L = 64, R = 16, T = 8, B = 24;
  const n = Math.max(...series.map(([, v]) => v.length));
  if (!n) return;
  const top = Math.max(...series.flatMap(([, v]) => v)) * 1.1;
  const x = (i) => L + (n > 1 ? (i / (n - 1)) * (W - L - R) : 0);
  const y = (v) => H - B - (v / top) * (H - T - B);
  const grid = [0, 0.5, 1].map((f) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(f * top)}" y2="${y(f * top)}"/><text x="${L - 8}" y="${y(f * top) + 4}" text-anchor="end">${usd(f * top)}</text>`);
  const lines = series.map(([c, v]) => `<polyline stroke="var(--${c})" points="${v.map((p, i) => `${x(i)},${y(p)}`).join(" ")}"/>`);
  $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Average cost per conversation">${grid.join("")}${lines.join("")}<text x="${W - R}" y="${H - 6}" text-anchor="end">${n} conversations</text></svg>`;
}

refresh();
setInterval(refresh, 3000);
