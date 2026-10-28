const $ = (id) => document.getElementById(id);
const usd = (x) => "$" + x.toFixed(4);
const pct = (x) => (Math.abs(x) * 100).toFixed(1) + "%";

async function refresh() {
  const d = await (await fetch("/api/ab")).json();
  if (d.change != null) {
    const [lo, hi] = d.ci;
    const less = d.change < 0;
    const range = less ? `${pct(-hi)} to ${pct(-lo)} less` : `${pct(lo)} to ${pct(hi)} more`;
    const sure = hi < 0 || lo > 0 ? "" : " The difference isn't clear yet.";
    $("summary").textContent = `Conversations with the policies cost ${pct(d.change)} ${less ? "less" : "more"} than control (95% CI ${range}).${sure}`;
  }
  for (const key of ["policies", "control"]) {
    const g = d.groups[key];
    $(`${key}-n`).textContent = g.n;
    $(`${key}-mean`).textContent = g.n ? usd(g.mean) : "";
    $(`${key}-hit`).textContent = g.calls ? (g.cache_hit * 100).toFixed(1) + "%" : "";
    $(`${key}-total`).textContent = usd(g.total);
  }
  const small = Math.min(d.groups.policies.n, d.groups.control.n) < d.min_per_group;
  $("note").textContent = [d.simulated && "Simulated traffic.", small && `Wait for ${d.min_per_group} conversations in each group before trusting this.`].filter(Boolean).join(" ");
  chart([["a", d.groups.policies.running_mean], ["b", d.groups.control.running_mean]]);
}

function chart(series) {
  const W = 620, H = 240, L = 56, R = 8, T = 8, B = 28;
  const n = Math.max(...series.map(([, v]) => v.length));
  if (!n) return;
  const top = Math.max(...series.flatMap(([, v]) => v)) * 1.1;
  const x = (i) => L + (n > 1 ? (i / (n - 1)) * (W - L - R) : 0);
  const y = (v) => H - B - (v / top) * (H - T - B);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => `<line x1="${L}" x2="${W - R}" y1="${y(f * top)}" y2="${y(f * top)}"/><text x="${L - 6}" y="${y(f * top) + 4}" text-anchor="end">${usd(f * top)}</text>`);
  const lines = series.map(([c, v]) => `<polyline stroke="var(--${c})" points="${v.map((p, i) => `${x(i)},${y(p)}`).join(" ")}"/>`);
  $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Average cost per conversation">${grid.join("")}${lines.join("")}<text x="${L}" y="${H - 8}">1</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${n}</text><text x="${(L + W) / 2}" y="${H - 8}" text-anchor="middle">Conversation</text></svg>`;
}

refresh();
setInterval(refresh, 3000);
