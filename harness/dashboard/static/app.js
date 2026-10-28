// Polls /api/ab and redraws. No libraries, so it works offline.

const GROUPS = [
  { key: "policies", name: "Policies", cls: "1" },
  { key: "control", name: "Control", cls: "2" },
];
const $ = (id) => document.getElementById(id);
const money = (x) => (x == null ? "-" : "$" + x.toFixed(4));
const pct = (x, sign = false) => (x == null ? "-" : (sign && x > 0 ? "+" : "") + (x * 100).toFixed(1) + "%");
let last = null;

async function refresh() {
  try {
    const res = await fetch("/api/ab", { cache: "no-store" });
    last = await res.json();
    render(last);
  } catch (err) {
    $("banner").hidden = false;
    $("banner").textContent = "Can't reach the dashboard server. Is it still running?";
  }
}

function render(d) {
  const banner = [];
  if (d.simulated) banner.push("Simulated demo traffic. These numbers show how the page works, not what you'll save.");
  const smaller = Math.min(d.groups.policies.n, d.groups.control.n);
  if (smaller < d.min_per_group) banner.push(`${smaller} conversations in the smaller half so far. Wait for ${d.min_per_group}+ before trusting the result.`);
  $("banner").hidden = banner.length === 0;
  $("banner").textContent = banner.join(" ");

  const change = $("change");
  if (d.change == null) {
    change.textContent = "waiting for traffic";
    change.classList.remove("good");
    $("ci-text").textContent = "";
  } else {
    change.textContent = pct(d.change, true);
    const [lo, hi] = d.ci;
    change.classList.toggle("good", hi < 0);
    $("ci-text").textContent = `95% CI ${pct(lo, true)} to ${pct(hi, true)}` + (hi < 0 ? ". Cheaper, and the interval doesn't cross zero." : lo > 0 ? ". More expensive." : ". No clear difference yet.");
    drawInterval(d.change, lo, hi);
  }

  for (const g of GROUPS) {
    const s = d.groups[g.key];
    $(`${g.key}-mean`).textContent = s.n ? money(s.mean) : "-";
    $(`${g.key}-n`).textContent = s.n;
    $(`${g.key}-hit`).textContent = s.calls ? pct(s.cache_hit) : "-";
    $(`${g.key}-total`).textContent = money(s.total);
  }
  $("table").innerHTML = GROUPS.map((g) => {
    const s = d.groups[g.key];
    return `<tr><td>${g.name}</td><td>${s.n}</td><td>${s.calls}</td><td>${s.n ? money(s.mean) : "-"}</td><td>${s.calls ? pct(s.cache_hit) : "-"}</td><td>${money(s.total)}</td></tr>`;
  }).join("");
  drawChart(d);
}

// The interval strip is centered on zero and scaled to the widest end.
function drawInterval(point, lo, hi) {
  const span = Math.max(Math.abs(lo), Math.abs(hi), Math.abs(point), 0.05) * 1.1;
  const at = (x) => 50 + (x / span) * 50;
  $("interval-bar").style.left = at(lo) + "%";
  $("interval-bar").style.width = at(hi) - at(lo) + "%";
  $("interval-point").style.left = at(point) + "%";
}

const W = 880, H = 300, M = { top: 12, right: 90, bottom: 28, left: 56 };

function drawChart(d) {
  const series = GROUPS.map((g) => ({ ...g, values: d.groups[g.key].running_mean }));
  const longest = Math.max(...series.map((s) => s.values.length));
  const chart = $("chart");
  if (longest === 0) {
    chart.innerHTML = '<div class="empty">No conversations yet</div>';
    return;
  }
  const maxY = Math.max(...series.flatMap((s) => s.values)) * 1.15 || 1;
  const x = (i) => M.left + (longest === 1 ? 0 : (i / (longest - 1)) * (W - M.left - M.right));
  const y = (v) => H - M.bottom - (v / maxY) * (H - M.top - M.bottom);

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * maxY);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Average cost per conversation for each half">`;
  svg += '<g class="grid">' + ticks.map((t) => `<line x1="${M.left}" x2="${W - M.right}" y1="${y(t)}" y2="${y(t)}"/>`).join("") + "</g>";
  svg += '<g class="axis">' + ticks.map((t) => `<text x="${M.left - 8}" y="${y(t) + 4}" text-anchor="end">$${t.toFixed(4)}</text>`).join("");
  svg += `<text x="${M.left}" y="${H - 6}">1</text><text x="${W - M.right}" y="${H - 6}" text-anchor="end">conversation ${longest}</text></g>`;
  for (const s of series) {
    if (!s.values.length) continue;
    const pts = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const lx = x(s.values.length - 1), ly = y(s.values[s.values.length - 1]);
    svg += `<polyline class="line l${s.cls}" points="${pts}"/>`;
    svg += `<circle class="dot" cx="${lx}" cy="${ly}" r="5" fill="var(--series-${s.cls})"/>`;
    svg += `<text class="label" x="${lx + 10}" y="${ly + 4}">${s.name}</text>`;
  }
  svg += `<line class="crosshair" id="crosshair" y1="${M.top}" y2="${H - M.bottom}" visibility="hidden"/>`;
  svg += `<rect id="hit" x="${M.left}" y="${M.top}" width="${W - M.left - M.right}" height="${H - M.top - M.bottom}" fill="transparent"/>`;
  svg += "</svg>";
  chart.innerHTML = svg;

  const el = chart.querySelector("svg"), hit = chart.querySelector("#hit"), cross = chart.querySelector("#crosshair"), tip = $("tooltip");
  hit.addEventListener("mousemove", (e) => {
    const box = el.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(longest - 1, Math.round(((sx - M.left) / (W - M.left - M.right)) * (longest - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<div><strong>Conversation ${i + 1}</strong></div>` + series.map((s) => `<div><span class="swatch s${s.cls}"></span>${s.name}: ${i < s.values.length ? money(s.values[i]) : "-"}</div>`).join("");
    tip.hidden = false;
    const card = chart.parentElement.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - card.left + 12, card.width - 180) + "px";
    tip.style.top = e.clientY - card.top + 12 + "px";
  });
  hit.addEventListener("mouseleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
  });
}

refresh();
setInterval(refresh, 3000);
