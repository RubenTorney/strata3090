/* Optional Community tab: use Strata's helpers, components and API-key handling. */
"use strict";
window.strataCommunity = (() => {
  let state = null, selected = null, poll = null, loading = false, reloadPending = false;
  const el = id => document.getElementById(`co-${id}`);
  const n = (v, unit = "", d = 2) => v == null ? "—" : `${Number(v).toFixed(d)}${unit}`;
  const fields = [["decode_tok_s", "Generation", " tok/s", true], ["ttft_s", "First token", " s", false],
    ["ram_peak_gib", "System RAM peak", " GiB", false], ["vram_peak_gib", "GPU memory peak", " GiB", false]];
  async function api(action, data) {
    const response = await fetch(`community/${action}`, data === undefined ? {headers: headers()} :
      {method: "POST", headers: headers(true), body: JSON.stringify(data)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || "Community is disabled. Start with --community-db PATH.");
    return body;
  }
  function error(e) { toast("error", "Community", e.message, 8000); }
  const runFor = id => state.runs.find(r => r.config_id === id);
  const cards = rig => rig.gpus || [{name:rig.gpu, vram_gib:rig.vram_gib}];
  const gpuCount = rig => rig.nodes?.length ? rig.nodes.reduce((n,x)=>n+x.count*cards(x).length,0) : cards(rig).length;
  const nodeCount = rig => rig.nodes?.length ? rig.nodes.reduce((n,x)=>n+x.count,0) : 1;
  const allCards = rig => rig.nodes?.length ? rig.nodes.flatMap(cards) : cards(rig);
  const platforms = rig => rig.nodes?.length ? rig.nodes.map(x=>x.platform) : [rig.platform || "PC / workstation"];
  const isUnified = rig => rig.memory_architecture === "Unified" || rig.nodes?.some(x=>x.memory_architecture === "Unified");
  const gpuKey = name => name.toLowerCase().replace(/nvidia|geforce|amd|intel/g, "").replace(/\s+/g," ").trim();
  const rigSummary = rig => cards(rig).map(g => `${g.name} (${g.vram_gib == null ? (rig.memory_architecture === "Unified" ? "unified memory" : "unknown VRAM") : n(g.vram_gib," GiB",0)})`).join(" + ");
  const warm = rows => rows?.find(r => r.phase === "warm");
  function metrics(r) { return `<div class="co-metrics">${fields.map(([k, label, unit]) =>
    `<div><strong>${n(r?.[k], unit)}</strong><span>${label}</span></div>`).join("")}</div>`; }
  function comparable(a, b) {
    return a && b && a.protocol === b.protocol && a.prompt_tokens === b.prompt_tokens && !a.simulated && !b.simulated;
  }
  function delta(a, b, key, high) {
    if (!comparable(a, b) || a[key] == null || b[key] == null || a[key] === 0) return "—";
    const value = (b[key] / a[key] - 1) * 100;
    return `<span class="${(high ? value > 0 : value < 0) ? "co-good" : value === 0 ? "" : "co-bad"}">${value > 0 ? "+" : ""}${value.toFixed(1)}%</span>`;
  }
  function renderList() {
    if (!state) return;
    const query = el("search").value.toLowerCase().replace(/tv100/g, "v100");
    const platformFilter = el("platform").value;
    const topology = el("topology").value, ramType = el("ram-type").value;
    const gpu = el("gpu").value, os = el("os").value, rating = +el("rating").value, date = el("date").value;
    let rows = state.configs.filter(c => `${c.title} ${c.author} ${c.rig.gpu} ${c.rig.cpu} ${c.rig.ram_type} ${platforms(c.rig).join(" ")} ${allCards(c.rig).map(g=>g.name).join(" ")}`.toLowerCase().includes(query) &&
      (!gpu || allCards(c.rig).some(g => gpuKey(g.name).includes(gpuKey(gpu)))) &&
      (!platformFilter || platforms(c.rig).includes(platformFilter)) &&
      (!ramType || (c.rig.ram_type || "Unknown") === ramType || c.rig.nodes?.some(x=>x.ram_type===ramType)) &&
      (!topology || (topology === "cluster" ? nodeCount(c.rig)>1 : topology === "unified" ? isUnified(c.rig) : topology === "single" ? cards(c.rig).length === 1 && nodeCount(c.rig)===1 : topology === "multi" ? gpuCount(c.rig) > 1 : new Set(allCards(c.rig).map(g => gpuKey(g.name))).size > 1)) && (!os || c.rig.os === os || c.rig.nodes?.some(x=>x.os===os)) &&
      c.rating >= rating && (!date || c.published.slice(0, 10) >= date) &&
      (!el("ram").value || (c.rig.nodes?.length ? c.rig.nodes : [c.rig]).every(x => x.ram_gib != null && x.ram_gib <= +el("ram").value)) &&
      (!el("favorites").checked || c.favorite));
    rows.sort((a, b) => el("sort").value === "rating" ? b.rating - a.rating : el("sort").value === "speed" ?
      (warm(b.results)?.decode_tok_s || 0) - (warm(a.results)?.decode_tok_s || 0) : b.published.localeCompare(a.published));
    if (!rows.some(c=>c.id===selected)) selected=rows[0]?.id || null;
    el("count").textContent = `${rows.length} configuration${rows.length === 1 ? "" : "s"} · stored locally`;
    el("list").innerHTML = rows.length ? rows.map(c => {
      const run = runFor(c.id), measurement = warm(run?.candidate) || warm(c.results);
      const source = run ? (measurement?.simulated ? "Simulation" : "Tested on this PC") : measurement ? "Reported · unverified" : "Not benchmarked";
      return `<button class="co-card" data-id="${esc(c.id)}" aria-pressed="${c.id === selected}"><div class="co-card-head"><h3>${esc(c.title)}</h3>${c.favorite ? '<span aria-label="Favorite">★</span>' : ""}</div>
        <p>${esc(rigSummary(c.rig))} · ${esc(c.rig.os)} · ${n(c.rig.ram_gib, " GiB", 0)} ${esc(c.rig.ram_type || "Unknown RAM type")}</p>
        <span class="co-tag">${nodeCount(c.rig)>1 ? `${nodeCount(c.rig)} nodes · ` : ""}${esc(platforms(c.rig).join(" + "))}${isUnified(c.rig) ? " · unified" : ""}</span> <span class="co-tag">${source}</span> <span class="co-note">${esc(c.author)} · ${esc(c.published.slice(0, 10))}${c.rating ? ` · ${"★".repeat(c.rating)} (personal)` : ""}</span>
        ${metrics(measurement)}${run ? `<p>Versus current settings: ${delta(warm(run.baseline), warm(run.candidate), "decode_tok_s", true)} generation · ${delta(warm(run.baseline), warm(run.candidate), "ttft_s", false)} first token</p>` : '<p>Reported numbers are a reference, not a predicted gain.</p>'}</button>`;
    }).join("") : '<div class="co-empty">No configurations yet, or no filter matches.<br>Save your settings or import a shared file.</div>';
    el("list").querySelectorAll("[data-id]").forEach(b => b.onclick = () => { selected = b.dataset.id; renderList(); });
    renderDetail();
  }
  function renderDetail() {
    const c = state?.configs.find(c => c.id === selected);
    if (!c) { el("detail").innerHTML='<h2>Select a configuration</h2><p class="co-note">Import a file or adjust the filters to find a profile.</p>'; el("comparison").innerHTML='<h2>Your local comparison</h2><p class="co-note">Select a configuration to see its results.</p>'; return; }
    const running = state.job.state === "running", incompatible = c.model !== state.model;
    const changes = Object.entries(c.settings).filter(([k,v]) => state.settings[k] !== v).map(([k, v]) => `${k}: ${state.settings[k] ?? "engine default"} → ${v}`).join("\n");
    el("detail").innerHTML = `<h2>${esc(c.title)}</h2><p class="co-note">${esc(c.description || "A community tuning preset.")}</p>
      <p class="co-note">${esc(c.model)}<br>${esc(c.rig.cpu)} · ${n(c.rig.ram_gib," GiB",0)} ${esc(c.rig.ram_type || "Unknown RAM type")}<br>${esc(rigSummary(c.rig))}</p>
      ${incompatible ? '<p class="co-bad">Different model. Testing is disabled.</p>' : ''}
      <div class="co-actions"><button class="st-btn st-btn--secondary" id="co-favorite">${c.favorite ? "★ Saved" : "☆ Favorite"}</button>
      <select class="st-input" id="co-stars" aria-label="Personal star rating"><option value="0">Not rated</option>${[1,2,3,4,5].map(x => `<option value="${x}" ${c.rating === x ? "selected" : ""}>${"★".repeat(x)} · personal</option>`).join("")}</select></div>
      ${c.rig.nodes?.length ? `<h3>Cluster · ${nodeCount(c.rig)} nodes</h3><ul class="co-note">${c.rig.nodes.map(x=>`<li>${x.count} × ${esc(x.platform)} · ${n(x.ram_gib," GiB per node",0)} ${esc(x.ram_type)} · ${esc(x.memory_architecture)} · ${esc(rigSummary(x))}</li>`).join("")}</ul><p class="co-note">Link: ${esc(c.rig.interconnect)}. Catalog description only: benchmarks below run on the connected server, not across these nodes.</p>` : ""}
      <p class="co-note">${gpuCount(c.rig)} GPU / SoC device(s) across ${nodeCount(c.rig)} node(s). VRAM is per card, not a single pooled allocation. Local device selection and layer split are retained.</p><h3>Settings to try</h3><pre>${esc(changes || "Same tuning as this PC")}</pre>
      <p class="co-note">Other settings stay unchanged. Testing restarts the engine, clears its conversation cache and temporarily pauses chat/API requests. Four short generations: current/selected × cold/warm. Original settings are restored automatically.</p>
      <div class="co-actions"><button class="st-btn st-btn--primary" id="co-benchmark" ${running || incompatible ? "disabled" : ""}>Test & compare</button></div>
      <div class="co-actions"><button class="st-btn st-btn--secondary" id="co-adopt" ${running || incompatible ? "disabled" : ""}>Use for this session</button><button class="st-btn st-btn--secondary" id="co-export">Share file</button></div>
      <div class="co-actions"><button class="st-btn st-btn--secondary" id="co-restore" ${running || !state.can_restore ? "disabled" : ""}>Restore previous settings</button></div>
      <p class="co-note">Adoption is temporary; the private run config stays unchanged. Export is anonymous by default. No upload.</p>`;
    el("favorite").onclick = async () => { try { await api("preference", {id:c.id, favorite:!c.favorite}); await load(); } catch(e) { error(e); } };
    el("stars").onchange = async e => { try { await api("preference", {id:c.id, rating:+e.target.value}); await load(); } catch(err) { error(err); } };
    for (const action of ["benchmark", "adopt", "restore"]) el(action).onclick = async () => {
      try { await api(action, {id:c.id}); await load(); } catch(e) { error(e); }
    };
    el("export").onclick = async () => {
      try {
        const run = runFor(c.id);
        const body = await api("export", {id:c.id, author:c.author === "Anonymous" ? "" : c.author, ...(run ? {run_id:run.id} : {})});
        const blob = new Blob([JSON.stringify(body, null, 2) + "\n"], {type:"application/json"});
        const url = URL.createObjectURL(blob), a = document.createElement("a");
        a.href = url; a.download = "strata-community.json"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
        toast("success", "Share file exported", "Review the nickname, description and hardware before sharing.");
      } catch(e) { error(e); }
    };
    renderComparison(c);
  }
  function renderComparison(c) {
    const run = runFor(c.id);
    if (!run) {
    el("comparison").innerHTML = `<h2>Your local comparison</h2><p class="co-note">No local test for ${esc(c.title)} yet. Imported results are not measurements on your PC.</p>${warm(c.results) ? '<h3>Shared warm result · unverified</h3>' + metrics(warm(c.results)) : ''}`;
      return;
    }
    let rows = "";
    for (const phase of ["cold", "warm"]) {
      const a = run.baseline.find(r => r.phase === phase), b = run.candidate.find(r => r.phase === phase);
      rows += fields.map(([k,label,unit,high]) => `<tr><td>${phase === "cold" ? "Fresh engine" : "Repeated prompt"}</td><td>${label}</td><td>${n(a?.[k],unit)}</td><td>${n(b?.[k],unit)}</td><td>${delta(a,b,k,high)}</td></tr>`).join("");
    }
    const sameAnswers = run.baseline.every((a,i) => a.answer_sha256 === run.candidate[i]?.answer_sha256);
    const perGpu = run.candidate.flatMap(r => (r.gpu_peaks || []).map(g => `${r.phase} · GPU ${g.index}: ${n(g.peak_gib," GiB")}`));
    el("comparison").innerHTML = `<h2>${warm(run.candidate)?.simulated ? "Simulation · not a performance result" : "Measured on your PC"}</h2>
      <p class="co-note">${esc(c.title)} · ${esc(run.created)} · ${esc(run.protocol)}<br>Measured rig: ${esc(rigSummary(run.rig))} · ${esc(run.rig.os)} · ${n(run.rig.ram_gib," GiB",1)} ${esc(run.rig.ram_type || "Unknown")}</p>
      <table class="co-table"><thead><tr><th>Cache state</th><th>Metric</th><th>Current settings</th><th>Selected configuration</th><th>Change</th></tr></thead><tbody>${rows}</tbody></table>
      ${perGpu.length ? `<p class="co-note">Selected configuration, per-device sampled peaks: ${esc(perGpu.join(" · "))}</p>` : ""}
      <p class="co-note">One pass, 128 output-token cap, fixed prompt and seed. Cold = fresh engine cache, not cold disk cache. First token = first streamed text event (includes preparation, excludes engine loading). Memory = sampled system/device usage, not process-only allocation; short peaks can be missed. — means unavailable, never zero. ${sameAnswers ? "Answer hashes match." : "Answers differ: speed is not a quality score."} No extrapolation to long context.</p>`;
  }
  function status() {
    const j = state.job;
    el("status").innerHTML = state.simulated ? '<div class="co-status">Mock engine · UI demonstration only. No real performance measurements.</div>' : "";
    if (j.state === "running") {
      el("status").innerHTML += `<div class="co-status">${esc(j.phase)}${j.action === "benchmark" ? '<button class="st-btn st-btn--secondary" id="co-cancel">Cancel test</button>' : ""}</div>`;
      if (el("cancel")) el("cancel").onclick = () => api("cancel", {}).catch(error);
    } else if (j.error) el("status").innerHTML += `<div class="co-status">${esc(j.error)}</div>`;
    else if (j.state === "done") el("status").innerHTML += `<div class="co-status">${esc(j.phase)} · complete</div>`;
  }
  async function load() {
    if (loading) { reloadPending = true; return; }
    loading = true;
    try {
      state = await api("state");
      if (!selected && state.configs.length) selected = state.configs[0].id;
      el("rig").innerHTML = `<span class="co-tag">YOUR RIG</span><strong>${esc(rigSummary(state.rig))}</strong><span>${esc(state.rig.os)} · ${n(state.rig.ram_gib, " GiB", 0)} ${esc(state.rig.ram_type || "Unknown RAM type")}</span><span class="co-tag">SQLite · local only</span>`;
      const filter = el("gpu").value;
      const gpus = [...new Set([...state.gpus.map(g => g.name), ...state.configs.flatMap(c => allCards(c.rig).map(g => g.name))])].sort();
      el("gpu").innerHTML = '<option value="">All graphics cards</option>' + gpus.map(g => `<option value="${esc(g)}">${esc(g)}</option>`).join("");
      el("gpu").value = filter;
      el("catalog").innerHTML = state.gpus.map(g => `<li>${esc(g.name)} · ${g.memory_architecture === "Unified" ? `Unified memory: ${g.unified_memory_options_gib.join(" / ")} GiB options (not dedicated VRAM)` : `${g.vram_gib} GiB VRAM`} · ${esc(g.architecture)} · <a href="${esc(g.source)}" target="_blank" rel="noopener noreferrer">${esc(g.vendor)} source</a></li>`).join("");
      renderList(); renderDetail(); status();
      clearTimeout(poll);
      if (state.job.state === "running") poll = setTimeout(load, 1000);
    } catch(e) { error(e); } finally { loading = false; if (reloadPending) { reloadPending = false; load(); } }
  }
  for (const id of ["search","gpu","topology","platform","ram-type","os","rating","date","ram","sort","favorites"]) el(id).addEventListener("input", renderList);
  el("reset").onclick = () => { for(const id of ["search","gpu","topology","platform","ram-type","os","date","ram"]) el(id).value=""; el("rating").value="0"; el("sort").value="newest"; el("favorites").checked=false; renderList(); };
  el("import").onclick = () => el("file").click();
  el("file").onchange = async () => {
    const file = el("file").files[0]; if (!file) return;
    try { if (file.size > 128 * 1024) throw new Error("Maximum file size: 128 KiB"); const body = await api("import", JSON.parse(await file.text())); selected = body.id; await load(); toast("success", "Configuration imported", "No settings have been applied."); } catch(e) { error(e); }
    el("file").value="";
  };
  function addGpu(card = {name:"",vram_gib:null}) {
    const row = document.createElement("div"); row.className="co-gpu-row";
    row.innerHTML = `<label>GPU model<input class="st-input" data-gpu-name list="co-gpu-options" maxlength="120" required value="${esc(card.name)}"></label><label>VRAM (GiB)<input class="st-input" data-gpu-vram type="number" min="1" max="1024" step="any" value="${card.vram_gib ?? ''}"></label><button class="st-btn st-btn--secondary" type="button" aria-label="Remove GPU">×</button>`;
    row.querySelector("button").onclick = () => { if(el("edit-gpus").children.length > 1) row.remove(); };
    row.querySelector("[data-gpu-name]").onchange = e => { const g = state.gpus.find(g => g.name === e.target.value); if(g) row.querySelector("[data-gpu-vram]").value=g.vram_gib ?? ""; };
    el("edit-gpus").appendChild(row);
  }
  el("add-gpu").onclick = () => { if(el("edit-gpus").children.length < 16) addGpu(); };
  function addNode() {
    if(el("edit-nodes").children.length>=16) return;
    const row=document.createElement("fieldset");row.className="co-node";
    row.innerHTML=`<legend>Node group</legend><div class="co-knobs"><label>Machine<select class="st-input" data-node-platform>${state.platforms.map(x=>`<option>${esc(x)}</option>`).join("")}</select></label><label>System<select class="st-input" data-node-os><option>Linux</option><option>Windows</option><option>macOS</option><option>Unknown</option></select></label><label>Count<input class="st-input" data-node-count type="number" min="1" max="64" value="1" required></label><label>RAM per node (GiB)<input class="st-input" data-node-ram type="number" min="1" max="8192" step="any" required></label><label>RAM type<select class="st-input" data-node-type>${state.ram_types.map(x=>`<option>${esc(x)}</option>`).join("")}</select></label><label>Memory architecture<select class="st-input" data-node-memory>${state.memory_architectures.map(x=>`<option>${esc(x)}</option>`).join("")}</select></label><label>GPU / SoC<input class="st-input" data-node-gpu list="co-gpu-options" maxlength="120" required></label><label>Dedicated VRAM (GiB)<input class="st-input" data-node-vram type="number" min="1" max="1024" step="any" placeholder="Blank if unified"></label></div><button class="st-btn st-btn--secondary" type="button">Remove node group</button>`;
    row.querySelector("button").onclick=()=>row.remove();el("edit-nodes").appendChild(row);
  }
  el("add-node").onclick=addNode;
  el("create").onclick = async () => {
    if (!state) await load(); if (!state) return;
    el("knobs").innerHTML = Object.entries(state.knobs).map(([k,v]) => `<label>${esc(k)}<input class="st-input" data-knob="${esc(k)}" type="number" min="${v.min}" max="${v.max}" step="${v.integer ? 1 : .01}" value="${state.settings[k] ?? ''}" placeholder="Inherit current"></label>`).join("");
    el("edit-platform").innerHTML=state.platforms.map(x=>`<option>${esc(x)}</option>`).join("");
    el("edit-memory").innerHTML=state.memory_architectures.map(x=>`<option>${esc(x)}</option>`).join("");
    el("edit-platform").value=state.rig.platform || "PC / workstation";
    el("edit-memory").value=state.rig.memory_architecture || "Unknown";
    el("edit-os").value=state.rig.os || "Unknown";
    el("edit-nodes").innerHTML="";el("edit-interconnect").value="";
    el("edit-gpus").innerHTML=""; cards(state.rig).forEach(addGpu);
    el("gpu-options").innerHTML = state.gpus.map(g => `<option value="${esc(g.name)}"></option>`).join("");
    el("edit-ram").value = state.rig.ram_gib == null ? "" : Math.round(state.rig.ram_gib);
    el("edit-ram-type").innerHTML = (state.ram_types || ["Unknown","DDR4","DDR5"]).map(t => `<option>${esc(t)}</option>`).join("");
    el("edit-ram-type").value=state.rig.ram_type || "Unknown";
    el("editor").showModal(); el("title").focus();
  };
  el("close").onclick = () => el("editor").close();
  el("form").onsubmit = async event => {
    event.preventDefault();
    try {
      const tuning = {};
      el("knobs").querySelectorAll("input").forEach(input => { if(input.value !== "") tuning[input.dataset.knob]=Number(input.value); });
      const nodes=[...el("edit-nodes").children].map(row=>{
        const value=k=>row.querySelector(`[data-node-${k}]`).value;
        return {platform:value("platform"),os:value("os"),count:Number(value("count")),ram_gib:Number(value("ram")),ram_type:value("type"),memory_architecture:value("memory"),
          gpus:[{name:value("gpu"),vram_gib:value("vram") ? Number(value("vram")) : null}]};
      });
      const rig = {...state.rig, os:el("edit-os").value, nodes, interconnect:el("edit-interconnect").value || "Unknown", platform:el("edit-platform").value,memory_architecture:el("edit-memory").value, ram_gib:el("edit-ram").value ? Number(el("edit-ram").value) : null, ram_type:el("edit-ram-type").value,
        gpus:[...el("edit-gpus").children].map(row => ({name:row.querySelector("[data-gpu-name]").value, vram_gib:row.querySelector("[data-gpu-vram]").value ? Number(row.querySelector("[data-gpu-vram]").value) : null}))};
      const body=await api("save", {rig, title:el("title").value, author:el("author").value, description:el("description").value, settings:tuning});
      selected=body.id;el("editor").close();await load();toast("success","Configuration saved");
    } catch(e) {error(e);}
  };
  if (location.hash === "#community") load();
  return {load};
})();
