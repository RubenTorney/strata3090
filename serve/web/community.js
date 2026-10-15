/* Optional Community tab: use Strata's helpers, components and API-key handling. */
"use strict";
window.strataCommunity = (() => {
  let state = null, selected = null, poll = null, loading = false, reloadPending = false, viewMode = "grid", incomingChecked = false;
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
  const supported = rig => !rig.nodes?.length && !isUnified(rig) && !["Mac mini","Mac Studio","DGX Spark"].includes(rig.platform);
  const runtimeTags = r => [r?.version && `Strata ${r.version}`, r?.max_context && `${n(r.max_context / 1024,"K",0)} context`, r?.kv && `KV ${r.kv}`, r?.images ? "Vision" : null].filter(Boolean);
  function ratingText(c) { return c.rating ? `★ ${c.rating.toFixed(1)} / 5 · 1 local vote` : "☆ No rating · 0 votes"; }
  function gainText(c) {
    const run = runFor(c.id), a=warm(run?.baseline), b=warm(run?.candidate);
    if (comparable(a,b)) return `<strong>${delta(a,b,"decode_tok_s",true)} generation</strong><span>${delta(a,b,"ttft_s",false)} first-token wait</span><small>Measured pair · ${esc(run.created.slice(0,10))}</small>`;
    return `<strong>Gain not measured</strong><span>${warm(c.results) ? "Shared result · unverified on your PC" : "Test this setup against yours"}</span><small>No predicted percentage without a local comparison</small>`;
  }
  function openConfig(id) { selected=id; renderList(); el("inspector").showModal(); }
  function renderList() {
    if (!state) return;
    const query=el("search").value.toLowerCase().replace(/tv100/g,"v100"), gpu=el("gpu").value, topology=el("topology").value, os=el("os").value;
    let rows=state.configs.filter(c=>supported(c.rig) && `${c.title} ${c.author} ${c.model} ${c.rig.cpu} ${rigSummary(c.rig)}`.toLowerCase().includes(query) &&
      (!gpu || cards(c.rig).some(g=>gpuKey(g.name).includes(gpuKey(gpu)))) &&
      (!topology || (topology==="single" ? cards(c.rig).length===1 : topology==="multi" ? cards(c.rig).length>1 : new Set(cards(c.rig).map(g=>gpuKey(g.name))).size>1)) &&
      (!os || c.rig.os===os) && c.rating>=+el("rating").value && (!el("date").value || c.published.slice(0,10)>=el("date").value) &&
      (!el("ram").value || (c.rig.ram_gib!=null && c.rig.ram_gib<=+el("ram").value)) && (!el("favorites").checked || c.favorite));
    rows.sort((a,b)=>el("sort").value==="rating" ? b.rating-a.rating : el("sort").value==="speed" ? (warm(runFor(b.id)?.candidate)?.decode_tok_s || warm(b.results)?.decode_tok_s || 0)-(warm(runFor(a.id)?.candidate)?.decode_tok_s || warm(a.results)?.decode_tok_s || 0) : b.published.localeCompare(a.published));
    el("count").textContent=`${rows.length} setups · local library · ratings are personal`;
    el("list").classList.toggle("co-list--rows",viewMode==="list");
    el("list").innerHTML=rows.length ? rows.map(c=>{
      const measured=warm(runFor(c.id)?.candidate) || warm(c.results);
      return `<article class="co-card"><div class="co-card-head"><span class="co-tag">${cards(c.rig).length>1 ? `${cards(c.rig).length} GPUs` : "Single GPU"}</span><button class="co-save-icon st-btn st-btn--secondary" data-fav="${esc(c.id)}" aria-label="${c.favorite ? "Remove favorite" : "Save favorite"}: ${esc(c.title)}" aria-pressed="${!!c.favorite}">${c.favorite ? "★" : "☆"}</button></div><h3><button class="co-title-button" data-open="${esc(c.id)}">${esc(c.title)}</button></h3><p class="co-card-gpu">${esc(rigSummary(c.rig))}</p><div class="co-tags"><span class="co-tag">${esc(c.rig.os)}</span><span class="co-tag">${n(c.rig.ram_gib," GiB RAM",0)}</span><span class="co-tag">${esc(c.model)}</span>${runtimeTags(c.runtime).map(t=>`<span class="co-tag">${esc(t)}</span>`).join("")}</div><div class="co-gain">${gainText(c)}</div><div class="co-card-stats"><span>${n(measured?.decode_tok_s," tok/s",1)}</span><span>${n(measured?.ttft_s," s first token",2)}</span></div><p class="co-note co-rating">${ratingText(c)}</p><div class="co-card-foot"><span class="co-note">${esc(c.author)} · ${esc(c.published.slice(0,10))}</span><button class="st-btn st-btn--primary" data-open="${esc(c.id)}">View & test →</button></div></article>`;
    }).join("") : '<div class="co-empty"><h3>Your next setup starts here</h3><p>Share your current setup or import a configuration. Try fewer filters if your library is empty.</p></div>';
    el("list").querySelectorAll("[data-open]").forEach(b=>b.onclick=()=>openConfig(b.dataset.open));
    el("list").querySelectorAll("[data-fav]").forEach(b=>b.onclick=async()=>{try {const c=state.configs.find(c=>c.id===b.dataset.fav);await api("preference",{id:c.id,favorite:!c.favorite});await load();}catch(e){error(e);}});
    renderDetail();
  }
  function renderDetail() {
    const c = state?.configs.find(c => c.id === selected);
    if (!c) { el("detail").innerHTML='<h2>Select a configuration</h2><p class="co-note">Import a file or adjust the filters to find a profile.</p>'; el("comparison").innerHTML='<h2>Your local comparison</h2><p class="co-note">Select a configuration to see its results.</p>'; return; }
    const running = state.job.state === "running", incompatible = c.model !== state.model || !supported(c.rig);
    const changes = Object.entries(c.settings).filter(([k,v]) => state.settings[k] !== v).map(([k, v]) => `${k}: ${state.settings[k] ?? "engine default"} → ${v}`).join("\n");
    el("detail").innerHTML = `<h2>${esc(c.title)}</h2><p class="co-note">${esc(c.description || "A community tuning preset.")}</p>
      <p class="co-note">${esc(c.model)} · ${runtimeTags(c.runtime).map(esc).join(" · ")}<br>${esc(c.rig.cpu)} · ${n(c.rig.ram_gib," GiB",0)} ${esc(c.rig.ram_type || "Unknown RAM type")}<br>${esc(rigSummary(c.rig))}</p>
      ${incompatible ? '<p class="co-bad">Different model or unsupported rig profile. Testing is disabled.</p>' : ''}
      <div class="co-actions"><button class="st-btn st-btn--secondary" id="co-favorite">${c.favorite ? "★ Saved" : "☆ Favorite"}</button>
      <select class="st-input" id="co-stars" aria-label="Personal star rating"><option value="0">Not rated</option>${[1,2,3,4,5].map(x => `<option value="${x}" ${c.rating === x ? "selected" : ""}>${"★".repeat(x)} · personal</option>`).join("")}</select></div>
      ${c.rig.nodes?.length ? `<h3>Cluster · ${nodeCount(c.rig)} nodes</h3><ul class="co-note">${c.rig.nodes.map(x=>`<li>${x.count} × ${esc(x.platform)} · ${n(x.ram_gib," GiB per node",0)} ${esc(x.ram_type)} · ${esc(x.memory_architecture)} · ${esc(rigSummary(x))}</li>`).join("")}</ul><p class="co-note">Link: ${esc(c.rig.interconnect)}. Catalog description only: benchmarks below run on the connected server, not across these nodes.</p>` : ""}
      <p class="co-note">${cards(c.rig).length} GPU(s). Testing keeps your local devices and layer split.</p><details><summary>Settings to try · ${Object.keys(c.settings).length} values</summary><pre>${esc(changes || "Same tuning as this PC")}</pre></details>
      <p class="co-note">Other settings stay unchanged. Testing restarts the engine, clears its conversation cache and temporarily pauses chat/API requests. Four short generations: current/selected × cold/warm. Original settings are restored automatically.</p>
      <div class="co-actions"><button class="st-btn st-btn--primary" id="co-benchmark" ${running || incompatible ? "disabled" : ""}>Test & compare</button></div>
      <div class="co-actions"><button class="st-btn st-btn--secondary" id="co-adopt" ${running || incompatible ? "disabled" : ""}>Use for this session</button><button class="st-btn st-btn--secondary" id="co-export">Download file</button><button class="st-btn st-btn--secondary" id="co-link">Share link</button></div>
      <div class="co-actions"><button class="st-btn st-btn--secondary" id="co-restore" ${running || !state.can_restore ? "disabled" : ""}>Restore previous settings</button></div>
      <p class="co-note">Adoption is temporary; the private run config stays unchanged. Export is anonymous by default. No upload.</p>`;
    el("favorite").onclick = async () => { try { await api("preference", {id:c.id, favorite:!c.favorite}); await load(); } catch(e) { error(e); } };
    el("stars").onchange = async e => { try { await api("preference", {id:c.id, rating:+e.target.value}); await load(); } catch(err) { error(err); } };
    for (const action of ["benchmark", "adopt", "restore"]) el(action).onclick = async () => {
      try { await api(action, {id:c.id}); await load(); } catch(e) { error(e); }
    };
    el("link").onclick = async () => {
      try {
        const run=runFor(c.id), body=await api("export",{id:c.id,author:c.author==="Anonymous" ? "" : c.author,...(run ? {run_id:run.id} : {})});
        const bytes=new TextEncoder().encode(JSON.stringify(body));
        const encoded=btoa(Array.from(bytes,x=>String.fromCharCode(x)).join(""));
        if(encoded.length>16000) throw new Error("This configuration is too large for a link. Download its file instead.");
        el("share-text").value=`${location.origin}${location.pathname}?setup=${encodeURIComponent(encoded)}#community`;
        el("share-dialog").showModal();
      } catch(e) {error(e);}
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
      <table class="co-table"><thead><tr><th>Cache state</th><th>Metric</th><th>Baseline at test time</th><th>Selected configuration</th><th>Change</th></tr></thead><tbody>${rows}</tbody></table>
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
      el("rig").innerHTML = `<div><span class="co-tag">DETECTED AUTOMATICALLY</span><strong>${esc(rigSummary(state.rig))}</strong><span class="co-note">${esc(state.rig.cpu)} · ${esc(state.rig.os)} · ${n(state.rig.ram_gib," GiB RAM",0)}</span></div><div class="co-tags"><span class="co-tag">${esc(state.model)}</span>${runtimeTags(state.runtime).map(t=>`<span class="co-tag">${esc(t)}</span>`).join("")}</div>`;
      const filter = el("gpu").value;
      const gpus = [...new Set([...state.gpus.filter(g=>g.memory_architecture!=="Unified").map(g => g.name), ...state.configs.filter(c=>supported(c.rig)).flatMap(c => allCards(c.rig).map(g => g.name))])].sort();
      el("gpu").innerHTML = '<option value="">All graphics cards</option>' + gpus.map(g => `<option value="${esc(g)}">${esc(g)}</option>`).join("");
      el("gpu").value = filter;
      el("catalog").innerHTML = state.gpus.filter(g=>g.memory_architecture!=="Unified").map(g => `<li>${esc(g.name)} · ${g.memory_architecture === "Unified" ? `Unified memory: ${g.unified_memory_options_gib.join(" / ")} GiB options (not dedicated VRAM)` : `${g.vram_gib} GiB VRAM`} · ${esc(g.architecture)} · <a href="${esc(g.source)}" target="_blank" rel="noopener noreferrer">${esc(g.vendor)} source</a></li>`).join("");
      renderList(); renderDetail(); status(); checkIncoming();
      clearTimeout(poll);
      if (state.job.state === "running") poll = setTimeout(load, 1000);
    } catch(e) { error(e); } finally { loading = false; if (reloadPending) { reloadPending = false; load(); } }
  }
  for (const id of ["search","gpu","topology","os","rating","date","ram","sort","favorites"]) el(id).addEventListener("input", renderList);
  el("reset").onclick = () => { for(const id of ["search","gpu","topology","os","date","ram"]) el(id).value=""; el("rating").value="0"; el("sort").value="newest"; el("favorites").checked=false; renderList(); };
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
  el("create").onclick = async () => {
    await load(); if (!state) return;
    el("knobs").innerHTML = Object.entries(state.knobs).map(([k,v]) => `<label>${esc(k)}<input class="st-input" data-knob="${esc(k)}" type="number" min="${v.min}" max="${v.max}" step="${v.integer ? 1 : .01}" value="${state.settings[k] ?? ''}" placeholder="Inherit current"></label>`).join("");
    el("override").checked=false;el("edit-os").value=state.rig.os || "Unknown";
    el("autofill").innerHTML=`<strong>${esc(rigSummary(state.rig))}</strong><p>${esc(state.rig.cpu)} · ${n(state.rig.ram_gib," GiB RAM",0)} · ${esc(state.rig.os)}</p><p>${esc(state.model)} · ${runtimeTags(state.runtime).map(esc).join(" · ")}</p><span class="co-note">Captured from Your PC and Model & engine when saved. No local paths or credentials.</span>`;
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
      let rig;
      if(el("override").checked) rig={...state.rig,os:el("edit-os").value,ram_gib:el("edit-ram").value ? Number(el("edit-ram").value) : null,ram_type:el("edit-ram-type").value,
        gpus:[...el("edit-gpus").children].map(row=>({name:row.querySelector("[data-gpu-name]").value,vram_gib:row.querySelector("[data-gpu-vram]").value ? Number(row.querySelector("[data-gpu-vram]").value) : null}))};
      const body=await api("save",{...(rig ? {rig} : {}),title:el("title").value,author:el("author").value,description:el("description").value,settings:tuning});
      selected=body.id;el("editor").close();await load();toast("success","Configuration saved");
    } catch(e) {error(e);}
  };
  el("filter-toggle").onclick=()=>{const open=el("filter-toggle").closest(".co-filters").classList.toggle("co-expanded");el("filter-toggle").setAttribute("aria-expanded",String(open));el("filter-toggle").textContent=open ? "Hide filters" : "Filters";};
  el("dismiss").onclick=()=>el("inspector").close();
  for(const [id,mode] of [["grid","grid"],["rows","list"]]) el(id).onclick=()=>{viewMode=mode;el("grid").setAttribute("aria-pressed",String(mode==="grid"));el("rows").setAttribute("aria-pressed",String(mode==="list"));renderList();};
  el("share-close").onclick=()=>el("share-dialog").close();
  el("copy").onclick=async()=>{try{await navigator.clipboard.writeText(el("share-text").value);toast("success","Link copied");}catch(e){el("share-text").select();toast("info","Select and copy the link","Clipboard access is unavailable.");}};
  function checkIncoming() {
    if(incomingChecked) return;incomingChecked=true;
    const params=new URLSearchParams(location.search), encoded=params.get("setup");if(!encoded)return;
    try {
      if(encoded.length>16000)throw new Error("Shared link is too large. Import the JSON file instead.");
      const doc=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))));
      const dialog=document.createElement("dialog");dialog.className="co-dialog";
      dialog.innerHTML=`<h2>Import shared configuration?</h2><p>${esc(String(doc.title || "Unnamed setup"))}</p><p class="co-note">${esc(String(doc.model || "Unknown model"))}. This adds a copy to your library. Nothing is applied or executed.</p><div class="co-actions"><button class="st-btn st-btn--secondary" data-dismiss>Cancel</button><button class="st-btn st-btn--primary" data-import>Import configuration</button></div>`;
      document.body.appendChild(dialog);dialog.showModal();dialog.querySelector("[data-dismiss]").onclick=()=>dialog.close();
      dialog.querySelector("[data-import]").onclick=async()=>{try{const r=await api("import",doc);selected=r.id;dialog.close();await load();openConfig(r.id);}catch(e){error(e);}};
      dialog.addEventListener("close",()=>dialog.remove());
    }catch(e){error(e);}
    params.delete("setup");history.replaceState(null,"",location.pathname+(params.size ? "?"+params : "")+"#community");
  }
  if (location.hash === "#community") load();
  return {load};
})();
