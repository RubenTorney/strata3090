/* Community benchmark desk. Local data, no chart CDN or model-generated commands. */
"use strict";
window.strataBench = (() => {
  const el = id => document.getElementById(`bw-${id}`);
  const num = (v, digits=2) => v == null ? "—" : Number(v).toFixed(digits);
  const labels = {cold:"Fresh engine",warm:"Repeated prompt","chat-1":"Code","chat-2":"Explanation","chat-3":"Advice","turn-0":"Opening",...Object.fromEntries([1,2,3,4,5].map(i=>[`turn-${i}`,`Turn ${i}`]))};
  const metrics = {ttft_s:["First-token wait","s",false],decode_tok_s:["Generation","tok/s",true],ram_peak_gib:["System RAM peak","GiB",false],vram_peak_gib:["GPU memory peak","GiB",false]};
  let state, target="", suite="quick", runId="", metric="ttft_s", pending=false, lastRun=null;
  async function act(action, data={}) {
    if(pending)return;
    pending=true;controls();
    try {
      const r=await fetch(`community/${action}`,{method:"POST",headers:headers(true),body:JSON.stringify(data)}), body=await r.json();
      if(!r.ok)throw new Error(body.error?.message || "Action failed");
      await window.strataCommunity.load();
    }catch(e){toast("error","Community",e.message,8000);}
    finally{pending=false;controls();}
  }
  function controls() {
    if(!state)return;
    const job=state.job, busy=job.state==="running", c=state.configs.find(c=>c.id===target);
    const invalid=!c || c.model!==state.model;
    const spec=state.benchmarks?.[suite];
    el("config").disabled=busy || pending;el("suite").disabled=busy || pending;
    el("start").disabled=busy || pending || invalid || !spec;
    el("load").disabled=busy || pending || invalid;
    el("undo").hidden=!state.can_restore;el("undo").disabled=busy || pending;
    el("start").textContent=runId ? "Run again" : "Run comparison";
    el("pause").hidden=el("stop").hidden=!busy || job.action!=="benchmark";
    el("pause").disabled=el("stop").disabled=pending || !!job.finishing;
    el("pause").textContent=job.pause_requested || job.paused ? "Resume" : "Pause";
    el("description").textContent=spec ? `${spec.description} Compared to your current settings. Engine restarts; chat pauses. Your settings are restored afterwards.` : "Reload the server to enable repository benchmarks.";
    let html="";
    if(busy)html=`<div class="bw-progress"><div><strong>${job.paused ? "Paused between requests" : job.pause_requested ? "Pausing after this request…" : esc(job.phase)}</strong><span>${job.completed || 0} / ${job.total || 0} measured requests</span></div><progress max="${job.total || 1}" value="${job.completed || 0}" aria-label="Benchmark progress"></progress><p class="co-note">${job.paused ? "Resume to continue, or stop to restore settings. Pausing keeps the engine reserved." : "You can close this window; the test continues and results are saved locally."}</p></div>`;
    else if(job.error==="Cancelled")html='<p class="co-note">Stopped · original settings restored. No partial comparison saved.</p>';
    else if(job.error)html=`<p class="co-status">${esc(job.error)}${job.action==="benchmark" ? " · Incomplete comparisons are not saved as results." : ""}</p>`;
    else if(job.state==="done")html=`<p class="co-note">${job.action==="benchmark" ? "Comparison saved · original settings restored" : job.action==="adopt" ? "Configuration loaded for this session" : job.action==="restore" ? "Previous configuration restored" : "Current model reloaded"}</p>`;
    el("status").innerHTML=html;
  }
  function renderReport() {
    const run=state?.runs.find(r=>r.id===runId);
    el("export").disabled=!run;
    if(!run){el("report").innerHTML='<div class="co-empty"><h3>A clear view of what changes</h3><p>Choose a configuration and a benchmark. First-token latency, generation speed and memory will appear here, request by request.</p></div>';return;}
    const sim=[...run.baseline,...run.candidate].some(r=>r.simulated);
    const name=state.configs.find(c=>c.id===run.config_id)?.title || "Saved configuration";
    const same=run.baseline.every((a,i)=>a.answer_sha256===run.candidate[i]?.answer_sha256);
    const total=key=>run.candidate.reduce((s,r)=>s+(r[key] || 0),0);
    el("report").innerHTML=`<div class="bw-report-title"><h3>${esc(name)}</h3><span class="co-tag">${sim ? "SIMULATION · NO PERFORMANCE CLAIM" : "LOCAL MEASUREMENT"}</span></div><p class="co-note">${esc(run.created)} · ${esc(run.protocol)} · one pass per setup</p>
      <div class="bw-kpis"><div><strong>${run.candidate.length}</strong><span>measured requests / setup</span></div><div><strong>${num(total("output_tokens"),0)}</strong><span>candidate output tokens</span></div><div><strong>${num(total("duration_s"),1)} s</strong><span>candidate request time</span></div></div>
      <p class="co-note">${same ? "Answer hashes match." : "Answers differ between setups; speed alone does not establish answer quality."}</p>
      <div class="bw-chart-head"><label>Compare<select class="st-input" id="bw-metric">${Object.entries(metrics).map(([key,[label]])=>`<option value="${key}" ${key===metric ? "selected" : ""}>${label}</option>`).join("")}</select></label><div class="bw-legend"><span><i class="bw-base"></i>Baseline at test time</span><span><i class="bw-candidate"></i>Selected configuration</span></div></div><div id="bw-chart"></div>
      <details class="co-method"><summary>Every request · tokens, latency and memory</summary><div class="bw-table-scroll" tabindex="0" role="region" aria-label="Request metrics"><table class="co-table"><thead><tr><th>Request / setup</th><th>Input tokens</th><th>Reused</th><th>Output</th><th>First token</th><th>Generation</th><th>RAM / VRAM peak</th></tr></thead><tbody>${run.baseline.flatMap((a,i)=>[a,run.candidate[i]].map((r,j)=>r ? `<tr><td>${esc(labels[r.phase] || r.phase)} · ${j ? "selected" : "baseline"}</td><td>${num(r.prompt_tokens,0)}</td><td>${num(r.reused_tokens,0)}</td><td>${num(r.output_tokens,0)}</td><td>${num(r.ttft_s)} s</td><td>${num(r.decode_tok_s,1)} tok/s</td><td>${num(r.ram_peak_gib,1)} / ${num(r.vram_peak_gib,1)} GiB</td></tr>` : "")).join("")}</tbody></table></div></details>
      <details class="co-method"><summary>Protocol and settings</summary><p class="co-note">${esc(run.source || "serve/community.py")} · ${esc(run.benchmark || "quick")}. Repository scenarios reuse the repository's suite with an in-process transport; TTFT includes preparation, excludes HTTP transport and engine loading. Warm-up and calibration requests are excluded. One pass is an exploration, not a statistical verdict. Pause time is excluded; the engine keeps its cache while paused.</p><p class="co-note">Memory is sampled whole-system/device use, not process allocation; short peaks can be missed. — means unavailable. ${same ? "Answer hashes match." : "Answers differ. Speed does not measure answer quality; agent continuations may differ in length."}</p><pre>${esc(JSON.stringify({baseline:run.baseline_settings,selected:run.effective_settings},null,2))}</pre></details>`;
    document.getElementById("bw-metric").onchange=e=>{metric=e.target.value;renderChart(run);};
    renderChart(run);
  }
  function renderChart(run) {
    const [label,unit,high]=metrics[metric], values=[...run.baseline,...run.candidate].map(r=>r[metric]).filter(v=>v!=null);
    const max=Math.max(0,...values), precision=metric==="ttft_s" ? 3 : metric==="decode_tok_s" ? 1 : 2;
    const width=v=>v==null || !max ? 0 : Math.max(0,Math.min(100,v/max*100));
    const chart=document.getElementById("bw-chart");
    chart.innerHTML=`<p class="co-note">${label} · ${high ? "higher" : "lower"} is better · shared scale from zero</p>${run.baseline.map((a,i)=>{
      const b=run.candidate[i], aligned=b && a.protocol===b.protocol && a.prompt_tokens===b.prompt_tokens && !a.simulated && !b.simulated;
      const change=aligned && a[metric]>0 && b[metric]!=null ? (b[metric]/a[metric]-1)*100 : null;
      return `<div class="bw-chart-row"><div class="bw-chart-label"><strong>${esc(labels[a.phase] || a.phase)}</strong><span class="${change==null || change===0 ? "" : (high ? change>0 : change<0) ? "co-good" : "co-bad"}">${change==null ? "" : `${change>0 ? "+" : ""}${change.toFixed(1)}%`}</span></div>${[a,b].map((r,j)=>`<div class="bw-bar-line"><div class="bw-track"><div class="bw-bar ${j ? "bw-candidate" : "bw-base"}" style="width:${width(r?.[metric])}%"></div></div><span>${num(r?.[metric],precision)} ${unit}</span></div>`).join("")}</div>`;
    }).join("")}<p class="co-note">Percent changes are shown only for matched prompt lengths and real measurements.</p>`;
  }
  function update(next) {
    state=next;
    if(state.job.state==="running" && state.job.action==="benchmark"){target=state.job.config_id;suite=state.job.benchmark || "quick";}
    const valid=state.configs.filter(c=>c.model===state.model && !c.rig.nodes?.length && c.rig.memory_architecture!=="Unified" && !["Mac mini","Mac Studio","DGX Spark"].includes(c.rig.platform));
    if(!valid.some(c=>c.id===target))target=valid[0]?.id || "";
    el("config").innerHTML=valid.length ? valid.map(c=>`<option value="${esc(c.id)}">${esc(c.title)}</option>`).join("") : '<option value="">Save or import a compatible setup first</option>';
    el("config").value=target;
    el("suite").innerHTML=Object.entries(state.benchmarks || {}).map(([id,s])=>`<option value="${id}">${esc(s.title)}</option>`).join("");el("suite").value=suite;
    if(state.job.run_id && state.job.run_id!==lastRun){lastRun=state.job.run_id;runId=lastRun;}
    if(!runId)runId=state.runs.find(r=>r.config_id===target)?.id || "";
    el("history").innerHTML='<option value="">'+(state.runs.length ? "Choose a saved comparison" : "No comparisons yet")+'</option>'+state.runs.map(r=>`<option value="${esc(r.id)}">${esc(r.created.replace("T"," ").slice(0,19))} · ${esc(state.configs.find(c=>c.id===r.config_id)?.title || "Configuration")} · ${esc(r.benchmark || "quick")}</option>`).join("");
    el("history").value=runId;controls();
    const signature=`${runId}/${metric}`;
    if(el("report").dataset.rendered!==signature){renderReport();el("report").dataset.rendered=signature;}
  }
  function open(id) {
    if(!state)return;
    if(id){target=id;runId=state.runs.find(r=>r.config_id===id)?.id || "";}
    update(state);document.getElementById("co-inspector").close();el("dialog").showModal();el("body").scrollTop=0;
  }
  el("close").onclick=()=>el("dialog").close();
  el("config").onchange=e=>{target=e.target.value;runId=state.runs.find(r=>r.config_id===target)?.id || "";el("history").value=runId;controls();renderReport();};
  el("suite").onchange=e=>{suite=e.target.value;controls();};
  el("history").onchange=e=>{
    runId=e.target.value;
    const run=state.runs.find(r=>r.id===runId);
    if(run && state.job.state!=="running"){target=run.config_id;suite=run.benchmark || "quick";el("config").value=target;el("suite").value=suite;}
    controls();renderReport();
  };
  el("start").onclick=()=>act("benchmark",{id:target,benchmark:suite});
  el("load").onclick=()=>act("adopt",{id:target});
  el("undo").onclick=()=>act("restore");
  el("pause").onclick=()=>act(state.job.pause_requested || state.job.paused ? "resume" : "pause");
  el("stop").onclick=()=>act("cancel");
  el("export").onclick=()=>{
    const run=state.runs.find(r=>r.id===runId);if(!run)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(run,null,2)+"\n"],{type:"application/json"})),a=document.createElement("a");
    a.href=url;a.download=`strata-benchmark-${run.benchmark || "quick"}-${run.created.slice(0,10)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  return {update,open,act};
})();
