#!/usr/bin/env python3
"""PC Monitor – live GPU & Ollama stats via WebSocket + Chart.js"""

import asyncio
import json
import subprocess
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# ── data collection helpers ────────────────────────────────────────────────

async def run(cmd: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    return stdout.decode(errors="replace").strip()


async def gpu_stats() -> dict:
    raw = await run(
        "nvidia-smi --query-gpu=utilization.gpu,"
        "memory.used,memory.total,power.draw "
        "--format=csv,noheader,nounits"
    )
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 4:
        return {}
    return {
        "gpu_util": float(parts[0]),
        "vram_used": float(parts[1]),
        "vram_total": float(parts[2]),
        "power": float(parts[3]),
    }


async def ollama_cpu() -> float:
    raw = await run("ps -eo pid,pcpu,comm,args --no-headers")
    total = 0.0
    for line in raw.splitlines():
        if "ollama" in line.lower():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total += float(parts[1])
                except ValueError:
                    pass
    return round(total, 1)


async def collect() -> dict:
    g, o = await asyncio.gather(gpu_stats(), ollama_cpu())
    return {"t": round(time.time() * 1000), "gpu": g, "ollama_cpu": o}


# ── websocket endpoint ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await collect()
            await ws.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, Exception):
        pass


# ── frontend ───────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PC Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{background:#111;color:#eee;font-family:system-ui,sans-serif;padding:10px;
     display:flex;flex-direction:column}
h1{text-align:center;font-size:1.1rem;color:#7cf;flex-shrink:0}
#status{text-align:center;color:#f55;font-size:.75rem;min-height:1em;flex-shrink:0}
.grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;
      gap:10px;flex:1;min-height:0}
.card{background:#1a1a2e;border-radius:8px;padding:8px 10px;
      display:flex;flex-direction:column;min-height:0}
.card-hdr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.card-hdr h2{font-size:.75rem;color:#aaa;text-transform:uppercase;letter-spacing:.5px}
.card-hdr .val{font-size:1.1rem;font-weight:700;font-variant-numeric:tabular-nums}
.chart-wrap{flex:1;min-height:0;position:relative}
.chart-wrap canvas{position:absolute;inset:0;width:100%!important;height:100%!important}
</style>
</head>
<body>
<h1>PC Monitor</h1>
<div id="status"></div>
<div class="grid">
  <div class="card">
    <div class="card-hdr"><h2>GPU Utilization</h2><span class="val" id="vGpu" style="color:#7cf">–</span></div>
    <div class="chart-wrap"><canvas id="cGpu"></canvas></div>
  </div>
  <div class="card">
    <div class="card-hdr"><h2>VRAM Usage</h2><span class="val" id="vVram" style="color:#f7a">–</span></div>
    <div class="chart-wrap"><canvas id="cVram"></canvas></div>
  </div>
  <div class="card">
    <div class="card-hdr"><h2>Power Draw</h2><span class="val" id="vPow" style="color:#bf5">–</span></div>
    <div class="chart-wrap"><canvas id="cPow"></canvas></div>
  </div>
  <div class="card">
    <div class="card-hdr"><h2>Ollama CPU</h2><span class="val" id="vOllama" style="color:#5df">–</span></div>
    <div class="chart-wrap"><canvas id="cOllama"></canvas></div>
  </div>
</div>
<script>
const MAX=300;
const STATUS=document.getElementById("status");

Chart.defaults.color="#ccc";
Chart.defaults.borderColor="#333";
Chart.defaults.font.size=10;
const gridOpts={grid:{color:"#222"}};

function mkLine(ctx,color,sugMax){
  return new Chart(ctx,{type:"line",data:{labels:[],datasets:[{
    data:[],borderColor:color,borderWidth:1.5,pointRadius:0,tension:.3,fill:false
  }]},options:{animation:false,responsive:true,maintainAspectRatio:false,
    scales:{x:{display:false},y:{min:0,suggestedMax:sugMax,...gridOpts}},
    plugins:{legend:{display:false}}}});
}

const chGpu   =mkLine(document.getElementById("cGpu"),   "#7cf",100);
const chVram  =mkLine(document.getElementById("cVram"),  "#f7a",33000);
const chPow   =mkLine(document.getElementById("cPow"),   "#bf5",500);
const chOllama=mkLine(document.getElementById("cOllama"),"#5df",800);

const vGpu=document.getElementById("vGpu");
const vVram=document.getElementById("vVram");
const vPow=document.getElementById("vPow");
const vOllama=document.getElementById("vOllama");

function push(chart,val){
  chart.data.labels.push("");
  chart.data.datasets[0].data.push(val);
  if(chart.data.labels.length>MAX){
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update();
}

let ws,retry=0;
function connect(){
  ws=new WebSocket("ws://"+location.host+"/ws");
  ws.onopen=()=>{STATUS.textContent="";retry=0;};
  ws.onclose=()=>{STATUS.textContent="Disconnected – reconnecting…";setTimeout(connect,Math.min(1000*2**retry++,8000));};
  ws.onerror=()=>ws.close();
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    const g=d.gpu;if(!g||!g.gpu_util&&g.gpu_util!==0)return;
    push(chGpu,g.gpu_util);
    push(chVram,g.vram_used);
    push(chPow,g.power);
    push(chOllama,d.ollama_cpu);
    vGpu.textContent=g.gpu_util.toFixed(0)+"%";
    vVram.textContent=(g.vram_used/1024).toFixed(1)+" / "+(g.vram_total/1024).toFixed(1)+" GiB";
    vPow.textContent=g.power.toFixed(1)+" W";
    vOllama.textContent=d.ollama_cpu.toFixed(1)+"%";
  };
}
connect();
</script>
</body>
</html>
"""


@app.get("/")
async def index():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
