//! PC Monitor — live GPU & Ollama stats streamed over WebSocket at 30 Hz.
//!
//! Architecture:
//!   • A single global collector runs at 30 Hz from the moment the server
//!     starts, regardless of whether any clients are connected.
//!   • Snapshots are stored in a global ring buffer (30 s of history).
//!   • On WebSocket connect, the full history is sent as a JSON array,
//!     then live snapshots stream individually.
//!   • Multiple clients share the same data — no per-client collection.
//!
//! CPU % for Ollama is computed from instantaneous `/proc/[pid]/stat` tick
//! deltas, not from the lifetime-averaged `pcpu` field of `ps`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Html,
    routing::get,
    Router,
};
use serde::Serialize;
use thiserror::Error;
use tokio::process::Command;
use tokio::sync::{broadcast, RwLock};

// ── Constants ──────────────────────────────────────────────────────────────

const LISTEN_ADDR: &str = "0.0.0.0:8765";
const POLL_INTERVAL: Duration = Duration::from_micros(33_333); // ~30 Hz
const MAX_HISTORY: usize = 900; // 30 s × 30 Hz
const CMD_TIMEOUT: Duration = Duration::from_secs(5);
const WS_SEND_TIMEOUT: Duration = Duration::from_secs(10);

// ── Global state ───────────────────────────────────────────────────────────

struct Global {
    /// Ring buffer of the last 30 seconds of pre-serialised JSON snapshots.
    history: RwLock<VecDeque<Arc<str>>>,
    /// Broadcast channel — each message is a pre-serialised JSON snapshot.
    tx: broadcast::Sender<Arc<str>>,
}

static STATE: LazyLock<Global> = LazyLock::new(|| {
    // 128 slots: even a client that stalls for ~4 s at 30 Hz won't lag out.
    let (tx, _) = broadcast::channel(128);
    Global {
        history: RwLock::new(VecDeque::with_capacity(MAX_HISTORY)),
        tx,
    }
});

// ── Error types ────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
enum CollectError {
    #[error("`{cmd}` timed out after {timeout_secs}s")]
    Timeout { cmd: String, timeout_secs: u64 },

    #[error("failed to execute `{cmd}`: {source}")]
    Spawn { cmd: String, source: std::io::Error },

    #[error("`{cmd}` exited with {status}: {stderr}")]
    NonZeroExit {
        cmd: String,
        status: std::process::ExitStatus,
        stderr: String,
    },

    #[error(
        "`{cmd}` produced unparseable output — \
         expected >= {expected} comma-separated fields, got {got}: \"{line}\""
    )]
    BadFieldCount {
        cmd: &'static str,
        expected: usize,
        got: usize,
        line: String,
    },

    #[error("`{cmd}` field {field} is not a valid f64: \"{value}\"")]
    BadFloat {
        cmd: &'static str,
        field: &'static str,
        value: String,
    },
}

// ── Data types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct Snapshot {
    /// Unix epoch milliseconds.
    t: u64,
    /// `None` if nvidia-smi failed on this tick.
    gpu: Option<GpuStats>,
    /// Instantaneous sum of CPU % across all Ollama processes.
    ollama_cpu: f64,
    /// Sum of RSS (MiB) across all Ollama processes.
    ollama_ram_mib: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GpuStats {
    gpu_util: f64,
    vram_used: f64,
    vram_total: f64,
    power: f64,
}

// ── Shell command helper ───────────────────────────────────────────────────

async fn run_cmd(cmd: &str) -> Result<String, CollectError> {
    let child = Command::new("sh")
        .args(["-c", cmd])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| CollectError::Spawn {
            cmd: cmd.to_owned(),
            source: e,
        })?;

    // wait_with_output() takes ownership of child.  On timeout the future
    // is dropped, and kill_on_drop(true) ensures the process is killed.
    match tokio::time::timeout(CMD_TIMEOUT, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            if !output.status.success() {
                return Err(CollectError::NonZeroExit {
                    cmd: cmd.to_owned(),
                    status: output.status,
                    stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
                });
            }
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
        }
        Ok(Err(e)) => Err(CollectError::Spawn {
            cmd: cmd.to_owned(),
            source: e,
        }),
        Err(_) => Err(CollectError::Timeout {
            cmd: cmd.to_owned(),
            timeout_secs: CMD_TIMEOUT.as_secs(),
        }),
    }
}

fn parse_f64(s: &str, cmd: &'static str, field: &'static str) -> Result<f64, CollectError> {
    s.parse::<f64>().map_err(|_| CollectError::BadFloat {
        cmd,
        field,
        value: s.to_owned(),
    })
}

// ── GPU collection ─────────────────────────────────────────────────────────

async fn gpu_stats() -> Result<GpuStats, CollectError> {
    const CMD: &str = "nvidia-smi \
        --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits";

    let raw = run_cmd(CMD).await?;
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();

    if parts.len() < 4 {
        return Err(CollectError::BadFieldCount {
            cmd: "nvidia-smi (gpu)",
            expected: 4,
            got: parts.len(),
            line: raw,
        });
    }

    Ok(GpuStats {
        gpu_util: parse_f64(parts[0], "nvidia-smi", "utilization.gpu")?,
        vram_used: parse_f64(parts[1], "nvidia-smi", "memory.used")?,
        vram_total: parse_f64(parts[2], "nvidia-smi", "memory.total")?,
        power: parse_f64(parts[3], "nvidia-smi", "power.draw")?,
    })
}

// ── Ollama /proc collection ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ProcTicks {
    ticks: u64,
}

#[derive(Debug, Clone)]
struct OllamaSnapshot {
    per_pid: HashMap<u32, ProcTicks>,
    when: std::time::Instant,
    rss_mib: f64,
}

fn page_size_bytes() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        assert!(ps > 0, "sysconf(_SC_PAGESIZE) returned {ps} — kernel bug?");
        ps as u64
    })
}

fn clk_tck() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        assert!(tck > 0, "sysconf(_SC_CLK_TCK) returned {tck} — kernel bug?");
        tck as u64
    })
}

/// Scan `/proc` for all Ollama processes and read their CPU ticks + RSS.
fn read_ollama_snapshot() -> OllamaSnapshot {
    let when = std::time::Instant::now();
    let page_bytes = page_size_bytes();
    let mut per_pid = HashMap::new();
    let mut rss_bytes_total: u64 = 0;

    let proc_dir = match std::fs::read_dir("/proc") {
        Ok(d) => d,
        Err(_) => return OllamaSnapshot { per_pid, when, rss_mib: 0.0 },
    };

    for entry in proc_dir.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        let pid: u32 = match name_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        let cmdline_path = format!("/proc/{pid}/cmdline");
        let cmdline = match std::fs::read(&cmdline_path) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).to_ascii_lowercase(),
            Err(_) => continue,
        };

        if !cmdline.contains("ollama") {
            continue;
        }

        let stat_path = format!("/proc/{pid}/stat");
        let stat_raw = match std::fs::read_to_string(&stat_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // The comm field (field 2) is in parens and may contain spaces/parens.
        // Anchor on the *last* ')' to find the end of comm reliably.
        let after_comm = match stat_raw.rfind(')') {
            Some(pos) if pos + 2 < stat_raw.len() => &stat_raw[pos + 2..],
            _ => continue,
        };

        let fields: Vec<&str> = after_comm.split_whitespace().collect();
        // After comm: field 3 is index 0.  utime=index 11, stime=index 12,
        // rss=index 21.
        if fields.len() < 22 {
            continue;
        }

        let utime: u64 = fields[11].parse().unwrap_or(0);
        let stime: u64 = fields[12].parse().unwrap_or(0);
        let rss_pages: u64 = fields[21].parse().unwrap_or(0);

        per_pid.insert(pid, ProcTicks { ticks: utime + stime });
        rss_bytes_total += rss_pages * page_bytes;
    }

    let rss_mib = rss_bytes_total as f64 / (1024.0 * 1024.0);
    OllamaSnapshot { per_pid, when, rss_mib }
}

/// Instantaneous CPU % from tick deltas between two consecutive snapshots.
fn compute_ollama_cpu(prev: &OllamaSnapshot, curr: &OllamaSnapshot) -> f64 {
    let wall_secs = curr.when.duration_since(prev.when).as_secs_f64();
    if wall_secs <= 0.0 {
        return 0.0;
    }

    let tck = clk_tck() as f64;
    let mut total_pct = 0.0;

    for (pid, curr_t) in &curr.per_pid {
        if let Some(prev_t) = prev.per_pid.get(pid) {
            let delta_ticks = curr_t.ticks.saturating_sub(prev_t.ticks);
            total_pct += (delta_ticks as f64 / tck) / wall_secs * 100.0;
        }
    }

    (total_pct * 10.0).round() / 10.0
}

// ── Background collector ───────────────────────────────────────────────────

/// Runs forever from server start.  Collects at 30 Hz, pushes to the global
/// ring buffer and broadcast channel.
async fn collector() {
    let mut prev_ollama: Option<OllamaSnapshot> = None;
    let mut next_tick = tokio::time::Instant::now();
    let mut consecutive_gpu_errors: u64 = 0;

    loop {
        // /proc scan on blocking pool (fast, but touches filesystem).
        let ollama_snap = tokio::task::spawn_blocking(read_ollama_snapshot)
            .await
            .expect("spawn_blocking panicked — tokio runtime is broken");

        let ollama_cpu = prev_ollama
            .as_ref()
            .map(|prev| compute_ollama_cpu(prev, &ollama_snap))
            .unwrap_or(0.0);
        let ollama_ram_mib = (ollama_snap.rss_mib * 10.0).round() / 10.0;
        prev_ollama = Some(ollama_snap);

        // GPU — may fail (driver reset, nvidia-smi not found, etc.).
        let gpu = match gpu_stats().await {
            Ok(g) => {
                if consecutive_gpu_errors > 0 {
                    eprintln!(
                        "[collector] nvidia-smi recovered after {consecutive_gpu_errors} consecutive failures"
                    );
                    consecutive_gpu_errors = 0;
                }
                Some(g)
            }
            Err(e) => {
                consecutive_gpu_errors += 1;
                // Log first failure, then every 300th (~10 s at 30 Hz)
                // to avoid flooding stderr.
                if consecutive_gpu_errors == 1 || consecutive_gpu_errors % 300 == 0 {
                    eprintln!(
                        "[collector] nvidia-smi failed (×{consecutive_gpu_errors}): {e}"
                    );
                }
                None
            }
        };

        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before Unix epoch — check your hardware clock")
            .as_millis() as u64;

        let snap = Snapshot { t, gpu, ollama_cpu, ollama_ram_mib };

        let json: Arc<str> = serde_json::to_string(&snap)
            .expect("Snapshot serialisation is infallible for well-typed structs")
            .into();

        // Push to ring buffer.
        {
            let mut hist = STATE.history.write().await;
            if hist.len() >= MAX_HISTORY {
                hist.pop_front();
            }
            hist.push_back(Arc::clone(&json));
        }

        // Broadcast to connected clients.  Ignore the error — it just means
        // nobody is listening right now, which is fine.
        let _ = STATE.tx.send(json);

        next_tick += POLL_INTERVAL;
        tokio::time::sleep_until(next_tick).await;
    }
}

// ── WebSocket handler ──────────────────────────────────────────────────────

async fn ws_upgrade(ws: WebSocketUpgrade) -> impl axum::response::IntoResponse {
    ws.on_upgrade(ws_loop)
}

async fn ws_loop(mut socket: WebSocket) {
    // Subscribe BEFORE reading history so we don't miss samples produced
    // between the read and the subscribe.  The client deduplicates by
    // timestamp if there's minor overlap.
    let mut rx = STATE.tx.subscribe();

    // Send full history as a JSON array (first message).
    {
        let hist = STATE.history.read().await;
        let mut buf = String::with_capacity(hist.len() * 120 + 2);
        buf.push('[');
        for (i, s) in hist.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            buf.push_str(s);
        }
        buf.push(']');

        match tokio::time::timeout(WS_SEND_TIMEOUT, socket.send(Message::Text(buf.into()))).await
        {
            Ok(Ok(())) => {}
            _ => return,
        }
    }

    // Stream live snapshots with send timeout to detect stalled clients.
    loop {
        match rx.recv().await {
            Ok(json) => {
                match tokio::time::timeout(
                    WS_SEND_TIMEOUT,
                    socket.send(Message::Text((*json).into())),
                )
                .await
                {
                    Ok(Ok(())) => {}
                    _ => return, // timeout or send error — client is gone
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("[ws] client lagged, skipped {n} messages");
            }
            Err(broadcast::error::RecvError::Closed) => {
                return;
            }
        }
    }
}

// ── HTTP handler ───────────────────────────────────────────────────────────

async fn index() -> Html<&'static str> {
    Html(HTML)
}

// ── Entrypoint ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Force initialisation of the global state and system constants.
    LazyLock::force(&STATE);
    let _ = page_size_bytes();
    let _ = clk_tck();

    // Start the collector — it runs for the lifetime of the server.
    tokio::spawn(collector());

    let app = Router::new()
        .route("/", get(index))
        .route("/ws", get(ws_upgrade));

    let listener = tokio::net::TcpListener::bind(LISTEN_ADDR)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "failed to bind to {LISTEN_ADDR}: {e} — \
                 is another process already listening on this port? \
                 (check with: lsof -i:{port})",
                port = LISTEN_ADDR.split(':').last().unwrap_or("?")
            );
        });

    eprintln!("PC Monitor listening on http://{LISTEN_ADDR}");

    axum::serve(listener, app)
        .await
        .expect("axum::serve returned an error — this is unexpected");
}

// ── Embedded HTML ──────────────────────────────────────────────────────────

const HTML: &str = r##"<!DOCTYPE html>
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
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:1;min-height:0}
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
    <div class="card-hdr"><h2>GPU Utilization</h2><span class="val" id="vGpu" style="color:#7cf">&ndash;</span></div>
    <div class="chart-wrap"><canvas id="cGpu"></canvas></div>
  </div>
  <div class="card">
    <div class="card-hdr"><h2>VRAM Usage</h2><span class="val" id="vVram" style="color:#f7a">&ndash;</span></div>
    <div class="chart-wrap"><canvas id="cVram"></canvas></div>
  </div>
  <div class="card">
    <div class="card-hdr"><h2>Power Draw</h2><span class="val" id="vPow" style="color:#bf5">&ndash;</span></div>
    <div class="chart-wrap"><canvas id="cPow"></canvas></div>
  </div>
  <div class="card">
    <div class="card-hdr"><h2>Ollama CPU</h2><span class="val" id="vOllama" style="color:#5df">&ndash;</span></div>
    <div class="chart-wrap"><canvas id="cOllama"></canvas></div>
  </div>
  <div class="card" style="grid-column:span 2">
    <div class="card-hdr"><h2>Ollama RAM</h2><span class="val" id="vRam" style="color:#da7">&ndash;</span></div>
    <div class="chart-wrap"><canvas id="cRam"></canvas></div>
  </div>
</div>
<script>
const MAX=900;
const STATUS=document.getElementById("status");
const STALE_MS=5000;

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
const chRam   =mkLine(document.getElementById("cRam"),   "#da7",64000);

const vGpu=document.getElementById("vGpu");
const vVram=document.getElementById("vVram");
const vPow=document.getElementById("vPow");
const vOllama=document.getElementById("vOllama");
const vRam=document.getElementById("vRam");

function pushOne(chart,val){
  chart.data.labels.push("");
  chart.data.datasets[0].data.push(val);
  if(chart.data.labels.length>MAX){
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
}

function ingestSnap(d){
  const g=d.gpu;
  pushOne(chGpu,   g?g.gpu_util:null);
  pushOne(chVram,  g?g.vram_used:null);
  pushOne(chPow,   g?g.power:null);
  pushOne(chOllama,d.ollama_cpu);
  pushOne(chRam,   d.ollama_ram_mib);
  return d;
}

function updateValues(d){
  const g=d.gpu;
  if(g){
    vGpu.textContent=g.gpu_util.toFixed(0)+"%";
    vVram.textContent=(g.vram_used/1024).toFixed(1)+" / "+(g.vram_total/1024).toFixed(1)+" GiB";
    vPow.textContent=g.power.toFixed(1)+" W";
  }else{
    vGpu.textContent="\u2013";
    vVram.textContent="\u2013";
    vPow.textContent="\u2013";
  }
  vOllama.textContent=d.ollama_cpu.toFixed(1)+"%";
  vRam.textContent=(d.ollama_ram_mib/1024).toFixed(2)+" GiB";
}

function redrawAll(){
  chGpu.update();chVram.update();chPow.update();chOllama.update();chRam.update();
}

/* ── Throttled rendering (~4 fps) ──────────────────────────────────── */
let pending=[],lastVal=null,dirty=false;
setInterval(()=>{
  if(pending.length){
    for(const s of pending) lastVal=ingestSnap(s);
    pending=[];
    dirty=true;
    if(lastVal) updateValues(lastVal);
  }
  if(dirty){dirty=false;redrawAll();}
},250);

/* ── WebSocket with reconnect ──────────────────────────────────────── */
let ws,retry=0,lastMsg=0;
function connect(){
  ws=new WebSocket("ws://"+location.host+"/ws");
  ws.onopen=()=>{STATUS.textContent="";retry=0;lastMsg=Date.now();};
  ws.onclose=()=>{
    STATUS.textContent="Disconnected \u2013 reconnecting\u2026";
    setTimeout(connect,Math.min(1000*2**retry++,8000));
  };
  ws.onerror=()=>ws.close();

  let gotHistory=false;
  ws.onmessage=e=>{
    lastMsg=Date.now();
    const parsed=JSON.parse(e.data);
    if(!gotHistory&&Array.isArray(parsed)){
      gotHistory=true;
      pending=parsed;
      return;
    }
    if(parsed.error){STATUS.textContent="Server: "+parsed.error;return;}
    pending.push(parsed);
  };
}

/* ── Stale-data watchdog ───────────────────────────────────────────── */
setInterval(()=>{
  if(!lastMsg) return;
  const age=Date.now()-lastMsg;
  if(age>STALE_MS){
    STATUS.textContent="Data stale \u2013 last update "+Math.round(age/1000)+"s ago";
    if(ws&&ws.readyState===WebSocket.OPEN) ws.close();
  }
},1000);

/* ── Page Visibility API ───────────────────────────────────────────── */
document.addEventListener("visibilitychange",()=>{
  if(!document.hidden&&lastMsg&&Date.now()-lastMsg>2000){
    pending=[];
    if(ws&&(ws.readyState===WebSocket.OPEN||ws.readyState===WebSocket.CONNECTING))
      ws.close();
  }
});

connect();
</script>
</body>
</html>
"##;
