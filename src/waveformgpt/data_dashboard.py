"""
WaveformGPT Data Dashboard

Web-based interface for:
1. Viewing collected waveforms
2. Labeling data (the REAL value)
3. Dataset statistics and health
4. Model performance tracking
5. Active learning queue
6. Real-time collection monitoring

This is what makes the data USABLE.
"""

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import json
import asyncio
import time
from pathlib import Path
import base64
import io

# Import our data pipeline
try:
    from waveformgpt.data_pipeline import (
        DataPipeline, DatasetManager, WaveformSample,
        ProblemLabel, ConfidenceLevel, WaveformSource
    )
except ImportError:
    from data_pipeline import (
        DataPipeline, DatasetManager, WaveformSample,
        ProblemLabel, ConfidenceLevel, WaveformSource
    )

# Initialize FastAPI
app = FastAPI(
    title="WaveformGPT Data Dashboard",
    description="Data collection, labeling, and model monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[DataPipeline] = None


def get_pipeline() -> DataPipeline:
    global pipeline
    if pipeline is None:
        pipeline = DataPipeline("~/.waveformgpt/data")
    return pipeline


# =============================================================================
# API Models
# =============================================================================

class LabelRequest(BaseModel):
    sample_id: str
    label: str
    confidence: str = "verified"
    notes: str = ""


class SampleResponse(BaseModel):
    id: str
    samples: List[float]
    sample_rate: float
    source: str
    primary_label: str
    confidence: str
    features: Dict[str, float]
    created_at: float


class StatsResponse(BaseModel):
    total_samples: int
    labeled_samples: int
    verified_samples: int
    samples_by_label: Dict[str, int]
    samples_by_source: Dict[str, int]


# =============================================================================
# HTML Templates
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WaveformGPT Data Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 24px;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status {
            display: flex;
            gap: 20px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        
        .status-dot.green { background: #4ade80; }
        .status-dot.yellow { background: #fbbf24; }
        .status-dot.red { background: #f87171; }
        
        .container {
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 80px);
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .panel h2 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 15px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .stat-card {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .stat-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-top: 5px;
        }
        
        .waveform-display {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .waveform-canvas {
            flex: 1;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .label-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .label-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            transition: all 0.2s;
        }
        
        .label-btn:hover {
            transform: translateY(-2px);
        }
        
        .label-btn.clean { background: #4ade80; color: #000; }
        .label-btn.noisy { background: #fbbf24; color: #000; }
        .label-btn.clipped { background: #f87171; color: #000; }
        .label-btn.ringing { background: #a78bfa; color: #000; }
        .label-btn.overshoot { background: #fb923c; color: #000; }
        .label-btn.emi { background: #38bdf8; color: #000; }
        
        .queue-list {
            flex: 1;
            overflow-y: auto;
        }
        
        .queue-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }
        
        .queue-item:hover {
            background: rgba(0, 0, 0, 0.4);
            border-left-color: #00d4ff;
        }
        
        .queue-item.active {
            border-left-color: #7b2ff7;
            background: rgba(123, 47, 247, 0.2);
        }
        
        .queue-item-id {
            font-family: monospace;
            font-size: 12px;
            color: #888;
        }
        
        .queue-item-source {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }
        
        .chart-container {
            height: 200px;
            margin-top: 15px;
        }
        
        .actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .action-btn {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .action-btn.primary {
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            color: white;
        }
        
        .action-btn.secondary {
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            transition: width 0.3s;
        }
        
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            background: #4ade80;
            color: #000;
            font-weight: 600;
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
        }
        
        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }
        
        .notes-input {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.2);
            color: white;
            font-size: 14px;
            margin-bottom: 15px;
        }
        
        .notes-input::placeholder {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ WaveformGPT Data Dashboard</h1>
        <div class="status">
            <div class="status-item">
                <div class="status-dot green" id="esp32-status"></div>
                <span>ESP32</span>
            </div>
            <div class="status-item">
                <div class="status-dot yellow" id="scope-status"></div>
                <span>Oscilloscope</span>
            </div>
            <div class="status-item">
                <div class="status-dot green" id="db-status"></div>
                <span>Database</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <!-- Left Panel: Stats -->
        <div class="panel">
            <h2>📊 Dataset Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-samples">0</div>
                    <div class="stat-label">Total Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="labeled-samples">0</div>
                    <div class="stat-label">Labeled</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="verified-samples">0</div>
                    <div class="stat-label">Verified</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="pending-samples">0</div>
                    <div class="stat-label">Pending</div>
                </div>
            </div>
            
            <div class="progress-bar" style="margin-top: 20px;">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
            <div style="font-size: 11px; color: #666; margin-top: 5px; text-align: center;">
                <span id="progress-text">0% labeled</span>
            </div>
            
            <h2 style="margin-top: 20px;">📈 Label Distribution</h2>
            <div class="chart-container">
                <canvas id="label-chart"></canvas>
            </div>
            
            <div class="actions">
                <button class="action-btn primary" onclick="exportDataset()">Export Dataset</button>
            </div>
        </div>
        
        <!-- Center Panel: Waveform Display & Labeling -->
        <div class="panel waveform-display">
            <h2>🎯 Label Sample</h2>
            <canvas id="waveform-canvas" class="waveform-canvas"></canvas>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span id="sample-id" style="font-family: monospace; color: #888;">No sample selected</span>
                <span id="sample-source" style="color: #666; font-size: 12px;">-</span>
            </div>
            
            <input type="text" class="notes-input" id="notes-input" placeholder="Add notes (optional)...">
            
            <div class="label-buttons">
                <button class="label-btn clean" onclick="labelSample('clean')">✓ Clean</button>
                <button class="label-btn noisy" onclick="labelSample('noisy')">〰 Noisy</button>
                <button class="label-btn clipped" onclick="labelSample('clipped')">📊 Clipped</button>
                <button class="label-btn ringing" onclick="labelSample('ringing')">〜 Ringing</button>
                <button class="label-btn overshoot" onclick="labelSample('overshoot')">↑ Overshoot</button>
                <button class="label-btn emi" onclick="labelSample('emi')">⚡ EMI</button>
            </div>
            
            <div class="actions">
                <button class="action-btn secondary" onclick="skipSample()">Skip</button>
                <button class="action-btn primary" onclick="nextSample()">Next →</button>
            </div>
        </div>
        
        <!-- Right Panel: Labeling Queue -->
        <div class="panel">
            <h2>📋 Labeling Queue</h2>
            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                <button class="action-btn secondary" onclick="loadQueue('unlabeled')" style="flex: 1;">Unlabeled</button>
                <button class="action-btn secondary" onclick="loadQueue('low')" style="flex: 1;">Low Conf</button>
            </div>
            <div class="queue-list" id="queue-list">
                <!-- Queue items will be inserted here -->
            </div>
            
            <h2 style="margin-top: 20px;">🔄 Recent Labels</h2>
            <div id="recent-labels" style="font-size: 12px; color: #888;">
                No recent labels
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast">Sample labeled!</div>
    
    <script>
        let currentSample = null;
        let queue = [];
        let queueIndex = 0;
        let labelChart = null;
        let recentLabels = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadStats();
            loadQueue('unlabeled');
            setInterval(loadStats, 5000);
        });
        
        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('total-samples').textContent = stats.total_samples;
                document.getElementById('labeled-samples').textContent = stats.labeled_samples;
                document.getElementById('verified-samples').textContent = stats.verified_samples;
                document.getElementById('pending-samples').textContent = stats.total_samples - stats.labeled_samples;
                
                const progress = stats.total_samples > 0 ? 
                    Math.round(stats.labeled_samples / stats.total_samples * 100) : 0;
                document.getElementById('progress-fill').style.width = progress + '%';
                document.getElementById('progress-text').textContent = progress + '% labeled';
                
                updateLabelChart(stats.samples_by_label);
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }
        
        // Update label distribution chart
        function updateLabelChart(data) {
            const ctx = document.getElementById('label-chart').getContext('2d');
            
            if (labelChart) {
                labelChart.destroy();
            }
            
            const labels = Object.keys(data);
            const values = Object.values(data);
            const colors = {
                'clean': '#4ade80',
                'noisy': '#fbbf24',
                'clipped': '#f87171',
                'ringing': '#a78bfa',
                'overshoot': '#fb923c',
                'emi': '#38bdf8',
                'unknown': '#6b7280'
            };
            
            labelChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: labels.map(l => colors[l] || '#6b7280'),
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#888', font: { size: 10 } }
                        }
                    }
                }
            });
        }
        
        // Load labeling queue
        async function loadQueue(type) {
            try {
                const response = await fetch(`/api/queue/${type}?limit=50`);
                queue = await response.json();
                queueIndex = 0;
                
                renderQueue();
                
                if (queue.length > 0) {
                    loadSample(queue[0]);
                }
            } catch (e) {
                console.error('Failed to load queue:', e);
            }
        }
        
        // Render queue list
        function renderQueue() {
            const container = document.getElementById('queue-list');
            container.innerHTML = queue.map((id, i) => `
                <div class="queue-item ${i === queueIndex ? 'active' : ''}" 
                     onclick="selectQueueItem(${i})">
                    <div class="queue-item-id">${id.substring(0, 12)}...</div>
                    <div class="queue-item-source">Click to load</div>
                </div>
            `).join('');
        }
        
        // Select queue item
        function selectQueueItem(index) {
            queueIndex = index;
            renderQueue();
            loadSample(queue[index]);
        }
        
        // Load sample
        async function loadSample(sampleId) {
            try {
                const response = await fetch(`/api/sample/${sampleId}`);
                currentSample = await response.json();
                
                document.getElementById('sample-id').textContent = currentSample.id;
                document.getElementById('sample-source').textContent = currentSample.source;
                document.getElementById('notes-input').value = '';
                
                drawWaveform(currentSample.samples);
            } catch (e) {
                console.error('Failed to load sample:', e);
            }
        }
        
        // Draw waveform on canvas
        function drawWaveform(samples) {
            const canvas = document.getElementById('waveform-canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);
            
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            
            // Clear
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(0, 0, width, height);
            
            // Draw grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            
            for (let i = 0; i <= 10; i++) {
                const y = height * i / 10;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
            
            // Draw waveform
            if (!samples || samples.length === 0) return;
            
            const min = Math.min(...samples);
            const max = Math.max(...samples);
            const range = max - min || 1;
            
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            
            for (let i = 0; i < samples.length; i++) {
                const x = (i / samples.length) * width;
                const y = height - ((samples[i] - min) / range) * (height - 20) - 10;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }
        
        // Label sample
        async function labelSample(label) {
            if (!currentSample) return;
            
            const notes = document.getElementById('notes-input').value;
            
            try {
                const response = await fetch('/api/label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sample_id: currentSample.id,
                        label: label,
                        confidence: 'verified',
                        notes: notes
                    })
                });
                
                if (response.ok) {
                    showToast(`Labeled as ${label}`);
                    
                    // Add to recent
                    recentLabels.unshift({ id: currentSample.id, label: label });
                    recentLabels = recentLabels.slice(0, 5);
                    updateRecentLabels();
                    
                    // Move to next
                    nextSample();
                    loadStats();
                }
            } catch (e) {
                console.error('Failed to label:', e);
            }
        }
        
        // Skip sample
        function skipSample() {
            nextSample();
        }
        
        // Next sample
        function nextSample() {
            if (queueIndex < queue.length - 1) {
                queueIndex++;
                renderQueue();
                loadSample(queue[queueIndex]);
            }
        }
        
        // Update recent labels display
        function updateRecentLabels() {
            const container = document.getElementById('recent-labels');
            container.innerHTML = recentLabels.map(r => `
                <div style="margin-bottom: 5px;">
                    <span style="color: #00d4ff;">${r.id.substring(0, 8)}</span>
                    → ${r.label}
                </div>
            `).join('');
        }
        
        // Export dataset
        async function exportDataset() {
            try {
                const response = await fetch('/api/export', { method: 'POST' });
                const result = await response.json();
                showToast(`Exported ${result.total_exported} samples`);
            } catch (e) {
                console.error('Failed to export:', e);
            }
        }
        
        // Show toast notification
        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            switch(e.key) {
                case '1': labelSample('clean'); break;
                case '2': labelSample('noisy'); break;
                case '3': labelSample('clipped'); break;
                case '4': labelSample('ringing'); break;
                case '5': labelSample('overshoot'); break;
                case '6': labelSample('emi'); break;
                case 's': skipSample(); break;
                case 'ArrowRight': nextSample(); break;
            }
        });
    </script>
</body>
</html>
"""


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML"""
    return DASHBOARD_HTML


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics"""
    p = get_pipeline()
    stats = p.get_stats()
    return {
        "total_samples": stats.total_samples,
        "labeled_samples": stats.labeled_samples,
        "verified_samples": stats.verified_samples,
        "samples_by_label": stats.samples_by_label,
        "samples_by_source": stats.samples_by_source
    }


@app.get("/api/queue/{queue_type}")
async def get_queue(queue_type: str, limit: int = 50):
    """Get samples for labeling queue"""
    p = get_pipeline()
    
    if queue_type == "unlabeled":
        samples = p.dataset.get_unlabeled(limit)
    elif queue_type == "low":
        # Get low confidence samples
        conn = __import__('sqlite3').connect(p.dataset.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM samples 
            WHERE confidence = 'low' OR confidence = 'medium'
            LIMIT ?
        """, (limit,))
        samples = [r[0] for r in cursor.fetchall()]
        conn.close()
    else:
        samples = []
    
    return samples


@app.get("/api/sample/{sample_id}")
async def get_sample(sample_id: str):
    """Get a specific sample"""
    p = get_pipeline()
    sample = p.dataset.get_sample(sample_id)
    
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    return {
        "id": sample.id,
        "samples": sample.samples.tolist()[:2000],  # Limit for UI
        "sample_rate": sample.sample_rate,
        "source": sample.source.value,
        "primary_label": sample.primary_label.value,
        "confidence": sample.confidence.value,
        "features": sample.features
    }


@app.post("/api/label")
async def label_sample(request: LabelRequest):
    """Label a sample"""
    p = get_pipeline()
    
    try:
        label = ProblemLabel(request.label)
        confidence = ConfidenceLevel(request.confidence)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid label or confidence")
    
    success = p.dataset.label_sample(
        request.sample_id,
        label,
        confidence,
        annotator="dashboard_user",
        notes=request.notes
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    return {"success": True}


@app.post("/api/export")
async def export_dataset():
    """Export dataset for training"""
    p = get_pipeline()
    
    output_path = Path("~/.waveformgpt/exports").expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    filename = output_path / f"training_data_{timestamp}.npz"
    
    result = p.dataset.export_for_training(
        str(filename),
        format="numpy",
        pad_length=1000
    )
    
    return result


@app.post("/api/upload")
async def upload_waveform(
    file: UploadFile = File(...),
    label: str = Form(None)
):
    """Upload a waveform file"""
    p = get_pipeline()
    
    # Save temporarily
    temp_path = Path(f"/tmp/{file.filename}")
    content = await file.read()
    
    with open(temp_path, 'wb') as f:
        f.write(content)
    
    # Import
    problem_label = ProblemLabel(label) if label else None
    count = p.import_from_file(str(temp_path), problem_label)
    
    # Cleanup
    temp_path.unlink()
    
    return {"imported": count}


@app.websocket("/ws/collection")
async def collection_websocket(websocket: WebSocket):
    """WebSocket for real-time collection monitoring"""
    await websocket.accept()
    
    p = get_pipeline()
    
    try:
        while True:
            stats = p.get_stats()
            await websocket.send_json({
                "type": "stats",
                "data": {
                    "total": stats.total_samples,
                    "labeled": stats.labeled_samples
                }
            })
            await asyncio.sleep(1)
    except:
        pass


# =============================================================================
# Run Server
# =============================================================================

def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server"""
    import uvicorn
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║       WaveformGPT Data Dashboard                          ║
║                                                           ║
║   Open in browser: http://localhost:{port}                  ║
║                                                           ║
║   Keyboard Shortcuts:                                     ║
║   1-6: Quick labels (clean, noisy, clipped, etc.)        ║
║   s: Skip sample                                          ║
║   →: Next sample                                          ║
╚═══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
