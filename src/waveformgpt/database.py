"""
WaveformGPT Database Layer

Persistent storage for:
- Waveform data
- Analysis results
- Trained models metadata
- User sessions

Uses SQLite for simplicity, with async support via aiosqlite.
"""

import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Async support (optional)
try:
    import aiosqlite
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class WaveformRecord:
    """Stored waveform data"""
    id: str
    name: str
    samples: List[float]
    sample_rate: float
    created_at: str
    metadata: Dict[str, Any]
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.samples, dtype=np.float32)


@dataclass
class AnalysisRecord:
    """Stored analysis result"""
    id: str
    waveform_id: str
    analysis_type: str  # 'dsp', 'cnn', 'full'
    result: Dict[str, Any]
    created_at: str
    processing_time_ms: float


@dataclass
class OptimizationRecord:
    """Stored optimization result"""
    id: str
    target_waveform_id: str
    circuit_type: str
    optimal_components: Dict[str, float]
    mse: float
    iterations: int
    created_at: str


# =============================================================================
# Database Manager
# =============================================================================

class WaveformDatabase:
    """
    SQLite database for waveform storage and analysis caching.
    
    Usage:
        db = WaveformDatabase("waveforms.db")
        
        # Store a waveform
        wf_id = db.save_waveform(samples, sample_rate=1e6, name="test_capture")
        
        # Get analysis (cached)
        analysis = db.get_or_analyze(wf_id, analyzer_fn)
        
        # Search
        results = db.search("overshoot")
    """
    
    def __init__(self, db_path: str = "~/.waveformgpt/waveforms.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Waveforms table
                CREATE TABLE IF NOT EXISTS waveforms (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    samples BLOB,
                    sample_rate REAL,
                    num_samples INTEGER,
                    vpp REAL,
                    created_at TEXT,
                    metadata TEXT
                );
                
                -- Analysis cache table
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id TEXT PRIMARY KEY,
                    waveform_id TEXT,
                    analysis_type TEXT,
                    result TEXT,
                    created_at TEXT,
                    processing_time_ms REAL,
                    FOREIGN KEY (waveform_id) REFERENCES waveforms(id)
                );
                
                -- Optimization history
                CREATE TABLE IF NOT EXISTS optimizations (
                    id TEXT PRIMARY KEY,
                    target_waveform_id TEXT,
                    circuit_type TEXT,
                    optimal_components TEXT,
                    mse REAL,
                    iterations INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (target_waveform_id) REFERENCES waveforms(id)
                );
                
                -- Tags for search
                CREATE TABLE IF NOT EXISTS waveform_tags (
                    waveform_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (waveform_id) REFERENCES waveforms(id)
                );
                
                -- Full-text search index
                CREATE VIRTUAL TABLE IF NOT EXISTS waveform_fts USING fts5(
                    id, name, tags, problems
                );
                
                -- Create indices
                CREATE INDEX IF NOT EXISTS idx_waveforms_created 
                    ON waveforms(created_at);
                CREATE INDEX IF NOT EXISTS idx_analysis_waveform 
                    ON analysis_cache(waveform_id);
                CREATE INDEX IF NOT EXISTS idx_tags_waveform 
                    ON waveform_tags(waveform_id);
            """)
    
    # =========================================================================
    # Waveform CRUD
    # =========================================================================
    
    def save_waveform(
        self,
        samples: np.ndarray,
        sample_rate: float = 1e6,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a waveform to the database.
        
        Returns:
            Waveform ID
        """
        # Generate ID from content hash
        wf_id = hashlib.sha256(samples.tobytes()).hexdigest()[:16]
        
        # Default name
        if name is None:
            name = f"waveform_{wf_id[:8]}"
        
        # Compute basic stats
        vpp = float(np.max(samples) - np.min(samples))
        
        # Serialize samples
        samples_blob = samples.astype(np.float32).tobytes()
        
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert waveform
            conn.execute("""
                INSERT OR REPLACE INTO waveforms 
                (id, name, samples, sample_rate, num_samples, vpp, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wf_id, name, samples_blob, sample_rate,
                len(samples), vpp, created_at,
                json.dumps(metadata or {})
            ))
            
            # Add tags
            if tags:
                conn.executemany(
                    "INSERT INTO waveform_tags (waveform_id, tag) VALUES (?, ?)",
                    [(wf_id, tag) for tag in tags]
                )
                
                # Update FTS index
                conn.execute(
                    "INSERT OR REPLACE INTO waveform_fts (id, name, tags, problems) VALUES (?, ?, ?, ?)",
                    (wf_id, name, " ".join(tags), "")
                )
        
        return wf_id
    
    def get_waveform(self, wf_id: str) -> Optional[WaveformRecord]:
        """Get a waveform by ID"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, name, samples, sample_rate, created_at, metadata FROM waveforms WHERE id = ?",
                (wf_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            samples = np.frombuffer(row[2], dtype=np.float32).tolist()
            
            return WaveformRecord(
                id=row[0],
                name=row[1],
                samples=samples,
                sample_rate=row[3],
                created_at=row[4],
                metadata=json.loads(row[5])
            )
    
    def list_waveforms(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at DESC"
    ) -> List[Dict]:
        """List waveforms with basic info (no sample data)"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(f"""
                SELECT id, name, sample_rate, num_samples, vpp, created_at
                FROM waveforms
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
            
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "sample_rate": r[2],
                    "num_samples": r[3],
                    "vpp": r[4],
                    "created_at": r[5]
                }
                for r in rows
            ]
    
    def delete_waveform(self, wf_id: str) -> bool:
        """Delete a waveform and its cached analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM waveform_tags WHERE waveform_id = ?", (wf_id,))
            conn.execute("DELETE FROM analysis_cache WHERE waveform_id = ?", (wf_id,))
            conn.execute("DELETE FROM waveform_fts WHERE id = ?", (wf_id,))
            result = conn.execute("DELETE FROM waveforms WHERE id = ?", (wf_id,))
            return result.rowcount > 0
    
    # =========================================================================
    # Analysis Cache
    # =========================================================================
    
    def get_cached_analysis(
        self,
        wf_id: str,
        analysis_type: str = "full"
    ) -> Optional[Dict]:
        """Get cached analysis result"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT result FROM analysis_cache
                WHERE waveform_id = ? AND analysis_type = ?
            """, (wf_id, analysis_type)).fetchone()
            
            if row:
                return json.loads(row[0])
            return None
    
    def cache_analysis(
        self,
        wf_id: str,
        analysis_type: str,
        result: Dict,
        processing_time_ms: float
    ) -> str:
        """Cache an analysis result"""
        analysis_id = hashlib.md5(f"{wf_id}:{analysis_type}".encode()).hexdigest()[:12]
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analysis_cache
                (id, waveform_id, analysis_type, result, created_at, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis_id, wf_id, analysis_type,
                json.dumps(result), created_at, processing_time_ms
            ))
            
            # Update FTS with detected problems
            if "problems" in result:
                problems = " ".join(result["problems"])
                conn.execute(
                    "UPDATE waveform_fts SET problems = ? WHERE id = ?",
                    (problems, wf_id)
                )
        
        return analysis_id
    
    def get_or_analyze(
        self,
        wf_id: str,
        analyze_fn,
        analysis_type: str = "full"
    ) -> Dict:
        """
        Get cached analysis or run analysis function.
        
        Args:
            wf_id: Waveform ID
            analyze_fn: Function that takes np.ndarray and returns Dict
            analysis_type: Type label for caching
        
        Returns:
            Analysis result
        """
        # Check cache
        cached = self.get_cached_analysis(wf_id, analysis_type)
        if cached:
            cached["_cached"] = True
            return cached
        
        # Get waveform
        wf = self.get_waveform(wf_id)
        if wf is None:
            raise ValueError(f"Waveform {wf_id} not found")
        
        # Run analysis
        start = time.time()
        samples = np.array(wf.samples, dtype=np.float32)
        result = analyze_fn(samples)
        processing_time = (time.time() - start) * 1000
        
        # Cache
        self.cache_analysis(wf_id, analysis_type, result, processing_time)
        
        result["_cached"] = False
        result["_processing_time_ms"] = processing_time
        
        return result
    
    # =========================================================================
    # Optimization History
    # =========================================================================
    
    def save_optimization(
        self,
        target_wf_id: str,
        circuit_type: str,
        optimal_components: Dict[str, float],
        mse: float,
        iterations: int
    ) -> str:
        """Save optimization result"""
        opt_id = hashlib.md5(
            f"{target_wf_id}:{circuit_type}:{time.time()}".encode()
        ).hexdigest()[:12]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimizations
                (id, target_waveform_id, circuit_type, optimal_components, mse, iterations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                opt_id, target_wf_id, circuit_type,
                json.dumps(optimal_components), mse, iterations,
                datetime.now().isoformat()
            ))
        
        return opt_id
    
    def get_optimization_history(
        self,
        target_wf_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get optimization history"""
        with sqlite3.connect(self.db_path) as conn:
            if target_wf_id:
                rows = conn.execute("""
                    SELECT * FROM optimizations
                    WHERE target_waveform_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (target_wf_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM optimizations
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()
            
            return [
                {
                    "id": r[0],
                    "target_waveform_id": r[1],
                    "circuit_type": r[2],
                    "optimal_components": json.loads(r[3]),
                    "mse": r[4],
                    "iterations": r[5],
                    "created_at": r[6]
                }
                for r in rows
            ]
    
    # =========================================================================
    # Search
    # =========================================================================
    
    def search(self, query: str, limit: int = 50) -> List[Dict]:
        """Full-text search across waveforms"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT w.id, w.name, w.sample_rate, w.num_samples, w.vpp, w.created_at
                FROM waveform_fts f
                JOIN waveforms w ON f.id = w.id
                WHERE waveform_fts MATCH ?
                LIMIT ?
            """, (query, limit)).fetchall()
            
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "sample_rate": r[2],
                    "num_samples": r[3],
                    "vpp": r[4],
                    "created_at": r[5]
                }
                for r in rows
            ]
    
    def search_by_problem(self, problem: str) -> List[Dict]:
        """Search waveforms by detected problem type"""
        return self.search(problem)
    
    def search_by_tag(self, tag: str) -> List[Dict]:
        """Search waveforms by tag"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT w.id, w.name, w.sample_rate, w.num_samples, w.vpp, w.created_at
                FROM waveform_tags t
                JOIN waveforms w ON t.waveform_id = w.id
                WHERE t.tag = ?
            """, (tag,)).fetchall()
            
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "sample_rate": r[2],
                    "num_samples": r[3],
                    "vpp": r[4],
                    "created_at": r[5]
                }
                for r in rows
            ]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            wf_count = conn.execute("SELECT COUNT(*) FROM waveforms").fetchone()[0]
            analysis_count = conn.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0]
            opt_count = conn.execute("SELECT COUNT(*) FROM optimizations").fetchone()[0]
            
            db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
            
            return {
                "waveform_count": wf_count,
                "cached_analyses": analysis_count,
                "optimizations": opt_count,
                "database_size_mb": round(db_size, 2),
                "database_path": str(self.db_path)
            }
    
    def vacuum(self):
        """Optimize database size"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")


# =============================================================================
# Async Database (for FastAPI)
# =============================================================================

if HAS_ASYNC:
    
    class AsyncWaveformDatabase:
        """Async version for use with FastAPI"""
        
        def __init__(self, db_path: str = "~/.waveformgpt/waveforms.db"):
            self.db_path = Path(db_path).expanduser()
            self._sync_db = WaveformDatabase(str(self.db_path))
        
        async def save_waveform(self, *args, **kwargs) -> str:
            async with aiosqlite.connect(self.db_path) as db:
                # Use sync implementation for now
                return self._sync_db.save_waveform(*args, **kwargs)
        
        async def get_waveform(self, wf_id: str) -> Optional[WaveformRecord]:
            return self._sync_db.get_waveform(wf_id)
        
        async def get_cached_analysis(self, *args, **kwargs):
            return self._sync_db.get_cached_analysis(*args, **kwargs)
        
        async def cache_analysis(self, *args, **kwargs):
            return self._sync_db.cache_analysis(*args, **kwargs)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Database - Demo")
    print("=" * 60)
    
    # Create database
    db = WaveformDatabase("/tmp/waveformgpt_demo.db")
    
    # Generate test waveforms
    print("\n📊 Storing test waveforms...")
    
    from waveformgpt.waveform_cnn import SyntheticDataGenerator
    gen = SyntheticDataGenerator()
    
    ids = []
    for name, gen_fn in [
        ("normal_step", gen.generate_normal),
        ("overshoot_response", gen.generate_overshoot),
        ("noisy_signal", gen.generate_noise),
        ("clipped_sine", gen.generate_clipping),
    ]:
        waveform = gen_fn()
        wf_id = db.save_waveform(
            waveform,
            sample_rate=1e6,
            name=name,
            tags=[name.split("_")[0], "test"]
        )
        ids.append(wf_id)
        print(f"   Saved: {name} -> {wf_id}")
    
    # List waveforms
    print("\n📋 Listing waveforms:")
    for wf in db.list_waveforms():
        print(f"   {wf['id'][:8]}... : {wf['name']} ({wf['num_samples']} samples)")
    
    # Cache some analysis
    print("\n💾 Caching analysis results...")
    for wf_id in ids[:2]:
        db.cache_analysis(
            wf_id,
            "full",
            {"vpp": 3.3, "problems": ["OVERSHOOT", "NOISE"]},
            10.5
        )
    
    # Search
    print("\n🔍 Search for 'overshoot':")
    results = db.search("overshoot")
    for r in results:
        print(f"   {r['id'][:8]}... : {r['name']}")
    
    # Stats
    print("\n📈 Database stats:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Demo complete!")
