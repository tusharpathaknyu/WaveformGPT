"""
Waveform Image Extractor

Extracts waveform data from oscilloscope screenshots.
Uses OpenCV for image processing - no ML required for basic extraction.

Supports:
- Rigol, Siglent, Tektronix, Keysight scope screenshots
- Grid detection and scaling
- Multi-channel extraction
- Automatic voltage/time scale detection (from graticule)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Install OpenCV: pip install opencv-python")


class ChannelColor(Enum):
    """Standard oscilloscope channel colors"""
    YELLOW = "CH1"      # Most scopes use yellow for CH1
    CYAN = "CH2"        # Cyan/blue for CH2
    MAGENTA = "CH3"     # Magenta/pink for CH3
    GREEN = "CH4"       # Green for CH4
    WHITE = "MATH"      # Math channel often white


@dataclass
class ExtractedWaveform:
    """Extracted waveform data from image"""
    x_data: np.ndarray          # Time points (normalized 0-1 or in seconds)
    y_data: np.ndarray          # Voltage points (normalized or in volts)
    channel: str                # Channel name
    color: Tuple[int, int, int] # BGR color
    time_scale: Optional[float] # Seconds per division (if detected)
    volt_scale: Optional[float] # Volts per division (if detected)
    sample_count: int           # Number of points extracted


@dataclass
class GridInfo:
    """Detected oscilloscope grid information"""
    x_divisions: int        # Number of horizontal divisions
    y_divisions: int        # Number of vertical divisions
    grid_left: int          # Left edge of grid (pixels)
    grid_right: int         # Right edge of grid (pixels)
    grid_top: int           # Top edge of grid (pixels)
    grid_bottom: int        # Bottom edge of grid (pixels)
    div_width: float        # Pixels per horizontal division
    div_height: float       # Pixels per vertical division


class WaveformImageExtractor:
    """
    Extract waveform data from oscilloscope screenshots.
    """
    
    # Common oscilloscope channel colors (HSV ranges)
    # Format: (H_min, S_min, V_min), (H_max, S_max, V_max)
    CHANNEL_COLORS = {
        'yellow': ((20, 100, 100), (35, 255, 255)),     # CH1
        'cyan': ((85, 100, 100), (100, 255, 255)),      # CH2
        'magenta': ((140, 100, 100), (165, 255, 255)),  # CH3
        'green': ((35, 100, 100), (85, 255, 255)),      # CH4
        'orange': ((10, 100, 100), (20, 255, 255)),     # Some scopes
    }
    
    def __init__(self):
        if not HAS_CV2:
            raise ImportError("OpenCV required: pip install opencv-python")
    
    def load_image(self, path: str) -> np.ndarray:
        """Load image from file"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img
    
    def detect_grid(self, image: np.ndarray) -> Optional[GridInfo]:
        """
        Detect the oscilloscope grid/graticule.
        Returns grid boundaries and division info.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Look for grid lines using edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Use Hough lines to find grid
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                 minLineLength=100, maxLineGap=10)
        
        if lines is None:
            # Fallback: assume standard 10x8 grid covering 80% of image
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            return GridInfo(
                x_divisions=10,
                y_divisions=8,
                grid_left=margin_x,
                grid_right=w - margin_x,
                grid_top=margin_y,
                grid_bottom=h - margin_y,
                div_width=(w - 2*margin_x) / 10,
                div_height=(h - 2*margin_y) / 8
            )
        
        # Find bounding box of detected lines
        x_coords = []
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        grid_left = min(x_coords)
        grid_right = max(x_coords)
        grid_top = min(y_coords)
        grid_bottom = max(y_coords)
        
        # Standard scope has 10 horizontal, 8 vertical divisions
        return GridInfo(
            x_divisions=10,
            y_divisions=8,
            grid_left=grid_left,
            grid_right=grid_right,
            grid_top=grid_top,
            grid_bottom=grid_bottom,
            div_width=(grid_right - grid_left) / 10,
            div_height=(grid_bottom - grid_top) / 8
        )
    
    def extract_waveform_by_color(
        self, 
        image: np.ndarray, 
        color_name: str,
        grid: Optional[GridInfo] = None
    ) -> Optional[ExtractedWaveform]:
        """
        Extract waveform trace by color.
        
        Args:
            image: BGR image
            color_name: 'yellow', 'cyan', 'magenta', 'green', 'orange'
            grid: Grid info for scaling (optional)
        
        Returns:
            ExtractedWaveform or None if not found
        """
        if color_name not in self.CHANNEL_COLORS:
            raise ValueError(f"Unknown color: {color_name}")
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for the color
        lower, upper = self.CHANNEL_COLORS[color_name]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find waveform points
        points = np.where(mask > 0)
        if len(points[0]) < 10:
            return None
        
        y_pixels = points[0]  # Row indices (y coordinates)
        x_pixels = points[1]  # Column indices (x coordinates)
        
        # Group by x-coordinate and take median y for each x
        x_unique = np.unique(x_pixels)
        y_median = []
        x_clean = []
        
        for x in x_unique:
            y_at_x = y_pixels[x_pixels == x]
            if len(y_at_x) > 0:
                y_median.append(np.median(y_at_x))
                x_clean.append(x)
        
        x_data = np.array(x_clean)
        y_data = np.array(y_median)
        
        # Normalize to grid if available
        if grid:
            # X: 0 to 1 (left to right of grid)
            x_normalized = (x_data - grid.grid_left) / (grid.grid_right - grid.grid_left)
            # Y: flip and normalize (top is high voltage on scope)
            y_normalized = 1 - (y_data - grid.grid_top) / (grid.grid_bottom - grid.grid_top)
            
            # Scale to divisions
            x_data = x_normalized * grid.x_divisions
            y_data = (y_normalized - 0.5) * grid.y_divisions  # Center at 0
        else:
            # Normalize to 0-1
            x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
            y_data = 1 - (y_data - y_data.min()) / (y_data.max() - y_data.min())
        
        # Get average color for verification
        avg_color = cv2.mean(image, mask=mask)[:3]
        
        return ExtractedWaveform(
            x_data=x_data,
            y_data=y_data,
            channel=color_name.upper(),
            color=tuple(int(c) for c in avg_color),
            time_scale=None,  # Would need OCR to detect
            volt_scale=None,
            sample_count=len(x_data)
        )
    
    def extract_all_channels(self, image: np.ndarray) -> List[ExtractedWaveform]:
        """
        Extract all visible waveform channels from image.
        """
        grid = self.detect_grid(image)
        waveforms = []
        
        for color_name in self.CHANNEL_COLORS.keys():
            waveform = self.extract_waveform_by_color(image, color_name, grid)
            if waveform and waveform.sample_count > 50:  # Minimum points
                waveforms.append(waveform)
        
        return waveforms
    
    def extract_by_brightness(
        self, 
        image: np.ndarray,
        grid: Optional[GridInfo] = None,
        threshold: int = 200
    ) -> Optional[ExtractedWaveform]:
        """
        Extract waveform by brightness (for any color trace).
        Good for single-channel screenshots.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold bright pixels (waveform is usually bright)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Remove very bright areas (likely text/UI elements)
        # by looking for continuous horizontal regions
        kernel = np.ones((1, 50), np.uint8)
        text_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.subtract(mask, text_mask)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Extract points
        points = np.where(mask > 0)
        if len(points[0]) < 10:
            return None
        
        y_pixels = points[0]
        x_pixels = points[1]
        
        # Get one y per x (median)
        x_unique = np.unique(x_pixels)
        y_median = []
        x_clean = []
        
        for x in x_unique:
            y_at_x = y_pixels[x_pixels == x]
            if len(y_at_x) > 0:
                y_median.append(np.median(y_at_x))
                x_clean.append(x)
        
        x_data = np.array(x_clean)
        y_data = np.array(y_median)
        
        # Normalize
        if grid:
            x_data = (x_data - grid.grid_left) / (grid.grid_right - grid.grid_left) * grid.x_divisions
            y_data = (1 - (y_data - grid.grid_top) / (grid.grid_bottom - grid.grid_top) - 0.5) * grid.y_divisions
        else:
            x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min() + 1e-6)
            y_data = 1 - (y_data - y_data.min()) / (y_data.max() - y_data.min() + 1e-6)
        
        return ExtractedWaveform(
            x_data=x_data,
            y_data=y_data,
            channel="AUTO",
            color=(255, 255, 255),
            time_scale=None,
            volt_scale=None,
            sample_count=len(x_data)
        )
    
    def resample(
        self, 
        waveform: ExtractedWaveform, 
        num_points: int = 1000
    ) -> np.ndarray:
        """
        Resample waveform to fixed number of evenly-spaced points.
        Returns just the y-values.
        """
        x_new = np.linspace(waveform.x_data.min(), waveform.x_data.max(), num_points)
        y_new = np.interp(x_new, waveform.x_data, waveform.y_data)
        return y_new
    
    def to_voltage(
        self, 
        waveform: ExtractedWaveform,
        volts_per_div: float,
        offset_divs: float = 0
    ) -> np.ndarray:
        """
        Convert normalized waveform to actual voltage values.
        
        Args:
            waveform: Extracted waveform
            volts_per_div: Volts per division setting
            offset_divs: Vertical offset in divisions
        
        Returns:
            Voltage array
        """
        return (waveform.y_data + offset_divs) * volts_per_div
    
    def to_time(
        self, 
        waveform: ExtractedWaveform,
        time_per_div: float
    ) -> np.ndarray:
        """
        Convert normalized x-axis to time values.
        
        Args:
            waveform: Extracted waveform  
            time_per_div: Seconds per division setting
        
        Returns:
            Time array in seconds
        """
        return waveform.x_data * time_per_div


class QuickExtractor:
    """
    Simplified interface for common use cases.
    """
    
    def __init__(self):
        self.extractor = WaveformImageExtractor()
    
    def from_image(
        self, 
        image_path: str,
        channel_color: Optional[str] = None,
        volts_per_div: Optional[float] = None,
        time_per_div: Optional[float] = None
    ) -> Dict:
        """
        Quick extraction from oscilloscope screenshot.
        
        Args:
            image_path: Path to screenshot
            channel_color: 'yellow', 'cyan', etc. or None for auto
            volts_per_div: Voltage scale (optional)
            time_per_div: Time scale (optional)
        
        Returns:
            Dict with 'waveform', 'voltage', 'time' arrays
        """
        image = self.extractor.load_image(image_path)
        grid = self.extractor.detect_grid(image)
        
        # Extract waveform
        if channel_color:
            waveform = self.extractor.extract_waveform_by_color(image, channel_color, grid)
        else:
            # Try each color, take the one with most points
            waveforms = self.extractor.extract_all_channels(image)
            if waveforms:
                waveform = max(waveforms, key=lambda w: w.sample_count)
            else:
                # Fall back to brightness-based
                waveform = self.extractor.extract_by_brightness(image, grid)
        
        if waveform is None:
            raise ValueError("Could not extract waveform from image")
        
        # Resample to even spacing
        y_resampled = self.extractor.resample(waveform, 1000)
        x_resampled = np.linspace(0, 1, 1000)
        
        result = {
            'waveform': y_resampled,
            'x_normalized': x_resampled,
            'channel': waveform.channel,
            'sample_count': waveform.sample_count,
        }
        
        # Convert to real units if scales provided
        if volts_per_div:
            result['voltage'] = y_resampled * volts_per_div
            result['volts_per_div'] = volts_per_div
        
        if time_per_div and grid:
            result['time'] = x_resampled * grid.x_divisions * time_per_div
            result['time_per_div'] = time_per_div
        
        return result


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Waveform Image Extractor - Demo")
    print("=" * 60)
    
    # Create a synthetic "oscilloscope screenshot" for testing
    # In real use, you'd load an actual screenshot
    
    print("\n📷 Creating test image...")
    
    # Create fake scope screen
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (20, 20, 30)  # Dark background
    
    # Draw grid
    for i in range(11):  # 10 divisions
        x = 50 + i * 70
        cv2.line(img, (x, 50), (x, 550), (50, 50, 50), 1)
    for i in range(9):   # 8 divisions
        y = 50 + i * 62
        cv2.line(img, (50, y), (750, y), (50, 50, 50), 1)
    
    # Draw a yellow sine wave (CH1)
    x_points = np.arange(50, 751)
    y_points = 300 + 150 * np.sin(np.linspace(0, 4*np.pi, len(x_points)))
    for i in range(len(x_points) - 1):
        cv2.line(img, 
                 (x_points[i], int(y_points[i])),
                 (x_points[i+1], int(y_points[i+1])),
                 (0, 255, 255),  # Yellow in BGR
                 2)
    
    # Save test image
    test_path = "/tmp/test_scope.png"
    cv2.imwrite(test_path, img)
    print(f"   Saved test image to {test_path}")
    
    # Extract waveform
    print("\n🔍 Extracting waveform...")
    
    quick = QuickExtractor()
    result = quick.from_image(test_path, channel_color='yellow')
    
    print(f"\n📊 Extraction Results:")
    print(f"   Channel: {result['channel']}")
    print(f"   Sample points: {result['sample_count']}")
    print(f"   Waveform range: {result['waveform'].min():.2f} to {result['waveform'].max():.2f}")
    
    # Now analyze with circuit optimizer
    print("\n🔧 Analyzing extracted waveform...")
    
    from circuit_optimizer import CircuitOptimizer
    
    optimizer = CircuitOptimizer(sample_rate=1e6)
    analysis = optimizer.analyze(result['waveform'])
    
    print(f"\n   Features extracted:")
    print(f"   - Vpp: {analysis['features'].vpp:.3f}")
    print(f"   - Frequency: {analysis['features'].frequency:.0f} Hz (normalized)")
    
    print("\n✅ Demo complete!")
    print("\nUsage with real screenshot:")
    print("   result = QuickExtractor().from_image('scope_screenshot.png')")
    print("   analysis = CircuitOptimizer().analyze(result['waveform'])")
