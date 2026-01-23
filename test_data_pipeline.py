#!/usr/bin/env python3
"""
Quick test for WaveformGPT Data Pipeline
"""

import sys
sys.path.insert(0, 'src')

from waveformgpt.data_pipeline import (
    DataPipeline, WaveformSample, ProblemLabel, 
    ConfidenceLevel, WaveformSource
)
import numpy as np

print('='*60)
print('WaveformGPT Data Pipeline - Quick Test')
print('='*60)

# Initialize
pipeline = DataPipeline('~/.waveformgpt/test_data')
print('✅ Pipeline initialized')

# Generate and store samples
print('\n📊 Generating test samples...')
labels = ['clean', 'noisy', 'clipped', 'ringing']
for label in labels:
    for i in range(25):
        t = np.linspace(0, 0.01, 1000)
        if label == 'clean':
            samples = np.sin(2 * np.pi * 1000 * t)
        elif label == 'noisy':
            samples = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.random.randn(len(t))
        elif label == 'clipped':
            samples = np.clip(1.5 * np.sin(2 * np.pi * 1000 * t), -1, 1)
        else:
            samples = np.sin(2 * np.pi * 1000 * t) * (1 + 0.2 * np.sin(2 * np.pi * 5000 * t))
        
        sample = WaveformSample(
            id='',
            samples=samples,
            sample_rate=100000.0,
            source=WaveformSource.SYNTHETIC,
            primary_label=ProblemLabel(label),
            confidence=ConfidenceLevel.VERIFIED
        )
        sample.features = pipeline._extract_features(samples)
        pipeline.dataset.add_sample(sample)

# Get stats
stats = pipeline.get_stats()
print(f'\n📈 Dataset Statistics:')
print(f'   Total samples: {stats.total_samples}')
print(f'   Labeled: {stats.labeled_samples}')
print(f'   Verified: {stats.verified_samples}')
print(f'\n   By label:')
for label, count in stats.samples_by_label.items():
    print(f'      {label}: {count}')

# Test export
print('\n📦 Exporting for training...')
export_result = pipeline.dataset.export_for_training(
    '~/.waveformgpt/test_data/training.npz',
    format='numpy',
    pad_length=1000
)
print(f'   Exported {export_result["total_exported"]} samples')

print('\n✅ Data Pipeline working correctly!')
print(f'\nData stored at: ~/.waveformgpt/test_data/')
