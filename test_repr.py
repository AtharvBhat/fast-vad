#!/usr/bin/env python3
"""Test script to verify __repr__ functionality"""

import fast_vad

# Test FeatureExtractor
fe_16k = fast_vad.FeatureExtractor(16000)
print("FeatureExtractor (16kHz):", repr(fe_16k))

fe_8k = fast_vad.FeatureExtractor(8000)
print("FeatureExtractor (8kHz):", repr(fe_8k))

# Test VAD
vad_16k = fast_vad.VAD(16000)
print("VAD (16kHz):", repr(vad_16k))

vad_8k = fast_vad.VAD(8000)
print("VAD (8kHz):", repr(vad_8k))

# Test VadStateful
vad_stateful_16k = fast_vad.VadStateful(16000)
print("VadStateful (16kHz):", repr(vad_stateful_16k))

vad_stateful_8k = fast_vad.VadStateful(8000)
print("VadStateful (8kHz):", repr(vad_stateful_8k))