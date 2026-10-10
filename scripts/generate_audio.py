"""Synthesize all launch video audio assets mathematically.
Generates 48kHz 16-bit PCM WAV files with exact envelopes and dBFS ceilings for the 30-second LinkedIn video.
"""

import math
import os
import wave
import numpy as np

OUTPUT_DIR = os.path.join("brag-output", "composition", "assets", "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SR = 48000


def save_wav(name: str, audio: np.ndarray, target_db: float = -12.0) -> float:
    """Normalize to target_db peak and write to 16-bit WAV stereo."""
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    
    peak = np.max(np.abs(audio))
    if peak > 0:
        desired_peak = 10.0 ** (target_db / 20.0)
        audio = audio * (desired_peak / peak)
    
    audio = np.clip(audio, -0.999, 0.999)
    i16 = (audio * 32767.0).astype(np.int16)
    
    path = os.path.join(OUTPUT_DIR, name)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(i16.tobytes())
    
    dur = len(audio) / SR
    print(f"Generated {name:20s}: {dur:.2f}s, peak: {target_db} dB")
    return dur


def gen_bed(duration: float = 30.0) -> np.ndarray:
    """30s ambient drone/low pad at -24 dB, fade in 1s, fade out 2s."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    tone = (
        0.5 * np.sin(2 * np.pi * 55.0 * t) +
        0.3 * np.sin(2 * np.pi * 55.4 * t + 0.3) +
        0.3 * np.sin(2 * np.pi * 110.0 * t) +
        0.2 * np.sin(2 * np.pi * 164.81 * t + 0.7) +
        0.15 * np.sin(2 * np.pi * 220.0 * t)
    )
    lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.15 * t)
    tone *= lfo
    
    fade_in_len = int(1.0 * SR)
    fade_out_len = int(2.0 * SR)
    env = np.ones_like(t)
    env[:fade_in_len] = np.linspace(0, 1, fade_in_len) ** 2
    env[-fade_out_len:] = np.linspace(1, 0, fade_out_len) ** 2
    return tone * env


def gen_bubble_tick(duration: float = 0.08) -> np.ndarray:
    """Soft subtle tick for chat bubble appearance."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tick = np.sin(2 * np.pi * 1400.0 * t) * np.exp(-t / 0.015)
    click = np.random.randn(n) * 0.2 * np.exp(-t / 0.005)
    return tick + click


def gen_pain_tone(duration: float = 3.5) -> np.ndarray:
    """Low dull tone that builds during chat scroll (4.5s to 8.0s)."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = (
        0.6 * np.sin(2 * np.pi * 48.0 * t) +
        0.5 * np.sin(2 * np.pi * 52.0 * t) +
        0.3 * np.sin(2 * np.pi * 96.0 * t)
    )
    env = (t / duration) ** 1.5
    return tone * env


def gen_impact_vanish(duration: float = 0.35) -> np.ndarray:
    """Sharp attack + deep thump, 0.35s decay for 'vanish'."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    f_env = 85.0 * (0.7 + 1.1 * np.exp(-t / 0.04))
    phase = 2 * np.pi * np.cumsum(f_env) / SR
    thump = np.sin(phase)
    click = np.random.randn(n) * np.exp(-t / 0.012)
    amp_env = (1.0 - np.exp(-t / 0.003)) * np.exp(-t / 0.11)
    return (thump * 0.85 + click * 0.35) * amp_env


def gen_pain_riser(duration: float = 0.8) -> np.ndarray:
    """Short riser from 8.0s to 8.8s (0.8s)."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    f_t = 80.0 + 380.0 * (t / duration) ** 2.4
    phase = 2 * np.pi * np.cumsum(f_t) / SR
    sweep = np.sin(phase)
    noise = np.random.randn(n) * 0.35
    env = (t / duration) ** 2.2
    return (sweep * 0.7 + noise * 0.3) * env


def gen_boom_shimmer(duration: float = 1.5) -> np.ndarray:
    """Deep bass boom plus soft rising shimmer for Reveal."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    boom_f = 50.0 * (0.7 + 0.5 * np.exp(-t / 0.15))
    boom_phase = 2 * np.pi * np.cumsum(boom_f) / SR
    boom = np.sin(boom_phase) * np.exp(-t / 0.45)
    
    shimmer = (
        0.3 * np.sin(2 * np.pi * 920.0 * t) +
        0.2 * np.sin(2 * np.pi * 1380.0 * t) +
        0.15 * np.sin(2 * np.pi * 1840.0 * t)
    )
    shimmer_env = (1.0 - np.exp(-t / 0.1)) * np.exp(-t / 0.6)
    return boom * 0.8 + shimmer * shimmer_env * 0.35


def gen_glow_hum(duration: float = 3.5) -> np.ndarray:
    """Gentle sustained warm glow hum while reveal box is shown."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    hum = (
        0.6 * np.sin(2 * np.pi * 130.81 * t) +
        0.3 * np.sin(2 * np.pi * 261.63 * t) +
        0.15 * np.sin(2 * np.pi * 392.44 * t)
    )
    env = np.ones_like(t)
    env[:int(SR * 0.3)] = np.linspace(0, 1, int(SR * 0.3))
    env[-int(SR * 0.4):] = np.linspace(1, 0, int(SR * 0.4))
    return hum * env


def gen_whoosh(duration: float = 0.5) -> np.ndarray:
    """Smooth whoosh during navbar movement."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.randn(n)
    env = np.sin(np.pi * (t / duration)) ** 2
    f_t = 180.0 + 500.0 * np.sin(np.pi * (t / duration))
    carrier = np.sin(2 * np.pi * np.cumsum(f_t) / SR)
    return (noise * 0.6 + carrier * 0.4) * env


def gen_click(duration: float = 0.08) -> np.ndarray:
    """Subtle mechanical click when chip docks."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    click = np.random.randn(n) * np.exp(-t / 0.008)
    return click


def gen_tick(duration: float = 0.08) -> np.ndarray:
    """Soft tick when title appears."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tick = np.sin(2 * np.pi * 2200.0 * t) * np.exp(-t / 0.012)
    return tick


def gen_dissonance(duration: float = 1.0) -> np.ndarray:
    """Dull low dissonant tone when left rule disappears."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = (
        0.45 * np.sin(2 * np.pi * 65.41 * t) +
        0.45 * np.sin(2 * np.pi * 69.30 * t) +
        0.25 * np.sin(2 * np.pi * 92.50 * t)
    )
    env = (1.0 - np.exp(-t / 0.04)) * np.exp(-t / 0.4)
    return tone * env


def gen_typing(duration: float = 2.7) -> np.ndarray:
    """Soft keyboard typing ticks during terminal typing (14.8s - 17.5s)."""
    n = int(SR * duration)
    audio = np.zeros(n)
    tick_times = np.linspace(0.05, duration - 0.1, 18) + np.random.uniform(-0.015, 0.015, 18)
    for tt in tick_times:
        idx = int(tt * SR)
        dur_tick = int(0.025 * SR)
        if idx + dur_tick < n:
            t_sub = np.linspace(0, 0.025, dur_tick)
            f = 1100 + np.random.uniform(-250, 250)
            k = (np.random.randn(dur_tick) * 0.35 + np.sin(2 * np.pi * f * t_sub) * 0.65) * np.exp(-t_sub / 0.006)
            audio[idx:idx + dur_tick] += k * 0.65
    return audio


def gen_chime(duration: float = 1.8) -> np.ndarray:
    """Clean bright confirmation chime when 10 (was 30) appears."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    chime = (
        0.5 * np.sin(2 * np.pi * 659.25 * t) * np.exp(-t / 0.7) +
        0.35 * np.sin(2 * np.pi * 987.77 * t) * np.exp(-t / 0.5) +
        0.25 * np.sin(2 * np.pi * 1318.51 * t) * np.exp(-t / 0.4) +
        0.15 * np.sin(2 * np.pi * 1975.53 * t) * np.exp(-t / 0.25)
    )
    return chime


def gen_blip(freq: float, duration: float = 0.12) -> np.ndarray:
    """Short soft blip for layer chain."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    blip = np.sin(2 * np.pi * freq * t)
    env = (1.0 - np.exp(-t / 0.008)) * np.exp(-t / 0.035)
    return blip * env


def gen_blip6_accent(duration: float = 0.6) -> np.ndarray:
    """Stronger accent note on Stage 6 (Shadow selection) at 23.0s."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    accent = (
        0.55 * np.sin(2 * np.pi * 739.99 * t) * np.exp(-t / 0.25) +
        0.35 * np.sin(2 * np.pi * 1108.73 * t) * np.exp(-t / 0.20) +
        0.20 * np.sin(2 * np.pi * 1479.98 * t) * np.exp(-t / 0.15)
    )
    return accent


def gen_chord(duration: float = 4.5) -> np.ndarray:
    """Single resolved clean chord/tone on Kernl.ai close (25.5s - 30.0s)."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    chord = (
        0.35 * np.sin(2 * np.pi * 130.81 * t) +
        0.30 * np.sin(2 * np.pi * 196.00 * t) +
        0.25 * np.sin(2 * np.pi * 261.63 * t) +
        0.20 * np.sin(2 * np.pi * 329.63 * t)
    )
    env = (1.0 - np.exp(-t / 0.2)) * np.exp(-t / 2.0)
    return chord * env


def main():
    print("Generating audio assets for 30s LinkedIn launch video...")
    save_wav("bed.wav", gen_bed(30.0), target_db=-24.0)
    save_wav("bubble_tick.wav", gen_bubble_tick(0.08), target_db=-14.0)
    save_wav("pain_tone.wav", gen_pain_tone(3.5), target_db=-14.0)
    save_wav("impact_vanish.wav", gen_impact_vanish(0.35), target_db=-9.0)
    save_wav("pain_riser.wav", gen_pain_riser(0.8), target_db=-11.0)
    
    save_wav("boom_shimmer.wav", gen_boom_shimmer(1.5), target_db=-9.0)
    save_wav("glow_hum.wav", gen_glow_hum(3.5), target_db=-22.0)
    save_wav("whoosh.wav", gen_whoosh(0.5), target_db=-13.0)
    save_wav("click.wav", gen_click(0.08), target_db=-13.0)
    save_wav("tick.wav", gen_tick(0.08), target_db=-14.0)
    
    save_wav("dissonance.wav", gen_dissonance(1.0), target_db=-11.0)
    save_wav("typing.wav", gen_typing(2.7), target_db=-14.0)
    save_wav("chime.wav", gen_chime(1.8), target_db=-8.0)
    
    blip_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 523.25]
    for i, f in enumerate(blip_freqs, 1):
        save_wav(f"blip{i}.wav", gen_blip(f, 0.12), target_db=-12.0)
    save_wav("blip6_accent.wav", gen_blip6_accent(0.6), target_db=-8.0)
    
    save_wav("chord.wav", gen_chord(4.5), target_db=-9.0)
    print("All 30-second audio assets generated successfully.")


if __name__ == "__main__":
    main()
