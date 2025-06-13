import librosa
import soundfile as sf

audio, sr = librosa.load('HYBS_TIP_TOE_short.m4a', sr=44100, mono=False)
sf.write('HYBS_TIP_TOE_short.wav', audio.T, sr, subtype='PCM_32')
print(f'Converted HYBS_TIP_TOE_short.m4a to HYBS_TIP_TOE_short.wav (32-bit PCM WAV)')
print(f'Shape: {audio.shape}, Sample rate: {sr}')
print(f'Min: {audio.min():.6f}, Max: {audio.max():.6f}')