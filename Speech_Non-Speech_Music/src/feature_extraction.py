import numpy as np
import librosa

def xtract(audio, sr, alpha=0.01, beta=0.01, n_interleavings=5,
  start_index=1, n_mfcc=20, n_mels=128, fmax=None):

  '''
  audio: raw audio
  sr: sampling rate of the raw audio
  alpha: window/sub-window size
  beta: hop length
  n_interleavings: number of sub windows
  '''

  if fmax is None:
    fmax = sr//2

  frames_per_window, hop_length = int(sr*alpha), int(sr*beta)
  window_length_in_s = alpha + (n_interleavings-1)*beta

  mfccs = librosa.feature.mfcc(audio, sr=sr, 
    n_mfcc=n_mfcc, n_fft=frames_per_window, 
    hop_length=hop_length, center=False,
    n_mels=n_mels, fmax=fmax)
  
  n_instances = mfccs.shape[1]-n_interleavings+1
  n_xfeatures = (n_mfcc-start_index)*n_interleavings

  res = np.zeros((n_instances, n_xfeatures)).astype(np.float32)
  for i in range(n_instances):
    res[i] = mfccs[start_index:, i:i+n_interleavings].reshape(n_xfeatures)
  return res

def xtract_coarse(audio, sr, alpha=0.01, beta=0.01, n_interleavings=5,
  start_index=1, n_mfcc=20, n_mels=128, fmax=None):

  '''
  audio: raw audio
  sr: sampling rate of the raw audio
  alpha: window/sub-window size
  beta: hop length
  n_interleavings: number of sub windows
  '''

  if fmax is None:
    fmax = sr//2

  frames_per_window, hop_length = int(sr*alpha), int(sr*beta)
  window_length_in_s = alpha + (n_interleavings-1)*beta

  mfccs = librosa.feature.mfcc(audio, sr=sr, 
    n_mfcc=n_mfcc, n_fft=frames_per_window, 
    hop_length=hop_length, center=False,
    n_mels=n_mels, fmax=fmax)

  mfccs2 = librosa.feature.mfcc(audio, sr=sr, 
    n_mfcc=n_mfcc, n_fft=int(sr*window_length_in_s), 
    hop_length=hop_length, center=False,
    n_mels=n_mels, fmax=fmax)
  
  n_instances = mfccs.shape[1]-n_interleavings+1
  n_xfeatures = (n_mfcc-start_index)*n_interleavings

  res = np.zeros((n_instances, n_xfeatures+(n_mfcc-start_index))).astype(np.float32)
  for i in range(n_instances):
    res[i] = np.concatenate([mfccs[start_index:, i:i+n_interleavings].reshape(n_xfeatures), mfccs2[start_index:, i]])
  return res




  