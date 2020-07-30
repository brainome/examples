import os
#os.environ['W'] = 'ignore'
import librosa
import argparse
import re
import numpy as np
from progress.bar import Bar
from functools import reduce
# ------------
from amitools import make_labels, get_ami_part
from DCASEtools import get_DCASE_paths
from feature_extraction import xtract, xtract_coarse

BITS_PER_GB = 8589934592
EXTENSIONS = ['.wav', '.mp3', '.flac']


def get_part(path):
  if 'ami' in path:
    tag = re.search("([a-zA-Z]{2}[0-9]{4}[a-zA-Z]*)", path).group(0)
    return get_ami_part(tag)
  if 'train' in path:
    return 'train'
  elif 'test' in path:
    return 'test'
  elif 'val' in path:
    return 'val'
  else:
    if np.random.uniform(low=0., high=1.) > 0.1:
      return 'train'
    else:
      return 'test'


def get_paths(source):
  if 'DCASE' in source:
    return get_DCASE_paths()
  if isinstance(source, str):
    return get_paths_single(source)
  elif isinstance(source, list):
    return get_paths_multisource(source)
  else:
    raise ValueError


def get_paths_single(source):
  root = source
  paths = [[x[0] + '/' + y for y in x[2] if 
    any(extension in y for extension in EXTENSIONS)] for 
      x in os.walk(root)]
  paths = reduce((lambda a,b: a+b), paths)
  np.random.shuffle(paths)
  return paths


def get_paths_multisource(sources):
  res = []
  for root in sources:
    paths = [[x[0] + '/' + y for y in x[2] if 
      any(extension in y for extension in EXTENSIONS)] for 
        x in os.walk(root)]
    paths = reduce((lambda a,b: a+b), paths)
    np.random.shuffle(paths)
    res.extend(paths)
  np.random.shuffle(res)
  return res


def make_mfcc_data(data_paths, 
                   save_path,
                   audio_sr=None, 
                   alpha=0.03, 
                   beta=0.01, 
                   n_interleavings=1, 
                   n_mfcc=19, 
                   start_index=1, 
                   n_mels=128, 
                   fmax=None,
                   center=False, 
                   max_gb=25.,
                   res_gb=1.,
                   coarse=False,
                   balance=False, 
                   delete=False,
                   reuse=False):  


  n_cols = (n_mfcc-start_index)*n_interleavings + 1
  if coarse: n_cols += (n_mfcc-start_index)
  max_rows = int(max_gb*BITS_PER_GB/(1.3*2*32*n_cols))
  labels = range(len(data_paths))


  if not os.path.exists(save_path):
    os.makedirs(save_path)


  audio_paths = {}
  for i, datpath in enumerate(data_paths):
    if datpath:
      audio_paths[i] = get_paths(datpath)
    else:
      audio_paths[i] = []

  
  if reuse:
    usage = 'r'
  else:
    usage =  'w+'


  fm = {}
  for label in labels:
    fm[label] = {part:open(save_path + '{}.{}.csv'.format(part, str(label)), 
      usage) for part in ['train', 'test']}

  
  for label in labels:


    if reuse:
      continue


    paths = audio_paths[label]
    n_rows = 0
    bar = Bar("Making MFCCs for {} data.".format(str(label)), max=len(paths))
    BREAK = False


    for pid, path in enumerate(paths):


      bar.next()
      is_ami = 'amicorpus' in path


      if BREAK:
        break


      part = get_part(path)
      if part == 'val':
        continue
      

      try:
        audio, sr = librosa.load(path, sr=audio_sr)
      except:
        continue


      if sr == 0:
        continue


      if len(audio) < int(sr*alpha):
      	continue


      if coarse:
        xfeatures = xtract_coarse(audio, sr, alpha, beta, n_interleavings, 
          start_index, n_mfcc, n_mels, fmax)        
      else:
        xfeatures = xtract(audio, sr, alpha, beta, n_interleavings, 
          start_index, n_mfcc, n_mels, fmax)


      n_instances = xfeatures.shape[0]


      if is_ami:
        window_length_in_s = alpha + (n_interleavings-1)*beta
        tag = re.search("([a-zA-Z]{2}[0-9]{4}[a-zA-Z]*)", path).group(0)
        amilabels = make_labels(tag, hop_length_in_s=beta, 
          window_length_in_s=window_length_in_s, irange=n_instances)


      for i in range(0, n_instances, 2):


        if is_ami and amilabels[i] != 1:
          continue
        

        file = fm[label][part]
        file.write(",".join(str(x) for x in xfeatures[i]) + 
          ',' + str(label) + '\n')
        

        if part == 'train':
          n_rows += 1


        if n_rows >= max_rows:
          BREAK = True
          break


    bar.finish()


  for part in ['train', 'test']:
    for label in labels:
      file = fm[label][part]
      file.close()

  
  if balance:


    for part in ['train', 'test']:
      

      res = open(save_path + 'combined_{}.csv'.format(str(part)), 'w+')
      fpaths = [fm[label][part].name for label in fm]
      sizes = [os.path.getsize(path)/1000000000 for path in fpaths]
      out_size = min(sizes)*len(sizes)
      

      try:
        reject_probs = [min(sizes)/size for size in sizes]
      except:
        if delete:
          for fpath in fpaths:
            os.remove(fpath)
        continue


      if out_size > res_gb:
        base_reject = res_gb/out_size
        reject_probs = [x*base_reject for x in reject_probs]


      for i, fpath in enumerate(fpaths):
        

        with open(fpath, 'r') as f:
          for line in f:
            if np.random.uniform() < reject_probs[i]:
              res.write(line)
        

        if delete:
          os.remove(fpath)
      

      res.close()






