import numpy as np

total = 72984
smallest_class = 972
DCASE_probs = {}
DCASE_probs['absence'] = 1.#smallest_class/18860
DCASE_probs['cooking'] = smallest_class/5124
DCASE_probs['dishwashing'] = smallest_class/1424
DCASE_probs['eating'] = smallest_class/2308
DCASE_probs['other'] = smallest_class/2060
DCASE_probs['social_activity'] = smallest_class/4944
DCASE_probs['vacuum_cleaner'] = smallest_class/972
DCASE_probs['watching_tv'] = smallest_class/18648
DCASE_probs['working'] = smallest_class/18644

def get_DCASE_paths():
  nonspeech_paths = []
  with open('../DCASE/meta.txt', 'r') as f:
    for line in f:
      try:
        path_to_wav, activity, idx = line.strip().split('\t')
        if activity in ['social_activity', 'watching_tv']:
          continue
        if activity not in DCASE_probs:
          print(activity)
          raise Exception
        if np.random.uniform() > 2*DCASE_probs[activity]:
          continue
        nonspeech_paths.append('../DCASE/' + path_to_wav)
      except:
        continue
  np.random.shuffle(nonspeech_paths)
  return nonspeech_paths