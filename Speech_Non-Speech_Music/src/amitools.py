import os
import numpy as np
import re
# ------------
from search_utils import binary_search

AUDIO_PATHS = [(x[0] + '/' + x[2][0], 
  re.search("([a-zA-Z]{2}[0-9]{4}[a-zA-Z]*)", x[2][0]).group(0)) 
    for x in os.walk('../amicorpus/') if x[2] and '.wav' in x[2][0]]
N_AUDIO_FILES = len(AUDIO_PATHS)

WORDS_DIR = '../ami_public_manual_1.6.2/words/'
TAGS_TO_WORDS = {tag:[WORDS_DIR + x for x in next(os.walk(WORDS_DIR))[2] if 
  tag in x] for _, tag in AUDIO_PATHS}

TAGS_TO_LENGTHS = {}
if not os.path.exists('./TAGS_TO_LENGTHS.txt'):
  print("Making text file for later use. This may take a minute.")
  with open('./TAGS_TO_LENGTHS.txt', 'w+') as file:
    for path, tag in AUDIO_PATHS:
      audio, sr = librosa.load(path, sr=None)
      length = librosa.core.get_duration(audio, sr=sr)
      file.write(f"{tag},{length}" + '\n')
      TAGS_TO_LENGTHS[tag] = length
else:
  print("Text file found. You've saved a minute.")
  with open('./TAGS_TO_LENGTHS.txt', 'r') as file:
    for line in file:
      data = line.strip().split(',')
      tag, length = data[0], float(data[1])
      TAGS_TO_LENGTHS[tag] = length

TEST_PARTITION = ['ES2004', 'IS1009', 'TS3003', 'EN2002']
VAL_PARTITION = ['ES2011', 'IS1008', 'TS3004', 'IB4001', 
                 'IB4002', 'IB4003', 'IB4004', 'IB4010', 
                 'IB4011']
TRAIN_PARTITION = ['ES2002', 'ES2003', 'ES2005', 'ES2006', 
                   'ES2007', 'ES2008', 'ES2009', 'ES2010', 
                   'ES2012', 'ES2013', 'ES2014', 'ES2015', 
                   'ES2016', 'IS1000', 'IS1001', 'IS1002', 
                   'IS1003', 'IS1004', 'IS1005', 'IS1006', 
                   'IS1007', 'TS3005', 'TS3006', 'TS3007', 
                   'TS3008', 'TS3009', 'TS3010', 'TS3011', 
                   'TS3012', 'EN2001', 'EN2003', 'EN2004a', 
                   'EN2005a', 'EN2006', 'EN2009', 'IN1001', 
                   'IN1002', 'IN1005', 'IN1007', 'IN1008', 
                   'IN1009', 'IN1012', 'IN1013', 'IN1014', 
                   'IN1016']

for _, tag in AUDIO_PATHS:
  idx = tag[0:6]
  if tag in TEST_PARTITION:
    assert tag not in VAL_PARTITION
    assert tag not in TRAIN_PARTITION
  elif idx in TEST_PARTITION:
    assert idx not in VAL_PARTITION
    assert idx not in TRAIN_PARTITION
  elif tag in TRAIN_PARTITION:
    assert tag not in VAL_PARTITION
  elif idx in TRAIN_PARTITION:
    assert idx not in VAL_PARTITION

def get_ami_part(tag):
  if tag in TRAIN_PARTITION or tag[0:6] in TRAIN_PARTITION:
    return 'train'
  elif tag in TEST_PARTITION or tag[0:6] in TEST_PARTITION:
    return 'test'
  else:
    return 'val'

def get_word_data(tag, tags_to_words=TAGS_TO_WORDS):

  # returns an array of data containing the start and end time 
  # of all utterrances in all annotations associated to 
  # the given tag, sorted by when the utterance begins.
  
  rgx = r'(starttime=\s*\")([0-9]*.[0-9]*)\"\s*' + \
    r'(endtime=\s*\")([0-9]*.[0-9]*)\"[^>]*>' + \
    r'([A-Za-z][A-Za-z]*[&#39;]*[A-Za-z]*)'

  data = []
  for path in tags_to_words[tag]:
    with open(path, 'r') as file:
      words = [(match.group(2), match.group(4), match.group(5)) for match 
        in re.finditer(rgx, file.read())]
      data.extend(words)
  data.sort(key=lambda x: float(x[0]))

  return data

def get_speech_segments(tags=None, tags_to_words=TAGS_TO_WORDS, 
                        tags_to_lengths=TAGS_TO_LENGTHS, cutoff=0):

  # returns two dictionaries for speech
  # and non-speech, which map tags to
  # an array of all contiguous segments
  # of speech/non-speech stored as a start
  # and end time.
  
  tags2speech, tags2nonspeech = {}, {}
  
  if tags == None:
    tags = list(tags_to_words.keys())

  for tag in tags:
    
    data = get_word_data(tag, tags_to_words)

    n_utterances = len(data)
    duration = tags_to_lengths[tag]

    speech_start, speech_end, i = float(data[0][0]), float(data[0][1]), 0
    speech, nonspeech = [], [(0, speech_start)] if speech_start > 0 else []
    
    while i < n_utterances-1:
      
      # for a given speech segment, we look ahead until
      # we find an utterance that begins a new segment.
      # some of the source data contains errors, we pass 
      # over these rows
      for j in range(i+1, n_utterances):
        
        utterance_start, utterance_end = float(data[j][0]), float(data[j][1])
        
        if utterance_end > utterance_start and utterance_end <= duration:

          if utterance_start <= speech_end:
            speech_end = max(utterance_end, speech_end)
          else:
            i = j
            nonspeech.append((speech_end, utterance_start))
            speech.append((speech_start, speech_end))
            speech_start, speech_end = utterance_start, utterance_end
          
        if j == n_utterances-1:
          i = j
          speech.append((speech_start, speech_end))
          assert duration >= speech_end
          if duration > speech_end:
            nonspeech.append((speech_end, duration))
    
    tags2speech[tag] = [(a,b) for a, b in speech if b-a > cutoff]
    tags2nonspeech[tag] = [(a,b) for a, b in nonspeech if b-a > cutoff]

  return tags2speech, tags2nonspeech

def make_labels(tag, hop_length_in_s, window_length_in_s, irange, 
                search_type='start', center=False, tags_to_words=TAGS_TO_WORDS, 
                tags_to_lengths=TAGS_TO_LENGTHS):
  labels = {}
  tags2speech, tags2nonspeech = get_speech_segments([tag], 
    tags_to_words=tags_to_words, tags_to_lengths=tags_to_lengths)

  for i in range(irange):
    
    if center == False:
      start = hop_length_in_s*i
      end = start + window_length_in_s
    else:
      start = min(hop_length_in_s*i - window_length_in_s/2, 0)
      end = start + window_length_in_s

    val = (start+end)/2 if search_type == 'midpoint' else start
    label = binary_search(tags2speech[tag], 0, len(tags2speech[tag])-1, 
      val, search_type, window_length_in_s)
    labels[i] = label
      
  return labels





