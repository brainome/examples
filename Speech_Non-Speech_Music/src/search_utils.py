def slow_search(arr, val, window_length_in_s):
  for (a,b) in arr:
    if val + window_length_in_s < a:
      break
    if val >= a and val + window_length_in_s <= b:
      return 1
    elif val >= a and val < b and val + window_length_in_s > b:
      return -1
    elif val < a and val + window_length_in_s > a:
      return -1
  return 0

def binary_search_by_start(arr, start, end, val, window_length_in_s):

  guess = (start+end)//2
  a, b = arr[guess]
  
  if val >= a and val + window_length_in_s <= b: # entirely in
    return 1
  elif val >= a and val < b and val + window_length_in_s > b: # partialy in
    return -1
  elif val < a and val + window_length_in_s > a: # partially in
    return -1
  elif val < a and val + window_length_in_s <= a: # left of
    if guess == 0 or start == end:
      return 0
    else:
      return binary_search_by_start(arr, start, max(start, guess-1), val, 
        window_length_in_s)
  elif val >= b: # right of
    if guess == len(arr) or start == end:
      return 0
    else:
      return binary_search_by_start(arr, min(end, guess+1), end, val, 
        window_length_in_s)
  print('check')

def binary_search_by_midpoint(arr, start, end, val):
  if start >= end:
    return 0

  guess = (start+end)//2
  
  if val >= arr[guess][0] and val <= arr[guess][1]: # in [guess[0], guess[1]]
    return 1
  elif val < arr[guess][0]: # left of [guess[0], guess[1]]
    return binary_search_by_midpoint(arr, start, guess-1, val)
  elif val > arr[guess][1]: # right of [guess[0], guess[1]]
    return binary_search_by_midpoint(arr, guess+1, end, val)

def binary_search(arr, start, end, val, search_type, window_length_in_s):
  if search_type == 'midpoint':
    return binary_search_by_midpoint(arr, start, end, val)
  elif search_type == 'start':
    return binary_search_by_start(arr, start, end, val, window_length_in_s)
  elif search_type == 'slow':
    return slow_search(arr, val, window_length_in_s)