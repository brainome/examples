import argparse
import yaml
from mfcc import make_mfcc_data


if __name__ == "__main__":

  
  parser = argparse.ArgumentParser()
  parser.add_argument('-gb', '--max_gb', type=float, default=1., help="Maximum size of each individual CSV of MFCCs per class.")
  parser.add_argument('--outsize', type=float, default=1., help="Maximum size of output csv which contains the combined data from all classes.")
  parser.add_argument('--reuse', '-r', action='store_true', help="If true, reuse existing saved files.")
  parser.add_argument('--yaml', '-y', type=str, default='foo.yaml', help="Name of .yaml file in working directory to be used.")
  parser.add_argument('--notcoarse', '-nc', action='store_false')
  args = parser.parse_args()


  doc = yaml.load(open(args.yaml, 'r'))


  make_mfcc_data(data_paths=doc['data_paths'],
                 save_path=doc['save_path'],
                 audio_sr=doc['sampling_rate'],
  	             alpha=doc['alpha'], 
                 beta=doc['beta'], 
                 n_interleavings=doc['n_interleavings'], 
                 n_mfcc=doc['n_mfcc'], 
                 start_index=doc['start_index'], 
                 n_mels=doc['n_mels'],
                 fmax=doc['fmax'],
                 delete=doc['delete'],
                 center=False,
                 coarse=args.notcoarse,
                 balance=True,
                 max_gb=args.max_gb,
                 res_gb=args.outsize,
                 reuse=args.reuse)




