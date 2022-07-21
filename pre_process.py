from pathlib import Path
import argparse
import json
from data_utils import PreProcAudio

if __name__ == "__main__":
  parser = argparse.ArgumentParser("sed_inference")
  parser.add_argument('--path', type=str, default='/home/teo/userdata/nia_dataset',
                      help='Directory path of the dataset')
  parser.add_argument('--vocab_path', type=str, default='vocab.json',
                      help='json path for the vocabulary')
  parser.add_argument('--sr', type=int, default=32000,
                      help='Target sampling rate')

  args = parser.parse_args()

  # with open(args.vocab_path, 'r') as f:
  #   vocab = json.load(f)

  dataset = PreProcAudio(args.path, args.sr, pre_process=True)

  sample_out = dataset[0]
  print(sample_out)