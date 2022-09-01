import json
from pathlib import Path
import argparse
import random

if __name__ == "__main__":
  parser = argparse.ArgumentParser("pred_json_divider")
  parser.add_argument('--path', type=str, default='translated_meta_prediction_threshold0.05.json',
                      help='directory path to the json file')
  parser.add_argument('--hop_size', type=int, default=30,
                      help='hop size between two consecutive data')
  parser.add_argument('--iter', type=int, default=40,
                      help='iter number for divide')
  parser.add_argument('--sampling', type=int, default=0,
                      help='choose to sample the data 0 is False, 1 is True')
  
  args = parser.parse_args()
  
  prediction_path = Path(args.path)
  with open(prediction_path, 'r') as f:
    pred_json = json.load(f)
  
  if args.sampling:
    random.seed(4)
    random.shuffle(pred_json)
  
  for i in range(args.iter):
    divided_pred_json = pred_json[args.hop_size*i:args.hop_size*(i+1)]
  
    with open(f"kor_data_v4_03_1200_sampled/{args.hop_size*i}to{args.hop_size*(i+1)}.json", "w") as json_file:
      json.dump(divided_pred_json, json_file, ensure_ascii=False)