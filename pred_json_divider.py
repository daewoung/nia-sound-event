import json
from pathlib import Path
import argparse
import random

if __name__ == "__main__":
  parser = argparse.ArgumentParser("pred_json_divider")
  parser.add_argument('--path', type=str, default='translated_meta_prediction_threshold0.05.json',
                      help='directory path to the json file')
  parser.add_argument('--inset', type=int, default=0,
                      help='index for divide in')
  parser.add_argument('--outset', type=int, default=85,
                      help='index for divied out')
  parser.add_argument('--sampling', type=int, default=0,
                    help='choose to sample the data 0 is False, 1 is True')
  
  args = parser.parse_args()
  
  prediction_path = Path(args.path)
  with open(prediction_path, 'r') as f:
    pred_json = json.load(f)
    
  if args.sampling:
    random.seed(4)
    random.shuffle(pred_json)
    
  divided_pred_json = pred_json[args.inset:args.outset]
  
  with open(f"{Path(args.path).stem}_{args.inset}to{args.outset}.json", "w") as json_file:
    json.dump(divided_pred_json, json_file, ensure_ascii=False)