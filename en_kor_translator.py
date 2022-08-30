import pandas as pd
import json
from pathlib import Path
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser("en_kor_translator")
  parser.add_argument('--target_path', type=str, default='prediction_threshold0.05.json',
                      help='json file to translate')
  
  args = parser.parse_args()
  
  ontology = pd.read_csv('audioset_ontology_translate.csv')
  
  ontology_dict = {}
  ontology_dict = {en_name:kor_name for en_name, kor_name in zip(ontology['name'], ontology['kor_name'])}

  prediction_path = Path(args.target_path)
  with open(prediction_path, 'r') as f:
    pred_json = json.load(f)
    
  for i in range(len(pred_json)):
    for j in range(len(pred_json[i]['predictions'][0]['result'])):
      en_name = pred_json[i]['predictions'][0]['result'][j]['value']['labels'][0]
      kor_name = ontology_dict[en_name]
      pred_json[i]['predictions'][0]['result'][j]['value']['labels'][0] = kor_name
  
  with open(f"translated_{args.target_path}", "w") as json_file:
    json.dump(pred_json, json_file, ensure_ascii=False)