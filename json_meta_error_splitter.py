from pathlib import Path
import argparse
from data_utils import NeutuneSet, OnFlyAudio
from meta_utils import MetaCreator
import json

if __name__ == "__main__":
  parser = argparse.ArgumentParser("json_meta_error_splitter")
  parser.add_argument('--target_path', type=str, default='prediction_threshold0.05.json',
                      help='json file to break error list')
  
  dataset = OnFlyAudio('/home/clay/label_studio_files/data_v4/02/')
  meta_creator = MetaCreator('vocab.json')
  entire_meta = [meta_creator.create_meta_for_wav(wav) for wav in dataset.wav_list]

  error_dict = meta_creator.error_dict
  error_dict_list = [item for values in error_dict.values() for item in values]
  
  prediction_path = Path('prediction_threshold0.05.json')
  with open(prediction_path, 'r') as f:
    pred_json = json.load(f)
    
  new_pred_json = []
  
  # for i in range(len(error_dict_list)):
  for j in range(len(pred_json)):
    # if error_dict_list[i][:-4] in pred_json[j]['data']['audio']:
    file_name = Path(pred_json[j]['data']['audio']).with_suffix(".wav").name
    if file_name in error_dict_list:
      continue
    else:
      new_pred_json.append(pred_json[j])
        
  with open("meta_prediction_threshold0.05.json", "w") as json_file:
    json.dump(new_pred_json, json_file, ensure_ascii=False)