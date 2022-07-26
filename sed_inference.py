from panns_inference import SoundEventDetection
import numpy as np
import argparse
from pathlib import Path
import torchaudio
from torch.utils.data import DataLoader
import torch
import json
from tqdm.auto import tqdm 
from data_utils import OnFlyAudio, pad_collate
from meta_utils import MetaCreator


DEV = 'cuda'



class Smoother(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.conv = torch.nn.Conv1d(1, 1, )

  def forward(self, pred):
    return


def quantize_prediction(pred, lens, threshold=0.1, shifted_index=0):
  assert pred.ndim == 3
  th_pred = (pred > threshold).astype(int)
  th_pred = np.concatenate([np.zeros([th_pred.shape[0], 1, th_pred.shape[2]]), th_pred], axis=1) # add initial step
  diff_pred = np.diff(th_pred, axis=1).transpose(0,2,1)
  onset = np.nonzero(diff_pred==1)
  offset =  np.nonzero(diff_pred==-1)
  events_of_piece = []
  total_events = []
  j = 0
  prev_batch_id = -1

  for i, (batch_id, tag_id, onset_frame) in enumerate(zip(onset[0], onset[1], onset[2])):
    event = {'data_id': int(batch_id + shifted_index), 'label':sed.labels[tag_id], 'onset': float(onset_frame/100)}
    if j<len(offset[0]) and batch_id == offset[0][j] and tag_id == offset[1][j]:
      event['offset'] = float(offset[2][j] / 100)
      j += 1
    else:
      event['offset'] = lens[batch_id] / 32000
#       print(len(offset[0]), batch_id,  offset[0][j], tag_id,  offset[1][j] )
    event['confidence'] = max(pred[batch_id, onset_frame:int(event['offset']*100), tag_id])
  
    if batch_id == prev_batch_id:
      events_of_piece.append(event)
    else:
      if events_of_piece != []:
        events_of_piece = retain_one_event_per_tag(events_of_piece)
        total_events.append(events_of_piece)
      events_of_piece = []
    prev_batch_id = batch_id
  if events_of_piece != []:
    events_of_piece = retain_one_event_per_tag(events_of_piece)

    # sort by onset time
    events_of_piece.sort(key=lambda x:x['onset'])
    total_events.append(events_of_piece)

  return total_events

def retain_one_event_per_tag(events):
  '''
  events = list of tag events
  '''
  unique_tags = list(set([x['label'] for x in events]))
  outputs = []
  for tag in unique_tags:
    selected_events = [x for x in events if x['label']==tag]
    # TODO: find better score formular
    scores = [x['confidence'] * (x['offset']-x['onset']) for x in selected_events]
    max_id = scores.index(max(scores))
    outputs.append(selected_events[max_id])
  return outputs


def jsonify(event_labels, dataset, meta_manager:MetaCreator):
  '''
  event_labels (list of dict):
    each item has {'data_id': int, 'label': str, 'onset': float, 'offset': float}

  dataset (AudioSet)
  
  '''
  json_list = []

  for piece_event in event_labels:
    event_id = piece_event[0]['data_id']
    sample_path = str(dataset.wav_list[event_id].relative_to(dataset.path))
    sample_path = '/data/local-files/?d=nia_dataset/'+sample_path

    annotations =[{'id': 1,
                    'result': [{
                                "value":{"start":event['onset'],"end":event['offset'],"labels":[event['label']],"score":float(event["confidence"])},
                                "from_name":"label",
                                "to_name":"audio",
                                "type":"labels",
                              } for event in piece_event]}]
    json_event = {'id':event_id, 'predictions': annotations, 'data': {'audio': sample_path, 'text': meta_manager.get_class_name_and_title(dataset.wav_list[event_id])}}
    json_list.append(json_event)

  return json_list


if __name__ == "__main__":
  parser = argparse.ArgumentParser("sed_inference")
  parser.add_argument('--path', type=str, default='/home/teo/label-studio_files/nia_dataset/1cycle',
                      help='directory path to the dataset')
  parser.add_argument('--vocab_path', type=str, default='vocab.json',
                      help='directory path to the dataset')
  parser.add_argument('--threshold', type=float, default=0.1,
                      help='sound event detection threshold value')

  args = parser.parse_args()

  dataset = OnFlyAudio(args.path)
  meta_manager = MetaCreator(args.vocab_path)

  data_loader = DataLoader(dataset, batch_size=10, collate_fn=pad_collate, pin_memory=True, num_workers=2, drop_last=False)
  sed = SoundEventDetection(checkpoint_path=None, device='cuda')
  pred = []

  for i, batch in tqdm(enumerate(data_loader)):
    audio, lens = batch
    framewise_output = sed.inference(audio)

    pred += quantize_prediction(framewise_output, lens, shifted_index= i * data_loader.batch_size, threshold=args.threshold)
    break
  jsonified_pred = jsonify(pred, dataset, meta_manager)
  with open(f"prediction_threshold{str(args.threshold)}.json", "w") as json_file:
    json.dump(jsonified_pred, json_file, ensure_ascii=False)
  print(pred)
