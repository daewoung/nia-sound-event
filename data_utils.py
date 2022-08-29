from unicodedata import category
import torch 
import torchaudio
from pathlib import Path
from tqdm.auto import tqdm


class NeutuneSet:
  def __init__(self, path, sr=32000) -> None:
    self.path = Path(path)
    self.wav_list = sorted(list(self.path.rglob('*.wav')))
    self.mp3_list = sorted(list(self.path.rglob('*.mp3')))
    self.sr = sr

  def __len__(self):
    return len(self.wav_list)
  
  def __getitem__(self, idx):
    wav_path = self.wav_list[idx]
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(0)
    if sr != self.sr:
      audio = torchaudio.functional.resample(audio, sr, self.sr)
    return audio

class OnFlyAudio(NeutuneSet):
  def __init__(self, path, sr=32000) -> None:
    super().__init__(path, sr)

  def __getitem__(self, idx):
    wav_path = self.wav_list[idx]
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(0)
    if sr != self.sr:
      audio = torchaudio.functional.resample(audio, sr, self.sr)
    return audio

class PreProcAudio(NeutuneSet):
  def __init__(self, path, sr=32000, pre_process=False) -> None:
    super().__init__(path, sr)
    if pre_process:
      self.check_and_save_pt()
  
  def check_and_save_pt(self, skip_exist=True, vocab=None):
    '''
    Arguments:
      skip_exist (bool): Skip pre-processing for file which have a corersponding pt file
      vocab (list, optional): index-to-category_name 
    '''
    for wav_path in tqdm(self.wav_list):
      pt_path = wav_path.with_suffix('.pt')
      if skip_exist and pt_path.exists() :
        continue
      audio, sr = torchaudio.load(wav_path)
      audio = audio.mean(0)
      if sr != self.sr:
        audio = torchaudio.functional.resample(audio, sr, self.sr)
      category_index = int(pt_path.stem.split('-')[3]) - 1
      data_dict = {'audio': audio, 'label':category_index}
      if isinstance(vocab, list):
        data_dict['label-str']: vocab[category_index]
      torch.save(data_dict, pt_path)

  
  def __getitem__(self, idx):
    '''
    
    Output:
      audio_sample (torch.Tensor): 1D tensor
      category_label (int)
    '''
    pt_path = self.wav_list[idx].with_suffix('.pt')
    datasample = torch.load(pt_path)

    return datasample['audio'], datasample['label']


def pad_collate_with_label(raw_batch):
  '''
  Collate function for classification task with arbitrary length of audio
  
  Outputs:
    audio_tensor (torch.Tensor): Padded audio tensor
    label_tensor (torch.LongTensor): Label for each data sample in the batch
  '''
  lens = [len(x[0]) for x in raw_batch]
  max_len = max(lens)
  output = torch.zeros(len(raw_batch), max_len)

  for i, sample in enumerate(raw_batch):
    output[i, :len(sample[0])] = sample[0]
  
  return output, torch.LongTensor([x[1] for x in raw_batch])

def pad_collate(raw_batch):
  lens = [len(x) for x in raw_batch]
  max_len = max(lens)
  output = torch.zeros(len(raw_batch), max_len)

  for i, sample in enumerate(raw_batch):
    output[i, :len(sample)] = sample
  
  return output, lens