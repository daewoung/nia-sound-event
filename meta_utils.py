import json
from pathlib import Path
from copy import copy
from docx.api import Document
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from math import isnan

from collections import defaultdict

class MetaCreator:
  def __init__(self, vocab_path, dataset_dir = None):
    with open(vocab_path, 'r') as f:
      self.vocab = json.load(f)
    self.error_list = []
    self.dataset_dir =dataset_dir
    self.audio_feature_extractor = AudioFeatureExtractor()
    self.error_dict = defaultdict(list)

  def get_recording_type(self, wav_path):
    if 'fs' in wav_path:
      return '폴리/합성'
    elif 'r' in wav_path:
      return '필드 레코딩'
    # if '폴리' in wav_path:
    #   return '폴리'
    # elif '합성' in wav_path:
    #   return '합성'
    # elif '필드' in wav_path:
    #   return '필드 레코딩'

  def check_name_is_wrong(self, wav_path):
    name = wav_path.stem
    high, middle, fine = [int(x) for x in name.split('-')[1:4]]
    voc = self.vocab[fine-1]
    assert voc["소분류_번호"] == fine
    if high != voc['대분류_번호']:
      print(f"Correct High: {voc['대분류_번호']} / Current High: {high} / Current Fine: {fine}")
      return True
    if middle != voc['중분류_번호']:
      print(f"Correct middle: {voc['중분류_번호']} / Current middle: {middle} / Current Fine: {fine}")

      return True
    return False

  def add_error(self, wav_path, error_str):
    self.error_list.append({'file': wav_path.name, 'error': error_str})
    self.error_dict[error_str].append(wav_path.name)

  def read_docx(self, wav_path):
    '''
    read docx for corresponding wav_path
    '''
    docx_path = wav_path.parent.parent / 'metadata' / wav_path.with_suffix('.docx').name
    detailed_meta = self.read_table_from_docx(docx_path)

    return detailed_meta

  def create_meta_for_wav(self, wav_path):
    if isinstance(wav_path, str):
      wav_path = Path(wav_path)
    category_idx = get_categorical_idx_from_fp(wav_path)
    
    # try:
    #   mp3_dur = self.check_mp3_format(wav_path)
    # except:
    #   print(f"Error occured while reading mp3 of {wav_path}")


    if self.check_name_is_wrong(wav_path):
      # print(f"Name is wrong for {wav_path}")
      self.add_error(wav_path, "Category is wrong")
    meta = copy(self.vocab[category_idx])
    meta['녹음 방식'] = self.get_recording_type(str(wav_path))

    # y, sr = torchaudio.load(wav_path)
    ob = sf.SoundFile(wav_path)
    # if abs(mp3_dur - ob.frames / ob.samplerate) > 0.2:
    #   print(f"Duration of MP3 and Wav is different: {ob.frames / ob.samplerate}, {mp3_dur}")

    meta['샘플링 레이트'] = ob.samplerate
    meta['채널 수'] = ob.channels
    meta['Bit_depth'] = int(ob.subtype.split('_')[1])
    if meta['Bit_depth'] != 24:
      # print(f"Bit-depth of {wav_path} is {meta['Bit_depth']}")
      self.add_error(wav_path, f"Bit-depth is {meta['Bit_depth']}")
    if meta['채널 수'] != 2:
      print(f"Num Channels of {wav_path} is {meta['채널 수']}")
      self.add_error(wav_path, f"Num channels is {meta['채널 수']}")
    if meta['샘플링 레이트'] != 96000:
      print(f"Sampling rate of {wav_path} is {meta['샘플링 레이트']}")
      self.add_error(wav_path, f"Sampling rate is {meta['샘플링 레이트']}")

    meta['길이'] = ob.frames / ob.samplerate

    docx_path = wav_path.parent.parent / 'metadata' / wav_path.with_suffix('.docx').name
    excel_path = docx_path.with_suffix('.xlsx')
    if excel_path.exists():
      try:
        detailed_meta = pd.read_excel(excel_path, index_col=0)
        if len(detailed_meta.columns) == 3:
          detailed_meta = pd.read_excel(excel_path,index_col=1)
          if "1차 원시데이터 기록표 (필드레코딩)" in detailed_meta:
            detailed_meta.pop("1차 원시데이터 기록표 (필드레코딩)")
          elif "1차 원시데이터 기록표 (폴리/합성)" in detailed_meta:
            detailed_meta.pop("1차 원시데이터 기록표 (폴리/합성)")
          detailed_meta = detailed_meta.rename(columns={'Unnamed: 2': 'Unnamed: 1'})
        # if meta['녹음 방식'] == "필드":
        detailed_meta = detailed_meta.T
        
        if "제목" in detailed_meta:
          meta["제목"] = detailed_meta.pop("제목")[0]
        elif "상황 묘사(제목) " in detailed_meta:
          meta["제목"] = detailed_meta.pop("상황 묘사(제목) ")[0]
        detailed_meta = detailed_meta.T
        detailed_dict = detailed_meta.to_dict()['Unnamed: 1']

        keys = detailed_dict.keys()
        change_keys = []
        for key in keys:
          if key[-1] == ' ':
            change_keys.append(key)
        for key in change_keys:
          detailed_dict[key[:-1]] = detailed_dict[key]
          del detailed_dict[key]

        meta["상세 정보"] = {}
        if "녹음/제작 연/월/일" in detailed_dict:
          meta["상세 정보"]["녹음/제작 연/월/일"] = detailed_dict["녹음/제작 연/월/일"]
        elif "제작 연/월/일" in detailed_dict:
          meta["상세 정보"]["녹음/제작 연/월/일"] = detailed_dict["제작 연/월/일"]
        else:
          meta["상세 정보"]["녹음/제작 연/월/일"] = detailed_dict["녹음 연/월/일"]
        
        if '사용 기자재' in detailed_dict:
          meta["상세 정보"]["사용 기자재"] = detailed_dict["사용 기자재"]
        else:
          for a in detailed_meta.loc['제작시 사용 개체'].values.flatten():
            if isinstance(a, str):
              meta['상세 정보']['사용 기자재'] = a
              break
        meta["상세 정보"]["보안 이슈"] = detailed_dict["보안 이슈"]
        if meta['녹음 방식'] == "필드 레코딩":
          meta["상세 정보"]["녹음 장소"] = detailed_dict["녹음 장소"]
          meta["상세 정보"]["녹음 지점"] = detailed_dict["녹음 지점"]
          if "기후 등 환경 조건" in detailed_dict:
            meta["상세 정보"]["기후 등 환경 조건"] = detailed_dict["기후 등 환경 조건"]
          else:
            meta["상세 정보"]["기후 등 환경 조건"] = detailed_dict["기후/환경 조건"]
          meta["상세 정보"]["기타 특이사항"] = detailed_dict["기타 특이사항"]

        for key in meta['상세 정보'].keys():
          if isinstance(meta['상세 정보'][key], float) and isnan(meta['상세 정보'][key]):
            meta['상세 정보'][key] = '없음'

      except Exception as e:
        self.add_error(wav_path, f"Error occured while handling the corresponding xlsx: {e}")

    elif docx_path.exists():
      try:
        detailed_meta = self.read_table_from_docx(docx_path)
        if meta['녹음 방식'] == "필드 레코딩":
          meta["필드 레코딩 메타"] = detailed_meta
        if '제목' in detailed_meta:
          pd.DataFrame(detailed_meta, index=['']).T.to_excel(docx_path.with_suffix('.xlsx'))
          print('excel created', excel_path)
        meta["제목"] = detailed_meta.pop("제목")
      except Exception as e:
        self.add_error(wav_path, f"Error occured while handling the corresponding docx")
    else:
      self.add_error(wav_path, "Meta docx does not exist")

    # self.get_audio_features(ob)
    return meta

  def read_table_from_docx(self, docx_path):
    document = Document(docx_path)
    table = document.tables[0]
    table_dict = {row.cells[0].text:row.cells[1].text for row in table.rows[1:]}

    return table_dict

  def get_class_name_and_title(self, wav_path):
    category_idx = get_categorical_idx_from_fp(wav_path)
    class_name =  self.vocab[category_idx]['소분류']
    try:
      title = self.read_docx(wav_path)['제목']
    except:
      title = ''

    return f"{class_name}: {title}"

  def __call__(self, label, save=False):
    audio_path = Path(label['data']['audio'])
    if 'r-1-1-2-3 허진만' in audio_path.stem:
      audio_path = audio_path.parent / 'r-1-1-2-3-허진만.wav'
    wav_path = audio_path.parent.parent / 'wav' / audio_path.with_suffix('.wav').name
    audio_path_in_container = self.dataset_dir / wav_path.relative_to('/data/local-files/?d=data_v4')
    json_path = audio_path_in_container.parent.parent / 'metadata' / audio_path.with_suffix('.json').name

    meta = self.create_meta_for_wav(audio_path_in_container)

    out = {}
    out['분류 정보'] = {}
    out['분류 정보']['소분류'] = meta['소분류']
    out['분류 정보']['중분류'] = meta['중분류']
    out['분류 정보']['대분류'] = meta['대분류']
    out['분류 정보']['소분류 번호'] = meta['소분류_번호']
    out['분류 정보']['중분류 번호'] = meta['중분류_번호']
    out['분류 정보']['대분류 번호'] = meta['대분류_번호']

    out['파일 정보'] = {}
    out['파일 정보']['파일 이름'] = audio_path.name
    out['파일 정보']['샘플링 레이트'] = meta['샘플링 레이트']
    out['파일 정보']['채널 수'] = meta['채널 수']
    out['파일 정보']['Bit Depth'] = meta['Bit_depth']
    out['파일 정보']['길이'] = meta['길이']

    out['오디오 이벤트 태그'] = [x['value']['labels'][0] for x in label['annotations'][0]['result']]
    out['오디오 음향 특성'] = self.audio_feature_extractor(audio_path_in_container)
    
    out['녹음 상세 정보'] = {}
    out['녹음 상세 정보']['제목'] = meta['제목']
    for key in meta['상세 정보'].keys():
      out['녹음 상세 정보'][key] = meta['상세 정보'][key]

    if save:
      with open(json_path, 'w') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    return out


  
  '''
  def check_mp3_format(self, wav_path):
    mp3_path = wav_path.parent.parent / 'mp3' / wav_path.with_suffix('.mp3').name
    mp3_info = mediainfo(mp3_path)
    if int(mp3_info["sample_rate"]) != 48000:
      print(f"Error in MP3 SR: {mp3_path} is {mp3_info['sample_rate']}")
    if abs(int(mp3_info["bit_rate"]) - 320000) > 300 :
      print(f"Error in MP3 bit_rate: {mp3_path} is {mp3_info['bit_rate']}")
    return float(mp3_info["duration"])
  '''

def get_categorical_idx_from_fp(wav_path):
  path = Path(wav_path)
  category_idx = int(path.stem.split('-')[3]) - 1
  return category_idx


class AudioFeatureExtractor:
  def __init__(self):
    pass 

  def get_audio_features(self, y:np.ndarray, sr:int):
    spec = self.get_spec(y, sr)
    
    spec_centroid = self.get_mean_and_std_of_audio_feature(spec, sr, librosa.feature.spectral_centroid, 'spectral centroid')
    
    spec_centroid.update(self.get_mean_and_std_of_audio_feature(spec, sr, librosa.feature.spectral_bandwidth, 'spectral bandwidth'))
    spec_centroid.update(self.get_mean_and_std_of_audio_feature(spec, sr, librosa.feature.spectral_contrast, 'spectral contrast'))
    spec_centroid.update(self.get_mean_and_std_of_audio_feature(spec, sr, librosa.feature.spectral_rolloff, 'spectral roll off'))
    spec_centroid.update(self.get_mean_and_std_of_audio_feature_wo_sr(y, librosa.feature.spectral_flatness, 'spectral flatness'))
    spec_centroid.update(self.get_mean_and_std_of_audio_feature_wo_sr(y, librosa.feature.zero_crossing_rate, 'zero crossing rate'))
    spec_centroid.update(self.get_mean_and_std_of_audio_feature_wo_sr(y, librosa.feature.rms, 'rms'))

    return spec_centroid
  
  def get_spec(self, y, sr):
    return np.abs(librosa.stft(y))
  
  def get_mean_and_std_of_audio_feature(self, y, sr, audio_feature, audio_feature_name:str):
    audio_features = audio_feature(S = y, sr = sr)
    return {f'{audio_feature_name} mean': np.mean(audio_features) , f'{audio_feature_name} std':np.std(audio_features) }
  
  def get_mean_and_std_of_audio_feature_wo_sr(self, y, audio_feature, audio_feature_name:str):
    audio_features = audio_feature(y = y)
    return {f'{audio_feature_name} mean': np.mean(audio_features) , f'{audio_feature_name} std':np.std(audio_features) }

  def read_soundfile(self, soundfile:sf.SoundFile):
    y = soundfile.read()
    y = y.mean(1)
    return y, soundfile.samplerate

  def __call__(self, soundfile:sf.SoundFile):
    if isinstance(soundfile, str) or isinstance(soundfile, Path):
      soundfile =  sf.SoundFile(soundfile)
    y, sr = self.read_soundfile(soundfile)
    return self.get_audio_features(y, sr)
