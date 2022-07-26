import json
from pathlib import Path
from copy import copy
from docx.api import Document
import soundfile as sf
import librosa

class MetaCreator:
  def __init__(self, vocab_path):
    with open(vocab_path, 'r') as f:
      self.vocab = json.load(f)
    self.error_list = []

  def get_recording_type(self, wav_path):
    if '폴리' in wav_path:
      return '폴리'
    elif '합성' in wav_path:
      return '합성'
    elif '필드' in wav_path:
      return '필드 레코딩'

  def check_name_is_wrong(self, wav_path):
    name = wav_path.stem
    high, middle, fine = [int(x) for x in name.split('-')[:3]]
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
    category_idx = int(wav_path.stem.split('-')[2]) - 1
    
    # try:
    #   mp3_dur = self.check_mp3_format(wav_path)
    # except:
    #   print(f"Error occured while reading mp3 of {wav_path}")


    if self.check_name_is_wrong(wav_path):
      # print(f"Name is wrong for {wav_path}")
      self.error_list.append(wav_path, "Category is wrong")
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
    if True: #if it is field recording
      docx_path = wav_path.parent.parent / 'metadata' / wav_path.with_suffix('.docx').name
      if docx_path.exists():
        try:
          detailed_meta = self.read_table_from_docx(docx_path)
          if meta['녹음 방식'] == "필드 레코딩":
            meta["필드 레코딩 메타"] = detailed_meta
          meta["제목"] = detailed_meta["제목"]
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


  def get_audio_features(self, soundfile:sf.SoundFile):
    y = soundfile.read()
    y = y.mean(1)
    spec_centroid = librosa.feature.spectral_centroid(y = y, sr=soundfile.samplerate)
    print(spec_centroid)

    return

  def get_class_name_and_title(self, wav_path):
    category_idx = int(wav_path.stem.split('-')[2]) - 1
    class_name =  self.vocab[category_idx]['소분류']
    try:
      title = self.read_docx(wav_path)['제목']
    except:
      title = ''

    return f"{class_name}: {title}"
  
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