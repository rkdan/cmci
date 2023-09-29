import torch
import torchaudio

from functools import lru_cache

import numpy as np
import pandas as pd

import os
from scipy.io import wavfile as wav
from tqdm import trange


SR = 44100
N_MELS = 40
FEATURIZER = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=N_MELS).to('cpu')

class Files:
    """Class to house file paths for labelled data.
    """
    data_loc = 'data/Calls for ML/'

    # create symlinks so that all the data can be seen from labelled_data
    lb_data_loc = 'data/Calls for ML/labelled_data/'

    state_dict = 'data/Calls for ML/simple_rnn_sd.pth'

    ml_test = 'ML_Test.wav'
    labels_file = 'Calls_ML.xlsx'


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.audio = {
            f.replace('.wav', ''): self.load_audio(Files.lb_data_loc + f).to(device) for f in os.listdir(Files.lb_data_loc) if '.wav' in f
        }
        self.audio_lens = {k: (len(a), len(a)/SR) for k, a in self.audio.items()}

        calls = pd.read_excel(os.path.join(Files.lb_data_loc, Files.labels_file))
        calls = self.preprocess_call_labels(calls, keep_only_conures=False)
        calls = calls.loc[
            (calls.call_type != 'interference'),
            ['file', 'call_type', 'start', 'end']
        ]

        calls_shaldon = pd.read_excel(os.path.join(Files.lb_data_loc, 'Shaldon_Training_Labels.xlsx'))
        calls_shaldon = calls_shaldon.loc[~calls_shaldon.Call_Type.isna(), ['File', 'Call_Type', 'Start', 'End']]
        calls_shaldon['File'] = 'Shaldon_Combined'
        calls_shaldon.columns = calls_shaldon.columns.str.lower()

        calls_blackpool = pd.read_excel(os.path.join(Files.lb_data_loc, 'Blackpool_Labels.xlsx'))
        calls_blackpool = calls_blackpool.loc[~calls_blackpool.Call_Type.isna(), ['File', 'Call_Type', 'Start', 'End']]
        calls_blackpool['File'] = 'Blackpool_Combined_FINAL'
        calls_blackpool.columns = calls_blackpool.columns.str.lower()

        self.labels = pd.concat([calls, calls_shaldon, calls_blackpool], axis=0).reset_index(drop=True)

        # l("Computing mfcc.")
        self.featurizer = FEATURIZER

        # l("Preprocessing label time series.")
        self.nps = self.featurizer(torch.zeros(1, SR).to(device)).shape[-1]

        self.features = {k: self.featurizer(a).T for k, a in self.audio.items()}

        self.label_ts = {k: None for k in self.audio.keys()}
        ts = {k: self.audio_lens[k][-1]*torch.arange(f.shape[0]).to(device)/f.shape[0] for k, f in self.features.items()}
        for k in ts.keys():
            temp_df = np.asarray(self.labels.loc[self.labels.file == k, ['start', 'end']])
            self.label_ts[k] = torch.zeros_like(ts[k])
            for start, end in temp_df:
                self.label_ts[k][(ts[k] >= start) & (ts[k] < end)] = 1.0
        self.ts = ts

    def __len__(self):
        return 1
    
    
    def load_audio(self, file_path):
        sr, audio = self.load_audio_file(file_path)
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, SR)
        return audio
    
    
    @staticmethod
    @lru_cache(maxsize=100)
    def load_audio_file(filename):
        sr, audio = wav.read(filename)
        if len(audio.shape) == 2:
            audio = audio[:, 0]  # take the first channel
        audio = audio.astype('f')/1000  # scale values down by 1000.
        return sr, audio
    

    @staticmethod
    def preprocess_call_labels(calls_og, keep_only_conures=True):
        calls = calls_og.copy()
        calls.columns = [c.lower().replace(' ', '_') for c in calls.columns]
        if keep_only_conures:
            calls = calls.loc[~calls.call_type.isna() | (calls.interference == 'Conure')].reset_index(drop=True) #drop conure calls?
        calls.loc[calls.call_type.isna(), 'call_type'] = 'interference' # set all unknown call types to interference
        calls = calls.loc[calls.start < calls.end].reset_index(drop=True)
        calls['call_type'] = calls.call_type.apply(lambda r: r.split(' ')[0])
        return calls


    def get_samples(self, audio_len=2.5):
        assert audio_len < 5
        segm_len = int(self.nps * audio_len)

        features = []
        labels = []
        zoos = []

        # l('Processing data.')
        files_to_process = [f for f in self.features.keys() if f != 'ML_Test_3']
        for file in files_to_process:
            # l(f'Processing {file}')
            lbs, feats = self.label_ts[file], self.features[file]
            for i in trange(max(0, len(feats)//segm_len)):
                start_idx = i*segm_len # np.random.choice(len(feats) - segm_len - 1)
                end_idx = start_idx + segm_len

                _ft = feats[None, start_idx:end_idx, :].clone()
                _lb = lbs[None, start_idx:end_idx].clone()
                if (len(_ft[0]) == len(_lb[0])) and (len(_lb[0]) == segm_len):
                    features.append(_ft)
                    labels.append(_lb)
                    zoos.append(np.array([file]*segm_len, dtype=str)[None, ...])

        return (torch.cat(features, axis=0), torch.cat(labels, axis=0),
                np.concatenate(zoos, axis=0))

    def __getitem__(self, *args):
        return self.get_samples()
    

def get_train_test(train_size=0.9, device='cpu'):
    dataset = AudioDataset(device=device)
    X_full, y_full, z_full = dataset[...]


    idx = np.random.choice(len(y_full), len(y_full), replace=False)
    train_idx, test_idx = idx[:int(0.9*len(idx))], idx[int(0.9*len(idx)):]

    X_train = X_full[train_idx, ...]
    y_train = y_full[train_idx, ...]

    X_test = X_full[test_idx, ...]
    y_test = y_full[test_idx, ...].cpu().numpy().reshape(-1)
    z_test = z_full[test_idx, ...].reshape(-1)

    conv = {'Blackpool_Combined_FINAL': 'blackpool', 'Shaldon_Combined': 'shaldon',
            'ML_Test': 'banham', 'ML_Test_2a': 'banham', 'ML_Test_2b': 'banham'}

    for k, repl in conv.items():
        z_test[z_test == k] = repl

    X_test_2 = dataset.featurizer(dataset.audio['ML_Test_3']).T[None, ...]
    y_test_2 = dataset.label_ts['ML_Test_3'].cpu().numpy()

    return X_train, y_train, X_test, y_test, z_test, X_test_2, y_test_2