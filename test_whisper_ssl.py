import numpy as np
import pandas as pd
import os, sys, re, math, random, time, json, pickle, gc
from tqdm import tqdm
from collections import defaultdict
import itertools as it
import traceback
import requests

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import librosa
import jiwer
import evaluate
from typing import Any, Dict, List, Union
from transformers import AdamW, WhisperModel, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

import whisper


class CSVPreProcessor:
    def __init__(self, test_path):
        self.test_df = pd.read_csv(test_path)
        self.test_df.dropna(axis=0, inplace=True)
        self.test_df['text'] = self.test_df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))
        self.test_df = self.test_df[self.test_df['text'].apply(lambda x: isinstance(x, str))]
        self.test_df = self.test_df[self.test_df['text'].apply(lambda x: len(x)) != 0]

    def collect_data(self, test_df):
        test_df = test_df
        return test_df
        
    def prepro_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        p1 = r'\([^)]*\)'
        p2 = r'\r\n'
        text = re.sub(p1, '', text)
        text = re.sub(p2, '', text)
        return text.strip()

    def apply_text_preprocessing(self, df):
        df['text'] = df['text'].apply(self.prepro_text)
        df['text'] = df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))
        df = df[df['text'].apply(lambda x: isinstance(x, str))]
        df = df[df['text'].apply(lambda x: len(x)) != 0]
        df.dropna(axis=0, inplace=True)
        return df

class TestDataset(Dataset):
    def __init__(self, df, processor, tokenizer, max_len):
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        audio = self.df.iloc[idx]['file_path']
        audio = r'{}'.format(audio)
        text = self.df.iloc[idx]['text']
        labels = self.tokenizer(text,
                                return_tensors='pt',
                                truncation=True,
                                max_length=self.max_len,
                                padding='max_length',
                                add_special_tokens=True)
        labels = labels['input_ids'][0]
        decoder_input_ids = torch.cat([torch.tensor([self.tokenizer.eos_token_id]), labels[:-1]]) # shape: (batch_size, target_sequence_length)

        return audio, labels, decoder_input_ids

def sample_test(test_df, meta_data, processor, tokenizer, shuffle_train = False):
    test_dataset = TestDataset(test_df, processor, tokenizer, meta_data['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=meta_data['batch_size'], shuffle=False)
    return test_loader


class Noise(nn.Module):
    def __init__(self, whisper, processor, feature_extractor, seed, noise_type=None, sampling_rate=16000,
                 inference=False):
        super(Noise, self).__init__()
        self.whisper = whisper
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.noise_type = noise_type

    def minus_sound(self, data, sr=16000):
        mn_data = (-1) * data
        return mn_data

    def pitch_shift(self, data, sr=16000, n_steps=-1):
        np.random.seed(self.seed)
        shift_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
        return shift_data

    def white_noise(self, data, sr=16000, noise_rate=0.005):
        np.random.seed(self.seed)
        wn = np.random.randn(len(data))
        data_wn = data + noise_rate * wn
        return data_wn

    def stretch_sound(self, data, sr=16000, rate=0.1):
        np.random.seed(self.seed)
        stretch_data = librosa.effects.time_stretch(data, rate=rate)
        return stretch_data

    def dropout(self, data, dropout_rate=0.1):
        np.random.seed(self.seed)
        drop_mask = np.random.binomial(1, 1 - dropout_rate, len(data))
        data_dropout = data * drop_mask
        return data_dropout

    def amplitude_modulation(self, data, rate=0.1):
        np.random.seed(self.seed)
        modulation = 1 + rate * np.sin(2 * np.pi * 0.5 * np.arange(len(data)) / self.sampling_rate)
        modulated_data = data * modulation
        return modulated_data

    def mix1(self, data, sr=16000, noise_rate=0.005, rate=0.1):
        np.random.seed(self.seed)
        wn = np.random.randn(len(data))
        data_wn = data + noise_rate * wn
        stretch_data = librosa.effects.time_stretch(data_wn, rate=rate)
        return stretch_data

    def mix2(self, data, sr=16000, noise_rate=0.005, rate=0.2):
        np.random.seed(self.seed)
        mn_data = (-1) * data
        wn = np.random.randn(len(mn_data))
        data_wn = mn_data + noise_rate * wn
        stretch_data = librosa.effects.time_stretch(data_wn, rate=rate)
        return stretch_data

    def mix3(self, data, sr=16000, n_steps=-1, noise_rate=0.005, rate=0.2):
        np.random.seed(self.seed)
        mn_data = (-1) * data
        shift_data = librosa.effects.pitch_shift(mn_data, sr=sr, n_steps=n_steps)
        wn = np.random.randn(len(shift_data))
        data_wn = mn_data + noise_rate * wn
        stretch_data = librosa.effects.time_stretch(data_wn, rate=rate)
        return stretch_data

    def mix4(self, data, sr=16000, n_steps=-2, noise_rate=0.05, dropout_rate=0.2):
        np.random.seed(self.seed)
        shift_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
        drop_mask = np.random.binomial(1, 1 - dropout_rate, len(shift_data))
        data_dropout = data * drop_mask
        wn = np.random.randn(len(data_dropout))
        data_wn = data_dropout + noise_rate * wn
        return data_wn

    def mix5(self, data, sr=16000, n_steps=-2, noise_rate=0.05, rate=0.2):
        np.random.seed(self.seed)
        shift_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
        modulation = 1 + rate * np.sin(2 * np.pi * 0.5 * np.arange(len(shift_data)) / self.sampling_rate)
        modulated_data = shift_data * modulation
        wn = np.random.randn(len(modulated_data))
        data_wn = modulated_data + noise_rate * wn
        return data_wn

    def forward(self, data_lst):
        input_list = []
        for data in data_lst:
            audio, _ = librosa.load(data, sr=16000)
            if self.noise_type == 'white':
                audio = self.white_noise(audio, sr=16000, noise_rate=0.01)
            elif self.noise_type == 'stretch':
                audio = self.stretch_sound(audio, sr=16000, rate=0.5)
            elif self.noise_type == 'mix1':
                audio = self.mix1(audio, sr=16000, noise_rate=0.02, rate=0.2)
            elif self.noise_type == 'mix2':
                audio = self.mix2(audio, sr=16000, noise_rate=0.04, rate=0.2)
            elif self.noise_type == 'mix3':
                audio = self.mix3(audio, sr=16000, n_steps=-2, noise_rate=0.04, rate=0.2)
            elif self.noise_type == 'mix4':
                audio = self.mix4(audio, sr=16000, n_steps=-2, noise_rate=0.04, dropout_rate=0.1)
            elif self.noise_type == 'mix5':
                audio = self.mix5(audio, sr=16000, n_steps=-2, noise_rate=0.05, rate=0.2)
            else:
                audio = audio

            audio = audio.astype(np.float32)
            audio = whisper.pad_or_trim(audio.flatten())
            mel = whisper.log_mel_spectrogram(audio)

            input_features = torch.tensor(mel, dtype=torch.float32)
            input_list.append(input_features)
        batch = torch.stack(input_list)

        return batch

def compute_metrics(pred, target, tokenizer, cer_metric, wer_metric):
    target[target == -100] = tokenizer.pad_token_id

    pred_str = [str(p).strip() for p in pred]
    target_str = [str(t).strip() for t in target]

    # return cer
    cer = 100 * cer_metric.compute(predictions=pred, references=target_str)
    wer = 100 * wer_metric.compute(predictions=pred, references=target_str)
    return cer, wer

class CustomWhisper(nn.Module):
    def __init__(self, model, processor, feature_extractor, noise_type, meta_data, seed, sampling_rate = 16000):
        super(CustomWhisper, self).__init__()
        self.whisper_model = model
        self.meta_data = meta_data
        self.noise = Noise(model, processor, feature_extractor, seed = seed, noise_type = noise_type, sampling_rate = 16000)

    def forward(self, audio_path, inference=False):
        if inference:
            self.noise.noise_type = None
        input_features = self.noise(audio_path).to(self.meta_data['device'])
        return input_features

def test(model, whisper_model, test_df, test_loader, optimizer, tokenizer, cer_metric, wer_metric, meta_data):
    model.to(meta_data['device'])
    model.eval()
    decoded_texts = []
    target_list = test_df['text'].tolist()
    options = whisper.DecodingOptions(language="Korean", without_timestamps=True)

    for batch in tqdm(test_loader):
        audio, labels, decoder_input_ids = batch
        input_features = model(audio, inference=True).to(meta_data['device'])
        labels = labels.to(meta_data['device'])
        results = whisper_model.decode(input_features, options)
        decoded_texts.extend([result.text for result in results])
#         target_list.append(labels.cpu().numpy().tolist())
    cer_score, wer_score = compute_metrics(decoded_texts, target_list, tokenizer, cer_metric, wer_metric)
    test_df['model_transcription'] = decoded_texts

    print(f'CER: {cer_score:.2f}%')
    print(f'WER: {wer_score:.2f}%')

    return cer_score, wer_score, test_df

def test_ssl(seed, meta_data, model_path, test_df, all_results):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(meta_data['model_name'])
    processor = WhisperProcessor.from_pretrained(meta_data['model_name'], language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(meta_data['model_name'], language="Korean", task="transcribe")

    test_loader = sample_test(test_df, meta_data, processor, tokenizer, shuffle_train = False)

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    whisper_model = whisper.load_model(meta_data['model_name'].split('-')[-1])
    whisper_model.to(meta_data['device'])

    model = CustomWhisper(whisper_model, processor, feature_extractor, seed = seed, noise_type = meta_data['noise_type'], meta_data = meta_data, sampling_rate = 16000).to(meta_data['device'])
    optimizer = AdamW(model.parameters(), lr=meta_data['lr'])

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(meta_data['device'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    res = all_results[str(model_path)]['results']
    cer_score, wer_score, test_df = test(model, whisper_model, test_df, test_loader, optimizer, tokenizer, cer_metric, wer_metric, meta_data)
    res.append([cer_score, wer_score, test_df['model_transcription'].tolist()])
    all_results[str(model_path)]['results'] = res

    return all_results