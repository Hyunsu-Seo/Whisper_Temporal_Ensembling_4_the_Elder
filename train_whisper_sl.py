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
import evaluate
import jiwer
from typing import Any, Dict, List, Union
from transformers import AdamW, WhisperModel, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

import whisper

class CSVPreProcessor:
    def __init__(self, diag_path, free_path, seed):
        self.seed = seed
        np.random.seed(self.seed)

        self.diag_df = pd.read_csv(diag_path)
        self.free_df = pd.read_csv(free_path)

        # self.diag_df = self.diag_df.drop(['age', 'region', 'gender', 'dialect'], axis=1)
        # self.free_df = self.free_df.drop(['age', 'region', 'gender'], axis=1)

        self.diag_df.dropna(axis=0, inplace = True)
        self.free_df.dropna(axis=0, inplace = True)

        self.diag_df['text'] = self.diag_df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))
        self.free_df['text'] = self.free_df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))
        # float 에러 나서
        self.diag_df = self.diag_df[self.diag_df['text'].apply(lambda x: isinstance(x, str))]
        self.free_df = self.free_df[self.free_df['text'].apply(lambda x: isinstance(x, str))]

        # self.diag_df = self.diag_df[self.diag_df['text'].apply(lambda x: len(x)) != 0]
        # self.free_df = self.free_df[self.free_df['text'].apply(lambda x: len(x)) != 0]
        self.diag_df = self.diag_df[self.diag_df['text'].apply(lambda x: len(x.strip()) != 0)]
        self.free_df = self.free_df[self.free_df['text'].apply(lambda x: len(x.strip()) != 0)]
        
        self.diag_df.dropna(axis=0, inplace = True)
        self.free_df.dropna(axis=0, inplace = True)
        
    def collect_data(self, free_df, diag_df, free_size, diag_size):
        np.random.seed(self.seed)

        shuffle_free_df = free_df.sample(frac=1, random_state = self.seed)
        shuffle_diag_df = diag_df.sample(frac = 1, random_state = self.seed)

        train_df = shuffle_free_df.iloc[:free_size].reset_index(drop=True)
        valid_df = shuffle_diag_df.iloc[:diag_size*2].reset_index(drop=True)

        return train_df, valid_df

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
        df = df[df['text'].apply(lambda x: len(x.strip()) != 0)]
        return df

class CustomDataset(Dataset):
    def __init__(self, df, feature_extractor, tokenizer, processor, max_len, seed):
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.df.iloc[idx]['file_path']
        filepath = r'{}'.format(filepath)
        text = self.df.iloc[idx]['text']
        audio, _ = librosa.load(filepath, sr=16000)

        # if self.mode == 'train':
        #     audio = self.mix1(audio, sr = 16000, noise_rate=0.01, rate = 0.5)
        audio = audio.astype(np.float32)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        input_features = torch.tensor(mel, dtype=torch.float32)

        tokenized = self.processor.tokenizer(text, return_tensors='pt', padding='max_length', return_attention_mask=True, max_length=self.max_len)
        labels = tokenized['input_ids'][0]

        decoder_input_ids = [self.tokenizer.bos_token_id] + labels[:-1].tolist()
        decoder_input_ids = torch.tensor(decoder_input_ids)

        return {"input_features": input_features, "labels": labels, "decoder_input_ids": decoder_input_ids}

# cer_metric = evaluate.load("cer")
# wer_metric = evaluate.load("wer")

def compute_metrics(pred, target, tokenizer, cer_metric, wer_metric):
    # target[target == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred, skip_special_tokens=True)
    target_str = tokenizer.batch_decode(target, skip_special_tokens=True)

    # cer = 100 * metric.compute(predictions=pred_str, references=target_str)

    # return cer
    cer = 100 * cer_metric.compute(predictions=pred_str, references=target_str)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=target_str)
    return cer, wer

def train(model, data_loader, loss_fn, optimizer, meta_data, tokenizer, cer_metric, wer_metric):
    model.to(meta_data['device'])
    model.train()
    pred_list = []
    target_list = []
    pbar = tqdm(data_loader)
    train_loss = 0
    cer_score = 0

    for i, batch in enumerate(pbar):
        if batch["input_features"].nelement() == 0:
            continue
        input_features = batch["input_features"].to(meta_data['device'])
        labels = batch["labels"].long().to(meta_data['device'])
        decoder_input_ids = batch["decoder_input_ids"].long().to(meta_data['device'])

        optimizer.zero_grad()

        audio_features = model.encoder(input_features).to(meta_data['device'])
        outputs = model.decoder(decoder_input_ids, audio_features)

        loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(train_loss / (i+1), 4)))

        pred = torch.argmax(outputs, dim=-1)
        pred_list.extend(pred.cpu().numpy().tolist())
        target_list.extend(labels.cpu().numpy().tolist())

    cer_score, wer_score = compute_metrics(pred_list, target_list, tokenizer, cer_metric, wer_metric)
    train_loss = train_loss / len(data_loader)

    t_epoch_message = 'CER: %.4f, WER: %.4f' % (cer_score, wer_score)
    print(t_epoch_message)

    torch.cuda.empty_cache()
    gc.collect()

    return model, cer_score, wer_score, train_loss

def valid(model, data_loader, loss_fn, meta_data, tokenizer, cer_metric, wer_metric):
    model.to(meta_data['device'])
    model.eval()
    pred_list = []
    target_list = []
    pbar = tqdm(data_loader)
    valid_loss = 0
    cer_score = 0

    for i, batch in enumerate(pbar):
        if batch["input_features"].nelement() == 0:
            continue
        input_features = batch["input_features"].to(meta_data['device'])
        labels = batch["labels"].to(meta_data['device'])
        decoder_input_ids = batch["decoder_input_ids"].to(meta_data['device'])

        audio_features = model.encoder(input_features)
        outputs = model.decoder(decoder_input_ids, audio_features)

        loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        valid_loss += loss.item()
        pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(valid_loss / (i+1), 4)))

        pred = torch.argmax(outputs, dim=-1)
        pred_list.extend(pred.cpu().numpy().tolist())
        target_list.extend(labels.cpu().numpy().tolist())

    cer_score, wer_score = compute_metrics(pred_list, target_list, tokenizer, cer_metric, wer_metric)
    valid_loss = valid_loss / len(data_loader)

    v_epoch_message = 'CER: %.4f, WER: %.4f' % (cer_score, wer_score)
    print(v_epoch_message)

    torch.cuda.empty_cache()
    gc.collect()

    return model, cer_score, wer_score, valid_loss

def train_sl(seed, meta_data, train_df, valid_df, all_results):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(meta_data['model_name'])
    processor = WhisperProcessor.from_pretrained(meta_data['model_name'], language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(meta_data['model_name'], language="Korean", task="transcribe")

    train_dataset = CustomDataset(train_df, feature_extractor, tokenizer, processor, meta_data['max_len'], seed = seed)
    val_dataset = CustomDataset(valid_df, feature_extractor, tokenizer, processor, meta_data['max_len'], seed = seed)

    train_loader = DataLoader(train_dataset, batch_size = meta_data['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = meta_data['batch_size'])
    
    whisper_model = whisper.load_model(meta_data['model_name'].split('-')[-1])
    optimizer = AdamW(whisper_model.parameters(), lr=meta_data['lr'])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if seed !=0 and meta_data['start_epoch'] != 0:
        print(os.path.isfile(meta_data['path'] + meta_data['save_model_path'].format(seed)))
        checkpoint = torch.load(meta_data['path'] + meta_data['save_model_path'].format(seed))
        whisper_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # elif seed !=0 and meta_data['start_epoch'] ==0:
    #     checkpoint = torch.load(meta_data['path'] + meta_data['save_model_path'].format(seed-1))
    #     whisper_model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    whisper_model.to(meta_data['device'])
    if meta_data['start_epoch'] != 0:
        res = all_results[str(seed)]['results']
    else:
        res = []
    for epoch in range(meta_data['epochs']):
        if epoch < meta_data['start_epoch']:
            continue
        print('\nEpoch: {}'.format(epoch+1))
        print('---------------------')

        whisper_model, train_cer_score, train_wer_score, train_loss = train(whisper_model, train_loader, loss_fn, optimizer, meta_data, tokenizer, cer_metric, wer_metric)
        whisper_model, valid_cer_score, valid_wer_score, valid_loss = valid(whisper_model, val_loader, loss_fn, meta_data, tokenizer, cer_metric, wer_metric)
        
        res.append([epoch, train_cer_score, train_wer_score, train_loss, valid_cer_score, valid_wer_score, valid_loss])

        all_results[str(seed)]['results'] = res

        with open(meta_data['path'] + meta_data['save_logging_file_name'], 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        torch.save({
            'model_state_dict': whisper_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, meta_data['path'] + meta_data['save_model_path'].format(seed))
            
    torch.cuda.empty_cache()
    gc.collect()

    return all_results