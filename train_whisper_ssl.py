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
    def __init__(self, diag_path, free_path, seed):
        self.seed = seed
        np.random.seed(self.seed)

        self.diag_df = pd.read_csv(diag_path)
        self.free_df = pd.read_csv(free_path)

        self.diag_df = self.diag_df.drop('dialect', axis=1)
        # self.free_df = self.free_df.drop(['age', 'region', 'gender'], axis=1)

        self.diag_df.dropna(axis=0, inplace = True)
        self.free_df.dropna(axis=0, inplace = True)

        self.diag_df['text'] = self.diag_df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))
        self.free_df['text'] = self.free_df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))

        self.diag_df = self.diag_df[self.diag_df['text'].apply(lambda x: isinstance(x, str))]
        self.free_df = self.free_df[self.free_df['text'].apply(lambda x: isinstance(x, str))] 
              
        self.diag_df = self.diag_df[self.diag_df['text'].apply(lambda x: len(x)) != 0]
        self.free_df = self.free_df[self.free_df['text'].apply(lambda x: len(x)) != 0]

    def collect_data(self, labeled_df, unlabeled_df, label_size, unlabel_size):
        np.random.seed(self.seed)

        shuffle_label_df = labeled_df.sample(frac=1, random_state = self.seed)
        shuffle_unlabel_df = unlabeled_df.sample(frac=1, random_state = self.seed)

        train_label_df = shuffle_label_df.iloc[:label_size]
        # print(train_label_df)
        train_unlabel_df = shuffle_unlabel_df.iloc[:unlabel_size]
        train_unlabel_df.loc[:, 'text'] = -1

        train_df = pd.concat([train_label_df, train_unlabel_df], axis=0).reset_index(drop=True)
        # print(train_df)
        valid_df = shuffle_unlabel_df.iloc[unlabel_size:unlabel_size*2].reset_index(drop=True)

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
        df['text'] = df['text'].apply(lambda x: jiwer.RemoveMultipleSpaces()(x))
        df = df[df['text'].apply(lambda x: isinstance(x, str))]   
        df = df[df['text'].apply(lambda x: len(x)) != 0]
        df.dropna(axis=0, inplace = True)
        return df

class CustomDataset(Dataset):
    def __init__(self, dataset, processor, tokenizer, max_len):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        audio = self.dataset.iloc[idx]['file_path']
        audio = r'{}'.format(audio)
        text = self.dataset.iloc[idx]['text']

        if sample["text"] == '-1':
            labels = torch.tensor([self.tokenizer.pad_token_id]*self.max_len)
            decoder_input_ids = torch.tensor([self.tokenizer.pad_token_id]*self.max_len)
            labeled_yn = torch.tensor([0])
        else:
            labels = self.tokenizer(text,
                                    return_tensors='pt',
                                    truncation=True,
                                    max_length=self.max_len,
                                    padding='max_length',
                                    add_special_tokens=True)
            labels = labels['input_ids'][0]
            decoder_input_ids = torch.cat([torch.tensor([self.tokenizer.eos_token_id]), labels[:-1]]) # shape: (batch_size, target_sequence_length)
            labeled_yn = torch.tensor([1])

        return audio, labels, decoder_input_ids, labeled_yn

def sample_train(train_df, valid_df, meta_data, processor, tokenizer, shuffle_train = False):
    train_dataset = CustomDataset(train_df, processor, tokenizer, meta_data['max_len'])
    train_loader = DataLoader(train_dataset, batch_size=meta_data['batch_size'], shuffle=shuffle_train)

    valid_dataset = CustomDataset(valid_df, processor, tokenizer, meta_data['max_len'])
    valid_loader = DataLoader(valid_dataset, batch_size=meta_data['batch_size'], shuffle=False)

    return train_loader, valid_loader

class Noise(nn.Module):
    def __init__(self, whisper, processor, feature_extractor, seed, noise_type = None, sampling_rate = 16000, inference = False):
        super(Noise, self).__init__()
        self.whisper = whisper
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.noise_type = noise_type
    
    def minus_sound(self, data, sr = 16000):
        mn_data = (-1) * data
        return mn_data

    def pitch_shift(self, data, sr = 16000, n_steps= -1):
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
        stretch_data = librosa.effects.time_stretch(data, rate = rate)
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

    def mix1(self, data, sr = 16000, noise_rate=0.005, rate = 0.1):
        np.random.seed(self.seed)
        wn = np.random.randn(len(data))
        data_wn = data + noise_rate * wn
        stretch_data = librosa.effects.time_stretch(data_wn, rate = rate)
        return stretch_data
    
    def mix2(self, data, sr = 16000, noise_rate=0.005, rate = 0.2):
        np.random.seed(self.seed)
        mn_data = (-1) * data
        wn = np.random.randn(len(mn_data))
        data_wn = mn_data + noise_rate * wn
        stretch_data = librosa.effects.time_stretch(data_wn, rate = rate)
        return stretch_data

    def mix3(self, data, sr = 16000, n_steps= -1, noise_rate=0.005, rate = 0.2):
        np.random.seed(self.seed)
        mn_data = (-1) * data
        shift_data = librosa.effects.pitch_shift(mn_data, sr=sr, n_steps=n_steps)
        wn = np.random.randn(len(shift_data))
        data_wn = mn_data + noise_rate * wn
        stretch_data = librosa.effects.time_stretch(data_wn, rate = rate)
        return stretch_data
    
    def mix4(self, data, sr = 16000, n_steps= -2, noise_rate=0.05, dropout_rate = 0.2):
        np.random.seed(self.seed)
        shift_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
        drop_mask = np.random.binomial(1, 1 - dropout_rate, len(shift_data))
        data_dropout = data * drop_mask
        wn = np.random.randn(len(data_dropout))
        data_wn = data_dropout + noise_rate * wn
        return data_wn
    
    def mix5(self, data, sr = 16000, n_steps = -2, noise_rate = 0.05, rate = 0.2):
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
                audio = self.mix1(audio, sr = 16000, noise_rate=0.02, rate = 0.2)
            elif self.noise_type == 'mix2':
                audio = self.mix2(audio, sr = 16000, noise_rate=0.04, rate = 0.2)
            elif self.noise_type == 'mix3':
                audio = self.mix3(audio, sr = 16000, n_steps = -2, noise_rate=0.04, rate = 0.2)
            elif self.noise_type == 'mix4':
                audio = self.mix4(audio, sr = 16000, n_steps = -2, noise_rate=0.04, dropout_rate = 0.1)
            elif self.noise_type == 'mix5':
                audio = self.mix5(audio, sr = 16000, n_steps = -2, noise_rate = 0.05, rate = 0.2)
            else:
                audio = audio

            audio = audio.astype(np.float32)
            audio = whisper.pad_or_trim(audio.flatten())
            mel = whisper.log_mel_spectrogram(audio)

            input_features = torch.tensor(mel, dtype=torch.float32)
            input_list.append(input_features)
        batch = torch.stack(input_list)

        return batch

def temporal_losses(decoder_output, encoder_output, z_output, w, labels, labeled_yn, meta_data):
    "ensemble output과 current output을 통해 supervised, unsupervised loss 및 total loss를 계산함"
    sup_loss, nbsup = masked_crossentropy(decoder_output, labels, labeled_yn, meta_data) # supervised loss
    unsup_loss = mse_loss(decoder_output, z_output, meta_data['device']) # unsupervised loss
    total_loss = sup_loss + w * unsup_loss # combine

    return total_loss, sup_loss, unsup_loss, nbsup

def mse_loss(out1, out2, device):
    "current output, ensemble output 간의 mean difference: unsupervised loss"
    quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1).to(device)) ** 2)

    # return quad_diff
    return quad_diff / out1.data.nelement()

def masked_crossentropy(out, labels, labeled_yn, meta_data):
    cond = labeled_yn.squeeze() == 1 # (4,1) --> (4)
    nnz = cond.nonzero(as_tuple=True)[0] # (4)
    if nnz.numel() > 0:
        audio_outputs = out[nnz].permute(0, 2, 1)
        text_labels = labels[nnz]
        loss = F.cross_entropy(audio_outputs, text_labels) # cross_entropy(input, target)
        nbsup = nnz.sum().item()
        return loss, nbsup
    else:
        loss = torch.tensor([0.], requires_grad=False).to(meta_data['device'])
        return loss, 0

def weight_scheduler(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)

def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(-mult * (float(epoch) / max_epochs) ** 2)

def compute_metrics(pred, target, tokenizer, cer_metric, wer_metric):
    # target[target == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred, skip_special_tokens=True)
    target_str = tokenizer.batch_decode(target, skip_special_tokens=True)

    # cer = 100 * metric.compute(predictions=pred_str, references=target_str)

    # return cer
    cer = 100 * cer_metric.compute(predictions=pred_str, references=target_str)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=target_str)
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

def train(model, train_loader, optimizer, tokenizer, cer_metric, wer_metric, Z, z, outputs, meta_data, max_val, mult, epoch, n_labeled, n_samples):

    model.to(meta_data['device'])
    model.train()

    train_loss = 0
    suplosses = []
    unsuplosses = []
    print('\nEpoch: {}'.format(epoch+1))
    w = weight_scheduler(epoch, meta_data['epochs'], max_val, mult, n_labeled, n_samples)

    w = torch.tensor(w, requires_grad=False).to(meta_data['device'])
    print('---------------------')
    pred_list = []
    target_list = []

    for i, (audio, labels, decoder_input_ids, labeled_yn) in enumerate(tqdm(train_loader)):
        input_features = model(audio, inference=False).to(meta_data['device'])
        labels = labels.to(meta_data['device'])
        decoder_input_ids = decoder_input_ids.to(meta_data['device'])

        optimizer.zero_grad()

        # encoder_ids = model(audio).to(meta_data['device'])
        encoder_ids = model.whisper_model.encoder(input_features).to(meta_data['device'])
        out = model.whisper_model.decoder(decoder_input_ids, encoder_ids)
        zcomp = z[i * meta_data['batch_size']: (i+1) * meta_data['batch_size']]
        zcomp.requires_grad_(False)
        loss, suploss, unsuploss, nbsup = temporal_losses(out, encoder_ids, zcomp, w, labels, labeled_yn, meta_data)

        outputs[i * meta_data['batch_size']: (i+1) * meta_data['batch_size']] = out.clone().detach()
        train_loss += loss.item()
        suplosses.append(nbsup * suploss.item())
        unsuplosses.append(unsuploss.item())

        for j in range(len(labeled_yn)):
            if labeled_yn[j] == 1:
                pred = torch.argmax(out[j], dim = -1)
                pred_list.append(pred.cpu().numpy().tolist())
                target_list.append(labels[j].cpu().numpy().tolist())

        loss.backward()
        optimizer.step()

    outputs[i * meta_data['batch_size']: (i+1) * meta_data['batch_size']] = out.clone().detach() # 다음 에폭
    cer_score, wer_score = compute_metrics(pred_list, target_list, tokenizer, cer_metric, wer_metric)
    train_loss = train_loss / len(train_loader)
    supl_mean = np.mean(suplosses)
    unsupl_mean = np.mean(unsuplosses)

    t_epoch_message = 'CER: %.4f, WER: %.4f, Loss: %.4f, Supervised Loss: %.4f, Unsupervised Loss: %.4f' % (cer_score, wer_score, train_loss, float(supl_mean), float(unsupl_mean))
    print(t_epoch_message)

    Z = meta_data['alpha'] * Z + (1. - meta_data['alpha']) * outputs
    # z = Z * (1. / (1. - meta_data['alpha'] ** (epoch + 1)))
    z = Z

    torch.cuda.empty_cache()
    gc.collect()

    return model, Z, z, outputs, cer_score, wer_score, train_loss

def evaluation(model, loader, meta_data, tokenizer, cer_metric, wer_metric):
    model.eval()
    val_loss = 0
    pred = []
    target = []

    for i, (audio, labels, decoder_input_ids, labeled_yn) in enumerate(tqdm(loader)):
        input_features = model(audio, inference=True).to(meta_data['device'])
        labels = labels.to(meta_data['device'])
        decoder_input_ids = decoder_input_ids.to(meta_data['device'])

        encoder_ids = model.whisper_model.encoder(input_features).to(meta_data['device'])
        out = model.whisper_model.decoder(decoder_input_ids, encoder_ids)
        
        loss = F.cross_entropy(out.permute(0, 2, 1), labels)
        val_loss += loss.item()
        for p, t in zip(out.argmax(dim=2).tolist(), labels.detach().cpu().numpy()) :
            pred.append(p)
            target.append(t)
    cer_score, wer_score = compute_metrics(pred, target, tokenizer, cer_metric, wer_metric)
    val_loss = val_loss / len(loader)
    v_epoch_message = 'CER: %.4f, WER: %.4f, Loss : %.4f' % (cer_score, wer_score, val_loss)
    print(v_epoch_message)

    torch.cuda.empty_cache()
    gc.collect()

    return cer_score, wer_score, val_loss

def train_ssl(seed, max_val, mult, meta_data, train_df, valid_df, all_results):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(meta_data['model_name'])
    processor = WhisperProcessor.from_pretrained(meta_data['model_name'], language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(meta_data['model_name'], language="Korean", task="transcribe")

    train_loader, valid_loader = sample_train(train_df, valid_df, meta_data, processor, tokenizer, shuffle_train = False)

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    whisper_model = whisper.load_model(meta_data['model_name'].split('-')[-1])
    whisper_model.to(meta_data['device'])

    model = CustomWhisper(whisper_model, processor, feature_extractor, seed = seed, noise_type = meta_data['noise_type'], meta_data = meta_data, sampling_rate = 16000).to(meta_data['device'])
    optimizer = optimizer = AdamW(model.parameters(), lr=meta_data['lr'])

    n_labeled = len(train_df[train_df['text'] != -1])
    n_samples = len(train_df)

    Z = torch.zeros(len(train_df), meta_data['max_len'], 51865).float() # intermediate values
    z = torch.zeros(len(train_df), meta_data['max_len'], 51865).float() # temporal outputs
    outputs = torch.zeros(len(train_df), meta_data['max_len'], 51865).float() # current outputs

    if meta_data['re_start'] == True and meta_data['start_epoch'] != 0:
        print(meta_data['path'] + meta_data['save_model_path'].format(seed, max_val, meta_data['labeled_data_size']))
        checkpoint1 = torch.load(meta_data['path'] + meta_data['save_model_path'].format(seed, max_val, meta_data['labeled_data_size']))
        model.load_state_dict(checkpoint1['model_state_dict'])
        optimizer.load_state_dict(checkpoint1['optimizer_state_dict'])

        # print(meta_data['path'] + 'outputs_ver{}_max_val_{}_label_{}.pt'.format(seed, max_val, meta_data['labeled_data_size']))
        # checkpoint3 = torch.load(meta_data['path'] + 'outputs_ver{}_max_val_{}_label_{}.pt'.format(seed, max_val, meta_data['labeled_data_size']))
        # outputs = checkpoint3['outputs']
    #     val_res = all_results[str(seed)]['results']

    # 기존 데이터 유지 및 업데이트
    val_res = all_results[str(seed)]['results'][str(max_val)]
    
    for epoch in range(meta_data['epochs']):
        if epoch < meta_data['start_epoch']:
            continue

        model, Z, z, outputs, train_cer_score, train_wer_score, train_loss = train(model, train_loader, optimizer, tokenizer, cer_metric, wer_metric, Z, z, outputs, meta_data, max_val, mult, epoch, n_labeled, n_samples)
        val_cer_score, val_wer_score, val_loss, = evaluation(model, valid_loader, meta_data, tokenizer, cer_metric, wer_metric)
        val_res.append([epoch, train_cer_score, train_wer_score, train_loss, val_cer_score, val_wer_score, val_loss])

        all_results[str(seed)]['results'][str(max_val)] = val_res

        with open(meta_data['path'] + meta_data['save_logging_file_name'], 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
    
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, meta_data['path']+ meta_data['save_model_path'].format(seed, max_val, meta_data['labeled_data_size']))

        # torch.save({'outputs': outputs}, meta_data['path'] + 'outputs_ver{}_max_val_{}_label_{}.pt'.format(seed, max_val, meta_data['labeled_data_size']))
        
        torch.cuda.empty_cache()
        gc.collect()

    return all_results
