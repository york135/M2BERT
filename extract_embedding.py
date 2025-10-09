import os
import pickle
import torch
import numpy as np
import argparse
from data_creation.prepare_data.model import CP
from M2BERT.model import MidiModernBERT
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser(description='Extract features from MIDI using MidiModernBERT')
    parser.add_argument('--dict_path', type=str, required=True, help='Path to event2word/word2event dictionary (.pkl)')
    parser.add_argument('--midi_path', type=str, required=True, help='Path to MIDI file')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to pretrained MidiModernBERT weights (.pt)')
    parser.add_argument('--output_pickle', type=str, required=True, help='Path to save output features (.pkl)')
    parser.add_argument('--max_len', type=int, default=1024, help='Maximum sequence length for each chunk (default: 1024)')
    args = parser.parse_args()

    # ==== LOAD DICTIONARY ====
    cp = CP(args.dict_path)

    # ==== EXTRACT EVENTS FROM MIDI ====
    events = cp.extract_events_stretch_to_4_4(args.midi_path)
    if not events:
        raise ValueError(f"No events extracted from {args.midi_path}")

    
    # print (events)

    # ==== CONVERT EVENTS TO TOKENS ====
    words = []
    for note_tuple in events:
        nts = []
        for e in note_tuple:
            e_text = '{} {}'.format(e.name, e.value)
            nts.append(cp.event2word[e.name][e_text])
        words.append(nts)

    # ==== SLICE INTO CHUNKS OF max_len AND PAD ====
    max_len = args.max_len
    slice_words = []
    for i in range(0, len(words), max_len):
        chunk = words[i:i+max_len]
        if len(chunk) < max_len:
            # Pad with pad_word
            pad_word = cp.pad_word
            chunk += [pad_word] * (max_len - len(chunk))
        slice_words.append(chunk)
    
    # print (slice_words)

    # ==== LOAD PRETRAINED MidiModernBERT ====
    print("\nBuilding ModernBERT model")
    configuration = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    configuration.vocab_size = 10
    configuration.max_position_embeddings = 1024
    configuration.num_hidden_layers = 12
    configuration.attention_dropout = 0.1
    configuration.mlp_dropout = 0.1
    # configuration.window_size = [256, 256]
    # configuration.position_embedding_type ='relative_key_query'
    configuration.pad_token_id = -1
    model = MidiModernBERT(bertConfig=configuration, e2w=cp.event2word, w2e=cp.word2event)
    model.load_state_dict(torch.load(args.model_ckpt, map_location='cpu')['state_dict'])
    model.eval()

    # ==== EXTRACT FEATURES FOR EACH CHUNK ====
    features_list = []
    with torch.no_grad():
        for idx, chunk in enumerate(slice_words):
            input_ids = np.array([chunk], dtype=np.int64)  # (1, max_len, 4)
            input_ids = torch.tensor(input_ids)
            hidden_states = model(input_ids, output_hidden_states=True)['hidden_states']
            # chunk_hidden_states = []
            features = []
            for layer_idx, h in enumerate(hidden_states):
                # print(f'  Layer {layer_idx}: {h.shape}')
                features.append(h.unsqueeze(1))
            # all_chunks_hidden_states.append(chunk_hidden_states)
            features = torch.cat(features, dim=1)
            print(f'Chunk {idx} feature shape:', features.shape)
            features_list.append(features.cpu().numpy())

    # ==== SAVE TO PICKLE ====
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(features_list, f)

    print(f'Features saved to {args.output_pickle}')


if __name__ == '__main__':
    main()
