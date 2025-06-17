import numpy as np
import pickle
from tqdm import tqdm
import data_creation.prepare_data.utils as utils
import miditoolkit
import csv, math
import xlrd
from utils import *

from music21 import converter
import music21


Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8
}

Genre = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}

Texture = {'pad': 0, 'mel': 1, 'rhythm': 2, 'harm': 3,
                'mel+rhythm': 4, 'mel+harm': 5, 
                'rhythm+harm': 6, 'mel+rhythm+harm': 7, 'rhythn': 2, 'melody': 1}

Chord_root = {'pad': 0, 
                'C': 1, 'B#': 13, 'D--': 25,
                'C#': 2, 'D-': 14, 'B##': 26,
                'D': 3, 'E--': 15, 'C##': 27,
                'D#': 4, 'E-': 16, 'F--': 28,
                'E': 5, 'F-': 17,'D##': 29,
                'F': 6, 'E#': 18, 'G--': 30,
                'F#': 7, 'G-': 19, 'E##': 31,
                'G': 8, 'A--': 20, 'F##': 32,
                'G#': 9, 'A-': 21,
                'A': 10, 'B--': 22, 'G##': 33,
                'A#': 11, 'B-': 23, 'C--': 34,
                'B': 12, 'C-': 24, 'A##': 35,}

Localkey = {'pad': 0, 'b': 1, 'A-': 2, 'f': 3, 'c': 4, 'D-': 5, 'd': 6, 'F': 7, 
            'A': 8, 'E': 9, 'C': 10, 'B-': 11, 'a': 12, 'c#': 13, 'D': 14, 'G-': 15, 
            'G': 16, 'e-': 17, 'E-': 18, 'g': 19, 'b-': 20, 'F#': 21, 'd#': 22, 'B': 23, 
            'f#': 24, 'C-': 25, 'e': 26, 'a-': 27, 'g#': 28, 'd-': 29, 'D#': 30, 'C#': 31, 
            'a#': 32, 'b#': 33, 'e#': 34, 'g-': 35, 'G#': 36, 'F-': 37}


class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # print (dict, self.event2word)
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]
        # print (self.pad_word)

    def quantize_data(self, notes):
        max_note_offset = max([notes[i][1] for i in range(len(notes))])
        dbeats = [i for i in range(0, int(round(max_note_offset + 4.0)), 4)]
        cur_downbeat_timing = 0
        groups = []
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            insiders = []
            if cur_downbeat_timing == 0:
                insiders.append(Item(
                    name='Tempo',
                    start=0,
                    end=None,
                    velocity=None,
                    pitch=120,
                    Type=-1))

            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][1] - db1) / (db2 - db1)
                    # print (notes[i][3])
                    insiders.append(Item(
                        name='Note',
                        start=cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        end=cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        velocity=64, 
                        pitch=max(min(notes[i][2], 107), 22),
                        Type=0))

            insiders.sort(key=lambda x: (x.start, x.pitch))
            overall = [cur_downbeat_timing] + insiders + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
            cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
            groups.append(overall)

        events = item2event(groups, task='na')
        return events

    def quantize_and_tokenize(self, data_npy_path, gt_npy_path, max_len):
        # Now, always assumes 4/4 (already rescaled)
        data = np.load(data_npy_path, allow_pickle=True)
        groundtruth = np.load(gt_npy_path, allow_pickle=True)

        all_words, all_ys = [], []
        for k in tqdm(range(len(data))):
            events = self.quantize_data(data[k])
            # print (len(events), len(groundtruth[k]))
            # print (events[:10], data[k][:10])
            words, ys = [], []
            for j in range(len(events)):
                nts, to_class = [], -1
                for e in events[j]:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)
                ys.append(groundtruth[k][j])

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)
        return all_words, all_ys

    def extract_events(self, input_path, task='pretrain'):
        if task == 'pretrain':
            note_items, tempo_items = utils.read_items(input_path, sanity_check=True)
        else:
            note_items, tempo_items = utils.read_items(input_path, sanity_check=False)

        if note_items is None:
            return None

        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        # print (items[:20])
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data

    def prepare_pretrain_data(self, midi_paths):
        all_words = []
        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words = []
            for note_tuple in events:
                nts, to_class = [], -1
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)

            slice_words = [words,]
            all_words = all_words + slice_words
        return all_words

    def prepare_finetune_seq_data(self, midi_paths, task, max_len=512):
        # max_len is always 512
        all_words, all_ys = [], []

        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words = []
            for note_tuple in events:
                nts, to_class = [], -1
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])

                if task == "composer":
                    name = path.split('/')[-2]
                    slice_ys.append(Composer[name])
                elif task == "emotion":
                    name = path.split('/')[-1].split('_')[0]
                    slice_ys.append(Emotion[name])
                elif task == "genre":
                    name = path.split('/')[-2]
                    slice_ys.append(Genre[name])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys


    def prepare_finetune_pop909_data(self, midi_paths, task, max_len):
        all_words, all_ys = [], []

        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                ys.append(to_class+1)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys

    def extract_events_from_csv(self, csv_path, dbeat_path, task):
        # Get notes
        notes = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    onset = float(row[0])
                    duration = float(row[3])

                    # 1: non-motif note; 2: motif note
                    if row[-1] == '':
                        note = [onset, int(row[1]), duration, 1]
                    else:
                        note = [onset, int(row[1]), duration, 2]

                    if duration > 0:
                        notes.append(note)

        if dbeat_path[-4:] == 'xlsx':
            workbook = xlrd.open_workbook(dbeat_path)
            sheet = workbook.sheet_by_index(0)
            dbeats = [0.0]
            for rowx in range(sheet.nrows):
                cols = sheet.row_values(rowx)
                dbeats.append(cols[0])
        else:
            dbeats = [0.0]
            with open(dbeat_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    dbeats.append(float(row[0]))

        dbeats[0] = dbeats[1] - (dbeats[2] - dbeats[1])
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))

        groups = []
        cur_downbeat_timing = 0
        # print (dbeats)
        note_count = 0
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            insiders = []
            if cur_downbeat_timing == 0:
                insiders.append(Item(
                    name='Tempo',
                    start=0,
                    end=None,
                    velocity=None,
                    pitch=120,
                    Type=-1))

            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                    # print (notes[i][3])
                    insiders.append(Item(
                        name='Note',
                        start=cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        end=cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        velocity=64, 
                        pitch=notes[i][1],
                        Type=notes[i][3]))

                    note_count = note_count + 1
            overall = [cur_downbeat_timing] + insiders + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
            cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
            groups.append(overall)

        events = item2event(groups, task=task)
        return events


    def prepare_finetune_bpsmotif_data(self, csv_paths, dbeat_paths, task, max_len):
        all_words, all_ys = [], []

        for j in tqdm(range(len(csv_paths))):
            csv_path = csv_paths[j]
            dbeat_path = dbeat_paths[j]
            # extract events
            events = self.extract_events_from_csv(csv_path, dbeat_path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                # print (note_tuple)
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                ys.append(to_class)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys

    def extract_s3_events(self, csv_path, annotation_path, dbeat_path, task):
        
        # Get texture annotation
        cur_texture_annotation = []
        with open(annotation_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    cur_texture_annotation.append([int(round(float(row[0]))), int(round(float(row[1]))), Texture[row[2]]])

        # Deal with possibly overlapping segments
        for i in range(len(cur_texture_annotation) - 1):
            if cur_texture_annotation[i][1] > cur_texture_annotation[i+1][0]:
                cur_texture_annotation[i][1] = cur_texture_annotation[i+1][0]
                cur_texture_annotation[i+1][0] = cur_texture_annotation[i+1][0]

        # Get notes
        notes = []
        cur_annotation_id = 0

        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    onset = float(row[0])
                    duration = float(row[1]) - float(row[0])
                    # Clip pitch value
                    pitch = max(min(int(row[2]), 107), 22)

                    while (cur_annotation_id < len(cur_texture_annotation) - 1 
                        and onset >= cur_texture_annotation[cur_annotation_id+1][0]):
                        cur_annotation_id += 1

                    if duration > 0:
                        notes.append([onset, pitch, duration
                            , cur_texture_annotation[cur_annotation_id][2]])

        notes = sorted(notes, key=lambda x: (x[0], x[1], x[2]))

        dbeats = [0.0]
        with open(dbeat_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    dbeats.append(float(row[0]))

        dbeats[0] = dbeats[1] - (dbeats[2] - dbeats[1])
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))

        groups = []
        cur_downbeat_timing = 0
        # print (dbeats)
        note_count = 0
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            insiders = []
            if cur_downbeat_timing == 0:
                insiders.append(Item(
                    name='Tempo',
                    start=0,
                    end=None,
                    velocity=None,
                    pitch=120,
                    Type=-1))

            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                    # print (notes[i][3])
                    insiders.append(Item(
                        name='Note',
                        start=cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        end=cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        velocity=64, 
                        pitch=notes[i][1],
                        Type=notes[i][3]))

                    note_count = note_count + 1
            overall = [cur_downbeat_timing] + insiders + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
            cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
            groups.append(overall)

        events = item2event(groups, task=task)
        return events

    def prepare_finetune_s3_data(self, csv_paths, annotation_paths, dbeat_paths, task, max_len):
        all_words, all_ys = [], []

        for j in tqdm(range(len(csv_paths))):
            csv_path = csv_paths[j]
            dbeat_path = dbeat_paths[j]
            annotation_path = annotation_paths[j]
            # extract events
            events = self.extract_s3_events(csv_path, annotation_path, dbeat_path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                # print (note_tuple)
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                ys.append(to_class)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys

    def extract_orch_events(self, midi_path, annotation_path, dbeat_path, task):
        all_events = []
        # Get texture annotation
        all_texture_annotation = np.load(annotation_path)
        # print (all_texture_annotation[0:5,0])

        dbeats = []
        with open(dbeat_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    dbeats.append(float(row[0]))
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))
        # print (len(dbeats))

        midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

        for j in range(all_texture_annotation.shape[1]):
            cur_texture_annotation = []
            for i in range(len(all_texture_annotation)):
                cur_texture_annotation.append([dbeats[i], dbeats[i+1], (all_texture_annotation[i][j][0]*4 
                    + all_texture_annotation[i][j][1]*2 + all_texture_annotation[i][j][2])])
            # print (cur_texture_annotation)

            midi_notes = midi_obj.instruments[j].notes
            midi_notes.sort(key=lambda x: (x.start, x.pitch))

            notes = []
            cur_annotation_id = 0

            for note in midi_notes:
                onset = note.start / midi_obj.ticks_per_beat
                duration = (note.end - note.start) / midi_obj.ticks_per_beat
                # Clip pitch value
                pitch = max(min(note.pitch, 107), 22)

                while (cur_annotation_id < len(cur_texture_annotation) - 1 
                    and onset >= cur_texture_annotation[cur_annotation_id+1][0]):
                    cur_annotation_id += 1

                if duration > 0:
                    notes.append([onset, pitch, duration
                        , cur_texture_annotation[cur_annotation_id][2]])

            notes = sorted(notes, key=lambda x: (x[0], x[1], x[2]))

            groups = []
            cur_downbeat_timing = 0
            # print (dbeats)
            note_count = 0
            for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
                insiders = []
                if cur_downbeat_timing == 0:
                    insiders.append(Item(
                        name='Tempo',
                        start=0,
                        end=None,
                        velocity=None,
                        pitch=120,
                        Type=-1))

                for i in range(len(notes)):
                    if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                        start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                        end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                        # print (notes[i][3])
                        insiders.append(Item(
                            name='Note',
                            start=cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                            end=cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                            velocity=64, 
                            pitch=notes[i][1],
                            Type=notes[i][3]))

                        note_count = note_count + 1
                overall = [cur_downbeat_timing] + insiders + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
                cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
                groups.append(overall)

            events = item2event(groups, task=task)
            all_events.append(events)
        return all_events

    def prepare_finetune_orch_data(self, midi_paths, annotation_paths
                        , dbeat_paths, task, max_len):
        all_words, all_ys = [], []

        for j in tqdm(range(len(midi_paths))):
            midi_path = midi_paths[j]
            annotation_path = annotation_paths[j]
            dbeat_path = dbeat_paths[j]
            # extract events
            all_events = self.extract_orch_events(midi_path, annotation_path, dbeat_path, task)
            if not all_events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            for events in all_events:
                # events to words
                words, ys = [], []
                for note_tuple in events:
                    nts, to_class = [], -1
                    # print (note_tuple)
                    for e in note_tuple:
                        e_text = '{} {}'.format(e.name, e.value)
                        nts.append(self.event2word[e.name][e_text])
                        if e.name == 'Pitch':
                            to_class = e.Type
                    words.append(nts)
                    ys.append(to_class)

                # slice to chunks so that max length = max_len
                slice_words, slice_ys = [], []
                for i in range(0, len(words), max_len):
                    slice_words.append(words[i:i+max_len])
                    slice_ys.append(ys[i:i+max_len])

                if len(slice_words[-1]) < max_len:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                    slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

                all_words = all_words + slice_words
                all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys

    def note2event(self, notes, dbeats, task):
        groups = []
        cur_downbeat_timing = 0
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            insiders = []
            if cur_downbeat_timing == 0:
                insiders.append(Item(
                    name='Tempo',
                    start=0,
                    end=None,
                    velocity=None,
                    pitch=120,
                    Type=-1))

            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)

                    insiders.append(Item(
                        name='Note',
                        start=cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        end=cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        velocity=64, 
                        pitch=notes[i][1],
                        Type=notes[i][3]))
            overall = [cur_downbeat_timing] + insiders + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
            cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
            groups.append(overall)
        # print (groups[1:4])
        # print (note_count)
        events = item2event(groups, task=task)
        return events

    def extract_augnet_events(self, mxl_path, annotation_path, task):
        # Get downbeat and anontation
        dbeats = [0, ]
        root = []
        with open(annotation_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            previous_gt_measure = 0
            for row in reader:
                if row[0] != 'j_offset':
                    if task == 'chordroot':
                        root.append([float(row[0]), float(row[0]) + 0.125, Chord_root[str(row[16])]])
                    elif task == 'localkey':
                        root.append([float(row[0]), float(row[0]) + 0.125, Localkey[str(row[20])]])
                    
                    if float(row[2]) > previous_gt_measure:
                        # Add a barline
                        dbeats.append(float(row[0]))
                        previous_gt_measure = float(row[2])

        if len(dbeats) > 2:
            dbeats[0] = dbeats[1] - (dbeats[2] - dbeats[1])
        else:
            # print (annotation_path)
            dbeats[0] = dbeats[1] - 4.0
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))

        # Parse mxl file
        note_file = converter.parse(mxl_path)

        note_list = []
        for note in note_file.flat.notes:
            # print (note)
            if isinstance(note, music21.note.Note):
                # onset, pitch, duration
                note_list.append([note.offset, note.pitch.midi, note.duration.quarterLength])
            elif isinstance(note, music21.chord.Chord):
                for j in range(len(note.pitches)):
                    note_list.append([note.offset, note.pitches[j].midi, note.duration.quarterLength])

        notes = []
        cur_annotation_id = 0
        for note in note_list:
            # onset, pitch, duration
            onset = note[0]
            duration = note[2]
            # Clip pitch value
            pitch = max(min(int(note[1]), 107), 22)

            while (cur_annotation_id < len(root) - 1 
                and onset >= root[cur_annotation_id+1][0]):
                cur_annotation_id += 1

            if duration > 0:
                notes.append([onset, pitch, duration, root[cur_annotation_id][2]])

        notes = sorted(notes, key=lambda x: (x[0], x[1], x[2]))
        events = self.note2event(notes, dbeats, task)
        return events

    
    def prepare_finetune_augnet_data(self, mxl_paths, annotation_paths, task, max_len):
        all_words, all_ys = [], []
        for j in tqdm(range(len(mxl_paths))):
            mxl_path = mxl_paths[j]
            annotation_path = annotation_paths[j]
            # extract events
            events = self.extract_augnet_events(mxl_path, annotation_path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue

            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                # print (note_tuple)
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                ys.append(to_class)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys

    def extract_pm2s_data(self, note_sequence, annotations, task, max_len):
        # 1 is always negative; 2 is always positive
        resolution = 0.01
        tolerance = 0.07
        beats = annotations['beats']
        downbeats = annotations['downbeats']

        # time to beat/downbeat/inter-beat-interval dictionaries
        end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
        time2beat = np.zeros(int(np.ceil(end_time / resolution)))
        time2downbeat = np.zeros(int(np.ceil(end_time / resolution)))
        time2ibi = np.zeros(int(np.ceil(end_time / resolution)))
        for idx, beat in enumerate(beats):
            l = np.round((beat - tolerance) / resolution).astype(int)
            r = np.round((beat + tolerance) / resolution).astype(int)
            time2beat[l:r+1] = 1.0

            ibi = beats[idx+1] - beats[idx] if idx+1 < len(beats) else beats[-1] - beats[-2]
            l = np.round((beat - tolerance) / resolution).astype(int) if idx > 0 else 0
            r = np.round((beat + ibi) / resolution).astype(int) if idx+1 < len(beats) else len(time2ibi)
            if ibi > 4:
                # reset ibi to 0 if it's too long, index 0 will be ignored during training
                ibi = np.array(0)
            time2ibi[l:r+1] = np.round(ibi / resolution)
        
        for downbeat in downbeats:
            l = np.round((downbeat - tolerance) / resolution).astype(int)
            r = np.round((downbeat + tolerance) / resolution).astype(int)
            time2downbeat[l:r+1] = 1.0
        
        # get beat probabilities at note onsets
        beat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        downbeat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        ibis = np.zeros(len(note_sequence), dtype=np.float32)
        for i in range(len(note_sequence)):
            onset = note_sequence[i][1]
            beat_probs[i] = time2beat[np.round(onset / resolution).astype(int)]
            downbeat_probs[i] = time2downbeat[np.round(onset / resolution).astype(int)]
            ibis[i] = time2ibi[np.round(onset / resolution).astype(int)]

        # print (note_sequence.shape, beat_probs.shape)
        if task == 'beat':
            note_sequence = np.concatenate((note_sequence, np.expand_dims(beat_probs, axis=1)), axis=1)
        elif task == 'downbeat':
            note_sequence = np.concatenate((note_sequence, np.expand_dims(downbeat_probs, axis=1)), axis=1)

        # print (note_sequence.shape)
        note_items, tempo_items = utils.read_items_from_note_seq(note_sequence)

        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)

        if not events:  # if midi contains nothing
            print ('?????', note_sequence)

        return events

    def prepare_finetune_pm2s_data(self, file_paths, task, max_len):
        all_words, all_ys = [], []
        for file_path in tqdm(file_paths):
            note_sequence, annotations = pickle.load(open(file_path, 'rb'))
            events = self.extract_pm2s_data(note_sequence, annotations, task, max_len)

            if not events:  # if midi contains nothing
                print(f'skip {file_path} because it is empty')
                continue

            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                # print (note_tuple)
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                ys.append(to_class)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys

    def extract_tnua_data(self, file_path, task):
        # Get notes
        notes = []

        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'pitch':
                    localbeat_onset = float(row[1])
                    duration = float(row[2])
                    pitch = int(row[0])
                    string = int(row[4])
                    position = int(row[5])
                    finger = int(row[6]) + 1
                    # String: 1-4
                    # position: 1-12
                    # finger: 0-4
                    if task == 'violin_position':
                        note = [localbeat_onset, pitch, duration, position]
                    elif task == 'violin_string':
                        note = [localbeat_onset, pitch, duration, string]
                    elif task == 'violin_all':
                        annotation = max((string - 1) * 60 + (max(position, 1) - 1) * 5 + finger + 1, 0)
                        note = [localbeat_onset, pitch, duration, annotation]
                    notes.append(note)

        # global_timesig = max([notes[i][0] for i in range(len(notes))])
        # # Possible timesig: 1.5, 2, 3, 4, ......
        # if global_timesig < 1.5 and global_timesig > 1.0:
        #     global_timesig = 1.5
        # else:
        #     global_timesig = math.ceil(global_timesig)
        global_timesig = 4

        dbeats = [-global_timesig, 0.0]
        for i in range(len(notes) - 1):
            if notes[i][0] > notes[i+1][0]:
                dbeats.append(dbeats[-1] + global_timesig)
        dbeats.append(dbeats[-1] + global_timesig)

        # Add bar position back
        current_bar_id = 0
        global_notes = []
        for i in range(len(notes)):
            if i >= 1 and notes[i-1][0] > notes[i][0]:
                current_bar_id += 1
            global_notes.append([notes[i][0] + current_bar_id * global_timesig, notes[i][1], notes[i][2], notes[i][3]])
        notes = global_notes

        groups = []
        cur_downbeat_timing = 0
        # print (dbeats)
        note_count = 0
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            insiders = []
            if cur_downbeat_timing == 0:
                insiders.append(Item(
                    name='Tempo',
                    start=0,
                    end=None,
                    velocity=None,
                    pitch=120,
                    Type=-1))

            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                    # print (notes[i][3])
                    insiders.append(Item(
                        name='Note',
                        start=cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        end=cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        velocity=64, 
                        pitch=notes[i][1],
                        Type=notes[i][3]))

                    note_count = note_count + 1
            overall = [cur_downbeat_timing] + insiders + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
            cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
            groups.append(overall)

        events = item2event(groups, task=task)
        return events

    def prepare_finetune_tnua_data(self, file_paths, task, max_len):
        all_words, all_ys = [], []
        for file_path in tqdm(file_paths):
            events = self.extract_tnua_data(file_path, task)

            if not events:  # if midi contains nothing
                print(f'skip {file_path} because it is empty')
                continue

            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                # print (note_tuple)
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                ys.append(to_class)

            # slice to chunks so that max length = 512
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                slice_ys.append(ys[i:i+max_len])

            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys