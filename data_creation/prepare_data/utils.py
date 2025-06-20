import numpy as np
import miditoolkit
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])     # np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, Type):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, Type={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.Type)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, sanity_check=False):
    try:
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    except:
        return None, None
        
    if sanity_check:
        if (len(midi_obj.time_signature_changes) != 1 or midi_obj.time_signature_changes[0].numerator != 4
            or midi_obj.time_signature_changes[0].denominator != 4):
            return None, None

    # note
    note_items = []
    num_of_instr = len(midi_obj.instruments) 
    
    for i in range(num_of_instr):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            if (note.end - note.start) / midi_obj.ticks_per_beat >= (1.0 / 16.0) - 0.0001:
                note_items.append(Item(
                    name='Note',
                    start=int(note.start * DEFAULT_RESOLUTION / midi_obj.ticks_per_beat), 
                    end=int(note.end * DEFAULT_RESOLUTION / midi_obj.ticks_per_beat), 
                    velocity=note.velocity, 
                    pitch=max(min(note.pitch, 107), 22),
                    Type=i))
                
    note_items.sort(key=lambda x: x.start)
    
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=int(tempo.time * DEFAULT_RESOLUTION / midi_obj.ticks_per_beat),
            end=None,
            velocity=None,
            pitch=int(tempo.tempo),
            Type=-1))
    tempo_items.sort(key=lambda x: x.start)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                Type=-1))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                Type=-1))
    tempo_items = output
    return note_items, tempo_items

import statistics

def read_items_from_note_seq(note_sequence):
    estimated_sec_per_beat = statistics.median(note_sequence[:,2])
    # 40~200 (1.5sec ~ 0.3sec)
    while estimated_sec_per_beat < 0.3:
        estimated_sec_per_beat = estimated_sec_per_beat * 2.0

    while estimated_sec_per_beat > 1.5:
        estimated_sec_per_beat = estimated_sec_per_beat / 2.0

    # Convert second to pseudo-bar position
    note_sequence[:,1] = note_sequence[:,1] / estimated_sec_per_beat
    note_sequence[:,2] = note_sequence[:,2] / estimated_sec_per_beat

    # note
    note_items = []
    # pitch, start, duration, velocity
    note_sequence = list(note_sequence)
    note_sequence.sort(key=lambda x: (x[1], x[0]))

    # print (note_sequence)
    
    for note in note_sequence:
        note_items.append(Item(
            name='Note',
            start=note[1] * DEFAULT_RESOLUTION, 
            end=(note[1] + note[2]) * DEFAULT_RESOLUTION, 
            velocity=int(note[3]), 
            pitch=max(min(int(round(note[0])), 107), 24),
            Type=int(round(note[4])) + 1))
                
    note_items.sort(key=lambda x: x.start)
    
    # tempo
    tempo_items = [Item(
            name='Tempo',
            start=0,
            end=None,
            velocity=None,
            pitch=int(60.0 / estimated_sec_per_beat),
            Type=-1),]

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                Type=-1))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                Type=-1))
    tempo_items = output
    return note_items, tempo_items

class Event(object):
    def __init__(self, name, time, value, text, Type):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
        self.Type = Type

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={}, Type={})'.format(
            self.name, self.time, self.value, self.text, self.Type)


def item2event(groups, task):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True
        
        for item in groups[i][1:-1]:
            if item.name != 'Note':
                continue
            note_tuple = []

            # Bar
            if new_bar:
                BarValue = 'New' 
                new_bar = False
            else:
                BarValue = "Continue"
            note_tuple.append(Event(
                name='Bar',
                time=None, 
                value=BarValue,
                text='{}'.format(n_downbeat),
                Type=-1))

            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            note_tuple.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start),
                Type=-1))
            
            # Pitch
            velocity_index = np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1

            if (task == 'melody' or task == 'mnid' or task == 'texture' 
                or task == 'chordroot' or task == 'localkey' or task == 'beat' or task == 'downbeat'
                or task == 'violin_string' or task == 'violin_position' or task == 'violin_all'):
                pitchType = item.Type
            elif task == 'velocity':
                pitchType = velocity_index
            else:
                pitchType = -1
                
            note_tuple.append(Event(
                name='Pitch',
                time=item.start, 
                value=item.pitch,
                text='{}'.format(item.pitch),
                Type=pitchType))

            # Duration
            duration = item.end - item.start
            index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
            note_tuple.append(Event(
                name='Duration',
                time=item.start,
                value=index,
                text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index]),
                Type=-1))

            events.append(note_tuple)

    return events


def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups