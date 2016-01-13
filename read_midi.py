import midi

def to_int_array(file_location):
    pattern = midi.read_midifile(file_location)

    return [
        event.data[0] for track in pattern for event in track
        if isinstance(event, midi.NoteOnEvent) and event.data[1] > 0
    ]
