import midi

def from_array(notes):
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    for note in notes:
        on = midi.NoteOnEvent(tick=0, velocity=127, pitch=note)
        track.append(on)

        off = midi.NoteOffEvent(tick=100, pitch=note)
        track.append(off)

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile('output.mid', pattern)
