import read_midi
import generate
import write_midi

training_data = read_midi.to_int_array('music_data/cs1-1pre.mid')
notes = generate.with_training(training_data)
print notes
write_midi.from_array(notes)
