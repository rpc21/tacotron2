import os
import numpy as np
import pandas as pd
import wave

PATH_TO_DATA = '/scratch/speech/IEMOCAP_full_release/'


def print_list(list_to_print):
	for item in list_to_print:
		print(item)


def get_text(session, dialog, utterance):
	path = PATH_TO_DATA + 'Session{}/dialog/transcriptions/{}'.format(session, dialog[:-3] + '.txt')
	with open(path, 'r') as f:
		for line in f:
			descriptor, text = line.strip().split(': ')
			if descriptor.split()[0] == dialog:
				return text
	return ''


def process_text_files():
	audio_paths = []
	categorical_emotion = []
	corresponding_text = []
	average_valence = []
	average_activation = []
	average_dominance = []
	for x in range(5):
		session_path = PATH_TO_DATA + 'Session' + str(x + 1)
		label_path = session_path + '/dialog/EmoEvaluation'
		audio_path = session_path + '/sentences/wav/'
		files = [label_path + '/' + f for f in os.listdir(label_path) if
				 os.path.isfile(os.path.join(label_path, f)) and f[0] != '.']
		# print_list(files)
		for file in files:
			lines = []
			get_next_line = False
			with open(file) as f:
				for line in f:
					if get_next_line and len(line.strip('\n').split('\t')) > 3:
						lines.append(line.strip().split('\t'))
					elif line.strip() == '':
						get_next_line = True
			for utterance in lines:
				audio_paths.append(audio_path + utterance[1][:14] + '/' + utterance[1] + '.wav')
				categorical_emotion.append(utterance[2])
				print(utterance)
				scores = utterance[3][1:-1].split(', ')
				average_valence.append(float(scores[0]))
				average_activation.append(float(scores[1]))
				average_dominance.append(float(scores[2]))
	for audio_path in audio_paths:
		_, scratch, release, session, sentence, wav, dialog, utterance = audio_path.split('/')
		corresponding_text.append(get_text(session, dialog, utterance))
	return pd.DataFrame(
		list(zip(audio_paths, corresponding_text, categorical_emotion, average_valence, average_activation, average_dominance)),
		columns=['file', 'text', 'emotion', 'valence', 'activation', 'dominance'])


if __name__ == '__main__':
	df = process_text_files()
	df.to_csv('/scratch/rpc21/Speech-Emotion-Analysis/src/preprocessing/audio_paths_labels_TTS.csv',index=False)
	# print('wrote to csv')
