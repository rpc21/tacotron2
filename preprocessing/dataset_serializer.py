import os
import pickle
import argparse

PATH_TO_DATA = '/scratch/speech/datasets/LibriTTS/train-clean-100/'


def serialize_dataset(args):
    dirs = [r for r, d, f in os.walk(args.data) if len(r.split('/')) == 8]
    files = set([])
    for dir in dirs:
        files.update([os.path.join(dir, f) for f in os.listdir(dir) if
                      os.path.isfile(os.path.join(dir, f)) and f.split('.')[-1] == 'wav'])
    audio_files = [x for x in files]
    original_text = []
    normalized_text = []
    for filename in audio_files:
        with open(filename[:-3] + 'original.txt', 'r') as f:
            original_text.append(f.read())
        with open(filename[:-3] + 'normalized.txt', 'r') as f:
            normalized_text.append(f.read())
    data = {'audio': audio_files, 'original_text': original_text, 'normalized_text': normalized_text}
    with open(args.pkl, 'wb+') as f:
        pickle.dump(data, f)


def process_for_tacotron(args):
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    with open(args.txt, 'w+') as f:
        for file, text in zip(data['audio'], data['normalized_text']):
            f.write('{}|{}\n'.format(file, text))


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_data', '-data', dest='data')
    parser.add_argument('-path_to_pickle', '-pkl', dest='pkl')
    parser.add_argument('-path_to_text', '-txt', dest='txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    serialize_dataset(args)
    process_for_tacotron(args)
