import os
import cPickle


def get_file_path():
    return os.path.dirname(os.path.realpath(__file__))


def load_text_data_set(dataset_name):
    data_set = cPickle.load(open(get_file_path() + '/textual/%s' % dataset_name, 'rb'))
    return data_set


def load_bimodal_data_set(dataset_name):
    data = cPickle.load(open(get_file_path() + '/bimodal/%s.p' % dataset_name, 'rb'))
    return data


def save_bimodal_data_set(bimodal_auto_encoder, data_set_name, embeddings_dict):
    output = {'embeddings_dict': embeddings_dict,
              'target_words': bimodal_auto_encoder.data_set['target_words'],
              'right_context_words': bimodal_auto_encoder.data_set['right_context_words']}
    path_ = get_file_path() + '/bimodal/' + data_set_name + '.p'
    cPickle.dump(output, open(path_, "wb"))