import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from RNNextraction import crnn
import tensorflow as tf


class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = './data/train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('./data/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values
                yield seg_id, x, -999

    def features(self, x, y, seg_id):

        outputs = crnn(tensor=x, kernel_size=self.chunk_size, stride=1, out_channels=10, rnn_n_layers=1, rnn_type='lstm', bidirectional=False, padding='SAME'
             , w_std=0, scope_name='feature')

        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id
        for i in range(10):
            feature_dict[str(i)] = outputs[:,:,i]


        # create features here
        # for example:
        # feature_dict['mean'] = np.mean(x)

        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)


training_fg = FeatureGenerator(dtype='train', n_jobs=4, chunk_size=150000)
training_data = training_fg.generate()


test_fg = FeatureGenerator(dtype='test', n_jobs=4, chunk_size=None)
test_data = test_fg.generate()

training_data.to_csv("./data/train_features.csv", index=False)
test_data.to_csv("./data/test_features.csv", index=False)