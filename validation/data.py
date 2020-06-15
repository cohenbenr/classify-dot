import pandas as pd
from validation.title_matching import layered_matcher, title_matcher, punct_lookup, exact_matcher
from validation.dot_data import get_dictionary, LemmaTokenizer
from embed_software.preprocess import Preprocessor, readme_processor
from diskcache import FanoutCache
from itertools import islice
from embed_software.preprocess import claims_processor, readme_processor, Preprocessor
import json
from toolz import dissoc

from gcsfs import GCSFileSystem


def get_indeed_texts(path, use_gcs = False, **kwargs):
    """Reads csv with indeed data that turns into test set"""
    if use_gcs:
        fs = GCSFileSystem(project='labor-market-data')
        path = path.replace('..','lmd-classify-dot',1)
        with fs.open(path) as f:
            indeed = pd.read_csv(f, **kwargs)
    else:
        indeed = pd.read_csv(path, **kwargs)

    indeed['title'] = indeed.title.str.lower()
    return indeed

def get_soc_n(socs, N):
    return socs.str.split('-').map(lambda l: ''.join(l)[:N]).astype(int)

def make_desc_lookup(path, N):
    desc_lookup = {
        2: 'desc_soc2',
        3: 'desc_soc3',
        6: 'soc_title6'
    }

    dot_codes = get_dictionary(path, N).groupby('soc').first()
    d = dot_codes[desc_lookup[N]].to_dict()
    def lookup(code):
        try:
            return d[int(code)]
        except KeyError:
            return code
    return lookup

def get_title_lookup(path='crosswalks'):
    """Returns function that matches titles to SOC code"""
    lookup = pd.read_csv(f'{path}/soc-title-lookup.csv')
    xwalk = (pd.read_excel(f'{path}/soc_2000_to_2010_crosswalk.xls',
                          skiprows=6)
             .drop(0))

    xwalk = xwalk.groupby('2010 SOC code').first().reset_index()
    lookup = (lookup.merge(xwalk,
                           how='left',
                           left_on='code',
                           right_on='2010 SOC code')
              [['title', '2000 SOC code']]
              .rename(columns={'2000 SOC code': 'code'})
              .dropna(subset=['code']))
    return lookup

def make_matcher():
    lookup = get_title_lookup()
    cache = FanoutCache('title_cache', shards=24)
    matcher = layered_matcher([
        exact_matcher(lookup),
        title_matcher(lookup, punct_lookup(cache, lookup))
    ])
    return matcher

def indeed_test_data(texts, lim, soc_n):
    """Make test data from indeed (pre-embedded)"""
    indeed = get_indeed_texts(texts, use_gcs=True, nrows=lim)
    matcher = make_matcher()
    matches = matcher(indeed.reset_index()).set_index('index')
    return matches.content, get_soc_n(matches.code, soc_n), matches.index

def dot_train_data(soc_n, include_tasks=True, include_dot=True):
    """Combine DOT Dictionary and Tasks descriptions for training set"""
    dot_dict = get_dictionary('', soc_n)
    tasks = pd.read_csv('tasks.txt', sep='\t')
    processor = Preprocessor(readme_processor, 1, 1, 6).process

    if include_tasks and include_dot:
        X_train = pd.concat([dot_dict.job_description, tasks.Task]).map(processor)
        y_train = pd.concat([dot_dict.soc, get_soc_n(tasks['O*NET-SOC Code'], soc_n)])
    elif include_dot:
        X_train = dot_dict.job_description.map(processor)
        y_train = dot_dict.soc
    elif include_tasks:
        X_train = tasks.Task.map(processor)
        y_train = get_soc_n(tasks['O*NET-SOC Code'], soc_n)

    return X_train, y_train


def virginia_data(path):
    with open(path) as f:
        dat = (json.loads(l) for l in f)
        dat = (dissoc(d, '@context', '@type', 'jobLocation', 'baseSalary', '_id') for d in dat)
        dat = list(dat)
    return dat

def virginia_test_data(path, N):
    string_processor = Preprocessor(readme_processor, 4).process
    dat = virginia_data(path)
    df = pd.DataFrame(dat)
    descriptions = df.description.map(string_processor)
    codes = df.onet_soc_code
    idx = (codes != '') & (descriptions.notna())
    df = df[idx]
    codes = get_soc_n(codes[idx], N)
    return descriptions[idx], codes, df
