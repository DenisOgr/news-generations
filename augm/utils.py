import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm_notebook
import os
import numpy as np
from datetime import datetime
pd.set_option('display.max_colwidth', -1)
from nltk.tokenize import sent_tokenize
import random
import os
from pymystem3 import Mystem
import gensim.downloader as api
import glob
from dt import RDT

def load_mystem_and_w2v():
    print("Start loading mydtem and word2vec")
    model_mystem = Mystem()
    model_w2w_ruscorpora_300 = api.load("word2vec-ruscorpora-300")
    print("Finish loading mydtem and word2vec")
    return model_mystem, model_w2w_ruscorpora_300

def load_td():
    print("Start loading RDT")
    model = RDT(dt_pkl_fpath="./rdt.pkl.3")
    print("Finish loading RDT")
    return model


def get_datasets():
    print("Start loading datasets")
    dtypes_br = {
        'match_id': 'int64',
        'team1': 'object',
        'team2': 'object',
        'name': 'object',
        'match_time': 'int64',
        'type': 'category',
        'minute': 'int64',
        'content': 'object',
        'message_time': 'int64'
    }
    broadcasts = pd.read_csv('../data/ods_broadcasts_201905301157.csv',
                             header=0,
                             usecols=dtypes_br.keys(),
                             skipinitialspace=True,
                             skip_blank_lines=True,
                             encoding='utf-8')
    broadcasts.content.apply(str)
    broadcasts = broadcasts.dropna(subset=['content'])
    dtypes_ns = {
        'id': 'int64',
        'name': 'object',
        'ctime': 'int64',
        'body': 'object',
        'match_id': 'int64',
    }

    news = pd.read_csv('../data/ods_match_news.csv',
                       header=0,
                       usecols=dtypes_ns.keys(),
                       skipinitialspace=True,
                       skip_blank_lines=True,
                       encoding='utf-8')
    news = news.dropna(subset=['body'])
    news = news[news.match_id != 787015]
    print("Finish loading datasets")
    return broadcasts, news


def time_type_news(one_news, one_broadcasts):
    assert type(one_news) == pd.core.series.Series, "one_news should be Series"
    assert type(one_broadcasts) == pd.core.series.Series, "one_news should be Series"
    before = 'before'
    after = 'after'
    time_match = datetime.fromtimestamp(one_broadcasts.match_time)
    time_news  = datetime.strptime(one_news.ctime, "%Y-%m-%d %H:%M:%S")
    #print("time match: ", time_match)
    #print("time news: ", time_news)
    return before if time_news < time_match else after

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def _add_points(s):
    return ". ".join(map(lambda x: x.strip(), s.split("."))).strip()
def _r(s, tag):
    s_a = sent_tokenize(s)
    return " ".join(filter(lambda x: x.lower().find(tag.lower()) == -1, s_a))


def clean_news_text(text):
    return  re.sub("(&#(?:\d)*;)", "", re.sub("<.*?>", "", text)).replace("\n","").replace("\r","").replace("\x96","")


def get_candidate_news(news, cur_one_broadcast, min_limit, max_limit):
    match_id = cur_one_broadcast['match_id']
    raw_news_scope = news[news.match_id == match_id]
    news_scope = []
    for i in range(len(raw_news_scope)):
        cur_news = raw_news_scope.iloc[i]
        ttn = time_type_news(cur_news, cur_one_broadcast)
        if ttn == 'after':
            cur_news = _add_points(clean_news_text(cur_news.body))
            cur_news = _r(_r(_r(cur_news, "онлайн-трансляц"), "здесь."), "таблица")
            news_scope.append(cur_news)

    # split news
    news_scope_result = []
    for idx in range(len(news_scope)):
        news_line_tokens = news_scope[idx].split()
        news_line_tokens = split_seq(news_line_tokens, min_limit, max_limit)
        if not news_line_tokens:
            continue
        news_scope_result.append(" ".join(news_line_tokens))

    return news_scope_result


def split_seq(s, min_l, max_l):
    if len(s) < min_l:
        return None
    if len(s) <= max_l:
        return s
    if len(s) > max_l:
        return s[0:max_l]
# assert "".join(split_seq(list("abgdthfsdsss"), 0, 2)) == "ab"
# assert "".join(split_seq(list("abgdthfsdsss"), 0, 10000)) == "abgdthfsdsss"
# assert split_seq(list("12"), 10000, 10000) == None


def shuffle(br_file, news_file):
    print("Start shuffling")
    assert os.path.exists(br_file)
    assert os.path.exists(news_file)
    d = os.path.dirname(br_file)

    tmp_br_file = os.path.join(d, ".br_tmp")
    tmp_news_file = os.path.join(d, ".news_tmp")

    br_lines = open(br_file, encoding='utf-8').readlines()
    news_lines = open(news_file, encoding='utf-8').readlines()

    tmp_br_lines = open(tmp_br_file, "w", encoding='utf-8')
    tmp_news_lines = open(tmp_news_file, "w", encoding='utf-8')

    assert len(br_lines) == len(news_lines)
    idxs = list(range(len(br_lines)))
    random.shuffle(idxs)
    for idx in idxs:
        tmp_br_lines.write(br_lines[idx])
        tmp_news_lines.write(news_lines[idx])

    tmp_br_lines.close()
    tmp_news_lines.close()
    # remove old
    os.remove(br_file)
    os.remove(news_file)

    # rename old
    os.rename(tmp_br_file, br_file)
    os.rename(tmp_news_file, news_file)



def tag(word, model):
    try:
        assert type(word) == str
        if len(word) < 2:
            #print("Warning in: ", word, 'len: ', len(word))
            return None
        processed = None
        processed = model.analyze(word)[0]
        lemma = processed["analysis"][0]["lex"].lower().strip()
        pos = processed["analysis"][0]["gr"].split(',')[0]
        pos = pos.split('=')[0].strip()
        _MAPPINGS = {
                    'A': 'ADJ',
                    'A-PRO': 'PRON',
                    'APRO': 'PRON',
                    'ADV': 'ADV',
                    'ADV-PRO': 'PRON',
                    'ADVPRO': 'PRON',
                    'ANUM': 'ADJ',
                    'CONJ': 'CONJ',
                    'INTJ': 'X',
                    'NONLEX': '.',
                    'NUM': 'NUM',
                    'PARENTH': 'PRT',
                    'PART': 'PRT',
                    'PR': 'ADP',
                    'PRAEDIC': 'PRT',
                    'PRAEDIC-PRO': 'PRON',
                    'S': 'NOUN',
                    'S-PRO': 'PRON',
                    'SPRO': 'PRON',
                    'V': 'VERB',
                }
        tagged = lemma+'_'+_MAPPINGS[pos]
    except:
        #print("Error in: ", word, 'len: ', len(word), 'processed: ', processed)
        return None
    return tagged


def create_dataset(all_matches, broadcasts, news, n, aug_func, to_dir, shard_num=0, aug_func_opt = {}):
    print("Start create_dataset")
    assert type(all_matches) == list
    assert type(broadcasts) == pd.core.frame.DataFrame
    assert type(news) == pd.core.frame.DataFrame
    assert n > 0
    assert callable(aug_func)
    assert type(to_dir) == str


    br_min_limit = 400
    br_max_limit = 2500
    news_min_limit = 0
    news_max_limit = 300
    create_dir(to_dir)
    count_lines = 0
    src_file = "%s/aug_%s_n%s_%s.src.txt" % (to_dir, aug_func.__name__, n, shard_num)
    tgt_file = "%s/aug_%s_n%s_%s.tgt.txt" % (to_dir, aug_func.__name__, n, shard_num)

    # #set custom options for augm functions
    # opt = {}
    # if aug_func.__name__ == 'af_stemb':
    #     opt['model_mystem'], opt['model_w2w_ruscorpora_300'] = load_mystem_and_w2v()
    #

    with(open(src_file, 'w', encoding='utf-8')) as f_broad:
        with(open(tgt_file, 'w', encoding='utf-8')) as f_news:
            for match_id in tqdm_notebook(all_matches):
                print("match_id: ", match_id)

                # getting and splitting broadcast
                origin_br = clean_news_text(" ".join(broadcasts[broadcasts['match_id'] == match_id]['content']).lower())
                origin_br_splited = split_seq(origin_br.split(), br_min_limit, br_max_limit)

                if not origin_br_splited:
                    print("Skip br")
                    continue

                # getting list of news (with splitting)
                one_broadcast = broadcasts[broadcasts['match_id'] == match_id].iloc[0]
                news_scope = get_candidate_news(news, one_broadcast, news_min_limit, news_max_limit)
                if not news_scope:
                    continue
                # print(news_scope)

                # storing origin broadcast and news
                f_broad.write(" ".join(origin_br_splited) + "\n")
                f_news.write(random.choice(news_scope) + "\n")

                # generating broadcasts
                cache = None
                for i in range(n):
                    cand_br, cache = aug_func(origin_br_splited, cache, aug_func_opt)
                    cand_news = random.choice(news_scope)
                    f_broad.write(" ".join(cand_br) + "\n")
                    f_news.write(cand_news + "\n")
                    count_lines += 1

    print("Source file: ", src_file)
    print("Target file: ", tgt_file)
    print("Count lines: ", count_lines)
    return src_file, tgt_file


def get_num_of_words(s, p):
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return 0
    while True:
        r = np.random.geometric(p=p, size=1)[0]
        if r < len(s):
            return r - 1


# Thesaurus
def aug_func_thesaurus(s, cache):
    assert type(s) == list
    idxs = list(range(len(s)))
    random.shuffle(idxs)
    for id in idxs[0:get_num_of_words(s)]:
        pass


# Static embedding
# try:
#     # this is cache, baby
#     print(type(model_mystem))
#     print(type(model_w2w_ruscorpora_300))
# except:
#     model_mystem = Mystem()
#     model_w2w_ruscorpora_300 = api.load("word2vec-ruscorpora-300")
#

def af_stemb(s, cache, opt={}):
    assert type(s) == list
    model_mystem = opt['model_mystem']
    model_w2w_ruscorpora_300 = opt['model_w2w_ruscorpora_300']

    no_similar = 0
    changed = 0
    r = -1
    if not cache:
        print("creating cache")
        # Getting similar words for all
        cache = {}
        for idx, word in enumerate(s):
            if len(word) < 5:
                continue
            word = tag(word, model_mystem)
            try:
                sim_words_raw = model_w2w_ruscorpora_300.similar_by_word(word)
            except:
                # print('no similar: ', word)
                no_similar += 1
                continue
            tag_word = word.split('_')[-1]
            # Getting words only with same part of speech
            sim_words = []
            for can_word in sim_words_raw:
                can_word = can_word[0]
                can_word, can_tag = can_word.split('_')[0], can_word.split('_')[1]
                if can_tag == tag_word:
                    sim_words.append(can_word)

            if not sim_words:
                continue

            cache[idx] = sim_words

    # Getting random words for cache
    if cache:
        cache_keys = list(cache.keys())
        random.shuffle(cache_keys)
        r = get_num_of_words(cache.keys(), 0.004)

        for idx in cache_keys[0:r]:
            # Getting random words from cache for idx position
            try:
                i = get_num_of_words(cache[idx], 0.4)
                word = cache[idx][i]
                # print('was: ', s[idx], 'new: ', word)
                s[idx] = word
                changed += 1
            except IndexError as e:
                print(e, 'index: ', i, sim_words)

    print('no similar: ', no_similar, 'changed: ', changed, 'r: ', r)
    return s, cache


def af_th(s, cache, opt={}):
    assert type(s) == list
    dt = opt['dt']

    no_similar = 0
    changed = 0
    r = -1
    if not cache:
        print("creating cache")
        # Getting similar words for all
        cache = {}
        for idx, word in enumerate(s):
            if len(word) < 5:
                continue
            sim_words = dt.most_similar(word.lower(), top_n=10)
            if not sim_words:
                print("no words for: %s" % word)
                continue

            cache[idx] = [word[0] for word in sim_words]

    # Getting random words for cache
    if cache:
        cache_keys = list(cache.keys())
        random.shuffle(cache_keys)
        r = get_num_of_words(cache.keys(), 0.004)

        for idx in cache_keys[0:r]:
            # Getting random words from cache for idx position
            try:
                i = get_num_of_words(cache[idx], 0.4)
                word = cache[idx][i]
                print('was: ', s[idx], 'new: ', word)
                s[idx] = word
                changed += 1
            except IndexError as e:
                print(e, 'index: ', idx, cache)

    print('no similar: ', no_similar, 'changed: ', changed, 'r: ', r)
    return s, cache

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def merge(args):
    print("Start merging.")
    src_file = "%s/aug_%s_n%s_all.src.txt" % (args.to_dir, args.aug_func.__name__, args.n)
    tgt_file = "%s/aug_%s_n%s_all.tgt.txt" % (args.to_dir, args.aug_func.__name__, args.n)
    is_file = False
    if os.path.isfile(src_file):
        print("File exist: %s" % src_file)
        is_file=True
    if os.path.isfile(tgt_file):
        print("File exist: %s" % tgt_file)
        is_file=True
    if is_file:
        print("Remove exists files")
        return

    src_files = glob.glob("%s/aug_%s_n%s_*.src.txt" % (args.to_dir, args.aug_func.__name__, args.n))
    tgt_files = glob.glob("%s/aug_%s_n%s_*.tgt.txt" % (args.to_dir, args.aug_func.__name__, args.n))
    assert len(src_files) == len(tgt_files)
    print("Found %s files" % len(src_files))
    all_files = [(open(file, encoding="utf-8"), open(file.replace(".src.", ".tgt."), encoding="utf-8")) for file in src_files]
    idx_all_files = list(range(len(all_files)))
    c = 0
    with(open(src_file, "w", encoding="utf-8")) as s:
        with(open(tgt_file, "w", encoding="utf-8")) as t:
            while idx_all_files:
                idx_file = random.choice(idx_all_files)
                try:
                    src_line = all_files[idx_file][0].readline()
                    tgt_line = all_files[idx_file][1].readline()
                except:
                    print("Error: idx: ", idx_file, "idx_all_files: ", idx_all_files)

                if not src_line or not tgt_line:
                    #del all_files[idx_file]
                    idx_all_files.remove(idx_file)
                    print("remove files.")
                    continue
                s.write(src_line)
                t.write(tgt_line)
                c += 1
                print("Count added files: %s" % c)
    print("Source file: ", src_file)
    print("Target file: ", tgt_file)
    pass


def to_train_set(args):
    print("Start transformation.")
    src_file = "%s/aug_%s_n%s_all.src.txt" % (args.to_dir, args.aug_func.__name__, args.n)
    tgt_file = "%s/aug_%s_n%s_all.tgt.txt" % (args.to_dir, args.aug_func.__name__, args.n)
    print("Source file: ", src_file)
    print("Target file: ", tgt_file)

    assert os.path.isfile(src_file), "File %s should be exist" % src_file
    assert os.path.isfile(tgt_file), "File %s should be exist" % tgt_file
    assert os.path.isdir(args.to_save), "Invalid path to dir: " % args.to_save
    assert args.to_save[-1] != '/'

    num_lines = sum(1 for line in open(src_file, encoding='utf-8'))
    X_train, X_test = train_test_split(list(range(num_lines)), test_size=0.01, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=0.11, random_state=42)
    print("Len all: ", num_lines)
    print("Len X_train: ", len(X_train))
    print("Len X_val: ", len(X_val))
    print("Len X_test: ", len(X_test))
    idx = 0
    files = [
        (X_train, open('%s/train_src.txt' % args.to_save, "w", encoding='utf-8'), open('%s/train_tgt.txt' % args.to_save, "w", encoding='utf-8')),
        (X_val, open('%s/valid_src.txt' % args.to_save, "w", encoding='utf-8'), open('%s/valid_tgt.txt' % args.to_save, "w", encoding='utf-8')),
        (X_test, open('%s/test_src.txt' % args.to_save, "w", encoding='utf-8'), open('%s/test_tgt.txt' % args.to_save, "w", encoding='utf-8'))]

    with(open(src_file, encoding='utf-8')) as s:
        with(open(tgt_file, encoding='utf-8')) as t:
            while True:
                sl = s.readline()
                tl = t.readline()
                if not sl or not tl:
                    break
                for file in files:
                    if idx in file[0]:
                        file[1].write(sl)
                        file[2].write(tl)
                idx += 1
                print(idx)






