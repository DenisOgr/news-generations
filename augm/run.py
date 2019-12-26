from utils import *
import random
from multiprocessing import Pool
import argparse
from os import path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type=str, default='create',  choices=['create', 'merge'])
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-to_dir",  type=str, default='./results/', required=False)
    parser.add_argument("-aug_func",  type=str, required=True, choices=['af_stemb', 'af_th', 'af_bert'])
    parser.add_argument("-shard_size",  type=int, required=True)

    args = parser.parse_args()
    assert args.n > 0, "n should more than 0"
    assert args.shard_size > 0, "shard_size should more than 0"
    assert path.isdir(args.to_dir), "invalid path to store dir: %s" % args.to_dir
    assert args.aug_func in locals(), "Invalid aug_func: %s" % args.aug_func
    args.aug_func = locals()[args.aug_func]
    assert callable(args.aug_func)

    if args.task == 'create':
        broadcasts, news = get_datasets()

        all_matches = list(news.match_id.unique())
        random.shuffle(all_matches)

        #options for augm functions
        opt = {}

        if args.aug_func.__name__ == 'af_stemb':
            opt['model_mystem'], opt['model_w2w_ruscorpora_300'] = load_mystem_and_w2v()
        if args.aug_func.__name__ == 'af_th':
            opt['dt'] = load_td()

        args_to_func = [(chunk, args.n, args.aug_func, args.to_dir, idx, opt) for idx, chunk in
                        enumerate(chunks(all_matches, args.shard_size))]

        def wrapper_create_and_shuffle(all_matches, n, af_stemb, to_dir, shard_num, opt):
            print("Start process with shard: %s" % str(shard_num))

            broadcasts, news = get_datasets()
            src_file, tgt_file = create_dataset(all_matches, broadcasts, news, n, af_stemb, to_dir, shard_num, opt)
            shuffle(src_file, tgt_file)
        with Pool() as pool:
            pool.starmap(wrapper_create_and_shuffle, args_to_func)

    if args.task == 'merge':
        merge(args)


