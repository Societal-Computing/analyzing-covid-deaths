import numpy as np
import pandas as pd
import preprocessor as p
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

# Options for tweets preprocessing
# only used for the MinhashLSH deduplication step
# the model prediction uses the default options
# to keep the information about user mentions and hashtags
options = [
    p.OPT.URL,
    p.OPT.HASHTAG,
    p.OPT.EMOJI,
    p.OPT.SMILEY,
    p.OPT.NUMBER,
    p.OPT.RESERVED,
    p.OPT.MENTION,
    p.OPT.ESCAPE_CHAR,
]


def _shingle(string, shingle_size=4):
    shings = {
        string[i : i + shingle_size] for i in range(len(string) - shingle_size + 1)
    }
    return set(shings)


def find_duplicates(df, similarity=0.8, num_perms=256, shingle_size=4):
    SIMILARITY_THRESHOLD = similarity
    NUM_PERMS = num_perms
    SHINGLE_SIZE = shingle_size

    lsh = MinHashLSH(
        threshold=SIMILARITY_THRESHOLD,
        num_perm=NUM_PERMS,
    )

    texts = df["full_text"].apply(p.clean).apply(str.lower).apply(str.strip)

    for id_, title in tqdm(texts.items()):
        title_shingles = _shingle(title, shingle_size=SHINGLE_SIZE)

        title_minhash = MinHash(num_perm=NUM_PERMS)

        for shing in title_shingles:
            title_minhash.update(shing.encode("utf8"))

        lsh.insert(id_, title_minhash, check_duplication=False)

    dup_dict = {}

    for id_, title in tqdm(texts.items()):
        title_shingles = _shingle(title, shingle_size=SHINGLE_SIZE)

        title_minhash = MinHash(num_perm=NUM_PERMS)

        for shing in title_shingles:
            title_minhash.update(shing.encode("utf8"))

        dups = lsh.query(title_minhash)
        dup_dict[id_] = dups

    return dup_dict


def remove_duplicates(df, dup_dict):
    possible_dups = set()
    already_seen = set()

    for idx in df.index.tolist():
        already_seen.add(idx)
        dup_indices = dup_dict[idx]

        for i in dup_indices:
            if i == idx:
                continue
            else:
                if i in already_seen:
                    continue
                else:
                    possible_dups.add(i)

    df = df.drop(possible_dups)
    return df
