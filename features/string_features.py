import icu, sys
sys.path.append('..')
from functools import partial
from utils.arabic_parser import arabicNameParser


def firstCharFunction(s1,s2,n):
    return int(s1[:n]==s2[:n])

def lastCharFunction(s1,s2,n):
    return int(s1[::-1][:n]==s2[::-1][:n])

def sameWordFunction(s1,s2,idx=0):
    try:
        return int(s1.split(' ')[idx]==s2.split(' ')[idx])
    except IndexError:
        return 0
    
def sameFirstWordsFunction(s1,s2,n=2):
    try:
        return int(s1.split(' ')[:2]==s2.split(' ')[:2])
    except IndexError:
        return 0
    
def sameNWordsFunction(s1,s2,n=3):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)
    counter =0
    for ss in s_short:
        counter += int(ss in s_long)
    return int(counter>=n)
    
    
def charThresholdFunction(s1,s2,threshold=0.5):
    min_l = min([len(s1), len(s2)])
    eq_vec = [int(s1[j]==s2[j]) for j in range(min_l)]
    return int((sum(eq_vec)/float(min_l))>threshold)

def wordThresholdFunction(s1,s2,threshold=0.5):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    min_l = min([len(s1_w), len(s2_w)])
    eq_vec = [int(s1_w[j]==s2_w[j]) for j in range(min_l)]
    return int((sum(eq_vec)/float(min_l))>threshold)

def sortedArabicFunction(s1, s2, collator=icu.Collator.createInstance(icu.Locale('de_DE.UTF-8'))):
    s1_s = ''.join(sorted(list(s1), key=collator.getSortKey))
    s2_s = ''.join(sorted(list(s2), key=collator.getSortKey))
    return int(s1_s==s2_s)

def obtainLongerShorterFunction(s1,s2):
    if len(s1) >= len(s2):
        s_short, s_long = s2, s1
    else:
        s_short, s_long = s1, s2
    return s_short, s_long
    
def samePrefixWordsFunction(s1, s2, n=3):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)
    
    same_suff = []
    for ss in s_short:
        same_suff.append(sum([int(ss[:n]==sl[:n]) for sl in s_long])>=1)
    return int(all(same_suff))

def sameSuffixWordsFunction(s1, s2, n=3):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)
    
    same_suff = []
    for ss in s_short:
        same_suff.append(sum([int(ss[::-1][:n]==sl[::-1][:n]) for sl in s_long])>=1)
    return int(all(same_suff))

def numberSameCharFunction(s1, s2, last=False):
    s_short, s_long = obtainLongerShorterFunction(s1, s2)
    if last:
        s_short, s_long = s_short[::-1], s_long[::-1]

    counter, flag, limit = 0, True, len(s_short)
    while flag and counter < limit:
        if s_short[counter] == s_long[counter]:
            counter +=1
        else:
            flag = False

    return counter


def numberSameWordFunction(s1, s2, last=False):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)
    if last:
        s_short, s_long = s_short[::-1], s_long[::-1]

    counter, flag, limit = 0, True, len(s_short)
    while flag and counter < limit:
        if s_short[counter] == s_long[counter]:
            counter +=1
        else:
            flag = False

    return counter


def alignedCharFunction(s1, s2, perc=True):
    s_short, s_long = obtainLongerShorterFunction(s1, s2)
    al_vec = [int(s_short[j]==s_long[j]) for j in range(len(s_short))]
    if perc:
        return float(sum(al_vec))/len(al_vec)
    else:
        return sum(al_vec)


def alignedWordFunction(s1,s2, perc=True):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)

    al_vec = [int(s_short[j]==s_long[j]) for j in range(len(s_short))]
    if perc:
        return float(sum(al_vec))/len(al_vec)
    else:
        return sum(al_vec)


def effectiveWordDiffFunction(s1, s2):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)

    inc, sshort_notfound = 0, []
    while inc<len(s_short):
        word_sel = s_short[inc]
        try:
            idx_2 = s_long.index(word_sel)
            del s_long[idx_2]
            inc +=1
        except ValueError:
            sshort_notfound.append(word_sel)
            inc +=1
    return sshort_notfound, s_long


def nWordDiffFunction(s1, s2, perc=True):
    sshort_notfound, s_long = effectiveWordDiffFunction(s1,s2)
    len_short = min([len(s1.split(' ')), len(s2.split(' '))])

    if perc:
        return float(len(sshort_notfound))/len_short
    else:
        return len(sshort_notfound)


def nSameSuffixFunction(s1, s2, n=3, perc=True):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)

    same_suff = []
    for ss in s_short:
        same_suff.append(sum([int(ss[::-1][:n]==sl[::-1][:n]) for sl in s_long])>=1)

    if perc:
        return float(sum(same_suff))/len(s_short)
    else:
        return sum(same_suff)


def nSamePrefixFunction(s1, s2, n=3, perc=True):
    s1_w, s2_w = s1.split(' '), s2.split(' ')
    s_short, s_long = obtainLongerShorterFunction(s1_w, s2_w)

    same_pre = []
    for ss in s_short:
        same_pre.append(sum([int(ss[:n]==sl[:n]) for sl in s_long])>=1)

    if perc:
        return float(sum(same_pre))/len(s_short)
    else:
        return sum(same_pre)


def sameArabicNamePartFunction(s1, s2, namepart='ISM', binary=True):
    namepart_vec = ['ISM', 'LAQAB', 'NASAB', 'KUNYA', 'NISBAH', 'LAQAB/NISBAH', 'OTHER']
    np_idx = namepart_vec.index(namepart)
    
    s1_parser = arabicNameParser(s1, print_flag=False)[np_idx][1]
    s2_parser = arabicNameParser(s2, print_flag=False)[np_idx][1]
    perc_al = alignedCharFunction(s1_parser, s2_parser) if s1_parser and s2_parser else 0.0
    
    if binary:
        return int(perc_al==1.0)
    else:
        return perc_al

string_feat_dict={
	'n_chars_first': (numberSameCharFunction, 'name_1', 'name_2'),
    'n_chars_last': (partial(numberSameCharFunction, last=True), 'name_1', 'name_2'),
    'n_words_first': (numberSameWordFunction, 'name_1', 'name_2'),
    'n_words_last': (partial(numberSameWordFunction, last=True), 'name_1', 'name_2'),
    'align_words_perc': (alignedWordFunction, 'name_1', 'name_2'),
    'align_char_perc': (alignedCharFunction, 'name_1', 'name_2'),
    'same_sorted_string': (sortedArabicFunction, 'name_1', 'name_2'),
    'perc_agreeing_words': (nWordDiffFunction, 'name_1', 'name_2'),
    'perc_same_suffixes': (nSameSuffixFunction, 'name_1', 'name_2'),
    'perc_same_prefixes': (nSamePrefixFunction, 'name_1', 'name_2'),
    'perc_ism': (partial(sameArabicNamePartFunction, binary=False, namepart='ISM'), 'name_1', 'name_2'),
    'perc_laqab_nisbah': (partial(sameArabicNamePartFunction, binary=False, namepart='LAQAB/NISBAH'), 'name_1', 'name_2')
}