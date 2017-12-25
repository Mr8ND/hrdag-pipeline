from ...utils.arabic_soundex_server import strip_accents
import nltk.stem.isri as stemmer


def removeAlChars(string):
    al_list = [u'آل', u'ال', u'من']
    return string.replace(al_list[0], '').replace(al_list[1], '').replace(al_list[2], '')

def removeSuffixesFunction(word):
    word_mod = stemmer.ISRIStemmer.suf1(stemmer.ISRIStemmer(), word)
    return stemmer.ISRIStemmer.suf32(stemmer.ISRIStemmer(), word_mod)

def removePrefixesFunction(word):
    word_mod = stemmer.ISRIStemmer.pre1(stemmer.ISRIStemmer(), word)
    return stemmer.ISRIStemmer.pre32(stemmer.ISRIStemmer(), word_mod)

def stemArabicWordFunction(word):
    return stemmer.ISRIStemmer.stem(stemmer.ISRIStemmer(), word)

def fullPrefSufRemFunction(string):
    return ' '.join([stemArabicWordFunction(removePrefixesFunction(removeSuffixesFunction(w))) for w in string.split(' ')])

def firstArabicRule(string):
	return strip_accents(string)

def secondArabicRule(string):
	return removeAlChars(strip_accents(string))

def thirdArabicRule(string):
	return fullPrefSufRemFunction(removeAlChars(strip_accents(string)))


arabic_rules_dict ={
	'no_accents': firstArabicRule,
	'no_accents_noAL': secondArabicRule,
	'no_accents_noAL_noprefsuf': thirdArabicRule
}