# -*- coding: utf-8 -*-
import unicodedata

lookup_dict = {'\u0627\u0623\u0625\u0622\u062d\u062e\u0647\u0639\u063a\u0634\u0648\u064a\u0621': '0',
			   '\u0641\u0628\u0649': '1',
			   '\u062c\u0632\u0633\u0635\u0638\u0642\u0643': '2',
			   '\u062a\u062b\u062f\u0630\u0636\u0637\u0629': '3',
			   '\u0644': '4',
			   '\u0645\u0646': '5',
			   '\u0631': '6'
			   }

lookup_dict_arab = {u'ا': 0, u'أ': 0, u'إ': 0, u'آ': 0, u'ح': 0, u'خ': 0, u'ه': 0, u'ع': 0, u'غ': 0, u'ش': 0, u'و': 0, u'ي': 0, u'ء': 0,
                	u'ف': 1, u'ب': 1, u'ى': 1,
                    u'ج': 2, u'ز': 2, u'س': 2, u'ص': 2, u'ظ': 2, u'ق': 2, u'ك': 2,
                	u'ت': 3, u'ث': 3, u'د': 3, u'ذ': 3, u'ض': 3, u'ط': 3, u'ة':3,
                	u'ل': 4, 
                	u'م': 5, u'ن': 5,
                	u'ر': 6
                	}



def emptyStringError(arab_str):
	if not arab_str:
		raise TypeError('The string passed into this function needs to be non-empty')
	else:
		pass


def readjustSpacesInString(string):
    """
    This function removes the double spaces in a string, on top of the spaces at the beginning/end.

    INPUT
    string - any kind of string, works with Arabic string if utf-8 encoded

    OUTPUT
    The same string without double spaces, or single spaces in the beginning/end of string.
    """

    return ' '.join([x for x in string.split(' ') if x])


def Astr(string):
	'''
	This function encodes in utf-8 a ASCII string with Arabic character in order for it to be
	compatible with the string formatting in Python.

	INPUT
	string - ASCII string with arabic characters

	OUTPUT
	The input string encoded with utf-8
	'''

	return unicode(string, encoding='utf-8')


tanwin_vec = (Astr('ً'), Astr('ٌ'), Astr('ٍ'))

def strip_accents(s, tanween_flag=False):
	'''
	This function takes any unicode string and strips the accents. As this function has been created with arabic in mind, if the
	tanween_flag is True, it will normalize all accents apart from tanween accents.

	INPUT
	- s : this is the unicode string. The function was thought for arabic strings, but it could be any strings
	- tanween_flag : whether tanween accents should be excluded from normalization or not

	OUTPUT
	The string normalized from accents, excluding tanween if tanween_flag is True.
	'''

	if isinstance(s, str):
		s = Astr(s)

	if tanween_flag:
		tanween_vec = [1 if c in tanwin_vec else 0 for c in s]
		return ''.join(unicodedata.normalize('NFD', c) if (unicodedata.category(c) != 'Mn' and tanween_vec[i]==0) else c for i,c in enumerate(s))
	else:
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def arabic_soundex_main(arab_str, firstcharuniforming=True, firstletter_rem=True, accents_strip=True, lim=3):

	emptyStringError(arab_str)

	if isinstance(arab_str, str):
		arab_str = Astr(arab_str)

	if accents_strip:
		arab_str = strip_accents(arab_str)


	#The first thing to do is to exclude the first letter in the firstletter_rem option is activated.
	#This apparently improves the performances of the soundex, according  to Tamman Koujian's C# version of 
	#this soundex.


	if firstletter_rem and arab_str[0] in [u'\u0627',u'\u0623',u'\u0625',u'\u0622']:
		arab_str = arab_str[1:]


	#Now we need to convert the string into the relative sound.
	#Again according to Tamman Koujian's C# version, the first letter is not useful.
	#It firstcharpruning is True, than this happens, otherwise it does not.

	transp = ''
	if firstcharuniforming:
		transp += 'x'
	else:
		transp += arab_str[0]


	#We now proceed to the transposition.

	def charInDictKeys(char, dict_chars):
		keys_unicode = [x for x in dict_chars.keys()]
		pos_vec = [1 if char in x else 0 for x in keys_unicode]
		return dict_chars.keys()[pos_vec.index(1)] if sum(pos_vec)>0 else None


	code, prevcode = None, None
	if len(arab_str)>1:
		for char in arab_str[1:]:

			key = charInDictKeys(char, lookup_dict_arab)

			if key and lookup_dict_arab[key]!='0': code = str(lookup_dict_arab[key])
			else: code = char

			if code != prevcode:
				transp += code
				prevcode = code

	if lim:
		lim = lim +1
		n_transp = len(transp)
		transp = transp[:lim] if len(transp) >= lim else transp+ ''.join(['0' for x in range(lim-n_transp)])

	return transp


def arabic_soundex_names(arab_str, *args, **kwargs):

	emptyStringError(arab_str)

	arab_str = readjustSpacesInString(arab_str)

	soundex_repr = []
	for arabname_part in arab_str.split(' '):
		soundex_repr.append(arabic_soundex_main(arabname_part,*args, **kwargs))

	return ' '.join(soundex_repr)


if __name__ == '__main__':
	print arabic_soundex_main(Astr('ًالسلميٌلسلميٍٍٍلسلمي'))
