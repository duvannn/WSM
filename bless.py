from collections import defaultdict
from scipy import sparse, io, spatial 
import math
import numpy

class Bless:
    def __init__(self):
        self.correctlist = []
        self.unknown_target = []
        self.unknown_answer = []
        self.unknown_cosine = []
        self.dubbles = []
        self.incorrect = []
        self.wordlist = []
        self.dictyp = {}
        self.t = ''

    def run(matrix,dic):
        correctlist = []
        unknown_target = []
        unknown_answer = []
        unknown_cosine = []
        dubbles = []
        incorrect = []
        wordlist = []
        dictyp = {}
        t = ''
        with open('BLESS.txt')  as bless:
            lines = bless.readlines()
            while len(lines)>=1:
                target, klass, typ, word = lines.pop().split()
                target = target.replace('-',' ').split()[0]
                word = word.replace('-',' ').split()[0]
                typ =  typ.replace('-',' ').split()[0]
                if(target != t):
                    if(t!=''):
                        if dic.has_key(target):
                            targetvec = matrix[dic[target],:]
                            highest = 0
                            answer = ''
                            for alt in wordlist:
                                if alt!=target and dic.has_key(alt):
                                    altvec = matrix[dic[alt],:]
                                    altlen = 1 - spatial.distance.cosine(targetvec,altvec)
                                    if math.isnan(altlen):
                                        unknown_cosine.append((target,alt))
                                    if(altlen>highest):
                                        highest = altlen
                                        answer = alt
                                else:
                                    dubbles.append(alt)
                            typen = dictyp[answer]
                            if(typen == 'hyper' or typen == 'coord'):
                                correctlist.append((target,answer,highest))
                            else:
                                incorrect.append((target,answer,highest))
                        else:
                            unknown_target.append(target)
                    t = target
                    wordlist = []
                    dictyp = {}
                wordlist.append(word)
                dictyp[word] = typ
        print ('correct:', correctlist)
        print ('incorect:', incorrect)
        print ('unknowns:', unknown_target, unknown_cosine)
        print float(len(correctlist))/float((len(correctlist)+len(incorrect)))
        return float(len(correctlist))/float((len(correctlist)+len(incorrect)))