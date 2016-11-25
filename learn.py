#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
from collections import defaultdict
import bleu
import random
import math
from numpy import array, dot, random



optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("-r", "--ref", dest="ref", default=os.path.join("data", "train.fr"), help="reference")

epoches=5
tau = 500
alpha=0.1
xi=100
w = random.rand(6)
eta=0.1

(opts, _) = optparser.parse_args()
#translations={}
translations=defaultdict(list)
#all_translations=[]
#translations=[]
source = open(opts.ref).read().splitlines()
a_translation=namedtuple('a_translation','sentence, features, stats')
nbests = [[] for i in range(len(source))]
for line in open(opts.nbest):
    (i, sentence, features) = line.strip().split("|||")
    ind=int(i)
    bleu_stat=bleu.bleu_stats(sentence, source[ind])
    #test=list(bleu_stat)[:2]
    #test1=test[0]
    bleu_score=bleu.bleu(bleu_stat)


    nbests[ind].append(a_translation(sentence,features,bleu.bleu_stats(sentence, source[ind])))

def get_sample(nbest):
    sample=[]
    for i in range(0,tau):
        random_items = random.sample(nbest, 2)
        s1 = random_items[0]
        s2 = random_items[1]
        if math.fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha:
            if s1.smoothed_bleu > s2.smoothed_bleu:
                sample += (s1, s2)
            else:
                sample += (s2, s1)
        else:
            continue
    return sample

for i in range(0,epoches):
    for nbest in nbests:
        #random_items = random.sample(nbest, 2)
        #s1 = random_items[0]
        #s2 = random_items[1]
        s=nbest[4]
        test=s.stats.smoothed_bleu
        '''
        sample=get_sample(nbest).sort(key=lambda tup:(math.fabs(tup[0]-tup[1])))[:xi]
        mistakes=0
        for item in sample:
            if dot(w,item[1])<=dot(w,item[0]):
                mistakes=mistakes+1
                w=eta*()
        '''


#features = [float(h) for h in features.strip().split()]
#w = [1.0/len(features) for _ in xrange(len(features))]
#break

#print "\n".join([str(weight) for weight in w])
