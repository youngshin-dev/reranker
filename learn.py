#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
from collections import defaultdict
import bleu
import random
import math
import numpy
from operator import itemgetter


optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("-r", "--ref", dest="ref", default=os.path.join("data", "train.fr"), help="reference")
optparser.add_option("-t", "--target", dest="tar", default=os.path.join("data", "train.en"), help="proper translation")

epoches=5
tau = 500
alpha=0.1
xi=100
theta = numpy.random.rand(6)
eta=0.1

(opts, _) = optparser.parse_args()

source = open(opts.ref).read().splitlines()
target = open(opts.tar).read().splitlines()
a_translation=namedtuple('a_translation','sentence, features, smoothed_bleu')
nbests = [[] for i in range(len(source))]


for line in open(opts.nbest):
    (i, sentence, features) = line.strip().split("|||")
    ind=int(i)
    #stats=bleu.bleu_stats(sentence, source[ind])
    stats=list(bleu.bleu_stats(sentence, target[ind]))
    #test1=test[0]
    bleu_smooth_score=bleu.smoothed_bleu(stats)
    feature_vec=numpy.fromstring(features, sep=' ')
    nbests[ind].append(a_translation(sentence,feature_vec,bleu_smooth_score))

def get_sample(nbest):
    sample=[]
    for i in range(0,tau):
        #random_items = random.sample(nbest, 2)
        #s1 = random_items[0]
        #s2 = random_items[1]
        s1=random.choice(nbest) 
        s2=random.choice(nbest) 
        if math.fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha:
            if s1.smoothed_bleu > s2.smoothed_bleu:
                #tuple1=(s1, s2)
                sample.append([s1,s2])
            else:
                #tuple2 = (s2 ,s1)
                sample.append([s2,s1])
        else:
            continue
    return sample

for i in range(0,epoches):
    for nbest in nbests:

        sample=get_sample(nbest)
        #sorted_sample=sample.sort(key=lambda tup:(math.fabs(tup[0].smoothed_bleu-tup[1].smoothed_bleu)))[:xi]
        sorted_sample=sorted(sample,key=lambda x: math.fabs(x[0].smoothed_bleu-x[1].smoothed_bleu))[:xi]
        mistakes=0
        for item in sorted_sample:
            feature1=item[0].features
            feature2=item[1].features
            if numpy.dot(theta,feature1)<=numpy.dot(theta,feature2):
                mistakes=mistakes+1
                theta=theta+eta*(feature1-feature2)
            else:
                mistakes = mistakes + 1
                theta = theta + eta * (feature2 - feature1)

def main():
    print "\n".join([str(weight) for weight in theta])

if __name__ == "__main__":
    main()
#features = [float(h) for h in features.strip().split()]
#w = [1.0/len(features) for _ in xrange(len(features))]
#break

#print "\n".join([str(weight) for weight in w])
