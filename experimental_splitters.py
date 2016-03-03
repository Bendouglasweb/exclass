import data_infra
import numpy as np
import matplotlib.pyplot as plt
import pprint
from sklearn.feature_selection import SelectKBest, chi2
import math
import time
from sklearn.ensemble import ExtraTreesClassifier

import sys
import timeit
import random
import os

data_file = open('hello','w')


MAXK = 10
#FILES = {"spambase.csv","humanbot.csv"}
FILES = {"humanbot.csv","spambase.csv"}
#FILES = {"humanbot.csv"}
#FILES = {"spambase.csv"}

for file_value in FILES:
    print("\nStarting analysis for file: %s\n" % file_value)
    # print("\n--------------- Starting analysis for file: %s\n" % file_value,file=data_file)
    # data_file.flush()
    # os.fsync(data_file)
    #FILENAME='spambase.csv'#'humanbot'
    FILENAME=file_value
    [X,Y]=data_infra.ReadFromFile(FILENAME, shuffle=True)
    n_features = len(X[0])

    print(n_features)

    # -------------- SELECT K BEST --------------
    selection=SelectKBest(k=10)
    model=selection.fit(X,Y)
    x=[i for i in range(len(X[0]))]

    scores=model.scores_
    scores=[i if not math.isnan(i) else 0 for i in scores]
    yx=zip(scores,x)
    #yx.sort(reverse=True)
    yx = sorted(yx,key=lambda x: x[0],reverse=True)


    #yx = yx[:MAXK*2]

    #for op in range(9):
    op = 0
    # print("\n----- SVM_LINEAR Op: %s --\n" % op,file=data_file)
    # data_file.flush()
    # os.fsync(data_file)
    results = {}
    # print("Starting Sim for KBest w/ SVM_LINEAR Op=%s.\nKs completed:" % op,end="")
    start_time = timeit.default_timer()
    for K in range(1,MAXK):
        print("----%s----" % K)
        attribute_list={}
        remaining = []
        results[K] = []

        for i, value in enumerate(yx):

            if i%K not in attribute_list:
                attribute_list[i%K]=[]
            if i < K:
                attribute_list[i%K].append(value[1])
            else:
                remaining.append(value[1])


        holding = [-1,0]     # K, K diff
        seen = []
        for rem in remaining:
            for i,val in attribute_list.items():
                # Get current accuracy
                attributes=sorted(val)
                new_X=[[sample[j] for j in attributes] for sample in X]
                model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                old_acc = res['metric']

                # Get accuracy with new element
                attributes.append(rem)
                new_X=[[sample[j] for j in attributes] for sample in X]
                model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                new_acc = res['metric']

                diff = new_acc - old_acc

                if diff > holding[1]:
                    holding[0] = i
                    holding[1] = diff

                #print("i: %s, val: %s, \t\t\trem: %s, diff: %s, holding: %s, %s. old/new = %s-%s" % (i,val,rem,diff,holding[0],holding[1],old_acc,new_acc))

            if holding[0] > 0:
                attribute_list[holding[0]].append(rem)
            else:
                if rem not in seen:
                    remaining.append(rem)

            seen.append(rem)
            holding = [-1,0]


        print("  Att: %s" % (attribute_list))
        for i,val in attribute_list.items():
            attributes=sorted(val)
            new_X=[[sample[j] for j in attributes] for sample in X]
            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
            results[K].append(res['metric'])
            print(results[K])
        # print("%s " % K,end="")
        sys.stdout.flush()  # Was needed to get it to actually print

    end_time = timeit.default_timer()


    print("\nAll Data: ")
    #pprint.pprint(results)
    print("------")
    for K in range (1,MAXK):
        print("K: %s, Avg: %s" % (K,sum(results[K]) / float(len(results[K]))))
        #print("%s,%s" % (K,sum(results[K]) / float(len(results[K]))),file=data_file)
        #data_file.flush()
        #os.fsync(data_file)

    # print("\n-Detailed Info\n",file=data_file)
    # data_file.flush()
    # os.fsync(data_file)
    # for K in range (1,MAXK):
    #     print("%s,%s" % (K,results[K]),file=data_file)
    #     data_file.flush()
    #     os.fsync(data_file)
    print("Done in: %s seconds" % (end_time - start_time))




    # print("\n------ END OF %s ------ \n" % file_value,file=data_file)
    # data_file.flush()
    # os.fsync(data_file)

    # -------------- END SELECT K BEST --------------




data_file.close()