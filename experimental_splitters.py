import data_infra
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import math

import pprint
import time
from sklearn.ensemble import ExtraTreesClassifier

import sys
import timeit
import random
import os

data_file = open('Results-2', 'a')

print("\n *** *** *** *** NEW FILE OPEN *** *** *** *** \n",file=data_file)
data_file.flush()
os.fsync(data_file)

# Max K defines the maximum number of K-groupings to loop to
MAXK = 21

# Files is a list of files to run the experiment on
#FILES = {"datasets/kdd10_2","datasets/humanbot.csv","datasets/spambase.csv","datasets/breast_cancer.csv","datasets/credit.csv","datasets/digits08.csv","datasets/qsar.csv","datasets/sonar.csv","datasets/theorem.csv"}
#FILES = {"datasets/humanbot.csv","datasets/spambase.csv","datasets/breast_cancer.csv","datasets/credit.csv","datasets/digits08.csv","datasets/qsar.csv","datasets/sonar.csv","datasets/theorem.csv"}
FILES = {"datasets/kdd10_2","datasets/spambase.csv","datasets/sonar.csv"}
#FILES = {"datasets/kdd10_2","datasets/spambase.csv","datasets/sonar.csv"}

# Loop through each file and perform all subsequent testing
for file_value in FILES:
    print("\nStarting analysis for file: %s\n" % file_value)
    print("\n--------------- Starting analysis for file: %s\n" % file_value,file=data_file)
    data_file.flush()
    os.fsync(data_file)
    FILENAME=file_value
    [X,Y]=data_infra.ReadFromFile(FILENAME, shuffle=True)
    n_features = len(X[0])

    print("Num features in file: %s" % n_features)
    print("Num features in file: %s" % n_features,file=data_file)
    data_file.flush()
    os.fsync(data_file)


    # ---- Do SelectKBest to organize features by impact, saves into yx sorted, descending
    selection=SelectKBest(k=10)
    model=selection.fit(X,Y)
    x=[i for i in range(len(X[0]))]
    scores=model.scores_
    scores=[i if not math.isnan(i) else 0 for i in scores]
    yx=zip(scores,x)
    yx = sorted(yx,key=lambda x: x[0],reverse=True)



    # This loop/testop loops through the various options for our tests

    # # testops:
    #     0: This is sorting by KBest, using all elements evenly split
    #     1: This is using Ben's algorithm with KBest
    #     2: This is using all elements evenly, but randomly sorted | 2 iterations
    #     3: Second iteration of 2
    #     4: This is using Ben's algorithm with random sorting | 2 iterations
    #     5: Second iteration of 4
    for testop in range(6):

        if testop == 0 or testop != 2 or testop != 3:
            print("Skipping iteration %s, continuing on to the next!" % testop)
            print("Skipping iteration %s, continuing on to the next!" % testop,file=data_file)
            data_file.flush()
            os.fsync(data_file)
            continue


        print("\nTest iteration: %s\n" % testop)
        print("\nTest iteration: %s\n" % testop,file=data_file)
        data_file.flush()
        os.fsync(data_file)

        # For testop 2 and greater, shuffle the order of elements
        if testop >= 2:
            random.shuffle(yx)

        print("\nyx: %s\n" % yx,file=data_file)
        data_file.flush()
        os.fsync(data_file)

        # this op variable is used to iterate through the classification options, as defined in the
        # TrainModel method within data_infra.
        for op in range(2):

            if op == 1:
                print("\nSkipping op 1\n")
                print("\nSkipping op 1\n",file=data_file)
                continue

            print("\n----- SVM_LINEAR Op: %s --\n" % op)
            print("\n----- SVM_LINEAR Op: %s --\n" % op,file=data_file)
            data_file.flush()
            os.fsync(data_file)

            results = {}

            start_time = timeit.default_timer() # Used for run time

            # Loops through all K groupings
            for K in range(1,MAXK):
                print("----%s----" % K)
                data_file.flush()
                os.fsync(data_file)

                attribute_list={}       # Used to hold column numbers for features to be used for each group
                remaining = []          # 
                results[K] = []

                # Get attribute list ready for each of the options:

                if testop in {0,2,3}:       # We want them all split evenly here
                    for i, value in enumerate(yx):
                        if i%K not in attribute_list:
                            attribute_list[i%K]=[]
                        attribute_list[i%K].append(value[1])
                elif testop in {1,4,5}:     # Here we just want the first set, then the rest put into remaining
                    # Populate attribute_list and remaining
                    for i, value in enumerate(yx):
                        if i%K not in attribute_list:
                            attribute_list[i%K]=[]
                        if i < K:
                            attribute_list[i%K].append(value[1])
                        else:
                            remaining.append(value[1])

                    # Now, we need to run through Ben's algorithm to determine if we want to use any from remaining
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

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem: %s, diff: %s, holding: %s, %s. old/new = %s-%s" % (i,val,rem,diff,holding[0],holding[1],old_acc,new_acc))

                        if holding[0] > 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem)
                        else:
                            if rem not in seen:
                                remaining.append(rem)

                        seen.append(rem)
                        holding = [-1,0]


                print("  Att: %s" % (attribute_list))
                print("Attribute list: %s" % (attribute_list),file=data_file)
                data_file.flush()
                os.fsync(data_file)
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
            print("\nAll Data: \n",file=data_file)
            data_file.flush()
            os.fsync(data_file)
            #pprint.pprint(results)
            print("------")
            print("{K,Avg,StD,Max,Min,Swing)",file=data_file)
            data_file.flush()
            os.fsync(data_file)
            for K in range (1,MAXK):


                print("(K:%s) Avg: %.5f, StD: %.5f, Max: %.5f, Min: %.5f, Swing: %.5f" % (K,sum(results[K]) / float(len(results[K])),np.std(results[K]),np.amax(results[K]),np.amin(results[K]),np.max(results[K]) - np.amin(results[K])))
                print("%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (K,sum(results[K]) / float(len(results[K])),np.std(results[K]),np.amax(results[K]),np.amin(results[K]),np.max(results[K]) - np.amin(results[K])),file=data_file)

                data_file.flush()
                os.fsync(data_file)

            print("\n-Detailed Info\n",file=data_file)
            data_file.flush()
            os.fsync(data_file)
            for K in range (1,MAXK):
                print("%s,%s" % (K,results[K]),file=data_file)
                data_file.flush()
                os.fsync(data_file)
            print("Done in: %s seconds" % (end_time - start_time))
            print("\nDone in: %s seconds\n" % (end_time - start_time),file=data_file)




    print("\n------ END OF %s ------ \n" % file_value,file=data_file)
    data_file.flush()
    os.fsync(data_file)

    # -------------- END SELECT K BEST --------------

data_file.close()
