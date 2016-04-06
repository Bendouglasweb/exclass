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

data_file = open('Results-4_6', 'a')
data_file_nv = open('Results-4_6-nv', 'a')    # "Non-verbose". Only logs final results to file

print("\n *** *** *** *** NEW FILE OPEN *** *** *** *** \n",file=data_file)
data_file.flush()
os.fsync(data_file)

print("\n *** *** *** *** NEW FILE OPEN *** *** *** *** \n",file=data_file_nv)
data_file_nv.flush()
os.fsync(data_file_nv)

# Max K defines the maximum number of K-groupings to loop to
MAXK = 13

# KFold determines how many groupings we split the data into for train/test purposes
# We train with k-1 groups, test against the remaining
KFOLD = 5

# Verbose logging includes each groupings individual columns and each groupings individual training score
VERBOSELOGGING = True

# Files is a list of files to run the experiment on
FILES = {"datasets/kdd10_2","datasets/humanbot.csv","datasets/spambase.csv","datasets/breast_cancer.csv","datasets/credit.csv","datasets/digits08.csv","datasets/qsar.csv","datasets/sonar.csv","datasets/theorem.csv"}
#FILES = {"datasets/humanbot.csv","datasets/breast_cancer.csv","datasets/credit.csv","datasets/digits08.csv","datasets/qsar.csv","datasets/sonar.csv","datasets/theorem.csv"}
#FILES = {"datasets/humanbot.csv","datasets/spambase.csv","datasets/breast_cancer.csv","datasets/credit.csv","datasets/digits08.csv","datasets/qsar.csv","datasets/sonar.csv","datasets/theorem.csv"}
#FILES = {"datasets/breast_cancer.csv","datasets/digits08.csv","datasets/qsar.csv"}
#FILES = {"datasets/qsar.csv"}

TOTALSVMS = 0

#FILES = {"datasets/kdd10_2","datasets/spambase.csv","datasets/sonar.csv"}

# Loop through each file and perform all subsequent testing
for file_value in FILES:
    print("\nStarting analysis for file: %s" % file_value)
    print("\n--------------- Starting analysis for file: %s\n" % file_value,file=data_file)
    data_file.flush()
    os.fsync(data_file)
    print("\n--------------- Starting analysis for file: %s\n" % file_value,file=data_file_nv)
    data_file_nv.flush()
    os.fsync(data_file_nv)
    FILENAME=file_value
    [X,Y]=data_infra.ReadFromFile(FILENAME, shuffle=True)

    n_features = len(X[0])
    n_tuples = len(X)
    kfold_n_tuples = np.floor(n_tuples/KFOLD)      # number of tuples per KFold

    print("Num features in file: %s" % n_features)
    print("Num features in file: %s" % n_features,file=data_file)
    data_file.flush()
    os.fsync(data_file)
    print("Num features in file: %s" % n_features,file=data_file_nv)
    data_file_nv.flush()
    os.fsync(data_file_nv)


    # ---- Do SelectKBest to organize features by impact, saves into yx sorted, descending
    selection=SelectKBest(k=1)
    model=selection.fit(X,Y)
    x=[i for i in range(len(X[0]))]
    scores=model.scores_
    scores=[i if not math.isnan(i) else 0 for i in scores]
    yx=zip(scores,x)
    yx = sorted(yx,key=lambda x: x[0],reverse=True)

    print(yx)

    # This loop/testop loops through the various options for our tests


    # # testops:
    #     0: This is sorting by KBest, using all elements evenly split
    #     1: This is using Ben's algorithm with KBest, v1
    #     2: This is using Ben's algorithm with KBest, v2
    #     3: This is using Ben's algorithm with KBest, v3
    #     4: This is using Ben's algorithm with random sorting, v1
    #     5: This is using Ben's algorithm with random sorting, v1, second iteration
    #     6: This is using Ben's algorithm with random sorting, v2
    #     7: This is using Ben's algorithm with random sorting, v2, second iteration
    #     8: This is using Ben's algorithm with random sorting, v3
    #     9: This is using Ben's algorithm with random sorting, v3, second iteration


    # # testops:

    #     0: This is sorting by KBest, using all elements evenly split
    #     1: This is randomly sorting, using all elements evenly split, iteration: 1
    #     2: This is randomly sorting, using all elements evenly split, iteration: 2
    #     3: This is randomly sorting, using all elements evenly split, iteration: 3
    #     4: This is randomly sorting, using all elements evenly split, iteration: 4
    #     5: This is randomly sorting, using all elements evenly split, iteration: 5
    #     6: This is randomly sorting, using all elements evenly split, iteration: 6
    #     7: This is randomly sorting, using all elements evenly split, iteration: 7
    #     8: This is randomly sorting, using all elements evenly split, iteration: 8
    #     9: This is randomly sorting, using all elements evenly split, iteration: 9
    #     10: This is randomly sorting, using all elements evenly split, iteration: 10


    #   v1: original, just do one round of "does it help?"
    #   v2: original + do k+1, remove lowest then redistribute
    #   v3: v2 + go through and see how many elements we can remove from each. Then redistribute




    for testop in range(11):


        print("\n[[[[ Test iteration: %s ]]]]" % testop)
        print("\nTest iteration: %s\n" % testop,file=data_file)
        data_file.flush()
        os.fsync(data_file)
        print("Test iteration: %s" % testop,file=data_file_nv,end="")
        data_file_nv.flush()
        os.fsync(data_file_nv)

        # For testop 4 and greater, shuffle the order of column
        if testop >= 1:
            random.shuffle(yx)

        print("\nyx: %s\n" % yx,file=data_file)
        data_file.flush()
        os.fsync(data_file)

        # this op variable is used to iterate through the classification options, as defined in the
        # TrainModel method within data_infra.
        for op in range(2):

            if op == 1:
                print("Skipping op 1")
                print("\nSkipping op 1\n",file=data_file)
                data_file.flush()
                os.fsync(data_file)
                print("\nSkipping op 1\n",file=data_file_nv)
                data_file_nv.flush()
                os.fsync(data_file_nv)
                continue

            print("\n----- SVM_LINEAR Op: %s --" % op)
            print("\n----- SVM_LINEAR Op: %s --\n" % op,file=data_file)
            data_file.flush()
            os.fsync(data_file)
            print(". Op: %s" % op,file=data_file_nv)
            data_file_nv.flush()
            os.fsync(data_file_nv)

            results = {}
            results_std = {}

            start_time = timeit.default_timer() # Used for run time

            # Loops through all K groupings, starting with 3
            for K in range(3,MAXK):
                #print("----%s----" % K)
                data_file.flush()
                os.fsync(data_file)

                attribute_list={}       # Used to hold column numbers for features to be used for each group
                remaining = []          # 
                results[K] = []
                results_std[K] = []

                # Get attribute list ready for each of the options:

                if testop in {0,1,2,3,4,5,6,7,8,9,10}:       # We want them all split evenly here
                    for i, value in enumerate(yx):
                        if i%K not in attribute_list:
                            attribute_list[i%K]=[]
                        attribute_list[i%K].append(value[1])

                elif testop in {99}:     # Here we just want the first set, then the rest put into variable remaining
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
                            TOTALSVMS = TOTALSVMS + 1

                            # Get accuracy with new element
                            attributes.append(rem)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            new_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            diff = new_acc - old_acc

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem: %s, diff: %s, holding: [%s,%s]. old/new = %s-%s" % (i,val,rem,diff,holding[0],holding[1],old_acc,new_acc))

                        #print("")

                        if holding[0] >= 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem)
                        else:
                            if rem not in seen:
                                remaining.append(rem)

                        seen.append(rem)
                        holding = [-1,0]

                elif testop in {99}:
                    # Populate attribute_list and remaining
                    for i, value in enumerate(yx):
                        if i%(K+1) not in attribute_list:
                            attribute_list[i%(K+1)]=[]
                        if i < (K+1):
                            attribute_list[i%(K+1)].append(value[1])
                        else:
                            remaining.append(value[1])

                    # Now, we need to run through Ben's algorithm to determine if we want to use any from remaining
                    holding = [-1,0]     # K, K diff
                    unused = []
                    unused2 = []
                    for rem in remaining:

                        for i,val in attribute_list.items():
                            # Get current accuracy
                            attributes=sorted(val)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            old_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            # Get accuracy with new element
                            attributes.append(rem)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            new_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            diff = new_acc - old_acc

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem: %s, diff: %s, holding: [%s,%s]. old/new = %s-%s" % (i,val,rem,diff,holding[0],holding[1],old_acc,new_acc))

                        #print("")

                        if holding[0] >= 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem)
                        else:
                            unused.append(rem)



                        holding = [-1,0]

                    # Now that we have built our first attribute list, we massage it a little.

                    # First, find and remove the lowest scoring grouping

                    tempresults = []

                    for i,val in attribute_list.items():
                        attributes=sorted(val)
                        new_X=[[sample[j] for j in attributes] for sample in X]
                        model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                        res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                        tempresults.append(res['metric'])
                        TOTALSVMS = TOTALSVMS + 1

                    #print("tempresults: %s, with min being %s" % (tempresults,np.argmin(tempresults)))

                    # Save features out of grouping to be re-distrubted
                    for item in attribute_list[np.argmin(tempresults)]:
                        unused.append(item)

                    # Remove lowest scoring group
                    del attribute_list[np.argmin(tempresults)]

                    # *** Now that we've built the first grouping, found the lowest scoring group and removed it,
                    # we shall now go back and see if any of the unused/deleted features can help any of the groups now

                    #print("Unused attributes: %s" % unused)

                    holding = [-1,0]     # K, K diff
                    unused2 = []
                    for rem2 in unused:
                        for i,val in attribute_list.items():
                            # Get current accuracy
                            attributes=sorted(val)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            old_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            # Get accuracy with new element
                            attributes.append(rem2)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            new_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            diff = new_acc - old_acc

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem2: %s, diff: %s, holding: [%s,%s]. old/new = %s-%s" % (i,val,rem2,diff,holding[0],holding[1],old_acc,new_acc))

                        #print("")

                        if holding[0] >= 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem2)
                        else:
                            unused2.append(rem2)



                        holding = [-1,0]

                    #print("Completely unused attributes: %s" % unused2)

                    # Now we have:
                    #   - Split into K+1 groups
                    #   - Removed lowest scoring group
                    #   - Redistrubted those features to other groups
                    #



                elif testop in {99}:
                    # Populate attribute_list and remaining
                    for i, value in enumerate(yx):
                        if i%(K+1) not in attribute_list:
                            attribute_list[i%(K+1)]=[]
                        if i < (K+1):
                            attribute_list[i%(K+1)].append(value[1])
                        else:
                            remaining.append(value[1])

                    # Now, we need to run through Ben's algorithm to determine if we want to use any from remaining
                    holding = [-1,0]     # K, K diff
                    unused = []
                    unused2 = []
                    for rem in remaining:

                        for i,val in attribute_list.items():
                            # Get current accuracy
                            attributes=sorted(val)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            old_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            # Get accuracy with new element
                            attributes.append(rem)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            new_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            diff = new_acc - old_acc

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem: %s, diff: %s, holding: [%s,%s]. old/new = %s-%s" % (i,val,rem,diff,holding[0],holding[1],old_acc,new_acc))

                        #print("")

                        if holding[0] >= 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem)
                        else:
                            unused.append(rem)



                        holding = [-1,0]

                    # Now that we have built our first attribute list, we massage it a little.

                    # First, find and remove the lowest scoring grouping

                    tempresults = []

                    for i,val in attribute_list.items():
                        attributes=sorted(val)
                        new_X=[[sample[j] for j in attributes] for sample in X]
                        model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                        res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                        tempresults.append(res['metric'])
                        TOTALSVMS = TOTALSVMS + 1

                    #print("tempresults: %s, with min being %s" % (tempresults,np.argmin(tempresults)))

                    # Save features out of grouping to be re-distrubted
                    for item in attribute_list[np.argmin(tempresults)]:
                        unused.append(item)

                    # Remove lowest scoring group
                    del attribute_list[np.argmin(tempresults)]

                    # *** Now that we've built the first grouping, found the lowest scoring group and removed it,
                    # we shall now go back and see if any of the unused/deleted features can help any of the groups now

                    #print("Unused attributes: %s" % unused)

                    holding = [-1,0]     # K, K diff
                    unused2 = []
                    for rem2 in unused:
                        for i,val in attribute_list.items():
                            # Get current accuracy
                            attributes=sorted(val)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            old_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            # Get accuracy with new element
                            attributes.append(rem2)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            new_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            diff = new_acc - old_acc

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem2: %s, diff: %s, holding: [%s,%s]. old/new = %s-%s" % (i,val,rem2,diff,holding[0],holding[1],old_acc,new_acc))

                        #print("")

                        if holding[0] >= 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem2)
                        else:
                            unused2.append(rem2)



                        holding = [-1,0]

                    #print("Completely unused attributes: %s" % unused2)

                    # Now we have:
                    #   - Split into K+1 groups
                    #   - Removed lowest scoring group
                    #   - Redistrubted those features to other groups
                    #
                    # So, next we're going to do kbest on each group, and see if removing the lowest value positively impacts the score

                    for i,val in attribute_list.items():
                        #attributes=sorted(val)
                        attributes=val
                        #print("attribute list: %s" % attribute_list)
                        # #print("\nattributes: %s" % attributes)
                        # new_X=[[sample[j] for j in attributes] for sample in X]
                        #
                        # selection=SelectKBest(k='all')
                        # model=selection.fit(new_X,Y)
                        # x=[i for i in range(len(new_X[0]))]
                        # scores=model.scores_
                        # scores=[i if not math.isnan(i) else 0 for i in scores]
                        # new_yx=zip(scores,x)
                        # new_yx=sorted(new_yx,key=lambda x: x[0],reverse=True)

                        #print("(i:%s), new_yx: %s" % (i,new_yx))
                        # print("new_yx: ", end="")
                        # for i,val in enumerate(new_yx):
                        #     print("(%.5f, %s), " % (val[0],val[1]),end="")
                        #
                        # print("")

                        tempa = attributes[:]

                        if (len(attributes) > 1):
                            looplogic = 1
                        else:
                            looplogic = 0

                        # This loops through, removing a low element, and then starting over again
                        while (looplogic == 1):
                            looplogic = 0
                            for l in range(len(attributes)):
                                tempa = attributes[:]
                                tempa.pop(l)

                                # print("attributes   : %s" % attributes)
                                # print("attri w/o low: %s" % tempa)
                                t_new_X=[[sample[k] for k in attributes] for sample in X]
                                model=data_infra.TrainModel(t_new_X,Y,"SVM_LINEAR",op,n_features)
                                res = (data_infra.ComputePerf(data_infra.PredictModel(model,t_new_X),Y))
                                old_acc = res['metric']
                                TOTALSVMS = TOTALSVMS + 1

                                # Get accuracy without low scoring element
                                t_new_X=[[sample[k] for k in tempa] for sample in X]
                                model=data_infra.TrainModel(t_new_X,Y,"SVM_LINEAR",op,n_features)
                                res = (data_infra.ComputePerf(data_infra.PredictModel(model,t_new_X),Y))
                                new_acc = res['metric']
                                TOTALSVMS = TOTALSVMS + 1

                                diff = new_acc - old_acc
                                # print("%s  " % diff,end="")



                                if diff > -0.002:
                                    unused2.append(attributes[l])
                                    attributes.pop(l)
                                    looplogic = 1
                                    break

                            if (len(attributes) <= 1):
                                looplogic = 0

                    #print("unused2: %s" % unused2)

                    # All removed elements were stored in unused3
                    # We're now going to iterate through again, and try to redistribute these among the groups

                    holding = [-1,0]     # K, K diff
                    unused3 = []
                    for rem3 in unused2:
                        for i,val in attribute_list.items():
                            # Get current accuracy
                            attributes=sorted(val)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            old_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            # Get accuracy with new element
                            attributes.append(rem3)
                            new_X=[[sample[j] for j in attributes] for sample in X]
                            model=data_infra.TrainModel(new_X,Y,"SVM_LINEAR",op,n_features)
                            res = (data_infra.ComputePerf(data_infra.PredictModel(model,new_X),Y))
                            new_acc = res['metric']
                            TOTALSVMS = TOTALSVMS + 1

                            diff = new_acc - old_acc

                            # If this one has a greater difference then any previous one, set it
                            if diff > holding[1]:
                                holding[0] = i
                                holding[1] = diff

                            #print("i: %s, val: %s, \t\t\trem3: %s, diff: %s, holding: [%s,%s]. old/new = %s-%s" % (i,val,rem3,diff,holding[0],holding[1],old_acc,new_acc))

                        #print("")

                        if holding[0] >= 0:  # if this was less than 0, then nothing saw a benefit from the additional attribute
                            attribute_list[holding[0]].append(rem3)
                        else:
                            unused3.append(rem3)

                        holding = [-1,0]


                #print("\nAtt: %s" % (attribute_list))
                print("Attribute list: %s" % (attribute_list),file=data_file)
                data_file.flush()
                os.fsync(data_file)

                # Break dataset into KFOLD number of groups, and test the SVMs against each group
                for i,val in attribute_list.items():
                    attributes=sorted(val)
                    new_X=[[sample[j] for j in attributes] for sample in X]

                    temp_acc = []               # Holds results from each fold for later averaging

                    for fn in range(KFOLD):
                        folds = []              # Fold numbers. I.e., fold 0, 1, etc.
                        for j in range(KFOLD):  # Build list of folds
                            folds.append(j)
                        folds.remove(fn)        # Remove one for testing

                        # Build tuples for training
                        first = 0
                        for f in folds:
                            k_start = int(f * kfold_n_tuples)
                            k_end = int(k_start + kfold_n_tuples - 1)

                            if (first == 0):    # If it's the first one, just set it equal to the slice
                                first = 1;
                                train_x = new_X[k_start:k_end]
                                train_y = Y[k_start:k_end]
                            else:               # If it's not the first, append the new slice to the current
                                train_x = np.concatenate((train_x,new_X[k_start:k_end]))
                                train_y = np.concatenate((train_y,Y[k_start:k_end]))

                        # Build tuples for testing
                        k_start = int(fn * kfold_n_tuples)
                        k_end = int(k_start + kfold_n_tuples - 1)
                        test_x = new_X[k_start:k_end]
                        test_y = Y[k_start:k_end]

                        # Train and Test!
                        model=data_infra.TrainModel(train_x,train_y,"SVM_LINEAR",op,n_features)
                        res = data_infra.ComputePerf(test_y,data_infra.PredictModel(model,test_x))
                        temp_acc.append(res['metric'])      # Append result for each FOLD
                        TOTALSVMS = TOTALSVMS + 1

                    # Append results from the folds
                    results[K].append(sum(temp_acc) / float(len(temp_acc)))
                    results_std[K].append(np.std(temp_acc))

                sys.stdout.flush()  # Was needed to get it to actually print

            end_time = timeit.default_timer()


            print("\nAll Data: ")
            print("\nAll Data: \n",file=data_file)
            data_file.flush()
            os.fsync(data_file)
            print("\nAll Data: \n",file=data_file_nv)
            data_file_nv.flush()
            os.fsync(data_file_nv)

            print("(K,Avg,StD,Max,Min,Swing)",file=data_file)
            data_file.flush()
            os.fsync(data_file)
            for K in range (3,MAXK):

                print("|| (K:%s) Avg: %.5f, StD: %.5f, Max: %.5f, Min: %.5f, Swing: %.5f" % (K,sum(results[K]) / float(len(results[K])),sum(results_std[K]) / float(len(results_std[K])),np.amax(results[K]),np.amin(results[K]),np.amax(results[K]) - np.amin(results[K])))
                print("%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (K,sum(results[K]) / float(len(results[K])),sum(results_std[K]) / float(len(results_std[K])),np.amax(results[K]),np.amin(results[K]),np.amax(results[K]) - np.amin(results[K])),file=data_file)
                data_file.flush()
                os.fsync(data_file)
                print("%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (K,sum(results[K]) / float(len(results[K])),sum(results_std[K]) / float(len(results_std[K])),np.amax(results[K]),np.amin(results[K]),np.amax(results[K]) - np.amin(results[K])),file=data_file_nv)
                data_file_nv.flush()
                os.fsync(data_file)

            print("\n-Detailed Info\nIndividual Group Accuracies\n",file=data_file)
            data_file.flush()
            os.fsync(data_file_nv)
            for K in range (3,MAXK):
                print("%s,%s" % (K,results[K]),file=data_file)
                data_file.flush()
                os.fsync(data_file)

            print("\nIndividual Group Standard Deviations",file=data_file)
            data_file.flush()
            os.fsync(data_file)
            for K in range (3,MAXK):
                print("%s,%s" % (K,results_std[K]),file=data_file)
                data_file_nv.flush()
                os.fsync(data_file_nv)

            print("\nDone in: %s seconds" % (end_time - start_time))
            print("\nDone in: %s seconds\n" % (end_time - start_time),file=data_file)
            data_file.flush()
            os.fsync(data_file)






    print("\n------ END OF %s ------ \n" % file_value,file=data_file)
    data_file.flush()
    os.fsync(data_file)
    print("\n------ END OF %s ------ \n" % file_value,file=data_file_nv)
    data_file_nv.flush()
    os.fsync(data_file_nv)

    # -------------- END SELECT K BEST --------------

print("\nTotal SVMs: %s\n" % TOTALSVMS,file=data_file)
data_file.flush()
os.fsync(data_file)
print("\nTotal SVMs: %s\n" % TOTALSVMS,file=data_file_nv)
data_file_nv.flush()
os.fsync(data_file_nv)

data_file.close()
data_file_nv.close()
