// programming Naive Bayes to classify SPAM


import pandas as pd
from glob import glob
import math
import os

spam_word_counts = {}
ham_word_counts = {}
total_words = 0


files_spam = glob("./Documents/MSCS/MachineLearning/HamSpam/spam/*")
files_ham = glob("./Documents/MSCS/MachineLearning/HamSpam/ham/*")

spam_files = len(files_spam)
ham_files = len(files_ham)

for file in files_spam: 
    with open(file, "r") as i_file: 
        for line in i_file: 
            line = line.strip()
            total_words += 1
            if line in spam_word_counts:
                spam_word_counts[line] += 1
            else: 
                spam_word_counts[line] = 1


for file in files_ham: 
    ham_files += 1
    with open(file, "r") as i_file: 
        for line in i_file: 
            line = line.strip()
            total_words += 1
            if line in ham_word_counts:
                ham_word_counts[line] += 1
            else: 
                ham_word_counts[line] = 1

total_files = spam_files + ham_files

# hyperparameters
alpha = .005
vocab = 150000

# data smoothing

for key in spam_word_counts:
    spam_word_counts[key] += alpha

for key in ham_word_counts:
    ham_word_counts[key] += alpha

spam_probability = spam_files/total_files


# 3. Classification

def classify(file):

    spam_final_probability = 0
    ham_final_probability = 0

    #SPAM

    spam_words_proability = 0
    with open(file, "r") as f:
        for line in f: 
            word_count = spam_word_counts.get(line.strip(), alpha)
            word_probability = word_count/((alpha * vocab) + len(spam_word_counts)) 
            spam_words_proability += math.log(word_probability)
    spam_final_probability = math.log(spam_probability) + spam_words_proability

    # HAM

    ham_words_proability = 0
    with open(file, "r") as f:
        for line in f: 
            word_count = ham_word_counts.get(line.strip(), alpha)
            word_probability = word_count/((alpha * vocab) + len(ham_word_counts)) 
            ham_words_proability += math.log(word_probability)
    ham_final_probability = math.log(1 - spam_probability) + ham_words_proability

    if ham_final_probability >= spam_final_probability:
        return "HAM"
    else: 
        return "SPAM"

# TESTING

test_results = {}
test_results["TP"] = 0
test_results["TN"] = 0
test_results["FP"] = 0
test_results["FN"] = 0

true_spam = {}

with open("./Documents/MSCS/MachineLearning/HamSpam/truthfile", "r") as truth:
    for line in truth:
        true_spam[line.strip()] = 0


test_files = glob("./Documents/MSCS/MachineLearning/HamSpam/test/*")

for file in sorted(test_files):
    prediction = classify(file)
    filename = os.path.basename(file).split(".")[0]
    if prediction == "SPAM":
        if filename in true_spam:
            test_results["TP"] += 1
        else:
            test_results["FP"] += 1

    if prediction == "HAM":
        if filename in true_spam:
            test_results["FN"] += 1
        else:
            test_results["TN"] += 1

TN = test_results["TN"]
TP = test_results["TP"]
FP = test_results["FP"]
FN = test_results["FN"]

print(TN, " is TN")
print(TP, " is TP")
print(FN, " is FN")
print(FP, " is FP")


accuracy = (TN + TP)/ (TN + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP/ (TP + FN) 
f_score = (2 * precision * recall)/(precision + recall)

print("Accuracy of the model is ", accuracy)
print("Precision of the model is ", precision)
print("Recall of the model is ", recall)
print("The f-score of the model is ", f_score)