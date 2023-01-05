import random
import os 
import sys

train_dir = sys.argv[1]
test_dir = sys.argv[2]
pos_train_dir = train_dir + '/pos/'
neg_train_dir = train_dir + '/neg/'
pos_test_dir = test_dir + '/pos/'
neg_test_dir = test_dir + '/neg/'

num_positive_test_data = 0
num_negative_test_data = 0

positive_correct_count = 0
negative_correct_count = 0
positive_incorrect_count = 0
negative_incorrect_count = 0

random_prediction = []

for _, _, files in os.walk(pos_test_dir):
    for filename in files:
        num_positive_test_data += 1
        pred_result = random.randrange(0, 2)
        random_prediction.append(pred_result)
        if pred_result == 1:
            positive_correct_count += 1
        else:
            positive_incorrect_count += 1

for _, _, files in os.walk(neg_test_dir):
    for filename in files:
        num_negative_test_data += 1
        pred_result = random.randrange(0, 2)
        random_prediction.append(pred_result)
        if pred_result == 0:
            negative_correct_count += 1
        else:
            negative_incorrect_count += 1


correct = positive_correct_count + negative_correct_count
incorrect = positive_incorrect_count + negative_incorrect_count
print("Accuracy on test data when predicting randomly: ", 100*correct/(correct + incorrect))
print("Accuracy on test data when predicting everything positive: ", 100*num_positive_test_data / (num_positive_test_data + num_negative_test_data))