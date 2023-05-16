file = open(r'log\TEST_classifier_squaresmall_1min_zscore_shuffle_ROC.txt','r')
file_content = file.readlines()
for content in file_content:
    if 'Total accuracy: ' in content:
        print(float(content[-7:-2]))