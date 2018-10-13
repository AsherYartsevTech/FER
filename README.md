# FER

Data Preparation:
1. clone the repository
2. download fer2013.tar.gz from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
3. run 'tar xvzf fer2013.tar.gz -C ~/FER/FERDataset'. Now you should have ~/FER/FERDataset/fer2013/fer2013.csv
4. run 'python3 csv_to_pngs.py'. Now you will have ~/FER/FERDataset/data with train, valid, test folders inside, each having a folder for each lable.
5. run 'python3 pngs_to_randomized_joblib.py'. Now you will have ~/FER/FERDataset/FER.joblib and you are ready to start training!

Phase #1:
run 'python3 vgglike.py'
