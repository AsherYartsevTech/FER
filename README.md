# FER

Data Preparation:
1. clone the repository
2. download fer2013.tar.gz from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data (we've also added it under nets/EyesWithGans/FerDataset)
3. run 'tar xvzf fer2013.tar.gz -C ~/FER/FERDataset'. Now you should have ~/FER/FERDataset/fer2013/fer2013.csv
4. run 'python3 csv_to_pngs.py'. Now you will have ~/FER/FERDataset/data with train, valid, test folders inside, each having a folder for each lable.
5. run 'python3 pngs_to_randomized_joblib.py'. Now you will have ~/FER/FERDataset/FER.joblib and you are ready to start training!

Phase #1:
run 'python3 nets/vgglike.py'

Phase #2:
1. run 'python3 pngs_to_randomized_joblib_4dims.py' to prepare a new joblib with the images array as a 4 dimensions array
2. run 'python3 nets/vgglike4ensemble.py'
3. take latest weights file from weights that relates to the vgglike net and put it's name in the nets/ensemble.py in the right place.
4. repeat steps 2-3 for the other architectures: dense4ensemble, convPoolCnn, allCnn, ninCnn
5. run 'python3 nets/ensemble.py'


EyesWithGans part:
1. in nets/EyesWithGans/eyeswithGuns execute csv_to_pngs.py and than pngs_to_randomized_pickle.py.
2. in nets/EyesWithGans/eyeswithGuns/GlobalVarsAndLibs.py you can set iterations and epochs numbers as you wish,
and also control some other hyper-parameters.
3. than, just execute nets/EyesWithGans/eyeswithGuns/main.py
4. pay attention that each run, a tensorboard file is created where the graph can be viewed.
5. we also commited a ready tensorboard file in results/eyesWithGansResAndPng/tf_logs/run-20180907023509/
