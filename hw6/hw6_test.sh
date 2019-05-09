wget 'https://github.com/chris94137/ML-model/releases/download/0.0.2/best_acc_0.76050.h5'
wget 'https://github.com/chris94137/ML-model/releases/download/0.0.2/embed.model'
python hw6_test.py $3 ./best_acc_0.76050.h5 $2 $1 ./embed.model 