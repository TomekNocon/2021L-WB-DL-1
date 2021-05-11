import numpy as np

from keras.models import Sequential
from data import BalanceCovidDataset
import os, argparse, pathlib
import keras_model
import tensorflow as tf


parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--bs', default=5, type=int, help='Batch size')
parser.add_argument('--weightspath', default='models/COVIDNet-CXR4-A', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='labels/under.txt', type=str, help='Path to train file') #'labels/train_COVIDx7A_new.txt' #over bylo dla ostanich wykonan # moge chceic znowu tu miec labels/under.txt
parser.add_argument('--testfile', default='labels/test_COVIDx7A_new.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='data', type=str, help='Path to data folder')
parser.add_argument('--covid_weight', default=4., type=float, help='Class weighting for covid') #bylo 4 #a wzialem 2 do under over samplingu
parser.add_argument('--covid_percent', default=0.3, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_1/MatMul:0', type=str, help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str, help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str, help='Name of sample weights tensor for loss')

args = parser.parse_args()

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1


generator = BalanceCovidDataset(data_dir=args.datadir,
                                csv_file=args.trainfile,
                                batch_size=batch_size,
                                input_shape=(args.input_size, args.input_size),
                                covid_percent=args.covid_percent,
                                class_weights=[1., 1., args.covid_weight],
                                top_percent=args.top_percent)

# loss_op = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
#     logits=pred_tensor, labels=labels_tensor) * sample_weights)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)


model = keras_model.keras_model_build()

optimizer=tf.keras.optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])# loss nie jest poprawiony i metryka jeszcze

model.fit_generator(generator=generator,
                    epochs=1,
                    #class_weight = [1.,1., args.covid_weight],
                    verbose=1)#,
                    #use_multiprocessing=True)

print("co jest xDDDDDDDDDDDD")
model.save('models/model.h5')

################################################################################################################################
#walidacja taka na szbyko

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}


#definiowane z palca normalnie przekwazywane w funkcji
testfile = args.testfile
testfolder = os.path.join(args.datadir,'test')
#
from data import process_image_file
from sklearn.metrics import confusion_matrix


image_tensor = args.in_tensorname
pred_tensor = args.out_tensorname

y_test = []
pred = []
for i in range(len(testfile)):
    line = testfile[i].split()
    x = process_image_file(os.path.join(testfolder, line[1]), 0.08, args.input_size)
    x = x.astype('float32') / 255.0
    y_test.append(mapping[line[2]])
    pred.append(np.array(model.predict(x)).argmax(axis=1))
y_test = np.array(y_test)
pred = np.array(pred)

matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float')
#cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
print(matrix)
#class_acc = np.array(cm_norm.diagonal())
class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                           class_acc[1],
                                                                           class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                         ppvs[1],
                                                                         ppvs[2]))
