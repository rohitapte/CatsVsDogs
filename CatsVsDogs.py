import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
import random
import math

#LEARNING_RATE = 1e-4
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY=0.1
NUM_GENS_TO_WAIT=250.0
TRAINING_ITERATIONS = 200000
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 5000
IMAGE_SIZE=64
CHANNELS=3
NUM_ANIMALS=2

#TRAIN_DIR='../data/CatsVsDogs/input/train/'
#TEST_DIR='../data/CatsVsDogs/input/test/'
TRAIN_DIR='input/train/'
TEST_DIR='input/test/'

train_images=[TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs=[TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats=[TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images=[TEST_DIR+i for i in os.listdir(TEST_DIR)]

def dense_to_one_hot(labels_dense, num_classes):
	num_labels=labels_dense.shape[0]
	index_offset=np.arange(num_labels)*num_classes
	labels_one_hot=np.zeros((num_labels,num_classes))
	labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
	return labels_one_hot

def read_image_and_resize(filename):
	#img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
	#return cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_LINEAR)
	img=cv2.imread(filename,cv2.IMREAD_COLOR)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	if (img.shape[0] >= img.shape[1]):
		resizeto=(IMAGE_SIZE,int(round(IMAGE_SIZE*(float (img.shape[1])/img.shape[0]))));
	else:
		resizeto = (int (round (IMAGE_SIZE * (float (img.shape[0])  / img.shape[1]))), IMAGE_SIZE);
	img2=cv2.resize(img,(resizeto[1],resizeto[0]),interpolation=cv2.INTER_CUBIC)
	img3=cv2.copyMakeBorder(img2,0,IMAGE_SIZE-img2.shape[0],0,IMAGE_SIZE-img2.shape[1],cv2.BORDER_CONSTANT,0)
	#return img3[:,:,::-1]
	return img3

def convert_image_to_vector(images):
	count=len(images)
	data=np.ndarray((count,IMAGE_SIZE,IMAGE_SIZE,CHANNELS),dtype=np.float32)
	for i,image_file in enumerate(images):
		animalimage=read_image_and_resize(image_file)
		data[i]=animalimage
		if i%100==0:print('Processed {} of {}'.format(i,count))
	return data

x_train=convert_image_to_vector(train_images)
x_test=convert_image_to_vector(test_images)
#mean=np.mean(x_train)
#stddev=np.std(x_train)
#x_test-=mean
#x_test/=stddev

train_labels=[]
for i in train_images:
	if 'dog' in i:train_labels.append(1)
	if 'cat' in i:train_labels.append(0)

y_train=dense_to_one_hot(np.array(train_labels),NUM_ANIMALS)
num_examples=x_train.shape[0]
perm = np.arange(num_examples)
np.random.shuffle(perm)
x_train=x_train[perm]
y_train=y_train[perm]
x_val=x_train[:VALIDATION_SIZE]
x_train=x_train[VALIDATION_SIZE:]
y_val=y_train[:VALIDATION_SIZE]
y_train=y_train[VALIDATION_SIZE:]

def updateImage(x_train_data,distort=True):
	#global mean
	#global stddev
	x_temp=x_train_data.copy()
	x_output=np.zeros(shape=(0,IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
	for i in range(0,x_temp.shape[0]):
		temp=x_temp[i]
		if distort:
			if random.random()>0.5:
				temp=np.fliplr(temp)
			brightness=random.randint(-63,63)
			temp=temp+brightness
			contrast=random.uniform(0.2,1.8)
			temp=temp*contrast
		mean=np.mean(temp)
		stddev=np.std(temp)
		temp=(temp-mean)/stddev
		temp=np.expand_dims(temp,axis=0)
		x_output=np.append(x_output,temp,axis=0)
	return x_output

x_test=updateImage(x_test,False)

def truncated_normal_var(name,shape,dtype):
	return(tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.truncated_normal_initializer(stddev=0.05)))
def zero_var(name,shape,dtype):
	return(tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.constant_initializer(0.0)))

x=tf.placeholder(tf.float32,shape=[None,x_train.shape[1],x_train.shape[2],x_train.shape[3]],name='x')
labels=tf.placeholder(tf.float32,shape=[None,y_train.shape[1]],name='labels')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')

with tf.variable_scope('conv1') as scope:
	conv1_kernel=truncated_normal_var(name='conv1_kernel',shape=[5,5,3,64],dtype=tf.float32)
	strides=[1,1,1,1]
	conv1=tf.nn.conv2d(x,conv1_kernel,strides,padding='SAME')
	conv1_bias=zero_var(name='conv1_bias',shape=[64],dtype=tf.float32)
	conv1_add_bias=tf.nn.bias_add(conv1,conv1_bias)
	relu_conv1=tf.nn.relu(conv1_add_bias)

pool_size=[1,3,3,1]
strides=[1,2,2,1]
pool1=tf.nn.max_pool(relu_conv1,ksize=pool_size,strides=strides,padding='SAME',name='pool_layer1')
norm1=tf.nn.lrn(pool1,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm1')

with tf.variable_scope('conv2') as scope:
	conv2_kernel=truncated_normal_var(name='conv2_kernel',shape=[5,5,64,64],dtype=tf.float32)
	strides=[1,1,1,1]
	conv2=tf.nn.conv2d(norm1,conv2_kernel,strides,padding='SAME')
	conv2_bias=zero_var(name='conv2_bias',shape=[64],dtype=tf.float32)
	conv2_add_bias=tf.nn.bias_add(conv2,conv2_bias)
	relu_conv2=tf.nn.relu(conv2_add_bias)

pool_size=[1,3,3,1]
strides=[1,2,2,1]
pool2=tf.nn.max_pool(relu_conv2,ksize=pool_size,strides=strides,padding='SAME',name='pool_layer2')
norm2=tf.nn.lrn(pool2,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm2')

with tf.variable_scope('conv3') as scope:
	conv3_kernel=truncated_normal_var(name='conv3_kernel',shape=[5,5,64,64],dtype=tf.float32)
	strides=[1,1,1,1]
	conv3=tf.nn.conv2d(norm2,conv3_kernel,strides,padding='SAME')
	conv3_bias=zero_var(name='conv3_bias',shape=[64],dtype=tf.float32)
	conv3_add_bias=tf.nn.bias_add(conv3,conv3_bias)
	relu_conv3=tf.nn.relu(conv3_add_bias)

pool_size=[1,3,3,1]
strides=[1,2,2,1]
pool3=tf.nn.max_pool(relu_conv3,ksize=pool_size,strides=strides,padding='SAME',name='pool_layer3')
norm3=tf.nn.lrn(pool3,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm3')

with tf.variable_scope('conv4') as scope:
	conv4_kernel=truncated_normal_var(name='conv4_kernel',shape=[5,5,64,64],dtype=tf.float32)
	strides=[1,1,1,1]
	conv4=tf.nn.conv2d(norm3,conv4_kernel,strides,padding='SAME')
	conv4_bias=zero_var(name='conv4_bias',shape=[64],dtype=tf.float32)
	conv4_add_bias=tf.nn.bias_add(conv4,conv4_bias)
	relu_conv4=tf.nn.relu(conv4_add_bias)

pool_size=[1,3,3,1]
strides=[1,2,2,1]
pool4=tf.nn.max_pool(relu_conv4,ksize=pool_size,strides=strides,padding='SAME',name='pool_layer4')
norm4=tf.nn.lrn(pool4,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm4')

reshaped_output=tf.reshape(norm4, [-1, 4*4*64])
reshaped_dim=reshaped_output.get_shape()[1].value

#with tf.variable_scope('full1') as scope:
full_weight1=truncated_normal_var(name='full_mult1',shape=[reshaped_dim,1024],dtype=tf.float32)
full_bias1=zero_var(name='full_bias1',shape=[1024],dtype=tf.float32)
full_layer1=tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight1),full_bias1))
full_layer1=tf.nn.dropout(full_layer1,keep_prob)

#with tf.variable_scope('full2') as scope:
full_weight2=truncated_normal_var(name='full_mult2',shape=[1024,256],dtype=tf.float32)
full_bias2=zero_var(name='full_bias2',shape=[256],dtype=tf.float32)
full_layer2=tf.nn.relu(tf.add(tf.matmul(full_layer1,full_weight2),full_bias2))
full_layer2=tf.nn.dropout(full_layer2,keep_prob)

#with tf.variable_scope('full3') as scope:
full_weight3=truncated_normal_var(name='full_mult3',shape=[256,NUM_ANIMALS],dtype=tf.float32)
full_bias3=zero_var(name='full_bias3',shape=[NUM_ANIMALS],dtype=tf.float32)
final_output=tf.add(tf.matmul(full_layer2,full_weight3),full_bias3)
logits=tf.identity(final_output,name='logits')
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels),name='cross_entropy')
#train_step=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
generation_run = tf.Variable(0, trainable=False)
model_learning_rate=tf.train.exponential_decay(LEARNING_RATE,generation_run,NUM_GENS_TO_WAIT,LEARNING_RATE_DECAY,staircase=True)
train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(final_output,1),tf.argmax(labels,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

epochs_completed=0
index_in_epoch = 0
num_examples=x_train.shape[0]

init=tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	def next_batch(batch_size):
		global x_train
		global y_train
		global index_in_epoch
		global epochs_completed
		start = index_in_epoch
		index_in_epoch += batch_size

		if index_in_epoch > num_examples:
			# finished epoch
			epochs_completed += 1
			# shuffle the data
			perm = np.arange(num_examples)
			np.random.shuffle(perm)
			x_train=x_train[perm]
			y_train=y_train[perm]
			# start next epoch
			start = 0
			index_in_epoch = batch_size
			assert batch_size <= num_examples
		end = index_in_epoch
		#return x_train[start:end], y_train[start:end]
		x_output=updateImage(x_train[start:end],True)
		
		return x_output,y_train[start:end]



	# visualisation variables
	train_accuracies = []
	validation_accuracies = []
	x_range = []

	display_step=1

	for i in range(TRAINING_ITERATIONS):

		#get new batch
		batch_xs, batch_ys = next_batch(BATCH_SIZE)

		# check progress on every 1st,2nd,...,10th,20th,...,100th... step
		if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
			train_accuracy=accuracy.eval(feed_dict={x:batch_xs,labels:batch_ys,keep_prob:1.0})
			validation_accuracy=0.0
			for j in range(0,x_val.shape[0]//BATCH_SIZE):
				validation_accuracy+=accuracy.eval(feed_dict={x: x_val[j*BATCH_SIZE:(j+1)*BATCH_SIZE],labels:y_val[j*BATCH_SIZE:(j+1)*BATCH_SIZE],keep_prob:1.0})
			validation_accuracy/=(j+1.0)
			print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
			validation_accuracies.append(validation_accuracy)
			#print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
			train_accuracies.append(train_accuracy)
			x_range.append(i)
			# increase display_step
			if i%(display_step*10) == 0 and i:
				display_step *= 10
		# train on batch
		sess.run(train_step,feed_dict={x:batch_xs,labels:batch_ys,keep_prob:DROPOUT})

	#validation_accuracy = accuracy.eval(feed_dict={x: x_test,y_: y_test,keep_prob: 1.0})
	#print('validation_accuracy => %.4f'%validation_accuracy)
	validation_accuracy=0.0
	for i in range(0,x_val.shape[0]//BATCH_SIZE):
		validation_accuracy+=accuracy.eval(feed_dict={x:x_val[i*BATCH_SIZE:(i+1)*BATCH_SIZE],labels:y_val[i*BATCH_SIZE:(i+1)*BATCH_SIZE],keep_prob:1.0})
	validation_accuracy/=(i+1.0)
	print('validation_accuracy => %.4f'%validation_accuracy)
	saver=tf.train.Saver()
	save_path=saver.save(sess,'./CatsDogs_model')
	sess.close()