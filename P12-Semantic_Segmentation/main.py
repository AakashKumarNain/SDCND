import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
	warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

	
#############################################################################################################################

def load_vgg(sess, vgg_path):
	"""
	Load Pretrained VGG Model into TensorFlow.
	:param sess: TensorFlow Session
	:param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
	:return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
	"""
	# TODO: Implement function
	#   Use tf.saved_model.loader.load to load the model and weights
	vgg_tag = 'vgg16'
	vgg_model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

	graph = tf.get_default_graph()

	# Input tensor
	input_tensor = graph.get_tensor_by_name('image_input:0')
	# Dropout 
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	layer_3_out = graph.get_tensor_by_name('layer3_out:0')
	layer_4_out = graph.get_tensor_by_name('layer4_out:0')
	layer_7_out = graph.get_tensor_by_name('layer7_out:0')

	return input_tensor, keep_prob, layer_3_out, layer_4_out, layer_7_out
tests.test_load_vgg(load_vgg, tf)

#############################################################################################################################3


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	"""
	Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
	:param vgg_layer7_out: TF Tensor for VGG Layer 3 output
	:param vgg_layer4_out: TF Tensor for VGG Layer 4 output
	:param vgg_layer3_out: TF Tensor for VGG Layer 7 output
	:param num_classes: Number of classes to classify
	:return: The Tensor for the last layer of output
	"""
	# Initialization for kernel 
	init = tf.truncated_normal_initializer(stddev = 0.001)
	
	
	def add_conv_1x1(x, num_classes, init = init):
		return tf.layers.conv2d(x, num_classes, 1, padding = 'same', kernel_initializer = init)

	def upsample(x, num_classes, depth, strides, init = init):
		return tf.layers.conv2d_transpose(x, num_classes, depth, strides, padding = 'same', kernel_initializer = init)

	# Add 1x1 conv layers
	layer_7_1x1 = add_conv_1x1(vgg_layer7_out, num_classes)
	layer_4_1x1 = add_conv_1x1(vgg_layer4_out, num_classes)
	layer_3_1x1 = add_conv_1x1(vgg_layer3_out, num_classes)

	upsample1 = upsample(layer_7_1x1, num_classes, 5, 2)
	layer1 = tf.layers.batch_normalization(upsample1)
	layer1 = tf.add(layer1, layer_4_1x1)

	upsample2 = upsample(layer1, num_classes, 5, 2)
	layer2 = tf.layers.batch_normalization(upsample2)
	layer2 = tf.add(layer2, layer_3_1x1)

	return upsample(layer2, num_classes, 14, 8)

tests.test_layers(layers)

###############################################################################################################################


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
	"""
	Build the TensorFLow loss and optimizer operations.
	:param nn_last_layer: TF Tensor of the last layer in the neural network
	:param correct_label: TF Placeholder for the correct label image
	:param learning_rate: TF Placeholder for the learning rate
	:param num_classes: Number of classes to classify
	:return: Tuple of (logits, train_op, cross_entropy_loss)
	"""
	# TODO: Implement function
	logits = tf.reshape(nn_last_layer, (-1, num_classes))
	correct_label = tf.reshape(correct_label, (-1, num_classes))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	return logits, optimizer, loss

tests.test_optimize(optimize)
############################################################################################################################


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
			 correct_label, keep_prob, learning_rate):
	"""
	Train neural network and print out the loss during training.
	:param sess: TF Session
	:param epochs: Number of epochs
	:param batch_size: Batch size
	:param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
	:param train_op: TF Operation to train the neural network
	:param cross_entropy_loss: TF Tensor for the amount of loss
	:param input_image: TF Placeholder for input images
	:param correct_label: TF Placeholder for label images
	:param keep_prob: TF Placeholder for dropout keep probability
	:param learning_rate: TF Placeholder for learning rate
	"""

	steps = 0
	for epoch_i in range(epochs):
		avg_cost = 0
		for images, labels in get_batches_fn(batch_size):
			_, loss = sess.run([train_op, cross_entropy_loss],
			feed_dict={input_image: images, correct_label: labels, keep_prob:0.5, learning_rate:0.001})
			avg_cost+=loss

		print("Epoch {}/{}...".format(epoch_i+1, epochs),"Training Loss: {:.4f}...".format(avg_cost/batch_size))    

tests.test_train_nn(train_nn)

##############################################################################################################################3

def run():
	num_classes = 2
	image_shape = (160, 576)
	data_dir = './data'
	runs_dir = './runs'
	tests.test_for_kitti_dataset(data_dir)
	epochs = 30
	batch_size = 8

	# Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(data_dir)

	# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
	# You'll need a GPU with at least 10 teraFLOPS to train on.
	#  https://www.cityscapes-dataset.com/

	with tf.Session() as sess:
		# Path to vgg model
		vgg_path = os.path.join(data_dir, 'vgg')
		# Create function to get batches
		get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

		# TODO: Build NN using load_vgg, layers, and optimize function
		input_image, keep_prob, layer_3_out, layer_4_out, layer_7_out = load_vgg(sess, vgg_path)
		output = layers(layer_3_out, layer_4_out, layer_7_out, num_classes)

		correct_label = tf.placeholder(tf.float32, shape = [None, None, None, num_classes])
		learning_rate = tf.placeholder(tf.float32)

		logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
		sess.run(tf.global_variables_initializer())
		train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,correct_label, keep_prob, learning_rate)
		helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


#####################################################################################################################################

if __name__ == '__main__':
	run()


