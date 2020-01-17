import tensorflow as tf


def weight_var(shape):
    init = tf.truncated_normal(shape, stddev=.04)
    return tf.Variable(init)


def bias_var(shape):
    init = tf.constant(0.0, shape=shape)
    return tf.Variable(init)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def depth_c(x, W, stride):
    return tf.nn.depthwise_conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def Conv_Layer(input_, shape, str, pool, unit_num):
    with tf.name_scope('weights_{}'.format(unit_num)):
        w = weight_var(shape)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        tf.summary.histogram("weights", w)
    with tf.name_scope('bias_{}'.format(unit_num)):
        b = bias_var([shape[3]])
        tf.summary.histogram("bias", b)
    with tf.name_scope('conv_{}'.format(unit_num)):
        h_conv = tf.layers.batch_normalization(tf.nn.elu(conv2d(input_, w, stride=str) +b))
    #tf.summary.histogram("Relu_{}".format(unit_num), h_conv)
    if pool == True:
        with tf.name_scope('pool_{}'.format(unit_num)):
            return max_pool_2x2(h_conv)
    return h_conv


def depth_conv(input_, shape, out_shape, str, unit_num):
    with tf.name_scope('weights_{}'.format(unit_num)):
        w = weight_var(shape)
        tf.summary.histogram("weigths", w)
    with tf.name_scope('bias_{}'.format(unit_num)):
        b = bias_var([shape[3]])
        tf.summary.histogram("bias", b)
    with tf.name_scope('d_conv_{}'.format(unit_num)):
        d_conv = tf.nn.relu(tf.layers.batch_normalization(depth_c(input_, w, str)+ b))

        conv = Conv_Layer(d_conv, out_shape, 1, False, unit_num=unit_num)
    return conv

def Alexnet(self, learn_rate, keep_prob):
    '''
    Function builds Neural network grpah along with accuracy and loss metrics
    :param learn_rate:
    :param keep_prob:
    :return:
    '''

    g = tf.Graph()
    with g.as_default():
        x_ = tf.placeholder(tf.float32, shape=[None, self.im_h, self.im_h, 3], name='input')
        y_e = tf.placeholder(tf.uint8, shape=[None], name='exp_out')

        x_in = tf.divide(x_, tf.constant(255.0, dtype=tf.float32))
        y_e = tf.one_hot(y_e, self.num_labels)

        # Input Layer
        conv1 = Conv_Layer(x_in, [5, 5, 3, 96], str=2, pool=True, unit_num='1')

        # Conv2
        conv2 = Conv_Layer(conv1, shape=[3, 3, 96, 256], str=2, pool=True, unit_num='2')

        # conv 3
        conv3 = Conv_Layer(conv2, shape=[3, 3, 256, 384], str=1, pool=False, unit_num='3')

        # Conv4
        conv4 = Conv_Layer(conv3, shape=[3, 3, 384, 384], str=1, pool=False, unit_num='4')

        # Conv5
        conv5 = Conv_Layer(conv4, shape=[3, 3, 384, 256], str=1, pool=True, unit_num='5')
        # Fully connected(Flattened)
        with tf.name_scope('weight_fc1'):
            w1 = weight_var([8 * 8 * 256, 512])
        with tf.name_scope('bias_fc1'):
            b1 = bias_var([512])
        fc_in = tf.reshape(conv5, [-1, 8 * 8 * 256], name='Flatten')
        with tf.name_scope("Dense_1"):
            fc1 = tf.nn.leaky_relu(tf.matmul(fc_in, w1) + b1)
        # FC2
        with tf.name_scope('weight_fc2'):
            w2 = weight_var([256, 256])
        with tf.name_scope('bias_fc2'):
            b2 = bias_var([256])
        with tf.name_scope("Dense_2"):
            keep_prob = tf.constant(keep_prob)
            fc2 = tf.nn.leaky_relu(tf.matmul(fc1, w2) + b2)
            tf.summary.histogram("Act", fc2)
            fc2_d = tf.nn.dropout(fc2, keep_prob=keep_prob)
        # Output
        with tf.name_scope('weight_fc3'):
            w3 = weight_var([256, self.num_labels])
        with tf.name_scope('bias_fc3'):
            b3 = bias_var([self.num_labels])
        with tf.name_scope("Output"):
            y_out = tf.matmul(fc2_d, w3) + b3
            tf.summary.histogram("Act", y_out)
            tf.identity(y_out, name='y_out')

        # Loss and accuracy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_e, logits=y_out))
        tf.identity(cross_entropy, "loss")
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy, name='train_step')

        loss_sum = tf.summary.scalar('Loss', cross_entropy)
        tf.identity(loss_sum, 'loss_sum')

        arg_max = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_e, 1))
        accuracy = tf.reduce_mean(tf.cast(arg_max, tf.float32), name='accuracy')
        acc_scal = tf.summary.scalar('Accuracy', accuracy)
        tf.identity(acc_scal, 'acc_scal')

    return g


def MobileNet(learn_rate, num_labels, im_h, im_w, labels, test_labels):
    '''
    Function builds Neural network grpah along with accuracy and loss metrics
    :param learn_rate:
    :param keep_prob:
    :return:
    '''

    g = tf.Graph()
    with g.as_default():
        y_hot =tf.one_hot(labels, num_labels, name="y_hot")
        y_hval = tf.one_hot(test_labels, num_labels, name="y_hval")
        
        x_ = tf.placeholder(tf.float32, shape=[None, im_h, im_h, 3], name='input')
        y_e = tf.placeholder(tf.uint8, shape=[None,num_labels], name='exp_out')

        x_in = tf.divide(x_, tf.constant(255.0, dtype=tf.float32))
        #1y_e = tf.one_hot(y_e, num_labels)
        #tf.summary.histogram("onehot", y_e)
        #im_ex = tf.summary.image("input_image", x_, 2)
       # tf.identity(im_ex, "im_ex")
        # Input Layer
        conv_1 = Conv_Layer(x_in, [5, 5, 3, 32], 2, False, '1')

        d_conv2 = depth_conv(conv_1, [3, 3, 32, 1], [1, 1, 32, 64], 1, '2')

        d_conv3 = depth_conv(d_conv2, [3, 3, 64, 1], [1, 1, 64, 128], 2, '3')

        d_conv4 = depth_conv(d_conv3, [3, 3, 128, 1], [1, 1, 128, 128], 1, '4')

        d_conv5 = depth_conv(d_conv4, [3, 3, 128, 1], [1, 1, 128, 256], 2, '5')

        d_conv6 = depth_conv(d_conv5, [3, 3, 256, 1], [1, 1, 256, 256], 1, '6')

        d_conv7 = depth_conv(d_conv6, [3, 3, 256, 1], [1, 1, 256, 512], 2, '7')

        d_conv8 = depth_conv(d_conv7, [3, 3, 512, 1], [1, 1, 512, 512], 1, '8')

        d_conv9 = depth_conv(d_conv8, [3, 3, 512, 1], [1, 1, 512, 512], 1, '9')

        d_conv10 = depth_conv(d_conv9, [3, 3, 512, 1], [1, 1, 512, 512], 1, '10')

        d_conv11 = depth_conv(d_conv10, [3, 3, 512, 1], [1, 1, 512, 512], 1, '11')

        d_conv12 = depth_conv(d_conv11, [3, 3, 512, 1], [1, 1, 512, 1024], 2, '12')

        d_conv13 = depth_conv(d_conv12, [3, 3, 1024, 1], [1, 1, 1024, 1024], 2, '8')
        # pool_14 = tf.nn.avg_pool(d_conv13, ksize=[1, 7, 7, 1],strides=[1,4,4,1], padding='SAME', name='pool_14')
        pool_14 = tf.layers.average_pooling2d(d_conv13, [im_h / 56, im_w / 56], 1)
        pool_14 = tf.reshape(pool_14, [-1, 1024])
        with tf.name_scope('FC_weights'):
            wfc = weight_var([1024, num_labels])
            tf.summary.histogram("Weightfc", wfc)
            bfc = bias_var([num_labels])
            tf.summary.histogram("biasfc", bfc)
        with tf.name_scope('Output'):
            y_out = tf.matmul(pool_14, wfc) + bfc
            tf.identity(y_out,name="y_out")
        weight_sum = tf.summary.merge_all()
        tf.identity(weight_sum, "w_sum")
        # Loss and accuracy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_e, logits=y_out))
        tf.identity(cross_entropy, "loss")
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

        loss_sum = tf.summary.scalar('Loss', cross_entropy)
        tf.identity(loss_sum, 'loss_sum')

        arg_max = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_e, 1))
        accuracy = tf.reduce_mean(tf.cast(arg_max, tf.float32), name='accuracy')
        acc_scal = tf.summary.scalar('Accuracy', accuracy)
        tf.identity(acc_scal, 'acc_scal')

    return g

def ConvNet(learn_rate, num_labels, im_h, im_w, labels, test_labels):
    '''
    Function builds Neural network grpah along with accuracy and loss metrics
    :param learn_rate:
    :param keep_prob:
    :return:
    '''

    g = tf.Graph()
    with g.as_default():
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.08)
        y_hot = tf.one_hot(labels, num_labels, name="y_hot")
        y_hval = tf.one_hot(test_labels, num_labels, name="y_hval")
        x_ = tf.placeholder(tf.float32, shape=[None, im_h, im_w, 3], name='input')
        y_e = tf.placeholder(tf.uint8, shape=[None, num_labels], name='exp_out')

        x_in = tf.divide(x_, tf.constant(255.0, dtype=tf.float32))


        #im_x = tf.summary.image("image",x_in,10)
        #tf.identity(im_x, "im_ex")
        # Input Layer
        conv1 = Conv_Layer(x_in, [5, 5, 3, 32], str=1, pool=True, unit_num='1')

        # Conv2
        conv2 = Conv_Layer(conv1, shape=[3, 3, 32, 64], str=1, pool=True, unit_num='2')

        # conv 3
        conv3 = Conv_Layer(conv2, shape=[3, 3, 64, 128], str=1, pool=False, unit_num='3')

        conv4 = Conv_Layer(conv3, shape=[3, 3, 128, 128], str=1, pool=False, unit_num='4')

        conv5 = Conv_Layer(conv4, shape=[3, 3, 128, 64], str=1, pool=True, unit_num='5')

        # Fully connected(Flattened)
        with tf.name_scope('weight_fc1'):
            w1 = weight_var([int((im_h/8) * (im_w/8) * 64), 512])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w1)
        with tf.name_scope('bias_fc1'):
            b1 = bias_var([512])
        fc_in = tf.reshape(conv5, [-1, int((im_h/8) * (im_w/8) * 64)], name='Flatten')
        with tf.name_scope("Dense_1"):
            fc1 = tf.nn.elu(tf.matmul(fc_in, w1) + b1)

        with tf.name_scope("weight_fc2"):
            w2=weight_var([512, 512])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w2)
            #tf.summary.histogram("weights", w2)
        with tf.name_scope("bias_fc1"):
            b2=bias_var([512])
        with tf.name_scope("Dense2"):
            fc2 = tf.nn.elu(tf.matmul(fc1,w2)+ b2)
        keep_prob = tf.placeholder(tf.float32, name="keep_p")
        drop_fc = tf.nn.dropout(fc2, keep_prob)

        # Output
        with tf.name_scope('weight_fc3'):
            w3 = weight_var([512, num_labels])
            print(w3)
            tf.identity(w3, "weighth1")
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w3)
            tf.summary.histogram("Weightfc", w3)
        with tf.name_scope('bias_fc3'):
            b3 = bias_var([num_labels])
        with tf.name_scope("Output"):
            y_out = tf.matmul(drop_fc, w3) + b3
            print(y_out)
            tf.identity(y_out, name='y_out')

        weight_sum = tf.summary.merge_all()
        tf.identity(weight_sum, "w_sum")

        # Loss and accuracy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_e, logits=y_out))
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        cross_entropy = cross_entropy + reg_term

        tf.identity(cross_entropy, "loss")
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

        loss_sum = tf.summary.scalar('Loss', cross_entropy)
        tf.identity(loss_sum, 'loss_sum')

        l_c = tf.placeholder(tf.float32, name="l_c")
        loss_cross = tf.summary.scalar("Loss_val", l_c)
        tf.identity(loss_cross, 'loss_cross')

        arg_max = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_e, 1))
        accuracy = tf.reduce_mean(tf.cast(arg_max, tf.float32), name='accuracy')
        acc_scal = tf.summary.scalar('Accuracy', accuracy)
        tf.identity(acc_scal, 'acc_scal')

    return g