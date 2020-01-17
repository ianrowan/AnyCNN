from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import json
from numpy import genfromtxt
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time
import Backend_Models as mod
import matplotlib
import matplotlib.pyplot as plt


#this is a cloud server test
# Class with fundtions to get full groups of images along with entire lists of keywords
class GetImages:

    def __init__(self):
        self


    def get_soup(self, url, header):
        return BeautifulSoup(urlopen(Request(url, headers=header)), 'html.parser')

    def img_scrape(self, name, num, save):
        '''
        Scrapes google images for keyword for 'num' images. Outputs full set to folder
        :param name:
        :param num:
        :param save:
        :return:
        '''
        query = name  # raw_input(args.search)
        max_images = int(num)
        save_directory = save
        image_type = "Action"
        query = query.split()
        query = '+'.join(query)
        url = "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"
        header = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        soup = self.get_soup(url, header)
        ActualImages = []  # contains the link for Large original images, type of  image
        for a in soup.find_all("div", {"class": "rg_meta"}):
            link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            ActualImages.append((link, Type))
        for i, (img, Type) in enumerate(ActualImages[0:max_images]):
            try:
                req = Request(img, headers={
                    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"})
                raw_ = urlopen(req)
                raw_img = raw_.read()
                if len(Type) == 0:
                    f = open(os.path.join(save_directory, name.replace(" ", "") + "_" + str(i) + ".jpg"), 'wb')
                else:
                    f = open(os.path.join(save_directory, name.replace(" ", "") + "_" + str(i) + "." + Type), 'wb')
                f.write(raw_img)
                f.close()
            except Exception as e:
                print("could not load : " + img)
                print(e)

    def all_images(self,cat_path, num_imgs, save_path):
        '''
        Takes input of csv file, turns it into array of style len(keyword, label).
        Then iterates over each keyword to scrape images and out put to file named after keyword.
        :param cat_path:
        :param num_imgs:
        :param save_path:
        :return:
        '''
        kwds = cat_path
        num = num_imgs
        queries = genfromtxt(kwds, delimiter=',', dtype=str)

        for i in tqdm(range(len(queries))):
            save_path1 = save_path+queries[i][1]
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            self.img_scrape(queries[i][0], num, save_path1)


class NeuralNet:
    '''
    Class includes functions to preprocess, build and train a convulutional neural network.
    '''

    def __init__(self, image_h, image_w, val_split, save_path):
        self.im_h = image_h
        self.im_w = image_w
        self.split = val_split

        train_path = save_path
        train_labels = next(os.walk(train_path))[1]
        print(train_labels)
        self.num_labels=int(len(train_labels))
        train_im = []
        train_label = []

        print('Pre-processing image set.......')
        for i in range(self.num_labels):
            print("\nCategory {} images processing......".format(i+1)+"\n")
            for f in tqdm(os.listdir(train_path + train_labels[i])):
                try:
                    file = train_path + train_labels[i] + "/" + f
                    img = imread(file)[:, :, :3]
                    img = resize(img, (self.im_h, self.im_w), mode='constant', preserve_range=True)
                    train_im.append(img)
                    train_label.append(int(train_labels[i])-1)
                except OSError as e:
                    pass
                except IndexError as e:
                    pass
                except ValueError as e:
                    pass

        train_x = np.asarray(train_im, dtype=np.uint8)
        train_y1 = np.asarray(train_label, dtype=np.uint8)
        print(train_y1)
        self.x_in, self.x_val, self.y_in, self.y_val = train_test_split(train_x, train_y1, test_size=self.split)
        print('Pre-processing Complete!')

        plt.hist(self.y_val)
        plt.show()





    def run_session(self, learn_rate, batch_size, epochs, keep_prob, early_stop):
        '''
        Runs given tensorflow graph and outputs data based on input params.
        :param learn_rate:
        :param batch_size:
        :param epochs:
        :param keep_prob:
        :param early_stop:
        :return:
        '''

        def next_batch(num, data, labels):
            '''
            Return a total of `num` random samples and labels.
            '''
            idx = np.arange(0, len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            data_shuffle = [data[i] for i in idx]
            labels_shuffle = [labels[i] for i in idx]

            return np.asarray(data_shuffle), np.asarray(labels_shuffle)

        graph = mod.ConvNet(learn_rate, self.num_labels, self.im_h, self.im_w, self.y_in, self.y_val)
        print("Currently running MobileNet")
        print("Number of categories found: {}".format(self.num_labels))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            wirter = tf.summary.FileWriter('/home/ian/Desktop/Cars', sess.graph)
            timer = 0
            train_start = time()
            losses = np.zeros([30000, 1])
            pat_count = 0
            j = 0
            y_hot = sess.run("y_hot:0")
            y_hval = sess.run("y_hval:0")
            for i in range(epochs):
                time_start = time()
                batch = next_batch(batch_size, self.x_in, y_hot)

                patience = 5
                loss_delt = .0001
                feed = {"input:0":batch[0], "exp_out:0":batch[1], "keep_p:0": 1.0}
                #if i%40==0:
                    #out = tf.nn.softmax(graph.get_tensor_by_name("Output/add:0"))
                    #print(sess.run("one_hot:0", feed_dict=feed))
                if i % 10 == 0:
                    train_accuracy = sess.run("accuracy:0",feed_dict=feed)
                    loss_s = sess.run("loss_sum:0",feed_dict=feed)
                    #acc_s = sess.run("acc_scal",feed_dict={"accuracy:0": train_accuracy})
                    wirter.add_summary(loss_s, i)
                    #wirter.add_summary(acc_s, i)
                    print('========================SUMMARY REPORT=============================')
                    print('step %d, accuracy: %g' % (i, train_accuracy))
                    print('Estimated Time Remaining = ' + str(round((epochs - i) * (timer / 60) / 60, 2)) + ' Hours')
                    print('===================================================================')

                if i % 75 == 0:

                    l = np.zeros([len(self.x_val), 1], np.float32)
                    acc = np.zeros([len(self.x_val), 1], np.float32)
                    for ii in range(len(l)):
                        x = np.zeros([1, self.im_h, self.im_w, 3], dtype='float32')
                        x[0][:][:][:] = self.x_val[ii]
                        yi = np.zeros([1, self.num_labels], dtype='float32')
                        yi[0] = y_hval[ii]
                        val_loss = sess.run("loss:0", feed_dict={"input:0": x, "exp_out:0": yi, "keep_p:0": 1.0})
                        val_acc = sess.run("accuracy:0", feed_dict={"input:0": x, "exp_out:0": yi, "keep_p:0": 1.0})
                        l[ii] = val_loss
                        acc[ii] = val_acc
                    loss_mean = np.mean(l)
                    chart_val = sess.run("loss_cross:0", feed_dict={"l_c:0":loss_mean})
                    wirter.add_summary(chart_val,i)
                    losses[j] = loss_mean
                    acc_mean  =np.mean(acc)

                    #im_sess = sess.run("im_ex:0", feed_dict=feed)
                    #wirter.add_summary(im_sess, i)


                    current_delt = losses[j - 1] - losses[j]

                    if early_stop == True:
                        if i > 0 and current_delt > loss_delt:
                            pat_count = 0
                        else:
                            pat_count += 1

                        if pat_count > patience:
                            print('Early Stop Initiated at step %g' % i)
                            break
                    j += 1

                    print('================================================================')
                    print('Valadation Accuracy %g' % acc_mean)
                    print('Validation Loss %g' % loss_mean)
                    print('Loss delta %g' % current_delt)
                    print('Stop Patience %g of ' % pat_count + str(patience))
                    print('================================================================')

                sess.run("Adam", feed_dict={"input:0":batch[0], "exp_out:0":batch[1], "keep_p:0": .4})
                hist = sess.run("w_sum:0", feed_dict=feed)
                wirter.add_summary(hist, i)
                time_stop = time()
                timer = time_stop - time_start
                if i == epochs-200:
                    w3 = sess.run("weight_fc3/Variable:0")
                    print(w3[1])

                print('Step: ' + str(i) + ' Epoch Time: ' + str(round(timer, 2)) + ' Secs.' + ' Time elapsed: ' +
                      str(round((time_stop - train_start) / 60, 2)) + ' Mins.' + str(
                    round((i / epochs) * 100, 1)) + " % complete")

            w3 = sess.run("weight_fc3/Variable:0")
            print(w3[1])
            inpt = input("Specify Model Savepath/model.ckpt: ")
            saver = tf.train.Saver(tf.global_variables(), name="saver")
            s_path = saver.save(sess, inpt)
            print("Model saved in {}".format(s_path))
            sess.close()


















