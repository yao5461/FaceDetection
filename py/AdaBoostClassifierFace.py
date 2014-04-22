__author__ = 'yao'

import re
import math
import time
import numpy
import matplotlib.pyplot as plt


class HaarFeature:
    """
    The class of haar-like feature
    """
    def __init__(self, type, x, y, w, h, x_scaled, y_scaled):
        """
        Initial the feature and describe it.
        The type of feature: 2h, 2v, 3h, 3v, 4c
        The left-top position of feature window is (x,y)
        The size of feature window is w * h = (base_w * x_scaled) * (base_y * y_scaled)
        """
        self.type = type
        self.x = y
        self.y = x
        self.w = w
        self.h = h
        self.axis_x_scaled = x_scaled
        self.axis_y_scaled = y_scaled


    def cal_feature_value(self, integral_img):
        """
        Calculate the value of the feature and then return it.
        @param integral_img: the integral image matrix of a image
        """
        #!!!Note: There is a mistake, the mapping is: (x, y) -> (h, w)
        x_one_scaled = self.axis_y_scaled
        x_two_scaled = self.axis_y_scaled * 2
        y_one_scaled = self.axis_x_scaled
        y_two_scaled = self.axis_x_scaled * 2

        feature_value = 0
        if self.type == '2h':  #horizontal two-rectangle feature: 6 array reference
            feature_value = integral_img[self.x+x_one_scaled, self.y+y_two_scaled] \
                            - integral_img[self.x, self.y+y_two_scaled] \
                            - 2 * integral_img[self.x+x_one_scaled, self.y+y_one_scaled] \
                            + 2 * integral_img[self.x, self.y+y_one_scaled] \
                            + integral_img[self.x+x_one_scaled, self.y] \
                            - integral_img[self.x, self.y]
            feature_value *= -1
        elif self.type == '2v': #vertical two-rectangle feature: 6 array reference
            feature_value = integral_img[self.x+x_two_scaled, self.y+y_one_scaled] \
                            - 2 * integral_img[self.x+x_one_scaled, self.y+y_one_scaled] \
                            - integral_img[self.x+x_two_scaled, self.y] \
                            + 2 * integral_img[self.x+x_one_scaled, self.y] \
                            + integral_img[self.x, self.y+y_one_scaled] \
                            - integral_img[self.x, self.y]
            feature_value *= -1
        elif self.type == '3h': #horizontal three-rectangle feature: 8 array reference
            y_three_scaled = self.axis_x_scaled * 3
            feature_value = 2 * integral_img[self.x+x_one_scaled, self.y+y_one_scaled] \
                            - 2 * integral_img[self.x, self.y+y_one_scaled] \
                            - integral_img[self.x+x_one_scaled, self.y] \
                            + integral_img[self.x, self.y] \
                            + integral_img[self.x+x_one_scaled, self.y+y_three_scaled] \
                            - integral_img[self.x, self.y+y_three_scaled] \
                            - 2 * integral_img[self.x+x_one_scaled, self.y+y_two_scaled]\
                            + 2 * integral_img[self.x, self.y+y_two_scaled]
        elif self.type == '3v': #vertical three-rectangle feature: 8 array reference
            x_three_scaled = self.axis_y_scaled * 3
            feature_value = 2 * integral_img[self.x+x_one_scaled, self.y+y_one_scaled] \
                            - integral_img[self.x, self.y+y_one_scaled] \
                            - 2 * integral_img[self.x+x_one_scaled, self.y] \
                            + integral_img[self.x, self.y] \
                            + integral_img[self.x+x_three_scaled, self.y+y_one_scaled] \
                            - 2 * integral_img[self.x+x_two_scaled, self.y+y_one_scaled] \
                            - integral_img[self.x+x_three_scaled, self.y] \
                            + 2 * integral_img[self.x+x_two_scaled, self.y]
        elif self.type == '4c': #diagonal four-rectangle feature: 9 array reference
            feature_value = 2 * integral_img[self.x+x_two_scaled, self.y+y_one_scaled] \
                            - 4 * integral_img[self.x+x_one_scaled, self.y+y_one_scaled] \
                            - integral_img[self.x+x_two_scaled, self.y] \
                            + 2 * integral_img[self.x+x_one_scaled, self.y] \
                            + 2 * integral_img[self.x+x_one_scaled, self.y+y_two_scaled] \
                            - integral_img[self.x, self.y+y_two_scaled] \
                            + 2 * integral_img[self.x, self.y+y_one_scaled] \
                            - integral_img[self.x, self.y] \
                            - integral_img[self.x+x_two_scaled, self.y+y_two_scaled]
            feature_value *= -1
        else:
            print "Error feature type!"

        return feature_value


    def print_feature_info(self):
        """
        """
        info = self.type + ':(' + str(self.x) + ',' + str(self.y) + ')(' +str(self.w) + ',' + str(self.h) + ')'
        return info


class ImgSample:
    """
    The class of sample
    """
    def __init__(self, img_path, label, widget):
        """
        Initial the information of sample
        @param img_path: the path of sample image
        @param label: 1 or 0 presents face or non-face
        @param widget: the widget of this sample
        """
        #self.img_path = img_path
        self.label = label
        self.widget = widget

        image_content = self.read_pgm_img(img_path, byteorder='<')
        #image_content = numpy.array(Image.open(img_path))
        #plt.imshow(image_content, plt.cm.gray)
        #plt.show()
        #print image_content
        row, col = image_content.shape
        self.integral_img = numpy.zeros((row+1, col+1), dtype=numpy.int)
        self.cal_integral_img(image_content)
        #print self.integral_img


    def read_pgm_img(self, img_path, byteorder = '>'):
        """
        Return image data from a raw PGm file as numpy array
        @param img_path: the path of the sample image
        """
        with open(img_path, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % img_path)

        return numpy.frombuffer(buffer,
                                dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                                count=int(width)*int(height),
                                offset=len(header)
                                ).reshape((int(height), int(width)))


    def cal_integral_img(self, img_content):
        """
        Calculate the integral image representation of raw image
        @param img_content: the data content of the raw image
        """
        row, col = img_content.shape
        self.integral_img[0, 0:] = 0
        self.integral_img[0:, 0] = 0

        for i in range(row):
            for j in range(col):
                self.integral_img[i+1, j+1] = self.integral_img[i, j+1] + self.integral_img[i+1, j] \
                                              - self.integral_img[i,j] + img_content[i,j]


class WeakClassifier:
    """
    The class of one weak classifier with one haar-like feature
    """
    def __init__(self, feature):
        """
        Initial the information of the weak classifier
        @param feature: the feature belongs to this weak classifier
        """
        self.feature = feature
        #the value of alpha
        self.widget = 0.0
        self.threshold = 0.0
        self.flag = 1


    def predict(self, integral_img):
        """
        Predict the result by comparing the feature value with the threshold
        """
        value = self.feature.cal_feature_value(integral_img)
        return 1 if (value * self.flag) < (self.threshold * self.flag) else 0


class AdaBoostFace:
    """
    Training a strong classifier for face detection using ada-boost algorithm
    """
    def __init__(self, config_file):
        """
        class initial
        @param config_file: the path of the file which stores the configuration formation
        """
        self.config_training(config_file)
        self.read_sample_data()
        self.training_init()


    def cal_one_attribute(self, classifier):
        """
        Calculate the value of this feature in each sample.
        Determine the minimum of the error and the threshold of this classifier with feature.
        """
        #the feature belongs to this classifier
        feature = classifier.feature
        #all feature values of each sample.  data:<id, value>
        feature_values = {}
        index = 0
        #pass though all samples to calculate the feature value
        for sample in self.all_samples:
            value = feature.cal_feature_value(sample.integral_img)
            feature_values[index] = value
            index += 1

        #sorted all samples by feature value
        sorted_feature_value_list = sorted(feature_values.items(), key=lambda item: item[1])

        #calculate the minimum number of error, and determine the current threshold(feature value)
        below_positive_sample_widget_sum = 0.0
        below_negative_sample_widget_sum = 0.0
        res_info = []

        for (index, value) in sorted_feature_value_list:
            temp1 = below_positive_sample_widget_sum + (self.negative_sample_widget_sum - below_negative_sample_widget_sum)
            temp2 = below_negative_sample_widget_sum + (self.positive_sample_widget_sum - below_positive_sample_widget_sum)

            e = min(temp1, temp2)
            flag = 1 if temp1 > temp2 else -1
            res_info.append((e, value, flag))

            if self.all_samples[index].label == 1:
                below_positive_sample_widget_sum += self.all_samples[index].widget
            else:
                below_negative_sample_widget_sum += self.all_samples[index].widget

        (res_e, res_value, res_flag) = min(res_info)
        classifier.threshold = res_value
        classifier.flag = res_flag

        return res_e


    def training_one_weak_classifier(self):
        """
        Choose a weak classifier in a feature set.
        Determine the threshold to form a classifier.
        """
        res_info = []

        for classifier in self.weak_classifiers:
            error_num = self.cal_one_attribute(classifier)
            error_num = abs(error_num)
            res_info.append((error_num, classifier))

        (res_error, res_classifier) = min(res_info)

        return res_error, res_classifier


    def do_adaboost_training(self):
        """
        The progress of training using AdaBoost algorithm.
        """
        for iter in range(self.max_iteration):
            #choose a current best weak classifier
            [err, classifier] = self.training_one_weak_classifier()

            #update the widget of classifier chosen
            classifier.widget = math.log( (1 - err) / err )
            self.classifiers.append(classifier)

            #test print
            print "iteration ", iter, ": "
            print 'test error: ', err
            print "feature: [" + classifier.feature.print_feature_info() + "], flag: " + str(classifier.flag)
            print "widget: " + str(classifier.widget) + ", threshold: " + str(classifier.threshold)

            #update the widget of each sample
            widget_sum = 0.0
            self.positive_sample_widget_sum = 0.0
            self.negative_sample_widget_sum = 0.0

            for sample in self.all_samples:
                #classify this sample using current classifier
                category = classifier.predict(sample.integral_img)
                #classified correctly modify the widget
                if category == sample.label:
                    sample.widget = sample.widget * (err / (1 - err))

                widget_sum += sample.widget

                if sample.label == 1:
                    self.positive_sample_widget_sum += sample.widget
                else:
                    self.negative_sample_widget_sum += sample.widget

            #normizal the widget of each sample
            for sample in self.all_samples:
                sample.widget = sample.widget / widget_sum


    def config_training(self, config_file):
        """
        Read some configuration parameters from file
        """
        #configuration information
        config_items = {}

        fp = open(config_file)
        for line in fp.readlines():
            item, value = line.strip('\r\n').split(': ')
            config_items[item] = value
            #print item, ' ', value
        fp.close()

        self.positive_sample_num = int(config_items['PositiveSamples'])
        self.negative_sample_num = int(config_items['NegativeSamples'])
        self.max_iteration = int(config_items['MaxIteration'])
        self.sample_type = config_items['SampleType']
        self.sample_width = int(config_items['SampleWidth'])
        self.sample_height = int(config_items['SampleHeight'])
        self.positive_sample_path = config_items['PositiveSamplePath']
        self.negative_sample_path = config_items['NegativeSamplePath']

        #the set of all samples
        self.all_samples = []

        #the sum widget of positive samples and negative samples
        self.positive_sample_widget_sum = 0.5
        self.negative_sample_widget_sum = 0.5

        #all weak classifiers
        self.weak_classifiers = []
        #the classifiers chosen
        self.classifiers = []


    def read_sample_data(self):
        """
        Read training data and set the widget value.
        """
        widget = 1.0 / (2 * self.positive_sample_num)

        #read positive samples images
        for i in range(self.positive_sample_num):
            img_path = self.positive_sample_path + str(i+1).zfill(5) + '.' + self.sample_type
            self.all_samples.append( ImgSample(img_path, 1, widget) )

        widget = 1.0 / (2 * self.negative_sample_num)

        #read negative samples images
        for i in range(self.negative_sample_num):
            img_path = self.negative_sample_path + str(i+1).zfill(5) + '.' + self.sample_type
            self.all_samples.append( ImgSample(img_path, 0, widget) )

        print 'sample number: ', len(self.all_samples)


    def training_init(self):
        """
        Define the weak classifiers
        """
        self.generate_weak_classifiers(2, 1, '2h')
        self.generate_weak_classifiers(1, 2, '2v')
        self.generate_weak_classifiers(3, 1, '3h')
        self.generate_weak_classifiers(1, 3, '3v')
        self.generate_weak_classifiers(2, 2, '4c')

        print "weak classifier num: ", len(self.weak_classifiers)


    def generate_weak_classifiers(self, w, h, type):
        """
        Generate weak classifiers with one type feature. The size of feature can be scaled.
        w: base width of feature window
        h: base height of feature window
        type: feature type
        """
        X = self.sample_width / w
        Y = self.sample_height / h

        for i in range(1, X+1):
            for j in range(1, Y+1):
                for x in range(0, self.sample_width - i * w + 1):
                    for y in range(0, self.sample_height - j * h + 1):
                        temp_classifier = WeakClassifier(HaarFeature(type, x, y, i*w, i*h, i, j))
                        self.weak_classifiers.append(temp_classifier)


    def predict(self, integral_img):
        """
        Check the result strong classifier
        """
        category = 0.0
        widget_sum = 0.0

        for classifier in self.classifiers:
            category += classifier.widget * classifier.predict(integral_img)
            widget_sum += classifier.widget

        print category, widget_sum
        if category >= (widget_sum / 2.0):
            return 1
        else:
            return 0


testAda = AdaBoostFace("config.txt")
print 'start time: ', time.strftime('%H:%M:%S', time.localtime(time.time()))
testAda.do_adaboost_training()
print 'finished time: ', time.strftime('%H:%M:%S', time.localtime(time.time()))


def test_strong_classifier():
    widget = 0.0
    all_test = []

    for i in range(5):
        img_path = testAda.positive_sample_path + str(i+1).zfill(5) + '.' + testAda.sample_type
        all_test.append( ImgSample(img_path, 1, widget) )
    for i in range(5):
        img_path = testAda.negative_sample_path + str(i+1).zfill(5) + '.' + testAda.sample_type
        all_test.append( ImgSample(img_path, 0, widget) )

    for i in range(5):
        img_path = testAda.positive_sample_path + str(i+251).zfill(5) + '.' + testAda.sample_type
        all_test.append( ImgSample(img_path, 1, widget) )
    for i in range(5):
        img_path = testAda.negative_sample_path + str(i+551).zfill(5) + '.' + testAda.sample_type
        all_test.append( ImgSample(img_path, 0, widget) )

    for img_sample in all_test:
        pre_label = testAda.predict(img_sample.integral_img)
        print 'raw label: ', img_sample.label, 'pre label: ', pre_label

test_strong_classifier()

#fp = open('samples/negative/B1_00390.pgm')
#for line in fp:
#    print line

