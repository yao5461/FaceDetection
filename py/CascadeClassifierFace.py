__author__ = 'yao'

import re
import math
import time
import numpy
import pp
from PIL import Image
from xml.dom.minidom import Document

# tuple of all parallel python servers to connect with
# ppservers = ("*",)
ppservers = ()

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

        #image_content = self.read_pgm_img(img_path, byteorder='<')
        image_content = numpy.array(Image.open(img_path))
        #plt.imshow(image_content, plt.cm.gray)
        #plt.show()
        #print image_content
        row, col = image_content.shape
        self.integral_img = numpy.zeros((row+1, col+1), dtype=numpy.int)
        self.cal_integral_img(image_content)
        #print self.integral_img


    """
    def read_pgm_img(self, img_path, byteorder = '>'):
    """
    """
    Return image data from a raw PGm file as numpy array
    @param img_path: the path of the sample image
    """
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
    """


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


class StrongClassifier:
    """
    The class of one strong classifier consist of several weak classifiers.
    """
    def __init__(self):
        """
        """
        self.num_classifier = 0
        self.classifiers = []
        self.threshold = 0.5


    def predict(self, integral_img):
        """
        Predict the result.
        """
        category = 0.0
        widget_sum = 0.0

        for classifier in self.classifiers:
            category += classifier.widget * classifier.predict(integral_img)
            widget_sum += classifier.widget

        #print category, widget_sum
        if category >= (widget_sum * self.threshold):
            return 1
        else:
            return 0


class AdaBoostFace:
    """
    Training a strong classifier for face detection using ada-boost algorithm
    """
    def __init__(self, sample_width, sample_height, cpu_num = 0):
        """
        class initial
        """
        self.sample_width = sample_width
        self.sample_height = sample_height

        #the set of all samples
        self.all_samples = []

        #the sum widget of positive samples and negative samples
        self.positive_sample_widget_sum = 0.5
        self.negative_sample_widget_sum = 0.5

        #feature number or the number of iteration
        self.max_iteration = 1
        self.iter = 1

        #number of cpu number
        self.cpu_number = cpu_num

        #all weak classifiers
        self.weak_classifiers = []
        #the classifiers chosen
        self.classifiers = []

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


    def training_one_weak_classifier_slider(self, start_index, end_index):
        """
        Training in parallel in sliders
        """
        res_info = []
        for index in range(start_index, end_index):
            classifier = self.weak_classifiers[index]
            error_num = self.cal_one_attribute(classifier)
            error_num = abs(error_num)
            #res_info.append((error_num, index))
            res_info.append((error_num, classifier))

        (res_error, res_classifier) = min(res_info)
        return res_error, res_classifier


    def training_one_weak_classifier_parallel(self):
        """
        Training a weak classifier by parallel computer framework
        """
        #job_server = pp.Server(ppservers=ppservers, secret='password')
        job_server = pp.Server(ppservers=ppservers)
        jobs = []
        res_info = []
        cpu_num = job_server.get_ncpus() if self.cpu_number == 0 else self.cpu_number
        slider_base = len(self.weak_classifiers) / cpu_num

        start_index = 0
        all_len = len(self.weak_classifiers)

        for index in range(all_len):
            if index % slider_base == 0 and index != 0:
                jobs.append(job_server.submit(self.training_one_weak_classifier_slider,
                                              (start_index, index), globals=globals()))
                #print 'submit job: [', start_index, index, ']'
                start_index = index

            if index == all_len - 1:
                jobs.append(job_server.submit(self.training_one_weak_classifier_slider,
                                              (start_index, index), globals=globals()))
                #print 'submit job: [', start_index, all_len, ']'


        #print 'finished submit! Do task!'
        job_server.wait()

        for job in jobs:
            #[err_num, index] = job()
            [err_num, classifier] = job()
            #res_info.append((err_num, index))
            res_info.append((err_num, classifier))

        #job_server.print_stats()
        job_server.destroy()
        
        (res_error, res_classifier) = min(res_info)

        return res_error, res_classifier


    def do_adaboost_training(self):
        """
        The progress of training using AdaBoost algorithm.
        """
        for iter in range(self.max_iteration):
            print "iteration ", self.iter, ": "
            self.iter += 1

            #choose a current best weak classifier
            #[err, classifier] = self.training_one_weak_classifier()
            [err, classifier] = self.training_one_weak_classifier_parallel()

            #update the widget of classifier chosen
            classifier.widget = math.log( (1 - err) / err )
            self.classifiers.append(classifier)

            #test print
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

        #return a strong classifier
        final_classifier = StrongClassifier()
        final_classifier.classifiers = self.classifiers
        final_classifier.num_classifier = len(self.classifiers)
        return final_classifier


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


class CascadeClassifierFace:
    """
    The class to train a cascade classifier.
    """
    def __init__(self, config_file):
        """
        Initial
        """
        #self.ada_boost = AdaBoostFace(config_file)
        self.config_training(config_file)
        self.read_sample_data()


    def train_cascade_classifier(self):
        """
        Train a cascade classifier by calling ada-boost algorithm.
        """
        false_positive_rate_last = 1.0
        detection_rate_last = 1.0

        while false_positive_rate_last > self.false_positive_rate_target:
            self.stage_num += 1

             #test print
            print 'layer-' + str(self.stage_num)

            [false_positive_rate_last, detection_rate_last] = self.train_one_strong_classifier(
                false_positive_rate_last, detection_rate_last)

            #test print
            print '\tdetection rate: ', detection_rate_last
            print '\tfalse positive rate: ', false_positive_rate_last

            #update negative sample set
            index = 0
            while index < len(self.negative_samples):
                if self.predict(self.negative_samples[index].integral_img) == 0:
                    self.negative_samples.pop(index)
                else:
                    index += 1
            self.negative_sample_num = len(self.negative_samples)

    
    def train_one_strong_classifier(self, false_positive_rate_last, detection_rate_last):
        """
        Train a strong classifier with feature_num weak classifiers using ada-boost algorithm.
        Add this strong classifier to new layer in cascade classifier.
        """
        #instance of AdaBoostFace
        ada_boost = AdaBoostFace(self.sample_width, self.sample_height, self.cpu_num)
        ada_boost.all_samples = self.positive_samples + self.negative_samples

        #finish condition
        iteration_num = 0 
        current_len = len(self.feature_num_layers)
        if self.stage_num > current_len:
            iteration_num = self.feature_num_layers[current_len - 1] + 25
            self.feature_num_layers.append(iteration_num)
        else:
            iteration_num = self.feature_num_layers[self.stage_num - 1]

        #training initial
        feature_num = 0
        false_positive_rate_current = false_positive_rate_last
        detection_rate_current = 0.0

        while false_positive_rate_current > self.false_positive_rate_layer * false_positive_rate_last:
            if feature_num >= iteration_num:
                break

            feature_num += 1

            #train a strong classifier
            #ada_boost.max_iteration = feature_num
            temp_strong_classifier = ada_boost.do_adaboost_training()
            #print 'one weak classifier created: ', time.strftime('%H:%M:%S', time.localtime(time.time()))

            #evaluate the current false positive rate and detection rate
            [false_positive_rate_current, detection_rate_current] = self.evaluate_current_cascade_classifier(
                temp_strong_classifier, detection_rate_last)

        return false_positive_rate_current, detection_rate_current


    def evaluate_current_cascade_classifier(self, strong_classifier, detection_rate_last):
        """
        Evaluate the current cascade classifier added new strong classifier.
        Decrease threshold of the new strong classifier to achieve the detection rate goal.
        Return current false positive rate and detection rate.
        """
        strong_classifier_num = len(self.cascade_classifiers)
        if strong_classifier_num  == self.stage_num:
            del self.cascade_classifiers[strong_classifier_num - 1]

        temp_positive_correct_samples = []
        temp_false_positive_samples = []

        #evalute detection rate until last layer
        if self.stage_num > 1:
            for sample in self.positive_tests:
                temp_res = self.predict(sample.integral_img)
                if temp_res == 1:
                    temp_positive_correct_samples.append(sample)

            for sample in self.negative_tests:
                temp_res = self.predict(sample.integral_img)
                if temp_res == 1:
                    temp_false_positive_samples.append(sample)
        else:   #if there is no layer
            temp_false_positive_samples = self.negative_tests
            temp_positive_correct_samples = self.positive_tests

        #evalute current detection rate by using current strong classifier as final layer
        positive_correct_num = 0
        false_positive_num = 0

        for sample in temp_positive_correct_samples:
            temp_res = strong_classifier.predict(sample.integral_img)
            if temp_res == 1:
                positive_correct_num += 1

        for sample in temp_false_positive_samples:
            temp_res = strong_classifier.predict(sample.integral_img)
            if temp_res == 1:
                false_positive_num += 1

        false_positive_rate_current = 0.0
        if self.negative_sample_num > 0:
            false_positive_rate_current = false_positive_num * 1.0 / self.negative_sample_num
        detection_rate_current = positive_correct_num * 1.0 / self.positive_sample_num

        #decrease the threshold of this strong classifier to increase detection rate
        while detection_rate_current < self.detection_rate_layer * detection_rate_last:
            strong_classifier.threshold -= 0.01
            #evalute again
            positive_correct_num = 0
            false_positive_num = 0

            for sample in temp_positive_correct_samples:
                temp_res = strong_classifier.predict(sample.integral_img)
                if temp_res == 1:
                    positive_correct_num += 1

            for sample in temp_false_positive_samples:
                temp_res = strong_classifier.predict(sample.integral_img)
                if temp_res == 1:
                    false_positive_num += 1

            if self.negative_sample_num == 0:
                false_positive_rate_current = 0.0
            else:
                false_positive_rate_current = false_positive_num * 1.0 / self.negative_sample_num
            detection_rate_current = positive_correct_num * 1.0 / self.positive_sample_num

        #push this strong classifier as current final layer
        self.cascade_classifiers.append(strong_classifier)

        return false_positive_rate_current, detection_rate_current


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
        self.sample_type = config_items['SampleType']
        self.sample_width = int(config_items['SampleWidth'])
        self.sample_height = int(config_items['SampleHeight'])
        self.positive_sample_path = config_items['PositiveSamplePath']
        self.negative_sample_path = config_items['NegativeSamplePath']
        #paramter of each layer training
        self.false_positive_rate_layer = float(config_items['FalsePositiveRateLayer'])
        self.detection_rate_layer = float(config_items['DetectionRateLayer'])
        self.false_positive_rate_target = float(config_items['FalsePositiveRateTarget'])
        #bits number to fill by 0
        self.z_fill = int(config_items['ZFill'])
        self.cpu_num = int(config_items['CpuNum'])

        #the set of all samples
        self.positive_samples = []
        self.negative_samples = []

        #the set of all tests
        self.positive_tests = []
        self.negative_tests = []

        #all strong classifier
        self.cascade_classifiers = []
        self.stage_num = 0

        #setting the number of features in each layer by human intervention
        self.feature_num_layers = [2, 10, 25, 25, 50, 50, 50]


    def read_sample_data(self):
        """
        Read training data and set the widget value.
        """
        widget = 1.0 / (2 * self.positive_sample_num)

        #read positive samples images
        for i in range(self.positive_sample_num):
            img_path = self.positive_sample_path + str(i+1).zfill(self.z_fill) + '.' + self.sample_type
            self.positive_samples.append( ImgSample(img_path, 1, widget) )
            img_path = self.positive_sample_path + str(i+self.positive_sample_num+1).zfill(self.z_fill) + '.' + self.sample_type
            self.positive_tests.append( ImgSample(img_path, 1, 0.0) )

        widget = 1.0 / (2 * self.negative_sample_num)

        #read negative samples images
        for i in range(self.negative_sample_num):
            img_path = self.negative_sample_path + str(i+1).zfill(self.z_fill) + '.' + self.sample_type
            self.negative_samples.append( ImgSample(img_path, 0, widget) )
            img_path = self.negative_sample_path + str(i+self.negative_sample_num+1).zfill(self.z_fill) + '.' + self.sample_type
            self.negative_tests.append( ImgSample(img_path, 1, 0.0) )

        print 'positive sample number: ', len(self.positive_samples)
        print 'negative sample number: ', len(self.negative_samples)
        print 'positive test number: ', len(self.positive_tests)
        print 'negative test number: ', len(self.negative_tests)


    def predict(self, integral_img):
        """
        Predict the face using current cascade classifier
        """
        result = 1
        for strong_classifier in self.cascade_classifiers:
            temp_res = strong_classifier.predict(integral_img)
            if temp_res == 0:
                result = 0
                break

        return result


    def create_classifier_file(self):
        """
        Write the cascade classifier information into a xml file.
        """
        #create instance of DOM
        doc = Document()

        #create root element
        cascade_classifier_xml = doc.createElement('cascade_frontal_face')
        cascade_classifier_xml.setAttribute('type_id', 'yao_cascade_classifier')
        doc.appendChild(cascade_classifier_xml)

        #sample size
        sample_size = doc.createElement('size')
        sample_size.setAttribute('w', str(self.sample_width))
        sample_size.setAttribute('h', str(self.sample_height))
        cascade_classifier_xml.appendChild(sample_size)

        #stage count
        stage_count = 0

        for strong_Classifier in self.cascade_classifiers:
            #stage information
            stage_info = doc.createElement('stages')
            stage_info.setAttribute('threshold', str(strong_Classifier.threshold))
            stage_comment = doc.createComment('stage-'+str(stage_count))
            stage_info.appendChild(stage_comment)
            cascade_classifier_xml.appendChild(stage_info)

            stage_count += 1
            #tree node count
            tree_count = 0

            for weak_classifier in strong_Classifier.classifiers:
                #decision tree node information
                tree_info = doc.createElement('trees')
                tree_comment = doc.createComment('tree-'+str(tree_count))
                tree_info.appendChild(tree_comment)
                stage_info.appendChild(tree_info)

                tree_count += 1

                #feature information
                feature_info = doc.createElement('features')
                #feature type
                feature_type = doc.createElement('feature_type')
                feature_type.setAttribute('type', str(weak_classifier.feature.type))
                feature_info.appendChild(feature_type)
                #feature position
                feature_pos = doc.createElement('pos')
                feature_pos.setAttribute('x', str(weak_classifier.feature.x))
                feature_pos.setAttribute('y', str(weak_classifier.feature.y))
                feature_info.appendChild(feature_pos)
                #feature size
                feature_size = doc.createElement('size')
                feature_info.appendChild(feature_size)
                feature_size.setAttribute('w', str(weak_classifier.feature.w))
                feature_size.setAttribute('h', str(weak_classifier.feature.h))
                #feature scaled size
                feature_scaled = doc.createElement('scaled')
                feature_info.appendChild(feature_scaled)
                feature_scaled.setAttribute('x_axis', str(weak_classifier.feature.axis_x_scaled))
                feature_scaled.setAttribute('y_axis', str(weak_classifier.feature.axis_y_scaled))
                tree_info.appendChild(feature_info)

                #threshold
                threshold = doc.createElement('threshold')
                threshold_node = doc.createTextNode(str(weak_classifier.threshold))
                threshold.appendChild(threshold_node)
                tree_info.appendChild(threshold)

                #flag
                flag = doc.createElement('flag')
                flag_node = doc.createTextNode(str(weak_classifier.flag))
                flag.appendChild(flag_node)
                tree_info.appendChild(flag)

                #widget
                widget = doc.createElement('widget')
                widget_node = doc.createTextNode(str(weak_classifier.widget))
                widget.appendChild(widget_node)
                tree_info.appendChild(widget)

        #write to file
        f = open('cascade_classifier.xml','w')
        f.write(doc.toprettyxml(indent = ' '))
        f.close()


    def write_information_to_log(self, start_time, end_time):
        """
        Write a log file to record information of this cascade classifier.
        """
        f = open('log.txt', 'w')
        #record time
        f.write('start time: ' + start_time)
        f.write('end time: ' + end_time)
        #record sample number
        f.write('positive sample: ' + str(self.positive_sample_num))
        f.write('negative sample: ' + str(self.negative_sample_num))
        #write size
        f.write('sample size: ' + str(self.sample_width) + ' * ' + str(self.sample_height))
        f.write('stage number: ' + str(len(self.cascade_classifiers)))

        stage_num = 1

        for strong_classifier in self.cascade_classifiers:
            f.write('stage-' + str(stage_num) + ':')
            sub_node_num = len(strong_classifier.classifiers)
            f.write('  sub node number: ' + str(sub_node_num))
            stage_num += 1

        f.close()


if __name__ == "__main__":
    cc = CascadeClassifierFace('cascade_config.txt')
    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print 'start time: ', start_time
    cc.train_cascade_classifier()
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print 'finished time: ', end_time
    cc.create_classifier_file()
    cc.write_information_to_log(start_time, end_time)
