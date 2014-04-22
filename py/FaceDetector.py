__author__ = 'yao'

from xml.dom import minidom
import scipy.misc
import matplotlib.pyplot
from PIL import Image
import numpy
import math


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


    def cal_feature_value(self, integral_img, w = 19, h = 19):
        """
        Calculate the value of the feature and then return it.
        @param integral_img: the integral image matrix of a image
        """
        #print 'type:', self.type
        #print 'raw x:', self.x, 'raw y:', self.y
        #print 'raw axis x:', self.axis_x_scaled, 'raw axis y:', self.axis_y_scaled

        #scaled
        record_x = self.x
        record_y = self.y
        record_axis_x = self.axis_x_scaled
        record_axis_y = self.axis_y_scaled
        row, col = integral_img.shape
        row -= 1
        col -= 1
        self.x = int(round(self.x * row / h))
        self.y = int(round(self.y * col / w))

        if self.type == '2h':
            self.axis_y_scaled = int(round(self.axis_y_scaled * row / h))
            self.axis_x_scaled = min(int(round(2 * self.axis_x_scaled * col / w + 1)) / 2, (col - self.y + 1) / 2)
        elif self.type == '2v':
            self.axis_y_scaled = min(int(round(2 * self.axis_y_scaled * row / h + 1)) / 2, (row - self.x + 1) / 2)
            self.axis_x_scaled = int(round(self.axis_x_scaled * col / w))
        elif self.type == '3h':
            self.axis_y_scaled = int(round(self.axis_y_scaled * row / h))
            self.axis_x_scaled = min(int(round(3 * self.axis_x_scaled * col / w)) / 3, (col - self.y + 1) / 3)
        elif self.type == '3v':
            self.axis_y_scaled = min(int(round(3 * self.axis_y_scaled * row / h)) / 3, (row - self.x + 1) / 3)
            self.axis_x_scaled = int(round(self.axis_x_scaled * col / w))
        elif self.type == '4c':
            self.axis_y_scaled = min(int(round(2 * self.axis_y_scaled * row / h + 1)) / 2, (row - self.x + 1) / 2)
            self.axis_x_scaled = min(int(round(2 * self.axis_x_scaled * col / w + 1)) / 2, (col - self.y + 1) / 2)
        else:
            print "Error feature type!"

        # 'row: ', row, 'col: ', col
        #print 'x:', self.x, 'y:', self.y
        #print 'x_axis:', self.axis_x_scaled, 'y_axis:', self.axis_y_scaled

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

        self.x = record_x
        self.y = record_y
        self.axis_x_scaled = record_axis_x
        self.axis_y_scaled = record_axis_y

        return feature_value


    def print_feature_info(self):
        """
        """
        info = self.type + ':(' + str(self.x) + ',' + str(self.y) + ')(' +str(self.w) + ',' + str(self.h) + ')'
        return info



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



class StrongClassifier():
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



class FaceDetector():
    """
    Detector the face by using cascade classifier in xml file
    """
    def __init__(self, classifier_file):
        """
        Initial the cascade classifier
        """
        self.sample_width = 0
        self.sample_height = 0
        self.scale_multiplier = 1.25
        self.cascade_classifiers = []

        self.load_xml_file(classifier_file)


    def load_xml_file(self, classifier_file):
        """
        Parse the xml file and store the cascade classifier information
        """
        doc = minidom.parse(classifier_file)
        root = doc.documentElement

        #sample size
        size_nodes = root.getElementsByTagName('size')
        size_node = size_nodes[0]
        if size_node:
            self.sample_width = int(size_node.getAttribute('w'))
            self.sample_height = int(size_node.getAttribute('h'))

        print 'sample size: ', self.sample_width, self.sample_height

        stage_num = 0

        #stages
        stage_nodes = root.getElementsByTagName('stages')
        for stage_node in stage_nodes:
            stage_threshold = float(stage_node.getAttribute('threshold'))

            print '='*20
            print 'layer: ', stage_num
            print 'stage threshold: ', stage_threshold
            stage_num += 1

            #all weak classifiers
            weak_classifiers = []

            tree_num = 0

            #decision trees
            tree_nodes = stage_node.getElementsByTagName('trees')
            for tree_node in tree_nodes:
                print '  tree: ', tree_num
                tree_num += 1

                #feature node
                feature_nodes = tree_node.getElementsByTagName('features')
                feature_node = feature_nodes[0]
                type_node = feature_node.getElementsByTagName('feature_type')
                feature_type = type_node[0].getAttribute('type')
                pos_node = feature_node.getElementsByTagName('pos')
                pos_x = int(pos_node[0].getAttribute('x'))
                pos_y = int(pos_node[0].getAttribute('y'))
                fsize_node = feature_node.getElementsByTagName('size')
                size_w = int(fsize_node[0].getAttribute('w'))
                size_h = int(fsize_node[0].getAttribute('h'))
                scaled_node = feature_node.getElementsByTagName('scaled')
                scaled_x = int(scaled_node[0].getAttribute('x_axis'))
                scaled_y = int(scaled_node[0].getAttribute('y_axis'))
                feature = HaarFeature(feature_type, pos_y, pos_x, size_w, size_h, scaled_x, scaled_y)
                print '\t', feature.print_feature_info()

                #threshold node
                threshold_node = tree_node.getElementsByTagName('threshold')[0]
                threshold = int(threshold_node.childNodes[0].data)
                print '\tthreshold: ', threshold

                #flag node
                flag_node = tree_node.getElementsByTagName('flag')[0]
                flag = int(flag_node.childNodes[0].data)
                print '\tflag: ', flag

                #widget node
                widget_node = tree_node.getElementsByTagName('widget')[0]
                widget = float(widget_node.childNodes[0].data)
                print '\twidget: ', widget

                #weak classifier
                weak_classifier = WeakClassifier(feature)
                weak_classifier.threshold = threshold
                weak_classifier.flag = flag
                weak_classifier.widget = widget
                weak_classifiers.append(weak_classifier)

            #a strong classifier
            strong_classifier = StrongClassifier()
            strong_classifier.classifiers = weak_classifiers
            strong_classifier.num_classifier = len(weak_classifiers)
            strong_classifier.threshold = stage_threshold
            #push in
            self.cascade_classifiers.append(strong_classifier)


    def generate_all_sub_windows(self, row, col):
        """
        Create all sub widows to through
        """
        k = 0
        slider_step = 3
        w = int(round( self.sample_width * math.pow(self.scale_multiplier, k) ))
        h = int(round( self.sample_height * math.pow(self.scale_multiplier, k) ))
        sub_windows = {}

        while w <= (col / 4) and h <= (row / 4):
            sub_window = []

            #move detection sub-window
            for t in range(0, (row - h + 1), slider_step):
                for l in range(0, (col - w + 1), slider_step):
                    sub_window.append((t, l, w, h))
            #add sub_window list in dictionary
            sub_windows[k] = sub_window
            #update scaled sub_window
            k += 1
            w = int(round( self.sample_width * math.pow(self.scale_multiplier, k) ))
            h = int(round( self.sample_height * math.pow(self.scale_multiplier, k) ))

            slider_step += 3

        return sub_windows


    def detect_face(self, image_file):
        """
        Detector the face in a image
        """
        #image and its copy
        img = Image.open(image_file)
        img_gray = img.convert('L')
        target_img = numpy.array(img_gray)

        row, col = target_img.shape
        #print 'img row:', row, 'img col:', col
        sub_windows = self.generate_all_sub_windows(row, col)
        positive_sub_windows = {}

        print 'all sub windows num:', len(sub_windows)

        #paint image
        #matplotlib.pyplot.imshow(numpy.array(img))

        #iterate all sub windows to predict
        for key, sub_window in sub_windows.items():
            matplotlib.pyplot.imshow(numpy.array(img))

            #positive sub_window of key multiplier scaled
            positive_sub_window = []
            for [t, l, w, h] in sub_window:
                #print '(', t, l, w, h, ')'

                #take sub image from raw image
                sub_window_img = numpy.zeros((h, w), dtype=numpy.int)
                for i in range(h):
                    for j in range(w):
                        sub_window_img[i,j] = target_img[t + i, l + j]

                #result = self.cal_standard_deviation(sub_window_img)
                integral_img = self.cal_integral_img(sub_window_img)
                result = self.predict(integral_img)

                if result == 1:
                    positive_sub_window.append(sub_window)
                    y = [t, t + h, t + h, t, t]
                    x = [l, l, l + w, l + w, l]
                    matplotlib.pyplot.plot(x, y, 'b-')

            print 'key:', key, 'positive sub window num:', len(positive_sub_window)
            positive_sub_windows[key] = positive_sub_window

            matplotlib.pyplot.axis('off')
            matplotlib.pyplot.title('face detection')
            matplotlib.pyplot.show()


    def cal_integral_img(self, img_content):
        """
        Calculate the integral image representation of raw image
        @param img_content: the data content of the raw image
        """
        row, col = img_content.shape
        integral_img = numpy.zeros((row+1, col+1), dtype=numpy.int)
        integral_img[0, 0:] = 0
        integral_img[0:, 0] = 0

        for i in range(row):
            for j in range(col):
                integral_img[i+1, j+1] = integral_img[i, j+1] + integral_img[i+1, j] - integral_img[i,j] + img_content[i,j]

        return integral_img


    def cal_standard_deviation(self, img):
        """
        Calculate the standard deviation of this sub window image
        """
        row, col = img.shape

        average = 0.0
        #calculate the average of this sub image
        for i in range(row):
            for j in range(col):
                average += img[i, j]
        average = average / (row * col)

        standard_deviation = 0.0
        #calculate the deviation
        for i in range(row):
            for j in range(col):
                standard_deviation += math.pow((img[i,j] - average), 2)
        standard_deviation = standard_deviation / (row * col)
        standard_deviation = math.sqrt(standard_deviation)

        if standard_deviation <= 1:
            return 0
        else:
            return 1


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

detector = FaceDetector('xml_data/cascade_classifier3.xml')
#lena = scipy.misc.lena()
#lena = numpy.array(Image.open('09010309.jpg'))
print '=' * 20
print 'start detection:'
detect_list = ['09010309.jpg', 't1.JPG', 't2.jpg', 't3.jpg']
detector.detect_face('pics/' + detect_list[3])

#matplotlib.pyplot.imshow(lena, matplotlib.pyplot.cm.gray)
#matplotlib.pyplot.imshow(lena)
#x = [50,50,100,100,50]
#y = [50,100,100,50,50]
#matplotlib.pyplot.plot(x, y, 'r*')
#matplotlib.pyplot.plot(x, y, 'r-')
#matplotlib.pyplot.title('plotting')
#matplotlib.pyplot.show()

