import pandas as pd
import time as time
import pptree
import node
from math import log2
import scipy


class phase():

    def __init__(self, examples=None, chosen_attribute=None, all_attributes_dict=None, parent_examples=None,parent = None,final=False,ans=None,prev_value=None):
        self.examples=examples
        self.chosen_attribute = chosen_attribute
        self.all_attributes_dict = all_attributes_dict
        self.parent_exampls = parent_examples
        self.parent = parent
        self.branches = {}
        self.childrens = []
        self.final=final
        self.ans=ans
        self.prev_value=prev_value


    def __str__(self): # STR -  printing the node Phase on the tree
        answer = ''
        if self.prev_value != None:
            answer=self.prev_value
        if not self.final:
            return answer+'-'+self.chosen_attribute+'?'
        elif self.ans == True:
            return answer+' - BUSY       '#      +' Pk='+ str(self.get_p()) + ' Nk='+str(self.get_n())+ '        P=' + str(self.parent.get_p()) + ' N='+str(self.parent.get_n()) +'        X2 = '+ str(self.parent.X_2) + ' all_same ='+str(self.parent.all_child_same)+' checked='+ str(self.checked)+' treeID='+ str(self.parent.phaseID)
        else: return answer+' - NOT BUSY      '# +' Pk='+ str(self.get_p()) + ' Nk='+str(self.get_n())+'        P=' + str(self.parent.get_p()) + ' N='+str(self.parent.get_n()) +'        X2 = '+str(self.parent.X_2) +' all_same ='+str(self.parent.all_child_same)+' checked=' +str(self.checked)+' treeID=' +str(self.parent.phaseID)


    def getAttributeValues(self): # return the possible values of node attribute
        return self.all_attributes_dict[self.chosen_attribute]

    def getMatchExamples(self,attribue_value): #return the relevant match examples to build child node
        return self.examples[(self.examples[self.chosen_attribute] == attribue_value)]

    def add_branch(self,subtree,attribue_value):
        self.branches[attribue_value] = subtree
        self.childrens.append(subtree)

    def get_p(self): # number of possitive answear examples
        return len(self.examples[(self.examples['busy_hour']==True)])

    def get_n(self):  # number of negative answear examples
        return len(self.examples[(self.examples['busy_hour']==False)])

    def get_x_2(self): # for pruning - calculatig X_2 for phase
        p = self.get_p()
        n = self.get_n()
        X_2 = 0

        for attribute_val in self.branches.keys():

            p_k = self.branches[attribute_val].get_p()
            n_k = self.branches[attribute_val].get_n()

            p_k_hat = p*((p_k+n_k)/(p+n))
            n_k_hat = n*((p_k+n_k)/(p+n))

            if p_k==0 and n_k==0:
                continue
            X_2 = X_2 + (((p_k-p_k_hat)**2)/p_k_hat) + (((n_k-n_k_hat)**2)/n_k_hat)

        self.X_2 = X_2
        return X_2

    def need_to_prune(self): #check if need to prone this phase
        X_2 = self.get_x_2()

        list_of_ans = [self.branches[attribute_val].ans for attribute_val in self.branches.keys()]
        same_value = False
        same_value= all(ans == list_of_ans[0] for ans in list_of_ans)

        ans = X_2<3.841 or same_value
        return ans

    def turn_phase_to_final(self): # actual pruning
        self.final = True
        self.ans = plurality_value(self.examples)
        self.childrens = []
        self.branches = {}


    def find_relevent_child(self,row_input):
        value = row_input[self.chosen_attribute]
        return self.branches[value]

    def check_if_busy(self,row_input): #check if inputed row is busy or not
        if self.final == True:
            return self.ans
        else:
            relevant_child = self.find_relevent_child(row_input)
            return relevant_child.check_if_busy(row_input)

    def row_is_busy(self,row_input):

        columns_name = ['Date', 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                        'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)',
                        'Seasons', 'Holiday', 'Functioning Day']
        df_row = pd.DataFrame(columns=columns_name)
        df_row.loc[0] = row_input

        df_row = bucketing_features(df_row)

        ans = self.check_if_busy(df_row.iloc[0])

        return ans


def entropy(p1,p2):

    if p1==0 or p2==0:
        return 0
    return -(p1*log2(p1)+p2*log2(p2))

def information_gain(attribute,examples):
    TARGET_BUSY = examples[examples['busy_hour']==True]
    TARGET_NOT_BUSY = examples[examples['busy_hour']==False]
    lentgh_of_examples = len(examples)
    prop_for_busy = len(TARGET_BUSY)/lentgh_of_examples
    prop_for_non_busy = len(TARGET_NOT_BUSY)/lentgh_of_examples

    gain = entropy(prop_for_busy,prop_for_non_busy)
    values_of_attribute = list(pd.unique(examples[attribute]))
    for value in values_of_attribute:
        examples_with_value = examples[examples[attribute]==value]
        length_of_examples_with_value = len(examples_with_value)
        prop_for_value = length_of_examples_with_value/lentgh_of_examples
        TARGET_BUSY_for_value = examples_with_value[examples_with_value['busy_hour']==True]
        TARGET_NOT_BUSY_for_value = examples_with_value[examples_with_value['busy_hour']==False]

        prop_for_value_busy = len(TARGET_BUSY_for_value)/length_of_examples_with_value
        prop_for_value_non_busy = len(TARGET_NOT_BUSY_for_value)/length_of_examples_with_value

        gain=gain-prop_for_value*entropy(prop_for_value_busy,prop_for_value_non_busy)

    return gain

def plurality_value(examples):
    return examples['busy_hour'].value_counts().idxmax()

def get_next_attribute(examples, attributes):
    gains = {}
    for attribute in attributes:
        gains[attribute] = information_gain(attribute,examples)

    return max(gains,key=gains.get)



def decision_tree_learning(examples, attributes_dict,parent_examples,attribute=None,parent=None ,i=None,prevvalue=None):

    if len(examples) == 0:
        return phase(final=True,parent=parent ,chosen_attribute=attribute,examples=examples,ans=plurality_value(parent_examples),prev_value=prevvalue)

    elif len(examples['busy_hour'].value_counts()) == 1:
        return phase(final=True,parent=parent,chosen_attribute=attribute,examples=examples,ans=examples['busy_hour'].value_counts().idxmax(),prev_value=prevvalue)

    elif len(attributes_dict)==0:
        return phase(final=True,parent=parent,chosen_attribute=attribute,examples=examples,ans=plurality_value(examples),prev_value=prevvalue)

    else:
        next_attribute = get_next_attribute(examples, attributes_dict.keys())
        new_phase = phase(examples, next_attribute, attributes_dict, parent_examples,parent=parent,prev_value=prevvalue)
        attributes_for_subtree = attributes_dict.copy()
        attributes_for_subtree.pop(next_attribute)

        for attribue_value in new_phase.getAttributeValues():
            match_examples = new_phase.getMatchExamples(attribue_value)
            subtree = decision_tree_learning(examples=match_examples,attribute=next_attribute,attributes_dict= attributes_for_subtree, parent_examples=examples,parent=new_phase,i=i+1,prevvalue=attribue_value)
            i=i+1
            new_phase.add_branch(subtree,attribue_value=attribue_value)
        return new_phase

# replace values with categorical values based on analyze
def rented_bikes_count_TO_busy_hour(rented_bike_count):
    if rented_bike_count > 650:
        return True
    else:
        return False


def hour_bucketing(hour):
    if hour < 7 or hour > 20:
        return 'night'
    if hour < 11:
        return 'morning'
    if hour < 16:
        return 'noon'
    else:
        return 'evening'


def temprature_bucketing(temp):
    if temp < 3.5:
        return 'super_cold'
    if temp < 13.7:
        return 'cold'
    if temp < 22.5:
        return 'normal'
    else:
        return 'hot'


def Humidity_bucketing(hum):
    if hum < 48:
        return 'low'
    if hum < 80:
        return 'medium'
    else:
        return 'high'


def Wind_bucketing(wind):
    if wind < 1.1:
        return 'slow'
    if wind < 1.8:
        return 'regular'
    else:
        return 'fast'


def Visibility_bucketing(meters_10):
    if meters_10 < 1191:
        return 'fog'
    if meters_10 < 1956:
        return 'ok'
    else:
        return 'clear'


def Dew_point_temperature_bucketinng(temp):
    if temp < -1.2:
        return 'low'
    if temp < 11.2:
        return 'mid'
    else:
        return 'high'


def Solar_Radiation_becketing(solar):
    if solar < 1.2:
        return 'ok'
    else:
        return 'burn'


def Rainfall_bucketing(mm):
    if mm == 0:
        return 'not_rain'
    else:
        return 'rain'


def Snowfall_bucketing(cm):
    if cm == 0:
        return 'not_snow'
    else:
        return 'snow'


def bucketing_features(data):

    data['Hour'] = data['Hour'].apply(lambda hour: hour_bucketing(hour))
    data['Temp'] = data['Temperature(°C)'].apply(lambda temp: temprature_bucketing(temp))
    data['Hum'] = data['Humidity(%)'].apply(lambda hum: Humidity_bucketing(hum))
    data['Wind speed'] = data['Wind speed (m/s)'].apply(lambda wind: Wind_bucketing(wind))
    data['Visibility'] = data['Visibility (10m)'].apply(lambda meters: Visibility_bucketing(meters))
    data['DP-temp'] = data['Dew point temperature(°C)'].apply(lambda temp: Dew_point_temperature_bucketinng(temp))
    data['Radiation'] = data['Solar Radiation (MJ/m2)'].apply(lambda solar: Solar_Radiation_becketing(solar))
    data['Rain'] = data['Rainfall(mm)'].apply(lambda mm: Rainfall_bucketing(mm))
    data['Snow'] = data['Snowfall (cm)'].apply(lambda cm: Snowfall_bucketing(cm))

    data['year'] = data['Date'].apply(lambda date: str(date)[-4:])
    data.drop(columns=['Date','Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature(°C)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)'], inplace=True)

    return data

def buketing_target(data):

    data['busy_hour'] = data['Rented Bike Count'].apply(lambda value: rented_bikes_count_TO_busy_hour(value))
    data.drop(columns='Rented Bike Count', inplace=True)

    return data


def split_train_test(data,ratio):
    train = data.sample(frac=ratio, random_state=150)
    test = data.drop(train.index)
    return train,test

def post_pruning(tree):
    if tree.final == True:
        parent = tree.parent
        prune_ans = parent.need_to_prune()
        if prune_ans:
            parent.turn_phase_to_final()

        return tree, prune_ans

    for subtree in tree.childrens:
        sometree, proned_ans = post_pruning(subtree)
        if proned_ans:
            sometree, proned_ans = post_pruning(tree)
    return tree, False


def test_error(tree,test):
    sum_of_true_predicted=0
    count_all=0
    for index, test_row in test.iterrows():
        count_all = count_all+1
        val_for_row = test_row['busy_hour']
        row = test_row.drop(columns='busy_hour')

        columns = ['Hour', 'Seasons', 'Holiday', 'Functioning Day', 'Temp', 'Hum', 'Wind speed', 'Visibility',
                   'DP-temp', 'Radiation', 'Rain', 'Snow', 'year']
        row = row[columns]
        predicted_val_for_row = tree.check_if_busy(row)
        if predicted_val_for_row == val_for_row:
            sum_of_true_predicted=sum_of_true_predicted+1

    if count_all==0:
        accuracy = 0
    else:
        accuracy = sum_of_true_predicted/count_all

    return accuracy

def get_data():
    all_data = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')

    all_data = bucketing_features(all_data)
    all_data = buketing_target(all_data)

    attributes = list(all_data.columns)
    attributes.remove('busy_hour')

    attributes_dict = {}
    for attribute in attributes:
        attributes_dict[attribute] = list(pd.unique(all_data[attribute]))

    return all_data,attributes_dict

def build_tree_with_tain_test(train,test,attributes_dict):

    tree = decision_tree_learning(train, attributes_dict, parent_examples=None, i=0)
    proned_tree, proned_ans = post_pruning(tree)
    accuracy = test_error(proned_tree, test)
    return proned_tree, accuracy


def build_tree_with_no_printing(ratio):

    all_data, attributes_dict = get_data()
    train, test = split_train_test(all_data, ratio)

    proned_tree, accuracy = build_tree_with_tain_test(train,test,attributes_dict)

    return proned_tree, accuracy

def build_tree(ratio):
    print('building the tree...')
    proned_tree,accuracy= build_tree_with_no_printing(ratio)
    pptree.print_tree(proned_tree, childattr='childrens', horizontal=True)
    print('---------------------------------------------------')
    print('---------------Test Error------',round(1-accuracy,2),'---------------')
    print('---------------------------------------------------')
    print('-----------Test Accuracy------', round(accuracy, 2),'---------------')
    print('---------------------------------------------------')
    return proned_tree,accuracy

def cross_validation_split(data,k):
    fold_dict = {}
    iterate = k+1
    ratio = 1/k
    while iterate>1:
        iterate = iterate - 1
        fold, rest = split_train_test(data,ratio)
        fold_dict[iterate] = fold
        data = rest
        if ratio==1:
            continue
        ratio = 1 / ((1 / ratio) - 1)
    return fold_dict

def tree_error(k):
    print('Cross Validation - K =',str(k))
    sum_of_accuarcy = 0
    ratio = 1/k
    fold = k

    data, attributes_dict = get_data()
    folds_dict = cross_validation_split(data,k)

    while fold>0:
        print('building the tree - fold number ',str(k-fold+1))
        train_list = [df for key, df in folds_dict.items() if key not in [fold]]
        train = pd.concat(train_list)
        test = folds_dict[fold]

        tree,treAccuarcy = build_tree_with_tain_test(train,test,attributes_dict)
        sum_of_accuarcy=sum_of_accuarcy+treAccuarcy
        fold=fold-1
    avg_accuracy = sum_of_accuarcy/k
    print('---------------------------------------------------')
    print('------------------CROSS - VALIDATION-------------------')
    print('---------------------------------------------------')
    print('---------------K-fold Error------', round(1 - avg_accuracy, 2), '---------------')
    print('---------------------------------------------------')
    print('-----------K-fold Accuracy------', round(avg_accuracy, 2), '---------------')
    print('---------------------------------------------------')

def is_busy(sample_prob):
    proned_tree,accuracy= build_tree_with_no_printing(ratio=1)
    ans = proned_tree.row_is_busy(sample_prob)
    if ans:
        return 1
    else: return 0

if __name__ == '__main__':

    k=3
    ratio = 0.5

    build_tree(ratio)
    tree_error(k)

    # sample = ['1/12/2017',2,-6,39,1,2000,-17.7,0,0,0,'Winter','No Holiday','Yes']
    # sample_2 = ['1/12/2017', 8, -7.6, 37, 1.1, 2000, -19.8, 0.01, 0, 0, 'Winter', 'No Holiday', 'Yes']
    # sample_3 = ['7/8/2018',8,28.5,64,2,2000,21,0.89,0,0,'Summer','No Holiday','Yes']
    # sample_4 = ['7/8/2018',8,28.5,64,2,2000,21,0.89,0,0,'Summer','No Holiday','Yes']
    #
    # sample_prob = ['7/8/2018',13,40,98,2,2000,21,0.89,0,0,'Winter','No Holiday','Yes']

    # ans_for_row = is_busy(sample_4)
    #print(ans_for_row)

