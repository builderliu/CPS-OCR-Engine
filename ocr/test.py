
import pickle
def get_label_dict():
    f=open('./chinese_labels_bak','rb')
    label_dict = pickle.load(f)
    f.close()
    print(label_dict)
    return label_dict



if __name__ == '__main__':
    label_dict=get_label_dict()
    for (value,chars) in label_dict.items():
        print (value,chars)

