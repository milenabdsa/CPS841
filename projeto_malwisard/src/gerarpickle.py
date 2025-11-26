import numpy as np
import pandas as pd
import pickle
import os
import cv2
import gc

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

print("Importing data")
class_mapping = {
    'Adposhel': {'y_26': 'Adposhel', 'y_2': 'Malign', 'y_6': 'adware'},
    'Agent': {'y_26': 'Agent', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Allaple': {'y_26': 'Allaple', 'y_2': 'Malign', 'y_6': 'Worm'},
    'Amonetize': {'y_26': 'Amonetize', 'y_2': 'Malign', 'y_6': 'adware'},
    'Androm': {'y_26': 'Androm', 'y_2': 'Malign', 'y_6': 'Backdoor'},
    'Autorun': {'y_26': 'Autorun', 'y_2': 'Malign', 'y_6': 'Worm'},
    'BrowseFox': {'y_26': 'BrowseFox', 'y_2': 'Malign', 'y_6': 'adware'},
    'Dinwod': {'y_26': 'Dinwod', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Elex': {'y_26': 'Elex', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Expiro': {'y_26': 'Expiro', 'y_2': 'Malign', 'y_6': 'Virus'},
    'Fasong': {'y_26': 'Fasong', 'y_2': 'Malign', 'y_6': 'Worm'},
    'HackKMS': {'y_26': 'HackKMS', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Hlux': {'y_26': 'Hlux', 'y_2': 'Malign', 'y_6': 'Worm'},
    'Injector': {'y_26': 'Injector', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'InstallCore': {'y_26': 'InstallCore', 'y_2': 'Malign', 'y_6': 'adware'},
    'MultiPlug': {'y_26': 'MultiPlug', 'y_2': 'Malign', 'y_6': 'adware'},
    'Neoreklami': {'y_26': 'Neoreklami', 'y_2': 'Malign', 'y_6': 'adware'},
    'Neshta': {'y_26': 'Neshta', 'y_2': 'Malign', 'y_6': 'Virus'},
    'Other': {'y_26': 'Benign', 'y_2': 'Benign', 'y_6': 'Benign'},
    'Regrun': {'y_26': 'Regrun', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Sality': {'y_26': 'Sality', 'y_2': 'Malign', 'y_6': 'Virus'},
    'Snarasite': {'y_26': 'Snarasite', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Stantinko': {'y_26': 'Stantinko', 'y_2': 'Malign', 'y_6': 'Backdoor'},
    'VBA': {'y_26': 'VBA', 'y_2': 'Malign', 'y_6': 'Virus'},
    'VBKrypt': {'y_26': 'VBKrypt', 'y_2': 'Malign', 'y_6': 'Trojan'},
    'Vilsel': {'y_26': 'MultiPlug', 'y_2': 'Malign', 'y_6': 'Trojan'}
}

data = []
y_26 = []
y_2 = []
y_6 = []

for class_name, labels in class_mapping.items():
    img = load_images_from_folder(f'malevis_train_val_224x224/train/{class_name}')
    length = len(img)
    
    if length > 0:
        data.extend(img)
        y_26.extend([labels['y_26']] * length)
        y_2.extend([labels['y_2']] * length)
        y_6.extend([labels['y_6']] * length)
    
    del img
    gc.collect()

data = np.array(data)
y_26 = np.array(y_26)
y_2 = np.array(y_2)
y_6 = np.array(y_6)


pickle.dump( data, open( "data.p", "wb" ) )
pickle.dump( y_26, open( "y26.p", "wb" ) )
pickle.dump( y_2, open( "y2.p", "wb" ) )
pickle.dump( y_6, open( "y6.p", "wb" ) )

# Process validation data
print("Processing validation data")
data_val = []
y_26_val = []
y_2_val = []
y_6_val = []

for class_name, labels in class_mapping.items():
    img = load_images_from_folder(f'malevis_train_val_224x224/val/{class_name}')
    length = len(img)
    
    if length > 0:
        data_val.extend(img)
        y_26_val.extend([labels['y_26']] * length)
        y_2_val.extend([labels['y_2']] * length)
        y_6_val.extend([labels['y_6']] * length)
    
    del img
    gc.collect()

data_val = np.array(data_val)
y_26_val = np.array(y_26_val)
y_2_val = np.array(y_2_val)
y_6_val = np.array(y_6_val)

pickle.dump( data_val, open( "data_val.p", "wb" ) )
pickle.dump( y_26_val, open( "y26_val.p", "wb" ) )
pickle.dump( y_2_val, open( "y2_val.p", "wb" ) )
pickle.dump( y_6_val, open( "y6_val.p", "wb" ) )


