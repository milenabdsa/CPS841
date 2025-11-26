import wisardpkg as wp
import numpy as np
import cv2
import copy
from sklearn.metrics import confusion_matrix
import time
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def most_frequent(List): 
    return max(set(List), key = List.count) 

def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    print(clf_matrix)
    
wisard_train = True
mental_images = True
differences = False
wisard_correlation = True
wisard_testing = False
wisard_pairs_voting_crop = False
wisard_pairs_voting = False
wisard_pairs = False
wisard_voting = False
cluswisard = False

if wisard_train:
    print("WiSARD")
    
    wsd = wp.Wisard(20, bleachingActivated = True, ignoreZero = False, verbose = True)
    print("Training...")
    ds = wp.DataSet("Expiro_crop_all.wpkds")
    wsd.train(ds)
    print("\n")
    ds = wp.DataSet("Sality_crop_all.wpkds")
    wsd.train(ds)
    print("\n")
    ds = wp.DataSet("Neshta_crop_all.wpkds")
    wsd.train(ds)
    print("\n")
    ds = wp.DataSet("VBA_crop_all.wpkds")
    wsd.train(ds)
    print("\n")

    # wsd.json(True, "virus_all")

if mental_images:
    print("Getting patterns...")

    patterns = wsd.getMentalImages()

    expiro_therm = patterns["Expiro"]
    neshta_therm = patterns["Neshta"]
    sality_therm = patterns["Sality"]
    vba_therm = patterns["VBA"]

    expiro = []
    neshta = []
    sality = []
    vba = []

    for i in range(int(len(expiro_therm)/12)):
        m = max(expiro_therm[12*i:12*(i+1)])
        keys = [i for i,j in enumerate(expiro_therm[12*i:12*(i+1)]) if j == m]
        try:
            expiro.append((keys[0]+1)*255/12)
        except:
            expiro.append((keys+1)*255/12)

        m = max(neshta_therm[12*i:12*(i+1)])
        keys = [i for i,j in enumerate(neshta_therm[12*i:12*(i+1)]) if j == m]
        try:
            neshta.append((keys[0]+1)*255/12)
        except:
            neshta.append((keys+1)*255/12)

        m = max(sality_therm[12*i:12*(i+1)])
        keys = [i for i,j in enumerate(sality_therm[12*i:12*(i+1)]) if j == m]
        try:
            sality.append((keys[0]+1)*255/12)
        except:
            sality.append((keys+1)*255/12)

        m = max(vba_therm[12*i:12*(i+1)])
        keys = [i for i,j in enumerate(vba_therm[12*i:12*(i+1)]) if j == m]
        try:
            vba.append((keys[0]+1)*255/12)
        except:
            vba.append((keys+1)*255/12)

    expiro_img = np.reshape(expiro, (75,224,3))
    neshta_img = np.reshape(neshta, (75,224,3))
    sality_img = np.reshape(sality, (75,224,3))
    vba_img = np.reshape(vba, (75,224,3))
  
    cv2.imwrite("expiro_crop.png", expiro_img)
    cv2.imwrite("neshta_crop.png", neshta_img)
    cv2.imwrite("sality_crop.png", sality_img)
    cv2.imwrite("vba_crop.png", vba_img)

if differences:
    print("Differences")
    if False:
        expiro_neshta = np.array(expiro_therm)-np.array(neshta_therm)
        expiro_neshta = np.where(expiro_neshta > 0, 1, 0)
        d_expiro_neshta = wp.Discriminator(20, 75*224*3*12, ignoreZero=False)
        d_expiro_neshta.train(expiro_neshta)

        neshta_expiro = np.array(neshta_therm)-np.array(expiro_therm)
        neshta_expiro = np.where(neshta_expiro > 0, 1, 0)
        d_neshta_expiro = wp.Discriminator(20, 75*224*3*12, ignoreZero=False)
        d_neshta_expiro.train(neshta_expiro)

        expiro_sality = np.array(expiro_therm)-np.array(sality_therm)
        expiro_sality = np.where(expiro_sality > 0, 1, 0)
        d_expiro_sality = wp.Discriminator(20, 75*224*3*12, ignoreZero=False)
        d_expiro_sality.train(expiro_sality)
        
        sality_expiro = np.array(sality_therm)-np.array(expiro_therm)
        sality_expiro = np.where(sality_expiro > 0, 1, 0)
        d_sality_expiro = wp.Discriminator(20, 75*224*3*12, ignoreZero=False)
        d_sality_expiro.train(sality_expiro)

        neshta_sality = np.array(neshta_therm)-np.array(sality_therm)
        neshta_sality = np.where(neshta_sality > 0, 1, 0)
        d_neshta_sality = wp.Discriminator(20, 75*224*3*12, ignoreZero=False)
        d_neshta_sality.train(neshta_sality)
        
        sality_neshta = np.array(sality_therm)-np.array(neshta_therm)
        sality_neshta = np.where(sality_neshta > 0, 1, 0)
        d_sality_neshta = wp.Discriminator(20, 75*224*3*12, ignoreZero=False)
        d_sality_neshta.train(sality_neshta)
    
    if True:
        expiro = [0 for i in range(75*224*3*12)]
        neshta = [0 for i in range(75*224*3*12)]
        sality = [0 for i in range(75*224*3*12)]
        vba = [0 for i in range(75*224*3*12)]

        for i in range(int(len(expiro_therm)/12)):
            m = expiro_therm[12*i:12*(i+1)]
            keys = [i for i,j in enumerate(m) if j > np.median(m)]
            for j in keys:
                expiro[12*i + j] = 1

            m = neshta_therm[12*i:12*(i+1)]
            keys = [i for i,j in enumerate(m) if j > np.median(m)]
            for j in keys:
                neshta[12*i + j] = 1

            m = sality_therm[12*i:12*(i+1)]
            keys = [i for i,j in enumerate(m) if j > np.median(m)]
            for j in keys:
                sality[12*i + j] = 1
      
        print("Training")
        wsd_not = wp.Wisard(20, bleachingActivated = True, ignoreZero = False, verbose = True, returnActivationDegree=True, returnConfidence=True, returnClassesDegrees=True)
        
        expiro = np.array(expiro)
        not_expiro = np.where(expiro > 0, 0, 1)
        neshta = np.array(neshta)
        not_neshta = np.where(neshta > 0, 0, 1)
        sality = np.array(sality)
        not_sality = np.where(sality > 0, 0, 1)
        
        wsd_not.train([expiro],["Expiro"])
        wsd_not.train([neshta],["Not-Expiro"])
        wsd_not.train([sality],["Not-Expiro"])
        wsd_not.train([neshta],["Neshta"])
        wsd_not.train([expiro],["Not-Neshta"])
        wsd_not.train([sality],["Not-Neshta"])
        wsd_not.train([sality],["Sality"])
        wsd_not.train([expiro],["Not-Sality"])
        wsd_not.train([neshta],["Not-Sality"])

    print("=== EXPIRO ===")
    ds = wp.DataSet("Expiro_test_crop_all.wpkds")
    a = ds.get(37)
    expiro_example = []
    for i in range(a.size()):
        expiro_example.append(a.get(i))
    out = wsd_not.classify([expiro_example])
    print(out)

    print("=== NESHTA ===")
    ds = wp.DataSet("Neshta_test_crop_all.wpkds")
    a = ds.get(37)
    expiro_example = []
    for i in range(a.size()):
        expiro_example.append(a.get(i))
    out = wsd_not.classify([expiro_example])
    print(out)
    
    print("=== SALITY ===")
    ds = wp.DataSet("Sality_test_crop_all.wpkds")
    a = ds.get(37)
    expiro_example = []
    for i in range(a.size()):
        expiro_example.append(a.get(i))
    out = wsd_not.classify([expiro_example])
    print(out)

if wisard_correlation:
    img = load_images_from_folder('malevis_train_val_224x224/val/Expiro')
    data = np.array(img).reshape(len(img), 224*224*3)
    expiro_data = []
    for i in range(len(data)):
        expiro_data.append(data[i][75*224*3:150*224*3])

    img = load_images_from_folder('malevis_train_val_224x224/val/Neshta')
    data = np.array(img).reshape(len(img), 224*224*3)
    neshta_data = []
    for i in range(len(data)):
        neshta_data.append(data[i][75*224*3:150*224*3])

    img = load_images_from_folder('malevis_train_val_224x224/val/Sality')
    data = np.array(img).reshape(len(img), 224*224*3)
    sality_data = []
    for i in range(len(data)):
        sality_data.append(data[i][75*224*3:150*224*3])

    out = 0
    total = 0

    for j in range(len(expiro_data)):
        a = [np.convolve(expiro, expiro_data[j], 'valid'), np.convolve(neshta, expiro_data[j], 'valid'), np.convolve(sality, expiro_data[j], 'valid')]
        if(a.index(max(a)) == 0):
            out += 1
        total += 1

    for j in range(len(neshta_data)):
        a = [np.convolve(expiro, neshta_data[j], 'valid'), np.convolve(neshta, neshta_data[j], 'valid'), np.convolve(sality, neshta_data[j], 'valid')]
        if(a.index(max(a)) == 1):
            out += 1
        total += 1

    for j in range(len(sality_data)):
        a = [np.convolve(expiro, sality_data[j], 'valid'), np.convolve(neshta, sality_data[j], 'valid'), np.convolve(sality, sality_data[j], 'valid')]
        if(a.index(max(a)) == 2):
            out += 1
        total += 1
   
    print("Out: " + str(out))
    print("Total: " + str(total))
    print("Fraction: " + str(out/total))

if wisard_testing:
    print("Testing...")
    
    out = []
    y = []
    
    ds = wp.DataSet("Expiro_test_crop_all.wpkds")
    out_expiro = wsd.classify(ds)
    for i in range(len(out_expiro)):
        out.append(out_expiro[i])
        y.append(ds.getLabel(i))            

    ds = wp.DataSet("Sality_test_crop_all.wpkds")
    out_sality = wsd.classify(ds)
    for i in range(len(out_sality)):
        out.append(out_sality[i])
        y.append(ds.getLabel(i))            

    ds = wp.DataSet("Neshta_test_crop_all.wpkds")
    out_neshta = wsd.classify(ds)
    for i in range(len(out_neshta)):
        out.append(out_neshta[i])
        y.append(ds.getLabel(i))            
    
    #ds = wp.DataSet("VBA_test_crop_all.wpkds")
    #out_vba = wsd.classify(ds)
    #for i in range(len(out_vba)):
    #    out.append(out_vba[i])
    #    y.append(ds.getLabel(i))            

    clf_eval('wisard_virus_crop_12', y, out, classes = list(dict.fromkeys(y)))

if wisard_pairs_voting_crop:

    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd02  = wp.Wisard( 2, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
    wsd04  = wp.Wisard( 4, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
    wsd08  = wp.Wisard( 8, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
    wsd16  = wp.Wisard(16, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
    wsd32  = wp.Wisard(32, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
    wsd64  = wp.Wisard(64, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)

    print("Training...")

    ds_expiro = wp.DataSet("Expiro_crop_all.wpkds")
    ds_sality = wp.DataSet("Sality_crop_all.wpkds")
    ds_neshta = wp.DataSet("Neshta_crop_all.wpkds")
    
    wsd02.train(ds_expiro)
    wsd02.train(ds_sality)
    wsd02.train(ds_neshta)
    wsd04.train(ds_expiro)
    wsd04.train(ds_sality)
    wsd04.train(ds_neshta)
    wsd08.train(ds_expiro)
    wsd08.train(ds_sality)
    wsd08.train(ds_neshta)
    wsd16.train(ds_expiro)
    wsd16.train(ds_sality)
    wsd16.train(ds_neshta)
    wsd32.train(ds_expiro)
    wsd32.train(ds_sality)
    wsd32.train(ds_neshta)
    wsd64.train(ds_expiro)
    wsd64.train(ds_sality)
    wsd64.train(ds_neshta)
    
    print("Netword trained")

    print("Testing...")
    
    out = []
    y = []

    print("=== EXPIRO ===")
    ds = wp.DataSet("Expiro_test_crop_all.wpkds")
    for i in range(ds.size()):
        a = ds.get(i)
        example = []
        for j in range(a.size()):
            example.append(a.get(j))
        out02    = wsd02.classify([example])
        out04    = wsd04.classify([example])
        out08    = wsd08.classify([example])
        out16    = wsd16.classify([example])
        out32    = wsd32.classify([example])   
        out64    = wsd64.classify([example])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        out.append(most_frequent(finalResult))
        y.append("Expiro")
    
    print("=== NESHTA ===")
    ds = wp.DataSet("Neshta_test_crop_all.wpkds")
    for i in range(ds.size()):
        a = ds.get(i)
        example = []
        for j in range(a.size()):
            example.append(a.get(j))
        out02    = wsd02.classify([example])
        out04    = wsd04.classify([example])
        out08    = wsd08.classify([example])
        out16    = wsd16.classify([example])
        out32    = wsd32.classify([example])   
        out64    = wsd64.classify([example])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        out.append(most_frequent(finalResult))
        y.append("Neshta")
    
    print("=== SALITY ===")
    ds = wp.DataSet("Sality_test_crop_all.wpkds")
    for i in range(ds.size()):
        a = ds.get(i)
        example = []
        for j in range(a.size()):
            example.append(a.get(j))
        out02    = wsd02.classify([example])
        out04    = wsd04.classify([example])
        out08    = wsd08.classify([example])
        out16    = wsd16.classify([example])
        out32    = wsd32.classify([example])   
        out64    = wsd64.classify([example])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        out.append(most_frequent(finalResult))
        y.append("Sality")

    print("Tests done\n\n")
    
    clf_eval("wisard_voting_crop", y, out, classes=list(dict.fromkeys(y)))

### WiSARD
if wisard_pairs_voting:

    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd02  = wp.Wisard( 2, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd04  = wp.Wisard( 4, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd08  = wp.Wisard( 8, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd16  = wp.Wisard(16, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd32  = wp.Wisard(32, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd64  = wp.Wisard(64, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality"])
    
    wsd02.train(X_traincv, y_traincv)
    wsd04.train(X_traincv, y_traincv)
    wsd08.train(X_traincv, y_traincv)
    wsd16.train(X_traincv, y_traincv)
    wsd32.train(X_traincv, y_traincv)
    wsd64.train(X_traincv, y_traincv)
    
    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        out02    = wsd02.classify([X_testcv[i]])
        out04    = wsd04.classify([X_testcv[i]])
        out08    = wsd08.classify([X_testcv[i]])
        out16    = wsd16.classify([X_testcv[i]])
        out32    = wsd32.classify([X_testcv[i]])   
        out64    = wsd64.classify([X_testcv[i]])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        if most_frequent(finalResult) == "Expiro":
            print("Predicted Expiro, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Neshta":
            print("Predicted Neshta, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Sality":
            print("Predicted Sality, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "VBA":
            print("Predicted VBA, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
    print("Tests done")

    # the output of classify is a string list in the same sequence as the input
    #total = 0
    #corrects = 0
    #for count in range(len(y_testcv)):
    #    if y_testcv[count] == out[count]:
    #        corrects = corrects + 1
    #    total = total + 1
    #print("Chegou 1")
    #print("Keys: " + str(list(dict.fromkeys(y))))
    #clf_eval('wisard_virus_forget', y_testcv, out, classes = list(dict.fromkeys(y)))
    #new_y_testcv = y_testcv
    #new_out = out

    #for i in range(len(out)):
    #    if out[i] == 'Malign':
    #        new_out.append(1)
    #    else:
    #        new_out.append(0)

    #for i in range(len(y_testcv)):
    #    if y_testcv[i] == 'Malign':
    #        new_y_testcv.append(1)
    #    else:
    #        new_y_testcv.append(0)
   
    #f1.append(f1_score(new_y_testcv, new_out,average='micro'))
    #precision.append(precision_score(new_y_testcv, new_out, average='micro'))
    #recall.append(recall_score(new_y_testcv, new_out, average='micro'))
    #accuracy.append(float(corrects)/total)

   # print("f1 mean " + str(np.mean(f1)))
   # print("f1 std " + str(np.std(f1)))
   # print("Precision mean "+str(np.mean(precision)))
   # print("Precision std "+str(np.std(precision)))
   # print("Recall mean "+str(np.mean(recall)))
   # print("Recall std "+str(np.std(recall)))
   # print("Accuracy mean "+str(np.mean(accuracy)))
   # print("Accuracy std "+str(np.std(accuracy)))
   # print("==============================")

if wisard_pairs:

    print("WiSARD")

    addressSize = 20
    bleachingActivated=True
    ignoreZero=False

    ExpiroVBA     = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    ExpiroNeshta  = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    ExpiroSality  = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    NeshtaSality  = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    NeshtaVBA     = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    SalityVBA     = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","VBA","Neshta","Sality"])
    for i in range(len(X_traincv)):
        if y_traincv[i] == "Expiro":
            ExpiroVBA.    train([X_traincv[i]], [y_traincv[i]])      
            ExpiroNeshta. train([X_traincv[i]], [y_traincv[i]])      
            ExpiroSality. train([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "Neshta":
            ExpiroNeshta. train([X_traincv[i]], [y_traincv[i]])
            NeshtaSality. train([X_traincv[i]], [y_traincv[i]])
            NeshtaVBA.    train([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "Sality":
            ExpiroSality. train([X_traincv[i]], [y_traincv[i]])
            NeshtaSality. train([X_traincv[i]], [y_traincv[i]])
            SalityVBA.    train([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "VBA":
            ExpiroVBA.    train([X_traincv[i]], [y_traincv[i]])
            NeshtaVBA.    train([X_traincv[i]], [y_traincv[i]])
            SalityVBA.    train([X_traincv[i]], [y_traincv[i]])

    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        outExpiroVBA    = ExpiroVBA.    classify([X_testcv[i]])
        outExpiroNeshta = ExpiroNeshta. classify([X_testcv[i]])
        outExpiroSality = ExpiroSality. classify([X_testcv[i]])
        outNeshtaSality = NeshtaSality. classify([X_testcv[i]])
        outNeshtaVBA    = NeshtaVBA.    classify([X_testcv[i]])   
        outSalityVBA    = SalityVBA.    classify([X_testcv[i]])
        finalResult     = [outExpiroVBA[0], outExpiroNeshta[0], outExpiroSality[0], outNeshtaSality[0], outNeshtaVBA[0], outSalityVBA[0]]
        if finalResult.count("Expiro") == 3:
            print("Predicted Expiro, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if finalResult.count("Neshta") == 3:
            print("Predicted Neshta, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if finalResult.count("Sality") == 3:
            print("Predicted Sality, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if finalResult.count("VBA") == 3:
            print("Predicted VBA, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
    print("Tests done")

    # the output of classify is a string list in the same sequence as the input
    #total = 0
    #corrects = 0
    #for count in range(len(y_testcv)):
    #    if y_testcv[count] == out[count]:
    #        corrects = corrects + 1
    #    total = total + 1
    #print("Chegou 1")
    #print("Keys: " + str(list(dict.fromkeys(y))))
    #clf_eval('wisard_virus_forget', y_testcv, out, classes = list(dict.fromkeys(y)))
    #new_y_testcv = y_testcv
    #new_out = out

    #for i in range(len(out)):
    #    if out[i] == 'Malign':
    #        new_out.append(1)
    #    else:
    #        new_out.append(0)

    #for i in range(len(y_testcv)):
    #    if y_testcv[i] == 'Malign':
    #        new_y_testcv.append(1)
    #    else:
    #        new_y_testcv.append(0)
   
    #f1.append(f1_score(new_y_testcv, new_out,average='micro'))
    #precision.append(precision_score(new_y_testcv, new_out, average='micro'))
    #recall.append(recall_score(new_y_testcv, new_out, average='micro'))
    #accuracy.append(float(corrects)/total)

    #print("f1 mean " + str(np.mean(f1)))
    #print("f1 std " + str(np.std(f1)))
    #print("Precision mean "+str(np.mean(precision)))
    #print("Precision std "+str(np.std(precision)))
    #print("Recall mean "+str(np.mean(recall)))
    #print("Recall std "+str(np.std(recall)))
    #print("Accuracy mean "+str(np.mean(accuracy)))
    #print("Accuracy std "+str(np.std(accuracy)))
    #print("==============================")

if wisard_voting:

    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd02  = wp.Wisard( 2, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd04  = wp.Wisard( 4, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd08  = wp.Wisard( 8, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd16  = wp.Wisard(16, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd32  = wp.Wisard(32, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd64  = wp.Wisard(64, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","VBA","Neshta","Sality"])
    
    wsd02.train(X_traincv, y_traincv)
    wsd04.train(X_traincv, y_traincv)
    wsd08.train(X_traincv, y_traincv)
    wsd16.train(X_traincv, y_traincv)
    wsd32.train(X_traincv, y_traincv)
    wsd64.train(X_traincv, y_traincv)
    
    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        out02    = wsd02.classify([X_testcv[i]])
        out04    = wsd04.classify([X_testcv[i]])
        out08    = wsd08.classify([X_testcv[i]])
        out16    = wsd16.classify([X_testcv[i]])
        out32    = wsd32.classify([X_testcv[i]])   
        out64    = wsd64.classify([X_testcv[i]])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        if most_frequent(finalResult) == "Expiro":
            print("Predicted Expiro, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Neshta":
            print("Predicted Neshta, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Sality":
            print("Predicted Sality, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "VBA":
            print("Predicted VBA, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
    print("Tests done")

    # the output of classify is a string list in the same sequence as the input
    total = 0
    corrects = 0
    for count in range(len(y_testcv)):
        if y_testcv[count] == out[count]:
            corrects = corrects + 1
        total = total + 1
    print("Chegou 1")
    print("Keys: " + str(list(dict.fromkeys(y))))
    clf_eval('wisard_virus_forget', y_testcv, out, classes = list(dict.fromkeys(y)))
    new_y_testcv = y_testcv
    new_out = out

    #for i in range(len(out)):
    #    if out[i] == 'Malign':
    #        new_out.append(1)
    #    else:
    #        new_out.append(0)

    #for i in range(len(y_testcv)):
    #    if y_testcv[i] == 'Malign':
    #        new_y_testcv.append(1)
    #    else:
    #        new_y_testcv.append(0)
   
    f1.append(f1_score(new_y_testcv, new_out,average='micro'))
    precision.append(precision_score(new_y_testcv, new_out, average='micro'))
    recall.append(recall_score(new_y_testcv, new_out, average='micro'))
    accuracy.append(float(corrects)/total)

    print("f1 mean " + str(np.mean(f1)))
    print("f1 std " + str(np.std(f1)))
    print("Precision mean "+str(np.mean(precision)))
    print("Precision std "+str(np.std(precision)))
    print("Recall mean "+str(np.mean(recall)))
    print("Recall std "+str(np.std(recall)))
    print("Accuracy mean "+str(np.mean(accuracy)))
    print("Accuracy std "+str(np.std(accuracy)))
    print("==============================")
# ClusWiSARD

if cluswisard:
    print("ClusWiSARD")

    minScore = 0.1
    threshold = int(len(y_testcv)/20)
    discriminatorLimit = 20

    clus = wp.ClusWisard(addressSize, minScore, threshold, discriminatorLimit, verbose = True)

    print("Training...")

    start_train = time.time()
    if False:
        print("Supervised")
        clus.train(X_traincv,y_traincv)
    elif False:
        print("Semi-supervised")
        print(list(dict.fromkeys(y)))
        classes_dict = {
            1: "Expiro",
            2: "Neshta",
            3: "Sality"
        }
        clus.train(X_traincv, classes_dict)
    elif True:
        print("Unsupervised")
        clus.trainUnsupervised(X_traincv)
        out = clus.classifyUnsupervised(X_traincv)
        class_numbers = {
            "Expiro": [],
            "Neshta": [],
            "Sality": [],
        }
        for count in range(len(y_traincv)):
            print("Correct: " + str(y_traincv[count]) + "\t" + "Unsuperv.: " + str(out[count]))
            class_numbers[y_traincv[count]].append(out[count])
    finish_train = time.time()
    print(class_numbers)

    print("Netword trained")

    print("Testing...")
    # classify some data
    start_classify = time.time()
    total = 0
    ok = 0
    if False:
        out = clus.classify(X_testcv)
    elif True:
        out = clus.classifyUnsupervised(X_testcv)
        for count in range(len(y_testcv)):
            if out[count] in class_numbers[y_testcv[count]]:
                correct = True
                ok = ok + 1
            else:
                correct = False
            total = total + 1
            print("Correct: " + str(y_testcv[count]) + "\t\t" + "Unsuperv.: " + str(out[count]))
            # class_numbers[y_testcv[count]].append(out[count])
    finish_classify = time.time()
    #print(class_numbers)

    print("Tests done")

    print(total)
    print(ok)

    # the output of classify is a string list in the same sequence as the input
    total = 0
    corrects = 0
    for count in range(len(y_testcv)):
        if y_testcv[count] == out[count]:
            corrects = corrects + 1
        total = total + 1

    print("Keys: " + str(list(dict.fromkeys(y))))
    clf_eval('confusion_clus.png', y_testcv, out, classes = list(dict.fromkeys(y)))
    print("----------------------")
    print(float(corrects)/total)
    print(finish_train-start_train)
    print(finish_classify-start_classify)
