import numpy as np
import pickle

# =========================================================================================================#
# ====================================== IMPORT DATA ======================================================#
# =========================================================================================================#
print("Importing data...")

preproc = "thermometer12_"

print("File prefix: " + preproc)

thermometer_0000_0999X = pickle.load( open( preproc + "0_999X.p", "rb" ) )
thermometer_1000_1999X = pickle.load( open( preproc + "1000_1999X.p", "rb" ) )
thermometer_2000_2999X = pickle.load( open( preproc + "2000_2999X.p", "rb" ) )
thermometer_3000_3999X = pickle.load( open( preproc + "3000_3999X.p", "rb" ) )
thermometer_4000_4999X = pickle.load( open( preproc + "4000_4999X.p", "rb" ) )
thermometer_5000_5999X = pickle.load( open( preproc + "5000_5999X.p", "rb" ) )
thermometer_6000_6999X = pickle.load( open( preproc + "6000_6999X.p", "rb" ) )
thermometer_7000_7999X = pickle.load( open( preproc + "7000_7999X.p", "rb" ) )
thermometer_8000_8999X = pickle.load( open( preproc + "8000_8999X.p", "rb" ) )
thermometer_9000_9099X = pickle.load( open( preproc + "9000_9099X.p", "rb" ) )

thermometer_0000_0999y = pickle.load( open( preproc + "0_999y.p", "rb" ) )
thermometer_1000_1999y = pickle.load( open( preproc + "1000_1999y.p", "rb" ) )
thermometer_2000_2999y = pickle.load( open( preproc + "2000_2999y.p", "rb" ) )
thermometer_3000_3999y = pickle.load( open( preproc + "3000_3999y.p", "rb" ) )
thermometer_4000_4999y = pickle.load( open( preproc + "4000_4999y.p", "rb" ) )
thermometer_5000_5999y = pickle.load( open( preproc + "5000_5999y.p", "rb" ) )
thermometer_6000_6999y = pickle.load( open( preproc + "6000_6999y.p", "rb" ) )
thermometer_7000_7999y = pickle.load( open( preproc + "7000_7999y.p", "rb" ) )
thermometer_8000_8999y = pickle.load( open( preproc + "8000_8999y.p", "rb" ) )
thermometer_9000_9099y = pickle.load( open( preproc + "9000_9099y.p", "rb" ) )

print("Joining...")

thermometer_X = thermometer_0000_0999X + thermometer_1000_1999X + thermometer_2000_2999X + thermometer_3000_3999X + thermometer_4000_4999X + thermometer_5000_5999X + thermometer_6000_6999X + thermometer_7000_7999X + thermometer_8000_8999X + thermometer_9000_9099X

thermometer_y = list(thermometer_0000_0999y) + list(thermometer_1000_1999y) + list(thermometer_2000_2999y) + list(thermometer_3000_3999y) + list(thermometer_4000_4999y) + list(thermometer_5000_5999y) + list(thermometer_6000_6999y) + list(thermometer_7000_7999y) + list(thermometer_8000_8999y) + list(thermometer_9000_9099y)

pickle.dump (thermometer_X, open( preproc + "X.p", "wb") )
pickle.dump (thermometer_y, open( preproc + "y.p", "wb") )

print("Done!")
