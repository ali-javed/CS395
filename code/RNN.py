import scipy.io as spio
import numpy as np




def readMat(dataPath):
    
    
    #readMat: Reads the .mat for mad river data files placed at data path
    #INPUT
    #dataPath == relative location of the data path to the code file, or absolute
    #
    #OUTPUT
    #still being worked on
    #eventToClass== np array of all classes belonging to all 616events. value at 0 index is class for 0th event.
    #sedimentData_events == list containing array for sediment data for each event
    #
    #
    #events == list containing 2-D array for each event. 1st column is sediment and 2nd is stream flow. Ideally we will be using this
    #maxEventLen == Longest event in terms of timesteps
    
    #sample call: eventToClass, myEvents, maxEventLen,streamFlow_Data,sedimentData_events = readMat('..\data\')
    
    #Programmer: Ali Javed
    #Date last modified: 27 Feb 2018
    #modified by: Ali Javed
    #Comments: Initial version.
    
    
    ##############################################################################################################
        
    
    
    
    classMat = spio.loadmat(dataPath + 'allMadSitesStormHystClassKey.mat', squeeze_me=True)
    dataMat = spio.loadmat(dataPath + 'allMadSitesEventTimeSeries.mat', squeeze_me=True)
    
    eventToClass = classMat['stormHystClass'][:,3] #index 3 refers to class of 3rd event. Event number start from 0
    eventToClass = eventToClass.astype(int) # we do not need float classes




    #gather 626 events
    events = []
    sedimentData_events = []
    streamFlowData_events = []
    
    maxEventLen = -1 #need this as fixed size input to keras RNN
    
    streamFlow = 1
    suspendedSedimentConcentration = 2
    
    for event in range(0,len(dataMat['dataTSOut'])):
    
    
        #not reading datetime and rainfall data for now
        #event_dataTime = np.zeros((len(dataMat['dataTSOut'][event][streamFlow]))) #can not extract datetime so setting it to one, for out purpose it does not matter anyways
        #event_rainFall = np.zeros((len(dataMat['dataTSOut'][event][streamFlow])))
                             
        event_streamflow = dataMat['dataTSOut'][event][streamFlow]
        event_suspendedSedimentConcentration = dataMat['dataTSOut'][event][suspendedSedimentConcentration]
    
        
        eventArray = np.column_stack((event_streamflow,event_suspendedSedimentConcentration))
        
        
        events.append(eventArray)
        sedimentData_events.append(event_streamflow)
        streamFlowData_events.append(event_suspendedSedimentConcentration)
    
        if len(event_streamflow)> maxEventLen:
            maxEventLen = len(event_streamflow)
    
    
        
        ##############################################################################################################
        #for classification based only on rain and sediment... i can not figure out how to give 2d input to RNN
        
        
        #classVector = np.repeat(eventToClass[event], len(event_streamflow))
        #print(np.shape(classVector))
        #print(np.shape(suspendedSedimentConcentration))
        #streamFlow_Data = np.column_stack((event_streamflow,classVector))
        #suspendedSedimentConcentration_Data = np.column_stack((event_suspendedSedimentConcentration,classVector))
        
    return eventToClass, events, maxEventLen, streamFlowData_events, sedimentData_events
    
    
##############################################################################################################
#MAIN STARTS HERE

       
                     
dataPath = '../data/'
#get data
eventToClass, myEvents, maxEventLen,streamFlow_Data,sedimentData_events = readMat(dataPath)


                     

##############################################################################################################


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical





#############################################################################
#DECLARE PARAMETERS FOR NN

# fix random seed for reproducibility
np.random.seed(7)


#these parameters are the architecture of the RNN. Still have to do more on this
embedding_vecor_length = 32  #each event is represented using a 32 length vector. 
len_input = 700  #i do not know what purpose these vectors do






###############################################################################






#train test split not being done for now

#preprocess data to required format, padding to make all sequence data same lenght
X_train = sequence.pad_sequences(sedimentData_events,dtype='float')
#create one hot representation for class values
y_train = to_categorical(eventToClass, num_classes=None)






###############################################################################
#CREATE SIMPLE RNN


# create the model

model = Sequential()

model.add(Embedding(len_input, embedding_vecor_length, input_length=maxEventLen))
#add the recurrent LSTM layer of 100 nodes
model.add(LSTM(100))

#out put layer with 16 nodes, one hot representation
model.add(Dense(16, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#test and train both on same data because did not split for now.
model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=1, batch_size=64)





 
