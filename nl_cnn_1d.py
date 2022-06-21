# NL_CNN MODEL
# Returns a precompiled model with a specific optimizer included
#==============================================================================================
import keras.optimizers
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Conv1D, MaxPooling1D, \
  GlobalAveragePooling1D
from keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D  # straturi convolutionale si max-pooling


def create_nl_cnn_model(input_shape, num_classes, k=1.5,separ=0, flat=0, width=80, nl=(3,2), add_layer=0, learning_rate=0.01, dropout=0.5):
  # Arguments: k - multiplication coefficient
  # Structure parameteres
  kfil=k
  filtre1=width ; filtre2=int(kfil*filtre1) ; filtre3=(kfil*filtre2)  # filters (kernels) per each layer - efic. pe primul
  nr_conv=3 # 0, 1, 2 sau 3  (number of convolution layers)
  csize1=3; csize2=3 ; csize3=3      # convolution kernel size (square kernel)
  psize1=4; psize2=4 ; psize3=4      # pooling size (square)
  str1=2; str2=2; str3=2             # stride pooling (downsampling rate)
  pad='same' # padding style ('valid' is also an alternative)
  nonlinlayers1=nl[0]  # total of layers (with RELU nonlin) in the first maxpool layer  # De parametrizat asta
  nonlinlayers2=nl[1]  #

  nonlin_type='relu' # may be other as well 'tanh' 'elu' 'softsign'
  bndrop=1 # include BatchNorm inainte de MaxPool si drop(0.3) dupa ..

  cvdrop=1 # droput
  if dropout == 0:
    cvdrop = 0
  drop_cv=dropout

  model = Sequential()
  # convolution layer1  ==========================================================================
  # Initially first layer was always a Conv2D one
  if separ==1:
    model.add(SeparableConv2D(filtre1, padding=pad, kernel_size=(csize1, csize1), input_shape=input_shape) )
  elif separ==0:
    model.add(Conv1D(filtre1, padding=pad, kernel_size=csize1, input_shape=input_shape) )

  # next are the additional layers
  for nl in range(nonlinlayers1-1):
    model.add(Activation(nonlin_type))  # Activ NL-CNN-1
    if separ==1:
      model.add(SeparableConv2D(filtre1, padding=pad, kernel_size=(csize1, csize1) ) ) # Activ NL-CNN-2
    elif separ==0:
      model.add(Conv1D(filtre1, padding=pad, kernel_size=csize1)) # Activ NL-CNN-2
  #  MaxPool in the end of the module
  if bndrop==1:
    model.add(BatchNormalization())
  model.add(MaxPooling1D(pool_size=psize1,strides=str1,padding=pad))
  if cvdrop==1:
    model.add(Dropout(drop_cv))

  # NL LAYER 2 =======================================================================================================

  if separ==1:
    model.add(SeparableConv2D(filtre2, padding=pad, kernel_size=(csize2, csize2)) )
  elif separ==0:
    model.add(Conv1D(filtre2, padding=pad, kernel_size=csize2))
  # aici se adauga un neliniar

  #=========== unul extra NL=2 pe strat 2 =====================
  for nl in range(nonlinlayers2-1):
    model.add(Activation(nonlin_type))  # Activ NL-CNN-1
    if separ==1:
        model.add(SeparableConv2D(filtre2, padding=pad, kernel_size=(csize2, csize2)) ) # Activ NL-CNN-2
    elif separ==0:
        model.add(Conv1D(filtre2, padding=pad, kernel_size=csize2)) # Activ NL-CNN-2

  # OUTPUT OF LAYER 2 (MAX-POOL)
  if bndrop==1:
      model.add(BatchNormalization())
  model.add(MaxPooling1D(pool_size=psize2,strides=str2,padding=pad))
  if cvdrop==1:
      model.add(Dropout(drop_cv))
  #-------------------------------------------------------------------------------------------
  # LAYER 3

  if separ==1:
      model.add(SeparableConv2D(filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
  elif separ==0:
      model.add(Conv1D(filtre3, padding=pad, kernel_size=csize3) ) # Activ NL-CNN-2
  # OUTPUT OF LAYER 3
  if bndrop==1:
      model.add(BatchNormalization())
  model.add(MaxPooling1D(pool_size=psize3,strides=str3,padding=pad))
  if cvdrop==1:
      model.add(Dropout(drop_cv))
  #-------------------
  #
  # LAYER 4  (only if requested - for large images ?? )
  if add_layer==1:
    if separ==1:
      model.add(SeparableConv2D(1.2*filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
    elif separ==0:
      model.add(Conv2D(1.2*filtre3, padding=pad, kernel_size=(csize3, csize3)) ) # Activ NL-CNN-2
    # OUTPUT OF LAYER 4
    if bndrop==1:
      model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(psize3, psize3),strides=(str3,str3),padding=pad))
    if cvdrop==1:
      model.add(Dropout(drop_cv))
  #========================================================================================
  # INPUT TO DENSE LAYER (FLATTEN - more data can overfit / GLOBAL - less data - may be a good choice )
  if flat==1:
      model.add(Flatten())  #
  elif flat==0:
      model.add(GlobalAveragePooling1D()) # Global average

  model.add(Dense(num_classes, activation='softmax'))
  # END OF MODEL DESCRIPTION
  # ------------------ COMPILE THE MODEL
  myopt = tensorflow.keras.optimizers.SGD(learning_rate=learning_rate)
  #myopt = Nadam()

  # --------------------------   LOSS function  ------------------------------------
  my_loss='categorical_crossentropy'
  model.compile(loss=my_loss,
              optimizer=myopt,
              metrics=['accuracy'])

  return model