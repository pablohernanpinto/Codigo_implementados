import os
import smtplib
from email.message import EmailMessage
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilitar GPU
import time

import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay, balanced_accuracy_score
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.optimizers import Adam
#from keras.backend import expand_dims
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential,Model
from keras.constraints import MaxNorm
from keras.layers import Activation, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, SpatialDropout1D,Lambda,Input
from imblearn.over_sampling import SMOTE
import gc
from tensorflow.keras.losses import mse
import torch
import torch.nn as nn
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

from sklearn.decomposition import PCA
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d

import xgboost as xgb
from xgboost import XGBClassifier
# Deshabilitar GPU en TensorFlow
#tf.config.set_visible_devices([], 'GPU')
from sklearn.model_selection import RandomizedSearchCV
torch.manual_seed(42)
np.random.seed(42)


from sklearn.decomposition import PCA
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def Crear_modelo_XGBOOST(X_train, y_train):
    # Espacio de búsqueda bayesiana
    param_search = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 10),
        'n_estimators': Integer(50, 300),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0, 5),
        'reg_alpha': Real(1e-3, 10, prior='log-uniform'),
        'reg_lambda': Real(1e-3, 10, prior='log-uniform')
    }

    # Modelo base
    # Configuración del modelo base
    # Configuración del modelo base
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist',  # Usar 'gpu_hist' si tienes GPU
        n_jobs=-1
    )
    # Búsqueda bayesiana
    bayes_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=param_search,
        n_iter=10,         # Aumenta para mejor ajuste
        scoring='recall',  # Cambia a 'f1', 'accuracy', etc., si lo prefieres
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Entrenamiento
    bayes_search.fit(X_train, y_train)

    return bayes_search.best_estimator_


def normalizacion_XGBOOST(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def calcular_metricas(a,b,y_test,y_pred,model,X_test_total_scaled):
    tp = b[1]
    tn = a[0]
    fp = a[1]
    fn = b[0]
    
    exactitud = (tp + tn) / (tp + tn + fp + fn)
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0
    especificidad = tn/(tn + fp) if (tn + fp) else 0
    vpp = tp / (tp + fp) if (tp + fp) else 0
    vpn = tn / (tn + fn) if (tn + fn) else 0
    y_prob = model.predict_proba(X_test_total_scaled)[:, 1]  # Probabilidades para la clase positiva
    PRC = average_precision_score(y_test, y_prob)  # PRC
    AUC = roc_auc_score(y_test, y_prob)
    #print('----------------------------------------------------------------')
    #print('accuracy',accuracy,'precision',precision,'recall',recall,'f1','PRC',PRC,'AUC',AUC)
    return exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn

# Función de entrenamiento base para XGBOOST
def entrenamiento_base_XGBOOST(X_train, X_test, y_train, y_test):
    X_train_scaled, X_test_scaled = normalizacion_XGBOOST(X_train, X_test)
    
    model = Crear_modelo_XGBOOST(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para la clase positiva
    a,b = confusion_matrix(y_test, y_pred)

    exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = calcular_metricas(a,b,y_test,y_pred,model,X_test_scaled)
    
    return '1. Entrenamiento base',exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn

# Función para aplicar SMOTE con XGBOOST
def Aplicar_Smote_XGBOOST(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)
    X_train_scaled, X_test_scaled = normalizacion_XGBOOST(X_resampled_smote, X_test)

    model = Crear_modelo_XGBOOST(X_train_scaled, y_resampled_smote)
    y_pred = model.predict(X_test_scaled)
    a,b = confusion_matrix(y_test, y_pred)
    exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = calcular_metricas(a,b,y_test,y_pred,model,X_test_scaled)

    return '2. SMOTE',exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn

def Aplicar_VAE_XGBOOST(X_train, X_test, y_train, y_test):

    df_xtrain = pd.DataFrame(X_train)
    df_xtrain.columns = df_xtrain.columns.astype(str)  # Convertir nombres a strings
    
    df_ytrain = pd.DataFrame(y_train, columns=['antibiotico'])
    minority_class = pd.concat([df_xtrain, df_ytrain], axis=1)
    minority_class = minority_class.query('antibiotico == 1')
    
    # Excluir la columna objetivo antes de escalar
    minority_class = minority_class.drop(columns=['antibiotico'])
    scaler = MinMaxScaler()
    X_minority_scaled = scaler.fit_transform(minority_class)
    # Dimensiones
    input_dim = X_minority_scaled.shape[1]
    latent_dim = 2  # Espacio latente

    # Encoder
    inputs = Input(shape=(input_dim,))
    hidden = Dense(16, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(hidden)
    z_log_var = Dense(latent_dim, name='z_log_var')(hidden)

    # Sampling
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    decoder_hidden = Dense(16, activation='relu')
    decoder_output = Dense(input_dim, activation='sigmoid')

    hidden_decoded = decoder_hidden(z)
    outputs = decoder_output(hidden_decoded)

    # Modelo VAE
    vae = Model(inputs, outputs)

    # Pérdida personalizada
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    vae.summary()
    vae.fit(X_minority_scaled, X_minority_scaled, epochs=200, batch_size=32, verbose=1)
    # Construir el generador (Decoder independiente)
    decoder_input = Input(shape=(latent_dim,))
    hidden_decoded_2 = decoder_hidden(decoder_input)
    output_decoded = decoder_output(hidden_decoded_2)
    generator = Model(decoder_input, output_decoded)

    # Generar datos sintéticos
    print(pd.Series(y_train).value_counts())
    num_samples = pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1]
    latent_points = np.random.normal(size=(num_samples, latent_dim))
    synthetic_data = generator.predict(latent_points)


    # Escalar de vuelta a los valores originales
    synthetic_data_original = scaler.inverse_transform(synthetic_data)
    X_train_balanced = np.concatenate([X_train, synthetic_data_original])
    y_train_balanced = np.concatenate([y_train, np.ones(num_samples)])

    X_train_reshaped,X_test_reshaped = normalizacion_XGBOOST(X_train_balanced, X_test)
    
    model = Crear_modelo_XGBOOST(X_train_reshaped,y_train_balanced)

    y_pred = model.predict(X_test_reshaped)
    a,b = confusion_matrix(y_test, y_pred)
    exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = calcular_metricas(a,b,y_test,y_pred,model,X_test_reshaped)
    return '3. VAE',exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn



def Aplicar_DifussionModel_XGBOOST(X_train, X_test, y_train, y_test):

    df_xtrain = pd.DataFrame(X_train)
    df_xtrain.columns = df_xtrain.columns.astype(str)  # Convertir nombres a strings
    
    df_ytrain = pd.DataFrame(y_train, columns=['antibiotico'])
    minority_class = pd.concat([df_xtrain, df_ytrain], axis=1)
    minority_class = minority_class.query('antibiotico == 1')
    
    # Excluir la columna objetivo antes de escalar
    minority_class = minority_class.drop(columns=['antibiotico'])
    # Preprocesamiento
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(minority_class)
    # Modelo de Difusión
    class DiffusionModel(nn.Module):
        def __init__(self, input_dim):
            super(DiffusionModel, self).__init__()
            self.model = nn.Sequential( 
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),  # Regularización Dropout
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2),  # Regularización Dropout
                nn.Linear(32, input_dim)
            )
        def forward(self, x):
            return self.model(x)
    # Función de ruido (Scheduler)
    def add_noise(data, timesteps, noise_scale=1.0):
        noise = np.random.normal(0, noise_scale, data.shape) * np.sqrt(timesteps / 100)
        noisy_data = data + noise
        return noisy_data, noise
    
    # Configuración del modelo
    input_dim = scaled_data.shape[1]
    model = DiffusionModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()  # O Huber Loss

    # Scheduler de tasa de aprendizaje
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Entrenamiento
    scaled_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    epochs = 700
    losses = []  # Para guardar la pérdida por época

    for epoch in range(epochs):
        timesteps = np.random.randint(1, 100)
        noisy_data, noise = add_noise(scaled_data, timesteps)
        noisy_data_tensor = torch.tensor(noisy_data, dtype=torch.float32)
        noise_tensor = torch.tensor(noise, dtype=torch.float32)

        optimizer.zero_grad()
        predicted_noise = model(noisy_data_tensor)
        loss = loss_fn(predicted_noise, noise_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Actualiza la tasa de aprendizaje

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item()}")

        # Generación de Datos Sintéticos
    def generate_synthetic_data(model, num_samples, input_dim):
        model.eval()
        with torch.no_grad():
            synthetic_data = np.random.normal(0, 1, (num_samples, input_dim))
            for t in range(100, 0, -1):  # Reverse diffusion
                synthetic_data = synthetic_data - model(torch.tensor(synthetic_data, dtype=torch.float32)).numpy() * (t / 100)
            return synthetic_data
        
        
    synthetic_data = generate_synthetic_data(model, pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1], input_dim)
    synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

        
    # Cambiar el tipo de datos a float32
    synthetic_samples_numpy = synthetic_data_rescaled.astype(np.float32)

    # Mostrar las muestras generadas
    synthetic_samples_numpy.shape

    X_train_resampled = np.concatenate([X_train,synthetic_samples_numpy])

    ones_array = np.ones(pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1])
    y_train_resampled = np.concatenate([y_train,ones_array])

    #termino de oversampling
    #    X_train_reshaped,X_test_reshaped,X_test_total_scaled = normalizacion_XGBOOST(X_train_balanced, X_test,X_test_total)

    X_train_reshaped,X_test_reshaped = normalizacion_XGBOOST(X_train_resampled, X_test)
    
    model = Crear_modelo_XGBOOST(X_train_reshaped,y_train_resampled)

    y_pred = model.predict(X_test_reshaped)

    a,b = confusion_matrix(y_test, y_pred)
    exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = calcular_metricas(a,b,y_test,y_pred,model,X_test_reshaped)

    #model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped
    return '4. DIFFUSION MODEL',exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn

def Aplicar_Copulas_XGBOOST(X_train, X_test, y_train, y_test):
    df_xtrain = pd.DataFrame(X_train)
    df_xtrain.columns = df_xtrain.columns.astype(str)  # Convertir nombres a strings
    
    df_ytrain = pd.DataFrame(y_train, columns=['antibiotico'])
    minority_class = pd.concat([df_xtrain, df_ytrain], axis=1)
    minority_class = minority_class.query('antibiotico == 1')
    
    # Excluir la columna objetivo antes de escalar
    minority_class = minority_class.drop(columns=['antibiotico'])

    # Paso 2: Escalar los datos (opcional pero recomendable)
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(minority_class), columns=minority_class.columns)

    pca = PCA(n_components=10)
    scaled_data_reduced = pca.fit_transform(scaled_data)


    # Paso 3: Entrenar la cópula
    copula = GaussianMultivariate()
    copula.fit(scaled_data_reduced)

    # Paso 4: Generar datos sintéticos
    n_samples = pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1]  # <-- define la cantidad que necesitas
    synthetic_scaled_data = copula.sample(n_samples)

    synthetic_scaled_data = pca.inverse_transform(synthetic_scaled_data)

    # Paso 5: Desescalar los datos para volver a su forma original
    synthetic_data = pd.DataFrame(scaler.inverse_transform(synthetic_scaled_data), columns=minority_class.columns)

    # Ahora tienes synthetic_data como oversampling del conjunto minoritario

    X_train_resampled = np.concatenate([X_train,synthetic_data])

    ones_array = np.ones(int((pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1])))
    y_train_resampled = np.concatenate([y_train,ones_array])



    X_train_reshaped,X_test_reshaped = normalizacion_XGBOOST(X_train_resampled, X_test)
    
    model = Crear_modelo_XGBOOST(X_train_reshaped,y_train_resampled)

    y_pred  = model.predict(X_test_reshaped)
    y_pred = (y_pred>0.5)
    a,b = confusion_matrix(y_test, y_pred)
    exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = calcular_metricas(a,b,y_test,y_pred,model,X_test_reshaped)

    return '5. COPULAS',exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn

def columnas_bacterias_fun(df):
    vocales = ['a','e','i','o','u','A','E','I','O','U']
    columnas_bacterias = []
    for i in vocales:
        for j in df.columns:
            if i in j:
                columnas_bacterias.append(j)
    columnas_bacterias = list(set(columnas_bacterias))
    return columnas_bacterias


#  mejor_modelo = comparativa_anidado(accuracy,precision,recall,PRC,AUC)

def comparativa_anidado(accuracy,precision,recall,PRC,AUC,best_accuracy,best_precision,best_recall,best_f1,best_PRC,best_AUC):

    if best_accuracy is None:
        return accuracy, precision, recall, f1, PRC, AUC
    # Actualiza solo si f1 y PRC son mejores o iguales
    if (f1 >= best_f1) and (PRC >= best_PRC):
        return accuracy, precision, recall, f1, PRC, AUC
    else:
        return best_accuracy, best_precision, best_recall, best_f1, best_PRC, best_AUC

# Función para evaluar métricas con XGBOOST                 inscripcion_resultados_XGBOOST(best_accuracy,best_precision,best_recall,best_f1,best_PRC,best_AUC)

def inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tiempo,tp,tn,fp,fn):
    
    
    df_resultados_aux.loc[len(df_resultados_aux)] = {
        'BD': archivo,
        'Nombre antibiotico': bacteria,
        'Metodo de Oversampling': tipo_entrenamiento,
        'exactitud': exactitud,
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'vpp': vpp,
        'vpn': vpn,
        'PRC': PRC,
        'AUC': AUC,
        'Tiempo (s)':tiempo,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        
    }


def Correo(correo,contenido):
    # Crear el mensaje
    msg = EmailMessage()
    msg['Subject'] = correo
    msg['From'] = 'pablohernanpinto@gmail.com'
    msg['To'] = 'pablo.pinto@alu.ucm.cl'
    msg.set_content(contenido)

    # Enviar el correo
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login('pablohernanpinto@gmail.com', 'vogt hjif tlmi caxm')
            smtp.send_message(msg)
        print("Correo enviado exitosamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")



files_list = os.listdir('SetDatos/')

for archivo in  ['e_coli_driams_b_2000_20000Da_v2 (1).csv',  'bin_10_2000Da_10000Da_talca_e_coli_v2.csv' ] : #files_list ['bin_10_2000Da_10000Da_talca_e_coli_v2.csv'] ['e_coli_driams_b_2000_20000Da_v2 (1).csv',  'bin_10_2000Da_10000Da_talca_e_coli_v2.csv' ]
    print(archivo,'imprimiendo archivo')
    df = pd.read_csv('SetDatos/'+archivo)
    #print(df)
    columnas_a_eliminar = [
        'code', 'species', 'filename', 'codigo_de_barras', 
        'fecha_de_extraccion', 'tipo_de_muestra', 'especie'
    ]   
    df.drop(columns=columnas_a_eliminar, errors='ignore', inplace=True)
    #print(df)
    columnas_bacterias = columnas_bacterias_fun(df)
    #print(columnas_bacterias,'aqui')

    df_resultados_aux = pd.DataFrame({
        'BD': [],
        'Nombre antibiotico': [],
        'Metodo de Oversampling': [],
        'exactitud': [],
        'sensibilidad': [],
        'especificidad': [],
        'vpp': [],
        'vpn': [],
        'AUC': [],
        'PRC': [],
        'Tiempo (s)':[],
        'tp': [],
        'tn': [],
        'fp': [],
        'fn': [],
    })
    
    
    

    for bacteria in columnas_bacterias : #   columnas_bacterias  ['CEFTRIAXONA'] 
        try:
            print(bacteria,'esta bacteria')
            columnas_bacterias_sin_bacteria = [b for b in columnas_bacterias if b != bacteria] #bacteria
            df_bacteria = df.drop(columns = columnas_bacterias_sin_bacteria)
            df_bacteria.dropna(axis=0, how="any", inplace=True)
            print(df_bacteria.shape)
            bacteria = df_bacteria.columns[-1]
            X = df_bacteria.iloc[:, 0:-1].values  # variables independientes (espectros de masa)
            y = df_bacteria.iloc[:, -1].values    # variable dependientes (resistencia a ciprofloxacin)
            X = np.asarray(X).astype(np.float32)
            y = np.asarray(y).astype(np.float32)

            #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

            folds = 10
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            
            for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
                # División
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                #resultado sin oversampling
                start_time = time.time()
                tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = entrenamiento_base_XGBOOST(X_train_fold, X_val_fold, y_train_fold, y_val_fold)
                end_time = time.time()
                tiempo = round(end_time - start_time,2)

                inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tiempo,tp,tn,fp,fn)
            
                # Liberar memoria
                del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
                gc.collect()  # Forzar recolección de basura

            #SMOTE
            for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
                # División
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                #resultado sin oversampling
                start_time = time.time()

                tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_Smote_XGBOOST(X_train_fold, X_val_fold, y_train_fold, y_val_fold)
                end_time = time.time()
                tiempo = round(end_time - start_time,2)

                inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tiempo,tp,tn,fp,fn)
            
                # Liberar memoria
                del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
                gc.collect()  # Forzar recolección de basura
            #VAE
            for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
                # División
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                #resultado sin oversampling
                start_time = time.time()

                tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_VAE_XGBOOST(X_train_fold, X_val_fold, y_train_fold, y_val_fold)

                end_time = time.time()
                tiempo = round(end_time - start_time,2)


                inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tiempo,tp,tn,fp,fn)
            
                # Liberar memoria
                del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
                gc.collect()  # Forzar recolección de basura
            #DIFFUSION MODEL
            for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
                # División
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                #resultado sin oversampling
                start_time = time.time()

                tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_DifussionModel_XGBOOST(X_train_fold, X_val_fold, y_train_fold, y_val_fold)
                end_time = time.time()
                tiempo = round(end_time - start_time,2)

                inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tiempo,tp,tn,fp,fn)
            
                # Liberar memoria
                del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
                gc.collect()  # Forzar recolección de basura

            #COPULAS
            for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
                # División
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                #resultado sin oversampling
                start_time = time.time()


                tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_Copulas_XGBOOST(X_train_fold, X_val_fold, y_train_fold, y_val_fold)

                end_time = time.time()
                tiempo = round(end_time - start_time,2)


                inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tiempo,tp,tn,fp,fn)
            
                # Liberar memoria
                del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
                gc.collect()  # Forzar recolección de basura 
            """ 
            tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = entrenamiento_base_XGBOOST(X_train, X_test, y_train, y_test)

            inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn)
            
            del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
            gc.collect() """
            """  
            #Entrenamiento Smote
            tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_Smote_XGBOOST(X_train, X_test, y_train, y_test)
            inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn)
            del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
            gc.collect()
             
            
            #Entrenamiento VAE
            tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_VAE_XGBOOST(X_train, X_test, y_train, y_test)
            inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn)
            del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
            gc.collect()
             
            #Entrenamiento Diffusion model
            tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_DifussionModel_XGBOOST(X_train, X_test, y_train, y_test)
            inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn)
            del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
            gc.collect()
             
            #Entrenamiento Copula
            tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn = Aplicar_Copulas_XGBOOST(X_train, X_test, y_train, y_test)
            inscripcion_resultados_XGBOOST(archivo, bacteria,tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn)
            del tipo_entrenamiento,exactitud,sensibilidad,especificidad,vpp,vpn,PRC,AUC,tp,tn,fp,fn
            gc.collect() 
            
            
            ['e_coli_driams_b_2000_20000Da_v2 (1).csv',  'bin_10_2000Da_10000Da_talca_e_coli_v2.csv' ] 


            """
             
            
        except Exception  as e:
            print('este es el error ', e)
            Correo('Error',str(e))
        else:
            pass
            df_resultados_aux = df_resultados_aux.sort_values(by=['Nombre antibiotico','Metodo de Oversampling'])
            #print(df_resultados_aux)
            df_resultados_aux.to_excel('resultadosExcelXGBOOST/XGBOOST-'+archivo+'.xlsx',index=False)
            Correo('se inscribio',archivo)
Correo('Finalizo todo el programa','XGBOOST')

