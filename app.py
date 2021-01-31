from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from keras.models import model_from_json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding,Dropout,LSTM, Input, InputLayer, Dense, Bidirectional
from tensorflow.keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

################################################################################################

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")

#Toeknizer loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

################################################################################################

#cv=pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		test_data = tokenizer.texts_to_sequences(data)
		test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=200,padding='post')
		my_prediction = loaded_model.predict_classes(test_data)
		print(my_prediction)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)