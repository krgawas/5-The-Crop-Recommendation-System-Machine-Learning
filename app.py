import pickle
import numpy as np
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/form', methods=["POST"])
def brain():
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])
     
    values = np.array([Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]).reshape(1, -1)
    
    if 0 < Ph <= 14 and Temperature < 100 and Humidity > 0:
        # Load the model with explicit dtype specification
        with open('savedmodel.sav', 'rb') as model_file:
            load_model = pickle.load(model_file)
            expected_dtype = [('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8'), ('missing_go_to_left', 'u1')]
            load_model.tree_.__dict__['values'] = load_model.tree_.__dict__['values'].astype(expected_dtype)
        
        acc = load_model.predict(values)[0]
        return render_template('prediction.html', prediction=str(acc))
    else:
        return "Sorry... Error in entered values in the form. Please check the values and fill it again"

if __name__ == '__main__':
    app.run(debug=True)
