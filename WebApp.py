from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask('__name__')
model=pickle.load(open('model.pkl','rb'))

@app.route('/View')
def home():
    return render_template('PredictView.html')

@app.route('/View',methods=["POST"])
def predict():
    feature=[int(x) for x in request.form.values()]
    feature_final=np.array(feature).reshape(-1,1)
    prediction=model.predict(feature_final)
    return render_template('PredictView.html',prediction_text='Price of Salary {}'.format(int(prediction)))

if(__name__=='__main__'):
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True)

