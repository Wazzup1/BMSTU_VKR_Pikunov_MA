from flask import Flask, request, render_template

import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def choose_prediction_method():
    return render_template('main.html')


def mn_prediction(params):
    model = tf.keras.models.load_model('models/mn_model_0.73')
    pred = model.predict([params])
    return pred

def pr_prediction(params):
    model = tf.keras.models.load_model('models/pr_model_375')
    pred = model.predict([params])
    return pred

def upr_prediction(params):
    model = tf.keras.models.load_model('models/upr_model_2.62')
    pred = model.predict([params])
    return pred


@app.route('/mn/', methods=['POST', 'GET'])
def mn_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mup', 'pr', 'ps', 'yn', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Спрогнозированное Соотношение матрица-наполнитель для введенных параметров: {mn_prediction(params)}'
    return render_template('mn.html', message=message)

@app.route('/pr/', methods=['POST', 'GET'])
def pr_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mup', 'ps', 'yn', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Спрогнозированное значение Прочности при растяжении для введенных параметров: {pr_prediction(params)} МПа'
    return render_template('pr.html', message=message)

@app.route('/upr/', methods=['POST', 'GET'])
def upr_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'pr', 'ps', 'yn', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Спрогнозированное значение Модуля упругости при растяжении для введенных параметров: {upr_prediction(params)} ГПа'
    return render_template('upr.html', message=message)

if __name__ == '__main__':
    app.run()
