import numpy as np
import scipy.stats as stats
import flask

app = flask.Flask(__name__)

def get_posteriori(w_prior, s_prior, w_actual, s_actual, w_measured_arr):
    l_measured = stats.norm.pdf(w_measured_arr, w_actual, s_actual)
    likelihood = np.prod(l_measured)
    weighting = stats.norm.pdf(w_actual, w_prior, s_prior)
    posteriori = likelihood * weighting
    return posteriori
    
def ValuePredictor(request_params):
    w_measured_arr = []
    for str in request_params['w_measured'].split(','):
        w_measured_arr.append(float(str.strip()))
    
    s_actual = float(request_params['s_actual'])
        
    w_prior = float(request_params['w_prior'])
    s_prior = float(request_params['s_prior'])
    
    print(f'w_measured : {w_measured_arr}')
    print(f's_actual   : {s_actual}')
    print(f'w_prior    : {w_prior}')
    print(f's_prior    : {s_prior}')
    
    w_actual_arr = np.arange(10, 200, 0.1)
    posteriori_arr = []
        
    for w_actual in w_actual_arr:
        posteriori = get_posteriori(w_prior, s_prior, w_actual, s_actual, w_measured_arr)
        posteriori_arr.append(posteriori)

    peak_location = w_actual_arr[np.argmax(posteriori_arr)]
    print(f'Peak location: {peak_location:.1f}')
    
    return peak_location

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    if flask.request.method == 'POST':
        request_params = flask.request.form.to_dict()
        
        result = ValuePredictor(request_params)
        prediction = f'{result:.1f}'
        
        return flask.render_template("index.html", 
                                     w_prior=request_params['w_prior'], 
                                     s_prior=request_params['s_prior'], 
                                     w_measured=request_params['w_measured'], 
                                     s_actual=request_params['s_actual'], 
                                     prediction=prediction)
