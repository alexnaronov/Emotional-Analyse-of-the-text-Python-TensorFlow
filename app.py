from models.test import is_text_positive
from flask import Flask, render_template, flash, redirect, url_for, request
from wtforms import Form, StringField, TextAreaField, SelectField
from pathlib import Path
from tensorflow.keras import models
from tensorflow.keras import utils

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form=TextAreaForm(request.form)
    link=""

    if request.method == 'POST':
        Text=form.Text.data
        Type=form.Type.data

        if Text=="":
            flash("Please enter text into the Text form")
            return render_template('index.html', form=form, link=link)
            
        saved_models_path = Path("models/saved_models")

        plots_path = Path("static/plots")
        plots_path.mkdir(parents=True, exist_ok=True)

        if Type=='Perceptron':

            fast_text = models.load_model(saved_models_path / "fast_text.h5")

            result = is_text_positive(fast_text, Text)

            utils.plot_model(fast_text, plots_path / "fast_text.png")
            link=plots_path / "fast_text.png"

            flash(GetResult(result))
        if Type=='CNN':

            text_cnn = models.load_model(saved_models_path / "text_cnn.h5")

            result = is_text_positive(text_cnn, Text)

            utils.plot_model(text_cnn, plots_path / "text_cnn.png")
            link=plots_path / "text_cnn.png"

            flash(GetResult(result))
        if Type=='RNN':

            text_rnn = models.load_model(saved_models_path / "text_rnn.h5")

            result = is_text_positive(text_rnn, Text)

            utils.plot_model(text_rnn, plots_path / "text_rnn.png")
            link=plots_path / "text_rnn.png"

            flash(GetResult(result))
        #return render_template('index.html')
    return render_template('index.html', form=form, link=link)

def GetResult(value):
    message = f'Positive: {value*100}% \n Negative: {((1-value)*100)}% \n'
    return message

class TextAreaForm(Form):
    Type=SelectField("Type of neural model: ", choices=[
    	("Perceptron", "Perceptron"),
    	("CNN", "CNN"),
    	("RNN", "RNN")])
    Text=TextAreaField('Text')

if __name__ == '__main__':
    app.secret_key = 'some secret key'
    app.debug = True
    app.run(debug=True);
