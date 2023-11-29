# app.py

from flask import Flask, render_template, request
from detect_emotion import detect_emotion

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None


    if request.method == 'POST':
        text = request.form['text']
        result = detect_emotion(text)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
