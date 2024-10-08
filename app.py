from flask import Flask, render_template, request, redirect, jsonify, url_for, send_file
import plotly.graph_objs as go
import plotly
import io
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)