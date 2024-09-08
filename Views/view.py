from flask import render_template, request, jsonify
from API.app import app
import os
from API.Trancription.transcription_file import *

UPLOAD_FOLDER = './/API//Trancription//upload'

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html', content="")

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        arquivo = file_path
        print(arquivo)
        file_arquivo, e, f = transcribe_file(arquivo)
        print(e)
        print(f)
        return render_template('index.html', content=e+','+f)

