from pyannote.audio import Pipeline
import uuid
import whisper
import torchaudio
import pandas as pd
import csv
import os
from collections import Counter
import re
import json

# Carrega o modelo Whisper
model = whisper.load_model("small")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIARIZATION_FOLDER = os.path.join(BASE_DIR, 'temp_diarization')

def create_unique_file():
    return uuid.uuid4()

def estatisticas_totais_transcricao(file):
    df = pd.read_csv(file)
    total_palavras = int(df['qtd_palavras'].sum())
    total_duracao_segundos = df['temp_duracao'].sum()
    total_minutos = total_duracao_segundos / 60
    palavras_por_minuto = total_palavras / total_minutos if total_minutos > 0 else 0

    texto_completo = ' '.join(df['fala_text'])
    texto_completo = re.sub(r'[^\w\s]', '', texto_completo).lower()
    contador_palavras = Counter(texto_completo.split())
    palavras_ordenadas = dict(contador_palavras.most_common())
    palavras_magicas = ["obrigado(a)", "por favor", "desculpa", "desculpe", "por gentileza", "bom dia",
                        "boa tarde", "boa noite", "agradeço", "agradece", "gratidão", "sinto muito",
                        "perdão", "me perdoe", "com licença"]
    texto_completo_lower = texto_completo.lower()
    total_palavras_magicas = sum(texto_completo_lower.count(palavra) for palavra in palavras_magicas)
    percentual_palavras_magicas = (total_palavras_magicas / sum(
        contador_palavras.values())) * 100 if total_palavras > 0 else 0

    estatisticas = {
        "qtd_total_palavras": total_palavras,
        "palavras_por_minuto": round(palavras_por_minuto, 2),
        "palavras_mais_falados": palavras_ordenadas,
        "percentual_palavras_magicas": f"{percentual_palavras_magicas:.2f}%"
    }

    # Imprimir o objeto JSON
    return json.dumps({"estatisticas": estatisticas}, indent=4, ensure_ascii=False)

def init_diarization(file):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")
    diarization = pipeline(file, min_speakers=1, max_speakers=5)

    name_unique = create_unique_file()


    rttm_name = f"{TEMP_DIARIZATION_FOLDER}\\{name_unique}_diarization.rttm"
    print(rttm_name)
    with open(rttm_name, "w") as rttm:
        diarization.write_rttm(rttm)
    return name_unique

def whisper_transcription_from_audio(audio_file_path, start_time, end_time, temp_file):
    # Carrega o áudio e corta de acordo com os tempos fornecidos
    waveform, sample_rate = torchaudio.load(audio_file_path)
    # Calcula os frames correspondentes aos tempos de início e fim
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)
    waveform_segment = waveform[:, start_frame:end_frame]
    # Salva o segmento de áudio temporariamente
    torchaudio.save(f"{temp_file}.wav", waveform_segment, sample_rate)

    # Transcreve o segmento
    result = model.transcribe(f"{temp_file}.wav", language='pt')
    # Retorna o texto transcrito
    return result['text']


def tratamento_files(file):
    # Leitura do arquivo
    df = pd.read_csv(f'{TEMP_DIARIZATION_FOLDER}\\{file}_diarization.rttm', sep=' ', header=None)
    df.columns = ['SPEAKER', 'ID1', 'ID2', 'col4', 'col5', 'col6', 'col7', 'speaker_name', 'col9', 'col10']
    result = []

    # Variáveis temporárias para agrupar linhas consecutivas
    prev_speaker = None
    start_time = None
    duration_sum = 0

    # Loop sobre cada linha do DataFrame
    for i, row in df.iterrows():
        current_speaker = row['speaker_name']

        if current_speaker == prev_speaker:
            # Se o speaker é o mesmo que o anterior, somamos a duração
            duration_sum += row['col5']
        else:
            # Se o speaker mudou (ou é a primeira linha), salvamos o grupo anterior
            if prev_speaker is not None:
                result.append([prev_speaker, start_time, duration_sum])

            # Resetamos os valores para o novo speaker
            prev_speaker = current_speaker
            start_time = row['col4']
            duration_sum = row['col5']

    # Adiciona o último grupo
    result.append([prev_speaker, start_time, duration_sum])

    # Criando o DataFrame final com os resultados
    final_df = pd.DataFrame(result)
    final_df.to_csv(f'{TEMP_DIARIZATION_FOLDER}\\{file}_tratado.csv', index=False)


def transcribe_file(file):
    file_complete = init_diarization(file)
    tratamento_files(file_complete)


    with open(f"{TEMP_DIARIZATION_FOLDER}\\{file_complete}_tratado.csv", "r") as rttm_file:
        lines = rttm_file.readlines()
    name = None
    for line in lines:
        vetor = line.strip().split(',')
        if str(vetor[0])!="0":
            trans = whisper_transcription_from_audio(file, float(vetor[1]),float(vetor[1])+float(vetor[2]), file_complete )
            name = transcribe_csv_content(file_complete, [vetor[0], trans, vetor[2]])

    return file_complete, json_return(name), estatisticas_totais_transcricao(name)


def transcribe_csv_content(name_file, text):
    file_path = f"{TEMP_DIARIZATION_FOLDER}\\{name_file}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline='', encoding='utf-8') as arquivo:
        writer = csv.writer(arquivo)

        # Adiciona o cabeçalho apenas se o arquivo não existir
        if not file_exists:
            writer.writerow(["locutor", "fala_text", "temp_duracao", "qtd_palavras"])

        writer.writerow([text[0], text[1].strip(), text[2], len(text[1].split())])
    return file_path

def csv_to_json(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        data = list(csv_reader)
    return data

def json_return(csv_file):
    data = csv_to_json(csv_file)
    return json.dumps({"transcricao": data}, indent=4, ensure_ascii=False)

