from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch
import numpy as np

from sentence_splitter import SentenceSplitter, split_text_into_sentences
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
config = LongformerConfig()
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)
config = LongformerConfig.from_json_file('E:/Research/Plagiarism/Plagiarism_Checker/Longformer_Epoch-1/config.json')
model = LongformerForSequenceClassification.from_pretrained('E:/Research/Plagiarism/Plagiarism_Checker/Longformer_Epoch-1/pytorch_model.bin',config=config)

splitter = SentenceSplitter(language='en')
#Create a list of sentences from a text file
def create_list(text):
    with open(text, 'r',encoding='utf-8') as file:
        data = file.read().replace('\n', '')
        #print(data)
        sentence_list = splitter.split(data)
    return sentence_list

def create_prediction(text):
    sentence_list = create_list(text)
    final_output = []
    for i in range(len(sentence_list)):
        tokenized_sentence = tokenizer.encode(sentence_list[i])
        input_ids = torch.tensor([tokenized_sentence])
        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=1)
        #print(int(label_indices))
        final_output.append(int(label_indices))
    count_1 = final_output.count(1)
    count_0 = final_output.count(0)
    pct_plagiarism = (count_1 / len(final_output)) * 100

    return str(pct_plagiarism) + '%' + ' ' + 'Plagiarised'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demon', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        cont = request.form['context']
        pred= generate_ner(cont)
        return render_template('index.html', data=pred)