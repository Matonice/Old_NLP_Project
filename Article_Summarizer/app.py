# importing the necessary library
import re
import math

import spacy
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

import gradio as gr

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import BartTokenizer, BartForConditionalGeneration


# initailizing the model pipeline   
model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
nlp = spacy.load("en_core_web_sm")
    
    
def clean_text(text):
  text = text
  text = text.encode("ascii", errors="ignore").decode(
          "ascii"
    )  # remove non-ascii, Chinese characters
    
  text = re.sub(r"\n", " ", text)
  text = re.sub(r"\n\n", " ", text)
  text = re.sub(r"\t", " ", text)
  text = text.strip(" ")
  text = re.sub(
        " +", " ", text
    ).strip()  # get rid of multiple spaces and replace with a single
  return text     


# Defining a function to get the summary of the article
def final_summary(text):
    # reading in the text and tokenizing it into sentence
    text = text
    bullet_points = 10

    while (bullet_points >= 10):

      chunks = []
      sentences = nlp(text)
      for sentence in sentences.sents:
        chunks.append(str(sentence))

      output = []
      sentences_remaining = len(chunks)
      i = 0

      #looping through the sentences in an equal batch based on their length and summarizing them
      while sentences_remaining > 0:
          chunks_remaining = math.ceil(sentences_remaining / 10.0)
          next_chunk_size = math.ceil(sentences_remaining / chunks_remaining)
          sentence = "".join(chunks[i:i+next_chunk_size])

          i += next_chunk_size
          sentences_remaining -= next_chunk_size

          inputs = tokenizer(sentence, return_tensors="pt", padding="longest")
          #inputs = inputs.to(DEVICE)
          original_input_length = len(inputs["input_ids"][0])

          # checking if the length of the input batch is less than 150
          if original_input_length < 100:
              split_sentences = nlp(sentence)
              for split_sentence in split_sentences.sents:
                output.append(str(split_sentence).rstrip("."))
              

          # checking if the length of the input batch is greater than 1024
          elif original_input_length > 1024:
              sent = sent_tokenize(sentence)
              length_sent = len(sent)

              j = 0
              sent_remaining = math.ceil(length_sent / 2)

              # going through the batch that is greater than 1024 and dividing them
              while length_sent > 0:
                  halved_sentence = "".join(sent[j:j+sent_remaining])
                  halved_inputs = tokenizer(halved_sentence, return_tensors="pt")
                  #halved_inputs = halved_inputs.to(DEVICE)
                  halved_summary_ids = model.generate(halved_inputs["input_ids"])
                  j += sent_remaining
                  length_sent -= sent_remaining

                  # checking if the length of the output summary is less than the original text
                  if len(halved_summary_ids[0]) < len(halved_inputs["input_ids"][0]):
                      halved_summary = tokenizer.batch_decode(halved_summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                      output.append(halved_summary)

          else:
              summary_ids = model.generate(inputs["input_ids"])

              if len(summary_ids[0]) < original_input_length:
                  summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                  output.append(summary)

      final_output = []
      for paragraphs in output:
        lines = paragraphs.split(" . ")	 
        for line in lines:
          final_output.append(line.replace(" .", "").strip()) 
      text = ".".join(final_output)
      bullet_points = len(final_output)


    for i in range(len(final_output)):
        final_output[i] = "* " + final_output[i] + "."

      # final sentences are incoherent, so we will join them by bullet separator
    summary_bullet = "\n".join(final_output)

    return summary_bullet
  
# creating an interface for the headline generator using gradio
demo = gr.Interface(final_summary, inputs=[gr.Textbox(label="Drop your article here")],
                                          title = "ARTICLE SUMMARIZER",
                                          outputs=[gr.Textbox(label="Summary")],
                                          )             

# launching the app
if __name__ == "__main__":
    demo.launch(debug=True)