#import gradio as gr
#import tensorflow as tf
#from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
#
## Load model and tokenizer
#model_save_path = "saved_model"  # replace with the actual path
#model = TFDistilBertForSequenceClassification.from_pretrained(model_save_path)
#tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
#
#def predict(question1, question2):
#    inputs = tokenizer(
#        [question1], [question2],
#        return_tensors='tf',
#        truncation=True,
#        padding=True,
#        max_length=50
#    )
#    outputs = model(inputs)
#    logits = outputs.logits
#    probabilities = tf.nn.softmax(logits, axis=-1)
#    prediction = tf.argmax(probabilities, axis=1).numpy()[0]
#    prob = probabilities.numpy()[0]
#    return f"{'Duplicate' if prediction == 1 else 'Not Duplicate'} (Probability: {prob})"
#
## Gradio interface
#interface = gr.Interface(
#    fn=predict,
#    inputs=["text", "text"],
#    outputs="text",
#    title="Duplicate Question Detection",
#    description="Enter two questions to check if they are duplicates."
#)
#
#interface.launch()

import streamlit as st
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

# Load model and tokenizer
model_save_path = "./saved_model"  # replace with the actual path
model = TFDistilBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)

# Streamlit app
st.title("Duplicate Question Detection (Approach 2)")
st.write("Using DistilBERT")

question1 = st.text_input("Enter the first question:")
question2 = st.text_input("Enter the second question:")

if st.button("Predict"):
    if question1 and question2:
        inputs = tokenizer(
            [question1], [question2],
            return_tensors='tf',
            truncation=True,
            padding=True,
            max_length=50
        )
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)
        prediction = tf.argmax(probabilities, axis=1).numpy()[0]  # 0 or 1
        prob = probabilities.numpy()[0]

        st.success(f"Prediction: {'Duplicate' if prediction == 1 else 'Not Duplicate'}")
        st.success(f"Probability: Not Duplicate {prob[0]}    Duplicate {prob[1]}")
    else:
        st.write("Please enter both questions.")


