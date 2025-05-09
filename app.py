import streamlit as st
from transformers import pipeline
import torch


# تحديد الجهاز: 0 لو في CUDA، -1 يعني CPU
device = 0 if torch.cuda.is_available() else -1

# إنشاء pipeline للنصوص
text_generator = pipeline("text-generation", model="openai-community/gpt2", device=device)

# باقي الموديلات زي ما هي

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
translator = pipeline("translation_en_to_fr", model="facebook/nllb-200-distilled-600M", device=-1)
question_answering = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)


# ✨ تطبيق Streamlit
st.title(" NLP App using 4 Hugging Face Models ")
st.write("Start with a short topic and let's work in three steps: Generate text about it ➔ Summarize ➔ Translate ➔  question_answering")

user_input = st.text_area(" Write a topic: ")
if "generated_text" not in st.session_state:
    st.session_state['generated_text'] = ""

if st.button("Start processing..."):
    if user_input.strip() != "":
        with st.spinner("Wait ...!⏳"):
            generated_output = text_generator(user_input, max_length=100, num_return_sequences=1,truncation=True,pad_token_id=50256)
            generated_text = generated_output[0]['generated_text']
            st.subheader("🔹generated_text: ")
            st.write(generated_text)
            st.session_state['generated_text'] = generated_text

            summarized_output = summarizer(generated_text, max_length=50, min_length=25, do_sample=False)
            summarized_text = summarized_output[0]['summary_text']
            st.subheader("🔹 summary_text: ")
            st.write(summarized_text)

            translated_output = translator(summarized_text)
            translated_text = translated_output[0]['translation_text']
            st.subheader("🔹 translation_text to fr: ")
            st.write(translated_text)
    else:
        st.warning("⚠️ try text first")

if st.session_state['generated_text']:
    question = st.text_area("Write a question:")
    if st.button("Get Answer"):
        if question.strip() != "":
            answer = question_answering(question=question, context=st.session_state['generated_text'])
            st.subheader("🔹 Question Answering:")
            st.write(answer['answer'])
        else:
            st.warning("⚠️ Please enter a question.")
