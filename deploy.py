import nltk
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import PyPDF2

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    st.header("Text Summarization App")
    # text = st.text_area(label="Input text")

    chapter = st.selectbox('Select Chapter:', ('1', '2', '3', '4', '5'))
    # st.write('You selected:', chapter)

    pdf_file_obj = open('wings-of-fire.pdf', 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)

    if '1' in chapter:
        text = ''
        for i in range(10, 13):
            page = pdf_reader.getPage(i)
            text += page.extractText()

    elif '2':
        text = ''
        for i in range(14, 26):
            page = pdf_reader.getPage(i)
            text += page.extractText()

    elif '3':
        text = ''
        for i in range(27, 32):
            page = pdf_reader.getPage(i)
            text += page.extractText()

    elif '4':
        text = ''
        for i in range(34, 36):
            page = pdf_reader.getPage(i)
            text += page.extractText()

    else:
        text = ''
        for i in range(37, 42):
            page = pdf_reader.getPage(i)
            text += page.extractText()

    def sentiment_scores(sentence):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(sentence)
        emotion = ""
        if sentiment_dict['compound'] >= 0.05:
            emotion = "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
            emotion = "Negative"
        else:
            emotion = "Neutral"
        return emotion

    if st.button("Summarize"):
        if text:
            summarizer_kl = KLSummarizer()
            language = "english"
            sentence_count = 5

            parser = PlaintextParser(text, Tokenizer(language))

            # Summarize using sumy KL Divergence
            summary = summarizer_kl(parser.document, 2)

            kl_summary = ""
            for sentence in summary:
                kl_summary += str(sentence)

            # st.success(kl_summary)

            summ_text = kl_summary
            st.write(summ_text)

    if st.button("Sentiment"):
        if text:
            summarizer_kl = KLSummarizer()
            language = "english"
            sentence_count = 5

            parser = PlaintextParser(text, Tokenizer(language))

            # Summarize using sumy KL Divergence
            summary = summarizer_kl(parser.document, 2)

            kl_summary = ""
            for sentence in summary:
                kl_summary += str(sentence)

                score = sentiment_scores(kl_summary)
            # st.success(score)

            senti_text = score
            st.write(senti_text)
