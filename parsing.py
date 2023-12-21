import pytesseract
import re
import fitz
import cv2
import numpy as np
import streamlit as st
import spacy
from spacy.matcher import Matcher


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

nlp = spacy.load('en_core_web_sm')

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

def ocr(file1):
        doc = fitz.open(file1, filetype="pdf")  # Open the PDF file
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  # Load each page of the PDF
            zoom = 2.0  # Set the zoom factor for better resolution (adjust as needed)
            mat = fitz.Matrix(zoom, zoom)  # Create a matrix for zooming
            pix = page.get_pixmap(matrix=mat)  # Render the page as a pixmap with zooming
            img = np.frombuffer(pix.samples, dtype=np.uint8)  # Convert pixmap to a NumPy array
            img = img.reshape(pix.height, pix.width, 3)  # Reshape array to the image dimensions (height, width, channels)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (thresh, binary) = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY)
        # Use pytesseract to extract text from the image
            page_text = pytesseract.image_to_string(binary)
            text += page_text

        return text

def extract_details(resume,keywords):

    text=ocr(resume)

    nlp_text = nlp(text)

    l = []
    for i in keywords:
        matches = re.findall(r'(?i)' + i, text)
        l.extend(matches)
        l = list(set(l))
    
    if len(l)>=1:
            
        # First name and Last name are always Proper Nouns
        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        matcher.add('NAME', [pattern])  # Provide the pattern as a list inside a list
        matches = matcher(nlp_text)


        for match_id, start, end in matches:
            span = nlp_text[start:end]
            st.write("Name:", span.text)
            break
        

        email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',text)
        if email:
            st.write("Email:", email.group())

        phone = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
        if phone:
            st.write("Mobile:", phone.group())



        st.write(l)

        st.write("Resume Qualified")

    else:
        st.write("Resume not Qualified")

    
def app():
    st.title("Resume Parser")
    resume=st.file_uploader("Upload a File",type=["pdf"])
    keywords = st.text_input("Enter keywords (separated by comma)")
    if st.button("Search"):
        keywords = [keyword.strip() for keyword in keywords.split(",")]
        if resume is not None:
            extract_details(resume,keywords)
        else:
            st.write("Please upload a resume ")

app()



