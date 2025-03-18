import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


Heart=pd.read_csv('./data/Heart.csv')
X =Heart.drop(columns=['HeartDisease'])
y =Heart.HeartDisease

model = GaussianNB()
model.fit(x, y)

# ตั้งค่าหน้าเว็บ Streamlit
st.title("Naïve Bayes Classifier - Iris Dataset")
st.write("ป้อนคุณสมบัติของดอกไม้เพื่อทำนายประเภท")

# รับค่าจากผู้ใช้ผ่าน slider
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")
A1 = st.number_input("กรุณาเลือกข้อมูลอายุ")
A2 = st.number_input("กรุณาเลือกเพศชาย=1 หญิง=0")
A3 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก")
A4 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก1")
A5 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก2")
A6 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก3")
A7 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก4")
A8 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก5")
A9 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก6")
A10 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก7")
A11 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก8")
# สร้างอาร์เรย์ข้อมูลจากค่าที่ป้อน
input_data = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])
# ทำนายผลลัพธ์
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# แสดงผลลัพธ์
st.subheader("ผลลัพธ์ที่ได้:")
st.write(f"ชนิดของดอกไม้ที่คาดการณ์: *{prediction[0]}*")

# แสดงความน่าจะเป็นของแต่ละคลาส
st.subheader("ความน่าจะเป็นของแต่ละประเภท:")
df_proba = pd.DataFrame(prediction_proba, columns=Heart.HeartDisease)
st.dataframe(df_proba.style.format("{:.2%}"))