import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


st.title("🐷🐷🐷การพยากรณ์โรคหัวใจล้มเหลวด้วยเทคนิคเหมืองข้อมูล🐷🐷")
st.header("🍖🍖การพยากรณ์โรคหัวใจล้มเหลวด้วยเทคนิคเหมืองข้อมูล🍖🍖")

st.image('./img/h1')
st.subheader("โรคหัวใจล้มเหลว")

dt = pd.read_csv("./data/Heart3.csv")
st.image('./img/h5')
st.subheader("โรคหัวใจล้มเหลว")


st.subheader("่สถิติข้อมูลโรคหัวใจ")
st.write('ผลรวม')
