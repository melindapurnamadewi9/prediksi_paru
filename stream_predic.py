import pickle 
import numpy as np
import streamlit as st

model = pickle.load(open('predic_tabel1.sav', 'rb'))

st.title('Prediksi Penyakit Paru-Paru')

col1, col2 = st.columns(2)

with col1 :
	usia = st.number_input('input usia')

	jenis_kelamin= st.number_input('input jenis kelamin (male : 1 female : 2)')

	merokok = st.number_input('input merokok')

	bekerja = st.number_input('input bekerja')

with col2:
    rumah_tangga = st.number_input('input rumah_tangga')
    aktivitas_begadang= st.number_input('input aktivitas_begadang')
    aktivitas_olahraga = st.number_input('input aktivitas_olahraga')
    asuransi = st.number_input('input asuransi')
    penyakit_bawaan = st.number_input('input penyakit_bawaan')
prediksi = ''

if st.button('prediksi penyakit paru paru'):
	penyakit_prediksi = model.predict([[usia, jenis_kelamin, merokok, bekerja, rumah_tangga, aktivitas_begadang, aktivitas_olahraga, asuransi, penyakit_bawaan]])


# print(penyakit_prediksi)

	if(penyakit_prediksi[0]==1):
            prediksi = 'positif penyakit paru paru'
	else :
		prediksi = 'negatif  penyakit paru paru'
st.success(prediksi)