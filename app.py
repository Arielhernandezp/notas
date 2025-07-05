import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y los encoders
try:
    modelo_data = joblib.load('modelo_naive_bayes.jb')
    model = modelo_data['model']
    label_encoders = modelo_data['encoders']
    st.success("Modelo y encoders cargados correctamente.")
except FileNotFoundError:
    st.error("Error: No se encontró el archivo 'modelo_naive_bayes.jb'. Por favor, asegúrese de que el modelo y los encoders han sido guardados previamente.")
    model = None
    label_encoders = None
except Exception as e:
    st.error(f"Error al cargar el modelo y los encoders: {e}")
    model = None
    label_encoders = None


st.title("Predicción de Clase")
st.markdown('<h2 style="color: red;">Elaborado por: ariel hernandez perez</h2>', unsafe_allow_html=True)

if model is not None and label_encoders is not None:
    st.write("Seleccione los valores para realizar la predicción:")

    # Entradas del usuario
    horas_estudio = st.selectbox("Horas de Estudio:", ["Alta", "Baja"])
    asistencia = st.selectbox("Asistencia:", ["Buena", "Mala"])

    # Crear un DataFrame con la entrada del usuario
    nueva_observacion = pd.DataFrame({
        "Horas de Estudio": [horas_estudio],
        "Asistencia": [asistencia]
    })

    # Codificar la nueva observación
    try:
        nueva_observacion_codificada = nueva_observacion.copy()
        for column in nueva_observacion_codificada.columns:
            if column in label_encoders:
                nueva_observacion_codificada[column] = label_encoders[column].transform(nueva_observacion_codificada[column])
            else:
                st.warning(f"Advertencia: No se encontró el encoder para la columna '{column}'. La predicción puede no ser correcta.")
                # Si no hay encoder, intenta dejar el valor tal cual si es numérico, o muestra un error.
                # Aquí asumimos que si falta el encoder es un problema.
                st.stop() # Detiene la ejecución si falta un encoder crucial

        # Realizar la predicción
        prediccion_numerica = model.predict(nueva_observacion_codificada)

        # Decodificar la predicción
        if "Resultado" in label_encoders:
            prediccion_etiqueta = label_encoders["Resultado"].inverse_transform(prediccion_numerica)

            # Mostrar el resultado con caritas
            st.subheader("Resultado de la predicción:")
            if prediccion_etiqueta[0] == "Sí":
                st.success(f"Felicitaciones {prediccion_etiqueta[0]} aprueba! 😊")
            else:
                st.error(f"No aprueba 😔")
        else:
            st.error("Error: No se encontró el encoder para la columna 'Resultado'. No se puede decodificar la predicción.")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
else:
    st.warning("El modelo no se cargó correctamente. No se puede realizar la predicción.")

