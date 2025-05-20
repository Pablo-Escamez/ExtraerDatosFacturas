import streamlit as st
import base64
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import json
import pandas as pd

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
try:
    from pydantic import BaseModel, Field, __version__ as pydantic_version
    # No mostrar advertencia de Pydantic aquí, ya que no es el foco principal del usuario
    # if not pydantic_version.startswith("2."):
    #     st.warning(f"Pydantic version {pydantic_version} detectada...")
    if not pydantic_version.startswith("2."): # Si no es V2
         from langchain_core.pydantic_v1 import BaseModel, Field # Usar v1 explícitamente
except ImportError:
    st.error("Pydantic no está instalado. Por favor, instálalo: pip install pydantic")
    from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import ChatOpenAI

# --- Configuración Inicial y Modelos (sin cambios) ---
load_dotenv()

class FacturaData(BaseModel):
    fecha: str = Field(description="Fecha de la factura en formato DD/MM/YYYY")
    categoria: str = Field(description="Categoría del gasto (ej: Material de Oficina, Comida, Transporte, Software, etc.)")
    remitente: str = Field(description="Nombre o empresa que emite la factura")
    descripcion: str = Field(description="Descripción concisa de los bienes o servicios facturados")
    importe_sin_iva: float = Field(description="Importe total antes de impuestos/IVA")
    iva_porcentaje: int = Field(description="Porcentaje de IVA aplicado (ej: 21 para 21%)")
    total: float = Field(description="Importe total final, incluyendo IVA")
    divisa: str = Field(description="Divisa de la factura (ej: EUR, USD). Por defecto EUR si no se especifica.", default="EUR")
    nota: str = Field(description="Cualquier nota adicional relevante. Por defecto 'N/A'.", default="N/A")

def encode_image_to_base64(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        format = img.format.lower() if img.format else 'jpeg'
    except Exception:
        format = 'jpeg'
    return f"data:image/{format};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def extraer_datos_factura_streamlit(image_bytes, openai_api_key, filename="factura"):
    try:
        base64_image = encode_image_to_base64(image_bytes)
    except Exception as e:
        # st.error(f"Error al procesar la imagen '{filename}': {e}") # Evitar st.error dentro de la función para mejor control del UI
        return {"filename": filename, "status": "error", "error_message": f"Error al procesar la imagen: {e}", "data": None}

    try:
        model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2048, openai_api_key=openai_api_key)
        parser = PydanticOutputParser(pydantic_object=FacturaData)
    except Exception as e:
        # st.error(f"Error al inicializar el modelo de OpenAI para '{filename}': {e}. Asegúrate de que la API key es válida y tienes acceso al modelo gpt-4o.")
        return {"filename": filename, "status": "error", "error_message": f"Error al inicializar OpenAI: {e}", "data": None}

    prompt_text = f"""
    Analiza la siguiente imagen de una factura ({filename}) y extrae la información solicitada.
    Debes ser muy preciso con los números, especialmente los importes y el porcentaje de IVA.
    La categoría debe ser una descripción general del tipo de gasto.
    Si alguna información no está presente, intenta deducirla o usa los valores por defecto.
    La fecha debe estar en formato DD/MM/YYYY.
    {parser.get_format_instructions()}
    Imagen de la factura:
    """
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": base64_image, "detail": "high"}},
        ]
    )
    try:
        response = model.invoke([message])
        parsed_object = parser.parse(response.content)
        data_dict = parsed_object.model_dump() if hasattr(parsed_object, "model_dump") else parsed_object.dict()
        return {"filename": filename, "status": "success", "data": data_dict, "image_bytes": image_bytes} # Guardamos bytes para posible visualización
    except Exception as e:
        error_message = f"Error durante la extracción o el parseo para '{filename}': {e}"
        raw_response_content = response.content if 'response' in locals() and hasattr(response, 'content') else "N/A"
        return {"filename": filename, "status": "error", "error_message": error_message, "data": None, "raw_response": raw_response_content, "image_bytes": image_bytes}

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Extractor de Facturas IA Múltiple", layout="wide")
st.title("🧾 Extractor de Datos de Múltiples Facturas con IA")
st.markdown("Sube una o varias imágenes de facturas y la IA intentará extraer la información relevante. Los resultados se acumularán.")

# Inicializar el estado de la sesión para guardar los resultados si no existe
if 'processed_invoices' not in st.session_state:
    st.session_state.processed_invoices = []

# Columna para la API Key
st.sidebar.header("Configuración")
openai_api_key_env = os.getenv("OPENAI_API_KEY")
openai_api_key_input = st.sidebar.text_input(
    "Tu API Key de OpenAI", type="password",
    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    help="Necesaria para usar los modelos de OpenAI.",
    value=openai_api_key_env if openai_api_key_env else "",
    key="api_key_input_widget"
)
OPENAI_API_KEY_TO_USE = openai_api_key_input if openai_api_key_input else openai_api_key_env

if OPENAI_API_KEY_TO_USE:
    st.sidebar.success("API Key detectada. Lista para procesar.")
else:
    st.sidebar.warning("Por favor, introduce tu API Key de OpenAI para habilitar el procesamiento.")

# Carga de múltiples archivos
uploaded_files = st.file_uploader(
    "Elige imágenes de facturas...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="file_uploader_widget"
)

# Lógica para habilitar el botón y mensajes
button_disabled = not (OPENAI_API_KEY_TO_USE and uploaded_files)
button_tooltip = ""
if not OPENAI_API_KEY_TO_USE and not uploaded_files:
    button_tooltip = "Introduce tu API Key y sube facturas para empezar."
elif not OPENAI_API_KEY_TO_USE:
    button_tooltip = "Introduce tu API Key en la barra lateral para procesar."
elif not uploaded_files:
    button_tooltip = "Sube una o más facturas para procesar."


if uploaded_files:
    st.info(f"{len(uploaded_files)} factura(s) nueva(s) seleccionada(s).")

# Botón para procesar facturas
if st.button("✨ Procesar Facturas Seleccionadas", disabled=button_disabled, help=button_tooltip if button_disabled else None, key="process_button_widget"):
    
    # NO limpiar st.session_state.processed_invoices aquí
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    # Usaremos un contenedor para los mensajes de procesamiento de este lote
    processing_messages_container = st.container()
    messages_this_batch = []

    for i, uploaded_file in enumerate(uploaded_files):
        messages_this_batch.append(f"--- \n### Procesando: {uploaded_file.name}")
        with processing_messages_container: # Escribir mensajes dentro del contenedor
            st.markdown("\n".join(messages_this_batch))
        
        with st.spinner(f"Analizando {uploaded_file.name}..."):
            image_bytes_current = uploaded_file.getvalue() # Obtener bytes aquí
            extracted_info = extraer_datos_factura_streamlit(image_bytes_current, OPENAI_API_KEY_TO_USE, uploaded_file.name)
            
            if extracted_info:
                # Añadir a la lista global de la sesión
                st.session_state.processed_invoices.append(extracted_info)
                if extracted_info["status"] == "success":
                    messages_this_batch.append(f"✅ **{uploaded_file.name}**: Datos extraídos con éxito.")
                else:
                    error_msg = extracted_info.get('error_message', 'Error desconocido')
                    messages_this_batch.append(f"❌ **{uploaded_file.name}**: Error - _{error_msg}_")
        
        with processing_messages_container: # Actualizar mensajes después del spinner
            st.markdown("\n".join(messages_this_batch))
        progress_bar.progress((i + 1) / total_files)
    
    processing_messages_container.success("¡Procesamiento de este lote de facturas completado!")
    st.balloons()
    # Forzar un rerun para que la sección de "Registro Acumulado" se actualice inmediatamente con los nuevos datos.
    # Esto es importante porque st.button causa un rerun, pero los cambios en session_state podrían no reflejarse
    # en la parte inferior de la página sin un rerun explícito después de que el bucle termine.
    # Sin embargo, añadir .append() a session_state dentro de un bucle y luego leerlo *debería* funcionar
    # sin un rerun explícito si toda la lógica de renderizado está después del botón.
    # Vamos a probar sin el rerun explícito primero, ya que puede tener efectos secundarios.
    # st.experimental_rerun() # Descomentar si la actualización del registro no es inmediata

# Mensajes contextuales si el botón está deshabilitado
if not uploaded_files and not st.session_state.processed_invoices:
    if not OPENAI_API_KEY_TO_USE:
        st.warning("Por favor, introduce tu API Key en la barra lateral y sube algunas facturas para empezar.")
    else:
        st.info("Sube una o más facturas para comenzar.")

# Mostrar registro acumulado y opciones de descarga
if st.session_state.processed_invoices:
    st.markdown("--- \n## 📋 Registro Acumulado de Facturas Procesadas")
    
    if st.button("🗑️ Limpiar Registro de Facturas", key="clear_log_button"):
        st.session_state.processed_invoices = []
        st.success("El registro de facturas ha sido limpiado.")
        st.experimental_rerun() # Forzar rerun para actualizar la UI inmediatamente

    successful_extractions = [item["data"] for item in st.session_state.processed_invoices if item["status"] == "success" and item["data"]]
    
    if not successful_extractions:
        st.info("Aún no se han procesado facturas con éxito o el registro está vacío.")
    else:
        df = pd.DataFrame(successful_extractions)
        json_string = json.dumps(successful_extractions, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Descargar Registro como JSON", data=json_string,
            file_name="registro_facturas_acumulado.json", mime="application/json",
        )
        csv_string = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Registro como CSV", data=csv_string,
            file_name="registro_facturas_acumulado.csv", mime="text/csv",
        )
        st.markdown("### Vista Previa de Datos Exitosos (Acumulado):")
        st.dataframe(df)

    st.markdown("### Detalles por Factura (Acumulado):")
    # Iterar en orden inverso para mostrar los más recientes primero
    for i, item in enumerate(reversed(st.session_state.processed_invoices)):
        unique_key_prefix = f"detail_{len(st.session_state.processed_invoices) - 1 - i}_{item['filename']}"
        with st.expander(f"Factura: {item['filename']} - Estado: {item['status'].upper()}", expanded=False):
            if item.get("image_bytes"): # Si guardamos los bytes de la imagen
                try:
                    st.image(item["image_bytes"], caption=f"Imagen: {item['filename']}", use_container_width=True)
                except Exception as img_e:
                    st.warning(f"No se pudo mostrar la imagen para {item['filename']}: {img_e}")

            if item["status"] == "success":
                st.json(item["data"])
            else:
                st.error(f"Error: {item.get('error_message', 'Desconocido')}")
                if "raw_response" in item and item["raw_response"] and item["raw_response"] != "N/A":
                     st.text_area("Respuesta cruda del modelo:", item["raw_response"], height=100, key=f"raw_resp_{unique_key_prefix}")
else: # Si no hay nada en st.session_state.processed_invoices
     if uploaded_files and OPENAI_API_KEY_TO_USE: # Y hay archivos listos para procesar
         pass # El botón de procesar estará activo
     elif not OPENAI_API_KEY_TO_USE and not uploaded_files:
         st.info("Introduce tu API Key y sube facturas para comenzar.")
     elif not OPENAI_API_KEY_TO_USE:
         st.info("Introduce tu API Key para poder procesar facturas.")
     elif not uploaded_files:
         st.info("Sube facturas para procesar.")


st.markdown("---")
st.markdown("Desarrollado con LangChain, OpenAI y Streamlit.")