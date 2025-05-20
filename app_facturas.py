import streamlit as st
import base64
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
# Usar PydanticOutputParser genérico
from langchain_core.output_parsers import PydanticOutputParser
# Importar BaseModel y Field desde pydantic (asumiendo que Pydantic V2 está instalado en el entorno)
try:
    from pydantic import BaseModel, Field, __version__ as pydantic_version
    if not pydantic_version.startswith("2."):
        st.warning(f"Pydantic version {pydantic_version} detectada. Se recomienda Pydantic V2 para una mejor compatibilidad con PydanticOutputParser. Si tienes problemas, considera usar langchain_core.pydantic_v1 y un parser específico si está disponible en tu versión de langchain-core, o actualiza tus paquetes.")
        from langchain_core.pydantic_v1 import BaseModel, Field # Fallback a v1 si pydantic global no es v2
except ImportError:
    st.error("Pydantic no está instalado. Por favor, instálalo: pip install pydantic")
    from langchain_core.pydantic_v1 import BaseModel, Field # Fallback

from langchain_openai import ChatOpenAI

# --- Configuración Inicial y Modelos ---
load_dotenv()

# 1. Definir el esquema de salida con Pydantic
#    (Ahora usando pydantic.BaseModel, que idealmente es V2)
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

# 2. Función para codificar la imagen a base64 (sin cambios)
def encode_image_to_base64(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        format = img.format.lower() if img.format else 'jpeg'
    except Exception:
        format = 'jpeg'
    return f"data:image/{format};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

# 3. Función de extracción principal (modificada para Streamlit)
def extraer_datos_factura_streamlit(image_bytes, openai_api_key):
    if not openai_api_key:
        st.error("Por favor, introduce tu API Key de OpenAI para continuar.")
        return None

    try:
        base64_image = encode_image_to_base64(image_bytes)
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

    try:
        model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2048, openai_api_key=openai_api_key)
        # Usar PydanticOutputParser genérico
        parser = PydanticOutputParser(pydantic_object=FacturaData)
    except Exception as e:
        st.error(f"Error al inicializar el modelo de OpenAI: {e}. Asegúrate de que la API key es válida y tienes acceso al modelo gpt-4o.")
        return None

    prompt_text = f"""
    Analiza la siguiente imagen de una factura y extrae la información solicitada.
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
        with st.spinner("Analizando la factura con IA... Esto puede tardar unos segundos."):
            response = model.invoke([message])
        # Para Pydantic V2, .dict() se llama .model_dump()
        # Para Pydantic V1, .dict() sigue siendo .dict()
        # El PydanticOutputParser debería devolver un objeto del mismo tipo que pydantic_object
        parsed_object = parser.parse(response.content)
        if hasattr(parsed_object, "model_dump"): # Pydantic V2
            return parsed_object.model_dump()
        else: # Pydantic V1
            return parsed_object.dict()
    except Exception as e:
        st.error(f"Error durante la extracción o el parseo: {e}")
        if 'response' in locals() and hasattr(response, 'content'):
            st.text_area("Respuesta cruda del modelo (para depuración):", response.content, height=150)
        return None

# --- Interfaz de Streamlit (sin cambios significativos aquí, solo el de use_container_width) ---
st.set_page_config(page_title="Extractor de Facturas IA", layout="wide")
st.title("🧾 Extractor de Datos de Facturas con IA")
st.markdown("Sube una imagen de una factura y la IA intentará extraer la información relevante en formato JSON.")

st.sidebar.header("Configuración")
openai_api_key_env = os.getenv("OPENAI_API_KEY")
openai_api_key_input = st.sidebar.text_input(
    "Tu API Key de OpenAI", type="password",
    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    help="Necesaria para usar los modelos de OpenAI. Si está en tu archivo .env, se usará automáticamente.",
    value=openai_api_key_env if openai_api_key_env else ""
)
OPENAI_API_KEY_TO_USE = openai_api_key_input if openai_api_key_input else openai_api_key_env

if not OPENAI_API_KEY_TO_USE:
    st.warning("Por favor, introduce tu API Key de OpenAI en la barra lateral para usar la aplicación.")
    st.stop()

uploaded_file = st.file_uploader("Elige una imagen de factura...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Factura Subida", use_container_width=True) # Corregido aquí
    st.markdown("---")
    if st.button("✨ Extraer Datos de la Factura"):
        if not OPENAI_API_KEY_TO_USE:
            st.error("API Key de OpenAI no configurada. Por favor, ingrésala en la barra lateral.")
        else:
            extracted_data = extraer_datos_factura_streamlit(image_bytes, OPENAI_API_KEY_TO_USE)
            if extracted_data:
                st.success("¡Datos extraídos con éxito!")
                st.json(extracted_data)
                st.markdown("---")
                st.subheader("Verificar y Editar Datos (Opcional)")
                col1, col2 = st.columns(2)
                with col1:
                    extracted_data["fecha"] = st.text_input("Fecha (DD/MM/YYYY)", extracted_data.get("fecha", ""))
                    extracted_data["categoria"] = st.text_input("Categoría", extracted_data.get("categoria", ""))
                    extracted_data["remitente"] = st.text_input("Remitente", extracted_data.get("remitente", ""))
                    extracted_data["descripcion"] = st.text_area("Descripción", extracted_data.get("descripcion", ""))
                with col2:
                    extracted_data["importe_sin_iva"] = st.number_input("Importe sin IVA", value=float(extracted_data.get("importe_sin_iva", 0.0)), format="%.2f")
                    extracted_data["iva_porcentaje"] = st.number_input("IVA Porcentaje (%)", value=int(extracted_data.get("iva_porcentaje", 0)), min_value=0, max_value=100)
                    extracted_data["total"] = st.number_input("Total", value=float(extracted_data.get("total", 0.0)), format="%.2f")
                    extracted_data["divisa"] = st.text_input("Divisa", extracted_data.get("divisa", "EUR"))
                    extracted_data["nota"] = st.text_input("Nota", extracted_data.get("nota", "N/A"))
                st.markdown("#### JSON Actualizado:")
                st.json(extracted_data)
            else:
                st.error("No se pudieron extraer los datos. Revisa los mensajes de error si los hay.")
else:
    st.info("Esperando que subas una imagen de factura.")

st.markdown("---")
st.markdown("Desarrollado con LangChain, OpenAI y Streamlit.")