import re
from transformers import pipeline

# Data Input
text = """Tengo un adosado con varias plantas de nueva construcción. Entre una planta y otra me costó configurarlos, uno en la planta baja, otro en la primera planta y el último en la buhardilla. Tras conseguirlo después de probar en varias ubicaciones hasta que la app me dijo que estaban bien configurados, hago un test de velocidad y me encuentro que no pasa de los 50mbps (adjunto foto). Ni siquiera probando en la misma habitación donde estaba el principal conectado al router de salida directo con el cable que trae.

Me da que el forjado de la casa hace que no sea muy buena la cobertura, pero que esté al lado del que tiene conexión directa y no supere los 50mbps es absurdo.

En el sótano ha sido desastroso, dónde antes llegaba de forma estable la señal del router del operador, ahora mi móvil salta todo el tiempo entre los 2,4ghz y los 5ghz, provocando cortes continuos.

Cunado intento configurar desde la aplicación que desactive los 5ghz me encuentro que es imposible. Entonces es cuando me doy cuenta que la personalización de la configuración de estos dispositivos es muy limitada, por lo que no es nada recomendable para usuarios medios/avanzados que quieran hacer algo más con la conectividad WiFi de casa... Es más, leyendo los comentarios he visto que un usuario llamó al soporte para desactivar los 5ghz le dijeron que sólo lo podían hacer ellos en remoto, por lo que tienen acceso a tu red... Puffff....

Lo he devuelto bastante disgustado y decepcionado. Esperaba más calidad para el precio que tiene. Probaré con otra marca."""

# Data Preprocess
def preprocess_text(text):

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    return text

cleaned_text = preprocess_text(text)

# Model Deployment
classifier = pipeline("sentiment-analysis", model="maxpe/bertin-roberta-base-spanish_sem_eval_2018_task_1")
output = classifier(cleaned_text)

# Display result
print(output)