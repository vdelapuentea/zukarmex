from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
def load_data_sniim():
    client = bigquery.Client()
    data_sniim_estado = client.query(f""" SELECT date_time AS Fecha, Estado, AVG(valor) AS Valor_Promedio_SNIIM FROM (
      SELECT A.*, B.Estado
      FROM ( SELECT * FROM `zmx-sugar-sales-ml-d.raw.SNIM_PRECIOS_REFERENCIA` ) A
      LEFT JOIN ( SELECT * FROM `zmx-sugar-sales-ml-d.raw.prueba_grupos` ) B
      ON A.Centro_distribucion = B.Ingenio)
    GROUP BY date_time, Estado;""").to_dataframe() 
    return data_sniim_estado

def load_data_bh():
    client = bigquery.Client()
    data_precio_bh_estado = client.query(f""" SELECT DATE(fecha) as Fecha, ingenio as Ingenio, Estado, AVG(precio_base) AS Precio_Base_Promedio FROM (
    SELECT A.*, B.Estado
    FROM ( SELECT * FROM `zmx-sugar-sales-ml-d.raw.precio-base-historico` ) A
    LEFT JOIN ( SELECT * FROM `zmx-sugar-sales-ml-d.raw.prueba_grupos` ) B
    ON A.ingenio = B.Ingenio)
    GROUP BY Fecha, Ingenio, Estado; """).to_dataframe()
    return data_precio_bh_estado