def convertir_a_minuscula(df):
	'''
	Convierte los nombres de columnas de dataframe a minuscula

	Params
	------
	df (dataframe): dataframe 

	Returns
	------
	dataframe con columnas en minuscula

	'''
    df.columns = map(str.lower, df.columns)
    return

def agregar_anio(df,variable):
	'''
	Agrega una columna con el anio extraido de una columna de dataframe de formato datetime

	Params
	------
	df (dataframe): dataframe que contiene columna datetime de la que se extraera el anio.
	variable (string): nombre de columna que contiene la fecha desde donde se quiere extraer el anio.

	Returns
	-------
	dataframe con columna con anio

	'''
    df['anio'] = df[variable].dt.year