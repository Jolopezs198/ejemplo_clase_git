#tratamiento de datos
import pandas as pd

import numpy as np

#visualizacion
import matplotlib.pyplot as plt
import seaborn as sns

#se carga y se revisa data "tmdb"

tmdb = pd.read_csv('cine/data/tmdb_movies_data.csv')
tmdb_b = pd.read_csv('cine/data/movies.csv')

tmdb_b['release_date'] = pd.to_datetime(tmdb_b['release_date'])
filtro = tmdb_b['release_date'].dt.year.between(2005,2015) 
tmdb_b = tmdb_b[filtro]

print(tmdb_b['release_date'].head(20))
print(tmdb_b.info())
filtro_columnas = tmdb_b[['id','popularity','original_language','vote_count','budget','revenue']]
filtro_columnas.info()

tmdb = pd.merge(tmdb, filtro_columnas, on='id', how='inner')
print(tmdb.head())
print(tmdb.info())


#revisamos que el dataset "tmdb" tenga registros unicos
tmdb.shape
tmbd_id = tmdb.id.unique()
tmbd_id.shape

#se intenta identificar registro duplicado
tmdb_n_pelis_x_id = tmdb.groupby(['id'], as_index=False)['original_title'].count() # ACA SI FUNCIONA ES DECIR 'as_index=False' evita que la clave quede como indice ademas 
tmdb_n_pelis_x_id.to_csv('cine/data/tmdb_n_pelis_x_id.csv', index = True)
tmdb_n_pelis_x_id.shape


#se quitan duplicados de "tmdb"
tmdb = tmdb.drop_duplicates()
tmdb.shape
tmdb.columns
total_productos = tmdb['director'].value_counts()
print(total_productos)

#se filtra tmdb por año 2005  - 2015
#filtro = (tmdb['release_year'].isin((2005,'2006','2007','2008','2009','2010','2011','2012','2013','2014','2015'))) 
filtro = (tmdb['release_year'].between(2005,2015)) 
tmdb = tmdb[filtro]

#contar productos por año
print(type(tmdb['movie_id']))
print(tmdb['runtime'].unique())
print(tmdb.info())
tmdb.shape
total_productos = tmdb['release_year'].value_counts()
print(total_productos)


# Mostrar la cantidad de valores nulos por columna
print(tmdb.isnull())
contador_nulos = tmdb.isnull().sum()
print(contador_nulos) # 2843 nulos en la columna Producto comprado
print(tmdb['keywords']) # 2843 nulos en la columna Producto comprado

# se reemplazan valores nulos por columna
tmdb.fillna({ 
            'imdb_id': 'sin imdb id'                   
            ,'cast' : 'sin cast'                     
            ,'homepage' : 'sin home page'               
            ,'director' : 'sin director'               
            ,'tagline'  : 'sin tagline'               
            ,'keywords' : 'sin keywords'              
            ,'overview' : 'sin overview'                  
            ,'genres'   : 'sin genres'                 
            ,'production_companies' : 'sin production companies'     
            }, inplace=True)




print(tmdb.columns)
print(tmdb.info())
print(tmdb['release_date'].tail)

#formatear tipos campos
tmdb = tmdb.astype({'imdb_id':'string'})
tmdb = tmdb.astype({'original_title':'string'})
tmdb = tmdb.astype({'cast':'string'})
tmdb = tmdb.astype({'homepage':'string'})
tmdb = tmdb.astype({'director':'string'})
tmdb = tmdb.astype({'tagline':'string'})
tmdb = tmdb.astype({'keywords':'string'})
tmdb = tmdb.astype({'overview':'string'})
tmdb = tmdb.astype({'genres':'string'})
tmdb = tmdb.astype({'production_companies':'string'})
tmdb = tmdb.astype({'release_date':'string'})

tmdb['release_date'] = pd.to_datetime(tmdb['release_date'])

print(tmdb.info())
print(tmdb['genres'].head(10))

#se crea tabla de actores
id_cast = tmdb['id'].astype(str)+ '|' + tmdb['cast'] 
tmdb_casting = id_cast.str.split("|",expand=True)
tmdb_casting.info()
print(tmdb_casting.head(30))
tmdb_casting.shape
tmdb_casting = pd.melt(tmdb_casting, id_vars=[0], value_vars=[1,2,3,4,5], value_name="actores", var_name="id_actor")
print(tmdb_casting.head(50))
tmdb_casting.info()
tmdb_casting = tmdb_casting.rename(columns = {0 : "id"})
tmdb_casting["id"] = pd.to_numeric(tmdb_casting['id'])
print(tmdb_casting.loc[tmdb_casting["id"] == 135397])
tmdb_casting = tmdb_casting.replace({'cast':{'PenÃ©lope Cruz' : 'Penelope Cruz'}})  

n_pelis_x_actor = tmdb_casting.groupby(['actores'], as_index=False)['actores'].aggregate(['count'])
print(n_pelis_x_actor.head())
print(n_pelis_x_actor.loc[n_pelis_x_actor["count"].between(20,30)])

# Se eliminan nulos en tabla actores
print(tmdb_casting.isnull())
contador_nulos = tmdb_casting.isnull().sum()
print(contador_nulos) # 2843 nulos en la columna Producto comprado
print(tmdb_casting.loc[tmdb_casting["id"] == 15651])
print(tmdb_casting.tail())
tmdb_casting = tmdb_casting.dropna()

#se crea tabla actores_peiliculas
tmdb_pelis_act =  pd.merge(tmdb, tmdb_casting, on='id', how='left')
print(tmdb_pelis_act[['id','original_title','actores']].head())
tmdb_pelis_act.info()

#cuales son los actores que han participado en peliculas top 5 mas populares por cada año
tmdb_pelis_act = tmdb_pelis_act.sort_values(['popularity_y'],ascending = False)
print(tmdb_pelis_act.info())
print(tmdb_pelis_act.head(25))
acts_pelis_pop_05 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2005]
acts_pelis_pop_05 = acts_pelis_pop_05.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_05 = acts_pelis_pop_05[:25]
acts_pelis_pop_06 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2006]
acts_pelis_pop_06 = acts_pelis_pop_06.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_06 = acts_pelis_pop_06[:25]
acts_pelis_pop_07 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2007]
acts_pelis_pop_07 = acts_pelis_pop_07.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_07 = acts_pelis_pop_07[:25]
acts_pelis_pop_08 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2008]
acts_pelis_pop_08 = acts_pelis_pop_08.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_08 = acts_pelis_pop_08[:25]
acts_pelis_pop_09 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2009]
acts_pelis_pop_09 = acts_pelis_pop_09.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_09 = acts_pelis_pop_09[:25]
acts_pelis_pop_10 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2010]
acts_pelis_pop_10 = acts_pelis_pop_10.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_10 = acts_pelis_pop_10[:25]
acts_pelis_pop_11 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2011]
acts_pelis_pop_11 = acts_pelis_pop_11.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_11 = acts_pelis_pop_11[:25]
acts_pelis_pop_12 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2012]
acts_pelis_pop_12 = acts_pelis_pop_12.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_12 = acts_pelis_pop_12[:25]
acts_pelis_pop_13 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2013]
acts_pelis_pop_13 = acts_pelis_pop_13.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_13 = acts_pelis_pop_13[:25]
acts_pelis_pop_14 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2014]
acts_pelis_pop_14 = acts_pelis_pop_14.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_14 = acts_pelis_pop_14[:25]
acts_pelis_pop_15 = tmdb_pelis_act.loc[tmdb_pelis_act["release_year"] == 2015]
acts_pelis_pop_15 = acts_pelis_pop_15.sort_values(['popularity_y'],ascending = False)
acts_pelis_pop_15 = acts_pelis_pop_15[:25]

actores_pelis_pop =  pd.concat([acts_pelis_pop_05,acts_pelis_pop_06,acts_pelis_pop_07,acts_pelis_pop_08,acts_pelis_pop_09,acts_pelis_pop_10,acts_pelis_pop_11,acts_pelis_pop_12,acts_pelis_pop_13,acts_pelis_pop_14,acts_pelis_pop_15])



# graficar TOP five periodo
años_pelis_pop = actores_pelis_pop.groupby(['release_year','original_title'], as_index = False).agg({'popularity_y' : 'first'})
print(acts_pelis_pop_13['original_title'].head(25))
print(años_pelis_pop['release_year'].unique())

info = pd.DataFrame(años_pelis_pop['popularity_y'].sort_values(ascending = False))
info['original_title'] = años_pelis_pop['original_title']
data = list(map(str,(info['original_title'])))
x = list(data[:20])
y = list(info['popularity_y'][:20])
print(info.head(15))

ax = sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(10,6)})
plt.subplots_adjust(bottom=0.1, top=0.95, left=0.3)
ax.set_title("Top 20 peliculas populares 2005-2015",fontsize = 15)
ax.set_xlabel("popularidad",fontsize = 13)
sns.set_style("darkgrid")

plt.show()


# graficar TOP 20 n actuaciones de esos años considerar % de peliculas en base al total de peliculas actuadas
actores_vs_n_pelis_pop = actores_pelis_pop.groupby(['actores'], as_index=False).agg({ 'id': 'count', 'budget_y' : 'mean', 'revenue_y' : 'mean', 'popularity_y' : 'mean' })
actores_vs_n_pelis_tot = tmdb_pelis_act.groupby(['actores'], as_index=False).agg({ 'id': 'count', 'budget_y' : 'mean', 'revenue_y' : 'mean', 'popularity_y' : 'mean' })

actores_vs_n_pelis_pop = actores_vs_n_pelis_pop.sort_values(['id'],ascending = False)
actores_vs_n_pelis_tot = actores_vs_n_pelis_tot.sort_values(['id'],ascending = False)

print(actores_vs_n_pelis_pop[:20])
print(actores_vs_n_pelis_tot[:20])

actores_vs_n_pelis = pd.merge(actores_vs_n_pelis_pop, actores_vs_n_pelis_tot, on='actores', how='inner')
print(actores_vs_n_pelis[:20])
print(actores_vs_n_pelis_pop.shape)
actores_vs_n_pelis = actores_vs_n_pelis[['actores','id_x','id_y']][:20]
actores_vs_n_pelis = actores_vs_n_pelis.rename(columns = {'id_x' : "numero actuaciones peliculas top", 'id_y' : "numero actuaciones totales"})
actores_vs_n_pelis = actores_vs_n_pelis.set_index('actores')

data = pd.DataFrame()
data['actores'] = actores_vs_n_pelis_pop['actores'][:20].tolist()
data['n_peliculas'] = actores_vs_n_pelis_pop['id'][:20].tolist()
print(data.head())
colores = sns.color_palette("flare", 20)

# Crea el gráfico de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño de la figura según tus preferencias
plt.subplots_adjust(bottom=0.25, top=0.95)
sns.barplot(data=data, x='actores', y='n_peliculas', palette=colores)

#plt.bar(actores, n_peliculas)
# Agrega etiquetas y título al gráfico
plt.xlabel('actor')
plt.ylabel('n actuaciones')
plt.title('Top 20 actores vs n actuaciones populares')
# Rota las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)
# Muestra el gráfico
plt.show()


#graficar eficiencia n actuacxioones top vs tot actuaciones
actores_vs_n_pelis.plot(kind ='bar', figsize=(10,6))
plt.subplots_adjust(bottom=0.25, top=0.95)
plt.title('Top 20 actores vs numero actuaciones populares y totales ')
plt.xlabel('actor')
plt.ylabel('n actuaciones')
plt.legend(title='n actuaciones tot vs pop')
plt.xticks(rotation=45)
plt.show()



# graficar TOP 10 actores de esos años considerar % de peliculas en base al total de peliculas actuadas
actores_vs_budget_y_pop = actores_pelis_pop.groupby(['actores'], as_index=False).agg({ 'id': 'count', 'budget_y' : 'mean', 'revenue_y' : 'mean', 'popularity_y' : 'mean' })
actores_vs_budget_y_pop = actores_vs_budget_y_pop.sort_values(['budget_y'],ascending = False)
print(actores_vs_budget_y_pop.head(10))
print(actores_vs_budget_y_pop[:10])

data = pd.DataFrame()
data['actores'] = actores_vs_budget_y_pop['actores'][:10].tolist()
data['presupuesto'] = actores_vs_budget_y_pop['budget_y'][:10].tolist()
colores = sns.color_palette("dark:#5A9_r", 10)

# Crea el gráfico de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño de la figura según tus preferencias
plt.subplots_adjust(bottom=0.25, top=0.95)
sns.barplot(data=data, x='actores', y='presupuesto',palette=colores)
#plt.bar(actores, n_peliculas)
# Agrega etiquetas y título al gráfico
plt.xlabel('actor')
plt.ylabel('presupuesto $')
plt.title('Top 10 actores vs presupuestos')
# Rota las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)
# Muestra el gráfico
plt.show()


# graficar TOP 20 actores de esos años considerar % de peliculas en base al total de peliculas actuadas
actores_vs_pop = actores_pelis_pop.groupby(['actores'], as_index=False).agg({ 'id': 'count', 'budget_y' : 'mean', 'revenue_y' : 'mean', 'popularity_y' : 'mean' })
actores_vs_pop = actores_vs_pop.sort_values(['popularity_y'],ascending = False)
print(actores_vs_pop.head(10))
print(actores_vs_pop[:10])

data = pd.DataFrame()
data['actores'] = actores_vs_pop['actores'][:10].tolist()
data['popularidad'] = actores_vs_pop['popularity_y'][:10].tolist()
colores = sns.color_palette("ch:s=.25,rot=-.25", 10)
# Crea el gráfico de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño de la figura según tus preferencias
plt.subplots_adjust(bottom=0.29, top=0.95)
sns.barplot(data=data, x='actores', y='popularidad', palette=colores)
#plt.bar(actores, n_peliculas)
# Agrega etiquetas y título al gráfico
plt.xlabel('actor')
plt.ylabel('popularidad')
plt.title('Top 10 actores vs prom popularidad')
# Rota las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)
# Muestra el gráfico
plt.show()


#graficar año vs popularidad
año_vs_pop = tmdb.loc[tmdb["release_year"] >= 2005]
año_vs_pop = año_vs_pop.groupby('release_year')['popularity_y'].mean()
plt.plot(año_vs_pop)
print(año_vs_pop.head(11))

#setup the title and labels of the figure.
plt.title("Año Vs popularidad promedio",fontsize = 14)
plt.xlabel('Año lanzamiento',fontsize = 13)
plt.ylabel('Popularidad promedio',fontsize = 13)

#setup the figure size.
sns.set(rc={'figure.figsize':(10,5)})
plt.subplots_adjust(top=0.95)
sns.set_style("whitegrid")
plt.show()


#graficar año vs rentabilidad
año_vs_rent = tmdb.loc[tmdb["release_year"] >= 2005]
año_vs_rent = año_vs_rent.groupby(['release_year'], as_index=False).agg({ 'budget_y': 'mean', 'revenue_y' : 'mean'})
print(año_vs_rent.head())
x=año_vs_rent['release_year']
y1=año_vs_rent['budget_y']
y2=año_vs_rent['revenue_y']

colors=['orange', 'purple']

plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Presupuesto")
plt.plot(x,y2,label="Retorno")

plt.title("Rentabilidad por año",fontsize=15)
plt.xlabel("Año lanzamiento",fontsize=13)
plt.ylabel("presupuesto vs retorno promedio $",fontsize=13)
plt.legend()
plt.show()


#graficar mes vs rentabilidad
print(tmdb.head())
mes_vs_rent = tmdb.loc[tmdb["release_year"] >= 2005]
mes_vs_rent['mes'] = mes_vs_rent['release_date'].dt.month
mes_vs_rent = mes_vs_rent.groupby(['mes'], as_index=False).agg({ 'budget_y': 'mean', 'revenue_y' : 'mean'})
print(mes_vs_rent['mes'].unique())
print(mes_vs_rent.head(12))


x=mes_vs_rent['mes']
y1=mes_vs_rent['budget_y']
y2=mes_vs_rent['revenue_y']

colors=['orange', 'purple']

plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Presupuesto")
plt.plot(x,y2,label="Retorno")

plt.title("Rentabilidad por mes",fontsize=15)
plt.xlabel("Mes lanzamiento",fontsize=13)
plt.ylabel("presupuesto vs retorno promedio $",fontsize=13)
plt.legend()
plt.show()


#graficar mes vs rentabilidad 2005 vs 2015
print(tmdb.head())
mes_vs_rent = tmdb.loc[tmdb["release_year"] == 2005]
mes_vs_rent['mes'] = mes_vs_rent['release_date'].dt.month
mes_vs_rent = mes_vs_rent.groupby(['mes'], as_index=False).agg({ 'budget_y': 'mean', 'revenue_y' : 'mean'})
print(mes_vs_rent['mes'].unique())
print(mes_vs_rent.head(12))

#sub plot
plt.figure(figsize=(8, 6))

#Histograma de altura
plt.subplot(2, 1, 1)
x=mes_vs_rent['mes']
y1=mes_vs_rent['budget_y']
y2=mes_vs_rent['revenue_y']

colors=['orange', 'purple']
plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Presupuesto")
plt.plot(x,y2,label="Retorno")
plt.title("Rentabilidad 2005",fontsize=15)
plt.xlabel("Mes lanzamiento",fontsize=13)
plt.ylabel("promedio $",fontsize=13)
plt.legend()


print(tmdb.head())
mes_vs_rent = tmdb.loc[tmdb["release_year"] == 2015]
mes_vs_rent['mes'] = mes_vs_rent['release_date'].dt.month
mes_vs_rent = mes_vs_rent.groupby(['mes'], as_index=False).agg({ 'budget_y': 'mean', 'revenue_y' : 'mean'})
print(mes_vs_rent['mes'].unique())
print(mes_vs_rent.head(12))

#Histograma de peso
plt.subplot(2, 1, 2)
x=mes_vs_rent['mes']
y1=mes_vs_rent['budget_y']
y2=mes_vs_rent['revenue_y']

colors=['orange', 'purple']
plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Presupuesto")
plt.plot(x,y2,label="Retorno")
plt.title("Rentabilidad 2015",fontsize=15)
plt.xlabel("Mes lanzamiento",fontsize=13)
plt.ylabel("Promedio $",fontsize=13)
plt.legend()

plt.tight_layout()
plt.show()





#EJEMPLO GRAFICO 3 BARRAS
#ventas por mes y producto---------------------------------------------------
ventas_mes_producto = df.groupby(['Mes', 'Producto comprado'])['Precio'].sum()

ventas_mes_producto.info()
print(ventas_mes_producto)


ventas_mes_producto = ventas_mes_producto.unstack()
#graficar barra agrupadas por producto en un mes
ventas_mes_producto.plot(kind ='bar', figsize=(10,6))

plt.title('Ventas por mes y producto')
plt.xlabel('Mes')
plt.ylabel('Ventas')
plt.legend(title='Producto')

plt.show()
#--------------------------------------------------------------------------------------

actores_pelis_pop[:60].to_csv('cine/data/imdb_id/prueba.csv', index = True)


print(acts_pelis_pop_06.head(25))

#se crea tabla de generos
id_gen = tmdb['id'].astype(str)+ '|' + tmdb['genres'] 
tmdb_generos = id_gen.str.split("|",expand=True)
tmdb_generos.info()
print(tmdb_generos.head(30))
tmdb_generos.shape
tmdb_generos = pd.melt(tmdb_generos, id_vars=[0], value_vars=[1,2,3,4,5], value_name="genero", var_name="id_genero")
print(tmdb_generos.head(50))
tmdb_generos.info()
tmdb_generos = tmdb_generos.rename(columns = {0 : "id"})
tmdb_generos["id"] = pd.to_numeric(tmdb_generos['id'])
print(tmdb_generos.loc[tmdb_generos["id"] == 135397])

# Se eliminan nulos en tabla generos
print(tmdb_generos.isnull())
contador_nulos = tmdb_generos.isnull().sum()
print(contador_nulos) # 2843 nulos en la columna Producto comprado
print(tmdb_generos.loc[tmdb_generos["id"] == 15651])
print(tmdb_generos.tail())
tmdb_generos = tmdb_generos.dropna()

#se crea tabla generos_peiliculas
tmdb_pelis_gen =  pd.merge(tmdb, tmdb_generos, on='id', how='left')
print(tmdb_pelis_gen[['id','original_title','director']].head())
tmdb_pelis_gen = tmdb_pelis_gen.sort_values(['popularity_y'],ascending = False)
#tmdb_pelis_gen = tmdb_pelis_gen[:200]
print(tmdb_pelis_gen.head(22))

# Se eliminan nulos en tabla tmdb_pelis_gen
print(tmdb_pelis_gen.isnull())
contador_nulos = tmdb_pelis_gen.isnull().sum()
print(contador_nulos) # 2843 nulos en la columna Producto comprado
print(tmdb_pelis_gen.loc[tmdb_pelis_gen["id"] == 15651])
print(tmdb_pelis_gen.tail())
tmdb_pelis_gen = tmdb_pelis_gen.dropna()


tmdb_pelis_gen.drop(tmdb_pelis_gen[(tmdb_pelis_gen['genero'] == 'sin genres')].index, inplace=True)
#Crear grafico generos mas PRODUCIDOS torta
gen_vs_pro = tmdb_pelis_gen.groupby('genero', as_index=False)['id'].count()
gen_vs_pro = gen_vs_pro.sort_values(['id'],ascending = False)
gen_vs_pro_otros = gen_vs_pro.loc[gen_vs_pro["id"] < 200]
gen_vs_pro_otros['genero'] = 'otros' 
gen_vs_pro_otros = gen_vs_pro_otros.groupby('genero', as_index=False)['id'].sum()
gen_vs_pro = gen_vs_pro.loc[gen_vs_pro["id"] >= 200]
gen_vs_pro = pd.concat([gen_vs_pro,gen_vs_pro_otros])
print(gen_vs_pro_otros.head(30))
print(gen_vs_pro.head(20))
plt.figure(figsize=(10, 6))
plt.pie(gen_vs_pro['id'], autopct='%1.1f%%',  pctdistance = 1.1, startangle=90)
plt.title('Distribución de generos mas producidos')
plt.tight_layout()
plt.legend(loc='best', bbox_to_anchor=(0.9, 0., 0.4, 0.8), labels=gen_vs_pro['genero'])
plt.subplots_adjust(top=0.95,  right= 0.8)
plt.show()


#Crear grafico generos mas POPULARES 
print(tmdb_pelis_gen.info())
gen_vs_pop = tmdb_pelis_gen.groupby('genero', as_index=False)['popularity_y'].mean()
gen_vs_pop = gen_vs_pop.sort_values(['popularity_y'],ascending = False)
gen_vs_pop_otros = gen_vs_pop.loc[gen_vs_pop["popularity_y"] < 0.5]
gen_vs_pop_otros['genero'] = 'otros' 
gen_vs_pop_otros = gen_vs_pop_otros.groupby('genero', as_index=False)['popularity_y'].sum()
gen_vs_pop = gen_vs_pop.loc[gen_vs_pop["popularity_y"] >= 0.5]
gen_vs_pop = pd.concat([gen_vs_pop,gen_vs_pop_otros])
print(gen_vs_pop_otros.head(30))
print(gen_vs_pop.head(20))

info = pd.DataFrame(gen_vs_pop['popularity_y'].sort_values(ascending = False))
print(info.info())
print(info.shape)
print(gen_vs_pop.index)
print(info.index.duplicated())

print(gen_vs_pop.head(20))
info = info[~info.index.duplicated()]
gen_vs_pop = gen_vs_pop[~gen_vs_pop.index.duplicated()]
info['genero'] = gen_vs_pop['genero']
print(info)
data = list(map(str,(info['genero'])))
x = list(data)
y = list(info['popularity_y'])
ax = sns.pointplot(x=y,y=x)
sns.set(rc={'figure.figsize':(10,6)})
plt.subplots_adjust(bottom=0.1, top=0.95, left=0.3)
ax.set_title("Genero vs popularidad 2005-2015",fontsize = 15)
ax.set_xlabel("Popularidad",fontsize = 13)
ax.set_ylabel("Genero",fontsize = 13)
sns.set_style("darkgrid")
plt.show()



# graficar generos mas populares por año
gen_pop_vs_año = pd.pivot_table(tmdb_pelis_gen.loc[tmdb_pelis_gen["release_year"] >= 2005], values=['popularity_y'], index=['release_year'], columns=['genero'], aggfunc=np.mean)#.sort_values(('popularity_y', 2015),ascending = False)
gen_pop_vs_año = pd.DataFrame(gen_pop_vs_año)
#columnas_año = gen_pop_vs_año.columns[7:]  # Excluye las columnas 'Country' y 'Total'
#gen_pop_vs_año = gen_pop_vs_año.loc[:, columnas_año]

#print(columnas_año)
print(gen_pop_vs_año.head(20))
print(gen_pop_vs_año.info())
gen_pop_vs_año = gen_pop_vs_año.round({('popularity_y', 2013): 1})
gen_pop_vs_año = gen_pop_vs_año.round({('popularity_y', 2014): 1})
gen_pop_vs_año = gen_pop_vs_año.round({('popularity_y', 2015): 1})

gen_pop_vs_año = gen_pop_vs_año.fillna(0)

gen_pop_vs_año.to_csv('cine/data/gen_pop_vs_año.csv', index = True)

gen_pop_vs_año.plot(kind='area')
plt.subplots_adjust(bottom=0.37, top=0.95)
plt.xlabel('Genero',size = 14)
plt.ylabel('Popularidad',size = 14)
plt.title('Popularidad generos por año')
plt.legend(loc='best', bbox_to_anchor=(0.8, 0.0, 0.0, 1.0))
#plt.xticks([i for i in range(16)])
#plt.set(xlim = (min(x), max(x)), xticks = x)
plt.xticks(rotation=45)
plt.show()


#graficar genero vs pupularidad por año arcaico
print(gen_pop_vs_año.head(20))
gen_pop_vs_año = gen_pop_vs_año.sort_values(('popularity_y', 2015),ascending = False)
x=gen_pop_vs_año.index.to_list()
print(x)
y1=gen_pop_vs_año[('popularity_y', 'Adventure')]
y2=gen_pop_vs_año[('popularity_y', 'Science Fiction')]
y3=gen_pop_vs_año[('popularity_y', 'Western')]
y4=gen_pop_vs_año[('popularity_y', 'Fantasy')]
y5=gen_pop_vs_año[('popularity_y', 'Action')]
#y6=gen_pop_vs_año[('popularity_y', 'Family')]
#y7=gen_pop_vs_año[('popularity_y', 'War')]
#y8=gen_pop_vs_año[('popularity_y', 'Animation')]
#y9=gen_pop_vs_año[('popularity_y', 'Crime')]
#y10=gen_pop_vs_año[('popularity_y', 'Thriller')]


print(y1.head(16))
print(y2.shape)

colors=['#C0392B', '#AF7AC5', '#5499C7', 'black', '#52BE80', '#0000CC', '#F8C471', '#2E4053', '#795548', 'black']

plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Adventure")
plt.plot(x,y2,label="Science Fiction")
plt.plot(x,y3,label="Western")
plt.plot(x,y4,label="Fantasy")
plt.plot(x,y5,label="Action")
#plt.plot(x,y6,label="Family")
#plt.plot(x,y7,label="War")
#plt.plot(x,y8,label="Animation")
#plt.plot(x,y9,label="Crime")
#plt.plot(x,y10,label="Thriller")


plt.subplots_adjust(bottom=0.22, top=0.95)
plt.title("Top 5 generos mas populares por año",fontsize=15)
plt.xlabel("Año",fontsize=13)
plt.ylabel("Popularidad",fontsize=13)
plt.legend()
plt.xticks(rotation=45)
plt.show()



#graficar genero vs rentabilidad
gen_vs_rent = tmdb_pelis_gen.groupby(['genero'], as_index=False).agg({ 'budget_y': 'mean', 'revenue_y' : 'mean'})
print(gen_vs_rent.head())
gen_vs_rent = gen_vs_rent.sort_values(['revenue_y'],ascending = False)
x=gen_vs_rent['genero']
y1=gen_vs_rent['budget_y']
y2=gen_vs_rent['revenue_y']
colors=['orange', 'purple']

plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Presupuesto")
plt.plot(x,y2,label="Retorno")
plt.subplots_adjust(bottom=0.22, top=0.95)
plt.title("Rentabilidad por genero",fontsize=15)
plt.xlabel("Genero",fontsize=13)
plt.ylabel("Presupuesto vs Retorno promedio $",fontsize=13)
plt.legend()
plt.xticks(rotation=45)
plt.show()



#se crea tabla de directores
print(tmdb['director'])
id_dir = tmdb['id'].astype(str)+ '|' + tmdb['director'] 
tmdb_directores = id_dir.str.split("|",expand=True)
tmdb_directores.info()
print(tmdb_directores.head(30))
tmdb_directores.shape
tmdb_directores = pd.melt(tmdb_directores, id_vars=[0], value_vars=[1,2,3,4,5], value_name="director", var_name="id_director")
print(tmdb_directores.head(50))
tmdb_directores.info()
tmdb_directores = tmdb_directores.rename(columns = {0 : "id"})
tmdb_directores["id"] = pd.to_numeric(tmdb_directores['id'])
print(tmdb_directores.loc[tmdb_directores["id"] == 135397])


# Se eliminan nulos en tabla directores
print(tmdb_directores.isnull())
contador_nulos = tmdb_directores.isnull().sum()
print(contador_nulos) # 2843 nulos en la columna Producto comprado
print(tmdb_directores.loc[tmdb_directores["id"] == 15651])
print(tmdb_directores.tail())
tmdb_directores = tmdb_directores.dropna()

#se crea tabla directores_peiliculas
tmdb_pelis_dir =  pd.merge(tmdb, tmdb_directores, on='id', how='left')
print(tmdb_pelis_dir[['id','original_title','genero']].head())
tmdb_pelis_dir.info()
print(tmdb_pelis_dir.head())

# Crea un boxplot de una variable específica
dir_vs_rent = tmdb_pelis_dir.groupby(['director_y'], as_index=False).agg({ 'budget_y': 'mean', 'revenue_y' : 'mean'}).sort_values(['revenue_y'],ascending = False)
dir_vs_rent = dir_vs_rent[:10]
print(dir_vs_rent.head(10))
print(tmdb_pelis_dir.head())
print(tmdb_pelis_dir['revenue_y'].loc[tmdb_pelis_dir["director_y"] == 'James Cameron'])

x=dir_vs_rent['director_y']
y1=dir_vs_rent['budget_y']
y2=dir_vs_rent['revenue_y']
colors=['orange', 'purple']

plt.gca().set_prop_cycle(color=colors)
plt.plot(x,y1,label="Presupuesto")
plt.plot(x,y2,label="Retorno")
plt.subplots_adjust(bottom=0.22, top=0.95)
plt.title("Top 10 Rentabilidad por director",fontsize=15)
plt.xlabel("Director",fontsize=13)
plt.ylabel("Presupuesto vs Retorno promedio $",fontsize=13)
plt.legend()
plt.xticks(rotation=45)
plt.show()


#se cargan datasets imdb por genero
accion = pd.read_csv('cine/data/imdb_id/action.csv')
aventura = pd.read_csv('cine/data/imdb_id/adventure.csv')
animacion = pd.read_csv('cine/data/imdb_id/animation.csv')
biografia = pd.read_csv('cine/data/imdb_id/biography.csv')
crimen = pd.read_csv('cine/data/imdb_id/crime.csv')
familia = pd.read_csv('cine/data/imdb_id/family.csv')
fantasia = pd.read_csv('cine/data/imdb_id/fantasy.csv')
cine_negro = pd.read_csv('cine/data/imdb_id/film-noir.csv')
historia = pd.read_csv('cine/data/imdb_id/history.csv')
terror = pd.read_csv('cine/data/imdb_id/horror.csv')
misterio = pd.read_csv('cine/data/imdb_id/mystery.csv')
romance = pd.read_csv('cine/data/imdb_id/romance.csv')
ficcion = pd.read_csv('cine/data/imdb_id/scifi.csv')
deporte = pd.read_csv('cine/data/imdb_id/sports.csv')
suspenso = pd.read_csv('cine/data/imdb_id/thriller.csv')
guerra = pd.read_csv('cine/data/imdb_id/war.csv')

#se unen datasets generos en dataset "imdb"
imdb = pd.concat([accion,aventura,animacion,biografia,crimen,familia,fantasia,cine_negro,historia,terror,misterio,romance,ficcion,deporte,suspenso,guerra])
imdb.shape
imdb.head()
imdb.columns

#se verifica cuantos resgistros deberia tener el data set "imdb" segun movie_id
movie_id_uni = imdb.movie_id.unique()
movie_id_uni.shape

#se intenta eliminar duplicados del data set "imdb"
imdb = imdb.drop_duplicates()
imdb.columns
imdb.shape


#reemplazando nulos
# Muestra una matriz de TRUE AND FALSE
print(imdb.isnull())

# Mostrar la cantidad de valores nulos por columna
contador_nulos = imdb.isnull().sum()
print(contador_nulos) # 2843 nulos en la columna Producto comprado

# Obtener los valores de una columna específica y contar las repeticiones
total_productos = imdb['year'].value_counts()
print(total_productos)

print(imdb)
print(imdb.info())
print(imdb['runtime'].unique())
imdb.fillna({ 'movie_name': 'sin titulo'
                 , 'year' : '0'
                 , 'certificate':'sin certificado'
                 , 'runtime':'0 min'
                 , 'rating': 0
                 , 'director': 'sin director'
                 , 'director_id': 'sin director id'
                 , 'star':'sin star'
                 ,'star_id':'sin star id'
                 , 'votes':0
                 ,'gross(in $)':0 }
                 , inplace=True)



#se formatean los tipos de datos de las columnas
imdb = imdb.astype({'movie_id':'string'})
imdb = imdb.astype({'movie_name':'string'})
imdb = imdb.astype({'year':'string'})
imdb = imdb.astype({'certificate':'string'})
imdb = imdb.astype({'runtime':'string'})
imdb = imdb.astype({'genre':'string'})
imdb = imdb.astype({'description':'string'})
imdb = imdb.astype({'director':'string'})
imdb = imdb.astype({'director_id':'string'})
imdb = imdb.astype({'star':'string'})
imdb = imdb.astype({'star_id':'string'})


def limpia_duracion(x):
    if (',' in x) :
        temp = x.replace(',','')
        return int(temp.split()[0])
    else: 
        return int(x.split()[0])

imdb['runtime'] = imdb['runtime'].apply(lambda x: limpia_duracion(x))# ESTO PERMITE ITERAR SIN USAR BUCLE X ASUME EL VALOR DE DF QUE EN ESTE CASO ES LA COLUMNA RUNTIME SIN NULOS TAMBIEN SE PUEDE ACCEDER DIRECTAMENTE DESDE EL DATASET A LA COLUMNA
#LOS NULOS JODEN EL SPLIT Y DA ERROR DE PUNTO FLOTANTE NO TIENE EL ATRIBUTO SPLIT

#finalmente se logra grupar dejando asi registros unicos 
print(imdb.info())
print(imdb.runtime)
imdb.columns
imdb = imdb.groupby(['movie_id'], as_index=False).agg({'gross(in $)':'max','director_id':'first','director':'first','star_id':'first','star':'first','year':'first','description':'first','movie_name':'first','rating':'max','genre':'first','runtime':'max','certificate':'first','votes': 'max'})#.reset_index()
print(imdb.info())
imdb.shape

#se verifica que no queden duplicados
imdb_repetidos = imdb.groupby(['movie_id'], as_index=False).agg({ 'movie_name': 'count'
                                                                    , 'year' : 'count'
                                                                    , 'certificate':'count'
                                                                    , 'runtime':'count'
                                                                    , 'genre' : 'count'
                                                                    , 'rating': 'count'
                                                                    , 'description' : 'count'
                                                                    , 'director': 'count'
                                                                    , 'director_id': 'count'
                                                                    , 'star':'count'
                                                                    ,'star_id':'count'
                                                                    , 'votes':'count'
                                                                    ,'gross(in $)':'count' })
imdb_repetidos.to_csv('cine/data/imdb_id/imdb_repetidos.csv', index = False)


imdb.to_csv('cine/data/imdb_id/imdb_uni.csv', index = False)


#verifico cantidad segun id
imdb_filtrados_uni = imdb['movie_id'].unique()
print(imdb_filtrados_uni.shape)


#se crea relacion tmdb_imdb_actores 
imdb = imdb.rename(columns = {'movie_id': 'imdb_id'})
tmdb_pelis_act.info()
tmdb_pelis_act.shape
imdb.info()
filtro_columnas = imdb[['imdb_id','rating']]
filtro_columnas.info()
tmdb_vs_imdb_actores = pd.merge(tmdb_pelis_act, filtro_columnas, on='imdb_id', how='inner')
tmdb_vs_imdb_actores.info()
tmdb_vs_imdb_actores.shape
print(tmdb_vs_imdb_actores.tail(10))

#se crea relacion tmdb_imdb_generos 
imdb = imdb.rename(columns = {'movie_id': 'imdb_id'})
tmdb_pelis_gen.info()
tmdb_pelis_gen.shape
imdb.info()
filtro_columnas = imdb[['imdb_id','rating']]
filtro_columnas.info()
tmdb_vs_imdb_generos = pd.merge(tmdb_pelis_gen, filtro_columnas, on='imdb_id', how='inner')
tmdb_vs_imdb_generos.info()
tmdb_vs_imdb_generos.shape
print(tmdb_vs_imdb_generos.tail(10))

#cantidad peliculas x actor
n_pelis_x_actor = tmdb_vs_imdb_actores.groupby(['id_actor','actores'], as_index=False).agg({ 'original_title': 'count'})
print(n_pelis_x_actor)
print(n_pelis_x_actor.info())
n_pelis_x_actor = n_pelis_x_actor.rename(columns = {"original_title" : "n_pelis_x_actor"})
print(n_pelis_x_actor.loc[n_pelis_x_actor["n_pelis_x_actor"] > 10])
#19261
tmdb_vs_imdb_actores = pd.merge(tmdb_vs_imdb_actores, n_pelis_x_actor, on=['id_actor','actores'], how='inner')

#promedio rating x actor
rating_x_actor = tmdb_vs_imdb_actores.groupby(['id_actor','actores'], as_index=False)['rating'].mean()#.aggregate({ 'rating': ['mean','max', 'min']})
print(rating_x_actor.tail(10))
print(rating_x_actor.info())
rating_x_actor = rating_x_actor.rename(columns = {"rating" : "mean_rat_actor"})
print(rating_x_actor.loc[rating_x_actor["mean_rat_actor"] > 5])
#19261
tmdb_vs_imdb_actores = pd.merge(tmdb_vs_imdb_actores, rating_x_actor, on=['id_actor','actores'], how='inner')
tmdb_vs_imdb_actores.to_csv('cine/data/imdb_id/tmdb_vs_imdb_actores.csv', index = False)

#cant horas x actor
horas_x_actor = tmdb_vs_imdb_actores.groupby(['id_actor','actores'], as_index=False)['runtime'].sum()#.aggregate({ 'rating': ['mean','max', 'min']})
print(horas_x_actor.tail(10))
print(horas_x_actor.info())
horas_x_actor = horas_x_actor.rename(columns = {"runtime" : "runtime_actor"})
print(horas_x_actor.loc[horas_x_actor["runtime_actor"] > 5])
#19261
tmdb_vs_imdb_actores = pd.merge(tmdb_vs_imdb_actores, horas_x_actor, on=['id_actor','actores'], how='inner')
tmdb_vs_imdb_actores.to_csv('cine/data/imdb_id/tmdb_vs_imdb_actores.csv', index = False)


matriz_correlacion = tmdb_vs_imdb_actores[['id','popularity_y','budget_y','revenue_y','runtime','vote_count','vote_average','release_year','rating','n_pelis_x_actor','mean_rat_actor','runtime_actor']].corr()
#graficar nuestra matriz de correalcion
sns.heatmap(matriz_correlacion, annot = True, cmap ='coolwarm')
plt.xticks(rotation=45)
plt.show()


print(tmdb_vs_imdb_actores.info())
#se agrupan valores para dejar registros unicos

gr_tmdb_vs_imdb_act = tmdb_vs_imdb_actores.groupby(['actores'], as_index=False).agg({ 'popularity_y': 'mean'
                                                                    , 'budget_y' : 'mean'
                                                                    , 'revenue_y':'mean'
                                                                    , 'runtime':'mean'
                                                                    , 'vote_count' : 'mean'
                                                                    , 'vote_average': 'mean'
                                                                    , 'release_year' : 'max'
                                                                    , 'budget_y_adj': 'mean'
                                                                    , 'revenue_y_adj': 'mean'
                                                                    ,'n_pelis_x_actor':'max'
                                                                    , 'mean_rat_actor':'max'
                                                                    ,'runtime_actor':'max' })
gr_tmdb_vs_imdb_act.to_csv('cine/data/imdb_id/gr_tmdb_vs_imdb_act.csv', index = False)



matriz_correlacion = gr_tmdb_vs_imdb_act[['popularity_y','budget_y','revenue_y','runtime','vote_count','vote_average','release_year','n_pelis_x_actor','mean_rat_actor','runtime_actor']].corr()
#graficar nuestra matriz de correalcion
sns.heatmap(matriz_correlacion, annot = True, cmap ='coolwarm')
plt.xticks(rotation=45)
plt.show()

umbral = 0.5

caracteristica_seleccionada = matriz_correlacion[abs(matriz_correlacion) > umbral].dropna(axis=0, how='all').dropna(axis=1,how='all')
print(caracteristica_seleccionada)

sns.heatmap(caracteristica_seleccionada, annot = True, cmap ='coolwarm')
plt.show()


print(imdb.info())
matriz_correlacion = imdb[['gross(in $)','rating','runtime','votes']].corr()
#graficar nuestra matriz de correalcion IMDB
sns.heatmap(matriz_correlacion, annot = True, cmap ='coolwarm')
plt.xticks(rotation=45)
plt.show()

print(tmdb.info())
matriz_correlacion = tmdb[['id','popularity_y','budget_y','revenue_y','runtime','vote_count','vote_average','release_year','budget_y_adj','revenue_y_adj']].corr()
#graficar nuestra matriz de correalcion TMDB
sns.heatmap(matriz_correlacion, annot = True, cmap ='coolwarm')
plt.xticks(rotation=45)
plt.show()


#id vs release_year
#popularyty vs revenue_y
#popularyty vs budget_y
#budget_y vs revenue_y






#FORMA INCORRECTA DE ACTUALIZAR CAMPO A TRAVEZ DE UN BUCLE
accion['duracion_numeric'] = 0

for i in range(accion.shape[0]):
   temp = df[i]    
   accion.duracion_numeric[i] = temp.split()[0] 

accion.duracion_numeric.unique()
#---------------------------------------------------------


accion.shape
accion.duracion_numeric.unique()