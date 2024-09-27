import streamlit as st
#Importation de packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Pour les tests statistiques
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import requests
from joblib import dump
import math
import os



df = pd.read_csv("Fichier_de_base .csv")
df_21 =pd.read_csv("Fichier_2021.csv")
df2 = pd.read_csv("climate_change_indicators .csv", sep=";")   
  


st.title ("Analyse du rapport du bien-être sur terre")
st.sidebar.title("Sommaire")
pages =["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Introduction")
  st.write ("Ce projet vise à conduire une analyse approfondie des données du World Happiness Report afin d'évaluer le bonheur des pays du monde en utilisant une variété d'indicateurs socio-économiques tels que la santé, l'éducation, la corruption, l'économie et l'espérance de vie. L'objectif principal est de présenter ces données à travers des visualisations interactives bien conçues tout en identifiant les combinaisons de facteurs qui expliquent pourquoi certains pays sont mieux classés que d'autres en termes de bonheur")
  st.write("")

  #importer le jeu de données du 2005 à 2020. Et 2021
  
  st.write("Affichage des 10 premières lignes du datafram 2005-2020")
  st.dataframe(df.head(10))
  st.write(df.shape)

    
  st.write("Résumé statistique des variables numériques")
  st.dataframe(df.describe())  
  
  st.write ("Adjonction de la variable région et la variable température à notre dataset ")
  def get_regional_indicator(country_name):
      country_to_region = {
        "Denmark": "Western Europe",
        "France": "Western Europe",
        "Mexico": "Latin America and Caribbean",
        "Germany": "Western Europe",
        "Poland": "Central and Eastern Europe",
        "Spain": "Western Europe",
        "Greece": "Western Europe",
        "Brazil": "Latin America and Caribbean",
        "Sweden": "Western Europe",
        "Egypt": "Middle East and North Africa",
        "Saudi Arabia": "Middle East and North Africa",
        "Lebanon": "Middle East and North Africa",
        "Netherlands": "Western Europe",
        "Australia": "Australia and New Zealand",
        "United Kingdom": "Western Europe",
        "Canada": "North America",
        "Iran": "Middle East and North Africa",
        "Pakistan": "South Asia",
        "Hungary": "Central and Eastern Europe",
        "Czech Republic": "Central and Eastern Europe",
        "Belgium": "Western Europe",
        "Turkey": "Middle East and North Africa",
        "Jordan": "Middle East and North Africa",
        "Venezuela": "Latin America and Caribbean",
        "Italy": "Western Europe",
        "Japan": "East Asia",
        "Romania": "Central and Eastern Europe",
        "Portugal": "Western Europe",
        "Singapore": "Southeast Asia",
        "Sierra Leone": "Sub-Saharan Africa",
        "Rwanda": "Sub-Saharan Africa",
        "Chile": "Latin America and Caribbean",
        "Senegal": "Sub-Saharan Africa",
        "Russia": "Commonwealth of Independent States",
        "Colombia": "Latin America and Caribbean",
        "Chad": "Sub-Saharan Africa",
        "China": "East Asia",
        "South Korea": "East Asia",
        "Slovenia": "Central and Eastern Europe",
        "Uganda": "Sub-Saharan Africa",
        "Belarus": "Commonwealth of Independent States",
        "Trinidad and Tobago": "Latin America and Caribbean",
        "Togo": "Sub-Saharan Africa",
        "Benin": "Sub-Saharan Africa",
        "Thailand": "Southeast Asia",
        "Tanzania": "Sub-Saharan Africa",
        "Bolivia": "Latin America and Caribbean",
        "Tajikistan": "Commonwealth of Independent States",
        "Taiwan Province of China": "East Asia",
        "Switzerland": "Western Europe",
        "Botswana": "Sub-Saharan Africa",
        "Sri Lanka": "South Asia",
        "Burkina Faso": "Sub-Saharan Africa",
        "Cambodia": "Southeast Asia",
        "South Africa": "Sub-Saharan Africa",
        "Cameroon": "Sub-Saharan Africa",
        "Slovakia": "Central and Eastern Europe",
        "Philippines": "Southeast Asia",
        "Costa Rica": "Latin America and Caribbean",
        "Cuba": "Latin America and Caribbean",
        "Malawi": "Sub-Saharan Africa",
        "Madagascar": "Sub-Saharan Africa",
        "Guatemala": "Latin America and Caribbean",
        "Lithuania": "Central and Eastern Europe",
        "Haiti": "Latin America and Caribbean",
        "Latvia": "Central and Eastern Europe",
        "Honduras": "Latin America and Caribbean",
        "Malaysia": "Southeast Asia",
        "Laos": "Southeast Asia",
        "Kyrgyzstan": "Commonwealth of Independent States",
        "Kuwait": "Middle East and North Africa",
        "Kenya": "Sub-Saharan Africa",
        "India": "South Asia",
        "Kazakhstan": "Commonwealth of Independent States",
        "Indonesia": "Southeast Asia",
        "Jamaica": "Latin America and Caribbean",
        "Ireland": "Western Europe",
        "Hong Kong S.A.R. of China": "East Asia",
        "Ghana": "Sub-Saharan Africa",
        "Mali": "Sub-Saharan Africa",
        "Georgia": "Commonwealth of Independent States",
        "Cyprus": "Middle East and North Africa",
        "Paraguay": "Latin America and Caribbean",
        "Panama": "Latin America and Caribbean",
        "Palestinian Territories": "Middle East and North Africa",
        "Bangladesh": "South Asia",
        "Dominican Republic": "Latin America and Caribbean",
        "Norway": "Western Europe",
        "Ecuador": "Latin America and Caribbean",
        "Nigeria": "Sub-Saharan Africa",
        "Niger": "Sub-Saharan Africa",
        "Nicaragua": "Latin America and Caribbean",
        "El Salvador": "Latin America and Caribbean",
        "New Zealand": "Australia and New Zealand",
        "Estonia": "Central and Eastern Europe",
        "Nepal": "South Asia",
        "Mozambique": "Sub-Saharan Africa",
        "Finland": "Western Europe",
        "Moldova": "Commonwealth of Independent States",
        "Peru": "Latin America and Caribbean",
        "Ukraine": "Commonwealth of Independent States",
        "Israel": "Middle East and North Africa",
        "Azerbaijan": "Commonwealth of Independent States",
        "Vietnam": "Southeast Asia",
        "Uruguay": "Latin America and Caribbean",
        "Zimbabwe": "Sub-Saharan Africa",
        "Armenia": "Commonwealth of Independent States",
        "Austria": "Western Europe",
        "Argentina": "Latin America and Caribbean",
        "United States": "North America",
        "Zambia": "Sub-Saharan Africa",
        "United Arab Emirates": "Middle East and North Africa",
        "Uzbekistan": "Commonwealth of Independent States",
        "Liberia": "Sub-Saharan Africa",
        "Bosnia and Herzegovina": "Central and Eastern Europe",
        "Montenegro": "Central and Eastern Europe",
        "Croatia": "Central and Eastern Europe",
        "Central African Republic": "Sub-Saharan Africa",
        "Mongolia": "East Asia",
        "Bulgaria": "Central and Eastern Europe",
        "Albania": "Central and Eastern Europe",
        "Mauritania": "Sub-Saharan Africa",
        "Yemen": "Middle East and North Africa",
        "Kosovo": "Central and Eastern Europe",
        "Serbia": "Central and Eastern Europe",
        "North Macedonia": "Central and Eastern Europe",
        "Belize": "Latin America and Caribbean",
        "Guyana": "Latin America and Caribbean",
        "Namibia": "Sub-Saharan Africa",
        "Afghanistan": "South Asia",
        "Djibouti": "Sub-Saharan Africa",
        "Congo (Brazzaville)": "Sub-Saharan Africa",
        "Iceland": "Western Europe",
        "Iraq": "Middle East and North Africa",
        "Syria": "Middle East and North Africa",
        "Burundi": "Sub-Saharan Africa",
        "Congo (Kinshasa)": "Sub-Saharan Africa",
        "Qatar": "Middle East and North Africa",
        "Ivory Coast": "Sub-Saharan Africa",
        "Tunisia": "Middle East and North Africa",
        "Turkmenistan": "Commonwealth of Independent States",
        "Comoros": "Sub-Saharan Africa",
        "Bahrain": "Middle East and North Africa",
        "Somaliland region": "Sub-Saharan Africa",
        "Luxembourg": "Western Europe",
        "Malta": "Western Europe",
        "Sudan": "Sub-Saharan Africa",
        "Algeria": "Middle East and North Africa",
        "Morocco": "Middle East and North Africa",
        "Swaziland": "Sub-Saharan Africa",
        "Guinea": "Sub-Saharan Africa",
        "Lesotho": "Sub-Saharan Africa",
        "Oman": "Middle East and North Africa",
        "Angola": "Sub-Saharan Africa",
        "Gabon": "Sub-Saharan Africa",
        "Mauritius": "Sub-Saharan Africa",
        "Myanmar": "Southeast Asia",
        "North Cyprus": "Western Europe",
        "Suriname": "Latin America and Caribbean",
        "Libya": "Middle East and North Africa",
        "Ethiopia": "Sub-Saharan Africa",
        "Bhutan": "South Asia",
        "Somalia": "Sub-Saharan Africa",
        "South Sudan": "Sub-Saharan Africa",
        "Gambia": "Sub-Saharan Africa",
        "Maldives": "South Asia"
    }
      if country_name in country_to_region:
          return country_to_region[country_name]
      else:
          return "Unknown"
# Ajouter une colonne "Regional indicator" dans le DataFrame
  df["Regional indicator"] = df["Country name"].apply(get_regional_indicator)
   # Renommage des colonnes
  df_a_renamed = df2.rename(columns={'NMGB': 'Country name', 'Year': 'year'})
  # Fusion des DataFrames
  df = pd.merge(df, df_a_renamed[['Country name', 'year', 'Temperature']], on=['Country name', 'year'], how='left')
  st.dataframe(df.head())
  
  if st.checkbox("Afficher les NA") :
      st.dataframe(df.isna().sum()) 



elif page == pages[1] : 
  st.write("### DataVizualization")

  st.write ("A- Un histogramme nous permettra de voir la distribution du score de bonheur parmi les différents pays.")
  fig = plt.figure()
  sns.histplot(df ["Life Ladder"], kde=True, bins=10, color="g", edgecolor="black")
  plt.title("Distribution du Life Ladder")
  plt.xlabel("Life Ladder")
  plt.ylabel("Fréquence") 
  # Calculer la médiane
  median_value = df["Life Ladder"].median()
  # Ajouter la ligne de médiane
  plt.axvline(x=median_value, color="red", linestyle="--", label="Médiane")
  # Afficher le graphique avec la légende
  plt.legend()
  st.pyplot(fig)

  st.write ("### Interprétation :")
  st.write ("- Le graphe montre la fréquence des différents scores de bonheur sur une échelle de 0 à 8, où 0 représente le moins heureux et 8 le plus heureux.")
  st.write ("- La forme du graphe est asymétrique, avec une queue plus longue vers la gauche. Cela indique qu’il y a plus de gens qui ont un score de bonheur inférieur à la moyenne que supérieur à la moyenne.")
  st.write ("")

  st.write ( "B- Boxplot du score du bonheur par années")
  fig = plt.figure()
  # Sélectionner les variables catégorielles
  variables_catégorielles = [ "year"]

  # Tracer des boxplots pour chaque variable catégorielle
  for variable in variables_catégorielles:
   
    sns.boxplot(x=variable, y="Life Ladder", data=df)
    plt.title(f"Boxplot du score de bonheur par {variable}")
    plt.xlabel(variable)
    plt.ylabel("Score de bonheur (Life Ladder)")
    plt.xticks(rotation=45)
    
  st.pyplot(fig)
  
  st.write ("")
  st.write ("### Interprétation :")
  st.write ("- Les scores de bonheur semblent avoir augmenté au fil des années, avec une légère variation d’une année à l’autre.")
  st.write ("- La médiane (ligne au milieu de chaque boîte) semble également augmenter progressivement.")
  st.write ("- Les valeurs aberrantes en 2020 pourraient être dues à des circonstances exceptionnelles (comme la pandémie de COVID-19).")



  st.write ("C- Affichage de la matrice de correlation des variables numériques")
  #Affichage de la matrice de corrélation par heatmap
  df_base_numeric = df.select_dtypes(include=['float64', 'int64'])
  # Calcul de la matrice de corrélation
  corr_matrix = df_base_numeric.corr()

  # Création de la heatmap avec seaborn
  fig = plt.figure ()
  sns.heatmap(data=corr_matrix, annot=True,cmap='viridis')
  st.write(fig)


  st.write ("### Interprétation :")
  
  st.write("- Le coefficient entre “Life Ladder” et “Log GDP per capita” est de 0,79,")
  st.write ("ce qui signifie qu’il y a une corrélation positive forte entre ces deux variables.")
  st.write("Cela implique que plus le produit intérieur brut par habitant est élevé, plus le score de bonheur est élevé, et vice versa.")

  st.write ("- Le coefficient entre “Freedom to make life choices” et “Perceptions of corruption” est de -0,44,")
  st.write ("ce qui signifie qu’il y a une corrélation négative modérée entre ces deux variables.")
  st.write ("Cela implique que plus la liberté de choix est élevée, plus la perception de la corruption est faible, et vice versa.")

  st.write("- Le coefficient entre “Generosity” et “Log GDP per capita” est de 0,000092,")
  st.write ("ce qui signifie qu’il n’y a pratiquement pas de corrélation entre ces deux variables.")
  st.write ("Cela implique que le niveau de générosité n’est pas lié au niveau de richesse, et qu’il peut varier indépendamment.")



  st.write ("D- Analyse du boxenplot du facture du score du bonheur par région")
  fig = plt.figure ()
  sns.boxenplot(x="Life Ladder", y= "Regional indicator", data = df2)
  plt.title("Distribution des scores de bonheur par région")
  plt.xlabel("Score de bonheur (Life Ladder)")
  plt.ylabel("Région")
  st.write(fig)

  st.write ("### Interprétation :")
  st.write ("- Pour l’Australie et la Nouvelle-Zélande, on peut dire :")
  st.write ("Cette région a le score de bonheur moyen le plus élevé, avec environ 5,2. Son boite à moustache est étroit et symétrique, ce qui signifie que les scores de bonheur sont peu dispersés et proches de la moyenne Il n’y a pas de valeurs aberrantes, ce qui signifie que tous les pays de cette région ont un niveau de bonheur similaire.")
  st.write ("")
  st.write ("- Pour l’Afrique subsaharienne, on peut dire :") 
  st.write ("Cette région a le score de bonheur moyen le plus bas, avec environ 4,1. Son boite à moustache est large et asymétrique, ce qui signifie que les scores de bonheur sont très dispersés et plus faibles que la moyenne.")
  st.write ("Il y a plusieurs valeurs aberrantes, ce qui signifie que certains pays de cette région ont un niveau de bonheur très différent des autres.") 
  st.write ("")
  st.write ("- Pour l’Europe occidentale, on peut dire :")
  st.write ("Cette région a un score de bonheur moyen élevé, avec environ 5,1.Son boite à moustache est étroit et légèrement asymétrique,")
  st.write ("Cette région a un score de bonheur moyen élevé, avec environ 5,1. Son boite à moustache est étroit et légèrement asymétrique,")
  st.write ("ce qui signifie que les scores de bonheur sont peu dispersés et légèrement supérieurs à la moyenne.")
  st.write ("Il y a quelques valeurs aberrantes, ce qui signifie que certains pays de cette région ont un niveau de bonheur plus bas ou plus haut que les autres.")


  st.write ("E- comparaison de Life Ladder and Log GDP per capita par country name pour chaque années")
  fig = plt.figure ()
  import plotly.express as px

  # Tracer le nuage de points interactif avec Plotly Express
  df = df.sort_values('year')
  fig = px.scatter(df,
                 x="Log GDP per capita",
                 y="Life Ladder",
                 animation_frame="year",
                 animation_group="Country name",

                 color="Social support",
                 hover_name="Country name",
                 size_max=200,
                 template="plotly_white")

  # Mettre à jour le titre du graphique
  fig.update_layout(title="comparaison de Life Ladder and Log GDP per capita par country name pour chaque années")

  # Afficher le graphique
  st.write(fig)

  st.write ("F -Life Ladder Comparison par Countries")
  fig = plt.figure ()
  fig = px.choropleth(df.sort_values("year"),
                   locations="Country name",
                   color="Life Ladder",
                   locationmode="country names",
                   animation_frame="year")
  fig.update_layout(title="Life Ladder Comparison par Countries")

  st.write(fig)
  st.write ("")
  st.write("### Analyse du facteur économique")
  st.write("1) choix de variable :")
  st.write ("- Log PIB par habitant (PIB par habitant) :")
  st.write ("Mesure la richesse économique moyenne d’un pays.")
  st.write ("Reflète le niveau de développement économique d’une nation.")
  st.write ("Un PIB par habitant élevé est souvent associé à une économie prospère et à un niveau de vie plus élevé pour les citoyens.")
  st.write ("- Soutien social :")
  st.write ("Bien que plus directement liée au bien-être social, une forte perception de soutien social est souvent corrélée avec une économie robuste.")
  st.write ("Les économies prospères peuvent offrir des filets de sécurité sociale plus solides et des infrastructures sociales mieux développées, contribuant ainsi au bien-être économique et social des individus.")
  st.write("- Espérance de vie en bonne santé à la naissance :")
  st.write ("Bien que liée à la santé, l’espérance de vie en bonne santé est également influencée par des facteurs économiques tels que l’accès aux soins de santé et les conditions de vie.")
  st.write ("Une économie forte peut investir dans des systèmes de santé robustes, améliorant ainsi la santé globale de la population et contribuant au bonheur et au bien-être économique.")

  st.write ("")
  st.write ("")
  st.write ("A- Comparaison par niveau de développement économique :")


  fig = plt.figure ()
  def economic_category(gdp):
      if gdp >= df['Log GDP per capita'].median():
          return 'Élevé'
      else:
          return 'Bas'
  
  sns.barplot(x=df['Log GDP per capita'].apply(economic_category), y='Life Ladder', data=df, palette="viridis")
  plt.title('Comparaison du score de bonheur par niveau de développement économique')
  plt.xlabel('Niveau de développement économique')
  plt.ylabel('Score de bonheur moyen (Life Ladder)')
  plt.xticks(rotation=45)
  st.write(fig) 

  st.write ("Ce graphique montre que le niveau de développement économique a une influence positive sur le score moyen de bonheur.")
  st.write ("Les pays avec un niveau de développement économique élevé ont un score de bonheur plus élevé que ceux avec un niveau bas.")

  st.write ("")
  st.write ("")
  st.write ("Après l'analyse de la matrice de corrélation on essaye d'afficher certaines variables les plus corrélées avec la variable cible.")
  st.write ("")
  st.write ("")

  st.write ("Nuage de points et régression de Log GDP per capita vs Life Ladder")
  fig = plt.figure ()
  # Premier sous-graphique
  plt.subplot(1, 2, 1)
  sns.scatterplot(y=df["Life Ladder"], x=df["Log GDP per capita"])
  sns.regplot(y=df["Life Ladder"], x=df["Log GDP per capita"], scatter=False, color='red')  # Ajouter une ligne de régression
  plt.ylabel("Life Ladder")
  plt.xlabel("Log GDP per capita")
  plt.title("Nuage de points et régression de Log GDP per capita vs Life Ladder")
  st.write (fig)

  st.write ("")
  st.write ("")

  st.write ("Nuage de points et régression de Social support vs Life Ladder")
  fig = plt.figure ()
  # Deuxième sous-graphique
  plt.subplot(1, 2, 2)
  sns.scatterplot(y=df["Life Ladder"], x=df["Social support"])
  sns.regplot(y=df["Life Ladder"], x=df["Social support"], scatter=False, color='green')  # Ajouter une ligne de régression
  plt.ylabel("Life Ladder")
  plt.xlabel("Social support")
  plt.title("Nuage de points et régression de Social support vs Life Ladder")
  # Afficher les deux sous-graphiques
  st.write(fig)



  st.write ("###Analyse statistique du premier graphique")

  st.write("Calcul des corrélations entre toutes les paires de variables numériques")
  correlation_matrix = df[['Log GDP per capita', 'Social support', 'Life Ladder']].corr()
  correlation_matrix

  st.write ("Les coefficients de corrélation que nous avons analysé indiquent une relation positive et significative entre les variables :")

  st.write ("- PIB par habitant (Log GDP per capita) et Score de bonheur (Life Ladder) : Avec un coefficient de 0.790166, il y a une forte corrélation positive. Cela suggère que les pays avec un PIB par habitant plus élevé ont tendance à avoir des scores de bonheur plus élevés.")
  st.write ("Soutien social (Social support) et Score de bonheur (Life Ladder) : Avec un coefficient de 0.707806, il y a également une forte corrélation positive. Cela indique que les pays où les individus perçoivent un plus grand soutien social ont tendance à avoir des scores de bonheur plus élevés.")
  st.write ("PIB par habitant (Log GDP per capita) et Soutien social (Social support) : Avec un coefficient de 0.692602, il y a une corrélation positive modérée, ce qui implique que les pays plus riches ont tendance à offrir un meilleur soutien social, mais cette relation n’est pas aussi forte que celle avec le score de bonheur.")


  st.write ("#même procédure sur d'autre variable, et entre 'Log GDP per capita' et 'Healthy life expectancy at birth'.")
  st.write ("# Life Ladder vs Log GDP per capita")
  st.write ("")
  fig = plt.figure ()
  sns.scatterplot(y=df["Life Ladder"], x=df["Healthy life expectancy at birth"])
  sns.regplot(y=df["Life Ladder"], x=df["Healthy life expectancy at birth"], scatter=False, color='red')  # Ajouter une ligne de régression
  plt.ylabel("Life Ladder")
  plt.xlabel("Healthy life expectancy at birth")
  plt.title("Nuage de points et régression de Healthy life expectancy at birth")
  st.write(fig)

  # Deuxième sous-graphique : Log GDP per capita vs Healthy life expectancy at birth
  st.write ("")
  st.write ("")
  st.write ("#Healthy life expectancy at birth vs Log GDP per capita")
  st.write ("")
  fig = plt.figure ()
  sns.scatterplot(x=df["Log GDP per capita"], y=df["Healthy life expectancy at birth"])
  sns.regplot(x=df["Log GDP per capita"], y=df["Healthy life expectancy at birth"], scatter=False, color='green')  # Ajouter une ligne de régression
  plt.xlabel("Log GDP per capita")
  plt.ylabel("Healthy life expectancy at birth")
  plt.title("Nuage de points et régression de Healthy life expectancy at birth vs Log GDP per capita")
  st.write (fig)

  st.write ("#Analyse statistique du deuxième graphique")

  correlation_matrix2 = df[['Log GDP per capita',  'Healthy life expectancy at birth', 'Life Ladder']].corr()
  # Afficher la matrice de corrélation
  correlation_matrix2
  
  st.write("")
  st.write("")
  st.write ("Les coefficients de corrélation que nous avons calculé confirment une forte relation positive entre les variables :")

  st.write("- PIB par habitant (Log GDP per capita) et Espérance de vie en bonne santé à la naissance (Healthy life expectancy at birth) : Avec un coefficient de 0.848049, cela indique que les pays avec un PIB par habitant plus élevé ont tendance à avoir une espérance de vie en bonne santé plus longue.")
  st.write("- PIB par habitant (Log GDP per capita) et Score de bonheur (Life Ladder) : Avec un coefficient de 0.790166, il y a une forte corrélation positive, suggérant que les pays plus riches ont tendance à avoir des scores de bonheur plus élevés.")
  st.write("- Espérance de vie en bonne santé à la naissance (Healthy life expectancy at birth) et Score de bonheur (Life Ladder) : Avec un coefficient de 0.744506, il y a également une corrélation positive significative, ce qui implique que les personnes vivant dans des pays avec une meilleure espérance de vie en bonne santé ont tendance à avoir un score de bonheur plus élevé.")
  st.write ("")
  st.write ("")


  st.write("#Analyse du facteur climat")

  st.write ("Comparaison du score du bonheur par changement climatique")
  sns.lmplot(x="Temperature", y="Life Ladder", data=df2)
  plt.title ("Score du bonheur et changement climatique" );

  # Définir les changements climatiques
  def climat_changement(climat):
      if climat > df2['Temperature'].median():
          return 'changement climatique important'
      else:
          return 'Peu de changement climatique'

  # Appliquer la fonction directement dans le graphique sans créer de nouvelle colonne
  couleurs = {'changement climatique important': 'red', 'Peu de changement climatique': 'green'}
  fig = plt.figure()
  sns.barplot(x=df2['Temperature'].apply(climat_changement), y='Life Ladder', palette=couleurs, data=df2)
  plt.title('Comparaison du score du bonheur par changement climatique')
  plt.ylabel('Score de bonheur moyen')
  plt.xticks(rotation=45)
  st.write (fig)
  
  st.write ("")
  st.write ("Cela confirme qu'il n'y a pas de corrélation entre le score du bonheur et le changement climatique")


  st.write ("# Analyse du facteur politique")

  df_pol = df2[['Country name', 'year', 'Life Ladder', 'Freedom to make life choices', 'Perceptions of corruption', 'Social support', 'Regional indicator']]
  df_pol.head(10)
  st.write("")
  
  fig= plt.figure()
  sns.set(style="whitegrid")
  fig, ax = plt.subplots(2, 1, figsize=(10, 12))
  sns.regplot(x='Freedom to make life choices', y='Life Ladder', data=df_pol, ax=ax[0])
  ax[0].set_title('Score de bonheur vs. Liberté de faire des choix de vie')
  ax[0].set_xlabel('Liberté de faire des choix de vie')
  ax[0].set_ylabel('Score de bonheur')

  sns.regplot(x='Perceptions of corruption', y='Life Ladder', data=df_pol, ax=ax[1], color='orange')
  ax[1].set_title('Score de bonheur vs. Perceptions de la corruption')
  ax[1].set_xlabel('Perceptions de la corruption')
  ax[1].set_ylabel('Score de bonheur')
  st.write (fig)

  st.write("Score de bonheur vs. Liberté de faire des choix de vie : Ce diagramme de dispersion montre une relation positive entre la liberté de faire des choix de vie et le score de bonheur. Les pays où les citoyens ressentent une plus grande liberté de choisir leur propre chemin dans la vie ont tendance à avoir des scores de bonheur plus élevés.")
  st.write("")
  st.write("Score de bonheur vs. Perceptions de la corruption : Le deuxième graphique, un diagramme de dispersion avec une ligne de régression, indique une relation négative entre les perceptions de la corruption et le score de bonheur. Cela suggère que dans les pays où la corruption est perçue comme étant élevée, les scores de bonheur tendent à être plus faibles.")
  
  st.write("#Life Ladder vs Social support")
  fig = plt.figure()
  # Premier graphique
  
  sns.scatterplot(y=df["Life Ladder"], x=df["Social support"])
  sns.regplot(y=df["Life Ladder"], x=df["Social support"], scatter=False, color='red')  # Ajouter une ligne de régression
  plt.ylabel("Life Ladder")
  plt.xlabel("Social support")
  plt.title("Nuage de points et régression de Life Ladder vs Social support")
  st.write (fig)
  st.write("Score de bonheur vs. Support social : Ce diagramme de dispersion montre une relation positive forte entre le support social et le score de bonheur. Les pays où les gens perçoivent un niveau élevé de soutien social ont tendance à avoir des scores de bonheur plus élevés, ce qui souligne l'importance des relations et du soutien communautaire pour le bien-être général.")


  st.write ("Conclusion :")
  st.write("Le heatmap de la matrice de corrélation illustre les relations entre les différentes variables de notre ensemble de données.")

  st.write("Voici quelques points clés à noter :")

  st.write ("Il existe une forte corrélation positive entre le support social et le score de bonheur (0.71), ce qui souligne l'importance du soutien social pour le bien-être général.")

  st.write ("La liberté de faire des choix de vie est également positivement corrélée avec le score de bonheur (0.53), ce qui suggère que la liberté personnelle joue un rôle clé dans la perception du bonheur.")

  st.write ("Les perceptions de la corruption sont négativement corrélées avec le score de bonheur (-0.43), indiquant que la corruption perçue peut diminuer le bien-être.")

  st.write ("Les relations entre la liberté de faire des choix de vie et les perceptions de la corruption (-0.49) montrent que dans les pays où les gens se sentent libres de faire leurs propres choix, la perception de la corruption tend à être plus faible.")

  st.write("")
  st.write("")
  
  st.write("# Analyse du facteur de la Santé")


  st.write ("I - Santé physique")

  st.write("A - Traitement / compréhension des données en lien avec la Santé au sens large (focus sur la colonne 'Healthy life at birth'")

  st.write("Nous nous proposons de démontrer dans cette partie qu'il existe un lien entre l'amélioration globale de la santé dans le monde se traduisant par une augmentation de l'espérance de vie à la naissance (calcul de l'âge moyen de l'espérance de vie collecté par l'Organisation Mondiale de la Santé) et 'évaluation du bien être")


  
  st.write ("3 - Vue statistiques du Dataframe")

  st.dataframe(df.describe())

  st.write ("1 - Analyse univariée (Age moyen d'espérance de vie)")
  
  fig = plt.figure()
  sns.histplot(df ["Healthy life expectancy at birth"], kde=True, bins=10, color="g", edgecolor="black")
  plt.title("Distribution du Healthy life expectancy at birth")
  plt.xlabel("Healthy life expectancy at birth")
  plt.ylabel("Fréquence") 
  # Calculer la médiane
  median_value = df["Healthy life expectancy at birth"].median()
  # Ajouter la ligne de médiane
  plt.axvline(x=median_value, color="red", linestyle="--", label="Médiane")
  # Afficher le graphique avec la légende
  plt.legend()
  st.pyplot(fig)
  
  
  st.write("2 - Evolution de l'espérance de vie entre 2005 et 2020")
  
  fig = plt.figure()
  sns.lineplot(x = "year", y = "Healthy life expectancy at birth", data = df, );
  st.pyplot(fig)
  
  """Nous notons une augmentation globale de l'espérance de vie entre 2008 et 2020."""

  """3 - Lien entre bien-être et espérance de vie"""
  fig = plt.figure ()
  sns.scatterplot(y=df["Life Ladder"], x=df["Healthy life expectancy at birth"])
  sns.regplot(y=df["Life Ladder"], x=df["Healthy life expectancy at birth"], scatter=False, color='red')  # Ajouter une ligne de régression
  plt.ylabel("Life Ladder")
  plt.xlabel("Healthy life expectancy at birth")
  plt.title("Nuage de points et régression de Healthy life expectancy at birth")
  st.write(fig)
  
  """Le coefficient de corrélation de 0,78 indique une corrélation positive forte entre 'évaluation due bien-êtr (Life Ladder) et l’espérance de vie en bonne santé à la naissance. La valeur p nulle suggère que cette corrélation n’est pas due au hasard et est statistiquement significative.

  Nous pouvons conclure que, bien qu'il existe de corrélation linéaire entre le score du bonheur et l'espérance de vie à la naissance, il existe, cependant, une corrélation forte entre les deux variables. En effet, en observant la carte heatmap ci-dessus, nous constatons que le niveau de corrélation entre les deux variables est significatif (7.4)"""


  
elif page == pages[2] : 
  
  st.write("### Modélisation")
  ## Traitement des valeurs manquantes
 #Scrapping de la variable "Log GDP per capita"
## Modelesations
  # Définir les features et la target
  df3= pd.read_csv("mon_fichier.csv")
  X = df3.drop('Life Ladder', axis=1)  # Toutes les colonnes sauf 'Life Ladder'
  y = df3['Life Ladder']               # La colonne 'Life Ladder'

# Séparer les données en ensembles d'entraînement et de test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #pour streamlit 3
  # Définition et entraînement des modèles
  def train_model(model_type, params):
      if model_type == 'Linear Regression':
          model = LinearRegression()
      elif model_type == 'Ridge':
          model = Ridge(**params)
      elif model_type == 'Lasso':
          model = Lasso(**params)
      elif model_type == 'Elastic Net':
          model = ElasticNet(**params)
      elif model_type == 'Random Forest':
          model = RandomForestRegressor(**params)
      elif model_type == 'XGBoost':
          model = XGBRegressor(**params)

      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      mse_train = mean_squared_error(y_train, y_pred_train)
      mse_test = mean_squared_error(y_test, y_pred_test)
      r2_train = r2_score(y_train, y_pred_train)
      r2_test = r2_score(y_test, y_pred_test)
      mae_train = mean_absolute_error(y_train, y_pred_train)
      mae_test = mean_absolute_error(y_test, y_pred_test)
      return model, mse_train, mse_test, r2_train, r2_test, mae_train, mae_test

# Streamlit interface
  model_type = st.selectbox('Choisissez le type de modèle', ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'Random Forest', 'XGBoost'])
  params = {}
  if model_type in ['Ridge', 'Lasso', 'Elastic Net']:
      params['alpha'] = st.slider('Alpha', 0.01, 1.0, 0.1)
      if model_type == 'Elastic Net':
          params['l1_ratio'] = st.slider('L1 Ratio', 0.01, 1.0, 0.5)
  if model_type in ['Random Forest', 'XGBoost']:
      params['n_estimators'] = st.slider('Number of Estimators', 10, 300, 100)
      if model_type == 'XGBoost':
          params['learning_rate'] = st.slider('Learning Rate', 0.01, 0.5, 0.1)

  if st.button('Entraîner et évaluer le modèle'):
      model, mse_train, mse_test, r2_train, r2_test, mae_train, mae_test = train_model(model_type, params)
      st.write(f"MSE Train: {mse_train:.4f}, R2 Train: {r2_train:.4f}, MAE Train: {mae_train:.4f}")
      st.write(f"MSE Test: {mse_test:.4f}, R2 Test: {r2_test:.4f}, MAE Test: {mae_test:.4f}")

      # Afficher les graphiques
      y_pred = model.predict(X_test)
      fig, ax = plt.subplots()
      ax.scatter(y_test, y_pred, alpha=0.5)
      ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
      ax.set_xlabel('Valeurs Réelles')
      ax.set_ylabel('Prédictions')
      ax.set_title('Comparaison des Prédictions et Valeurs Réelles')
      st.pyplot(fig)