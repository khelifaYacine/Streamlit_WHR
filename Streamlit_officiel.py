

import streamlit as st
from PIL import Image 
#Importation de packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
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
import shap




df = pd.read_csv("Fichier_de_base.csv")
df_21 =pd.read_csv("Fichier_2021.csv")
df2 = pd.read_csv("climate_change_indicators.csv", sep=";")  

  


st.sidebar.title("Sommaire")
pages = ["Exploration", "DataVizualisation", "Préprocessing","Modélisation", "Interprétabilité"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("# ANALYSE DU BIEN ETRE SUR TERRE")
  # Charger une image à partir d'un fichier local
  image = "Happiness.png"
  # Afficher l'image sur l'application Streamlit
  st.image(image, caption='Les clés du bonheur', use_column_width=True)

  st.write("## Introduction")
  st.write ("Ce projet vise à conduire une analyse approfondie des données du World Happiness Report afin d'évaluer le bonheur des pays du monde en utilisant une variété d'indicateurs socio-économiques tels que la santé, la corruption, l'économie et l'espérance de vie.")
  st.write ("L'objectif principal est de présenter ces données à travers des visualisations interactives tout en identifiant les combinaisons de facteurs qui expliquent les raisons pour lesquelles certains pays sont mieux classés que d'autres en termes de bonheur.")
  st.write("")

  #importer le jeu de données du 2005 à 2020. Et 2021
  
  st.write("Affichage des 10 premières lignes du world-happiness-report.csv 2005-2020")
  st.dataframe(df.head(10))
  st.write(df.shape)

    
  #st.write("Résumé statistique des variables numériques")
  #st.dataframe(df.describe())  
  
  st.write ("Adjonction des variables 'région' et 'température' à notre dataset ")
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
  # Enregistrement du DataFrame dans un fichier CSV
  df.to_csv('fichier_concat.csv', index=False)

  
  if st.checkbox("Afficher les NA") :
      st.dataframe(df.isna().sum()) 



elif page == pages[1] : 
  st.write("# DATAVIZUALISATION")

  st.write ("")
  st.write ("### Distribution du score du bonheur des pays")
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

  #st.write ("### Interprétation :")
  st.write ("0 le moins heureux et 8 le plus heureux.")
  #st.write ("- Graphe , avec une queue plus longue vers la gauche. Cela indique qu’il y a plus de gens qui ont un score de bonheur inférieur à la moyenne que supérieur à la moyenne.")
  st.write ("")
  st.write ( "### Boxplot du score du bonheur par années")
  fig = plt.figure()
  # Sélectionner les variables catégorielles
  variables_catégorielles = [ "year"]

  # Tracer des boxplots pour chaque variable catégorielle
  for variable in variables_catégorielles:
   
    sns.boxplot(x=variable, y="Life Ladder", data=df)
    plt.title(f"Boxplot Ladder score by {variable}")
    plt.xlabel(variable)
    plt.ylabel("Ladder score")
    plt.xticks(rotation=45)
    
  st.pyplot(fig)
  
  st.write ("")
  #st.write ("### Interprétation :")
  st.write ("Légère augementation au fil des années")
  st.write ("Quelques valeurs aberrantes en 2020")

  st.write ("")
  st.write ("")
  st.write ("## Matrice de correlation par heatmap")
  #Affichage de la matrice de corrélation par heatmap
 # df_base_numeric = df.select_dtypes(include=['float64', 'int64'])
  # Calcul de la matrice de corrélation
 # corr_matrix = df_base_numeric.corr()
 # Charger une image à partir d'un fichier local
  image2 = "Heatmap.png"
  # Afficher l'image sur l'application Streamlit
  st.image(image2, use_column_width=True)
  
  st.write("")

  st.write ("### Boxenplot du score du bonheur par région")
  fig = plt.figure ()
  sns.boxenplot(x="Life Ladder", y= "Regional indicator", data = df2)
  plt.title("Distribution des scores de bonheur par région")
  plt.xlabel("Score de bonheur")
  plt.ylabel("Région")
  st.write(fig)

  st.write("")
  #st.write ("### comparaison de Life Ladder and Log GDP per capita par pays et par années")
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
  fig.update_layout(title="Scrore du bonheur et PIB par pays selon l'année")

  # Afficher le graphique
  st.write(fig)

  st.write("")
  st.write ("### Life Ladder par pays selon l'année")
  fig = plt.figure ()
  fig = px.choropleth(df.sort_values("year"),
                   locations="Country name",
                   color="Life Ladder",
                   locationmode="country names",
                   animation_frame="year")
  fig.update_layout(title="Comparaison du Life Ladder par pays")

  st.write(fig)
  st.write ("")
   
  st.write("### Analyse du facteur climat")

  image3 = "Nuage_Temperature.png"
  # Afficher l'image sur l'application Streamlit
  st.image(image3, use_column_width=True)

 
elif page == pages[2] :

    missing = pd.read_csv ("fichier_concat.csv")    


    st.write ("# PREPROCESSING")
    # Présentation des valeurs manquantes
    st.write("## Traitement des valeurs manquantes")
    st.write("### Aperçu des valeurs manquantes dans notre dataset")
    
    # Afficher les valeurs manquantes
    st.write (missing.isnull().sum())

    # Scraping des données manquantes
    st.write("### 1. Scraping des données manquantes")
    st.write("Utilisation de l'API de la banque mondiale pour récuprer certaines valeurs manquantes de la variable 'Log GDP per capita'. ")
    #st.write("Pour la variable `Log GDP per capita`, nous avons utilisé une API pour récupérer certaines des valeurs manquantes. L'API utilisée est celle de la Banque mondiale. Cela a constitué un avantage majeur, car nous avons pu combler directement plusieurs lacunes dans notre dataset, sans devoir estimer toutes les valeurs de manière indirecte.")
    
    # Exemple de code pour scraper les données (si disponible)
    st.code("""
import requests

# Exemple de code pour récupérer les données manquantes
url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json"
response = requests.get(url)
data = response.json()
# Extraire les données nécessaires et les intégrer dans le dataset
    """, language='python')
    

    # Estimation des valeurs manquantes par régression linéaire
    st.write("### 2. Estimation des valeurs manquantes par régression linéaire")
    st.write("Utilisation d'estimations par régression linéaire pour les variables qui n'ont pas pu être obtenues via l'API")
    #st.write("Pour les données que nous n'avons pas pu obtenir via l'API, nous avons utilisé une méthode d'estimation par régression linéaire. Cette technique a permis de combler les valeurs manquantes en utilisant les données historiques disponibles pour chaque pays.")
    
    # Exemple de code pour la régression linéaire
    st.code("""
# Exemple simple de régression linéaire pour estimer des valeurs manquantes
for column in df.columns:
    if df[column].isnull().sum() > 0:
        X = df.dropna()[['Year']]  # Par exemple, en utilisant l'année comme variable explicative
        y = df.dropna()[column]
        model = LinearRegression().fit(X, y)
        df.loc[df[column].isnull(), column] = model.predict(df.loc[df[column].isnull(), ['Year']])
    """, language='python')

    # Imputation par KNN
    st.write("### 3. Imputation par KNN")
    st.write("Lorsque la régression linéaire n'était pas suffisante nous avons appliqué une méthode d'imputation par K-nearest neighbors (KNN).")
    #st.write("Pour certaines variables où la régression linéaire n'était pas suffisante, nous avons appliqué une méthode d'imputation par K-nearest neighbors (KNN).")
    
    # Exemple de code pour l'imputation par KNN
    st.code("""
# Imputation par KNN
imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    """, language='python')

    # Encodage des variables catégorielles
    st.write("## Encodage des variables catégorielles")
    st.write("Une fois les valeurs manquantes traitées, nous avons procédé à l'encodage des variables catégorielles afin de préparer les données pour les modèles de machine learning.")
    
    # Exemple de code pour l'encodage
    st.code("""
# One-Hot Encoding
encoder = OneHotEncoder()
encoded_vars = encoder.fit_transform(df[['Country name', 'Regional indicator']])
df_encoded = pd.concat([df.drop(columns=['Country name', 'Regional indicator']), pd.DataFrame(encoded_vars.toarray(), columns=encoder.get_feature_names_out())], axis=1)
    """, language='python')

     
        
    if st.checkbox(" ### Afficher le jeu de données après encodage") :
      final = pd.read_csv ("mon_fichier.csv")
      st.dataframe (final.isna().sum())
      st.dataframe(final)

 # Standardiastion
    st.write("## Strandardisation des variables numériques par StandardScaler")

    st.write("Ces étapes nous ont permis de finaliser notre dataset, qui est désormais prêt pour la phase de modélisation.")

    
elif page == pages[3] : 
  
    st.write("# MODELISATION")
  
    # Scrapping de la variable "Log GDP per capita"
    ## Modelesations
    # Définir les features et la target
    df3 = pd.read_csv("mon_fichier.csv")
    X = df3.drop('Life Ladder', axis=1)  # Toutes les colonnes sauf 'Life Ladder'
    y = df3['Life Ladder']            

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
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Stocker le modèle et les données dans st.session_state pour la page d'interprétabilité
        st.session_state['model'] = model
        st.session_state['X_train'] = X_train

        return model, mse_train, mse_test, mae_train, mae_test, r2_train, r2_test

    # Streamlit interface
    model_type = st.selectbox('Séléctionner le type de modèle', ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'Random Forest', 'XGBoost'])
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
        model, mse_train, mse_test, mae_train, mae_test, r2_train, r2_test = train_model(model_type, params)
        
        # Affichage des résultats sous forme de tableau
        metrics_table = pd.DataFrame({
            'Métrique': ['MSE', 'MAE', 'R2'],
            'Entraînement': [mse_train, mae_train, r2_train],
            'Test': [mse_test, mae_test, r2_test]
        })

        st.write("### Tableau des scores des métriques")
        st.table(metrics_table)

        # Afficher les graphiques
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Valeurs Réelles')
        ax.set_ylabel('Prédictions')
        ax.set_title('Comparaison des Prédictions et Valeurs Réelles')
        st.pyplot(fig)

elif page == "Interprétabilité":
    st.header("Interprétabilité des Modèles")

    # Vérifier si le modèle a été entraîné et stocké dans st.session_state
    if 'model' not in st.session_state:
        st.write("Aucun modèle n'a été entraîné. Veuillez entraîner un modèle sur la page Modélisation.")
    else:
        model = st.session_state['model']
        X_train = st.session_state['X_train']
        X_sample = shap.sample(X_train, 100)
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        st.subheader("Importance des caractéristiques")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig)

        st.subheader("Graphique en nuage de points")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig)

    st.write("### Interprétation des résultats")
    st.markdown("""
    - **Importance des variables** :
      - Le modèle Random Forest a identifié les variables les plus influentes sur le score de bonheur : "Log GDP per capita", "Social support", et "Healthy life expectancy at birth".
      - Les résultats soulignent l'importance des facteurs économiques et sociaux pour le bien-être des populations.
    """)

    st.write("### Conclusion")
    st.markdown("""
    - **Analyse du bien-être** :
      - L'analyse du World Happiness Report a révélé que les aspects économiques, tels que le PIB par habitant, et les facteurs sociaux, comme le support social et l'espérance de vie, sont cruciaux pour la perception du bonheur.
      - Parmi les modèles testés, le Random Forest s'est montré le plus efficace, avec un R² de 0.8919 sur les données de test, capturant bien les relations complexes entre les variables.
    """)
